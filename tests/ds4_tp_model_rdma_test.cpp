// Isolated synthetic-identity two-host model experiment; not a serving path.
#include "../src/ds4/ds4_model.hpp"
#include "../src/ds4/ds4_tp_channel.hpp"
#include "strata/net/verbs.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace strata::ds4::tp2;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"line %d: %s (%s)\n",__LINE__,#x,error.c_str()); return 1; } } while (0)

int ds4_tp_model_rdma_test(int argc,char** argv) {
    const bool reference=std::string(argv[1])=="--tp-reference";
    if ((reference && argc!=4) || (!reference && argc!=8)) {
        std::fprintf(stderr,"usage: --tp-reference pack output | --tp-model rank host port rdma pack reference\n"); return 2;
    }
    std::string error; long rank=-1,port=0; char* end=nullptr;
    if (!reference) {
        rank=std::strtol(argv[2],&end,10); if (!*argv[2] || *end || rank<0 || rank>1) return 2;
        port=std::strtol(argv[4],&end,10); if (!*argv[4] || *end || port<1 || port>65535) return 2;
    }
    CHECK(setenv("DS4_STREAM","0",1)==0);
    strata::ds4::Options o; o.pack=reference?argv[2]:argv[6]; o.max_context=32768; o.gpus={0};
    o.reserve_mib=1024; o.tp=1; o.threads=false; o.vision=false; o.dspark=false; o.lookup=false;
    o.prefill_chunk=0; o.expert_cache=false; o.mmap_experts=1; o.profile_readonly=true; o.cpu_threads=14;
    o.vram_experts=reference?1.0:0.0;
    if (!reference) { o.tp2_rank=static_cast<int>(rank); o.tp2_expert_budget=60ull<<30; o.tp_worker_verify=true; }
    strata::ds4::Model model; CHECK(model.load(o,error));
    std::fprintf(stderr,"%s model loaded\n",reference?"reference":"rank");
    const std::vector<std::vector<int>> batches={{5,6,7},{8},{9},{10}};
    std::vector<float> all;
    if (reference) {
        for (const auto& tokens:batches) {
            CHECK(model.feed(tokens.data(),static_cast<int>(tokens.size()),1,error));
            std::vector<float> logits; CHECK(model.logits(logits) && logits.size()==static_cast<std::size_t>(model.vocab()));
            all.insert(all.end(),logits.begin(),logits.end());
        }
        std::ofstream out(argv[3],std::ios::binary|std::ios::trunc); CHECK(out.good());
        out.write(reinterpret_cast<const char*>(all.data()),static_cast<std::streamsize>(all.size()*sizeof(float)));
        CHECK(out.good()); std::printf("reference rows=%zu vocab=%d\n",batches.size(),model.vocab()); return 0;
    }
    std::ifstream in(argv[7],std::ios::binary|std::ios::ate); CHECK(in.good());
    const auto expected_bytes=batches.size()*static_cast<std::size_t>(model.vocab())*sizeof(float);
    CHECK(in.tellg()==static_cast<std::streamoff>(expected_bytes)); in.seekg(0);
    all.resize(expected_bytes/sizeof(float)); in.read(reinterpret_cast<char*>(all.data()),static_cast<std::streamsize>(expected_bytes)); CHECK(in.good());
    strata::net::IbrverbsConfig cfg; cfg.rank=static_cast<std::uint32_t>(rank); cfg.bootstrap_host=argv[3];
    cfg.bootstrap_port=static_cast<std::uint16_t>(port); cfg.rdma_device=argv[5];
    auto comm=strata::net::CreateIbrverbsCommunicator(cfg,&error); CHECK(comm);
    // A real pack/build identity will be mandatory in production; this fixture
    // uses a synthetic identity and the separately verified identical XS pack.
    Identity id; id.protocol=1; id.world=2; id.rank=cfg.rank; id.hidden=5120; id.layers=40;
    id.experts=384; id.expert_width=2304; id.max_context=32768; id.feature_flags=1;
    id.max_partial_bytes=comm->MaxPartialBytes(); id.model_digest[0]=1; id.partition_digest[0]=2; id.build_digest[0]=3;
    Identity peer; std::uint64_t limit=0; CHECK(exchange_identity(*comm,id,peer,limit,error));
    CHECK(model.tp2_attach_test(*comm,static_cast<std::size_t>(limit),error));
    const int probe=5; CHECK(!model.feed_mark(&probe,1,1,0,error));
    CHECK(!model.tp_verify_begin(&probe,1,error) && !model.restore(0));
    error.clear();
    hipStream_t stream=nullptr; float* device=nullptr;
    const auto bytes=static_cast<std::size_t>(model.vocab())*sizeof(float); CHECK(bytes<=limit);
    CHECK(hipStreamCreate(&stream)==hipSuccess && hipMalloc(&device,bytes)==hipSuccess);
    for (std::size_t i=0;i<batches.size();++i) {
        CHECK(comm->BeginOperation(i+1,&error));
        CHECK(model.feed(batches[i].data(),static_cast<int>(batches[i].size()),1,error));
        std::vector<float> logits; CHECK(model.logits(logits) && logits.size()==static_cast<std::size_t>(model.vocab()));
        CHECK(hipMemcpyAsync(device,logits.data(),bytes,hipMemcpyHostToDevice,stream)==hipSuccess);
        const float* other=comm->ExchangePartial(device,bytes,stream,&error); CHECK(other);
        std::vector<float> peer_logits(logits.size());
        CHECK(hipMemcpyAsync(peer_logits.data(),other,bytes,hipMemcpyDeviceToHost,stream)==hipSuccess);
        CHECK(hipStreamSynchronize(stream)==hipSuccess && comm->CheckQueued(&error) && comm->EndOperation(i+1,&error));
        float pair_delta=0,ref_delta=0; double sq=0;
        const auto* ref=all.data()+i*logits.size();
        for (std::size_t j=0;j<logits.size();++j) {
            CHECK(std::isfinite(logits[j]) && std::isfinite(peer_logits[j]) && std::isfinite(ref[j]));
            pair_delta=std::max(pair_delta,std::fabs(logits[j]-peer_logits[j]));
            const float diff=std::fabs(logits[j]-ref[j]); ref_delta=std::max(ref_delta,diff); sq+=double(diff)*diff;
        }
        const auto got=std::max_element(logits.begin(),logits.end())-logits.begin();
        const auto want=std::max_element(ref,ref+logits.size())-ref;
        std::printf("rank=%ld row=%zu pair_max=%.8g reference_max=%.8g reference_rms=%.8g top=%td reference_top=%td\n",
                    rank,i,pair_delta,ref_delta,std::sqrt(sq/logits.size()),got,want);
        std::fflush(stdout); CHECK(pair_delta<=1e-5f && got==want);
    }
    CHECK(hipFree(device)==hipSuccess && hipStreamDestroy(stream)==hipSuccess);
    Control done;
    if (rank==0) {
        CHECK(send_message(*comm,{Command::Shutdown,5,6,{}},16,error));
        CHECK(receive_message(*comm,16,done,error));
    } else {
        CHECK(receive_message(*comm,16,done,error));
        CHECK(send_message(*comm,{Command::Shutdown,5,6,{}},16,error));
    }
    CHECK(done.command==Command::Shutdown && done.sequence==5);
    return 0;
}
