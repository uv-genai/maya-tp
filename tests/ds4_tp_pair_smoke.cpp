// Paired control smoke with full replicated models; NOT tensor-parallel inference.
#include "../src/ds4/ds4_model.hpp"
#include "../src/ds4/ds4_tp_model_worker.hpp"
#include "../src/ds4/ds4_tp_worker.hpp"
#include "strata/net/verbs.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace strata::ds4::tp2;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"line %d: %s (%s)\n",__LINE__,#x,error.c_str()); return 1; } } while (0)

int ds4_tp_pair_smoke(int argc, char** argv) {
    if (argc != 7) {
        std::fprintf(stderr,"usage: ds4_tp_verify_test --rdma rank bootstrap-host port rdma-device pack-dir\n");
        return 2;
    }
    std::string error;
    char* end=nullptr;
    const long rank=std::strtol(argv[2],&end,10);
    if (*argv[2]==0 || *end || rank<0 || rank>1) return 2;
    const long port=std::strtol(argv[4],&end,10);
    if (*argv[4]==0 || *end || port<=0 || port>65535) return 2;
    strata::ds4::Options o;
    // An adaptive per-host CPU/stream split can change rounding and expert
    // routing; force the same CPU path on both ranks for this state smoke.
    CHECK(setenv("DS4_STREAM","0",1)==0);
    o.pack=argv[6]; o.max_context=32768; o.gpus={0}; o.reserve_mib=1024;
    o.tp=1; o.threads=true; o.vision=false; o.prefill_chunk=0; o.vram_experts=0;
    o.cpu_threads=14; o.mmap_experts=1;
    o.profile_readonly=true; o.profile=o.pack+"/expert-profile.bin";
    o.dspark=rank==0; o.tp_worker_verify=true;
    strata::ds4::Model model;
    CHECK(model.load(o,error));
    std::fprintf(stderr,"rank %ld: replicated XS model loaded, establishing test pair\n",rank);

    strata::net::IbrverbsConfig cfg;
    cfg.rank=static_cast<std::uint32_t>(rank); cfg.bootstrap_host=argv[3];
    cfg.bootstrap_port=static_cast<std::uint16_t>(port); cfg.rdma_device=argv[5];
    auto comm=strata::net::CreateIbrverbsCommunicator(cfg,&error); CHECK(comm);
    // Fixed synthetic identity: pack content matching is established separately;
    // this smoke must not be mistaken for a production identity implementation.
    Identity id; id.protocol=1; id.world=2; id.rank=cfg.rank; id.hidden=5120; id.layers=40;
    id.experts=384; id.expert_width=2304; id.max_context=32768; id.feature_flags=1;
    id.max_partial_bytes=comm->MaxPartialBytes();
    id.model_digest[0]=1; id.partition_digest[0]=2; id.build_digest[0]=3;
    Identity peer; std::uint64_t limit=0; CHECK(exchange_identity(*comm,id,peer,limit,error));
    ModelWorker<strata::ds4::Model> model_worker(model);
    const std::vector<Control> commands={
        {Command::Feed,1,0,{5,6,7}}, {Command::Step,2,3,{8}},
        {Command::Verify,3,4,{9,10,11}}, {Command::Verdict,4,6,{1,12}},
        {Command::Step,5,6,{12}}, {Command::Shutdown,6,7,{}}
    };
    if (rank==1) {
        Worker worker(*comm,32768,static_cast<std::uint32_t>(model.vocab()),16);
        for (std::size_t i=0;i<commands.size();++i)
            CHECK(worker.run_one([&](const Control& c, std::string& e) { return model_worker.execute(c,e); },error));
        CHECK(worker.state().stopped() && worker.state().position()==7);
    } else {
        for (const Control& c:commands) {
            CHECK(send_message(*comm,c,16,error));
            CHECK(comm->BeginOperation(c.sequence,&error));
            CHECK(model_worker.execute(c,error));
            CHECK(comm->CheckQueued(&error));
            CHECK(comm->EndOperation(c.sequence,&error));
        }
    }
    CHECK(model.pos()==7);
    // Probe the resulting full replicated models, then exchange logits through
    // the RDMA data path for an actual cross-host state comparison.
    const int probe=13;
    CHECK(model.feed(&probe,1,1,error));
    std::vector<float> logits; CHECK(model.logits(logits) && !logits.empty());
    const std::size_t bytes=logits.size()*sizeof(float); CHECK(bytes<=limit);
    hipStream_t stream=nullptr; float* dev=nullptr;
    CHECK(hipStreamCreate(&stream)==hipSuccess && hipMalloc(&dev,bytes)==hipSuccess);
    CHECK(hipMemcpyAsync(dev,logits.data(),bytes,hipMemcpyHostToDevice,stream)==hipSuccess);
    CHECK(comm->BeginOperation(7,&error));
    const float* other=comm->ExchangePartial(dev,bytes,stream,&error); CHECK(other);
    std::vector<float> peer_logits(logits.size());
    CHECK(hipMemcpyAsync(peer_logits.data(),other,bytes,hipMemcpyDeviceToHost,stream)==hipSuccess);
    CHECK(hipStreamSynchronize(stream)==hipSuccess);
    CHECK(comm->CheckQueued(&error) && comm->EndOperation(7,&error));
    float max_abs=0;
    for (std::size_t i=0;i<logits.size();++i) {
        CHECK(std::isfinite(logits[i]) && std::isfinite(peer_logits[i]));
        max_abs=std::max(max_abs,std::fabs(logits[i]-peer_logits[i]));
    }
    CHECK(hipFree(dev)==hipSuccess && hipStreamDestroy(stream)==hipSuccess);
    std::printf("rank %ld: replicated DeepSeek control smoke, %zu logits, max_abs=%.8g\n",rank,logits.size(),max_abs);
    CHECK(max_abs<=1e-3f);
    Control done;
    if (rank==1) {
        CHECK(send_message(*comm,{Command::Shutdown,8,8,{}},16,error));
        CHECK(receive_message(*comm,16,done,error));
    } else {
        CHECK(receive_message(*comm,16,done,error));
        CHECK(send_message(*comm,{Command::Shutdown,8,8,{}},16,error));
    }
    CHECK(done.command==Command::Shutdown && done.sequence==8 && done.position==8);
    return 0;
}
