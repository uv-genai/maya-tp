// Paired real-text experiment. Synthetic identity, not a production launcher.
#include "../src/ds4/ds4_model.hpp"
#include "../src/ds4/ds4_tp_channel.hpp"
#include "strata/net/verbs.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

using namespace strata::ds4::tp2;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"line %d: %s (%s)\n",__LINE__,#x,error.c_str()); return 1; } } while (0)
namespace { constexpr std::uint32_t magic=0x32545854; constexpr int max_steps=64; }

int ds4_tp_text_rdma_test(int argc,char** argv) {
    const bool reference=std::string(argv[1])=="--text-reference";
    if ((reference && argc!=5) || (!reference && argc!=10)) {
        std::fprintf(stderr,"usage: --text-reference pack prompt-tokens reference | --text-model rank host port rdma pack prompt-tokens reference output-tokens\n"); return 2;
    }
    std::string error; long rank=-1,port=0; char* end=nullptr;
    if (!reference) {
        rank=std::strtol(argv[2],&end,10); if (!*argv[2] || *end || rank<0 || rank>1) return 2;
        port=std::strtol(argv[4],&end,10); if (!*argv[4] || *end || port<1 || port>65535) return 2;
    }
    std::vector<int> prompt;
    { std::ifstream f(reference?argv[3]:argv[7]); CHECK(f.good()); int t;
      while (f>>t) prompt.push_back(t); CHECK(f.eof() && !prompt.empty() && prompt.size()<=1024); }
    CHECK(setenv("DS4_STREAM","0",1)==0);
    strata::ds4::Options o; o.pack=reference?argv[2]:argv[6]; o.max_context=32768; o.gpus={0};
    o.reserve_mib=1024; o.tp=1; o.threads=false; o.vision=false; o.dspark=false; o.lookup=false;
    o.prefill_chunk=0; o.expert_cache=false; o.mmap_experts=1; o.profile_readonly=true; o.cpu_threads=14;
    o.vram_experts=reference?1.0:0.0;
    if (!reference) { o.tp2_rank=static_cast<int>(rank); o.tp2_expert_budget=90ull<<30; o.tp_worker_verify=true; }
    strata::ds4::Model model; CHECK(model.load(o,error));
    std::fprintf(stderr,"%s model loaded; prompt=%zu\n",reference?"reference":"rank",prompt.size());
    for (int token:prompt) CHECK(token>=0 && token<model.vocab());
    const auto vocab=static_cast<std::size_t>(model.vocab());
    std::vector<std::uint32_t> reference_tokens; std::vector<float> reference_logits;
    if (reference) {
        CHECK(model.feed(prompt.data(),static_cast<int>(prompt.size()),1,error));
        for (int i=0;i<max_steps;++i) {
            std::vector<float> logits; CHECK(model.logits(logits) && logits.size()==vocab);
            const int token=model.sample(0,1,0,0,0,0); CHECK(token>=0 && token<model.vocab());
            reference_tokens.push_back(static_cast<std::uint32_t>(token));
            reference_logits.insert(reference_logits.end(),logits.begin(),logits.end());
            if (token==1 || i+1==max_steps) break;
            CHECK(model.step(token,true,error));
        }
        std::ofstream f(argv[4],std::ios::binary|std::ios::trunc); CHECK(f.good());
        const std::uint32_t h[]={magic,static_cast<std::uint32_t>(vocab),static_cast<std::uint32_t>(reference_tokens.size())};
        f.write(reinterpret_cast<const char*>(h),sizeof(h));
        f.write(reinterpret_cast<const char*>(reference_tokens.data()),static_cast<std::streamsize>(reference_tokens.size()*4));
        f.write(reinterpret_cast<const char*>(reference_logits.data()),static_cast<std::streamsize>(reference_logits.size()*4));
        CHECK(f.good()); std::printf("reference prompt=%zu generated=%zu\n",prompt.size(),reference_tokens.size()); return 0;
    }
    {
        std::ifstream f(argv[8],std::ios::binary); CHECK(f.good()); std::uint32_t h[3];
        f.read(reinterpret_cast<char*>(h),sizeof(h)); CHECK(f.good() && h[0]==magic && h[1]==vocab && h[2]>0 && h[2]<=max_steps);
        reference_tokens.resize(h[2]); reference_logits.resize(static_cast<std::size_t>(h[2])*vocab);
        f.read(reinterpret_cast<char*>(reference_tokens.data()),static_cast<std::streamsize>(reference_tokens.size()*4));
        f.read(reinterpret_cast<char*>(reference_logits.data()),static_cast<std::streamsize>(reference_logits.size()*4));
        CHECK(f.good() && f.peek()==std::ifstream::traits_type::eof());
    }
    strata::net::IbrverbsConfig cfg; cfg.rank=static_cast<std::uint32_t>(rank); cfg.bootstrap_host=argv[3];
    cfg.bootstrap_port=static_cast<std::uint16_t>(port); cfg.rdma_device=argv[5];
    auto comm=strata::net::CreateIbrverbsCommunicator(cfg,&error); CHECK(comm);
    Identity id; id.protocol=1; id.world=2; id.rank=cfg.rank; id.hidden=5120; id.layers=40; id.experts=384;
    id.expert_width=2304; id.max_context=32768; id.feature_flags=1; id.max_partial_bytes=comm->MaxPartialBytes();
    id.model_digest[0]=1; id.partition_digest[0]=2; id.build_digest[0]=3; // synthetic test identity
    Identity peer; std::uint64_t limit=0; CHECK(exchange_identity(*comm,id,peer,limit,error));
    CHECK(model.tp2_attach_test(*comm,static_cast<std::size_t>(limit),error));
    const std::size_t bytes=vocab*sizeof(float); CHECK(bytes<=limit);
    hipStream_t stream=nullptr; float* device=nullptr;
    CHECK(hipStreamCreate(&stream)==hipSuccess && hipMalloc(&device,bytes)==hipSuccess);
    std::uint64_t sequence=0; std::size_t top_matches=0; float max_reference_delta=0; double reference_square=0;
    std::vector<std::uint32_t> generated;
    for (int phase=0;phase<2;++phase) {
        if (phase==1) {
            CHECK(comm->BeginOperation(++sequence,&error)); CHECK(model.tp_worker_reset(error));
            CHECK(comm->CheckQueued(&error) && comm->EndOperation(sequence,&error));
        }
        int prev=-1; const std::size_t count=phase==0?reference_tokens.size():max_steps;
        for (std::size_t i=0;i<count;++i) {
            if (i%8==0) std::fprintf(stderr,"rank=%ld phase=%s step=%zu/%zu\n",rank,
                                      phase==0?"reference":"greedy",i,count);
            CHECK(comm->BeginOperation(++sequence,&error));
            if (i==0) CHECK(model.feed(prompt.data(),static_cast<int>(prompt.size()),1,error));
            else CHECK(model.step(prev,true,error));
            std::vector<float> logits; CHECK(model.logits(logits) && logits.size()==vocab);
            CHECK(hipMemcpyAsync(device,logits.data(),bytes,hipMemcpyHostToDevice,stream)==hipSuccess);
            const float* remote=comm->ExchangePartial(device,bytes,stream,&error); CHECK(remote);
            std::vector<float> other(vocab);
            CHECK(hipMemcpyAsync(other.data(),remote,bytes,hipMemcpyDeviceToHost,stream)==hipSuccess);
            CHECK(hipStreamSynchronize(stream)==hipSuccess && comm->CheckQueued(&error) && comm->EndOperation(sequence,&error));
            const auto top=std::max_element(logits.begin(),logits.end())-logits.begin();
            float pair_delta=0;
            for (std::size_t j=0;j<vocab;++j) {
                CHECK(std::isfinite(logits[j]) && std::isfinite(other[j]));
                pair_delta=std::max(pair_delta,std::fabs(logits[j]-other[j]));
                if (phase==0) {
                    const float r=reference_logits[i*vocab+j]; CHECK(std::isfinite(r));
                    const float d=std::fabs(logits[j]-r); max_reference_delta=std::max(max_reference_delta,d);
                    reference_square+=double(d)*d;
                }
            }
            CHECK(pair_delta<=1e-5f);
            if (phase==0 && top==reference_tokens[i]) ++top_matches;
            std::uint32_t token=0;
            if (rank==0) {
                const int picked=phase==0?static_cast<int>(reference_tokens[i]):model.sample(0,1,0,0,0,0);
                CHECK(picked>=0 && picked<model.vocab()); token=static_cast<std::uint32_t>(picked);
                CHECK(comm->SendControl(&token,sizeof(token),&error));
            } else CHECK(comm->RecvControl(&token,sizeof(token),&error));
            CHECK(token<vocab && (phase!=0 || token==reference_tokens[i]));
            if (phase==1) generated.push_back(token);
            prev=static_cast<int>(token);
            if (token==1) break;
        }
    }
    std::ofstream out(argv[9]); CHECK(out.good());
    for (auto token:generated) out<<token<<' '; out<<'\n'; CHECK(out.good());
    std::printf("rank=%ld prompt=%zu reference_rows=%zu top_matches=%zu reference_max=%.8g reference_rms=%.8g generated=%zu\n",
                rank,prompt.size(),reference_tokens.size(),top_matches,max_reference_delta,
                std::sqrt(reference_square/(reference_tokens.size()*vocab)),generated.size());
    CHECK(hipFree(device)==hipSuccess && hipStreamDestroy(stream)==hipSuccess);
    Control done;
    if (rank==0) {
        CHECK(send_message(*comm,{Command::Shutdown,sequence+1,static_cast<std::uint64_t>(model.pos()),{}},16,error));
        CHECK(receive_message(*comm,16,done,error));
    } else {
        CHECK(receive_message(*comm,16,done,error));
        CHECK(send_message(*comm,{Command::Shutdown,sequence+1,static_cast<std::uint64_t>(model.pos()),{}},16,error));
    }
    CHECK(done.command==Command::Shutdown && done.sequence==sequence+1 && done.position==static_cast<std::uint64_t>(model.pos()));
    return 0;
}
