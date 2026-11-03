// Paired real-text experiment. Synthetic identity, not a production launcher.
#include "../src/ds4/ds4_model.hpp"
#include "../src/ds4/ds4_tp_channel.hpp"
#include "strata/net/verbs.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
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
      while (f>>t) prompt.push_back(t); CHECK(f.eof() && !prompt.empty() && prompt.size()<=32768); }
    CHECK(setenv("DS4_STREAM","0",1)==0);
    const char* chunk_env=std::getenv("DS4_TP_TEST_CHUNK");
    const int chunk=chunk_env ? std::atoi(chunk_env) : 0;
    if (chunk_env) { CHECK(chunk>=256 && chunk%16==0);
        CHECK(setenv("DS4_FLOW","0",1)==0 && setenv("DS4_YOCO","0",1)==0 &&
              setenv("DS4_CHUNK_MIN","17",1)==0); }
    strata::ds4::Options o; o.pack=reference?argv[2]:argv[6]; o.max_context=32768; o.gpus={0};
    o.reserve_mib=1024; o.tp=1; o.threads=false; o.vision=false; o.dspark=false; o.lookup=false;
    o.prefill_chunk=chunk; o.tp2_chunk_enabled=!reference && chunk>0;
    o.expert_cache=false; o.mmap_experts=1; o.profile_readonly=true; o.cpu_threads=14;
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
    id.expert_width=2304; id.max_context=32768; id.feature_flags=1|(static_cast<std::uint64_t>(chunk)<<16);
    id.max_partial_bytes=comm->MaxPartialBytes();
    id.model_digest[0]=1; id.partition_digest[0]=2; id.build_digest[0]=3; // synthetic test identity
    Identity peer; std::uint64_t limit=0; CHECK(exchange_identity(*comm,id,peer,limit,error));
    CHECK(model.tp2_attach(*comm,static_cast<std::size_t>(limit),error));
    const std::size_t bytes=vocab*sizeof(float); CHECK(bytes<=limit);
    hipStream_t stream=nullptr; float* device=nullptr;
    CHECK(hipStreamCreate(&stream)==hipSuccess && hipMalloc(&device,bytes)==hipSuccess);
    std::uint64_t sequence=0; std::size_t top_matches=0; float max_reference_delta=0; double reference_square=0;
    std::vector<std::uint32_t> generated;
    if (std::getenv("DS4_TP_TEST_PREFILL_SWEEP")) {
        CHECK(chunk>=256);
        const auto reset=[&] {
            return comm->BeginOperation(++sequence,&error) && model.tp_worker_reset(error) &&
                   model.tp_worker_synchronize(error) && comm->CheckQueued(&error) && comm->EndOperation(sequence,&error);
        };
        std::vector<int> lengths={17,128,129,256,257,512};
        if (const char* selected=std::getenv("DS4_TP_TEST_SWEEP_ONLY")) {
            char* end=nullptr; const long value=std::strtol(selected,&end,10);
            CHECK(*selected && end && !*end && value>=17 && value<=32768);
            lengths={static_cast<int>(value)};
        }
        for (int length : lengths) {
            std::vector<int> tokens;
            if (std::getenv("DS4_TP_TEST_USE_PROMPT_EXACT")) {
                CHECK(static_cast<std::size_t>(length)==prompt.size()); tokens=prompt;
            } else for (int i=0;i<length;++i) tokens.push_back(prompt[static_cast<std::size_t>(i)%prompt.size()]);
            CHECK(reset());
            const auto window_start=std::chrono::steady_clock::now();
            for (int at=0;at<length;) {
                const int n=std::min(16,length-at);
                CHECK(comm->BeginOperation(++sequence,&error));
                CHECK(model.feed(tokens.data()+at,n,at+n==length?1:0,error));
                CHECK(model.tp_worker_synchronize(error) && comm->CheckQueued(&error) && comm->EndOperation(sequence,&error));
                at+=n;
            }
            const double window_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-window_start).count();
            std::vector<float> baseline; CHECK(model.logits(baseline) && baseline.size()==vocab);
            CHECK(reset());
            std::vector<float> actual,other(vocab);
            const auto chunk_start=std::chrono::steady_clock::now();
            for (int at=0;at<length;) {
                const int n=std::min(chunk,length-at); const bool last=at+n==length;
                CHECK(comm->BeginOperation(++sequence,&error));
                CHECK(model.feed(tokens.data()+at,n,last?1:0,error));
                if (last) {
                    CHECK(model.logits(actual) && actual.size()==vocab);
                    CHECK(hipMemcpyAsync(device,actual.data(),bytes,hipMemcpyHostToDevice,stream)==hipSuccess);
                    const float* remote=comm->ExchangePartial(device,bytes,stream,&error); CHECK(remote);
                    CHECK(hipMemcpyAsync(other.data(),remote,bytes,hipMemcpyDeviceToHost,stream)==hipSuccess);
                    CHECK(hipStreamSynchronize(stream)==hipSuccess);
                }
                CHECK(model.tp_worker_synchronize(error) && comm->CheckQueued(&error) && comm->EndOperation(sequence,&error));
                at+=n;
            }
            const double chunk_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-chunk_start).count();
            float pair_delta=0, reference_delta=0;
            for (std::size_t i=0;i<vocab;++i) {
                CHECK(std::isfinite(actual[i]) && std::isfinite(other[i]) && std::isfinite(baseline[i]));
                pair_delta=std::max(pair_delta,std::fabs(actual[i]-other[i]));
                reference_delta=std::max(reference_delta,std::fabs(actual[i]-baseline[i]));
            }
            const auto a=std::max_element(actual.begin(),actual.end())-actual.begin();
            const auto b=std::max_element(baseline.begin(),baseline.end())-baseline.begin();
            std::fprintf(stderr,"rank=%ld prefill length=%d pair_max=%.8g window_max=%.8g chunk_top=%td window_top=%td window_s=%.4f chunk_s=%.4f\n",
                         rank,length,pair_delta,reference_delta,a,b,window_seconds,chunk_seconds);
            if (rank==0) if (const char* path=std::getenv("DS4_TP_TEST_LOGITS")) {
                std::ofstream f(path,std::ios::binary|std::ios::trunc); CHECK(f.good());
                f.write(reinterpret_cast<const char*>(baseline.data()),static_cast<std::streamsize>(baseline.size()*sizeof(float)));
                f.write(reinterpret_cast<const char*>(actual.data()),static_cast<std::streamsize>(actual.size()*sizeof(float)));
                CHECK(f.good());
            }
            if (length>chunk && std::getenv("DS4_TP_TEST_COMPARE_WHOLE")) {
                CHECK(reset());
                CHECK(comm->BeginOperation(++sequence,&error));
                CHECK(model.feed(tokens.data(),length,1,error));
                std::vector<float> whole; CHECK(model.logits(whole) && whole.size()==vocab);
                CHECK(hipMemcpyAsync(device,whole.data(),bytes,hipMemcpyHostToDevice,stream)==hipSuccess);
                const float* remote=comm->ExchangePartial(device,bytes,stream,&error); CHECK(remote);
                CHECK(hipMemcpyAsync(other.data(),remote,bytes,hipMemcpyDeviceToHost,stream)==hipSuccess);
                CHECK(hipStreamSynchronize(stream)==hipSuccess && model.tp_worker_synchronize(error) &&
                      comm->CheckQueued(&error) && comm->EndOperation(sequence,&error));
                float whole_delta=0, whole_pair=0;
                for (std::size_t i=0;i<vocab;++i) {
                    CHECK(std::isfinite(whole[i]) && std::isfinite(other[i]));
                    whole_delta=std::max(whole_delta,std::fabs(whole[i]-actual[i]));
                    whole_pair=std::max(whole_pair,std::fabs(whole[i]-other[i]));
                }
                const auto w=std::max_element(whole.begin(),whole.end())-whole.begin();
                std::fprintf(stderr,"rank=%ld whole length=%d pair_max=%.8g split_max=%.8g whole_top=%td split_top=%td\n",
                             rank,length,whole_pair,whole_delta,w,a);
                CHECK(whole_pair<=1e-5f && w==a);
                if (std::getenv("DS4_TP_TEST_REQUIRE_SPLIT_PARITY")) CHECK(whole_delta<=1e-5f);
            }
            if (length>chunk && std::getenv("DS4_TP_TEST_COMPARE_SPLIT_REPLAY")) {
                CHECK(reset());
                std::vector<float> replay;
                for (int at=0;at<length;) {
                    const int n=std::min(chunk,length-at); const bool last=at+n==length;
                    CHECK(comm->BeginOperation(++sequence,&error));
                    CHECK(model.feed(tokens.data()+at,n,last?1:0,error));
                    if (last) {
                        CHECK(model.logits(replay) && replay.size()==vocab);
                        CHECK(hipMemcpyAsync(device,replay.data(),bytes,hipMemcpyHostToDevice,stream)==hipSuccess);
                        const float* remote=comm->ExchangePartial(device,bytes,stream,&error); CHECK(remote);
                        CHECK(hipMemcpyAsync(other.data(),remote,bytes,hipMemcpyDeviceToHost,stream)==hipSuccess);
                        CHECK(hipStreamSynchronize(stream)==hipSuccess);
                    }
                    CHECK(model.tp_worker_synchronize(error) && comm->CheckQueued(&error) && comm->EndOperation(sequence,&error));
                    at+=n;
                }
                float replay_delta=0, replay_pair=0;
                for (std::size_t i=0;i<vocab;++i) {
                    CHECK(std::isfinite(replay[i]) && std::isfinite(other[i]));
                    replay_delta=std::max(replay_delta,std::fabs(replay[i]-actual[i]));
                    replay_pair=std::max(replay_pair,std::fabs(replay[i]-other[i]));
                }
                std::fprintf(stderr,"rank=%ld split replay length=%d pair_max=%.8g replay_max=%.8g\n",
                             rank,length,replay_pair,replay_delta);
                CHECK(replay_pair<=1e-5f && replay_delta<=1e-5f);
            }
            CHECK(pair_delta<=1e-5f && a==b);
        }
        CHECK(reset());
    }
    if (std::getenv("DS4_TP_TEST_SWEEP_ONLY_EXIT")) {
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
    const bool greedy_only=std::getenv("DS4_TP_TEST_GREEDY_ONLY")!=nullptr;
    for (int phase=0;phase<2;++phase) {
        if (phase==0 && greedy_only) continue;
        if (phase==1 && !greedy_only) {
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
                rank,prompt.size(),greedy_only?0:reference_tokens.size(),top_matches,max_reference_delta,
                greedy_only?0.0:std::sqrt(reference_square/(reference_tokens.size()*vocab)),generated.size());
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
