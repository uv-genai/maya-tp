// Internal rank loading is deliberately inaccessible to inference until RDMA
// reductions are installed. Exercise this guard against a real model pack.
#include "../src/ds4/ds4_model.hpp"
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"line %d: %s (%s)\n",__LINE__,#x,error.c_str()); return 1; } } while (0)
int ds4_tp_model_load_test(int argc, char** argv) {
    if (argc!=5) { std::fprintf(stderr,"usage: --rank-model-load rank budget-GiB pack-dir\n"); return 2; }
    char* end=nullptr;
    const long rank=std::strtol(argv[2],&end,10);
    if (!*argv[2] || *end || rank<0 || rank>1 || *argv[3]=='-') return 2;
    const auto gib=std::strtoull(argv[3],&end,10);
    if (!*argv[3] || *end || gib==0 || gib>std::numeric_limits<std::size_t>::max()/(1ull<<30)) return 2;
    if (setenv("DS4_STREAM","0",1)!=0) return 2;
    std::string error;
    strata::ds4::Options o;
    o.pack=argv[4]; o.max_context=32768; o.gpus={0}; o.reserve_mib=1024;
    o.tp=1; o.prefill_chunk=0; o.vision=false; o.dspark=false; o.lookup=false;
    o.expert_cache=false; o.mmap_experts=1; o.vram_experts=0; o.profile_readonly=true;
    o.tp_worker_verify=true; o.tp2_rank=static_cast<int>(rank);
    o.tp2_expert_budget=static_cast<std::size_t>(gib)*(std::size_t{1}<<30); o.verbose=true;
    {
        auto bad=o; bad.tp2_rank=2; strata::ds4::Model m;
        CHECK(!m.load(bad,error) && error.find("invalid rank")!=std::string::npos);
    }
    {
        auto bad=o; bad.reserve_mib=-1; strata::ds4::Model m;
        CHECK(!m.load(bad,error) && error.find("experimental rank loading")!=std::string::npos);
    }
    {
        auto bad=o; bad.dspark=true; strata::ds4::Model m;
        CHECK(!m.load(bad,error) && error.find("experimental rank loading")!=std::string::npos);
    }
    strata::ds4::Model m;
    error.clear(); CHECK(m.load(o,error));
    const int token=1; std::vector<float> logits;
    CHECK(!m.feed(&token,1,1,error) && error.find("load-only")!=std::string::npos);
    CHECK(!m.step(token,true,error) && error.find("load-only")!=std::string::npos);
    CHECK(!m.tp_verify_begin(&token,1,error) && error.find("load-only")!=std::string::npos);
    CHECK(m.sample(0,1,0,0,0,0)==-1 && !m.logits(logits) && !m.restore(0));
    CHECK(m.pos()==0);
    std::printf("rank=%ld model loaded; inference and state restoration rejected\n",rank);
    return 0;
}
