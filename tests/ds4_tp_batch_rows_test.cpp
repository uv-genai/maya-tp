// Real XS/S ordinary batch rows compared with independent selected-slot calls.
#include "../src/ds4/ds4_model.hpp"
#include "../src/ds4/ds4_tp_identity.hpp"
#include "../src/ds4/ds4_tp_channel.hpp"
#include "../src/ds4/ds4_tp_worker.hpp"
#include "../src/ds4/ds4_tp_model_worker.hpp"
#include "strata/net/verbs.hpp"
#include "ds4_tp_test_identity.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"line %d: %s (%s)\n",__LINE__,#x,error.c_str()); return 1; } } while (0)
int ds4_tp_batch_rows_test(int argc,char** argv) {
    if (argc!=7) return 2;
    std::string error; char* end=nullptr;
    const long rank=std::strtol(argv[2],&end,10); if (!*argv[2] || *end || rank<0 || rank>1) return 2;
    const long port=std::strtol(argv[4],&end,10); if (!*argv[4] || *end || port<1 || port>65535) return 2;
    std::int64_t context=0; CHECK(ds4_tp_test_context(context,error));
    CHECK(setenv("DS4_STREAM","0",1)==0 && setenv("DS4_FLOW","0",1)==0 && setenv("DS4_YOCO","0",1)==0);
    strata::ds4::Options o; o.pack=argv[6]; o.max_context=context; o.gpus={0}; o.reserve_mib=1024;
    o.tp=1; o.threads=false; o.vision=false; o.dspark=rank==0; o.lookup=false;
    o.prefill_chunk=256; o.tp2_chunk_enabled=true; o.expert_cache=false; o.mmap_experts=1;
    o.profile_readonly=true; o.cpu_threads=14; o.vram_experts=rank==0?1:0;
    o.tp2_rank=static_cast<int>(rank); o.tp2_expert_budget=90ull<<30; o.tp_worker_verify=true; o.seqs=4;
    strata::ds4::Model model; CHECK(model.load(o,error)); CHECK(model.tp2_sequence_layout(error));
    strata::net::IbrverbsConfig cfg; cfg.rank=static_cast<std::uint32_t>(rank); cfg.bootstrap_host=argv[3];
    cfg.bootstrap_port=static_cast<std::uint16_t>(port); cfg.rdma_device=argv[5];
    auto comm=strata::net::CreateIbrverbsCommunicator(cfg,&error); CHECK(comm);
    strata::ds4::tp2::Identity id,peer; const auto g=model.tp2_geometry(); CHECK(ds4_tp_test_identity(id,error));
    id.world=2; id.rank=cfg.rank; id.hidden=g.hidden; id.layers=g.layers; id.experts=g.experts;
    id.expert_width=g.expert_width; id.max_context=context; id.feature_flags=1ull|(256ull<<16);
    CHECK(strata::ds4::tp2::configure_batch_identity(id,4,error));
    id.max_partial_bytes=comm->MaxPartialBytes();std::uint64_t limit=0;
    CHECK(strata::ds4::tp2::exchange_identity(*comm,id,peer,limit,error));
    CHECK(model.tp2_attach(*comm,static_cast<std::size_t>(limit),error,rank==0));
    if (rank==1) {
        strata::ds4::tp2::Worker worker(*comm,context,model.vocab(),256,4);
        strata::ds4::tp2::ModelWorker<strata::ds4::Model> adapter(model,true,4);
        while (!worker.state().stopped()) CHECK(worker.run_one([&](const auto& c,std::string& e) { return adapter.execute(c,e); },error));
        CHECK(model.tp2_sequence_layout(error));
        std::printf("{\"passed\":true,\"rank\":1}\n");return 0;
    }
    std::vector<std::vector<int>> prompts(4);
    for (int s=0;s<4;++s) {
        for (int i=0;i<505+s*2;++i) prompts[s].push_back(5+(i+s)%3);
        CHECK(model.tp2_begin_sequence(s,error));
    }
    const auto prepare=[&](int s) {
        model.seq_bind(s);model.reset();
        if (!model.tp2_failure().empty()) return false;
        for (std::size_t at=0;at<prompts[s].size();at+=256) {
            const int n=static_cast<int>(std::min<std::size_t>(256,prompts[s].size()-at));
            if (!model.feed(prompts[s].data()+at,n,0,error)) return false;
        }
        return true;
    };
    std::vector<std::vector<float>> refs(4);
    for (int s=0;s<4;++s) {
        CHECK(prepare(s));CHECK(model.step(8+s,true,error));CHECK(model.logits(refs[s]));
        CHECK(refs[s].size()==static_cast<std::size_t>(model.vocab()));
    }
    float max_abs=0;std::size_t compared=0;
    for (int n=1;n<=4;++n) {
        int slots[4]{3,1,0,2},tokens[4]{};
        for (int i=0;i<n;++i) { CHECK(prepare(slots[i]));tokens[i]=8+slots[i]; }
        CHECK(model.step_multi(slots,tokens,n,error));std::vector<float> all;CHECK(model.logits(all));
        CHECK(all.size()==static_cast<std::size_t>(n)*model.vocab());
        for (int i=0;i<n;++i) {
            const auto& ref=refs[slots[i]];float worst=0;
            for (int v=0;v<model.vocab();++v) {
                const float a=all[static_cast<std::size_t>(i)*model.vocab()+v];CHECK(std::isfinite(a) && std::isfinite(ref[v]));
                worst=std::max(worst,std::abs(a-ref[v]));
            }
            const int want=static_cast<int>(std::max_element(ref.begin(),ref.end())-ref.begin());
            CHECK(model.sample_multi(i,0,1,0,0,0,0)==want);
            CHECK(model.seq_pos(slots[i])==static_cast<std::int64_t>(prompts[slots[i]].size()+1));
            std::fprintf(stderr,"rows=%d slot=%d max_abs=%g\n",n,slots[i],worst);
            CHECK(worst<=1e-5f);max_abs=std::max(max_abs,worst);++compared;
        }
    }
    CHECK(model.tp2_shutdown(error));
    std::printf("{\"passed\":true,\"rank\":0,\"compared\":%zu,\"max_abs\":%g}\n",compared,max_abs);return 0;
}
#undef CHECK
