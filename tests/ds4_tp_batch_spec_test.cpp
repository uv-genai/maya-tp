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
int ds4_tp_batch_spec_test(int argc,char** argv) {
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
    float max_abs=0;std::size_t compared=0,offered=0;
    const int slots[4]{0,1,2,3},anchors[4]{8,9,10,11},lens[4]{6,4,3,3};
    std::vector<std::vector<int>> canonical(4);
    for (int s=0;s<4;++s) for (int j=0;j<lens[s];++j) canonical[s].push_back(8+s+j);
    for (int pass=1;pass<=6;++pass) {
        const int keep[4]{pass,2,3,1};
        std::vector<std::vector<float>> refs(4);
        std::vector<std::vector<int>> refdrafts(4),emitted(4);
        std::vector<bool> more(4,true);more[0]=pass!=6;
        for (int s=0;s<4;++s) {
            CHECK(prepare(s));
            for (int j=0;j<keep[s];++j) CHECK(model.step(canonical[s][j],false,error));
            CHECK(model.step(20+s,true,error));CHECK(model.logits(refs[s]));
            std::vector<float> confidence;CHECK(model.draft(30+s,refdrafts[s],confidence,error));
        }
        for (int s=0;s<4;++s) CHECK(prepare(s));
        std::vector<std::vector<int>> drafts;std::vector<std::vector<float>> confs;
        CHECK(model.draft_tp2_groups(slots,anchors,4,drafts,confs,error));
        for (int s=0;s<4;++s) { CHECK(drafts[s].size()==static_cast<std::size_t>(model.draft_block()) && confs[s].size()==drafts[s].size());offered+=drafts[s].size(); }
        const int* inputs[4]{canonical[0].data(),canonical[1].data(),canonical[2].data(),canonical[3].data()};
        CHECK(model.step_multi_spec(slots,inputs,lens,4,error));
        for (int s=0;s<4;++s) {
            const std::vector<int> proposals(canonical[s].begin()+1,canonical[s].end());int accepted=0,next=-1;
            model.set_penalties(pass==2?64:0,1.1f,0.2f,0.1f);
            model.set_logit_bias(pass==2?std::vector<std::pair<int,float>>{{42,1.0f}}:std::vector<std::pair<int,float>>{});
            CHECK(model.verify_multi(s,proposals,pass==2?0.7f:0,0.9f,40,0.01f,17+s,3,accepted,next,error));
            CHECK(accepted>=0 && accepted<lens[s] && next>=0 && next<model.vocab());
            CHECK(model.seq_pos(s)==static_cast<std::int64_t>(prompts[s].size()+lens[s])); // selection did not commit
            emitted[s].assign(canonical[s].begin()+1,canonical[s].begin()+keep[s]);
            emitted[s].push_back(20+s); // canonical forced final pending token
        }
        model.set_penalties(0,1,0,0);model.set_logit_bias({});
        CHECK(model.tp2_commit_multi(emitted,more,error));
        for (int s=0;s<4;++s) {
            CHECK(model.seq_pos(s)==static_cast<std::int64_t>(prompts[s].size()+keep[s]));
            if (!more[s]) CHECK(model.tp2_begin_sequence(s,error));
            model.seq_bind(s);CHECK(model.step(20+s,true,error));std::vector<float> actual;CHECK(model.logits(actual));
            CHECK(actual.size()==refs[s].size());float worst=0;
            for (std::size_t v=0;v<actual.size();++v) {
                CHECK(std::isfinite(actual[v]) && std::isfinite(refs[s][v]));
                worst=std::max(worst,std::abs(actual[v]-refs[s][v]));
            }
            std::fprintf(stderr,"pass=%d slot=%d keep=%d max_abs=%g\n",pass,s,keep[s],worst);
            CHECK(worst<=1e-5f);max_abs=std::max(max_abs,worst);++compared;
            std::vector<int> actualdrafts;std::vector<float> confidence;
            CHECK(model.draft(30+s,actualdrafts,confidence,error));CHECK(actualdrafts==refdrafts[s]);
        }
    }
    CHECK(model.tp2_shutdown(error));
    std::printf("{\"passed\":true,\"rank\":0,\"compared\":%zu,\"offered\":%zu,\"max_abs\":%g}\n",compared,offered,max_abs);return 0;
}
#undef CHECK
