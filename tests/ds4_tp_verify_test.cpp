// Model-backed worker verification rollback parity. Requires a real pack + GPU.
#include "../src/ds4/ds4_model.hpp"
#include "../src/ds4/ds4_tp_model_worker.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
int ds4_tp_rank_load_test(int argc, char** argv);
int ds4_tp_dense_load_test(int argc, char** argv);
#if defined(STRATA_DS4_TP2_TEST_RDMA)
int ds4_tp_pair_smoke(int argc, char** argv);
int ds4_tp_expert_rdma_test(int argc, char** argv);
int ds4_tp_dense_rdma_test(int argc, char** argv);
#endif
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"line %d: %s (%s)\n",__LINE__,#x,error.c_str()); return 1; } } while (0)
int main(int argc,char** argv) {
    if (argc > 1 && std::string(argv[1])=="--rank-load") return ds4_tp_rank_load_test(argc,argv);
    if (argc > 1 && std::string(argv[1])=="--dense-load") return ds4_tp_dense_load_test(argc,argv);
#if defined(STRATA_DS4_TP2_TEST_RDMA)
    if (argc > 1 && std::string(argv[1])=="--rdma") return ds4_tp_pair_smoke(argc,argv);
    if (argc > 1 && std::string(argv[1])=="--rdma-expert") return ds4_tp_expert_rdma_test(argc,argv);
    if (argc > 1 && std::string(argv[1])=="--rdma-dense") return ds4_tp_dense_rdma_test(argc,argv);
#endif
    if(argc!=3 || (std::string(argv[2])!="on" && std::string(argv[2])!="none" && std::string(argv[2])!="off" && std::string(argv[2])!="dispatch")) {
        std::fprintf(stderr,"usage: pack-dir on|none|off|dispatch (none/dispatch reserve worker rollback buffers)\n"); return 2;
    }
    std::string error;
    strata::ds4::Options o; o.pack=argv[1]; o.max_context=32768; o.gpus={0}; o.reserve_mib=1024;
    o.tp=1; o.threads=true; o.vision=false; o.profile_readonly=true; o.profile=o.pack+"/expert-profile.bin";
    o.prefill_chunk=0; // Compare the same window path as speculative verification.
    o.dspark=std::string(argv[2])=="on";
    o.tp_worker_verify=std::string(argv[2])=="none" || std::string(argv[2])=="dispatch";
    strata::ds4::Model m; CHECK(m.load(o,error));
    if (std::string(argv[2])=="dispatch") {
        using namespace strata::ds4::tp2;
        ModelWorker<strata::ds4::Model> worker(m);
        CHECK(worker.execute({Command::Feed,1,0,{5,6,7}},error) && m.pos()==3);
        CHECK(worker.execute({Command::Step,2,3,{8}},error) && m.pos()==4);
        CHECK(worker.execute({Command::Verify,3,4,{9,10,11}},error) && m.pos()==7);
        CHECK(worker.execute({Command::Verdict,4,6,{1,12}},error) && m.pos()==6);
        CHECK(worker.execute({Command::Reset,5,0,{}},error) && m.pos()==0);
        CHECK(worker.execute({Command::Shutdown,6,0,{}},error));
        std::puts("DeepSeek real model worker dispatch passed");
        return 0;
    }
    const int prompt[]={5,6,7}, window[]={8,9,10}, probe=11;
    CHECK(!m.tp_verify_commit(0,error));
    CHECK(!m.tp_verify_begin(window,1,error));
    if (!o.dspark && !o.tp_worker_verify) {
        CHECK(!m.tp_verify_begin(window,3,error));
        CHECK(error.find("rollback buffers")!=std::string::npos);
        int accepted=-1, next=-1;
        CHECK(!m.verify(window[0],{window[1]},0.0f,1.0f,0,0.0f,0,0,accepted,next,error));
        CHECK(error.find("rollback buffers")!=std::string::npos);
        CHECK(m.pos()==0);
        std::puts("DeepSeek worker correctly rejected verification without rollback buffers");
        return 0;
    }
    for(int accepted=0;accepted<=2;++accepted) {
        std::vector<float> reference, actual, ref_conf, actual_conf;
        std::vector<int> ref_drafts, actual_drafts;
        m.reset(); CHECK(m.feed(prompt,3,0,error));
        CHECK(m.feed(window,accepted+1,0,error));
        const auto position=m.pos(); CHECK(position==4+accepted);
        if(m.has_drafter()) CHECK(m.draft(probe,ref_drafts,ref_conf,error));
        CHECK(m.feed(&probe,1,1,error)); CHECK(m.logits(reference));
        m.reset(); CHECK(m.feed(prompt,3,0,error));
        CHECK(m.tp_verify_begin(window,3,error)); CHECK(m.pos()==6);
        CHECK(!m.tp_verify_begin(window,3,error));
        CHECK(!m.tp_verify_commit(-1,error));
        CHECK(!m.tp_verify_commit(3,error));
        CHECK(m.pos()==6);
        CHECK(m.tp_verify_commit(accepted,error) && m.pos()==position);
        CHECK(!m.tp_verify_commit(0,error));
        if(m.has_drafter()) CHECK(m.draft(probe,actual_drafts,actual_conf,error));
        if (ref_drafts!=actual_drafts || ref_conf.size()!=actual_conf.size()) {
            std::fprintf(stderr,"accepted=%d draft mismatch: reference",accepted);
            for (auto t:ref_drafts) std::fprintf(stderr," %d",t);
            std::fprintf(stderr,"; worker");
            for (auto t:actual_drafts) std::fprintf(stderr," %d",t);
            std::fprintf(stderr,"\n");
        }
        bool drafts_match = ref_drafts==actual_drafts && ref_conf.size()==actual_conf.size();
        if (drafts_match) for(std::size_t i=0;i<ref_conf.size();++i)
            drafts_match = drafts_match && std::fabs(ref_conf[i]-actual_conf[i])<=1e-4f;
        CHECK(m.feed(&probe,1,1,error)); CHECK(m.logits(actual));
        CHECK(reference.size()==actual.size() && !reference.empty());
        float max_delta=0;
        for(std::size_t i=0;i<reference.size();++i) {
            CHECK(std::isfinite(reference[i]) && std::isfinite(actual[i]));
            max_delta=std::max(max_delta,std::fabs(reference[i]-actual[i]));
        }
        std::printf("accepted=%d logits max_abs=%.8g drafts=%zu\n",accepted,max_delta,actual_drafts.size());
        CHECK(max_delta<=1e-3f);
        CHECK(drafts_match);
    }
    // A failed begin cannot be committed; the model is discarded/reset.
    m.reset(); const int invalid[]={8,-1};
    CHECK(!m.tp_verify_begin(invalid,2,error)); CHECK(!m.tp_verify_commit(0,error));
    m.reset(); CHECK(m.pos()==0);
    std::puts("DeepSeek worker verify/commit model state parity passed");
}
