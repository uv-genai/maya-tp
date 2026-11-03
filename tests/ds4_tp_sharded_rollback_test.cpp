// Force every acceptance count on a sharded model and compare its next logits
// and, optionally, rank-0 DSpark drafts with a target-only continuation.
#include "../src/ds4/ds4_model.hpp"
#include "../src/ds4/ds4_tp_channel.hpp"
#include "../src/ds4/ds4_tp_model_worker.hpp"
#include "../src/ds4/ds4_tp_worker.hpp"
#include "strata/net/verbs.hpp"
#include "ds4_tp_test_identity.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>

using namespace strata::ds4::tp2;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"line %d: %s (%s)\n",__LINE__,#x,error.c_str()); return 1; } } while (0)
int ds4_tp_sharded_rollback_test(int argc,char** argv) {
    if (argc!=7) { std::fprintf(stderr,"usage: --sharded-rollback[-dspark] rank host port rdma pack\n"); return 2; }
    const bool dspark=std::string(argv[1])=="--sharded-rollback-dspark";
    std::string error; char* end=nullptr;
    const long rank=std::strtol(argv[2],&end,10); if (!*argv[2] || *end || rank<0 || rank>1) return 2;
    const long port=std::strtol(argv[4],&end,10); if (!*argv[4] || *end || port<1 || port>65535) return 2;
    CHECK(setenv("DS4_STREAM","0",1)==0);
    const char* chunk_env=std::getenv("DS4_TP_TEST_CHUNK");
    const int chunk=chunk_env ? std::atoi(chunk_env) : 0;
    if (chunk_env) { CHECK(chunk==256); CHECK(setenv("DS4_FLOW","0",1)==0 && setenv("DS4_YOCO","0",1)==0); }
    const auto max_words=static_cast<std::uint32_t>(chunk ? chunk : 16);
    std::int64_t context=0; CHECK(ds4_tp_test_context(context,error));
    strata::ds4::Options o; o.pack=argv[6]; o.max_context=context; o.gpus={0}; o.reserve_mib=1024;
    o.tp=1; o.threads=false; o.vision=false; o.dspark=dspark && rank==0; o.lookup=false; o.prefill_chunk=chunk;
    o.tp2_chunk_enabled=chunk>0;
    o.expert_cache=false; o.mmap_experts=1; o.profile_readonly=true; o.cpu_threads=14; o.vram_experts=0;
    o.tp2_rank=static_cast<int>(rank); o.tp2_expert_budget=90ull<<30; o.tp_worker_verify=true;
    if (o.dspark) o.vram_experts=1;
    strata::ds4::Model model; CHECK(model.load(o,error));
    CHECK(model.has_drafter()==o.dspark);
    std::fprintf(stderr,"rank %ld: rollback test model loaded\n",rank);
    strata::net::IbrverbsConfig cfg; cfg.rank=static_cast<std::uint32_t>(rank); cfg.bootstrap_host=argv[3];
    cfg.bootstrap_port=static_cast<std::uint16_t>(port); cfg.rdma_device=argv[5];
    auto comm=strata::net::CreateIbrverbsCommunicator(cfg,&error); CHECK(comm);
    Identity id; id.protocol=1; id.world=2; id.rank=cfg.rank; id.hidden=5120; id.layers=40; id.experts=384;
    id.expert_width=2304; id.max_context=static_cast<std::uint64_t>(context);
    id.feature_flags=1|(static_cast<std::uint64_t>(chunk)<<16);
    id.max_partial_bytes=comm->MaxPartialBytes();
    CHECK(ds4_tp_test_identity(id,error));
    Identity peer; std::uint64_t limit=0; CHECK(exchange_identity(*comm,id,peer,limit,error));
    CHECK(model.tp2_attach(*comm,static_cast<std::size_t>(limit),error));
    std::uint64_t sequence=0;
    if (rank==1) {
        Worker worker(*comm,static_cast<std::uint64_t>(context),static_cast<std::uint32_t>(model.vocab()),max_words);
        ModelWorker<strata::ds4::Model> adapter(model,chunk>0); std::uint64_t processed=0;
        while (!worker.state().stopped()) {
            CHECK(worker.run_one([&](const Control& c,std::string& e){return adapter.execute(c,e);},error)); ++processed;
        }
        CHECK(worker.state().position()==static_cast<std::uint64_t>(model.pos()));
        CHECK(send_message(*comm,{Command::Shutdown,processed+1,static_cast<std::uint64_t>(model.pos()),{}},max_words,error));
        std::printf("rank 1 completed rollback control; position=%lld\n",static_cast<long long>(model.pos())); return 0;
    }
    auto execute=[&](Command command,const std::vector<std::uint32_t>& words,bool logits,std::uint64_t position)->bool {
        Control c{command,++sequence,position,words};
        if (!send_message(*comm,c,max_words,error) || !comm->BeginOperation(sequence,&error)) return false;
        bool ok=false;
        if (command==Command::Feed || command==Command::Verify) {
            std::vector<int> ts(words.begin(),words.end());
            ok=command==Command::Feed ? model.feed(ts.data(),static_cast<int>(ts.size()),logits?1:0,error)
                                      : model.tp_verify_begin(ts.data(),static_cast<int>(ts.size()),error);
        } else if (command==Command::Step) ok=model.step(static_cast<int>(words[0]),logits,error);
        else if (command==Command::Verdict) ok=model.tp_verify_commit(static_cast<int>(words[0]),error);
        else if (command==Command::Reset) ok=model.tp_worker_reset(error);
        else if (command==Command::Shutdown) ok=true;
        if (!ok || !model.tp_worker_synchronize(error) || !comm->CheckQueued(&error) || !comm->EndOperation(sequence,&error)) return false;
        return true;
    };
    int prompt_len=3;
    if (const char* v=std::getenv("DS4_TP_TEST_PROMPT_LEN")) {
        char* end=nullptr; const long n=std::strtol(v,&end,10);
        CHECK(*v && end && !*end && n>=1 && n<=8192 && n<=context-32);
        prompt_len=static_cast<int>(n);
    }
    std::vector<std::uint32_t> prompt;
    for (int i=0;i<prompt_len;++i) prompt.push_back(static_cast<std::uint32_t>(5+i%3));
    int verify_rows=4;
    if (const char* v=std::getenv("DS4_TP_TEST_VERIFY_ROWS")) {
        char* end=nullptr; const long n=std::strtol(v,&end,10);
        CHECK(*v && end && !*end && n>=2 && n<=16); verify_rows=static_cast<int>(n);
    }
    std::vector<std::uint32_t> window;
    for (int i=0;i<verify_rows;++i) window.push_back(static_cast<std::uint32_t>(8+i));
    const std::uint32_t probe=static_cast<std::uint32_t>(8+verify_rows);
    std::fprintf(stderr,"rollback prefix=%d verify_rows=%d context=%lld\n",
                 prompt_len,verify_rows,static_cast<long long>(context));
    const auto feed_prompt=[&]() {
        for (std::size_t at=0;at<prompt.size();) {
            const auto end_at=std::min(prompt.size(),at+static_cast<std::size_t>(max_words));
            const std::vector<std::uint32_t> part(prompt.begin()+static_cast<std::ptrdiff_t>(at),
                                                   prompt.begin()+static_cast<std::ptrdiff_t>(end_at));
            if (!execute(Command::Feed,part,false,static_cast<std::uint64_t>(model.pos()))) return false;
            at=end_at;
        }
        return true;
    };
    for (int accepted=0;accepted<verify_rows;++accepted) {
        CHECK(execute(Command::Reset,{},false,0));
        CHECK(feed_prompt());
        const std::vector<std::uint32_t> kept(window.begin(),window.begin()+accepted+1);
        CHECK(execute(Command::Feed,kept,false,static_cast<std::uint64_t>(model.pos())));
        std::vector<int> baseline_drafts; std::vector<float> baseline_conf;
        if (dspark) CHECK(model.draft(probe,baseline_drafts,baseline_conf,error) && !baseline_drafts.empty());
        CHECK(execute(Command::Step,{probe},true,static_cast<std::uint64_t>(model.pos())));
        std::vector<float> baseline; CHECK(model.logits(baseline) && !baseline.empty());
        CHECK(execute(Command::Reset,{},false,0));
        CHECK(feed_prompt());
        const auto start=static_cast<std::uint64_t>(model.pos());
        CHECK(execute(Command::Verify,window,false,start));
        CHECK(execute(Command::Verdict,{static_cast<std::uint32_t>(accepted),probe},false,start+1+accepted));
        CHECK(model.pos()==static_cast<std::int64_t>(start+1+accepted));
        std::vector<int> actual_drafts; std::vector<float> actual_conf;
        if (dspark) {
            CHECK(model.draft(probe,actual_drafts,actual_conf,error));
            CHECK(actual_drafts==baseline_drafts && actual_conf.size()==baseline_conf.size());
            for (std::size_t i=0;i<actual_conf.size();++i) CHECK(std::fabs(actual_conf[i]-baseline_conf[i])<=1e-4f);
        }
        CHECK(execute(Command::Step,{probe},true,static_cast<std::uint64_t>(model.pos())));
        std::vector<float> actual; CHECK(model.logits(actual) && actual.size()==baseline.size());
        float delta=0;
        for (std::size_t i=0;i<actual.size();++i) {
            CHECK(std::isfinite(actual[i]) && std::isfinite(baseline[i]));
            delta=std::max(delta,std::fabs(actual[i]-baseline[i]));
        }
        std::printf("accepted=%d next_logits_max=%.8g drafts=%zu\n",accepted,delta,actual_drafts.size()); std::fflush(stdout);
        CHECK(delta<=1e-3f);
    }
    CHECK(execute(Command::Shutdown,{},false,static_cast<std::uint64_t>(model.pos())));
    Control done; CHECK(receive_message(*comm,max_words,done,error));
    CHECK(done.command==Command::Shutdown && done.sequence==sequence+1 && done.position==static_cast<std::uint64_t>(model.pos()));
    std::puts("rank 0: zero, partial and full rollback parity passed"); return 0;
}
