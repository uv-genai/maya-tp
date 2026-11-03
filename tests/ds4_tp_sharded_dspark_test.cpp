// Real rank-0 DSpark proposal and greedy acceptance against a sharded worker.
// Synthetic identity and fixed fixture: this is not a serving implementation.
#include "../src/ds4/ds4_model.hpp"
#include "../src/ds4/ds4_tp_channel.hpp"
#include "../src/ds4/ds4_tp_model_worker.hpp"
#include "../src/ds4/ds4_tp_worker.hpp"
#include "strata/net/verbs.hpp"
#include "ds4_tp_test_identity.hpp"
#include <algorithm>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

using namespace strata::ds4::tp2;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"line %d: %s (%s)\n",__LINE__,#x,error.c_str()); return 1; } } while (0)
int ds4_tp_sharded_dspark_test(int argc,char** argv) {
    if (argc!=10) { std::fprintf(stderr,"usage: --sharded-dspark rank host port rdma pack prompt-tokens expected-tokens output-tokens\n"); return 2; }
    std::string error; char* end=nullptr;
    const long rank=std::strtol(argv[2],&end,10); if (!*argv[2] || *end || rank<0 || rank>1) return 2;
    const long port=std::strtol(argv[4],&end,10); if (!*argv[4] || *end || port<1 || port>65535) return 2;
    std::vector<int> prompt,expected;
    { std::ifstream f(argv[7]); CHECK(f.good()); int t; while(f>>t) prompt.push_back(t); CHECK(f.eof() && !prompt.empty() && prompt.size()<=32768); }
    if (std::string(argv[8])!="-") {
        std::ifstream f(argv[8]); CHECK(f.good()); int t; while(f>>t) expected.push_back(t);
        CHECK(f.eof() && !expected.empty() && expected.size()<=256);
    }
    const bool target_only=std::getenv("DS4_TP_TEST_TARGET_ONLY")!=nullptr;
    CHECK(!expected.empty() || target_only);
    CHECK(setenv("DS4_STREAM","0",1)==0);
    const char* chunk_env=std::getenv("DS4_TP_TEST_CHUNK");
    const int chunk=chunk_env ? std::atoi(chunk_env) : 0;
    if (chunk_env) { CHECK(chunk>=256 && chunk%16==0);
        CHECK(setenv("DS4_FLOW","0",1)==0 && setenv("DS4_YOCO","0",1)==0); }
    std::int64_t context=0; CHECK(ds4_tp_test_context(context,error));
    strata::ds4::Options o; o.pack=argv[6]; o.max_context=context; o.gpus={0}; o.reserve_mib=1024;
    o.tp=1; o.threads=false; o.vision=false; o.dspark=rank==0; o.lookup=false; o.prefill_chunk=chunk;
    o.tp2_chunk_enabled=chunk>0;
    o.expert_cache=false; o.mmap_experts=1; o.profile_readonly=true; o.cpu_threads=14; o.vram_experts=rank==0?1:0;
    o.tp2_rank=static_cast<int>(rank); o.tp2_expert_budget=90ull<<30; o.tp_worker_verify=true;
    strata::ds4::Model model; CHECK(model.load(o,error)); CHECK(model.has_drafter()==(rank==0));
    std::fprintf(stderr,"rank %ld: sharded model loaded, drafter=%d\n",rank,model.has_drafter());
    strata::net::IbrverbsConfig cfg; cfg.rank=static_cast<std::uint32_t>(rank); cfg.bootstrap_host=argv[3];
    cfg.bootstrap_port=static_cast<std::uint16_t>(port); cfg.rdma_device=argv[5];
    auto comm=strata::net::CreateIbrverbsCommunicator(cfg,&error); CHECK(comm);
    Identity id; id.protocol=1; id.world=2; id.rank=cfg.rank; id.hidden=5120; id.layers=40; id.experts=384;
    id.expert_width=2304; id.max_context=static_cast<std::uint64_t>(context);
    id.feature_flags=1|(static_cast<std::uint64_t>(chunk)<<16);
    id.max_partial_bytes=comm->MaxPartialBytes();
    CHECK(ds4_tp_test_identity(id,error));
    Identity peer; std::uint64_t limit=0; CHECK(exchange_identity(*comm,id,peer,limit,error));
    CHECK(model.tp2_attach(*comm,static_cast<std::size_t>(limit),error,rank==0));
    if (rank==1) {
        Worker worker(*comm,static_cast<std::uint64_t>(context),static_cast<std::uint32_t>(model.vocab()),chunk>0?chunk:16);
        ModelWorker<strata::ds4::Model> adapter(model,chunk>0); std::uint64_t processed=0;
        while (!worker.state().stopped()) {
            CHECK(worker.run_one([&](const Control& c,std::string& e) {
                const char* stop=std::getenv("DS4_TP_TEST_STOP_COMMAND");
                if (stop && ((std::string(stop)=="verify" && c.command==Command::Verify) ||
                             (std::string(stop)=="verdict" && c.command==Command::Verdict))) {
                    std::fprintf(stderr,"DS4_TEST_STOP %s\n",stop); std::fflush(stderr);
                    std::raise(SIGSTOP); // The owned monitor kills this stopped test worker.
                }
                return adapter.execute(c,e);
            },error)); ++processed;
        }
        CHECK(worker.state().position()==static_cast<std::uint64_t>(model.pos()));
        CHECK(send_message(*comm,{Command::Shutdown,processed+1,static_cast<std::uint64_t>(model.pos()),{}},chunk>0?chunk:16,error));
        std::printf("rank 1: DSpark worker stopped at position %lld\n",static_cast<long long>(model.pos())); return 0;
    }
    int notifications=0; std::string notified;
    model.tp2_set_failure_handler([&](const std::string& cause) { ++notifications; notified=cause; });
    const bool expect_loss=std::getenv("DS4_TP_TEST_EXPECT_SPEC_LOSS")!=nullptr;
    const auto check_loss=[&]() {
        const std::string first=error;
        if (first.find("peer closed the bootstrap connection")==std::string::npos) return false;
        error.clear();
        if (model.step(prompt[0],true,error) || error!=first) return false;
        if (model.feed(prompt.data(),1,1,error) || error!=first) return false;
        if (notifications!=1 || notified!=first || model.tp2_failure()!=first) return false;
        std::printf("rank 0 preserved first speculative peer-loss cause and rejected later operations: %s\n",first.c_str());
        return true;
    };
    const auto feed_prompt=[&]() {
        for (std::size_t at=0;at<prompt.size();) {
            const auto last=std::min(prompt.size(),at+static_cast<std::size_t>(model.feed_unit()));
            if (!model.feed(prompt.data()+at,static_cast<int>(last-at),last==prompt.size()?1:0,error)) return false;
            at=last;
        }
        return true;
    };
    CHECK(feed_prompt());
    std::vector<int> generated; int y=model.sample(0,1,0,0,0,0); CHECK(y>=0 && y<model.vocab());
    std::size_t rounds=0,offered=0,kept=0;
    const std::size_t count=expected.empty()?64:expected.size();
    while (generated.size()<count) {
        if (!expected.empty() && y!=expected[generated.size()]) {
            const int probe=expected[generated.size()];
            std::fprintf(stderr,"DSpark mismatch at %zu: actual=%d expected=%d; checking committed state\n",
                         generated.size(),y,probe);
            CHECK(model.step(probe,true,error));
            std::vector<float> speculative; CHECK(model.logits(speculative) && !speculative.empty());
            model.reset(); CHECK(model.tp2_failure().empty()); CHECK(feed_prompt());
            for (std::size_t i=0;i<generated.size();++i)
                CHECK(model.step(generated[i],true,error));
            const int replay=model.sample(0,1,0,0,0,0);
            CHECK(model.step(probe,true,error));
            std::vector<float> plain; CHECK(model.logits(plain) && plain.size()==speculative.size());
            float max_delta=0; double sum_square=0;
            for (std::size_t i=0;i<plain.size();++i) {
                CHECK(std::isfinite(plain[i]) && std::isfinite(speculative[i]));
                const float d=std::fabs(plain[i]-speculative[i]); max_delta=std::max(max_delta,d);
                sum_square+=double(d)*d;
            }
            std::fprintf(stderr,"DSpark replay next=%d expected=%d post-probe logits max=%.8g rms=%.8g\n",
                         replay,probe,max_delta,std::sqrt(sum_square/plain.size()));
            CHECK(model.tp2_shutdown(error));
            return 1; // Preserve the failed output-equivalence gate.
        }
        if (y==1 || generated.size()+1==count) { generated.push_back(y); break; }
        if (target_only) {
            CHECK(model.step(y,true,error)); generated.push_back(y);
            y=model.sample(0,1,0,0,0,0); CHECK(y>=0 && y<model.vocab()); continue;
        }
        std::vector<int> drafts; std::vector<float> confidence;
        CHECK(model.draft(y,drafts,confidence,error)); CHECK(drafts.size()==confidence.size());
        drafts.resize(std::min(drafts.size(),count-generated.size()-1));
        const auto eos=std::find(drafts.begin(),drafts.end(),1); if (eos!=drafts.end()) drafts.erase(eos+1,drafts.end());
        if (drafts.empty()) {
            CHECK(model.step(y,true,error)); generated.push_back(y);
            y=model.sample(0,1,0,0,0,0); CHECK(y>=0 && y<model.vocab()); continue;
        }
        for (int t:drafts) CHECK(t>=0 && t<model.vocab());
        const auto start=static_cast<std::uint64_t>(model.pos());
        int accepted=-1,next=-1;
        if (!model.verify(y,drafts,0,1,0,0,0,0,accepted,next,error)) {
            CHECK(expect_loss && check_loss()); return 0;
        }
        CHECK(accepted>=0 && static_cast<std::size_t>(accepted)<=drafts.size() && next>=0 && next<model.vocab());
        CHECK(model.pos()==static_cast<std::int64_t>(start+1+accepted));
        generated.push_back(y);
        for (int j=0;j<accepted;++j) generated.push_back(drafts[static_cast<std::size_t>(j)]);
        CHECK(generated.size()<=count);
        for (std::size_t i=generated.size()-1-static_cast<std::size_t>(accepted);i<generated.size();++i)
            CHECK(generated[i]==expected[i]);
        ++rounds; offered+=drafts.size(); kept+=static_cast<std::size_t>(accepted); y=next;
        std::fprintf(stderr,"DSpark round=%zu offered=%zu accepted=%d emitted=%zu/%zu\n",rounds,drafts.size(),accepted,generated.size(),count);
        if (generated.back()==1) break;
    }
    CHECK(!expect_loss);
    CHECK(expected.empty() || generated==expected);
    CHECK(target_only || (rounds>0 && offered>0 && kept>0));
    CHECK(model.tp2_shutdown(error));
    std::ofstream out(argv[9]); CHECK(out.good()); for (int t:generated) out<<t<<' '; out<<'\n'; CHECK(out.good());
    std::printf("DSpark generated=%zu rounds=%zu offered=%zu accepted=%zu%s\n",generated.size(),rounds,offered,kept,
                expected.empty()?"; target-only diagnostic":"; target-only sequence matched");
    return 0;
}
