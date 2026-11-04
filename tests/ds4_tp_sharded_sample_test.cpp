// Statistical p/q speculative-sampling check on a sharded pair. Uses the
// same two-sample chi-square grouping as maya-ds4 --sampler-test.
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
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>

using namespace strata::ds4::tp2;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"line %d: %s (%s)\n",__LINE__,#x,error.c_str()); return 1; } } while (0)
namespace {
double chi_tail(double x,int df) {
    const double a=.5*df,z=.5*x; if (df<=0 || z<=0) return 1;
    const double front=std::exp(-z+a*std::log(z)-std::lgamma(a));
    if (z<a+1) {
        double term=1/a,sum=term;
        for (int i=1;i<10000 && term>sum*1e-15;++i) { term*=z/(a+i); sum+=term; }
        return std::clamp(1-front*sum,0.0,1.0);
    }
    constexpr double tiny=1e-300; double b=z+1-a,c=1/tiny,d=1/b,h=d;
    for (int i=1;i<10000;++i) {
        const double an=-i*(i-a); b+=2; d=an*d+b; if (std::fabs(d)<tiny) d=tiny;
        c=b+an/c; if (std::fabs(c)<tiny) c=tiny; d=1/d; h*=d*c;
        if (std::fabs(d*c-1)<1e-15) break;
    }
    return std::clamp(front*h,0.0,1.0);
}
struct Chi { double x=0,p=1; int groups=0; };
Chi compare(const std::vector<int>& a,const std::vector<int>& b) {
    std::map<int,std::pair<double,double>> counts;
    for (int t:a) counts[t].first++; for (int t:b) counts[t].second++;
    const double na=a.size(),nb=b.size(),small=std::min(na,nb); std::vector<std::pair<double,double>> groups;
    double left_a=0,left_b=0;
    for (const auto& [_,v]:counts) {
        if ((v.first+v.second)*small/(na+nb)>=5) groups.push_back(v);
        else { left_a+=v.first; left_b+=v.second; }
    }
    if (left_a+left_b>0) groups.emplace_back(left_a,left_b);
    const double ka=std::sqrt(nb/na),kb=std::sqrt(na/nb); Chi r;
    for (const auto& [x,y]:groups) r.x+=(ka*x-kb*y)*(ka*x-kb*y)/(x+y);
    r.groups=static_cast<int>(groups.size()); r.p=chi_tail(r.x,r.groups-1); return r;
}
struct Trial { std::vector<int> drafts; std::vector<float> confidence; int accepted=-1,next=-1,first=-1; bool sampled=false; };
}
int ds4_tp_sharded_sample_test(int argc,char** argv) {
    if (argc!=9) { std::fprintf(stderr,"usage: --sharded-sample rank host port rdma pack prompt-tokens trials\n"); return 2; }
    std::string error; char* end=nullptr;
    const long rank=std::strtol(argv[2],&end,10); if (!*argv[2] || *end || rank<0 || rank>1) return 2;
    const long port=std::strtol(argv[4],&end,10); if (!*argv[4] || *end || port<1 || port>65535) return 2;
    const long trials=std::strtol(argv[8],&end,10); if (!*argv[8] || *end || trials<50 || trials>1000) return 2;
    std::vector<int> prompt;
    { std::ifstream f(argv[7]); CHECK(f.good()); int t; while(f>>t) prompt.push_back(t); CHECK(f.eof() && !prompt.empty()); }
    CHECK(setenv("DS4_STREAM","0",1)==0 && setenv("DS4_DSPARK_SAMPLE","1",1)==0);
    const char* chunk_env=std::getenv("DS4_TP_TEST_CHUNK");
    const int chunk=chunk_env ? std::atoi(chunk_env) : 0;
    if (chunk_env) { CHECK(chunk==256); CHECK(setenv("DS4_FLOW","0",1)==0 && setenv("DS4_YOCO","0",1)==0); }
    const auto max_words=static_cast<std::uint32_t>(chunk ? chunk : 16);
    CHECK(prompt.size()<=max_words);
    std::int64_t context=0; CHECK(ds4_tp_test_context(context,error));
    strata::ds4::Options o; o.pack=argv[6]; o.max_context=context; o.gpus={0}; o.reserve_mib=1024;
    o.tp=1; o.threads=false; o.vision=false; o.dspark=rank==0; o.lookup=false; o.prefill_chunk=chunk;
    o.tp2_chunk_enabled=chunk>0;
    o.expert_cache=false; o.mmap_experts=1; o.profile_readonly=true; o.cpu_threads=14; o.vram_experts=rank==0?1:0;
    o.tp2_rank=static_cast<int>(rank); o.tp2_expert_budget=90ull<<30; o.tp_worker_verify=true;
    strata::ds4::Model model; CHECK(model.load(o,error) && model.has_drafter()==(rank==0));
    std::fprintf(stderr,"rank %ld: sampled DSpark test loaded\n",rank);
    strata::net::IbrverbsConfig cfg; cfg.rank=static_cast<std::uint32_t>(rank); cfg.bootstrap_host=argv[3];
    cfg.bootstrap_port=static_cast<std::uint16_t>(port); cfg.rdma_device=argv[5];
    auto comm=strata::net::CreateIbrverbsCommunicator(cfg,&error); CHECK(comm);
    Identity id; id.protocol=1; id.world=2; id.rank=cfg.rank; id.hidden=5120; id.layers=40; id.experts=384;
    id.expert_width=2304; id.max_context=static_cast<std::uint64_t>(context);
    id.feature_flags=1|(static_cast<std::uint64_t>(chunk)<<16); id.max_partial_bytes=comm->MaxPartialBytes();
    CHECK(ds4_tp_test_identity(id,error));
    Identity peer; std::uint64_t limit=0; CHECK(exchange_identity(*comm,id,peer,limit,error));
    CHECK(model.tp2_attach(*comm,static_cast<std::size_t>(limit),error));
    if (rank==1) {
        Worker worker(*comm,static_cast<std::uint64_t>(context),static_cast<std::uint32_t>(model.vocab()),max_words);
        ModelWorker<strata::ds4::Model> adapter(model,chunk>0); std::uint64_t processed=0;
        while (!worker.state().stopped()) {
            CHECK(worker.run_one([&](const Control& c,std::string& e){return adapter.execute(c,e);},error)); ++processed;
        }
        CHECK(worker.state().position()==static_cast<std::uint64_t>(model.pos()));
        CHECK(send_message(*comm,{Command::Shutdown,processed+1,static_cast<std::uint64_t>(model.pos()),{}},max_words,error));
        std::printf("rank 1: sampled worker stopped at position %lld\n",static_cast<long long>(model.pos())); return 0;
    }
    std::uint64_t sequence=0;
    auto begin=[&](Command c,const std::vector<std::uint32_t>& words,std::uint64_t pos)->bool {
        return send_message(*comm,{c,++sequence,pos,words},max_words,error) && comm->BeginOperation(sequence,&error);
    };
    auto finish=[&]()->bool { return model.tp_worker_synchronize(error) && comm->CheckQueued(&error) && comm->EndOperation(sequence,&error); };
    auto reset_feed=[&](bool logits)->bool {
        if (!begin(Command::Reset,{},0) || !model.tp_worker_reset(error) || !finish()) return false;
        std::vector<std::uint32_t> words(prompt.begin(),prompt.end());
        return begin(Command::Feed,words,0) && model.feed(prompt.data(),static_cast<int>(prompt.size()),logits?1:0,error) && finish();
    };
    constexpr float temp=1.f,top_p=1.f,min_p=0.f; constexpr int top_k=40; constexpr std::uint64_t seed=0x734d26a5ULL;
    CHECK(reset_feed(true)); const int greedy=model.sample(0,1,0,0,0,0); CHECK(greedy>=0 && greedy<model.vocab());
    for (int mode=0;mode<3;++mode) {
        const bool penalties=mode==1, lookup=mode==2;
        model.set_penalties(penalties?64:0,penalties?1.3f:1.f,penalties?.1f:0.f,penalties?.1f:0.f);
        int y=-1, point=-1; std::vector<int> plain;
        // Pick the first predetermined continuation with enough probability
        // mass outside its most common token for the chi-square smoke to have
        // at least two meaningful groups. Selection uses only plain draws.
        for (int candidate:{greedy,35,671,6111,5,2677,53624,9544,43,4246,3870}) {
            CHECK(reset_feed(false)); CHECK(begin(Command::Step,{static_cast<std::uint32_t>(candidate)},static_cast<std::uint64_t>(model.pos())));
            CHECK(model.step(candidate,true,error) && finish());
            plain.clear(); plain.reserve(static_cast<std::size_t>(trials)*20); std::map<int,int> counts;
            for (std::uint64_t i=0;i<static_cast<std::uint64_t>(trials)*20;++i) {
                const int t=model.sample(temp,top_p,top_k,min_p,seed,i); CHECK(t>=0 && t<model.vocab());
                plain.push_back(t); ++counts[t];
            }
            int max_count=0;
            for (const auto& [token,n]:counts) if (n>max_count) { max_count=n; point=token; }
            std::fprintf(stderr,"sample mode=%d candidate=%d unique=%zu top=%d/%zu\n",mode,candidate,counts.size(),max_count,plain.size());
            const bool varied=max_count>=static_cast<int>(plain.size()*15/100) &&
                              max_count<=static_cast<int>(plain.size()*85/100);
            if (!varied) continue;
            if (penalties) {
                // Point-q with penalties may always propose a target-impossible
                // token in a particular context. Probe its first deterministic
                // draft, and require at least 10% target mass in independent
                // plain draws before running fresh-counter acceptance trials.
                CHECK(reset_feed(false)); std::vector<int> probe; std::vector<float> conf;
                CHECK(model.draft(candidate,probe,conf,error,temp,top_p,top_k,min_p,seed,0));
                CHECK(!model.drafts_sampled());
                const int support=probe.empty()?0:counts[probe[0]];
                std::fprintf(stderr,"sample mode=1 candidate=%d point-draft target support=%d/%zu\n",candidate,support,plain.size());
                if (support<static_cast<int>(plain.size()/10)) continue;
            }
            y=candidate; break;
        }
        CHECK(y>=0 && point>=0);
        auto trial=[&](std::uint64_t ctr,Trial& out,bool stale_control=false)->bool {
            if (!reset_feed(false) || !model.draft(y,out.drafts,out.confidence,error,temp,top_p,top_k,min_p,seed,ctr)) return false;
            out.sampled=model.drafts_sampled();
            if (out.sampled!=!penalties) { error="unexpected sampled-draft mode"; return false; }
            if (lookup) {
                // Leave a matching sampled-q cache in place, then verify an
                // independently chosen lookup token as a point proposal.
                out.drafts.assign(1,point);
                out.confidence.assign(1,1.0f);
            }
            if (out.drafts.empty()) {
                if (!begin(Command::Step,{static_cast<std::uint32_t>(y)},static_cast<std::uint64_t>(model.pos())) ||
                    !model.step(y,true,error) || !finish()) return false;
                out.accepted=0; out.next=model.sample(temp,top_p,top_k,min_p,seed,ctr); out.first=out.next;
                return out.next>=0 && out.next<model.vocab();
            }
            std::vector<std::uint32_t> words{static_cast<std::uint32_t>(y)};
            for (int t:out.drafts) { if (t<0 || t>=model.vocab()) { error="invalid draft"; return false; } words.push_back(static_cast<std::uint32_t>(t)); }
            const auto at=static_cast<std::uint64_t>(model.pos());
            if (!begin(Command::Verify,words,at) ||
                !model.verify(y,out.drafts,temp,top_p,top_k,min_p,seed,ctr,out.accepted,out.next,error,
                              lookup && !stale_control) || !finish()) return false;
            if (out.accepted<0 || static_cast<std::size_t>(out.accepted)>out.drafts.size() || out.next<0 || out.next>=model.vocab()) {
                error="invalid sampled verdict"; return false;
            }
            if (!begin(Command::Verdict,{static_cast<std::uint32_t>(out.accepted),static_cast<std::uint32_t>(out.next)},
                       static_cast<std::uint64_t>(model.pos())) || !finish()) return false;
            out.first=out.accepted>0?out.drafts[0]:out.next; return true;
        };
        std::vector<int> spec; spec.reserve(trials); int kept=0,offered=0,control_differences=0;
        for (long i=0;i<trials;++i) {
            const auto ctr=(1ull<<32)+static_cast<std::uint64_t>(i)*64; Trial t;
            CHECK(trial(ctr,t)); spec.push_back(t.first); offered+=static_cast<int>(t.drafts.size()); kept+=t.accepted;
            if (i<3) {
                Trial again; CHECK(trial(ctr,again));
                CHECK(again.drafts==t.drafts && again.accepted==t.accepted && again.next==t.next && again.first==t.first &&
                      again.sampled==t.sampled && again.confidence.size()==t.confidence.size());
                for (std::size_t j=0;j<t.confidence.size();++j) CHECK(std::fabs(t.confidence[j]-again.confidence[j])<=1e-6f);
            }
            if (lookup && i<32) {
                Trial stale; CHECK(trial(ctr,stale,true));
                if (stale.first!=t.first || stale.accepted!=t.accepted || stale.next!=t.next) ++control_differences;
            }
            if ((i+1)%10==0) std::fprintf(stderr,"sample mode=%d trials=%ld/%ld\n",mode,i+1,trials);
        }
        const Chi x=compare(plain,spec);
        std::printf("mode=%s plain=%zu speculative=%zu groups=%d chi2=%.5g p=%.6g offered=%d accepted=%d\n",
                    lookup?"lookup-stale-q":penalties?"point+penalties":"drawn",
                    plain.size(),spec.size(),x.groups,x.x,x.p,offered,kept); std::fflush(stdout);
        if (lookup) std::printf("lookup stale-q control: %d of 32 verdicts differed\n",control_differences);
        CHECK(offered>0 && kept>0 && x.groups>=2 && x.p>=.001);
    }
    CHECK(begin(Command::Shutdown,{},static_cast<std::uint64_t>(model.pos())) && finish());
    Control done; CHECK(receive_message(*comm,max_words,done,error));
    CHECK(done.command==Command::Shutdown && done.sequence==sequence+1 && done.position==static_cast<std::uint64_t>(model.pos()));
    return 0;
}
