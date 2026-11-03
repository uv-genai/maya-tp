// Exercise the typed worker/control boundary with sharded model computation.
// Synthetic identity; this is a review test, not a production launcher.
#include "../src/ds4/ds4_model.hpp"
#include "../src/ds4/ds4_tp_channel.hpp"
#include "../src/ds4/ds4_tp_model_worker.hpp"
#include "../src/ds4/ds4_tp_worker.hpp"
#include "strata/net/verbs.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

using namespace strata::ds4::tp2;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"line %d: %s (%s)\n",__LINE__,#x,error.c_str()); return 1; } } while (0)
int ds4_tp_sharded_control_test(int argc,char** argv) {
    if (argc!=10) {
        std::fprintf(stderr,"usage: --sharded-control rank host port rdma pack prompt-tokens expected-tokens|- output-tokens\n"); return 2;
    }
    std::string error; char* end=nullptr;
    const long rank=std::strtol(argv[2],&end,10); if (!*argv[2] || *end || rank<0 || rank>1) return 2;
    const long port=std::strtol(argv[4],&end,10); if (!*argv[4] || *end || port<1 || port>65535) return 2;
    std::vector<int> prompt, expected;
    { std::ifstream f(argv[7]); CHECK(f.good()); int token; while (f>>token) prompt.push_back(token); CHECK(f.eof() && !prompt.empty() && prompt.size()<=1024); }
    if (std::string(argv[8])!="-") {
        std::ifstream f(argv[8]); CHECK(f.good()); int token; while (f>>token) expected.push_back(token);
        CHECK(f.eof() && !expected.empty() && expected.size()<=256);
    }
    CHECK(setenv("DS4_STREAM","0",1)==0);
    strata::ds4::Options o; o.pack=argv[6]; o.max_context=32768; o.gpus={0}; o.reserve_mib=1024;
    o.tp=1; o.threads=false; o.vision=false; o.dspark=false; o.lookup=false; o.prefill_chunk=0;
    o.expert_cache=false; o.mmap_experts=1; o.profile_readonly=true; o.cpu_threads=14; o.vram_experts=0;
    o.tp2_rank=static_cast<int>(rank); o.tp2_expert_budget=90ull<<30; o.tp_worker_verify=true;
    strata::ds4::Model model; CHECK(model.load(o,error));
    for (int token:prompt) CHECK(token>=0 && token<model.vocab());
    for (int token:expected) CHECK(token>=0 && token<model.vocab());
    std::fprintf(stderr,"rank %ld: sharded model loaded; prompt=%zu\n",rank,prompt.size());
    strata::net::IbrverbsConfig cfg; cfg.rank=static_cast<std::uint32_t>(rank); cfg.bootstrap_host=argv[3];
    cfg.bootstrap_port=static_cast<std::uint16_t>(port); cfg.rdma_device=argv[5];
    auto comm=strata::net::CreateIbrverbsCommunicator(cfg,&error); CHECK(comm);
    Identity id; id.protocol=1; id.world=2; id.rank=cfg.rank; id.hidden=5120; id.layers=40; id.experts=384;
    id.expert_width=2304; id.max_context=32768; id.feature_flags=1; id.max_partial_bytes=comm->MaxPartialBytes();
    id.model_digest[0]=1; id.partition_digest[0]=2; id.build_digest[0]=3;
    Identity peer; std::uint64_t limit=0; CHECK(exchange_identity(*comm,id,peer,limit,error));
    CHECK(model.tp2_attach_test(*comm,static_cast<std::size_t>(limit),error));
    std::uint64_t sequence=0;
    if (rank==1) {
        Worker worker(*comm,32768,static_cast<std::uint32_t>(model.vocab()),16);
        ModelWorker<strata::ds4::Model> adapter(model);
        std::uint64_t processed=0;
        while (!worker.state().stopped()) {
            CHECK(worker.run_one([&](const Control& c,std::string& e){return adapter.execute(c,e);},error));
            ++processed;
        }
        CHECK(worker.state().position()==static_cast<std::uint64_t>(model.pos()));
        CHECK(send_message(*comm,{Command::Shutdown,processed+1,
              static_cast<std::uint64_t>(model.pos()),{}},16,error));
        std::printf("rank 1 worker stopped at position %lld\n",static_cast<long long>(model.pos()));
        return 0;
    }
    auto execute=[&](Command command,const std::vector<std::uint32_t>& words,bool logits)->bool {
        Control c{command,++sequence,command==Command::Reset ? 0u : static_cast<std::uint64_t>(model.pos()),words};
        if (!send_message(*comm,c,16,error) || !comm->BeginOperation(sequence,&error)) return false;
        bool ok=false;
        if (command==Command::Feed) {
            std::vector<int> ts(words.begin(),words.end()); ok=model.feed(ts.data(),static_cast<int>(ts.size()),logits?1:0,error);
        } else if (command==Command::Step) ok=model.step(static_cast<int>(words[0]),logits,error);
        else if (command==Command::Reset) ok=model.tp_worker_reset(error);
        else if (command==Command::Shutdown) ok=true;
        if (!ok || !model.tp_worker_synchronize(error) || !comm->CheckQueued(&error) || !comm->EndOperation(sequence,&error)) return false;
        return true;
    };
    auto feed_prompt=[&]()->bool {
        for (std::size_t at=0;at<prompt.size();) {
            const auto end_at=std::min(prompt.size(),at+std::size_t{16});
            std::vector<std::uint32_t> words(prompt.begin()+static_cast<std::ptrdiff_t>(at),prompt.begin()+static_cast<std::ptrdiff_t>(end_at));
            if (!execute(Command::Feed,words,end_at==prompt.size())) return false;
            at=end_at;
        }
        return true;
    };
    CHECK(feed_prompt());
    std::vector<int> generated; const auto count=expected.empty()?std::size_t{64}:expected.size();
    for (std::size_t i=0;i<count;++i) {
        const int token=model.sample(0,1,0,0,0,0); CHECK(token>=0 && token<model.vocab());
        if (!expected.empty()) CHECK(token==expected[i]);
        generated.push_back(token);
        if (i%8==0) std::fprintf(stderr,"rank 0: %zu/%zu generated\n",i+1,count);
        if (token==1 || i+1==count) break;
        CHECK(execute(Command::Step,{static_cast<std::uint32_t>(token)},true));
    }
    CHECK(expected.empty() || generated==expected);
    CHECK(!generated.empty() && execute(Command::Reset,{},false) && model.pos()==0 && feed_prompt());
    CHECK(model.sample(0,1,0,0,0,0)==generated[0]);
    CHECK(execute(Command::Shutdown,{},false));
    Control done; CHECK(receive_message(*comm,16,done,error));
    CHECK(done.command==Command::Shutdown && done.sequence==sequence+1 && done.position==static_cast<std::uint64_t>(model.pos()));
    std::ofstream out(argv[9]); CHECK(out.good()); for (int token:generated) out<<token<<' '; out<<'\n'; CHECK(out.good());
    std::printf("rank 0 typed control: prompt=%zu generated=%zu position=%lld%s\n",prompt.size(),generated.size(),
                static_cast<long long>(model.pos()),expected.empty()?"":"; previous pair sequence matched");
    return 0;
}
