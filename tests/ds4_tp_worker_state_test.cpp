#include "../src/ds4/ds4_tp_worker_state.hpp"
#include <cstdio>
using namespace strata::ds4::tp2;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"line %d: %s (%s)\n",__LINE__,#x,error.c_str()); return 1; } } while (0)
int main() {
    std::string error;
    WorkerState w(16, 100, 16);
    CHECK(w.begin({Command::Feed,1,0,{3,4}},error));
    CHECK(w.position()==0); // no state commit before engine completion
    CHECK(w.complete(true,"",error) && w.position()==2);
    CHECK(w.begin({Command::Step,2,2,{5}},error));
    CHECK(w.complete(true,"",error) && w.position()==3);
    CHECK(w.begin({Command::Verify,3,3,{6,7,8,9}},error));
    CHECK(w.complete(true,"",error) && w.awaiting_verdict() && w.position()==3);
    CHECK(w.begin({Command::Verdict,4,6,{2,10}},error));
    CHECK(w.complete(true,"",error) && !w.awaiting_verdict() && w.position()==6);
    CHECK(w.begin({Command::Reset,5,0,{}},error));
    CHECK(w.complete(true,"",error) && w.position()==0);
    CHECK(w.begin({Command::Shutdown,6,0,{}},error));
    CHECK(w.complete(true,"",error) && w.stopped());
    CHECK(!w.begin({Command::Step,7,0,{1}},error));

    // Each failed protocol transition poisons this pair with its first cause.
    WorkerState seq(10,100,16);
    CHECK(!seq.begin({Command::Step,2,0,{1}},error));
    const auto first=error; CHECK(seq.failed());
    CHECK(!seq.begin({Command::Step,1,0,{1}},error) && error==first);
    WorkerState vocab(10,100,16);
    CHECK(!vocab.begin({Command::Step,1,0,{100}},error) && vocab.failed());
    WorkerState bounds(3,100,16);
    CHECK(!bounds.begin({Command::Feed,1,0,{1,2,3,4}},error) && bounds.failed());
    WorkerState position(10,100,16);
    CHECK(!position.begin({Command::Step,1,1,{1}},error) && position.failed());
    WorkerState invalid_verdict(10,100,16);
    CHECK(invalid_verdict.begin({Command::Verify,1,0,{1,2,3}},error));
    CHECK(invalid_verdict.complete(true,"",error));
    CHECK(!invalid_verdict.begin({Command::Verdict,2,4,{3,2}},error) && invalid_verdict.failed());
    WorkerState missing_verdict(10,100,16);
    CHECK(missing_verdict.begin({Command::Verify,1,0,{1,2}},error));
    CHECK(missing_verdict.complete(true,"",error));
    CHECK(!missing_verdict.begin({Command::Step,2,0,{1}},error) && missing_verdict.failed());
    WorkerState transport(10,100,16);
    CHECK(transport.begin({Command::Step,1,0,{1}},error));
    CHECK(!transport.complete(false,"verbs peer closed",error) && transport.position()==0);
    CHECK(!transport.begin({Command::Step,1,0,{1}},error) && error=="verbs peer closed");
    std::puts("DeepSeek TP2 worker state: transaction, rollback verdict, boundaries and sticky failure passed");
}
