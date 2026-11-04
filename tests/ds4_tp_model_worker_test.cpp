#include "../src/ds4/ds4_tp_model_worker.hpp"
#include <cstdio>
#include <string>

using namespace strata::ds4::tp2;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "line %d: %s (%s)\n", __LINE__, #x, error.c_str()); return 1; } } while (0)

struct FakeModel {
    int at=0, pending_start=0, pending_count=0, syncs=0, calls=0;
    int chunk=0, limit=32, vocabulary=100;
    bool sync_fail=false, operation_fail=false;
    int pos() const { return at; }
    int max_context() const { return limit; }
    int vocab() const { return vocabulary; }
    int prompt_chunk() const { return chunk; }
    bool feed(const int*, int n, int, std::string& error) {
        ++calls; if (operation_fail) { error="model failure"; return false; } at+=n; return true;
    }
    bool step(int, bool, std::string& error) { return feed(nullptr, 1, 0, error); }
    bool tp_verify_begin(const int*, int n, std::string& error) {
        pending_start=at; pending_count=n; return feed(nullptr, n, 0, error);
    }
    bool tp_verify_commit(int accepted, std::string&) { ++calls; at=pending_start+1+accepted; return true; }
    bool tp_worker_reset(std::string&) { ++calls; at=0; return true; }
    bool tp_worker_synchronize(std::string& error) {
        ++syncs; if (sync_fail) { error="GPU failure"; return false; } return true;
    }
};

int main() {
    std::string error;
    {
        FakeModel m; ModelWorker<FakeModel> w(m);
        CHECK(w.execute({Command::Feed,1,0,{1,2,3}},error) && m.at==3);
        CHECK(w.execute({Command::Step,2,3,{4}},error) && m.at==4);
        CHECK(w.execute({Command::Verify,3,4,{5,6,7}},error) && m.at==7);
        CHECK(w.execute({Command::Verdict,4,6,{1,8}},error) && m.at==6);
        CHECK(w.execute({Command::Reset,5,0,{}},error) && m.at==0);
        CHECK(w.execute({Command::Shutdown,6,0,{}},error));
        CHECK(m.syncs==6 && m.calls==5);
        CHECK(!w.execute({Command::Step,7,0,{2}},error));
    }
    {
        FakeModel m; m.chunk=128; ModelWorker<FakeModel> w(m);
        CHECK(!w.execute({Command::Feed,1,0,{1}},error));
        CHECK(error.find("window-only")!=std::string::npos && m.calls==0);
        const std::string first=error;
        CHECK(!w.execute({Command::Step,2,0,{1}},error) && error==first && m.calls==0);
    }
    {
        FakeModel m; m.chunk=128; ModelWorker<FakeModel> w(m,true);
        CHECK(w.execute({Command::Feed,1,0,std::vector<std::uint32_t>(20,1)},error));
        CHECK(m.at==20 && m.calls==1 && m.syncs==1);
    }
    {
        FakeModel m; ModelWorker<FakeModel> w(m);
        CHECK(w.execute({Command::Verify,1,0,{1,2,3}},error));
        CHECK(!w.execute({Command::Verdict,2,1,{2,4}},error));
        CHECK(error.find("invalid verify verdict")!=std::string::npos && m.calls==1);
    }
    {
        FakeModel m; ModelWorker<FakeModel> w(m);
        CHECK(w.execute({Command::Verify,1,0,{1,2}},error));
        CHECK(!w.execute({Command::Reset,2,0,{}},error));
        CHECK(error.find("verdict")!=std::string::npos && m.calls==1);
    }
    {
        FakeModel m; m.sync_fail=true; ModelWorker<FakeModel> w(m);
        CHECK(!w.execute({Command::Step,1,0,{1}},error) && error=="GPU failure");
        m.sync_fail=false;
        CHECK(!w.execute({Command::Step,2,1,{2}},error) && error=="GPU failure" && m.calls==1);
    }
    {
        FakeModel m; m.operation_fail=true; m.sync_fail=true; ModelWorker<FakeModel> w(m);
        CHECK(!w.execute({Command::Step,1,0,{1}},error) && error=="model failure");
        CHECK(m.syncs==1);
    }
    {
        FakeModel m; ModelWorker<FakeModel> w(m);
        CHECK(!w.execute({Command::Step,1,0,{100}},error) && m.calls==0);
    }
    {
        FakeModel m; m.at=31; ModelWorker<FakeModel> w(m);
        CHECK(!w.execute({Command::Feed,1,31,{1,2}},error) && m.calls==0);
    }
    {
        FakeModel m; m.at=33; ModelWorker<FakeModel> w(m);
        CHECK(!w.execute({Command::Step,1,33,{1}},error) && m.calls==0);
    }
    std::puts("DeepSeek TP2 model worker dispatch and sticky failure passed");
}
