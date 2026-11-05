#include "../src/ds4/ds4_tp_worker_state.hpp"
#include <cstdio>
#include <limits>
using namespace strata::ds4::tp2;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"line %d: %s (%s)\n",__LINE__,#x,error.c_str()); return 1; } } while(0)
int main() {
    std::string error;BatchLimits limits{4,100,128};
    BatchPlan p{{{2,1ull<<40,5,{1,2,3}},{0,7,20,{4,5}},{3,9,63,{6}},{1,11,0,{7,8,9,10}}}};
    std::vector<std::uint32_t> wire;CHECK(encode_batch(p,limits,false,wire,error));
    BatchPlan decoded;CHECK(decode_batch(wire,limits,false,decoded,error));
    CHECK(decoded.members[0].generation==(1ull<<40) && decoded.members[3].tokens==p.members[3].tokens);
    for (std::size_t n=0;n<wire.size();++n) {
        auto short_wire=std::vector<std::uint32_t>(wire.begin(),wire.begin()+static_cast<std::ptrdiff_t>(n));
        CHECK(!decode_batch(short_wire,limits,false,decoded,error));
    }
    auto badwire=wire;badwire.push_back(0);CHECK(!decode_batch(badwire,limits,false,decoded,error));
    badwire=wire;badwire[2]++;CHECK(!decode_batch(badwire,limits,false,decoded,error));
    badwire=wire;badwire[0]++;CHECK(!decode_batch(badwire,limits,false,decoded,error));
    CHECK(!valid_batch(p,limits,true,error));
    for (int kind=0;kind<8;++kind) {
        auto bad=p;
        switch (kind) {
            case 0:bad.members[1].slot=2;break;
            case 1:bad.members[0].slot=4;break;
            case 2:bad.members[0].generation=0;break;
            case 3:bad.members[0].position=std::numeric_limits<std::uint64_t>::max();break;
            case 4:bad.members[0].tokens.clear();break;
            case 5:bad.members[0].tokens.assign(17,1);break;
            case 6:bad.members[0].tokens[0]=100;break;
            case 7:bad.members.clear();break;
        }
        CHECK(!valid_batch(bad,limits,false,error));
    }
    auto wide=p;wide.members[0].position=1ull<<41;
    CHECK(encode_batch(wide,{4,100,1ull<<44},false,wire,error));
    CHECK(decode_batch(wire,{4,100,1ull<<44},false,decoded,error));
    CHECK(decoded.members[0].position==(1ull<<41));
    const std::uint64_t target=(1ull<<42)+19;
    BatchCommit commit{target,{}};
    for (const auto& m:p.members) commit.members.push_back({m.slot,m.generation,m.position,true,{99}});
    for (std::size_t keep=1;keep<=4;++keep) {
        auto c=commit;
        for (std::size_t i=0;i<p.members.size();++i) {
            const auto n=std::min(keep,p.members[i].tokens.size());
            c.members[i].emitted.assign(p.members[i].tokens.begin()+1,p.members[i].tokens.begin()+static_cast<std::ptrdiff_t>(n));
            c.members[i].emitted.push_back(99);
        }
        CHECK(encode_commit(c,p,target,limits,wire,error));
        BatchCommit back;CHECK(decode_commit(wire,p,target,limits,back,error));
        CHECK(back.target==target && back.members[3].emitted.size()==keep);
        for (std::size_t n=0;n<wire.size();++n)
            CHECK(!decode_commit(std::vector<std::uint32_t>(wire.begin(),wire.begin()+static_cast<std::ptrdiff_t>(n)),p,target,limits,back,error));
        badwire=wire;badwire.push_back(0);CHECK(!decode_commit(badwire,p,target,limits,back,error));
        badwire=wire;badwire[10]=2;CHECK(!decode_commit(badwire,p,target,limits,back,error));
    }
    for (int kind=0;kind<7;++kind) {
        auto c=commit;
        switch (kind) {
            case 0:c.target--;break;
            case 1:c.members.pop_back();break;
            case 2:c.members[0].generation++;break;
            case 3:c.members[0].position++;break;
            case 4:c.members[0].emitted={99,99};break;
            case 5:c.members[0].emitted={100};break;
            case 6:c.members[0].emitted.clear();break;
        }
        CHECK(!valid_commit(c,p,target,limits,error));
    }
    WorkerState state(128,100,16,4);std::uint64_t sequence=1;
    const auto run=[&](Command cmd,std::uint64_t pos,std::vector<std::uint32_t> words) {
        Control c{cmd,sequence++,pos,std::move(words)};
        return state.begin(c,error) && state.complete(true,"",error);
    };
    for (std::uint32_t slot=0;slot<4;++slot) CHECK(run(Command::BeginSlot,0,{slot,1,0}));
    CHECK(run(Command::Feed,0,{1,2})); // slot 3
    CHECK(run(Command::Bind,0,{0,1,0}));CHECK(run(Command::Step,0,{1}));
    CHECK(state.position(0)==1 && state.position(3)==2 && state.generation(3)==1);
    auto overfeed=state;
    CHECK(!overfeed.begin({Command::Feed,sequence,1,std::vector<std::uint32_t>(17,1)},error));
    BatchPlan plan{{{0,1,1,{2,3,4}},{3,1,2,{5}}}};
    CHECK(encode_batch(plan,limits,false,wire,error));const auto target_sequence=sequence;
    CHECK(run(Command::BatchVerify,0,wire));CHECK(state.awaiting_verdict());
    CHECK(state.position(0)==1 && state.position(3)==2);
    auto pending=state;CHECK(!pending.begin({Command::Reset,sequence,0,{}},error));
    BatchCommit result{target_sequence,{{0,1,1,true,{3,9}},{3,1,2,true,{9}}}};
    auto terminal_result=result;terminal_result.members[0].more=false;
    CHECK(encode_commit(terminal_result,plan,target_sequence,limits,wire,error));
    auto ended=state;CHECK(ended.begin({Command::BatchCommit,sequence,0,wire},error));CHECK(ended.complete(true,"",error));
    auto retired=ended;CHECK(!retired.begin({Command::Step,sequence+1,3,{9}},error));
    CHECK(ended.begin({Command::BeginSlot,sequence+1,3,{0,2,0}},error));CHECK(ended.complete(true,"",error));
    CHECK(ended.begin({Command::Step,sequence+2,3,{8}},error));CHECK(ended.complete(true,"",error));
    CHECK(encode_commit(result,plan,target_sequence,limits,wire,error));CHECK(run(Command::BatchCommit,0,wire));
    CHECK(!state.awaiting_verdict() && state.position(0)==3 && state.position(3)==3);
    auto wrong_pending=state;
    CHECK(!wrong_pending.begin({Command::Step,sequence,3,{8}},error));
    auto terminal=state;
    CHECK(terminal.begin({Command::Step,sequence,3,{9}},error));CHECK(terminal.complete(true,"",error));
    CHECK(run(Command::BeginSlot,3,{3,2,0}));
    auto stale=state;CHECK(encode_batch(plan,limits,false,wire,error));
    CHECK(!stale.begin({Command::BatchVerify,sequence,0,wire},error));
    const auto first=error;CHECK(!stale.fail("later",error) && error==first);
    CHECK(run(Command::Verify,3,{4,5,6}));
    CHECK(run(Command::Verdict,6,{2,7}));CHECK(state.position(3)==6);
    CHECK(run(Command::Shutdown,6,{}));CHECK(state.stopped());
    CHECK(!state.begin({Command::Shutdown,sequence,6,{}},error));
    WorkerState legacy(128,100,16);CHECK(!legacy.begin({Command::BeginSlot,1,0,{0,1,0}},error));
    WorkerState bounds(128,100,16,5);CHECK(!bounds.begin({Command::Shutdown,1,0,{}},error));
    std::puts("DeepSeek bounded batch plans/commits and per-slot state passed");return 0;
}
