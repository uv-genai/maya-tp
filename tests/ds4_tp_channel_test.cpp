#include "../src/ds4/ds4_tp_channel.hpp"
#include "../src/ds4/ds4_tp_worker.hpp"
#include "../src/ds4/ds4_tp_controller.hpp"

#include <cstring>
#include <cstdio>

using namespace strata::ds4::tp2;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); return 1; } } while (0)

#include "ds4_tp_control_fake.hpp"


int main() {
    Identity a; a.rank=0; a.world=2; a.protocol=1; a.hidden=5120; a.layers=40;
    a.experts=384; a.expert_width=2304; a.max_context=262144; a.max_partial_bytes=1024;
    a.model_digest[0]=1; a.partition_digest[0]=2; a.build_digest[0]=3;
    Identity b=a; b.rank=1; b.max_partial_bytes=512;
    std::uint64_t limit=0; Identity peer; std::string error;
    Fake f; f.inbox=encode_identity(b);
    CHECK(exchange_identity(f,a,peer,limit,error));
    CHECK(f.order=="SR" && limit==512 && peer.rank==1 && f.sent==encode_identity(a));
    Fake g; g.rank_value=1; g.inbox=encode_identity(a);
    CHECK(exchange_identity(g,b,peer,limit,error) && g.order=="RS" && peer.rank==0);
    b.model_digest[1]=9; Fake mismatch; mismatch.inbox=encode_identity(b);
    CHECK(!exchange_identity(mismatch,a,peer,limit,error) && error.find("model")!=std::string::npos);
    Fake badrank; badrank.rank_value=1;
    CHECK(!exchange_identity(badrank,a,peer,limit,error) && badrank.order.empty());
    Fake sendfail; sendfail.fail_send=true; CHECK(!exchange_identity(sendfail,a,peer,limit,error));
    CHECK(error=="send failed" && sendfail.order=="S");
    Control request{Command::Verify, 3, 42, {5,6,7}}, decoded;
    Fake channel; CHECK(send_message(channel,request,100,error)); channel.inbox=channel.sent;
    CHECK(receive_message(channel,100,decoded,error));
    CHECK(decoded.command==request.command && decoded.words==request.words && decoded.position==42);
    CHECK(channel.recv_calls==2);
    Fake oversized; oversized.inbox=channel.sent; oversized.inbox[12]=0xff;
    CHECK(!receive_message(oversized,100,decoded,error) && oversized.recv_calls==1);
    Fake truncated; truncated.inbox=channel.sent; truncated.inbox.pop_back();
    CHECK(!receive_message(truncated,100,decoded,error) && error=="receive failed");
    Fake worker_comm; worker_comm.rank_value=1;
    for (const Control& c : {Control{Command::Feed,1,0,{1,2}}, Control{Command::Verify,2,2,{3,4,5}},
                             Control{Command::Verdict,3,4,{1,6}}, Control{Command::Shutdown,4,4,{}}}) {
        std::vector<std::uint8_t> frame; CHECK(encode_control(c,16,frame,error));
        worker_comm.inbox.insert(worker_comm.inbox.end(),frame.begin(),frame.end());
    }
    Worker worker(worker_comm,32,100,16); int calls=0;
    auto handle=[&](const Control&,std::string&) { ++calls; return true; };
    CHECK(worker.run_one(handle,error) && worker.state().position()==2);
    CHECK(worker.run_one(handle,error) && worker.state().awaiting_verdict());
    CHECK(worker.run_one(handle,error) && worker.state().position()==4);
    CHECK(worker.run_one(handle,error) && worker.state().stopped());
    CHECK(calls==4 && worker_comm.begins==4 && worker_comm.ends==4);
    const auto calls_before=worker_comm.recv_calls;
    CHECK(!worker.run_one(handle,error) && worker_comm.recv_calls==calls_before);
    Fake failed; failed.rank_value=1; CHECK(encode_control(Control{Command::Step,1,0,{1}},16,failed.inbox,error));
    failed.fail_queued=true; Worker failed_worker(failed,32,100,16); calls=0;
    CHECK(!failed_worker.run_one(handle,error) && error=="queued peer failure");
    CHECK(failed_worker.state().failed() && calls==1 && failed.ends==0);
    const auto recv_before=failed.recv_calls;
    CHECK(!failed_worker.run_one(handle,error) && error=="queued peer failure" && failed.recv_calls==recv_before && calls==1);
    Fake disconnected; disconnected.rank_value=1; Worker dead(disconnected,32,100,16);
    CHECK(!dead.run_one(handle,error) && error=="receive failed");
    CHECK(!dead.run_one(handle,error) && error=="receive failed" && disconnected.recv_calls==1);
    Fake wrong_rank; Worker not_worker(wrong_rank,32,100,16);
    CHECK(!not_worker.run_one(handle,error) && wrong_rank.recv_calls==0 && not_worker.state().failed());
    Fake leader;
    CHECK(encode_control(Control{Command::Shutdown,5,4,{}},16,leader.inbox,error));
    Controller controller(leader,32,100,16); calls=0;
    CHECK(controller.run(Command::Feed,0,{1,2},handle,error));
    CHECK(controller.run(Command::Verify,2,{3,4,5},handle,error));
    CHECK(controller.state().awaiting_verdict());
    CHECK(controller.run(Command::Verdict,4,{1,6},handle,error));
    CHECK(controller.run(Command::Shutdown,4,{},handle,error));
    CHECK(controller.state().stopped() && controller.state().position()==4);
    CHECK(calls==4 && leader.begins==4 && leader.ends==4 && leader.recv_calls==1);
    Fake failed_leader; failed_leader.fail_queued=true;
    Controller broken(failed_leader,32,100,16); calls=0;
    CHECK(!broken.run(Command::Step,0,{1},handle,error) && error=="queued peer failure");
    const auto sent_before=failed_leader.sent.size();
    CHECK(!broken.run(Command::Step,0,{1},handle,error) && error=="queued peer failure");
    CHECK(failed_leader.sent.size()==sent_before && calls==1 && failed_leader.ends==0);
    Fake bad_verdict; Controller guarded(bad_verdict,32,100,16);
    CHECK(guarded.run(Command::Verify,0,{1,2},handle,error));
    const auto verify_bytes=bad_verdict.sent.size();
    CHECK(!guarded.run(Command::Step,0,{1},handle,error));
    CHECK(guarded.state().failed() && bad_verdict.sent.size()==verify_bytes);
    Fake bad_ack;
    CHECK(encode_control(Control{Command::Shutdown,2,1,{}},16,bad_ack.inbox,error));
    Controller ack_guard(bad_ack,32,100,16);
    CHECK(!ack_guard.run(Command::Shutdown,0,{},handle,error));
    CHECK(ack_guard.state().failed() && error.find("acknowledgement")!=std::string::npos);
    for (int point=0;point<4;++point) {
        Fake fault; fault.fail_send=point==0; fault.fail_begin=point==1; fault.fail_end=point==3;
        Controller guarded_fault(fault,32,100,16); int ran=0;
        const auto operation=[&](const Control&,std::string& e) {
            ++ran; if (point==2) { e="model failed"; return false; } return true;
        };
        CHECK(!guarded_fault.run(Command::Step,0,{1},operation,error));
        const std::string first=error; const auto bytes=fault.sent.size();
        CHECK(!guarded_fault.run(Command::Step,0,{1},operation,error) && error==first);
        CHECK(fault.sent.size()==bytes && ran==(point<2?0:1));
    }
    Fake not_leader; not_leader.rank_value=1;
    Controller wrong_leader(not_leader,32,100,16);
    CHECK(!wrong_leader.run(Command::Step,0,{1},handle,error) && not_leader.sent.empty());
    Fake idle; idle.fail_peer=true; Controller idle_guard(idle,32,100,16);
    CHECK(!idle_guard.run(Command::Step,0,{1},handle,error) && error=="idle peer failure");
    CHECK(idle.sent.empty() && idle.begins==0);
    std::puts("DeepSeek TP2 channel: identity ordering, bounded receive and transport errors passed");
}
