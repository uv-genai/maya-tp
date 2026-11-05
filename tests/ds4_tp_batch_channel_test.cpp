#include "../src/ds4/ds4_tp_controller.hpp"
#include "../src/ds4/ds4_tp_worker.hpp"
#include "ds4_tp_control_fake.hpp"
#include <cstdio>
#include <stdexcept>
using namespace strata::ds4::tp2;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"line %d: %s (%s)\n",__LINE__,#x,error.c_str()); return 1; } } while (0)
int main() {
    std::string error; Fake reply;
    CHECK(send_batch_ack(reply,1,AckStage::Accepted,"",error));
    CHECK(send_batch_ack(reply,1,AckStage::Completed,"",error));
    Fake lead;lead.inbox=reply.sent;Controller controller(lead,128,100,16,4);int called=0;
    CHECK(controller.run(Command::BeginSlot,0,{0,1,0},[&](const Control&,std::string&) {
        ++called;return lead.offset==20 && lead.begins==1 && controller.state().generation(0)==0;
    },error));
    CHECK(called==1 && lead.offset==40 && controller.state().generation(0)==1);
    const auto bytes=lead.sent.size();
    CHECK(!controller.run(Command::BeginSlot,0,{0,1,0},[&](const Control&,std::string&){++called;return true;},error));
    CHECK(called==1 && lead.sent.size()==bytes);
    Fake worker;worker.rank_value=1;worker.inbox=lead.sent;Worker peer(worker,128,100,16,4);
    CHECK(peer.run_one([&](const Control&,std::string&){return worker.sent.size()==20 && worker.begins==1;},error));
    CHECK(peer.state().generation(0)==1 && worker.sent.size()==40);
    Fake wrong;Fake bad_reply;CHECK(send_batch_ack(bad_reply,2,AckStage::Accepted,"",error));
    wrong.inbox=bad_reply.sent;Controller guarded(wrong,128,100,16,4);called=0;
    CHECK(!guarded.run(Command::BeginSlot,0,{0,1,0},[&](const Control&,std::string&){++called;return true;},error));
    CHECK(called==0 && wrong.begins==0 && guarded.state().failed());
    Fake completion;Fake completion_reply;
    CHECK(send_batch_ack(completion_reply,1,AckStage::Accepted,"",error));
    CHECK(send_batch_ack(completion_reply,1,AckStage::Completed,"first peer failure",error));
    completion.inbox=completion_reply.sent;Controller failed(completion,128,100,16,4);
    CHECK(!failed.run(Command::BeginSlot,0,{0,1,0},[](const Control&,std::string&){return true;},error));
    CHECK(error=="first peer failure" && failed.state().generation(0)==0);
    completion.fail_peer=true;const auto sent=completion.sent.size();
    CHECK(!failed.run(Command::BeginSlot,0,{0,1,0},[](const Control&,std::string&){return true;},error));
    CHECK(error=="first peer failure" && completion.sent.size()==sent);
    Fake bad_worker;bad_worker.rank_value=1;
    CHECK(encode_control({Command::BeginSlot,1,0,{4,1,0}},48,bad_worker.inbox,error));
    Worker rejected(bad_worker,128,100,16,4);called=0;
    CHECK(!rejected.run_one([&](const Control&,std::string&){++called;return true;},error));
    CHECK(called==0 && bad_worker.begins==0 && rejected.state().failed());
    Fake received;received.inbox=bad_worker.sent;
    CHECK(!receive_batch_ack(received,1,AckStage::Accepted,error));CHECK(error==rejected.state().failure());
    Fake thrown;thrown.rank_value=1;thrown.inbox=lead.sent;Worker exception(thrown,128,100,16,4);
    CHECK(!exception.run_one([](const Control&,std::string&)->bool{throw std::runtime_error("original model error");},error));
    CHECK(error=="original model error" && thrown.ends==0);
    Fake ack;ack.inbox=thrown.sent;CHECK(receive_batch_ack(ack,1,AckStage::Accepted,error));
    CHECK(!receive_batch_ack(ack,1,AckStage::Completed,error) && error=="original model error");
    Identity identity;identity.feature_flags=1;CHECK(configure_batch_identity(identity,4,error));
    CHECK(identity.protocol==2 && ((identity.feature_flags>>48)&255)==4 && (identity.feature_flags>>56)==16);
    CHECK(configure_batch_identity(identity,3,error));CHECK(((identity.feature_flags>>48)&255)==3);
    CHECK(configure_batch_identity(identity,1,error));CHECK(identity.protocol==1 && identity.feature_flags==1);
    Fake oversized;oversized.inbox=reply.sent;oversized.inbox[18]=4;oversized.inbox[19]=1;
    CHECK(!receive_batch_ack(oversized,1,AckStage::Accepted,error) && oversized.offset==20);
    std::puts("DeepSeek batch acceptance/completion ordering and sticky first failure passed");return 0;
}
