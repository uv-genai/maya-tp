#include "../src/ds4/ds4_tp_channel.hpp"

#include <cstring>
#include <cstdio>

using namespace strata::ds4::tp2;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); return 1; } } while (0)

class Fake final : public strata::net::Communicator {
public:
    unsigned rank_value = 0;
    std::vector<std::uint8_t> inbox, sent;
    std::size_t offset = 0, recv_calls = 0;
    std::string order;
    bool fail_send = false;
    std::uint32_t rank() const noexcept override { return rank_value; }
    std::uint32_t world_size() const noexcept override { return 2; }
    bool BeginOperation(std::uint64_t, std::string*) override { return true; }
    bool EndOperation(std::uint64_t, std::string*) override { return true; }
    bool SendControl(const void* data, std::size_t n, std::string* e) override {
        order += 'S';
        if (fail_send) { *e = "send failed"; return false; }
        auto p = static_cast<const std::uint8_t*>(data); sent.insert(sent.end(), p, p + n); return true;
    }
    bool RecvControl(void* data, std::size_t n, std::string* e) override {
        order += 'R'; ++recv_calls;
        if (n > inbox.size() - offset) { *e = "receive failed"; return false; }
        std::memcpy(data, inbox.data() + offset, n); offset += n; return true;
    }
    const float* ExchangePartial(const float*, std::size_t, hipStream_t, std::string*) override { return nullptr; }
    bool StartPartial(const float*, std::size_t, hipStream_t, std::string*) override { return false; }
    const float* FinishPartial(std::size_t, std::string*) override { return nullptr; }
    bool QueuePartial(std::size_t, QueuedPartial*, std::string*) override { return false; }
    bool CheckQueued(std::string*) override { return true; }
};

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
    std::puts("DeepSeek TP2 channel: identity ordering, bounded receive and transport errors passed");
}
