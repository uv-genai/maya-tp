#pragma once
#include "strata/net/communicator.hpp"
#include <cstring>
#include <vector>

class Fake final : public strata::net::Communicator {
public:
    unsigned rank_value = 0;
    std::vector<std::uint8_t> inbox, sent;
    std::size_t offset = 0, recv_calls = 0;
    std::string order;
    bool fail_send = false;
    bool fail_queued = false;
    bool fail_peer = false;
    bool fail_begin = false, fail_end = false;
    int begins = 0, ends = 0;
    std::uint32_t rank() const noexcept override { return rank_value; }
    std::uint32_t world_size() const noexcept override { return 2; }
    bool BeginOperation(std::uint64_t, std::string* e) override {
        ++begins; if (fail_begin) { *e="begin failed"; return false; } return true;
    }
    bool EndOperation(std::uint64_t, std::string* e) override {
        ++ends; if (fail_end) { *e="end failed"; return false; } return true;
    }
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
    bool CheckQueued(std::string* e) override { if (fail_queued) { *e = "queued peer failure"; return false; } return true; }
    bool CheckPeer(std::string* e) override { if (fail_peer) { *e = "idle peer failure"; return false; } return true; }
};
