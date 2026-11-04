// DeepSeek TP2 worker operation dispatch; model execution is supplied by a caller.
#pragma once

#include "ds4_tp_channel.hpp"
#include "ds4_tp_worker_state.hpp"
#include <exception>
#include <functional>

namespace strata::ds4::tp2 {

// The callback must execute precisely the requested operation on the worker,
// report the first error, and drain its GPU stream before returning. It must
// not emit an independent token or change the control/collective order. On any
// failure this object refuses further reads/operations; its owner must tear down
// the entire communicator and pair. The callback and real model are not wired
// by this class.
class Worker {
public:
    using Handler = std::function<bool(const Control&, std::string&)>;
    Worker(strata::net::Communicator& comm, std::uint64_t context, std::uint32_t vocab,
           std::uint32_t max_words) : comm_(comm), state_(context, vocab, max_words), max_words_(max_words) {}

    bool run_one(const Handler& handle, std::string& error) {
        if (state_.failed()) { error = state_.failure(); return false; }
        if (state_.stopped()) { error = "ds4 tp2 worker: shut down"; return false; }
        if (comm_.world_size() != 2 || comm_.rank() != 1)
            return state_.fail("ds4 tp2 worker: expected rank 1 of 2", error);
        Control c;
        if (!receive_message(comm_, max_words_, c, error)) return state_.fail(error, error);
        if (!state_.begin(c, error)) return false;
        if (!comm_.BeginOperation(c.sequence, &error)) return state_.complete(false, error, error);
        try {
            if (!handle(c, error)) return state_.complete(false, error, error);
        } catch (const std::exception& ex) {
            return state_.complete(false, ex.what(), error);
        } catch (...) {
            return state_.complete(false, "ds4 tp2 worker: operation threw", error);
        }
        if (!comm_.CheckQueued(&error)) return state_.complete(false, error, error);
        if (!comm_.EndOperation(c.sequence, &error)) return state_.complete(false, error, error);
        return state_.complete(true, "", error);
    }
    const WorkerState& state() const { return state_; }
private:
    strata::net::Communicator& comm_;
    WorkerState state_;
    std::uint32_t max_words_;
};

}  // namespace strata::ds4::tp2
