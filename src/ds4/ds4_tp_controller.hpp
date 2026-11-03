// Rank-zero control sequencing for a DeepSeek TP2 pair.
#pragma once

#include "ds4_tp_channel.hpp"
#include "ds4_tp_worker_state.hpp"
#include <exception>
#include <functional>
#include <utility>

namespace strata::ds4::tp2 {

// The handler runs rank zero's matching operation and drains its GPU stream.
// A failure after a control message may have reached the worker is fatal to
// the entire pair. In particular, do not retry a command or restart one rank.
class Controller {
public:
    using Handler = std::function<bool(const Control&, std::string&)>;
    Controller(strata::net::Communicator& comm, std::uint64_t context, std::uint32_t vocab,
               std::uint32_t max_words)
        : comm_(comm), state_(context, vocab, max_words), max_words_(max_words) {}

    bool run(Command command, std::uint64_t position, std::vector<std::uint32_t> words,
             const Handler& handle, std::string& error) {
        if (state_.failed()) { error = state_.failure(); return false; }
        if (state_.stopped()) { error = "ds4 tp2 controller: shut down"; return false; }
        if (comm_.world_size() != 2 || comm_.rank() != 0)
            return state_.fail("ds4 tp2 controller: expected rank 0 of 2", error);
        // A monitor may already have poisoned the pair while it was idle.
        // Check before writing the next command, retaining its first cause.
        if (!comm_.CheckPeer(&error)) return state_.fail(error,error);
        Control c{command, sequence_, position, std::move(words)};
        if (!state_.begin(c, error)) return false;
        if (!send_message(comm_, c, max_words_, error) || !comm_.BeginOperation(sequence_, &error))
            return state_.complete(false, error, error);
        try {
            if (!handle(c, error)) return state_.complete(false, error, error);
        } catch (const std::exception& ex) {
            return state_.complete(false, ex.what(), error);
        } catch (...) {
            return state_.complete(false, "ds4 tp2 controller: operation threw", error);
        }
        if (!comm_.CheckQueued(&error) || !comm_.EndOperation(sequence_, &error))
            return state_.complete(false, error, error);
        if (!state_.complete(true, "", error)) return false;
        ++sequence_;
        if (command == Command::Shutdown) {
            Control done;
            if (!receive_message(comm_, max_words_, done, error)) return state_.fail(error, error);
            if (done.command != Command::Shutdown || done.sequence != sequence_ ||
                done.position != state_.position() || !done.words.empty())
                return state_.fail("ds4 tp2 controller: invalid shutdown acknowledgement", error);
        }
        error.clear();
        return true;
    }

    const WorkerState& state() const { return state_; }

private:
    strata::net::Communicator& comm_;
    WorkerState state_;
    std::uint32_t max_words_;
    std::uint64_t sequence_ = 1;
};

} // namespace strata::ds4::tp2
