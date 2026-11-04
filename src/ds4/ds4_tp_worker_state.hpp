// Host-side sequencing for DeepSeek TP2 worker operations.
#pragma once

#include "ds4_tp_control.hpp"
#include <cstdint>
#include <limits>
#include <string>

namespace strata::ds4::tp2 {

// Validates control before entering collectives. The engine caller must call
// complete only after its operation and queued collectives have drained.
// This class tracks *logical* committed positions; it does not implement or
// claim to restore the model's KV, compressor, Engram, or drafter state.
class WorkerState {
public:
    WorkerState(std::uint64_t context, std::uint32_t vocab, std::uint32_t max_words)
        : context_(context), vocab_(vocab), max_words_(max_words) {}

    bool begin(const Control& c, std::string& error) {
        if (!failure_.empty()) { error = failure_; return false; }
        if (stopped_ || pending_) { error = "ds4 tp2 worker: stopped or operation already in progress"; return false; }
        const auto fatal = [&](const char* cause) { failure_ = cause; error = failure_; return false; };
        if (context_ == 0 || vocab_ == 0) return fatal("ds4 tp2 worker: invalid model limits");
        if (!valid_control(c, max_words_, error)) { failure_ = error; return false; }
        if (c.sequence != next_sequence_ || c.sequence == std::numeric_limits<std::uint64_t>::max()) {
            return fatal("ds4 tp2 worker: control sequence mismatch or exhausted");
        }
        if (awaiting_verdict_ != (c.command == Command::Verdict)) {
            return fatal("ds4 tp2 worker: expected verify verdict before next operation");
        }
        if (c.command == Command::Verdict) {
            const auto accepted = c.words[0];
            if (accepted > offered_ || c.words[1] >= vocab_ ||
                c.position != position_ + 1 + accepted) {
                return fatal("ds4 tp2 worker: invalid speculative verdict or position");
            }
        } else {
            if (c.command == Command::Reset) {
                if (c.position != 0) return fatal("ds4 tp2 worker: reset must target position zero");
            } else if (c.position != position_) {
                return fatal("ds4 tp2 worker: committed position mismatch");
            }
            if (c.command == Command::Feed || c.command == Command::Step || c.command == Command::Verify) {
                if (position_ > context_ || c.words.size() > context_ - position_) {
                    return fatal("ds4 tp2 worker: operation exceeds context");
                }
                for (const auto token : c.words) if (token >= vocab_) {
                    return fatal("ds4 tp2 worker: token outside model vocabulary");
                }
            }
        }
        active_ = c; pending_ = true; error.clear(); return true;
    }

    bool complete(bool success, const std::string& cause, std::string& error) {
        if (!failure_.empty()) { error = failure_; return false; }
        if (!pending_) { error = "ds4 tp2 worker: no operation in progress"; return false; }
        pending_ = false;
        if (!success) return fail(cause, error);
        switch (active_.command) {
            case Command::Feed: case Command::Step: position_ += active_.words.size(); break;
            case Command::Verify: awaiting_verdict_ = true; offered_ = active_.words.size() - 1; break;
            case Command::Verdict: position_ = active_.position; awaiting_verdict_ = false; offered_ = 0; break;
            case Command::Reset: position_ = 0; break;
            case Command::Shutdown: stopped_ = true; break;
        }
        ++next_sequence_;
        error.clear(); return true;
    }

    std::uint64_t position() const { return position_; }
    bool awaiting_verdict() const { return awaiting_verdict_; }
    bool failed() const { return !failure_.empty(); }
    bool stopped() const { return stopped_; }
    const std::string& failure() const { return failure_; }

    // A failed receive or operation is fatal to the pair. Never replace the
    // first cause or allow another operation on this state.
    bool fail(const std::string& cause, std::string& error) {
        if (failure_.empty()) failure_ = cause.empty() ? "ds4 tp2 worker: operation failed" : cause;
        pending_ = false;
        error = failure_; return false;
    }

private:
    std::uint64_t context_, position_ = 0, next_sequence_ = 1;
    std::uint32_t vocab_, max_words_;
    std::size_t offered_ = 0;
    bool pending_ = false, awaiting_verdict_ = false, stopped_ = false;
    Control active_;
    std::string failure_;
};

}  // namespace strata::ds4::tp2
