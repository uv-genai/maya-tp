// Execute a validated DeepSeek TP2 control operation on one rank's model.
#pragma once

#include "ds4_tp_control.hpp"
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace strata::ds4::tp2 {

// ModelT is Model or a test double with the same worker-facing operations.
// This adapter supplies no transport, partitioning, sampling, or all-reduces.
// Its owner must serialize calls, scope/check the communicator, and discard
// the pair after any failure. Only deterministic window-based feed is allowed
// until both ranks negotiate chunking and collective order.
template <class ModelT> class ModelWorker {
public:
    explicit ModelWorker(ModelT& model) : model_(model) {}

    bool execute(const Control& c, std::string& error) {
        if (!failure_.empty()) { error = failure_; return false; }
        const auto fail = [&](const std::string& cause) {
            failure_ = cause.empty() ? "ds4 tp2 model worker: operation failed" : cause;
            error = failure_;
            return false;
        };
        std::string why;
        if (stopped_) return fail("ds4 tp2 model worker: already shut down");
        if (!valid_control(c, kControlMaxWords, why)) return fail(why);
        if (pending_ != (c.command == Command::Verdict))
            return fail("ds4 tp2 model worker: expected verify verdict");
        if (model_.pos() < 0 || model_.max_context() <= 0 || model_.pos() > model_.max_context() || model_.vocab() <= 0)
            return fail("ds4 tp2 model worker: invalid model limits or position");
        if (c.command != Command::Verdict && c.command != Command::Reset &&
            c.position != static_cast<std::uint64_t>(model_.pos()))
            return fail("ds4 tp2 model worker: position mismatch");
        if (c.command == Command::Reset && c.position != 0)
            return fail("ds4 tp2 model worker: reset position must be zero");

        std::vector<int> tokens;
        if (c.command == Command::Feed || c.command == Command::Step || c.command == Command::Verify) {
            if (c.words.size() > static_cast<std::uint64_t>(model_.max_context() - model_.pos()))
                return fail("ds4 tp2 model worker: context exceeded");
            tokens.reserve(c.words.size());
            for (std::uint32_t word : c.words) {
                if (word >= static_cast<std::uint32_t>(model_.vocab()) ||
                    word > static_cast<std::uint32_t>(std::numeric_limits<int>::max()))
                    return fail("ds4 tp2 model worker: token outside vocabulary");
                tokens.push_back(static_cast<int>(word));
            }
        }
        bool ok = false;
        switch (c.command) {
            case Command::Feed:
                if (model_.prompt_chunk() != 0)
                    return fail("ds4 tp2 model worker: feed requires window-only prefill");
                ok = model_.feed(tokens.data(), static_cast<int>(tokens.size()), 0, why);
                break;
            case Command::Step: ok = model_.step(tokens[0], false, why); break;
            case Command::Verify:
                verify_start_ = static_cast<std::uint64_t>(model_.pos());
                verify_count_ = static_cast<std::uint32_t>(tokens.size());
                ok = model_.tp_verify_begin(tokens.data(), static_cast<int>(tokens.size()), why);
                break;
            case Command::Verdict:
                if (c.words[0] >= verify_count_ || c.words[1] >= static_cast<std::uint32_t>(model_.vocab()) ||
                    c.position != verify_start_ + 1 + c.words[0] ||
                    static_cast<std::uint64_t>(model_.pos()) != verify_start_ + verify_count_)
                    return fail("ds4 tp2 model worker: invalid verify verdict");
                ok = model_.tp_verify_commit(static_cast<int>(c.words[0]), why);
                break;
            case Command::Reset: ok = model_.tp_worker_reset(why); break;
            case Command::Shutdown: ok = true; break;
        }
        // Preserve the first model error even when draining reports another.
        std::string sync_error;
        const bool synced = model_.tp_worker_synchronize(sync_error);
        if (!ok) return fail(why);
        if (!synced) return fail(sync_error);
        const auto expected = c.command == Command::Reset ? std::uint64_t{0}
            : c.command == Command::Verdict ? c.position
            : c.command == Command::Feed || c.command == Command::Step || c.command == Command::Verify
                ? c.position + c.words.size() : c.position;
        if (model_.pos() < 0 || static_cast<std::uint64_t>(model_.pos()) != expected)
            return fail("ds4 tp2 model worker: operation returned the wrong position");
        if (c.command == Command::Verify) pending_ = true;
        if (c.command == Command::Verdict) pending_ = false;
        if (c.command == Command::Shutdown) stopped_ = true;
        error.clear();
        return true;
    }

private:
    ModelT& model_;
    std::uint64_t verify_start_ = 0;
    std::uint32_t verify_count_ = 0;
    bool pending_ = false, stopped_ = false;
    std::string failure_;
};

} // namespace strata::ds4::tp2
