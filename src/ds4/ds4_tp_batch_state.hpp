// Logical per-slot state for the negotiated multi-sequence protocol only.
#pragma once
#include "ds4_tp_batch.hpp"
#include <array>
#include <limits>
#include <optional>

namespace strata::ds4::tp2 {
inline std::uint32_t batch_control_words(std::uint32_t words,std::uint32_t slots) {
    return words==0 || slots==1 ? words : std::max(words,static_cast<std::uint32_t>(kBatchCommitWords));
}
class BatchState {
public:
    BatchState(std::uint64_t context,std::uint32_t vocab,std::uint32_t words,std::uint32_t slots)
        : limits_{slots,vocab,context},words_(batch_control_words(words,slots)),feed_words_(words) {}
    bool begin(const Control& c,std::string& error) {
        if (!failure_.empty()) { error=failure_;return false; }
        const auto bad=[&](const std::string& why){return fail(why,error);};
        if (stopped_ || pending_ || limits_.slots<2 || limits_.slots>kBatchSlots ||
            limits_.context==0 || limits_.vocab==0) return bad("ds4 batch state: invalid limits or operation in progress");
        if (!valid_control(c,words_,error)) return bad(error);
        if (c.sequence!=sequence_ || c.sequence==std::numeric_limits<std::uint64_t>::max())
            return bad("ds4 batch state: stale, skipped or exhausted operation");
        if (plan_ && c.command!=Command::BatchCommit)
            return bad("ds4 batch state: complete batch verdict required");
        if (legacy_verify_ && c.command!=Command::Verdict)
            return bad("ds4 batch state: selected-slot verdict required");
        if (!plan_ && c.command==Command::BatchCommit) return bad("ds4 batch state: no pending batch");
        if (!legacy_verify_ && c.command==Command::Verdict) return bad("ds4 batch state: no pending selected-slot window");
        std::optional<BatchPlan> next_plan;
        std::optional<BatchCommit> next_commit;
        switch (c.command) {
            case Command::Bind: case Command::BeginSlot: {
                const auto slot=c.words[0]; const auto generation=batch_get64(c.words,1);
                if (slot>=limits_.slots || c.position!=positions_[slot])
                    return bad("ds4 batch state: bind slot or position mismatch");
                if (c.command==Command::Bind ? generation!=generations_[slot]
                    : generation==0 || generation==std::numeric_limits<std::uint64_t>::max() ||
                      generations_[slot]==std::numeric_limits<std::uint64_t>::max() || generation!=generations_[slot]+1)
                    return bad("ds4 batch state: slot generation mismatch or exhausted");
                break;
            }
            case Command::Rows: case Command::BatchVerify: {
                if (c.position!=0) return bad("ds4 batch state: batch header position must be zero");
                BatchPlan p;
                if (!decode_batch(c.words,limits_,c.command==Command::Rows,p,error)) return bad(error);
                for (const auto& m:p.members)
                    if (m.generation!=generations_[m.slot] || m.position!=positions_[m.slot])
                        return bad("ds4 batch state: stale member generation or position");
                next_plan=std::move(p);break;
            }
            case Command::BatchCommit: {
                if (c.position!=0) return bad("ds4 batch state: commit header position must be zero");
                BatchCommit commit;
                if (!decode_commit(c.words,*plan_,target_,limits_,commit,error)) return bad(error);
                next_commit=std::move(commit);break;
            }
            case Command::Feed: case Command::Step: case Command::Verify:
                if (c.command==Command::Feed && c.words.size()>feed_words_)
                    return bad("ds4 batch state: feed exceeds negotiated prompt piece");
                if (generations_[selected_]==0 || c.position!=positions_[selected_] ||
                    positions_[selected_]>limits_.context || c.words.size()>limits_.context-positions_[selected_])
                    return bad("ds4 batch state: selected slot position or context mismatch");
                for (auto token:c.words) if (token>=limits_.vocab) return bad("ds4 batch state: invalid input token");
                break;
            case Command::Verdict:
                if (c.words[0]>legacy_offered_ || c.words[1]>=limits_.vocab ||
                    c.position!=positions_[selected_]+1+c.words[0])
                    return bad("ds4 batch state: selected-slot verdict mismatch");
                break;
            case Command::Reset:
                if (generations_[selected_]==0 || c.position!=0) return bad("ds4 batch state: invalid selected-slot reset");
                break;
            case Command::Shutdown:
                if (c.position!=positions_[selected_]) return bad("ds4 batch state: shutdown position mismatch");
                break;
        }
        active_=c;active_plan_=std::move(next_plan);active_commit_=std::move(next_commit);
        pending_=true;error.clear();return true;
    }
    bool complete(bool success,const std::string& cause,std::string& error) {
        if (!failure_.empty()) { error=failure_;return false; }
        if (!pending_) { error="ds4 batch state: no active operation";return false; }
        if (!success) return fail(cause,error);
        switch (active_.command) {
            case Command::Bind: case Command::BeginSlot:
                selected_=active_.words[0];generations_[selected_]=batch_get64(active_.words,1);break;
            case Command::Feed: case Command::Step: positions_[selected_]+=active_.words.size();break;
            case Command::Verify: legacy_verify_=true;legacy_offered_=active_.words.size()-1;break;
            case Command::Verdict: positions_[selected_]=active_.position;legacy_verify_=false;legacy_offered_=0;break;
            case Command::Reset: positions_[selected_]=0;break;
            case Command::Rows:
                for (const auto& m:active_plan_->members) positions_[m.slot]+=m.tokens.size();
                break;
            case Command::BatchVerify: plan_=std::move(active_plan_);target_=active_.sequence;break;
            case Command::BatchCommit:
                for (const auto& m:active_commit_->members) positions_[m.slot]=m.position+m.emitted.size();
                plan_.reset();target_=0;break;
            case Command::Shutdown: stopped_=true;break;
        }
        pending_=false;active_plan_.reset();active_commit_.reset();++sequence_;error.clear();return true;
    }
    bool fail(const std::string& cause,std::string& error) {
        if (failure_.empty()) failure_=cause.empty()?"ds4 batch state: operation failed":cause;
        pending_=false;error=failure_;return false;
    }
    std::uint64_t position() const { return positions_[selected_]; }
    std::uint64_t position(std::uint32_t slot) const { return slot<limits_.slots && slot<kBatchSlots?positions_[slot]:0; }
    std::uint64_t generation(std::uint32_t slot) const { return slot<limits_.slots && slot<kBatchSlots?generations_[slot]:0; }
    std::uint32_t selected() const { return selected_; }
    bool awaiting_verdict() const { return legacy_verify_ || plan_.has_value(); }
    bool failed() const { return !failure_.empty(); }
    bool stopped() const { return stopped_; }
    const std::string& failure() const { return failure_; }
private:
    BatchLimits limits_;std::uint32_t words_,feed_words_,selected_=0;
    std::array<std::uint64_t,kBatchSlots> positions_{},generations_{};
    std::uint64_t sequence_=1,target_=0;
    bool pending_=false,legacy_verify_=false,stopped_=false;
    std::size_t legacy_offered_=0;
    Control active_;
    std::optional<BatchPlan> plan_,active_plan_;
    std::optional<BatchCommit> active_commit_;
    std::string failure_;
};
} // namespace strata::ds4::tp2
