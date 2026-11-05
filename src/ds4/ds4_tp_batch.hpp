// DeepSeek's opt-in TP2 batch metadata; tensors remain on APU RDMA.
#pragma once
#include "ds4_tp_control.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::ds4::tp2 {
inline constexpr std::uint32_t kBatchVersion=1, kBatchSlots=4, kBatchRows=16;
inline constexpr std::size_t kBatchPlanWords=3+kBatchSlots*6+kBatchRows;
inline constexpr std::size_t kBatchCommitWords=4+kBatchSlots*7+kBatchRows;
static_assert(kBatchPlanWords==43 && kBatchCommitWords==48,"control payload bounds must match the batch codec");
struct BatchLimits { std::uint32_t slots=0,vocab=0; std::uint64_t context=0; };
struct BatchMember {
    std::uint32_t slot=0; std::uint64_t generation=0,position=0;
    std::vector<std::uint32_t> tokens;
};
struct BatchPlan { std::vector<BatchMember> members; };
struct CommitMember {
    std::uint32_t slot=0; std::uint64_t generation=0,position=0;
    bool more=false;
    // Consumed inputs = emitted.size(); the final emitted token is pending,
    // unconsumed. Earlier emitted tokens must equal canonical draft inputs.
    std::vector<std::uint32_t> emitted;
};
struct BatchCommit { std::uint64_t target=0; std::vector<CommitMember> members; };
inline void batch_put64(std::vector<std::uint32_t>& w,std::uint64_t v) {
    w.push_back(static_cast<std::uint32_t>(v));w.push_back(static_cast<std::uint32_t>(v>>32));
}
inline std::uint64_t batch_get64(const std::vector<std::uint32_t>& w,std::size_t at) {
    return w[at]|(static_cast<std::uint64_t>(w[at+1])<<32);
}
inline bool valid_batch(const BatchPlan& p,const BatchLimits& limits,bool ordinary,std::string& error) {
    const auto bad=[&](const char* why){error=why;return false;};
    if (limits.slots<2 || limits.slots>kBatchSlots || limits.vocab==0 || limits.context==0 ||
        p.members.empty() || p.members.size()>limits.slots) return bad("ds4 batch: invalid capacities or members");
    std::uint32_t seen=0; std::size_t rows=0;
    for (const auto& m:p.members) {
        if (m.slot>=limits.slots || (seen&(1u<<m.slot)) || m.generation==0 || m.tokens.empty() ||
            m.tokens.size()>kBatchRows-rows || (ordinary && m.tokens.size()!=1))
            return bad("ds4 batch: invalid, duplicate or oversized member");
        seen|=1u<<m.slot;rows+=m.tokens.size();
        if (m.position>limits.context || m.tokens.size()>limits.context-m.position)
            return bad("ds4 batch: member exceeds context");
        for (auto token:m.tokens) if (token>=limits.vocab) return bad("ds4 batch: token outside vocabulary");
    }
    error.clear();return true;
}
inline bool encode_batch(const BatchPlan& p,const BatchLimits& limits,bool ordinary,
                         std::vector<std::uint32_t>& out,std::string& error) {
    if (!valid_batch(p,limits,ordinary,error)) return false;
    std::size_t rows=0;for (const auto& m:p.members) rows+=m.tokens.size();
    std::vector<std::uint32_t> w{kBatchVersion,static_cast<std::uint32_t>(p.members.size()),static_cast<std::uint32_t>(rows)};
    for (const auto& m:p.members) {
        w.push_back(m.slot);batch_put64(w,m.generation);batch_put64(w,m.position);
        w.push_back(static_cast<std::uint32_t>(m.tokens.size()));w.insert(w.end(),m.tokens.begin(),m.tokens.end());
    }
    out=std::move(w);return true;
}
inline bool decode_batch(const std::vector<std::uint32_t>& w,const BatchLimits& limits,bool ordinary,
                         BatchPlan& out,std::string& error) {
    const auto bad=[&]{error="ds4 batch: malformed bounded plan";return false;};
    if (w.size()<3 || w.size()>kBatchPlanWords || w[0]!=kBatchVersion || w[1]<1 ||
        w[1]>kBatchSlots || w[2]<w[1] || w[2]>kBatchRows) return bad();
    BatchPlan p;std::size_t at=3,rows=0;
    for (std::uint32_t i=0;i<w[1];++i) {
        if (w.size()-at<6) return bad();
        BatchMember m{w[at],batch_get64(w,at+1),batch_get64(w,at+3),{}};
        const auto n=w[at+5];at+=6;
        if (n==0 || n>kBatchRows-rows || n>w.size()-at) return bad();
        rows+=n;m.tokens.assign(w.begin()+static_cast<std::ptrdiff_t>(at),w.begin()+static_cast<std::ptrdiff_t>(at+n));
        at+=n;p.members.push_back(std::move(m));
    }
    if (at!=w.size() || rows!=w[2]) return bad();
    if (!valid_batch(p,limits,ordinary,error)) return false;
    out=std::move(p);return true;
}
inline bool valid_commit(const BatchCommit& c,const BatchPlan& p,std::uint64_t target,
                         const BatchLimits& limits,std::string& error) {
    const auto bad=[&](const char* why){error=why;return false;};
    if (!valid_batch(p,limits,false,error)) return false;
    if (target==0 || c.target!=target || c.members.size()!=p.members.size())
        return bad("ds4 batch: incomplete or stale commit");
    for (std::size_t i=0;i<p.members.size();++i) {
        const auto& m=c.members[i];const auto& original=p.members[i];const auto keep=m.emitted.size();
        if (m.slot!=original.slot || m.generation!=original.generation || m.position!=original.position ||
            keep<1 || keep>original.tokens.size() || (m.more && keep>=limits.context-m.position))
            return bad("ds4 batch: commit identity, position or retained count mismatch");
        for (std::size_t j=0;j<keep;++j)
            if (m.emitted[j]>=limits.vocab || (j+1<keep && m.emitted[j]!=original.tokens[j+1]))
                return bad("ds4 batch: commit differs from canonical retained inputs");
    }
    error.clear();return true;
}
inline bool encode_commit(const BatchCommit& c,const BatchPlan& p,std::uint64_t target,
                          const BatchLimits& limits,std::vector<std::uint32_t>& out,std::string& error) {
    if (!valid_commit(c,p,target,limits,error)) return false;
    std::vector<std::uint32_t> w{kBatchVersion,static_cast<std::uint32_t>(c.members.size())};batch_put64(w,c.target);
    for (const auto& m:c.members) {
        w.push_back(m.slot);batch_put64(w,m.generation);batch_put64(w,m.position);
        w.push_back(static_cast<std::uint32_t>(m.emitted.size()));w.push_back(m.more?1:0);
        w.insert(w.end(),m.emitted.begin(),m.emitted.end());
    }
    out=std::move(w);return true;
}
inline bool decode_commit(const std::vector<std::uint32_t>& w,const BatchPlan& p,std::uint64_t target,
                          const BatchLimits& limits,BatchCommit& out,std::string& error) {
    const auto bad=[&]{error="ds4 batch: malformed bounded commit";return false;};
    if (w.size()<4 || w.size()>kBatchCommitWords || w[0]!=kBatchVersion || w[1]<1 || w[1]>kBatchSlots) return bad();
    BatchCommit c{batch_get64(w,2),{}};std::size_t at=4,rows=0;
    for (std::uint32_t i=0;i<w[1];++i) {
        if (w.size()-at<7 || w[at+6]>1) return bad();
        CommitMember m{w[at],batch_get64(w,at+1),batch_get64(w,at+3),w[at+6]!=0,{}};
        const auto keep=w[at+5];at+=7;
        if (keep==0 || keep>kBatchRows-rows || keep>w.size()-at) return bad();
        rows+=keep;m.emitted.assign(w.begin()+static_cast<std::ptrdiff_t>(at),w.begin()+static_cast<std::ptrdiff_t>(at+keep));
        at+=keep;c.members.push_back(std::move(m));
    }
    if (at!=w.size()) return bad();
    if (!valid_commit(c,p,target,limits,error)) return false;
    out=std::move(c);return true;
}
} // namespace strata::ds4::tp2
