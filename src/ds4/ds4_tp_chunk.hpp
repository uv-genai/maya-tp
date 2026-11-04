// Shared bound for DeepSeek TP2 activation rows and control messages.
#pragma once

#include "ds4_tp_control.hpp"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>

namespace strata::ds4::tp2 {

// Both ranks call this with the agreed request and receive limit. It does not
// enable a chunk execution path; callers must honor the returned cap on every
// feed, activation exchange and worker control message.
inline bool chunk_rows(std::uint32_t requested, std::uint32_t hidden, std::uint64_t limit,
                       std::uint32_t& rows, std::string& error) {
    constexpr std::uint32_t window=16;
    if (hidden==0 || requested==0 || requested>kControlMaxWords) {
        error="ds4 tp2: invalid prefill row request or hidden size"; return false;
    }
    const std::uint64_t row_bytes=static_cast<std::uint64_t>(hidden)*sizeof(float);
    const std::uint64_t capacity=limit/row_bytes;
    if (capacity<window) { error="ds4 tp2: collective window is too small for verification"; return false; }
    rows=static_cast<std::uint32_t>(std::min<std::uint64_t>(std::max(requested,window),capacity));
    error.clear(); return true;
}

} // namespace strata::ds4::tp2
