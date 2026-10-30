#pragma once
// Host-only bounds for a TP prefill batch. The transport window includes its own
// header, so use MaxPartialBytes(), not the nominal window allocation.
#include <algorithm>
#include <cstddef>

namespace strata::core {
inline int tp_prefill_limit(int local_rows, int hidden, std::size_t payload_bytes) {
    if (local_rows <= 0 || hidden <= 0) return 0;
    const std::size_t rows = payload_bytes / sizeof(float) / (std::size_t) hidden;
    const int bounded = (int) std::min<std::size_t>(rows, (std::size_t) local_rows);
    // Prefill layouts normally use 64-row blocks; never round UP past the wire budget.
    return bounded >= 64 ? bounded / 64 * 64 : bounded;
}
}  // namespace strata::core
