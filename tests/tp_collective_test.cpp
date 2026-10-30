#include "strata/core/tp_collective.hpp"
#include <cstdio>
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s\n", #x); return 1; } } while (0)
#include <cstddef>
#include <limits>

int main() {
    using strata::core::tp_prefill_limit;
    constexpr std::size_t payload = (48u << 20) - 64;
    constexpr int hidden = 4096;
    const int cap = tp_prefill_limit(32768, hidden, payload);
    CHECK(cap == 3008);
    CHECK((std::size_t) cap * hidden * sizeof(float) <= payload);
    // User's failing 7234 rows (118521856 bytes) must cross batch boundaries.
    CHECK((std::size_t) 7234 * hidden * sizeof(float) == 118521856);
    CHECK(tp_prefill_limit(25088, hidden, payload) == cap); // different host layout budgets
    CHECK(tp_prefill_limit(32768, hidden, payload) == cap);
    CHECK(tp_prefill_limit(64, hidden, payload) == 64);
    CHECK(tp_prefill_limit(63, hidden, payload) == 63);
    CHECK(tp_prefill_limit(32768, hidden, (std::size_t) hidden * sizeof(float) - 1) == 0);
    CHECK(tp_prefill_limit(32768, hidden, (std::size_t) hidden * sizeof(float)) == 1);
    CHECK(tp_prefill_limit(32768, hidden, (std::size_t) hidden * sizeof(float) * 3072) == 3072);
    CHECK(tp_prefill_limit(0, hidden, payload) == 0);
    CHECK(tp_prefill_limit(32768, 0, payload) == 0);
    CHECK(tp_prefill_limit(-1, hidden, payload) == 0);
    CHECK(tp_prefill_limit(32768, hidden, std::numeric_limits<std::size_t>::max()) == 32768);
}
