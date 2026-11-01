#include "../src/ds4/ds4_chunk_policy.hpp"
#include <climits>
#include <initializer_list>
#include <cstdio>
namespace policy = strata::ds4::chunk_policy;
#define CHECK(expr) do { if (!(expr)) { std::fprintf(stderr, "line%d: %s\n", __LINE__, #expr); return 1; } } while (0)
int main() {
    CHECK(policy::configured_cap(nullptr, nullptr, nullptr) == 0);
    CHECK(policy::configured_cap("4096", "8192", "16384") == 4096);
    CHECK(policy::configured_cap(nullptr, "4096", "8192") == 4096);
    CHECK(policy::configured_cap(nullptr, nullptr, "4096") == 4096);
    CHECK(policy::configured_cap("0", "4096", "8192") == 0);
    for (const char* bad : {"", "-1", "1", "15", "4096junk", "9999999999999999999999999999"})
        CHECK(policy::configured_cap(bad, "4096", nullptr) == 0);
    CHECK(policy::configured_cap("16", nullptr, nullptr) == 16);
    CHECK(policy::capped_rows(32768, 4097) == 4096);
    CHECK(policy::capped_rows(32, 4096) == 32);
    CHECK(policy::capped_rows(INT_MAX, INT_MAX - 1) > 0);
    for (int candidate = 1; candidate <= 512; ++candidate)
        for (int cap = -16; cap <= 512; ++cap) {
            const int rows = policy::capped_rows(candidate, cap);
            CHECK(rows > 0 && rows <= candidate);
            if (rows != candidate) CHECK(cap >= 16 && rows <= cap && rows % 16 == 0);
        }
    CHECK(policy::gfx1151_arch("gfx1151"));
    CHECK(policy::gfx1151_arch("gfx1151:xnack-"));
    for (const char* other : {"", "gfx11", "gfx1100", "gfx1150", "gfx11510", "gfx1201"})
        CHECK(!policy::gfx1151_arch(other));
    CHECK(!policy::gfx1151_arch(nullptr));
    std::puts("ds4_chunk_policy_test: aliases, bounds, architecture and270848 plan properties passed");
}
