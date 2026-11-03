#include "../src/ds4/ds4_tp_partition.hpp"

#include <array>
#include <climits>
#include <cstdio>
#include <string>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); return 1; } } while (0)

int main() {
    using strata::ds4::tp2::Range;
    using strata::ds4::tp2::expert_ranges;
    std::array<Range, 2> p{};
    std::string error;
    CHECK(expert_ranges(2304, 256, 0, p, error));
    CHECK(p[0].begin == 0 && p[0].count == 1280 && p[1].begin == 1280 && p[1].count == 1024);
    CHECK(expert_ranges(2304, 256, 1, p, error));
    CHECK(p[0].count == 1024 && p[1].begin == 1024 && p[1].count == 1280);
    for (int block : {1, 32, 128, 256}) {
        for (int blocks = 2; blocks <= 257; ++blocks) {
            for (int layer = 0; layer != 4; ++layer) {
                const int width = block * blocks;
                CHECK(expert_ranges(width, block, layer, p, error));
                CHECK(error.empty());
                CHECK(p[0].begin == 0 && p[1].begin == p[0].count);
                CHECK(p[0].count > 0 && p[1].count > 0);
                CHECK(p[0].count % block == 0 && p[1].count % block == 0);
                CHECK(p[0].count + p[1].count == width);
                CHECK(p[0].count - p[1].count <= block && p[1].count - p[0].count <= block);
            }
        }
    }
    p = {{{3, 4}, {5, 6}}};
    for (const auto bad : {std::array<int, 3>{0, 256, 0}, {-1, 256, 0}, {2304, 0, 0},
                           {2304, -1, 0}, {2304, 256, -1}, {2303, 256, 0}, {256, 256, 0}}) {
        CHECK(!expert_ranges(bad[0], bad[1], bad[2], p, error));
        CHECK(!error.empty() && p[0].begin == 3 && p[0].count == 4 && p[1].begin == 5 && p[1].count == 6);
    }
    CHECK(expert_ranges(INT_MAX, 1, INT_MAX, p, error));
    CHECK(p[0].count + p[1].count == INT_MAX);
    std::puts("DeepSeek TP2 whole-block ranges: geometry, alternation, coverage and invalid inputs passed");
}
