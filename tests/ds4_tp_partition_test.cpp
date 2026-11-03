#include "../src/ds4/ds4_tp_partition.hpp"

#include <array>
#include <climits>
#include <cstdio>
#include <limits>
#include <string>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); return 1; } } while (0)

int main() {
    using strata::ds4::tp2::Range;
    using strata::ds4::tp2::expert_ranges;
    using strata::ds4::tp2::expert_spans;
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
                if (blocks % 2) CHECK((layer % 2 == 0) == (p[0].count > p[1].count));
                else CHECK(p[0].count == p[1].count);
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
    // Actual Maya-S layer-0 IQ2_S gate/up and IQ3_S down row sizes from the pack.
    strata::ds4::tp2::ExpertSpans a{}, b{};
    CHECK(expert_ranges(2304, 256, 0, p, error));
    CHECK(expert_spans(2304, 5120, 256, p[0], 1640, 1640, 990, a, error));
    CHECK(expert_spans(2304, 5120, 256, p[1], 1640, 1640, 990, b, error));
    CHECK(a.gate_offset == 0 && a.gate_bytes == 2099200 && b.gate_offset == a.gate_bytes);
    CHECK(a.up_offset == 0 && a.up_bytes == a.gate_bytes && b.up_offset == a.up_bytes);
    CHECK(a.down_row_offset == 0 && a.down_row_bytes == 550 && b.down_row_offset == 550);
    CHECK(b.down_row_bytes == 440 && a.down_source_stride == 990 && a.down_rows == 5120);
    CHECK(a.compact_bytes + b.compact_bytes == 2 * 3778560 + 5068800);
    CHECK(!expert_spans(2304, 5120, 256, {128, 1024}, 1640, 1640, 990, a, error));
    CHECK(!expert_spans(2304, 5120, 256, {0, 1280}, 1640, 1640, 991, a, error));
    CHECK(!expert_spans(2304, 5120, 256, {0, 1280}, std::numeric_limits<std::size_t>::max(),
                        1640, 990, a, error));
    std::puts("DeepSeek TP2 whole-block ranges: geometry, alternation, coverage and invalid inputs passed");
}
