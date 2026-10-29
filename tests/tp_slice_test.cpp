// tests/tp_slice_test.cpp - tp_slice_expert: a rank's compact blob must make the GPU kernels' view
// (n_ff = ffn) equal that band of the full blob.  Synthetic, CPU only.
#include "strata/core/tp_partition.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace strata;

static int g_fail = 0;

static void ok(bool cond, const char* what) {
    if (!cond) {
        std::printf("FAIL %s\n", what);
        ++g_fail;
    }
}

int main() {
    // a small stand-in for IQ2_XXS/IQ2_S: gu_row 8 bytes/row (n_embd 8), d_row 4 bytes/row (n_ff 8)
    const int64_t n_embd = 8, n_ff = 8, ffn = 4, ff0 = 4;
    const size_t gu_row = 8, d_row = 4;
    // full blob: gate rows 0..7, up rows 0..7, down rows 0..7 x 8 cols
    std::vector<uint8_t> full(2 * n_ff * gu_row + n_embd * d_row);
    for (size_t i = 0; i < full.size(); ++i) full[i] = (uint8_t) (i + 1);
    // rank 1's band: gate rows [4,8), up rows [4,8), down cols [4,8) of each row
    std::vector<uint8_t> dst(2 * ffn * gu_row + n_embd * (d_row * ffn / n_ff));
    tp_slice_expert(full.data(), full.data() + n_ff * gu_row, full.data() + 2 * n_ff * gu_row, ff0, ffn, n_ff,
                    n_embd, gu_row, d_row, dst.data());

    // gate: rows 4..7 of the gate plane
    for (int r = 0; r < ffn; ++r)
        for (size_t b = 0; b < gu_row; ++b)
            ok(dst[(size_t) r * gu_row + b] == full[(size_t) (ff0 + r) * gu_row + b], "gate band");
    // up: same, offset into the up plane
    const size_t up_full = (size_t) n_ff * gu_row, up_dst = (size_t) ffn * gu_row;
    for (int r = 0; r < ffn; ++r)
        for (size_t b = 0; b < gu_row; ++b)
            ok(dst[up_dst + (size_t) r * gu_row + b] == full[up_full + (size_t) (ff0 + r) * gu_row + b], "up band");
    // down: cols [4,8) of each output row
    const size_t dn_full = 2 * up_full, dn_dst = 2 * up_dst;
    const size_t row_band = d_row * ffn / n_ff;   // 2
    for (int64_t r = 0; r < n_embd; ++r)
        for (size_t b = 0; b < row_band; ++b)
            ok(dst[dn_dst + (size_t) r * row_band + b] ==
                   full[dn_full + (size_t) r * d_row + d_row * ff0 / n_ff + b],
               "down band");

    // rank 0: [0,4)
    tp_slice_expert(full.data(), full.data() + n_ff * gu_row, full.data() + 2 * n_ff * gu_row, 0, ffn, n_ff, n_embd,
                    gu_row, d_row, dst.data());
    for (int r = 0; r < ffn; ++r)
        ok(dst[(size_t) r * gu_row] == full[(size_t) r * gu_row], "rank 0 gate start");
    for (int64_t r = 0; r < n_embd; ++r)
        ok(dst[dn_dst + (size_t) r * row_band] == full[dn_full + (size_t) r * d_row], "rank 0 down start");

    if (g_fail) {
        std::printf("tp_slice_test: %d FAILED\n", g_fail);
        return 1;
    }
    std::printf("tp_slice_test OK\n");
    return 0;
}
