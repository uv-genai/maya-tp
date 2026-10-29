// tests/tp_partition_test.cpp - host-side checks of the two-host TP split planner (no GPU, no model).
#include "strata/core/tp_partition.hpp"

#include <cstdio>
#include <string>

using namespace strata;

static int g_fail = 0;

static void ok(bool cond, const char* what) {
    if (!cond) {
        std::printf("FAIL %s\n", what);
        ++g_fail;
    }
}

int main() {
    std::string err;
    TpPartition p;

    // world=1 takes everything, no block constraint
    ok(TpPartition::create(2048, 0, 1, 256, p, err) && p.begin == 0 && p.count == 2048, "world 1");

    // Maya-S expert intermediate: 2048 -> 1024, IQ2_XXS/IQ2_S blocks of 256
    ok(TpPartition::create(2048, 0, 2, 256, p, err) && p.begin == 0 && p.count == 1024, "expert rank 0");
    ok(TpPartition::create(2048, 1, 2, 256, p, err) && p.begin == 1024 && p.count == 1024, "expert rank 1");
    ok(p.distributed(), "distributed");

    // dense lead 12288 -> 6144
    ok(TpPartition::create(12288, 1, 2, 256, p, err) && p.begin == 6144 && p.count == 6144, "dense rank 1");

    // refuse a width that does not divide
    ok(!TpPartition::create(2047, 0, 2, 256, p, err), "odd ff refused");
    ok(!err.empty(), "odd ff names the error");

    // refuse a share that cuts a quant block (world=4: 2048/4=512 ok; 1024/8=128 not a 256-block)
    ok(TpPartition::create(2048, 0, 4, 256, p, err) && p.count == 512, "world 4 share 512");
    ok(!TpPartition::create(1024, 0, 8, 256, p, err), "share 128 refused");

    // bad rank
    ok(!TpPartition::create(2048, 2, 2, 256, p, err), "rank 2 of world 2 refused");

    // blob spans: geometry [gate | up | down]
    // n_embd=4096, n_ff=2048, gu_row=160, d_row=1680 -> up_off=327680, down_off=655360, bytes=7536640
    TpBlobGeom g;
    g.n_embd = 4096;
    g.n_ff = 2048;
    g.gu_row = 160;
    g.d_row = 1680;
    g.up_off = (size_t) g.n_ff * g.gu_row;
    g.down_off = 2 * g.up_off;
    g.bytes = g.down_off + g.d_row * (size_t) g.n_embd;

    TpPartition::create(2048, 1, 2, 256, p, err);
    TpBlobSpans s;
    ok(tp_blob_spans(g, p, s, err), "blob spans ok");
    ok(s.gate_off == (size_t) 1024 * 160 && s.gate_bytes == (size_t) 1024 * 160, "gate slice");
    ok(s.up_off == g.up_off + (size_t) 1024 * 160 && s.up_bytes == s.gate_bytes, "up slice");
    ok(s.down_off == g.down_off && s.down_row_bytes == g.d_row, "down plane");
    ok(s.down_span_bytes == tp_span_bytes(g.d_row, 2048, 1024), "down span");
    ok(s.down_span_bytes == 840, "down span 1024/2048 of 1680");
    ok(s.n_embd == 4096, "n_embd");

    // span math
    ok(tp_span_bytes(1680, 2048, 1024) == 840, "span half");
    ok(tp_span_bytes(1680, 2048, 2048) == 1680, "span whole");
    ok(tp_span_bytes(1680, 2048, 2049) == 0, "span overflow");

    // geometry mismatch refused
    TpBlobGeom bad = g;
    bad.n_ff = 1024;
    ok(!tp_blob_spans(bad, p, s, err), "ff mismatch refused");
    bad = g;
    bad.down_off = 0;
    ok(!tp_blob_spans(bad, p, s, err), "bad geometry refused");

    // ggml block table
    ok(tp_ggml_block(16) == 256, "IQ2_XXS block");
    ok(tp_ggml_block(18) == 256, "IQ3_XXS block");
    ok(tp_ggml_block(22) == 256, "IQ2_S block");
    ok(tp_ggml_block(8) == 32, "Q8_0 block");
    ok(tp_ggml_block(999) == 0, "unknown type");

    if (g_fail) {
        std::printf("tp_partition_test: %d FAILED\n", g_fail);
        return 1;
    }
    std::printf("tp_partition_test OK\n");
    return 0;
}
