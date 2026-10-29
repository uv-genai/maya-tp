// tests/tp_context_test.cpp - TpContext partition + no-op checks (host only, no GPU, no model).
#include "strata/core/tp_context.hpp"

#include <cstdio>
#include <string>

using namespace strata::core;

static int g_fail = 0;

static void ok(bool cond, const char* what) {
    if (!cond) {
        std::printf("FAIL %s\n", what);
        ++g_fail;
    }
}

int main() {
    std::string err;
    TpContext tp;

    // world 1: no-op, full widths
    ok(TpContext::create(0, 1, 2048, 12288, 256, nullptr, tp, err), "world 1 create");
    ok(!tp.active(), "world 1 inactive");
    ok(tp.experts.count == 2048 && tp.dense.count == 12288, "world 1 full widths");
    ok(tp.drain(err), "world 1 drain no-op");

    // world 2: halves
    TpContext r0, r1;
    ok(TpContext::create(0, 2, 2048, 12288, 256, nullptr, r0, err), "rank 0 create");
    ok(TpContext::create(1, 2, 2048, 12288, 256, nullptr, r1, err), "rank 1 create");
    ok(r0.experts.begin == 0 && r0.experts.count == 1024, "r0 experts");
    ok(r1.experts.begin == 1024 && r1.experts.count == 1024, "r1 experts");
    ok(r0.dense.begin == 0 && r0.dense.count == 6144, "r0 dense");
    ok(r1.dense.begin == 6144 && r1.dense.count == 6144, "r1 dense");
    ok(!r0.active(), "no comm yet");   // comm is null

    // refuse a width that does not split
    ok(!TpContext::create(0, 2, 2047, 12288, 256, nullptr, tp, err), "odd ff refused");

    if (g_fail) {
        std::printf("tp_context_test: %d FAILED\n", g_fail);
        return 1;
    }
    std::printf("tp_context_test OK\n");
    return 0;
}
