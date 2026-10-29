// src/kernels/cpu/native_down_cols_test.cpp - two-host TP: a rank's down column band sums to the full down.
// Synthetic, CPU only.  Two bands of a Q8_0 down (n_ff 64 = 2 blocks) must add up to native_down_rows.
#include "strata/kernels/cpu/native_expert.hpp"

#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "native_down_cols_test: %s\n", what);
        std::exit(1);
    }
}

int main() {
    namespace cpu = strata::kernels::cpu;
    cpu::NativeFmt f;
    std::string err;
    // Q8_0 gate/up and down: block 32, so n_ff 64 splits into two block-aligned bands
    check(cpu::native_fmt(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 32, 64, f, err), err.c_str());
    check(f.n_ff == 64 && f.bytes > 0, "geometry");

    std::vector<unsigned char> blob(f.bytes), hq(f.h_bytes);
    // distinct weights so the bands do not accidentally match
    std::vector<float> w((size_t) f.n_embd * 64);
    for (size_t i = 0; i < w.size(); ++i) w[i] = (float) ((i * 17) % 23) * 0.25f - 2.0f;
    ggml_quantize_chunk(GGML_TYPE_Q8_0, w.data(), blob.data() + f.down_off, 0, (int64_t) f.n_embd, 64, nullptr);
    // activation: two tokens, every value distinct
    std::vector<float> h0(64), h1(64);
    for (int i = 0; i < 64; ++i) {
        h0[(size_t) i] = 0.05f * (float) i - 1.5f;
        h1[(size_t) i] = -0.03f * (float) i + 0.7f;
    }
    cpu::native_quant_h(f, h0.data(), hq.data());
    // second token's activation in its own buffer (native_quant_h writes one)
    std::vector<unsigned char> hq1(f.h_bytes);
    cpu::native_quant_h(f, h1.data(), hq1.data());

    const void* both[] = {hq.data(), hq1.data()};
    std::vector<float> full0((size_t) f.n_embd), full1((size_t) f.n_embd);
    float* fout[] = {full0.data(), full1.data()};
    cpu::native_down_rows(f, blob.data(), both, 2, fout, 0, (int) f.n_embd);

    std::vector<float> a0((size_t) f.n_embd), a1((size_t) f.n_embd), b0((size_t) f.n_embd), b1((size_t) f.n_embd);
    float* a[] = {a0.data(), a1.data()};
    float* b[] = {b0.data(), b1.data()};
    cpu::native_down_cols(f, blob.data() + f.down_off, 0, 32, both, 2, a);
    cpu::native_down_cols(f, blob.data() + f.down_off, 32, 32, both, 2, b);

    for (int r = 0; r < (int) f.n_embd; ++r) {
        const float s0 = a0[(size_t) r] + b0[(size_t) r];
        const float s1 = a1[(size_t) r] + b1[(size_t) r];
        check(std::abs(s0 - full0[(size_t) r]) < 1e-4f, "token 0 band sum equals full");
        check(std::abs(s1 - full1[(size_t) r]) < 1e-4f, "token 1 band sum equals full");
    }
    std::puts("native_down_cols_test: ok");
    return 0;
}
