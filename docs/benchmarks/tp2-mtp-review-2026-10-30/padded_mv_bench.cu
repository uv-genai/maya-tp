// Padded Q6_K TP projection microbenchmark, derived from tests/hip/glm_mv_bench.cu.
// Compare the original fallback and default RDNA dispatch at NT=2/3/8 with rotating
// weights and bit-exact output/padding checks. Build/run instructions: RESULTS.md.
#include "strata/kernels/glm_expert_bench.hpp"
#include "strata/kernels/glm_fast.hpp"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace gf = strata::kernels::glmf;
namespace {
void check(hipError_t e, const char* what) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
        std::exit(2);
    }
}
uint16_t half_bits(float f) { return std::bit_cast<uint16_t>(__float2half(f)); }

struct Shape {
    int type, n_in, n_out;
};
struct Case {
    const char* name;
    std::vector<Shape> jobs;
};
// the engine's batches (glm_fast_path.cu / glm_mtp.cu), Maya-S v2 types
const std::vector<Case> kCases = {
    {"kda_proj", {{14, 4096, 8192}, {14, 4096, 8192}, {14, 4096, 8192}, {30, 4096, 128}, {30, 4096, 128}, {30, 4096, 64}}},
    {"kda_out", {{14, 8192, 4096}}},
    {"dsa_proj", {{14, 4096, 1536}, {14, 4096, 512}, {30, 4096, 128}, {30, 4096, 128}, {30, 4096, 32}}},
    {"dsa_q", {{14, 1536, 16384}, {30, 1536, 4096}}},
    {"dsa_out", {{14, 16384, 4096}}},
    {"router_shexp", {{30, 4096, 288}, {14, 4096, 2048}, {14, 4096, 2048}}},
    {"dense_gu", {{14, 4096, 12288}, {14, 4096, 12288}}},
    {"dense_down", {{14, 12288, 4096}}},
    {"head", {{14, 4096, 154880}}},
};
size_t wbytes(const Shape& s) { return gf::row_bytes(s.type, s.n_in) * (size_t) s.n_out; }

void fill_weights(std::vector<uint8_t>& v, const Shape& s, std::mt19937& rng) {
    v.resize(wbytes(s));
    if (s.type == 14) {
        for (size_t i = 0; i < v.size(); i += sizeof(block_q6_K)) {
            for (size_t j = 0; j < sizeof(block_q6_K); ++j) v[i + j] = rng() & 255;
            const float m = (1 + (rng() % 1024)) / 65536.f;
            const uint16_t d = half_bits((rng() & 1) ? m : -m);
            std::memcpy(v.data() + i + offsetof(block_q6_K, d), &d, 2);
        }
    } else {   // BF16 ~ N(0, 0.05)
        std::normal_distribution<float> nd(0.f, 0.05f);
        for (size_t i = 0; i < v.size(); i += 2) {
            const uint32_t b = std::bit_cast<uint32_t>(nd(rng));
            const uint16_t h = (uint16_t) (b >> 16);
            std::memcpy(v.data() + i, &h, 2);
        }
    }
}
// nt rows of n values: f32 and their q8_1
void fill_act(int n, int nt, std::mt19937& rng, std::vector<float>& xf, std::vector<uint8_t>& xq) {
    std::normal_distribution<float> nd(0.f, 1.f);
    xf.resize((size_t) n * nt);
    for (auto& f : xf) f = nd(rng);
    xq.resize((size_t) n / 32 * nt * sizeof(block_q8_1));
    for (size_t b = 0; b < xf.size() / 32; ++b) {
        float amax = 0, sum = 0;
        for (int i = 0; i < 32; ++i) { amax = std::max(amax, std::fabs(xf[32 * b + i])); sum += xf[32 * b + i]; }
        const float d = amax / 127.f;
        uint8_t* p = xq.data() + b * sizeof(block_q8_1);
        const uint16_t dh = half_bits(d), sh = half_bits(sum);
        std::memcpy(p, &dh, 2);
        std::memcpy(p + 2, &sh, 2);
        for (int i = 0; i < 32; ++i) p[4 + i] = uint8_t(int8_t(d == 0 ? 0 : std::lround(xf[32 * b + i] / d)));
    }
}

struct Config {
    int r, u, wpb, lds;
    std::string label() const {
        if (r == 0) return "original";
        if (r < 0) return "default";
        return "r" + std::to_string(r) + "u" + std::to_string(u) + "w" + std::to_string(wpb) + (lds ? "L" : "");
    }
};

// a pure read of `bytes` (uint4 loads, U in flight per lane): what the memory gives a streaming kernel
__global__ void read_kernel(const uint4* __restrict__ p, size_t n16, unsigned* out) {
    uint32_t acc = 0;
    const size_t stride = (size_t) gridDim.x * blockDim.x;
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    for (; i + 3 * stride < n16; i += 4 * stride) {
        const uint4 a = p[i], b = p[i + stride], c = p[i + 2 * stride], d = p[i + 3 * stride];
        acc ^= a.x ^ a.y ^ a.z ^ a.w ^ b.x ^ b.y ^ b.z ^ b.w ^ c.x ^ c.y ^ c.z ^ c.w ^ d.x ^ d.y ^ d.z ^ d.w;
    }
    for (; i < n16; i += stride) acc ^= p[i].x ^ p[i].y ^ p[i].z ^ p[i].w;
    if (acc == 0x12345678u) out[0] = acc;
}

double time_us(hipStream_t s, int calls, const std::function<void(int)>& launch) {
    hipEvent_t a, b;
    check(hipEventCreate(&a), "event");
    check(hipEventCreate(&b), "event");
    for (int i = 0; i < 4; ++i) launch(i);
    check(hipStreamSynchronize(s), "warmup");
    std::vector<double> v;
    for (int rnd = 0; rnd < 7; ++rnd) {
        check(hipEventRecord(a, s), "record");
        for (int i = 0; i < calls; ++i) launch(i);
        check(hipEventRecord(b, s), "record");
        check(hipEventSynchronize(b), "sync");
        float ms = 0;
        check(hipEventElapsedTime(&ms, a, b), "elapsed");
        v.push_back(ms * 1000.0 / calls);
    }
    std::sort(v.begin(), v.end());
    check(hipEventDestroy(a), "event");
    check(hipEventDestroy(b), "event");
    return v[v.size() / 2];
}

} // namespace
int main() {
    check(hipSetDevice(0), "device");
    hipStream_t stream; check(hipStreamCreate(&stream), "stream");
    std::mt19937 rng(0x25ab1151u);
    bool all_exact=true;
    for (const Shape shape : {Shape{14,4096,4096},Shape{14,1536,8192},Shape{14,4096,6144}}) {
        const size_t bytes=wbytes(shape);
        const int copies=(int)std::min<size_t>(32,(256u<<20)/bytes+1);
        std::vector<void*> weights(copies);
        for(int i=0;i<copies;++i){
            std::vector<uint8_t> w;fill_weights(w,shape,rng);
            check(hipMalloc(&weights[i],w.size()),"weights");
            check(hipMemcpy(weights[i],w.data(),w.size(),hipMemcpyHostToDevice),"weights H2D");
        }
        std::vector<float> values;std::vector<uint8_t> quant;
        fill_act(shape.n_in,8,rng,values,quant);
        void* activation;float* output;
        check(hipMalloc(&activation,quant.size()),"activation");
        check(hipMemcpy(activation,quant.data(),quant.size(),hipMemcpyHostToDevice),"activation H2D");
        const size_t size=(size_t)8*2*shape.n_out;
        check(hipMalloc(&output,size*4),"output");
        for(int nt:{2,3,8}) {
            auto launch=[&](int index){
                gf::MvJob job={weights[index%copies],activation,nullptr,output,nullptr,1.0f,14,shape.n_in,shape.n_out};
                job.ncol=nt;job.xq_cstride=(int64_t)shape.n_in/32*sizeof(block_q8_1);job.y_cstride=2*shape.n_out;
                if(!gf::mv(&job,1,stream))std::exit(2);
            };
            gf::hip_expert_bench::mv_config(0,1,8,0);
            const double before=time_us(stream,20,launch);
            check(hipMemset(output,0xff,size*4),"reference sentinel");launch(0);
            check(hipStreamSynchronize(stream),"reference sync");
            std::vector<uint32_t> ref(size),got(size);
            check(hipMemcpy(ref.data(),output,size*4,hipMemcpyDeviceToHost),"reference D2H");
            gf::hip_expert_bench::mv_config(-1,2,8,-1);
            const double after=time_us(stream,20,launch);
            check(hipMemset(output,0xff,size*4),"output sentinel");launch(0);
            check(hipStreamSynchronize(stream),"output sync");
            check(hipMemcpy(got.data(),output,size*4,hipMemcpyDeviceToHost),"output D2H");
            const bool exact=ref==got;all_exact&=exact;
            std::printf("PADDED nin=%d nout=%d nt=%d copies=%d original_us=%.3f rdna_us=%.3f speedup=%.4f exact=%d\n",shape.n_in,shape.n_out,nt,copies,before,after,before/after,(int)exact);
            std::fflush(stdout);
        }
        for(void* p:weights)check(hipFree(p),"free weights");
        check(hipFree(activation),"free activation");check(hipFree(output),"free output");
    }
    check(hipStreamDestroy(stream),"destroy");
    return all_exact?0:1;
}
