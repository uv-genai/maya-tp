// src/net/tp_kernels.cu - GPU stage / wait / add for queued TP2 exchanges (gfx1151 HIP).
#include "strata/net/tp_kernels.hpp"

#include <algorithm>

namespace strata::net {
namespace {

constexpr int kThreads = 256;

// Every thread's copies reach the system scope before its block is counted; the last block publishes
// `value`, so the host that reads it also sees the whole partial.
__global__ void StagePartialKernel(const float* __restrict__ partial, float* __restrict__ send, size_t count,
                                   uint32_t* arrivals, uint64_t* ready, uint64_t value) {
    const size_t stride = (size_t) gridDim.x * blockDim.x;
    const size_t first = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t vectors = count / 4;
    for (size_t i = first; i < vectors; i += stride) reinterpret_cast<float4*>(send)[i] = reinterpret_cast<const float4*>(partial)[i];
    for (size_t i = vectors * 4 + first; i < count; i += stride) send[i] = partial[i];
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0 && __hip_atomic_fetch_add(arrivals, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_SYSTEM) + 1 == gridDim.x) {
        __hip_atomic_store(arrivals, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        __hip_atomic_store(ready, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
}

__global__ void WaitValueKernel(const uint64_t* flag, uint64_t value, uint64_t ticks, uint32_t* timed_out) {
    const uint64_t start = wall_clock64();
    while (__hip_atomic_load(flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < value) {
        if (wall_clock64() - start > ticks) {
            __hip_atomic_store(timed_out, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
            return;
        }
        __builtin_amdgcn_s_sleep(1);
    }
}

// Fused WaitValue + AddRowsBroadcast: spin until `arrived`, then add the peer's partial.
// Saves one kernel launch per exchange over the two-kernel form.
__global__ void WaitAddKernel(const uint64_t* flag, uint64_t value, uint64_t ticks, uint32_t* timed_out,
                              const float* __restrict__ addend, float* __restrict__ residual,
                              uint32_t hidden, uint32_t streams) {
    // one block for the wait, then the add over the whole grid
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const uint64_t start = wall_clock64();
        while (__hip_atomic_load(flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < value) {
            if (wall_clock64() - start > ticks) {
                __hip_atomic_store(timed_out, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
                return;
            }
            __builtin_amdgcn_s_sleep(1);
        }
    }
    __syncthreads();
    // the grid-wide add (same as AddRowsBroadcastKernel)
    const uint32_t t = blockIdx.x;
    const uint32_t i = blockIdx.y * blockDim.x + threadIdx.x;
    if (t >= streams || i >= hidden) return;
    residual[(size_t) t * streams * hidden + i] += addend[(size_t) t * hidden + i % hidden];
}

__global__ void AddRowsBroadcastKernel(const float* addend, float* residual, uint32_t hidden, uint32_t streams) {
    const uint32_t t = blockIdx.x;
    const uint32_t i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= streams * hidden) return;
    residual[(size_t) t * streams * hidden + i] += addend[(size_t) t * hidden + i % hidden];
}

__global__ void AddRowsBroadcastVec4Kernel(const float4* addend, float4* residual, uint32_t hidden4,
                                           uint32_t streams) {
    const uint32_t t = blockIdx.x;
    const uint32_t i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= streams * hidden4) return;
    const float4 a = addend[(size_t) t * hidden4 + i % hidden4];
    float4& r = residual[(size_t) t * streams * hidden4 + i];
    float4 v = r;
    v.x += a.x;
    v.y += a.y;
    v.z += a.z;
    v.w += a.w;
    r = v;
}

inline unsigned Blocks(size_t n) {
    return (unsigned) std::min<size_t>(65535, (n + kThreads - 1) / kThreads);
}

}  // namespace

void StagePartial(const float* partial, float* send, size_t count, uint32_t* arrivals, uint64_t* ready,
                  uint64_t value, hipStream_t stream) {
    const size_t threads = std::max<size_t>(1, count / 4);
    const unsigned blocks = (unsigned) std::min<size_t>(1024, (threads + kThreads - 1) / kThreads);
    hipLaunchKernelGGL(StagePartialKernel, dim3(blocks), dim3(kThreads), 0, stream, partial, send, count, arrivals,
                       ready, value);
}

void WaitValue(const uint64_t* flag, uint64_t value, uint64_t ticks, uint32_t* timed_out, hipStream_t stream) {
    hipLaunchKernelGGL(WaitValueKernel, dim3(1), dim3(1), 0, stream, flag, value, ticks, timed_out);
}

void WaitAdd(const uint64_t* flag, uint64_t value, uint64_t ticks, uint32_t* timed_out,
             const float* addend, float* residual, uint32_t rows, uint32_t hidden, uint32_t streams,
             hipStream_t stream) {
    // grid: (streams, ceil(hidden/threads)) — block (0,0) does the wait, all blocks do the add
    const dim3 grid(streams, (unsigned) ((hidden + kThreads - 1) / kThreads));
    hipLaunchKernelGGL(WaitAddKernel, grid, dim3(kThreads), 0, stream,
                       flag, value, ticks, timed_out, addend, residual, hidden, streams);
}

void AddRowsBroadcast(const float* addend, float* residual, uint32_t rows, uint32_t hidden, uint32_t streams,
                      hipStream_t stream) {
    if (rows == 0) return;
    if (hidden % 4 == 0 && streams == 1) {
        const uint32_t hidden4 = hidden / 4;
        hipLaunchKernelGGL(AddRowsBroadcastVec4Kernel, dim3(rows, Blocks(hidden4)), dim3(kThreads), 0, stream,
                           reinterpret_cast<const float4*>(addend), reinterpret_cast<float4*>(residual), hidden4,
                           streams);
        return;
    }
    hipLaunchKernelGGL(AddRowsBroadcastKernel, dim3(rows, Blocks((size_t) streams * hidden)), dim3(kThreads), 0,
                       stream, addend, residual, hidden, streams);
}

}  // namespace strata::net
