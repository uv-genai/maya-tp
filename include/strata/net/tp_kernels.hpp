// include/strata/net/tp_kernels.hpp - the GPU half of a queued TP2 exchange (gufo's 6 us path).
//
// QueuePartial hands the host-side window flags; these kernels do the rest on the stream so the CPU never
// round-trips: StagePartial copies the partial into the registered send window and publishes `ready`, WaitValue
// holds the stream until the peer's write raised `arrived`, AddRowsBroadcast adds the peer partial in place.
#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace strata::net {

/// Copies `count` floats from `partial` to `send` (the registered window) and publishes `value` in `ready`
/// once every block has finished (system scope, so the host/peer sees the whole partial).
void StagePartial(const float* partial, float* send, std::size_t count, std::uint32_t* arrivals,
                  std::uint64_t* ready, std::uint64_t value, hipStream_t stream);

/// One-wave spin until `*flag >= value`, or `ticks` of the device wall clock (then stores 1 in `timed_out`).
void WaitValue(const std::uint64_t* flag, std::uint64_t value, std::uint64_t ticks, std::uint32_t* timed_out,
               hipStream_t stream);

/// residual[t] += addend  for every row (broadcast): the two-rank all-reduce's add of the peer partial.
void AddRowsBroadcast(const float* addend, float* residual, std::uint32_t rows, std::uint32_t hidden,
                      std::uint32_t streams, hipStream_t stream);

}  // namespace strata::net
