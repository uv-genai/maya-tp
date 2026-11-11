#pragma once
// Opaque GPU handles for CPU-only metadata/layout fixtures. No runtime API is emulated.
// Production HIP builds include the real compatibility header, never tests/host.
struct ihipStream_t;
struct ihipEvent_t;
using cudaStream_t = ihipStream_t*;
using cudaEvent_t = ihipEvent_t*;
using cudaError_t = int;
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
