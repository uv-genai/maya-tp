#pragma once
// Opaque stream handle for CPU-only control fixtures. No GPU API is emulated.
// Only the host-test compiler includes this directory; HIP builds use ROCm.
struct ihipStream_t;
using hipStream_t=ihipStream_t*;
