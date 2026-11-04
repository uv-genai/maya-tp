// HIP upload of a single rank's block-aligned GGUF expert slices.
#pragma once

#include "ds4_tp_pack.hpp"
#include <hip/hip_runtime.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

namespace strata::ds4::tp2 {

struct DeviceExpertLayout {
    std::array<std::size_t, 3> offsets{};
    std::size_t bytes = 0;
};

// Matches the existing GGUF GPU expert layout: every matrix starts at a
// 256-byte boundary. Invalid/overflowing plans leave the output untouched.
inline bool device_expert_layout(const ExpertSpans& s, DeviceExpertLayout& out, std::string& error) {
    constexpr auto max = std::numeric_limits<std::size_t>::max();
    if (!s.gate_bytes || !s.up_bytes || !s.down_row_bytes || !s.down_rows ||
        s.down_rows > max / s.down_row_bytes) {
        error = "ds4 tp2 upload: invalid expert sizes"; return false;
    }
    auto align = [&](std::size_t n, std::size_t& a) {
        if (n > max - 255) return false;
        a = (n + 255) & ~std::size_t{255}; return true;
    };
    DeviceExpertLayout l;
    std::size_t down_bytes = s.down_rows * s.down_row_bytes, up_size = 0, down_size = 0;
    if (!align(s.gate_bytes, l.offsets[1]) || !align(s.up_bytes, up_size) ||
        !align(down_bytes, down_size) || l.offsets[1] > max - up_size) {
        error = "ds4 tp2 upload: expert layout overflows"; return false;
    }
    l.offsets[2] = l.offsets[1] + up_size;
    if (l.offsets[2] > max - down_size) {
        error = "ds4 tp2 upload: expert layout overflows"; return false;
    }
    l.bytes = l.offsets[2] + down_size;
    out = l; error.clear(); return true;
}

// Source spans are complete host matrices and must remain alive until the
// stream completes. dst must be a device allocation on stream's device. All
// geometry is validated before enqueueing; on a HIP failure the caller must
// discard the destination and drain/fail its operation.
inline bool upload_expert(const ExpertSpans& s, const std::array<ConstBytes, 3>& src,
                          std::uint8_t* dst, std::size_t dst_bytes, hipStream_t stream,
                          DeviceExpertLayout& out, std::string& error) {
    DeviceExpertLayout l;
    if (!device_expert_layout(s, l, error)) return false;
    if (dst == nullptr || dst_bytes < l.bytes || s.gate_offset > src[0].size() ||
        s.gate_bytes > src[0].size() - s.gate_offset || s.up_offset > src[1].size() ||
        s.up_bytes > src[1].size() - s.up_offset || s.down_source_stride == 0 ||
        s.down_row_offset > s.down_source_stride ||
        s.down_row_bytes > s.down_source_stride - s.down_row_offset ||
        s.down_rows > src[2].size() / s.down_source_stride) {
        error = "ds4 tp2 upload: source or destination does not cover expert slices"; return false;
    }
    hipError_t st = hipMemcpyAsync(dst + l.offsets[0], src[0].data() + s.gate_offset,
                                   s.gate_bytes, hipMemcpyHostToDevice, stream);
    if (st == hipSuccess)
        st = hipMemcpyAsync(dst + l.offsets[1], src[1].data() + s.up_offset,
                            s.up_bytes, hipMemcpyHostToDevice, stream);
    if (st == hipSuccess)
        st = hipMemcpy2DAsync(dst + l.offsets[2], s.down_row_bytes,
                              src[2].data() + s.down_row_offset, s.down_source_stride,
                              s.down_row_bytes, s.down_rows, hipMemcpyHostToDevice, stream);
    if (st != hipSuccess) { error = std::string("ds4 tp2 upload: ") + hipGetErrorString(st); return false; }
    out = l; error.clear(); return true;
}

} // namespace strata::ds4::tp2
