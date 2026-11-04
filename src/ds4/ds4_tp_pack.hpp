// Safe host packing of a rank's GGUF expert tensor slices.
#pragma once

#include "ds4_tp_partition.hpp"
#include <array>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>

namespace strata::ds4::tp2 {
using ConstBytes = std::span<const std::uint8_t>;
using Bytes = std::span<std::uint8_t>;

inline bool byte_ranges_overlap(ConstBytes a, ConstBytes b) {
    if (a.empty() || b.empty()) return false;
    const auto x = reinterpret_cast<std::uintptr_t>(a.data());
    const auto y = reinterpret_cast<std::uintptr_t>(b.data());
    return x <= y ? y - x < a.size() : x - y < b.size();
}

// Source arrays contain the complete gate, up and down tensors, individually.
// Destination arrays hold the compact rank slices; callers own layout/alignment.
// All sizes and overlap constraints are checked before any output is written.
inline bool pack_expert(const ExpertSpans& s, const std::array<ConstBytes, 3>& source,
                        const std::array<Bytes, 3>& dest, std::string& error) {
    const auto fail = [&] { error = "ds4 tp2: invalid expert buffers or overlapping slices"; return false; };
    if (s.gate_bytes == 0 || s.up_bytes == 0 || s.down_row_bytes == 0 || s.down_rows == 0 ||
        s.down_source_stride == 0 || s.gate_offset > source[0].size() ||
        s.gate_bytes > source[0].size() - s.gate_offset || s.up_offset > source[1].size() ||
        s.up_bytes > source[1].size() - s.up_offset || s.down_row_offset > s.down_source_stride ||
        s.down_row_bytes > s.down_source_stride - s.down_row_offset ||
        s.down_rows > source[2].size() / s.down_source_stride ||
        s.down_rows > dest[2].size() / s.down_row_bytes || dest[0].size() < s.gate_bytes ||
        dest[1].size() < s.up_bytes) return fail();
    for (const auto& d : dest) {
        for (const auto& input : source) if (byte_ranges_overlap(d, input)) return fail();
    }
    for (std::size_t i = 0; i < dest.size(); ++i)
        for (std::size_t j = i + 1; j < dest.size(); ++j)
            if (byte_ranges_overlap(dest[i], dest[j])) return fail();
    std::memcpy(dest[0].data(), source[0].data() + s.gate_offset, s.gate_bytes);
    std::memcpy(dest[1].data(), source[1].data() + s.up_offset, s.up_bytes);
    for (std::size_t row = 0; row < s.down_rows; ++row)
        std::memcpy(dest[2].data() + row * s.down_row_bytes,
                    source[2].data() + row * s.down_source_stride + s.down_row_offset, s.down_row_bytes);
    error.clear(); return true;
}
}  // namespace strata::ds4::tp2
