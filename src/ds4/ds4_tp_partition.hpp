// Two-host DeepSeek tensor partition geometry; no model or transport state.
#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <string>

namespace strata::ds4::tp2 {

struct Range {
    int begin = 0;
    int count = 0;
};

// Split a full FFN width into two contiguous ranges of whole quantization blocks.
// The extra block goes to alternating ranks so odd block counts balance across
// layers. The caller must supply a block compatible with every sliced tensor's
// quant format and separately validate head/output-group geometry.
inline bool expert_ranges(int width, int block, int layer, std::array<Range, 2>& out, std::string& error) {
    if (width <= 0 || block <= 0 || layer < 0 || width % block != 0 || width / block < 2) {
        error = "ds4 tp2: expert width must contain at least two complete blocks and layer must be nonnegative";
        return false;
    }
    const int blocks = width / block;
    const int rank0_blocks = blocks / 2 + ((blocks % 2 != 0 && layer % 2 == 0) ? 1 : 0);
    const int cut = rank0_blocks * block;  // <= width, so cannot overflow a positive int.
    const std::array<Range, 2> ranges{{{0, cut}, {cut, width - cut}}};
    out = ranges;
    error.clear();
    return true;
}

// Byte ranges in separate packed GGUF expert matrices. Gate/up select complete
// output rows; down selects the same input columns in *each* hidden row. The
// latter is strided in the source and must be packed row by row for a rank.
struct ExpertSpans {
    std::size_t gate_offset = 0, gate_bytes = 0;
    std::size_t up_offset = 0, up_bytes = 0;
    std::size_t down_row_offset = 0, down_row_bytes = 0, down_source_stride = 0;
    std::size_t down_rows = 0, compact_bytes = 0;
};

inline bool expert_spans(int width, int hidden, int block, Range part, std::size_t gate_row,
                         std::size_t up_row, std::size_t down_row, ExpertSpans& out, std::string& error) {
    const auto fail = [&] { error = "ds4 tp2: invalid or overflowing expert tensor geometry"; return false; };
    if (width <= 0 || hidden <= 0 || block <= 0 || width % block != 0 || part.begin < 0 || part.count <= 0 ||
        part.begin % block != 0 || part.count % block != 0 || part.begin > width || part.count > width - part.begin ||
        gate_row == 0 || up_row == 0 || down_row == 0) return fail();
    const std::size_t blocks = static_cast<std::size_t>(width / block);
    if (down_row % blocks != 0) return fail();
    const auto mul = [](std::size_t a, std::size_t b, std::size_t& r) {
        if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) return false;
        r = a * b; return true;
    };
    ExpertSpans s;
    std::size_t down_bytes = 0, source_bytes = 0;
    const auto begin = static_cast<std::size_t>(part.begin), count = static_cast<std::size_t>(part.count);
    const auto down_block_bytes = down_row / blocks;
    if (!mul(static_cast<std::size_t>(width), gate_row, source_bytes) ||
        !mul(static_cast<std::size_t>(width), up_row, source_bytes) ||
        !mul(static_cast<std::size_t>(hidden), down_row, source_bytes) ||
        !mul(begin, gate_row, s.gate_offset) || !mul(count, gate_row, s.gate_bytes) ||
        !mul(begin, up_row, s.up_offset) || !mul(count, up_row, s.up_bytes) ||
        !mul(begin / static_cast<std::size_t>(block), down_block_bytes, s.down_row_offset) ||
        !mul(count / static_cast<std::size_t>(block), down_block_bytes, s.down_row_bytes) ||
        !mul(static_cast<std::size_t>(hidden), s.down_row_bytes, down_bytes)) return fail();
    const auto max = std::numeric_limits<std::size_t>::max();
    if (s.gate_bytes > max - s.up_bytes || s.gate_bytes + s.up_bytes > max - down_bytes) return fail();
    s.down_source_stride = down_row;
    s.down_rows = static_cast<std::size_t>(hidden);
    s.compact_bytes = s.gate_bytes + s.up_bytes + down_bytes;
    out = s;
    error.clear();
    return true;
}

}  // namespace strata::ds4::tp2
