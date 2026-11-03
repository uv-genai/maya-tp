// Two-host DeepSeek tensor partition geometry; no model or transport state.
#pragma once

#include <array>
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

}  // namespace strata::ds4::tp2
