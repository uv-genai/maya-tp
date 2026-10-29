// include/strata/core/tp_partition.hpp - two-host tensor parallel: which slice of each split tensor a rank holds.
//
// Intra-expert split (gufo TP2): every rank holds [ff_begin, ff_begin + ff_count) of every expert's intermediate
// dimension - those rows of gate/up and those columns of down.  Both ranks run all selected experts on their half
// and one all-reduce sums the shares, so routing stays identical and neither rank waits on the other.
// Slices are byte spans of the existing quant blob (no re-encode).  A down share must land on a quantization
// block boundary or the load refuses it.
//
// Blob layout (tools/iq_pack.py, NativeFmt): [gate rows | up rows | down rows] = 2 * n_ff * gu_row + n_embd * d_row.
// Gate/up split is a whole-row slice (always block-safe).  Down is one row per output unit, columns of length
// n_ff; the share is ff_count values from each of n_embd rows (strided).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata {

/// One rank's share of a split width `whole` in units of `block` (quant block or head group).
struct TpShare {
    int begin = 0;  ///< first unit this rank owns
    int count = 0;  ///< units it owns
};

/// Two-rank tensor-parallel partition of an intermediate dimension.
struct TpPartition {
    int rank = 0;
    int world = 1;
    int ff = 0;       ///< full intermediate width (n_ff_exp / n_ff_dense)
    int begin = 0;    ///< this rank's first unit
    int count = 0;    ///< this rank's unit count

    bool distributed() const { return world > 1; }

    /// Equal split.  `block` is the quant block length in intermediate units (256 for IQ2_*/IQ3_*/Q6_K, 32 for
    /// Q8_0).  Fails when `ff` does not divide evenly, or a share would cut a quant block.  world=1 takes all.
    static bool create(int ff, int rank, int world, int block, TpPartition& out, std::string& err) {
        if (ff <= 0 || world <= 0 || rank < 0 || rank >= world) {
            err = "tp partition: bad rank/world/ff";
            return false;
        }
        if (world == 1) {
            out = {0, 1, ff, 0, ff};
            return true;
        }
        if (ff % world != 0) {
            err = "tp partition: " + std::to_string(ff) + " does not split into " + std::to_string(world);
            return false;
        }
        const int count = ff / world;
        if (block > 0 && count % block != 0) {
            err = "tp partition: share " + std::to_string(count) + " is not a multiple of block " +
                  std::to_string(block);
            return false;
        }
        out = {rank, world, ff, count * rank, count};
        return true;
    }
};

/// Bytes of `n` consecutive values of one ggml row, from the row's byte width `row_bytes` covering `row_values`.
/// Used for the down-projection column share (ff_count values of each output row).
inline size_t tp_span_bytes(size_t row_bytes, int64_t row_values, int64_t n) {
    if (row_values <= 0 || n <= 0 || n > row_values) return 0;
    // ggml rows are whole blocks; n must be a multiple of the block length the caller already checked.
    return (size_t) ((row_bytes * (int64_t) n) / row_values);
}

/// One expert's [gate | up | down] blob spans for this rank (offsets from the blob start).
struct TpBlobSpans {
    size_t gate_off = 0, gate_bytes = 0;
    size_t up_off = 0, up_bytes = 0;
    /// Down is strided: `down_row_bytes` stride, `down_span_bytes` per output row, `n_embd` rows starting at
    /// `down_off`.  A contiguous copy is not possible; callers copy row by row or keep a packed share.
    size_t down_off = 0, down_row_bytes = 0, down_span_bytes = 0;
    size_t n_embd = 0;
};

/// Native expert blob geometry (mirrors NativeFmt without pulling in ggml).
struct TpBlobGeom {
    int64_t n_embd = 0, n_ff = 0;
    size_t gu_row = 0, d_row = 0;   ///< bytes per gate/up row and per down row
    size_t up_off = 0, down_off = 0;
    size_t bytes = 0;
};

/// Spans for rank `p` of a native expert blob.  `geom` is the full blob (n_ff == p.ff).
inline bool tp_blob_spans(const TpBlobGeom& g, const TpPartition& p, TpBlobSpans& out, std::string& err) {
    if (p.ff != (int) g.n_ff) {
        err = "tp blob: partition ff " + std::to_string(p.ff) + " != blob n_ff " + std::to_string(g.n_ff);
        return false;
    }
    if (g.up_off != (size_t) g.n_ff * g.gu_row || g.down_off != 2 * g.up_off ||
        g.bytes != g.down_off + g.d_row * (size_t) g.n_embd) {
        err = "tp blob: geometry does not match [gate | up | down]";
        return false;
    }
    out.gate_off = (size_t) p.begin * g.gu_row;
    out.gate_bytes = (size_t) p.count * g.gu_row;
    out.up_off = g.up_off + out.gate_off;
    out.up_bytes = out.gate_bytes;
    out.down_off = g.down_off;
    out.down_row_bytes = g.d_row;
    out.down_span_bytes = tp_span_bytes(g.d_row, g.n_ff, p.count);
    out.n_embd = (size_t) g.n_embd;
    return true;
}

/// Block length (values) of the ggml types Maya-S uses for experts and dense FF.  0 = unknown (caller must supply).
/// IDs: Q8_0=8, Q2_K=10, Q6_K=14, IQ2_XXS=16, IQ2_XS=17, IQ3_XXS=18, IQ1_S=19, IQ3_S=21, IQ2_S=22, IQ4_XS=23.
inline int tp_ggml_block(int type) {
    switch (type) {
        case 8: return 32;    // Q8_0
        case 10: return 256;  // Q2_K
        case 14: return 256;  // Q6_K
        case 16: return 256;  // IQ2_XXS
        case 17: return 256;  // IQ2_XS
        case 18: return 256;  // IQ3_XXS
        case 19: return 256;  // IQ1_S
        case 21: return 256;  // IQ3_S
        case 22: return 256;  // IQ2_S
        case 23: return 256;  // IQ4_XS
        default: return 0;
    }
}

}  // namespace strata
