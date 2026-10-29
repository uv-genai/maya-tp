// include/strata/core/tp_context.hpp - two-host tensor parallel state for the GLM engine.
//
// When `world == 1` every method is a no-op and the engine is bit-identical to a single-host run.  When
// `world == 2` each rank holds a slice of every split weight (TpPartition) and the two all-reduce points
// per block - after the mixer out and after the MoE/shared sum - complete the residual the same way on
// both ranks (gufo's two-operand sum: each rank adds local+peer, so both hold the same bits).
#pragma once

#include "strata/core/tp_partition.hpp"
#include "strata/net/communicator.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <memory>
#include <string>

namespace strata::core {

struct TpContext {
    int rank = 0;
    int world = 1;
    int block = 256;  ///< quant block length in intermediate units (IQ2_*/Q6_K)
    std::shared_ptr<net::Communicator> comm;
    TpPartition experts;   ///< routed + shared expert n_ff (2048 -> 1024)
    TpPartition dense;     ///< dense lead n_ff_dense (12288 -> 6144)

    bool active() const {
        // MAYA_TP_NOAR=1: keep the slices but skip the all-reduce (isolates slice vs AR bugs)
        return world > 1 && comm != nullptr && std::getenv("MAYA_TP_NOAR") == nullptr;
    }
    /// Whether weights are sliced to this rank's band (MAYA_TP_NOSLICE=1: full weights on both ranks).
    bool slicing() const { return world > 1 && std::getenv("MAYA_TP_NOSLICE") == nullptr; }

    /// Builds the partitions; fails when widths do not split.  `comm` may be null for unit tests.
    static bool create(int rank, int world, int n_ff, int n_ff_dense, int block,
                       std::shared_ptr<net::Communicator> comm, TpContext& out, std::string& err) {
        if (world < 1) {
            err = "tp: bad world";
            return false;
        }
        out.rank = rank;
        out.world = world;
        out.block = block;
        out.comm = std::move(comm);
        if (world == 1) {
            out.experts = {0, 1, n_ff, 0, n_ff};
            out.dense = {0, 1, n_ff_dense, 0, n_ff_dense};
            return true;
        }
        if (!TpPartition::create(n_ff, rank, world, block, out.experts, err)) return false;
        if (!TpPartition::create(n_ff_dense, rank, world, block, out.dense, err)) return false;
        return true;
    }

    /// Two-rank all-reduce of `n` floats in place (the row-parallel partial sums).  No-op when world==1.
    /// `hidden` is the residual width (n_embd); `n` is a multiple of it (rows x hidden).
    bool all_reduce(float* dev, std::size_t n, int hidden, hipStream_t stream, std::string& err) const;

    /// After a call's collectives: the stream must drain before the scope ends (gufo's rule).
    bool drain(std::string& err) const {
        if (world == 1 || comm == nullptr) return true;
        return comm->CheckQueued(&err);
    }
};

}  // namespace strata::core
