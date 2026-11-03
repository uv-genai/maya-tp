// Identity checks for a DeepSeek two-host pair. This describes logical model
// and execution identity, not physical per-rank paths or rank-local allocations.
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>

namespace strata::ds4::tp2 {

struct Identity {
    std::uint32_t protocol = 0, world = 0, rank = 0;
    std::uint32_t hidden = 0, layers = 0, experts = 0, expert_width = 0;
    std::uint64_t max_context = 0, feature_flags = 0, max_partial_bytes = 0;
    // 256-bit digests supplied by the caller after validating model files and
    // canonical rank-independent partition description. Zero is never valid.
    std::array<std::uint8_t, 32> model_digest{}, partition_digest{}, build_digest{};
};

inline bool nonzero(const std::array<std::uint8_t, 32>& digest) {
    return std::any_of(digest.begin(), digest.end(), [](auto b) { return b != 0; });
}

// Both peers must compute this before model collectives. Return the smaller
// RDMA payload limit; the caller must agree a row/chunk size within that limit.
// Identity must be exchanged via a separately versioned, bounded wire codec;
// this in-memory struct must never be sent as raw bytes.
inline bool agree_identity(const Identity& local, const Identity& peer, std::uint64_t& max_partial,
                           std::string& error) {
    const auto fail = [&](const char* message) { error = message; return false; };
    if (local.world != 2 || peer.world != 2 || local.rank > 1 || peer.rank > 1 || local.rank == peer.rank)
        return fail("ds4 tp2: expected complementary ranks in a two-host pair");
    if (local.protocol == 0 || local.protocol != peer.protocol)
        return fail("ds4 tp2: protocol version mismatch");
    if (!nonzero(local.model_digest) || !nonzero(peer.model_digest) || local.model_digest != peer.model_digest)
        return fail("ds4 tp2: missing or mismatched model identity");
    if (!nonzero(local.partition_digest) || !nonzero(peer.partition_digest) ||
        local.partition_digest != peer.partition_digest)
        return fail("ds4 tp2: missing or mismatched partition identity");
    if (!nonzero(local.build_digest) || !nonzero(peer.build_digest) || local.build_digest != peer.build_digest)
        return fail("ds4 tp2: missing or mismatched build identity");
    if (local.hidden == 0 || local.layers == 0 || local.experts == 0 || local.expert_width == 0 ||
        local.hidden != peer.hidden || local.layers != peer.layers || local.experts != peer.experts ||
        local.expert_width != peer.expert_width)
        return fail("ds4 tp2: model geometry mismatch");
    if (local.max_context == 0 || local.max_context != peer.max_context || local.feature_flags != peer.feature_flags)
        return fail("ds4 tp2: context or requested features mismatch");
    if (local.max_partial_bytes == 0 || peer.max_partial_bytes == 0)
        return fail("ds4 tp2: invalid collective receive window");
    max_partial = std::min(local.max_partial_bytes, peer.max_partial_bytes);
    error.clear();
    return true;
}

}  // namespace strata::ds4::tp2
