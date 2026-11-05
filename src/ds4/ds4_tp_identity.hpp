// Identity checks for a DeepSeek two-host pair. This describes logical model
// and execution identity, not physical per-rank paths or rank-local allocations.
#pragma once

#include "ds4_tp_control.hpp"
#include "ds4_tp_batch.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <string_view>

namespace strata::ds4::tp2 {

struct Identity {
    std::uint32_t protocol = 0, world = 0, rank = 0;
    std::uint32_t hidden = 0, layers = 0, experts = 0, expert_width = 0;
    std::uint64_t max_context = 0, feature_flags = 0, max_partial_bytes = 0;
    // 256-bit digests supplied by the caller after validating model files and
    // canonical rank-independent partition description. Zero is never valid.
    std::array<std::uint8_t, 32> model_digest{}, partition_digest{}, build_digest{};
};

inline bool configure_batch_identity(Identity& id,std::uint32_t slots,std::string& error) {
    if (slots<1 || slots>kBatchSlots) { error="ds4 tp2: batch slots must be 1..4";return false; }
    id.protocol=slots==1?1:2;
    id.feature_flags&=~((255ull<<8)|(255ull<<48)|(255ull<<56));
    if (slots>1) id.feature_flags|=(static_cast<std::uint64_t>(kBatchVersion)<<8) |
        (static_cast<std::uint64_t>(slots)<<48) | (static_cast<std::uint64_t>(kBatchRows)<<56);
    error.clear();return true;
}

inline constexpr std::uint32_t kIdentityMagic = 0x44534944u; // DSID
inline constexpr std::uint32_t kIdentityVersion = 1;
inline constexpr std::size_t kIdentityBytes = 156;

inline std::vector<std::uint8_t> encode_identity(const Identity& id) {
    std::vector<std::uint8_t> b;
    b.reserve(kIdentityBytes);
    put32(b, kIdentityMagic); put32(b, kIdentityVersion);
    put32(b, id.protocol); put32(b, id.world); put32(b, id.rank); put32(b, id.hidden);
    put32(b, id.layers); put32(b, id.experts); put32(b, id.expert_width);
    put64(b, id.max_context); put64(b, id.feature_flags); put64(b, id.max_partial_bytes);
    for (const auto* d : {&id.model_digest, &id.partition_digest, &id.build_digest})
        b.insert(b.end(), d->begin(), d->end());
    return b;
}

inline bool decode_identity(std::span<const std::uint8_t> b, Identity& out, std::string& error) {
    if (b.size() != kIdentityBytes || get32(b, 0) != kIdentityMagic || get32(b, 4) != kIdentityVersion) {
        error = "ds4 tp2: invalid identity frame"; return false;
    }
    Identity id;
    id.protocol = get32(b, 8); id.world = get32(b, 12); id.rank = get32(b, 16);
    id.hidden = get32(b, 20); id.layers = get32(b, 24); id.experts = get32(b, 28);
    id.expert_width = get32(b, 32); id.max_context = get64(b, 36);
    id.feature_flags = get64(b, 44); id.max_partial_bytes = get64(b, 52);
    std::copy_n(b.begin() + 60, 32, id.model_digest.begin());
    std::copy_n(b.begin() + 92, 32, id.partition_digest.begin());
    std::copy_n(b.begin() + 124, 32, id.build_digest.begin());
    out = id; error.clear(); return true;
}

inline bool nonzero(const std::array<std::uint8_t, 32>& digest) {
    return std::any_of(digest.begin(), digest.end(), [](auto b) { return b != 0; });
}

// Parse the lowercase SHA-256 emitted by the preflight helper. Do not accept
// partial values or change the caller's digest on malformed input.
inline bool parse_digest(std::string_view text, std::array<std::uint8_t, 32>& out, std::string& error) {
    const auto nibble = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        return -1;
    };
    if (text.size() != 64) { error = "ds4 tp2: invalid SHA-256 identity"; return false; }
    std::array<std::uint8_t, 32> parsed{};
    for (std::size_t i = 0; i < parsed.size(); ++i) {
        const int hi = nibble(text[i * 2]), lo = nibble(text[i * 2 + 1]);
        if (hi < 0 || lo < 0) { error = "ds4 tp2: invalid SHA-256 identity"; return false; }
        parsed[i] = static_cast<std::uint8_t>((hi << 4) | lo);
    }
    if (!nonzero(parsed)) { error = "ds4 tp2: zero SHA-256 identity"; return false; }
    out = parsed; error.clear(); return true;
}

// Both peers must compute this before model collectives. Return the smaller
// RDMA payload limit; the caller must agree a row/chunk size within that limit.
// Exchange with encode_identity/decode_identity; never send this in-memory
// struct as raw bytes.
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
