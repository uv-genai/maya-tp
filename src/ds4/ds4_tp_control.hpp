// DeepSeek TP2 control messages: bounded, versioned, endian-independent codec.
// This is control-plane metadata only; tensor data must use RDMA collectives.
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace strata::ds4::tp2 {

inline constexpr std::uint32_t kControlMagic = 0x44533432u; // DS42
inline constexpr std::uint32_t kControlVersion = 1;
inline constexpr std::uint32_t kControlMaxWords = 1u << 20;
inline constexpr std::size_t kControlHeaderBytes = 32;

enum class Command : std::uint32_t {
    Feed = 1, Step = 2, Verify = 3, Verdict = 4, Reset = 5, Shutdown = 6,
    Bind = 16, BeginSlot = 17, Rows = 18, BatchVerify = 19, BatchCommit = 20
};
inline bool batch_command(Command command) {
    return command>=Command::Bind && command<=Command::BatchCommit;
}

struct Control {
    Command command = Command::Reset;
    std::uint64_t sequence = 0; // nonzero scoped operation identity
    std::uint64_t position = 0;
    // Feed: input tokens; Step: one token; Verify: input token followed by
    // drafts; Verdict: accepted count then next token. Reset/Shutdown: empty.
    std::vector<std::uint32_t> words;
};

inline bool valid_control(const Control& c, std::uint32_t max_words, std::string& error) {
    if (c.sequence == 0 || max_words == 0 || max_words > kControlMaxWords || c.words.size() > max_words) {
        error = "ds4 tp2 control: invalid sequence or message length"; return false;
    }
    const auto n = c.words.size();
    switch (c.command) {
        case Command::Feed: if (n > 0) return true; break;
        case Command::Step: if (n == 1) return true; break;
        case Command::Verify: if (n >= 2 && n <= 16) return true; break;
        case Command::Verdict: if (n == 2 && c.words[0] <= 15) return true; break;
        case Command::Reset: case Command::Shutdown: if (n == 0) return true; break;
        case Command::Bind: case Command::BeginSlot: if (n == 3) return true; break;
        case Command::Rows: case Command::BatchVerify: if (n >= 10 && n <= 43) return true; break;
        case Command::BatchCommit: if (n >= 12 && n <= 48) return true; break;
        default: break;
    }
    error = "ds4 tp2 control: invalid command or payload"; return false;
}

inline void put32(std::vector<std::uint8_t>& b, std::uint32_t x) {
    for (int shift = 24; shift >= 0; shift -= 8) b.push_back(static_cast<std::uint8_t>(x >> shift));
}
inline void put64(std::vector<std::uint8_t>& b, std::uint64_t x) {
    put32(b, static_cast<std::uint32_t>(x >> 32)); put32(b, static_cast<std::uint32_t>(x));
}
inline std::uint32_t get32(std::span<const std::uint8_t> b, std::size_t off) {
    return (static_cast<std::uint32_t>(b[off]) << 24) | (static_cast<std::uint32_t>(b[off+1]) << 16) |
           (static_cast<std::uint32_t>(b[off+2]) << 8) | static_cast<std::uint32_t>(b[off+3]);
}
inline std::uint64_t get64(std::span<const std::uint8_t> b, std::size_t off) {
    return (static_cast<std::uint64_t>(get32(b, off)) << 32) | get32(b, off+4);
}

// Called on exactly the fixed header before a transport allocates or receives
// the body. A peer controls the length; never allocate from it unchecked.
inline bool control_frame_size(std::span<const std::uint8_t> header, std::uint32_t max_words,
                               std::size_t& frame_bytes, std::string& error) {
    if (header.size() != kControlHeaderBytes || get32(header, 0) != kControlMagic ||
        get32(header, 4) != kControlVersion) {
        error = "ds4 tp2 control: truncated header or incompatible protocol"; return false;
    }
    const auto words = get32(header, 12);
    if (max_words == 0 || max_words > kControlMaxWords || words > max_words) {
        error = "ds4 tp2 control: invalid frame length"; return false;
    }
    frame_bytes = kControlHeaderBytes + static_cast<std::size_t>(words) * 4;
    error.clear(); return true;
}

inline bool encode_control(const Control& c, std::uint32_t max_words, std::vector<std::uint8_t>& out,
                           std::string& error) {
    if (!valid_control(c, max_words, error)) return false;
    std::vector<std::uint8_t> b;
    b.reserve(kControlHeaderBytes + c.words.size() * 4);
    put32(b, kControlMagic); put32(b, kControlVersion); put32(b, static_cast<std::uint32_t>(c.command));
    put32(b, static_cast<std::uint32_t>(c.words.size())); put64(b, c.sequence); put64(b, c.position);
    for (auto word : c.words) put32(b, word);
    out = std::move(b); error.clear(); return true;
}

inline bool decode_control(std::span<const std::uint8_t> b, std::uint32_t max_words, Control& out,
                           std::string& error) {
    if (b.size() < kControlHeaderBytes) {
        error = "ds4 tp2 control: truncated header or incompatible protocol"; return false;
    }
    std::size_t expected_bytes = 0;
    if (!control_frame_size(b.first(kControlHeaderBytes), max_words, expected_bytes, error)) return false;
    const std::uint32_t n = get32(b, 12);
    if (b.size() != expected_bytes) {
        error = "ds4 tp2 control: invalid frame length"; return false;
    }
    Control c;
    c.command = static_cast<Command>(get32(b, 8)); c.sequence = get64(b, 16); c.position = get64(b, 24);
    c.words.reserve(n);
    for (std::uint32_t i = 0; i < n; ++i) c.words.push_back(get32(b, kControlHeaderBytes + std::size_t(i) * 4));
    if (!valid_control(c, max_words, error)) return false;
    out = std::move(c); error.clear(); return true;
}

}  // namespace strata::ds4::tp2
