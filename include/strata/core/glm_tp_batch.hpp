#pragma once

// GLM application frames. The shared GUFT/GUCH tensor transport is independent.
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::core::glm_tp {

constexpr int32_t kVersion = 1;
constexpr int kMaxSlots = 4;
constexpr int kMaxRows = 64;
constexpr size_t kMaxWords = 4 + kMaxSlots * 6 + kMaxRows;

struct Limits {
    int slots = 1;
    int rows = 1;
    int vocab = 0;
    uint64_t context = 0;
};

struct Member {
    int slot = 0;
    uint64_t generation = 0;
    uint64_t position = 0;
    std::vector<int32_t> tokens;
};

struct Batch {
    uint64_t operation = 0;
    std::vector<Member> members;
};

inline int legacy_rows(size_t words, int slots) {
    if (slots < 1 || slots > kMaxSlots || words == 0 || words % 2 != 0 || words / 2 > (size_t) slots)
        return 0;
    return (int) (words / 2);
}

inline bool validate(const Batch& batch, const Limits& limits, uint64_t expected_operation, std::string& error) {
    if (limits.slots < 1 || limits.slots > kMaxSlots || limits.rows < 1 || limits.rows > kMaxRows ||
        limits.vocab < 1 || limits.context < 2 || limits.context > (uint64_t) INT64_MAX) {
        error = "GLM TP2 batch has invalid negotiated limits";
        return false;
    }
    if (batch.operation == 0 || batch.operation != expected_operation) {
        error = "GLM TP2 batch has a stale or skipped operation";
        return false;
    }
    if (batch.members.empty() || batch.members.size() > (size_t) limits.slots) {
        error = "GLM TP2 batch has an invalid member count";
        return false;
    }
    std::array<bool, kMaxSlots> seen{};
    size_t rows = 0;
    for (const Member& member : batch.members) {
        if (member.slot < 0 || member.slot >= limits.slots || seen[(size_t) member.slot] || member.generation == 0) {
            error = "GLM TP2 batch has an invalid or duplicate slot identity";
            return false;
        }
        seen[(size_t) member.slot] = true;
        if (member.tokens.empty() || member.tokens.size() > (size_t) limits.rows - rows ||
            member.position >= limits.context || member.tokens.size() >= limits.context - member.position) {
            error = "GLM TP2 batch exceeds its row or context budget";
            return false;
        }
        rows += member.tokens.size();
        for (int32_t token : member.tokens) {
            if (token < 0 || token >= limits.vocab) {
                error = "GLM TP2 batch token is outside the vocabulary";
                return false;
            }
        }
    }
    return true;
}

inline void append_u64(std::vector<int32_t>& words, uint64_t value) {
    words.push_back(std::bit_cast<int32_t>((uint32_t) value));
    words.push_back(std::bit_cast<int32_t>((uint32_t) (value >> 32)));
}

inline uint64_t read_u64(const int32_t* words) {
    return (uint64_t) std::bit_cast<uint32_t>(words[0]) |
           ((uint64_t) std::bit_cast<uint32_t>(words[1]) << 32);
}

inline bool encode(const Batch& batch, const Limits& limits, std::vector<int32_t>& words, std::string& error) {
    words.clear();
    if (!validate(batch, limits, batch.operation, error)) return false;
    words.reserve(kMaxWords);
    words.push_back(kVersion);
    append_u64(words, batch.operation);
    words.push_back((int32_t) batch.members.size());
    for (const Member& member : batch.members) {
        words.push_back(member.slot);
        append_u64(words, member.generation);
        append_u64(words, member.position);
        words.push_back((int32_t) member.tokens.size());
        words.insert(words.end(), member.tokens.begin(), member.tokens.end());
    }
    return true;
}

inline bool decode(const std::vector<int32_t>& words, const Limits& limits, uint64_t expected_operation,
                   Batch& output, std::string& error) {
    output = {};
    if (words.size() < 4 || words.size() > kMaxWords || words[0] != kVersion ||
        words[3] < 1 || words[3] > kMaxSlots) {
        error = "GLM TP2 batch has an invalid header or version";
        return false;
    }
    Batch batch;
    batch.operation = read_u64(words.data() + 1);
    size_t at = 4;
    for (int i = 0; i < words[3]; ++i) {
        if (words.size() - at < 6) {
            error = "GLM TP2 batch has a truncated member";
            return false;
        }
        Member member;
        member.slot = words[at];
        member.generation = read_u64(words.data() + at + 1);
        member.position = read_u64(words.data() + at + 3);
        const int32_t count = words[at + 5];
        at += 6;
        if (count < 1 || (size_t) count > words.size() - at) {
            error = "GLM TP2 batch has a truncated token window";
            return false;
        }
        member.tokens.assign(words.begin() + (ptrdiff_t) at, words.begin() + (ptrdiff_t) (at + count));
        at += (size_t) count;
        batch.members.push_back(std::move(member));
    }
    if (at != words.size()) {
        error = "GLM TP2 batch has trailing words";
        return false;
    }
    if (!validate(batch, limits, expected_operation, error)) return false;
    output = std::move(batch);
    return true;
}

}  // namespace strata::core::glm_tp
