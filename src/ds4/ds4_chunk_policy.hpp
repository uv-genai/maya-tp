// Host policy for DeepSeek's optional gfx1151 automatic prompt-chunk cap.
#pragma once
#include <charconv>
#include <cstring>
#include <system_error>

namespace strata::ds4::chunk_policy {

// MAYA wins even when it disables the cap; STRATA is the config alias, DS4 the legacy shell name.
// Default/invalid/sub16/overflow values preserve the current plan until locally measured (#79).
inline int configured_cap(const char* maya, const char* strata, const char* legacy) {
    const char* value = maya ? maya : strata ? strata : legacy;
    if (value == nullptr) return 0;
    const char* end = value + std::strlen(value);
    int cap = 0;
    const auto parsed = std::from_chars(value, end, cap);
    return parsed.ec == std::errc{} && parsed.ptr == end && cap >= 16 ? cap : 0;
}

inline int capped_rows(int candidate, int cap) {
    if (cap < 16 || candidate <= cap) return candidate;
    return cap / 16 * 16;
}

inline bool gfx1151_arch(const char* arch) {
    return arch != nullptr && std::strncmp(arch, "gfx1151", 7) == 0 && (arch[7] == '\0' || arch[7] == ':');
}

}  // namespace strata::ds4::chunk_policy
