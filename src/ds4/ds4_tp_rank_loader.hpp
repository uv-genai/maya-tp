// Real-pack DeepSeek rank-local expert loading. Does not execute a model.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace strata::ds4::tp2 {

struct RankExpertEntry {
    int layer=0, expert=0, ff_begin=0, ff_count=0;
    std::array<std::size_t,3> offsets{};
    std::array<int,3> gg_types{}; // gate, up, down ggml type IDs
    std::size_t bytes=0;
    const std::uint8_t* device=nullptr; // valid only for the loader's lifetime
};

// The loader owns its pack mappings and GPU allocation. It exposes no entries
// until every selected expert is loaded. max_experts=0 selects every trunk
// expert; a nonzero bound is for staged validation. This class provides no
// attention/shared weights, model execution, network or CPU fallback.
class RankExpertLoader {
public:
    RankExpertLoader();
    ~RankExpertLoader();
    RankExpertLoader(const RankExpertLoader&)=delete;
    RankExpertLoader& operator=(const RankExpertLoader&)=delete;
    bool load(const std::string& pack, int device, int rank, std::size_t budget,
              std::size_t max_experts, std::string& error,
              std::function<void(std::size_t,std::size_t)> progress = {});
    std::size_t size() const;
    std::size_t bytes() const;
    std::size_t staging_bytes() const;
    const RankExpertEntry* entry(std::size_t index) const;
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

} // namespace strata::ds4::tp2
