// DeepSeek rank-local GGUF dense-matrix loading; no model execution.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace strata::ds4::tp2 {

enum class DenseAxis { Whole, Output, Input };
enum class DenseEncoding { Bf16=1, Q8_0=3, Ggml=4 }; // engine x3::Mat format values

struct RankDenseEntry {
    int n=0, k=0, row_begin=0, column_begin=0;
    int full_n=0, full_k=0, gg_type=0;
    DenseEncoding encoding=DenseEncoding::Bf16;
    std::size_t row_bytes=0, bytes=0;
    const std::uint8_t* device=nullptr; // valid for the loader's lifetime
};

// Owns one loaded matrix, its pack mappings and device allocation. Whole is
// for replicated matrices or rank-owned output groups; Output splits output
// rows, Input splits each row in complete quantization blocks.
class RankDenseLoader {
public:
    RankDenseLoader();
    ~RankDenseLoader();
    RankDenseLoader(const RankDenseLoader&)=delete;
    RankDenseLoader& operator=(const RankDenseLoader&)=delete;
    bool load(const std::string& pack, const std::string& tensor, int device, int rank,
              DenseAxis axis, std::size_t budget, std::string& error);
    const RankDenseEntry* entry() const;
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

} // namespace strata::ds4::tp2
