// Bounded HIP pool for one rank's packed GGUF expert slices.
#pragma once

#include "ds4_tp_upload.cuh"
#include <algorithm>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace strata::ds4::tp2 {

// Initialization owns the device allocation and one reusable pinned staging
// buffer. A pool is usable only once every expert has uploaded successfully;
// after any failure it must be discarded. Its owning thread must serialize
// calls, and device pointers become invalid when the pool is destroyed.
class RankExpertPool {
public:
    RankExpertPool() = default;
    RankExpertPool(const RankExpertPool&) = delete;
    RankExpertPool& operator=(const RankExpertPool&) = delete;
    ~RankExpertPool() { release(); }

    bool allocate(int device, const std::vector<ExpertSpans>& spans, std::size_t budget,
                  std::string& error) {
        if (attempted_) return fail("ds4 tp2 expert pool: already initialized", error);
        attempted_ = true;
        if (device < 0 || spans.empty() || budget == 0)
            return fail("ds4 tp2 expert pool: invalid device, geometry or budget", error);
        std::size_t total=0, largest=0;
        std::vector<DeviceExpertLayout> layouts; layouts.reserve(spans.size());
        std::vector<std::size_t> offsets; offsets.reserve(spans.size());
        for (const auto& s:spans) {
            DeviceExpertLayout l;
            if (!device_expert_layout(s,l,error)) return fail(error,error);
            if (l.bytes > std::numeric_limits<std::size_t>::max()-total)
                return fail("ds4 tp2 expert pool: allocation size overflows",error);
            offsets.push_back(total); layouts.push_back(l);
            total+=l.bytes; largest=std::max(largest,l.bytes);
        }
        if (total>budget) return fail("ds4 tp2 expert pool: rank budget is too small",error);
        device_=device;
        hipError_t st=hipSetDevice(device_);
        if (st==hipSuccess) st=hipMalloc(&device_data_,total);
        if (st==hipSuccess) st=hipHostMalloc(&staging_,largest,hipHostMallocDefault);
        if (st==hipSuccess) st=hipStreamCreateWithFlags(&stream_,hipStreamNonBlocking);
        if (st!=hipSuccess) {
            const std::string cause=std::string("ds4 tp2 expert pool: ")+hipGetErrorString(st);
            release(); return fail(cause,error);
        }
        spans_=spans; layouts_=std::move(layouts); offsets_=std::move(offsets);
        loaded_.assign(spans.size(),false); bytes_=total; staging_bytes_=largest;
        error.clear(); return true;
    }

    bool upload(std::size_t index, const std::array<ConstBytes,3>& source, std::string& error) {
        if (!failure_.empty()) { error=failure_; return false; }
        if (device_data_==nullptr || index>=spans_.size() || loaded_[index])
            return fail("ds4 tp2 expert pool: invalid or repeated upload",error);
        const auto& s=spans_[index]; const auto& l=layouts_[index];
        std::memset(staging_,0,l.bytes);
        Bytes stage(static_cast<std::uint8_t*>(staging_),l.bytes);
        const std::size_t down=s.down_row_bytes*s.down_rows; // validated at allocation
        if (!pack_expert(s,source,{stage.subspan(l.offsets[0],s.gate_bytes),
                         stage.subspan(l.offsets[1],s.up_bytes),stage.subspan(l.offsets[2],down)},error))
            return fail(error,error);
        hipError_t st=hipSetDevice(device_);
        if (st==hipSuccess) st=hipMemcpyAsync(static_cast<std::uint8_t*>(device_data_)+offsets_[index],
                                               staging_,l.bytes,hipMemcpyHostToDevice,stream_);
        if (st==hipSuccess) st=hipStreamSynchronize(stream_);
        if (st!=hipSuccess) return fail(std::string("ds4 tp2 expert pool: ")+hipGetErrorString(st),error);
        loaded_[index]=true; ++loaded_count_; error.clear(); return true;
    }

    bool ready() const { return failure_.empty() && !spans_.empty() && loaded_count_==spans_.size(); }
    const std::uint8_t* data(std::size_t index) const {
        return ready() && index<offsets_.size() ? static_cast<const std::uint8_t*>(device_data_)+offsets_[index] : nullptr;
    }
    std::size_t bytes() const { return bytes_; }
    std::size_t staging_bytes() const { return staging_bytes_; }
    const std::string& failure() const { return failure_; }

private:
    bool fail(const std::string& cause, std::string& error) {
        if (failure_.empty()) failure_=cause.empty()?"ds4 tp2 expert pool: failed":cause;
        error=failure_; return false;
    }
    void release() noexcept {
        if (device_>=0) (void) hipSetDevice(device_);
        if (stream_!=nullptr) (void) hipStreamSynchronize(stream_);
        if (device_data_!=nullptr) (void) hipFree(device_data_);
        if (staging_!=nullptr) (void) hipHostFree(staging_);
        if (stream_!=nullptr) (void) hipStreamDestroy(stream_);
        device_data_=nullptr; staging_=nullptr; stream_=nullptr;
    }

    int device_=-1;
    bool attempted_=false;
    void* device_data_=nullptr;
    void* staging_=nullptr;
    hipStream_t stream_=nullptr;
    std::size_t bytes_=0, staging_bytes_=0, loaded_count_=0;
    std::vector<ExpertSpans> spans_;
    std::vector<DeviceExpertLayout> layouts_;
    std::vector<std::size_t> offsets_;
    std::vector<bool> loaded_;
    std::string failure_;
};

} // namespace strata::ds4::tp2
