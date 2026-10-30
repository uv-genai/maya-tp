// src/core/tp_context.cu - the GLM engine's two-rank all-reduce (queued GPU path).
#include "strata/core/tp_context.hpp"
#include "strata/net/tp_kernels.hpp"

namespace strata::core {

bool TpContext::agree_min(int& value, float* scratch, hipStream_t stream, std::string& err) const {
    if (!active()) return true;
    if (world != 2 || rank < 0 || rank > 1 || value < 0 || value > (1 << 24) || scratch == nullptr) {
        err = "tp decision: needs two ranks, device scratch and a nonnegative exact integer";
        return false;
    }
    float votes[2] = {};
    votes[rank] = (float) value;
    hipError_t status = hipMemcpyAsync(scratch, votes, sizeof votes, hipMemcpyHostToDevice, stream);
    if (status == hipSuccess && !all_reduce(scratch, 2, 2, stream, err)) {
        (void) hipStreamSynchronize(stream);   // the host vote must outlive its queued H2D copy
        return false;
    }
    if (status == hipSuccess)
        status = hipMemcpyAsync(votes, scratch, sizeof votes, hipMemcpyDeviceToHost, stream);
    if (status == hipSuccess) status = hipStreamSynchronize(stream);
    if (status != hipSuccess) {
        err = std::string("tp decision: ") + hipGetErrorString(status);
        return false;
    }
    if (!drain(err)) return false;
    value = (int) (votes[0] < votes[1] ? votes[0] : votes[1]);
    return true;
}

bool TpContext::all_reduce(float* dev, std::size_t n, int hidden, hipStream_t stream, std::string& err) const {
    if (world == 1 || comm == nullptr) return true;
    if (n == 0) return true;
    if (hidden <= 0 || n % (std::size_t) hidden != 0) {
        err = "tp all-reduce: n is not a multiple of hidden";
        return false;
    }
    const std::size_t bytes = n * sizeof(float);
    net::Communicator::QueuedPartial q;
    if (!comm->QueuePartial(bytes, &q, &err)) return false;
    net::StagePartial(dev, q.send, n, q.arrivals, q.ready, q.value, stream);
    net::WaitValue(q.arrived, q.value, q.wait_ticks, q.timed_out, stream);
    net::AddRowsBroadcast(q.peer, dev, (uint32_t) (n / (std::size_t) hidden), (uint32_t) hidden, 1, stream);
    return true;
}

bool TpContext::start_reduce(float* dev, std::size_t n, int hidden, hipStream_t stream, std::string& err) const {
    if (world == 1 || comm == nullptr) return true;
    if (n == 0) return true;
    if (hidden <= 0 || n % (std::size_t) hidden != 0) {
        err = "tp start-reduce: n is not a multiple of hidden";
        return false;
    }
    return comm->StartPartial(dev, n * sizeof(float), stream, &err);
}

bool TpContext::finish_reduce(float* dev, std::size_t n, int hidden, hipStream_t stream, std::string& err) const {
    if (world == 1 || comm == nullptr) return true;
    if (n == 0) return true;
    if (hidden <= 0 || n % (std::size_t) hidden != 0) {
        err = "tp finish-reduce: n is not a multiple of hidden";
        return false;
    }
    const float* peer = comm->FinishPartial(n * sizeof(float), &err);
    if (peer == nullptr) return false;
    net::AddRowsBroadcast(peer, dev, (uint32_t) (n / (std::size_t) hidden), (uint32_t) hidden, 1, stream);
    return true;
}

}  // namespace strata::core
