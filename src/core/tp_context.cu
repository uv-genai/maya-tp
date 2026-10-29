// src/core/tp_context.cu - the GLM engine's two-rank all-reduce (queued GPU path).
#include "strata/core/tp_context.hpp"
#include "strata/net/tp_kernels.hpp"

namespace strata::core {

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

}  // namespace strata::core
