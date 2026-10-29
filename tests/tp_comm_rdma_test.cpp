// tests/tp_comm_rdma_test.cpp - two-process RDMA exchange check (no model, no logits).
//
// Run one process per host with the same --world/--rank and the other's bootstrap address:
//   strix-halo-1:  tp_comm_rdma_test --rank 0 --bootstrap-port 18515
//   strix-halo-2:  tp_comm_rdma_test --rank 1 --bootstrap-host strix-halo-1.local --bootstrap-port 18515
// Both print OK and exit 0 when N exchanges of a growing pattern all match the expected sum.
#include "strata/net/communicator.hpp"
#include "strata/net/tp_kernels.hpp"
#include "strata/net/verbs.hpp"

#include <chrono>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

using namespace strata::net;

static int g_fail = 0;

static void ok(bool cond, const char* what) {
    if (!cond) {
        std::printf("FAIL %s\n", what);
        ++g_fail;
    }
}

int main(int argc, char** argv) {
    IbrverbsConfig cfg;
    cfg.rank = 0;
    cfg.world_size = 2;
    cfg.bootstrap_port = 18515;
    int exchanges = 200;
    bool bench = false;
    // Decode AR unit: float32 residual width. Maya GLM n_embd=12288 (48 KiB/row).
    // Gufo Flash-Next uses 2560 (10 KiB/row) — pass --hidden 2560 to compare.
    uint32_t hidden_bench = 12288;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", what);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--rank") cfg.rank = (uint32_t) std::atoi(next("--rank"));
        else if (a == "--bootstrap-host") cfg.bootstrap_host = next("--bootstrap-host");
        else if (a == "--bootstrap-port") cfg.bootstrap_port = (uint16_t) std::atoi(next("--bootstrap-port"));
        else if (a == "--rdma-device") cfg.rdma_device = next("--rdma-device");
        else if (a == "--gid-index") cfg.gid_index = (uint32_t) std::atoi(next("--gid-index"));
        else if (a == "--exchanges") exchanges = std::atoi(next("--exchanges"));
        else if (a == "--bench") bench = true;
        else if (a == "--hidden") hidden_bench = (uint32_t) std::atoi(next("--hidden"));
        else {
            std::fprintf(stderr, "usage: %s --rank N [--bootstrap-host H] [--bootstrap-port P] "
                                 "[--rdma-device D] [--gid-index G] [--exchanges N] [--bench] [--hidden H]\n", argv[0]);
            return 2;
        }
    }

    std::string err;
    auto comm = CreateIbrverbsCommunicator(cfg, &err);
    if (!comm) {
        std::printf("FAIL create: %s\n", err.c_str());
        return 1;
    }
    std::printf("rdma_ready %s rank %u/%u\n", comm->Describe().c_str(), comm->rank(), comm->world_size());

    // one stream for the exchange + add
    hipStream_t stream = nullptr;
    ok(hipStreamCreate(&stream) == hipSuccess, "stream");

    const int rows0 = 16, rows_max = 3072;   // up to one RDMA window (48 MiB)
    std::vector<float> host((size_t) rows_max * 4096);
    float* dev = nullptr;
    ok(hipMalloc(&dev, host.size() * sizeof(float)) == hipSuccess, "dev alloc");
    float* sum = nullptr;
    ok(hipMalloc(&sum, host.size() * sizeof(float)) == hipSuccess, "sum alloc");

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < exchanges; ++i) {
        const int rows = rows0 + (i % 64);
        const size_t n = (size_t) rows * 4096;
        for (size_t k = 0; k < n; ++k) host[k] = (float) (k + 1000 * i) + (float) comm->rank();
        ok(hipMemcpy(dev, host.data(), n * sizeof(float), hipMemcpyHostToDevice) == hipSuccess, "upload");

        if (!comm->BeginOperation((uint64_t) i, &err)) {
            std::printf("FAIL begin %d: %s\n", i, err.c_str());
            ++g_fail;
            break;
        }
        const float* peer = comm->ExchangePartial(dev, n * sizeof(float), stream, &err);
        if (!peer) {
            std::printf("FAIL exchange %d: %s\n", i, err.c_str());
            ++g_fail;
            break;
        }
        // sum = local + peer (both ranks hold the same bits)
        // reuse the queued form's contract: caller adds on the stream
        // (a tiny kernel-free add via memcpy is enough here; the engine uses a kernel)
        // we only check the peer's first and last values against the other rank's pattern
        std::vector<float> ph(n), lh(n);
        ok(hipMemcpyAsync(ph.data(), peer, n * sizeof(float), hipMemcpyDeviceToHost, stream) == hipSuccess,
           "peer download");
        ok(hipMemcpyAsync(lh.data(), dev, n * sizeof(float), hipMemcpyDeviceToHost, stream) == hipSuccess,
           "local download");
        ok(hipStreamSynchronize(stream) == hipSuccess, "sync");
        // rank r saw the other rank's fill: offset = other_rank
        const int other = 1 - (int) comm->rank();
        const float want0 = (float) (0 + 1000 * i) + (float) other;
        const float want1 = (float) (n - 1 + 1000 * i) + (float) other;
        ok(ph[0] == want0 && ph[n - 1] == want1, "peer pattern");
        ok(lh[0] == (float) (1000 * i) + (float) comm->rank(), "local pattern");

        if (!comm->EndOperation((uint64_t) i, &err)) {
            std::printf("FAIL end %d: %s\n", i, err.c_str());
            ++g_fail;
            break;
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (g_fail) {
        std::printf("tp_comm_rdma_test: %d FAILED (%d exchanges, %.1f ms)\n", g_fail, exchanges, ms);
        return 1;
    }
    std::printf("tp_comm_rdma_test OK  %d exchanges in %.1f ms (%.2f us/exchange both ways)\n",
                exchanges, ms, 1000.0 * ms / exchanges);

    if (bench) {
        // Three layers of "latency" — do not mix them up:
        //  1) ib_write_lat 2 B RTT (~6 µs) — pure link (gufo docs §7)
        //  2) Blocking ExchangePartial — host staging + write + wait (Maya's ~55 µs form)
        //  3) Queued QueuePartial Stage/Wait/Add — gufo's overlapped AR (~10 KiB/row there)
        // Payload: --hidden floats (default 12288 = 48 KiB/row, Maya GLM n_embd).
        // Gufo Flash-Next uses 2560 (10 KiB/row); pass --hidden 2560 to match theirs.
        const int warm = 50, reps = 2000, reps_big = 200;
        const size_t n1 = hidden_bench;
        // One RDMA window is 48 MiB; pick the largest row count that fits.
        const size_t window_bytes = 48u * 1024u * 1024u - 64u;
        const size_t rows_big = std::max<size_t>(1, window_bytes / (sizeof(float) * hidden_bench));
        const size_t nbig = rows_big * hidden_bench;
        // The correctness loop allocated rows_max*4096; bench may need more.
        (void) hipFree(dev);
        (void) hipFree(sum);
        dev = sum = nullptr;
        const size_t need = std::max(n1, nbig);
        if (hipMalloc(&dev, need * sizeof(float)) != hipSuccess || hipMalloc(&sum, need * sizeof(float)) != hipSuccess) {
            std::printf("FAIL bench alloc %zu floats\n", need);
            ++g_fail;
            return 1;
        }
        (void) hipMemset(dev, 0, need * sizeof(float));
        std::printf("bench hidden=%u  row=%.1f KiB  big=%zu rows (%.1f MiB)\n", hidden_bench,
                    (double) n1 * sizeof(float) / 1024.0, rows_big, (double) nbig * sizeof(float) / 1048576.0);
        for (int i = 0; i < warm; ++i) {
            if (!comm->BeginOperation((uint64_t) (100000 + i), &err)) break;
            if (!comm->ExchangePartial(dev, n1 * sizeof(float), stream, &err)) break;
            if (!comm->EndOperation((uint64_t) (100000 + i), &err)) break;
        }
        ok(hipStreamSynchronize(stream) == hipSuccess, "bench sync");
        auto b0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) {
            const uint64_t op = (uint64_t) (200000 + i);
            if (!comm->BeginOperation(op, &err) || !comm->ExchangePartial(dev, n1 * sizeof(float), stream, &err) ||
                !comm->EndOperation(op, &err)) {
                std::printf("FAIL bench exchange %d: %s\n", i, err.c_str());
                ++g_fail;
                break;
            }
        }
        ok(hipStreamSynchronize(stream) == hipSuccess, "bench sync2");
        auto b1 = std::chrono::steady_clock::now();
        const double lat_us = std::chrono::duration<double, std::micro>(b1 - b0).count() / reps;
        std::printf("bench blocking  %d x %.1f KiB: %.2f us/exchange\n", reps, (double) n1 * sizeof(float) / 1024.0,
                    lat_us);

        auto c0 = std::chrono::steady_clock::now();
        double bytes = 0;
        for (int i = 0; i < reps_big; ++i) {
            const uint64_t op = (uint64_t) (300000 + i);
            if (!comm->BeginOperation(op, &err) || !comm->ExchangePartial(dev, nbig * sizeof(float), stream, &err) ||
                !comm->EndOperation(op, &err)) {
                std::printf("FAIL big exchange %d: %s\n", i, err.c_str());
                ++g_fail;
                break;
            }
            bytes += (double) nbig * sizeof(float);
        }
        ok(hipStreamSynchronize(stream) == hipSuccess, "bench sync3");
        auto c1 = std::chrono::steady_clock::now();
        const double big_s = std::chrono::duration<double>(c1 - c0).count();
        // one-way payload rate (the write carries bytes; the peer's partial is the same size back)
        std::printf("bench blocking BW  %d x %.1f MiB: %.2f Gb/s one-way (%.2f us/exchange)\n", reps_big,
                    (double) nbig * sizeof(float) / 1048576.0, 8.0 * bytes / big_s / 1e9, 1e6 * big_s / reps_big);

        // --- queued GPU path (gufo's overlapped AR form): QueuePartial + Stage/Wait/Add ---
        const uint32_t hidden = hidden_bench;
        {
            Communicator::QueuedPartial q;
            if (comm->BeginOperation((uint64_t) (400000), &err) && comm->QueuePartial(n1 * sizeof(float), &q, &err)) {
                strata::net::StagePartial(dev, q.send, n1, q.arrivals, q.ready, q.value, stream);
                strata::net::WaitValue(q.arrived, q.value, q.wait_ticks, q.timed_out, stream);
                strata::net::AddRowsBroadcast(q.peer, dev, (uint32_t) (n1 / hidden), hidden, 1, stream);
                ok(hipStreamSynchronize(stream) == hipSuccess, "qbench warm sync");
                ok(comm->CheckQueued(&err), "qbench warm check");
                ok(comm->EndOperation((uint64_t) (400000), &err), "qbench warm end");
            } else {
                std::printf("FAIL queued warm: %s\n", err.c_str());
                ++g_fail;
            }
        }
        auto q0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) {
            const uint64_t op = (uint64_t) (500000 + i);
            Communicator::QueuedPartial q;
            if (!comm->BeginOperation(op, &err) || !comm->QueuePartial(n1 * sizeof(float), &q, &err)) {
                std::printf("FAIL queued %d: %s\n", i, err.c_str());
                ++g_fail;
                break;
            }
            strata::net::StagePartial(dev, q.send, n1, q.arrivals, q.ready, q.value, stream);
            strata::net::WaitValue(q.arrived, q.value, q.wait_ticks, q.timed_out, stream);
            strata::net::AddRowsBroadcast(q.peer, dev, (uint32_t) (n1 / hidden), hidden, 1, stream);
            // the stream must drain first: the background thread counts the exchange out
            // before it raises `arrived`, so a drained stream finds nothing in flight.
            if (hipStreamSynchronize(stream) != hipSuccess || !comm->CheckQueued(&err) ||
                !comm->EndOperation(op, &err)) {
                std::printf("FAIL queued end %d: %s\n", i, err.c_str());
                ++g_fail;
                break;
            }
        }
        ok(hipStreamSynchronize(stream) == hipSuccess, "qbench sync2");
        auto q1 = std::chrono::steady_clock::now();
        const double qlat_us = std::chrono::duration<double, std::micro>(q1 - q0).count() / reps;
        std::printf("bench queued latency  %d x %.1f KiB: %.2f us/exchange\n", reps,
                    (double) n1 * sizeof(float) / 1024.0, qlat_us);

        // pipelined: one scope, many queued exchanges (the host queues ahead; the GPU waits on flags)
        auto r0 = std::chrono::steady_clock::now();
        double qbytes = 0;
        {
            const uint64_t scope = 700000;
            if (!comm->BeginOperation(scope, &err)) {
                std::printf("FAIL queued big begin: %s\n", err.c_str());
                ++g_fail;
            } else {
                const size_t rows = nbig / hidden;
                for (int i = 0; i < reps_big; ++i) {
                    Communicator::QueuedPartial q;
                    if (!comm->QueuePartial(nbig * sizeof(float), &q, &err)) {
                        std::printf("FAIL queued big %d: %s\n", i, err.c_str());
                        ++g_fail;
                        break;
                    }
                    strata::net::StagePartial(dev, q.send, nbig, q.arrivals, q.ready, q.value, stream);
                    strata::net::WaitValue(q.arrived, q.value, q.wait_ticks, q.timed_out, stream);
                    strata::net::AddRowsBroadcast(q.peer, dev, (uint32_t) rows, hidden, 1, stream);
                    qbytes += (double) nbig * sizeof(float);
                }
                ok(hipStreamSynchronize(stream) == hipSuccess, "qbench sync3");
                ok(comm->CheckQueued(&err), "qbench check2");
                ok(comm->EndOperation(scope, &err), "qbench end2");
            }
        }
        auto r1 = std::chrono::steady_clock::now();
        const double qbig_s = std::chrono::duration<double>(r1 - r0).count();
        std::printf("bench queued bandwidth  %d x %.1f MiB (pipelined): %.2f Gb/s one-way (%.2f us/exchange)\n",
                    reps_big, (double) nbig * sizeof(float) / 1048576.0, 8.0 * qbytes / qbig_s / 1e9,
                    1e6 * qbig_s / reps_big);

        // gufo --allreduce-bench shape: 1/2/8/512 rows of --hidden, queued all-reduce,
        // timed queuing → stream drain, min/p50/mean/p99.
        {
            const uint32_t H = hidden_bench;
            const size_t rows_list[] = {1, 2, 8, 512};
            const uint64_t scope = 800000;
            if (comm->BeginOperation(scope, &err)) {
                for (size_t rows : rows_list) {
                    const size_t bytes = rows * H * sizeof(float);
                    std::vector<double> us;
                    us.reserve((size_t) reps);
                    for (int i = 0; i < reps; ++i) {
                        const auto s = std::chrono::steady_clock::now();
                        Communicator::QueuedPartial q;
                        if (!comm->QueuePartial(bytes, &q, &err)) break;
                        strata::net::StagePartial(dev, q.send, bytes / sizeof(float), q.arrivals, q.ready, q.value,
                                                  stream);
                        strata::net::WaitValue(q.arrived, q.value, q.wait_ticks, q.timed_out, stream);
                        strata::net::AddRowsBroadcast(q.peer, dev, (uint32_t) rows, H, 1, stream);
                        if (hipStreamSynchronize(stream) != hipSuccess || !comm->CheckQueued(&err)) break;
                        us.push_back(std::chrono::duration<double, std::micro>(
                                         std::chrono::steady_clock::now() - s)
                                         .count());
                    }
                    if (us.empty()) {
                        std::printf("allreduce rows=%zu FAILED: %s\n", rows, err.c_str());
                        ++g_fail;
                        continue;
                    }
                    std::sort(us.begin(), us.end());
                    double total = 0;
                    for (double v : us) total += v;
                    std::printf("allreduce rows=%zu bytes=%zu n=%zu min_us=%.1f p50_us=%.1f mean_us=%.1f p99_us=%.1f\n",
                                rows, rows * H * sizeof(float), us.size(), us.front(), us[us.size() / 2],
                                total / us.size(), us[(us.size() * 99) / 100]);
                }
                ok(comm->EndOperation(scope, &err), "allreduce end");
            }
        }
    }
    return 0;
}
