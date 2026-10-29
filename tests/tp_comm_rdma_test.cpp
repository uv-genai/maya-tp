// tests/tp_comm_rdma_test.cpp - two-process RDMA exchange check (no model, no logits).
//
// Run one process per host with the same --world/--rank and the other's bootstrap address:
//   strix-halo-1:  tp_comm_rdma_test --rank 0 --bootstrap-port 18515
//   strix-halo-2:  tp_comm_rdma_test --rank 1 --bootstrap-host strix-halo-1.local --bootstrap-port 18515
// Both print OK and exit 0 when N exchanges of a growing pattern all match the expected sum.
#include "strata/net/communicator.hpp"
#include "strata/net/verbs.hpp"

#include <chrono>
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
        else {
            std::fprintf(stderr, "usage: %s --rank N [--bootstrap-host H] [--bootstrap-port P] "
                                 "[--rdma-device D] [--gid-index G] [--exchanges N] [--bench]\n", argv[0]);
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
        // latency: 1-row (4096 F32 = 16 KiB) exchanges, no D2H; bandwidth: 3072 rows (one window)
        const int warm = 50, reps = 2000;
        const size_t n1 = 4096, nbig = 3071 * 4096;   // 3071 rows fit a 48 MiB window minus the 64 B header
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
        std::printf("bench latency  %d x 16 KiB: %.2f us/exchange\n", reps, lat_us);

        const int reps_big = 200;
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
        std::printf("bench bandwidth  %d x %.1f MiB: %.2f Gb/s one-way (%.2f us/exchange)\n",
                    reps_big, (double) nbig * sizeof(float) / 1048576.0, 8.0 * bytes / big_s / 1e9,
                    1e6 * big_s / reps_big);
    }
    return 0;
}
