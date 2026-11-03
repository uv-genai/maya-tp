// Two-host integration: control data arriving while RDMA waits is not EOF.
#include "../src/ds4/ds4_tp_channel.hpp"
#include "strata/net/verbs.hpp"
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

using namespace strata::ds4::tp2;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "line %d: %s; %s\n", __LINE__, #x, error.c_str()); return 1; } } while (0)
int main(int argc, char** argv) {
    std::string error;
    if (argc != 5 && argc != 6) { std::fprintf(stderr, "usage: rank bootstrap-host port rdma-device [peer-loss]\n"); return 2; }
    const bool peer_loss = argc == 6 && std::string(argv[5]) == "peer-loss";
    if (argc == 6 && !peer_loss) return 2;
    strata::net::IbrverbsConfig cfg;
    cfg.rank = static_cast<std::uint32_t>(std::atoi(argv[1]));
    cfg.bootstrap_host = argv[2]; cfg.bootstrap_port = static_cast<std::uint16_t>(std::atoi(argv[3]));
    cfg.rdma_device = argv[4];
    auto comm = strata::net::CreateIbrverbsCommunicator(cfg, &error); CHECK(comm);
    Identity me; me.protocol=1; me.world=2; me.rank=cfg.rank; me.hidden=5120; me.layers=40;
    me.experts=384; me.expert_width=2304; me.max_context=262144;
    me.max_partial_bytes=comm->MaxPartialBytes();
    me.model_digest[0]=1; me.partition_digest[0]=2; me.build_digest[0]=3;
    Identity peer; std::uint64_t limit=0;
    CHECK(exchange_identity(*comm, me, peer, limit, error));
    CHECK(peer.rank == 1u - cfg.rank && limit > 0);
    if (peer_loss && cfg.rank == 1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        std::puts("rank 1: intentionally closing owned test peer");
        return 0;
    }
    hipStream_t stream=nullptr; float* device=nullptr;
    CHECK(hipStreamCreate(&stream)==hipSuccess);
    CHECK(hipMalloc(&device, sizeof(float))==hipSuccess);
    float local=static_cast<float>(cfg.rank + 1);
    CHECK(hipMemcpy(device,&local,sizeof(float),hipMemcpyHostToDevice)==hipSuccess);
    CHECK(comm->BeginOperation(1,&error));
    if (cfg.rank==0 && !peer_loss) {
        CHECK(send_message(*comm, Control{Command::Step, 1, 0, {42}}, 16, error));
        // Rank 1 enters its RDMA wait with unread control bytes already queued.
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    const float* remote=comm->ExchangePartial(device,sizeof(float),stream,&error);
    if (peer_loss) {
        CHECK(!remote && error.find("peer closed the bootstrap connection") != std::string::npos);
        error.clear();
        CHECK(!comm->BeginOperation(2, &error));
        CHECK(error.find("poisoned") != std::string::npos &&
              error.find("peer closed the bootstrap connection") != std::string::npos);
        CHECK(hipFree(device)==hipSuccess); CHECK(hipStreamDestroy(stream)==hipSuccess);
        std::puts("DeepSeek TP2 peer loss detected with sticky original cause");
        return 0;
    }
    CHECK(remote);
    float got=0;
    CHECK(hipMemcpyAsync(&got,remote,sizeof(float),hipMemcpyDeviceToHost,stream)==hipSuccess);
    CHECK(hipStreamSynchronize(stream)==hipSuccess);
    CHECK(got==static_cast<float>(2-cfg.rank));
    CHECK(comm->CheckQueued(&error)); CHECK(comm->EndOperation(1,&error));
    Control message;
    if (cfg.rank==1) {
        CHECK(receive_message(*comm,16,message,error));
        CHECK(message.command==Command::Step && message.words==std::vector<std::uint32_t>{42});
        CHECK(send_message(*comm, Control{Command::Verdict, 2, 0, {0,43}}, 16, error));
    } else {
        CHECK(receive_message(*comm,16,message,error));
        CHECK(message.command==Command::Verdict && message.words==std::vector<std::uint32_t>({0,43}));
    }
    CHECK(hipFree(device)==hipSuccess); CHECK(hipStreamDestroy(stream)==hipSuccess);
    std::printf("DeepSeek TP2 real RDMA/control coexistence passed, rank %u\n", cfg.rank);
}
