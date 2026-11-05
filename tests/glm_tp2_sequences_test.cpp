// Paired sequence allocation, bind, prompt and snapshot regression for GLM TP2.
// Run manually on both gfx1151 ranks with identical packs and prompt IDs.
#include "strata/core/glm_model.hpp"
#include "strata/net/verbs.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "line %d failed: %s (%s)\n", __LINE__, #condition, error.c_str()); \
    return 1; \
} } while (false)

int main(int argc, char** argv) {
    if (argc != 10) {
        std::fprintf(stderr, "usage: rank host port rdma pack context sequences prompt-ids output-json\n");
        return 2;
    }
    const int rank = std::stoi(argv[1]);
    const int port = std::stoi(argv[3]);
    const int64_t context = std::stoll(argv[6]);
    const int sequences = std::stoi(argv[7]);
    std::string error;
    CHECK(rank >= 0 && rank < 2 && port > 0 && port <= 65535);
    CHECK(context >= 256 && sequences >= 2 && sequences <= 4);
    const std::string count = std::to_string(sequences);
    CHECK(setenv("STRATA_GLM_SEQS", count.c_str(), 1) == 0);
    CHECK(setenv("STRATA_GLM_SPLIT", "0", 1) == 0);
    std::vector<int32_t> prompt;
    {
        std::ifstream input(argv[8]);
        CHECK(input.good());
        int32_t token;
        while (input >> token) prompt.push_back(token);
        CHECK(input.eof() && prompt.size() >= 2 && prompt.size() < (size_t) context);
    }
    strata::core::Glm5Geometry geometry;
    CHECK(strata::core::Glm5Model::peek_pack_geometry(argv[5], geometry, error));
    for (int32_t token : prompt) CHECK(token >= 0 && token < geometry.n_vocab);
    strata::net::IbrverbsConfig config;
    config.rank = (uint32_t) rank;
    config.world_size = 2;
    config.bootstrap_host = argv[2];
    config.bootstrap_port = (uint16_t) port;
    config.rdma_device = argv[4];
    auto communicator = strata::net::CreateIbrverbsCommunicator(config, &error);
    CHECK(communicator);
    strata::core::TpContext tp;
    CHECK(strata::core::TpContext::create(rank, 2, geometry.n_ff_exp, geometry.n_ff_dense,
                                        256, communicator, tp, error));
    CHECK(communicator->BeginOperation(0, &error));
    strata::core::Glm5Model model;
    model.set_tp(tp);
    CHECK(model.load_pack_env(argv[5], context, error, "auto"));
    CHECK(model.seq_count() == sequences);
    const std::vector<int32_t> prefix(prompt.begin(), prompt.end() - 1);
    std::vector<float> reference, logits;
    std::vector<std::vector<float>> replay((size_t) sequences);
    std::vector<int> picks;
    const auto agrees = [](const std::vector<float>& left, const std::vector<float>& right) {
        if (left.size() != right.size() || left.empty()) return false;
        for (size_t i = 0; i < left.size(); ++i)
            if (!std::isfinite(left[i]) || !std::isfinite(right[i]) ||
                std::abs(left[i] - right[i]) > 1e-4f) return false;
        return true;
    };
    for (int slot = 0; slot < sequences; ++slot) {
        model.seq_bind_all(slot);
        model.reset();
        CHECK(model.prefill(prefix, error, prompt.back()));
        CHECK(model.forward({prompt.back()}, logits, error));
        CHECK(model.seq_pos(slot) == (int64_t) prompt.size());
        CHECK(!logits.empty());
        if (slot == 0) reference = logits;
        CHECK(agrees(reference, logits));
        CHECK(model.snapshot_save());
        picks.push_back((int) std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
        std::fprintf(stderr, "rank %d slot %d prompt passed, position=%lld context=%lld pick=%d\n",
                     rank, slot, (long long) model.seq_pos(slot), (long long) model.seq_max_ctx(slot), picks.back());
    }
    for (int slot = 0; slot < sequences; ++slot) {
        model.seq_bind_all(slot);
        CHECK(model.forward({picks[(size_t) slot]}, replay[(size_t) slot], error));
        CHECK(model.seq_pos(slot) == (int64_t) prompt.size() + 1);
    }
    for (int slot = sequences - 1; slot >= 0; --slot) {
        model.seq_bind_all(slot);
        CHECK(model.snapshot_restore());
        CHECK(model.seq_pos(slot) == (int64_t) prompt.size());
        CHECK(model.forward({picks[(size_t) slot]}, logits, error));
        CHECK(agrees(replay[(size_t) slot], logits));
        for (int other = 0; other < sequences; ++other)
            CHECK(model.seq_pos(other) == (int64_t) prompt.size() + 1);
    }
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(communicator->CheckQueued(&error));
    CHECK(communicator->EndOperation(0, &error));
    std::vector<int> peer_picks(picks.size());
    CHECK(communicator->SendControl(picks.data(), picks.size() * sizeof(int), &error));
    CHECK(communicator->RecvControl(peer_picks.data(), peer_picks.size() * sizeof(int), &error));
    CHECK(picks == peer_picks);
    size_t free_bytes = 0, total_bytes = 0;
    CHECK(cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess);
    std::ofstream output(argv[9]);
    CHECK(output.good());
    output << "{\"rank\":" << rank << ",\"sequences\":" << sequences
           << ",\"context\":" << context << ",\"free_bytes\":" << free_bytes
           << ",\"total_bytes\":" << total_bytes << ",\"prompt_tokens\":" << prompt.size()
           << ",\"passed\":true}\n";
    CHECK(output.good());
    std::printf("PASS rank %d, %d isolated sequences, snapshots and replay\n", rank, sequences);
    return 0;
}
