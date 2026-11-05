// Paired sequence allocation, bind, prompt and snapshot regression for GLM TP2.
// Run manually on both gfx1151 ranks with identical packs and prompt IDs.
#include "strata/core/glm_model.hpp"
#include "strata/kernels/sampler.hpp"
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
    if (argc != 10 && (argc != 11 || (std::string(argv[10]) != "--rows" && std::string(argv[10]) != "--mtp"))) {
        std::fprintf(stderr, "usage: rank host port rdma pack context sequences prompt-ids output-json [--rows|--mtp]\n");
        return 2;
    }
    const int rank = std::stoi(argv[1]);
    const int port = std::stoi(argv[3]);
    const int64_t context = std::stoll(argv[6]);
    const int sequences = std::stoi(argv[7]);
    const bool check_mtp = argc == 11 && std::string(argv[10]) == "--mtp";
    std::string error;
    CHECK(rank >= 0 && rank < 2 && port > 0 && port <= 65535);
    CHECK(context >= 256 && sequences >= 2 && sequences <= 4);
    const std::string count = std::to_string(sequences);
    CHECK(setenv("STRATA_GLM_SEQS", count.c_str(), 1) == 0);
    CHECK(setenv("STRATA_GLM_SPLIT", "0", 1) == 0);
    if (check_mtp) {
        CHECK(setenv("STRATA_GLM_MTP", "1", 1) == 0);
        CHECK(setenv("STRATA_GLM_MULTI_MTP", "2", 1) == 0);
    }
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
    if (argc == 11) {
        // Give each slot a different retained position/history before testing
        // packed rows against independent ordinary forwards of the same state.
        for (int slot = 0; slot < sequences; ++slot) {
            model.seq_bind_all(slot);
            CHECK(model.snapshot_restore());
            for (int step = 0; step < slot; ++step)
                CHECK(model.forward({picks[(size_t) slot]}, logits, error));
            CHECK(model.snapshot_save());
        }
        for (int count = 1; count <= sequences; ++count) {
            std::vector<int> slots, expected, actual((size_t) count);
            std::vector<int32_t> inputs;
            std::vector<strata::kernels::SamplerParams> sampling((size_t) count);
            for (int slot = 0; slot < sequences; ++slot) {
                model.seq_bind_all(slot);
                CHECK(model.snapshot_restore());
                if (slot >= count) continue;
                CHECK(model.forward({picks[(size_t) slot]}, logits, error));
                expected.push_back((int) std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
                CHECK(model.snapshot_restore());
                slots.push_back(slot);
                inputs.push_back(picks[(size_t) slot]);
                sampling[(size_t) slot].greedy = true;
            }
            CHECK(model.multi_rows(slots.data(), inputs.data(), sampling.data(), count, actual.data(), error));
            for (int row = 0; row < count; ++row)
                if (actual[(size_t) row] != expected[(size_t) row])
                    std::fprintf(stderr, "rank %d batch %d row %d pick %d expected %d\n", rank, count, row,
                                 actual[(size_t) row], expected[(size_t) row]);
            CHECK(actual == expected);
            for (int slot = 0; slot < sequences; ++slot)
                CHECK(model.seq_pos(slot) == (int64_t) prompt.size() + slot + (slot < count ? 1 : 0));
            CHECK(model.multi_boundary(error));
            std::fprintf(stderr, "rank %d passed %d packed rows against independent forwards\n", rank, count);
        }
    }
    if (check_mtp) {
        CHECK(model.mtpw_ok());
        std::vector<int> slots, counts;
        std::vector<strata::kernels::SamplerParams> sampling((size_t) sequences);
        for (int slot = 0; slot < sequences; ++slot) {
            model.seq_bind_all(slot);
            CHECK(model.snapshot_save());
            slots.push_back(slot);
            counts.push_back(slot % 3 == 0 ? 3 : slot % 3 == 1 ? 2 : 1);
            sampling[(size_t) slot].greedy = true;
        }
        for (int retained = 1; retained <= 3; ++retained) {
            std::vector<int32_t> inputs, reference_picks, last, written;
            std::vector<int> keep, drafts;
            std::vector<int64_t> positions;
            std::vector<std::vector<float>> expected((size_t) sequences);
            for (int slot = 0; slot < sequences; ++slot) {
                model.seq_bind_all(slot);
                CHECK(model.snapshot_restore());
                positions.push_back(model.seq_pos(slot));
                const int count = counts[(size_t) slot];
                const int kept = std::min(retained, count);
                std::vector<int32_t> window;
                int32_t token = picks[(size_t) slot];
                int32_t pending = -1;
                for (int row = 0; row < count; ++row) {
                    window.push_back(token);
                    CHECK(model.forward({token}, logits, error));
                    token = (int32_t) std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));
                    reference_picks.push_back(token);
                    if (row + 1 == kept) pending = token;
                }
                CHECK(model.snapshot_restore());
                for (int row = 0; row < kept; ++row)
                    CHECK(model.forward({window[(size_t) row]}, logits, error));
                CHECK(model.forward({pending}, expected[(size_t) slot], error));
                CHECK(model.snapshot_restore());
                inputs.insert(inputs.end(), window.begin(), window.end());
                written.insert(written.end(), window.begin() + 1, window.begin() + kept);
                keep.push_back(kept);
                last.push_back(pending);
                drafts.push_back(slot == 2 ? 0 : 2);
            }
            std::vector<int32_t> actual(inputs.size());
            CHECK(model.mtpw_front(slots.data(), counts.data(), sequences, inputs.data(), error));
            CHECK(model.mtpw_back(slots.data(), counts.data(), sequences, inputs.data(), sampling.data(), error));
            CHECK(model.mtpw_picks(slots.data(), counts.data(), sequences, actual.data(), error));
            CHECK(actual == reference_picks);
            CHECK(model.mtpw_commit(slots.data(), counts.data(), keep.data(), sequences, error));
            CHECK(model.mtpw_draft_queue(slots.data(), keep.data(), sequences, written.data(), last.data(), drafts.data(), error));
            std::vector<int32_t> proposals((size_t) sequences * 2, -1);
            CHECK(model.mtpw_draft_take(proposals.data(), error));
            for (int slot = 0; slot < sequences; ++slot) {
                for (int draft = 0; draft < 2; ++draft) {
                    const int32_t token = proposals[(size_t) slot * 2 + (size_t) draft];
                    if (drafts[(size_t) slot] == 0) CHECK(token == -1);
                    else CHECK(token >= 0 && token < geometry.n_vocab);
                }
                CHECK(model.seq_pos(slot) == positions[(size_t) slot] + keep[(size_t) slot]);
                model.seq_bind_all(slot);
                CHECK(model.forward({last[(size_t) slot]}, logits, error));
                if (!agrees(logits, expected[(size_t) slot])) {
                    float largest = 0;
                    for (size_t token = 0; token < logits.size(); ++token)
                        largest = std::max(largest, std::abs(logits[token] - expected[(size_t) slot][token]));
                    std::fprintf(stderr, "rank %d keep %d slot %d replay max difference %g\n", rank, retained, slot, largest);
                }
                CHECK(agrees(logits, expected[(size_t) slot]));
            }
            std::fprintf(stderr, "rank %d MTP retained %d, ragged windows, replay and per-slot drafts passed\n", rank, retained);
        }
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
           << ",\"rows\":" << (argc == 11 ? "true" : "false")
           << ",\"mtp\":" << (check_mtp ? "true" : "false") << ",\"passed\":true}\n";
    CHECK(output.good());
    std::printf("PASS rank %d, %d isolated sequences, snapshots and replay\n", rank, sequences);
    return 0;
}
