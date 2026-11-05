#include "strata/core/glm_tp_batch.hpp"

#include <algorithm>
#include <cstdio>

using namespace strata::core::glm_tp;

#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "line %d failed: %s (%s)\n", __LINE__, #condition, error.c_str()); \
    return 1; \
} } while (false)

int main() {
    std::string error;
    Limits limits{4, 16, 1000, 32768};
    Batch batch;
    batch.operation = (1ull << 40) + 7;
    for (int count = 1; count <= 4; ++count) {
        batch.members.push_back({count - 1, (1ull << 35) + (uint64_t) count,
                                 (uint64_t) count * 10, {30 + count}});
        std::vector<int32_t> wire;
        CHECK(encode(batch, limits, wire, error));
        Batch decoded;
        CHECK(decode(wire, limits, batch.operation, decoded, error));
        CHECK(decoded.operation == batch.operation && decoded.members.size() == batch.members.size());
        for (size_t i = 0; i < batch.members.size(); ++i) {
            CHECK(decoded.members[i].slot == batch.members[i].slot);
            CHECK(decoded.members[i].generation == batch.members[i].generation);
            CHECK(decoded.members[i].position == batch.members[i].position);
            CHECK(decoded.members[i].tokens == batch.members[i].tokens);
        }
        CHECK(legacy_rows((size_t) count * 2, 4) == count);
        CHECK(!decode(wire, limits, batch.operation + 1, decoded, error));
        for (size_t cut = 0; cut < wire.size(); ++cut) {
            const std::vector<int32_t> short_wire(wire.begin(), wire.begin() + (ptrdiff_t) cut);
            CHECK(!decode(short_wire, limits, batch.operation, decoded, error));
            CHECK(decoded.members.empty());
        }
    }
    std::vector<int32_t> wire;
    CHECK(encode(batch, limits, wire, error));
    Batch decoded;
    auto bad = wire;
    bad[0] = kVersion + 1;
    CHECK(!decode(bad, limits, batch.operation, decoded, error));
    bad = wire;
    bad[3] = INT32_MAX;
    CHECK(!decode(bad, limits, batch.operation, decoded, error));
    bad = wire;
    bad[9] = INT32_MAX;
    CHECK(!decode(bad, limits, batch.operation, decoded, error));
    bad = wire;
    bad.push_back(0);
    CHECK(!decode(bad, limits, batch.operation, decoded, error));
    bad = wire;
    bad[11] = 0;
    CHECK(!decode(bad, limits, batch.operation, decoded, error));
    for (uint64_t position : {uint64_t{32767}, uint64_t{32768}, UINT64_MAX}) {
        Batch invalid = batch;
        invalid.members[0].position = position;
        CHECK(!encode(invalid, limits, bad, error));
    }
    for (int token : {-1, 1000}) {
        Batch invalid = batch;
        invalid.members[0].tokens[0] = token;
        CHECK(!encode(invalid, limits, bad, error));
    }
    Batch invalid = batch;
    invalid.operation = 0;
    CHECK(!encode(invalid, limits, bad, error));
    invalid = batch;
    invalid.members[0].generation = 0;
    CHECK(!encode(invalid, limits, bad, error));
    invalid = batch;
    invalid.members[0].tokens.clear();
    CHECK(!encode(invalid, limits, bad, error));
    invalid = batch;
    invalid.members[0].tokens.assign(16, 1);
    CHECK(!encode(invalid, limits, bad, error));
    CHECK(legacy_rows(0, 4) == 0 && legacy_rows(3, 4) == 0 && legacy_rows(10, 4) == 0);
    CHECK(legacy_rows(2, 0) == 0 && legacy_rows(2, 5) == 0);
    const Batch target{1, {{0, (1ull << 35) + 5, 10, {10, 11, 12}}, {1, 6, 20, {20, 21}}}};
    Verdict verdict;
    for (int keep = 1; keep <= 3; ++keep) {
        verdict = {2, 1, {}};
        for (size_t member = 0; member < target.members.size(); ++member) {
            const auto& window = target.members[member];
            const int retained = std::min(keep, (int) window.tokens.size());
            VerdictMember result{window.slot, window.generation, window.position,
                                 member == 0 ? 2 : 0, member == 0, {}};
            result.emitted.assign(window.tokens.begin() + 1, window.tokens.begin() + retained);
            result.emitted.push_back(30 + (int) member);
            verdict.members.push_back(std::move(result));
        }
        CHECK(encode_verdict(verdict, target, limits, 2, wire, error));
        Verdict decoded_verdict;
        CHECK(decode_verdict(wire, target, limits, 2, 2, decoded_verdict, error));
        CHECK(decoded_verdict.operation == verdict.operation && decoded_verdict.target_operation == target.operation);
        for (size_t i = 0; i < verdict.members.size(); ++i) {
            CHECK(decoded_verdict.members[i].slot == verdict.members[i].slot);
            CHECK(decoded_verdict.members[i].generation == verdict.members[i].generation);
            CHECK(decoded_verdict.members[i].position == verdict.members[i].position);
            CHECK(decoded_verdict.members[i].emitted == verdict.members[i].emitted);
            CHECK(decoded_verdict.members[i].drafts == verdict.members[i].drafts);
            CHECK(decoded_verdict.members[i].more == verdict.members[i].more);
        }
        for (size_t cut = 0; cut < wire.size(); ++cut) {
            const std::vector<int32_t> truncated(wire.begin(), wire.begin() + (ptrdiff_t) cut);
            CHECK(!decode_verdict(truncated, target, limits, 2, 2, decoded_verdict, error));
            CHECK(decoded_verdict.members.empty());
        }
        CHECK(!decode_verdict(wire, target, limits, 3, 2, decoded_verdict, error));
    }
    for (int mutation = 0; mutation < 9; ++mutation) {
        Verdict invalid_verdict = verdict;
        switch (mutation) {
            case 0: invalid_verdict.members.pop_back(); break;
            case 1: invalid_verdict.members[0].generation++; break;
            case 2: invalid_verdict.members[0].position++; break;
            case 3: invalid_verdict.members[0].emitted.clear(); break;
            case 4: invalid_verdict.members[0].emitted[0] = 99; break;
            case 5: invalid_verdict.members[0].drafts = 3; break;
            case 6: invalid_verdict.members[1].drafts = 1; break;
            case 7: invalid_verdict.target_operation++; break;
            case 8: invalid_verdict.members[0].emitted.back() = 1000; break;
        }
        CHECK(!encode_verdict(invalid_verdict, target, limits, 2, bad, error));
    }
    Verdict decoded_verdict;
    bad = wire;
    bad[13] = 2;
    CHECK(!decode_verdict(bad, target, limits, 2, 2, decoded_verdict, error));
    bad = wire;
    bad[11] = INT32_MAX;
    CHECK(!decode_verdict(bad, target, limits, 2, 2, decoded_verdict, error));
    limits.rows = 0;
    CHECK(!encode(batch, limits, bad, error));
    std::puts("GLM TP2 batch framing, bounds, identities and legacy word counts passed");
    return 0;
}
