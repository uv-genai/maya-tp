#include "strata/core/glm_tp_batch.hpp"

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
    limits.rows = 0;
    CHECK(!encode(batch, limits, bad, error));
    std::puts("GLM TP2 batch framing, bounds, identities and legacy word counts passed");
    return 0;
}
