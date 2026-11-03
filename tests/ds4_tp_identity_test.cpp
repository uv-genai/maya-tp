#include "../src/ds4/ds4_tp_identity.hpp"
#include <cstdio>

using namespace strata::ds4::tp2;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); return 1; } } while (0)
int main() {
    Identity a;
    a.protocol = 1; a.world = 2; a.rank = 0; a.hidden = 5120; a.layers = 40;
    a.experts = 384; a.expert_width = 2304; a.max_context = 524288;
    a.feature_flags = 1; a.max_partial_bytes = 1048576;
    a.model_digest[0] = 1; a.partition_digest[0] = 2; a.build_digest[0] = 3;
    Identity b = a; b.rank = 1; b.max_partial_bytes = 524288;
    std::uint64_t limit = 0; std::string error;
    auto wire = encode_identity(b);
    CHECK(wire.size() == kIdentityBytes && wire[0] == 'D' && wire[1] == 'S' && wire[2] == 'I' && wire[3] == 'D');
    Identity decoded;
    CHECK(decode_identity(wire, decoded, error));
    CHECK(decoded.rank == b.rank && decoded.model_digest == b.model_digest &&
          decoded.partition_digest == b.partition_digest && decoded.build_digest == b.build_digest &&
          decoded.max_partial_bytes == b.max_partial_bytes && decoded.max_context == b.max_context);
    for (std::size_t n = 0; n < wire.size(); ++n) {
        decoded.rank = 42;
        CHECK(!decode_identity(std::span(wire).first(n), decoded, error) && decoded.rank == 42);
    }
    auto broken = wire; broken[0] ^= 1; CHECK(!decode_identity(broken, decoded, error));
    broken = wire; broken[7] = 2; CHECK(!decode_identity(broken, decoded, error));
    broken = wire; broken.push_back(0); CHECK(!decode_identity(broken, decoded, error));
    CHECK(agree_identity(a, b, limit, error) && limit == 524288 && error.empty());
    CHECK(agree_identity(b, a, limit, error) && limit == 524288);
    auto rejected = [&](const Identity& x) {
        limit = 99; error.clear();
        return !agree_identity(a, x, limit, error) && limit == 99 && !error.empty();
    };
    Identity x = b; x.rank = 0; CHECK(rejected(x));
    x = b; x.rank = 2; CHECK(rejected(x));
    x = b; x.world = 1; CHECK(rejected(x));
    x = b; x.protocol = 2; CHECK(rejected(x));
    x = b; x.model_digest[31] ^= 1; CHECK(rejected(x));
    x = b; x.model_digest.fill(0); CHECK(rejected(x));
    x = b; x.partition_digest[31] ^= 1; CHECK(rejected(x));
    x = b; x.build_digest[31] ^= 1; CHECK(rejected(x));
    x = b; ++x.expert_width; CHECK(rejected(x));
    x = b; ++x.layers; CHECK(rejected(x));
    x = b; ++x.max_context; CHECK(rejected(x));
    x = b; x.feature_flags = 0; CHECK(rejected(x));
    x = b; x.max_partial_bytes = 0; CHECK(rejected(x));
    std::puts("DeepSeek TP2 identity: matching peers, negotiated window and mismatch rejection passed");
}
