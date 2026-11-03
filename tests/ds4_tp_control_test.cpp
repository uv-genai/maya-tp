#include "../src/ds4/ds4_tp_control.hpp"

#include <cstdio>
#include <limits>

using namespace strata::ds4::tp2;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); return 1; } } while (0)

int main() {
    std::string error;
    std::vector<std::uint8_t> bytes;
    Control in{Command::Verify, 0x0102030405060708ULL, 0x1122334455667788ULL, {1, 258, 129280}};
    CHECK(encode_control(in, 100, bytes, error));
    CHECK(bytes.size() == kControlHeaderBytes + 12 && bytes[0] == 'D' && bytes[1] == 'S' &&
          bytes[2] == '4' && bytes[3] == '2' && bytes[16] == 1 && bytes[23] == 8 &&
          bytes[24] == 0x11 && bytes[31] == 0x88);
    Control out{Command::Reset, 7, 9, {}};
    CHECK(decode_control(bytes, 100, out, error));
    CHECK(out.command == in.command && out.sequence == in.sequence && out.position == in.position && out.words == in.words);
    for (std::size_t n = 0; n < bytes.size(); ++n) {
        Control untouched{Command::Reset, 42, 0, {}};
        CHECK(!decode_control(std::span(bytes).first(n), 100, untouched, error));
        CHECK(untouched.sequence == 42);
    }
    auto corrupt = bytes;
    corrupt[0] ^= 1; CHECK(!decode_control(corrupt, 100, out, error));
    corrupt = bytes; corrupt[7] = 2; CHECK(!decode_control(corrupt, 100, out, error));
    corrupt = bytes; corrupt[11] = 99; CHECK(!decode_control(corrupt, 100, out, error));
    corrupt = bytes; corrupt[12] = 0xff; CHECK(!decode_control(corrupt, 100, out, error));
    corrupt = bytes; corrupt.push_back(0); CHECK(!decode_control(corrupt, 100, out, error));
    CHECK(!decode_control(bytes, 2, out, error));
    CHECK(!decode_control(bytes, kControlMaxWords + 1, out, error));
    for (const Control& good : {Control{Command::Feed, 1, 0, {1, 2}}, Control{Command::Step, 2, 9, {3}},
                                Control{Command::Verdict, 3, 9, {0, 3}}, Control{Command::Reset, 4, 0, {}},
                                Control{Command::Shutdown, 5, 0, {}}}) {
        CHECK(encode_control(good, 100, bytes, error)); CHECK(decode_control(bytes, 100, out, error));
        CHECK(out.command == good.command && out.sequence == good.sequence && out.words == good.words);
    }
    for (const Control& bad : {Control{Command::Feed, 1, 0, {}}, Control{Command::Step, 1, 0, {1, 2}},
                               Control{Command::Verify, 1, 0, {1}}, Control{Command::Verdict, 1, 0, {16, 1}},
                               Control{Command::Reset, 1, 0, {1}}, Control{Command::Shutdown, 0, 0, {}}}) {
        bytes = {7}; CHECK(!encode_control(bad, 100, bytes, error)); CHECK(bytes == std::vector<std::uint8_t>{7});
    }
    std::puts("DeepSeek TP2 control: portable round-trip and malformed frames passed");
}
