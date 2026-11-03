#include "../src/ds4/ds4_tp_pack.hpp"
#include <array>
#include <cstdio>
#include <vector>
using namespace strata::ds4::tp2;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"line %d: %s (%s)\n",__LINE__,#x,error.c_str()); return 1; } } while (0)
int main() {
    constexpr int width=2304, hidden=4, block=256;
    std::string error;
    std::array<std::vector<std::uint8_t>,3> input{std::vector<std::uint8_t>(width*2),
        std::vector<std::uint8_t>(width*3),std::vector<std::uint8_t>(hidden*18)};
    for (std::size_t j=0;j<input.size();++j)
        for (std::size_t i=0;i<input[j].size();++i) input[j][i]=static_cast<std::uint8_t>((i*17+j*31)%251);
    const std::array<ConstBytes,3> src{input[0],input[1],input[2]};
    for (int layer=0;layer<2;++layer) {
        std::array<Range,2> ranges;
        CHECK(expert_ranges(width,block,layer,ranges,error));
        std::vector<std::uint8_t> gathered_gate, gathered_up;
        std::array<std::vector<std::uint8_t>,2> packed_down;
        for (int rank=0;rank<2;++rank) {
            ExpertSpans s;
            CHECK(expert_spans(width,hidden,block,ranges[rank],2,3,18,s,error));
            std::vector<std::uint8_t> gate(s.gate_bytes),up(s.up_bytes),down(s.down_row_bytes*hidden);
            CHECK(pack_expert(s,src,{Bytes(gate),Bytes(up),Bytes(down)},error));
            gathered_gate.insert(gathered_gate.end(),gate.begin(),gate.end());
            gathered_up.insert(gathered_up.end(),up.begin(),up.end());
            packed_down[rank]=std::move(down);
            // A too-short destination must fail before any output is modified.
            std::fill(gate.begin(),gate.end(),0xa5);
            CHECK(!pack_expert(s,src,{Bytes(gate).first(gate.size()-1),Bytes(up),Bytes(packed_down[rank])},error));
            for (auto v:gate) CHECK(v==0xa5);
            // In-place packing would overwrite a later input and is forbidden.
            CHECK(!pack_expert(s,src,{Bytes(input[0]).first(s.gate_bytes),Bytes(up),Bytes(packed_down[rank])},error));
        }
        CHECK(gathered_gate==input[0] && gathered_up==input[1]);
        const std::size_t n0=packed_down[0].size()/hidden,n1=packed_down[1].size()/hidden;
        for (int row=0;row<hidden;++row) {
            for (std::size_t i=0;i<n0;++i) CHECK(packed_down[0][row*n0+i]==input[2][row*18+i]);
            for (std::size_t i=0;i<n1;++i) CHECK(packed_down[1][row*n1+i]==input[2][row*18+n0+i]);
        }
    }
    std::puts("DeepSeek TP2 expert packing: rotated ranks reconstruct gate/up/down and reject unsafe buffers");
}
