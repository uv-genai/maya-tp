// Read one real packed GGUF expert, split it, and reconstruct its tensor bytes.
// This is read-only and does not require a GPU. It validates byte extraction,
// not quantized math or remote execution.
#include "../src/ds4/ds4_tp_pack.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
using namespace strata::ds4::tp2;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"line %d: %s (%s)\n",__LINE__,#x,error.c_str()); return 1; } } while (0)

int main(int argc, char** argv) {
    if (argc != 4) { std::fprintf(stderr,"usage: pack-dir layer expert\n"); return 2; }
    std::string error; const std::filesystem::path pack=argv[1];
    const int layer=std::atoi(argv[2]), expert=std::atoi(argv[3]);
    CHECK(layer >= 0 && expert >= 0);
    std::ifstream cfg(pack/"ds4.txt"); CHECK(cfg.good());
    std::string key, value, dir; int width=0, hidden=0, heads=0, groups=0;
    while (std::getline(cfg,value)) {
        auto p=value.find(' '); if(p==std::string::npos) continue;
        key=value.substr(0,p); auto v=value.substr(p+1);
        if(key=="model_dir") dir=v;
        else if(key=="moe_intermediate_size") width=std::stoi(v);
        else if(key=="hidden_size") hidden=std::stoi(v);
        else if(key=="num_attention_heads") heads=std::stoi(v);
        else if(key=="o_groups") groups=std::stoi(v);
    }
    CHECK(!dir.empty() && width>0 && hidden>0);
    struct Tensor { int file=-1; std::uint64_t offset=0, size=0; int a=0,b=0; };
    std::array<Tensor,3> tensor{};
    std::vector<std::string> files;
    std::ifstream index(pack/"tensors.txt"); CHECK(index.good());
    std::string line;
    const auto prefix="layers."+std::to_string(layer)+".ffn.experts."+std::to_string(expert)+".w";
    while(std::getline(index,line)) {
        std::istringstream ss(line); std::string kind; ss>>kind;
        if(kind=="file") { std::size_t i; std::string name; ss>>i; std::getline(ss,name); if(name.size() && name[0]==' ') name.erase(0,1);
            CHECK(!name.empty() && i<10000); if(files.size()<=i) files.resize(i+1); files[i]=name; }
        else if(kind=="t") { std::string name,dtype; ss>>name>>dtype;
            if(name.size()!=prefix.size()+8 || name.compare(0,prefix.size(),prefix)!=0 || name.substr(prefix.size()+1)!=".weight") continue;
            int role=name[prefix.size()]-'0'; CHECK(role>=1 && role<=3);
            auto& t=tensor[role==1?0:role==3?1:2]; CHECK(t.file==-1);
            ss>>t.file>>t.offset>>t.size>>t.a>>t.b; CHECK(!ss.fail());
        }
    }
    std::array<std::vector<std::uint8_t>,3> source;
    for(int i=0;i<3;++i) {
        const auto& t=tensor[i]; CHECK(t.file>=0 && static_cast<std::size_t>(t.file)<files.size());
        CHECK(t.a==(i==2?hidden:width) && t.b==(i==2?width:hidden));
        CHECK(t.size>0 && t.size<=(1ull<<30)); source[i].resize(static_cast<std::size_t>(t.size));
        std::ifstream f(std::filesystem::path(dir)/files[t.file],std::ios::binary); CHECK(f.good());
        f.seekg(static_cast<std::streamoff>(t.offset)); f.read(reinterpret_cast<char*>(source[i].data()),static_cast<std::streamsize>(t.size));
        CHECK(f.good());
    }
    CHECK(tensor[0].size%width==0 && tensor[1].size%width==0 && tensor[2].size%hidden==0);
    const auto gate_row=tensor[0].size/width, up_row=tensor[1].size/width, down_row=tensor[2].size/hidden;
    const std::array<ConstBytes,3> src{source[0],source[1],source[2]};
    std::array<RankGeometry,2> ranks; CHECK(rank_geometry(width,256,layer,heads,groups,ranks,error));
    std::array<std::vector<std::uint8_t>,3> restored{std::vector<std::uint8_t>(source[0].size()),
        std::vector<std::uint8_t>(source[1].size()),std::vector<std::uint8_t>(source[2].size())};
    for(int rank=0;rank<2;++rank) {
        ExpertSpans s; CHECK(expert_spans(width,hidden,256,ranks[rank].experts,gate_row,up_row,down_row,s,error));
        std::array<std::vector<std::uint8_t>,3> part{std::vector<std::uint8_t>(s.gate_bytes),
            std::vector<std::uint8_t>(s.up_bytes),std::vector<std::uint8_t>(s.down_row_bytes*s.down_rows)};
        CHECK(pack_expert(s,src,{Bytes(part[0]),Bytes(part[1]),Bytes(part[2])},error));
        std::copy(part[0].begin(),part[0].end(),restored[0].begin()+s.gate_offset);
        std::copy(part[1].begin(),part[1].end(),restored[1].begin()+s.up_offset);
        for(std::size_t row=0;row<s.down_rows;++row)
            std::copy_n(part[2].begin()+row*s.down_row_bytes,s.down_row_bytes,
                        restored[2].begin()+row*s.down_source_stride+s.down_row_offset);
    }
    CHECK(restored==source);
    std::printf("real expert layer=%d id=%d: %d heads and %d groups/rank; rank slices reconstruct all three quantized tensors\n",
                layer,expert,ranks[0].attention_heads.count,ranks[0].output_groups.count);
}
