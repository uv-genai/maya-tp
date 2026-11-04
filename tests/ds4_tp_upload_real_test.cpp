// Upload real pack expert slices to a GPU and compare them to CPU-packed bytes.
#include "../src/ds4/ds4_tp_upload.cuh"
#include "../src/ds4/ds4_tp_expert_pool.cuh"
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
    if (argc!=4) { std::fprintf(stderr,"usage: pack-dir layer expert\n"); return 2; }
    std::string error; const std::filesystem::path pack=argv[1];
    const int layer=std::atoi(argv[2]), expert=std::atoi(argv[3]);
    CHECK(layer>=0 && expert>=0);
    std::ifstream cfg(pack/"ds4.txt"); CHECK(cfg.good());
    std::string line, dir; int width=0, hidden=0;
    while (std::getline(cfg,line)) {
        const auto p=line.find(' '); if (p==std::string::npos) continue;
        const auto key=line.substr(0,p), value=line.substr(p+1);
        if (key=="model_dir") dir=value;
        else if (key=="moe_intermediate_size") width=std::stoi(value);
        else if (key=="hidden_size") hidden=std::stoi(value);
    }
    CHECK(!dir.empty() && width>0 && hidden>0);
    struct Tensor { int file=-1; std::uint64_t offset=0, size=0; int rows=0, cols=0; };
    std::array<Tensor,3> tensors{}; std::vector<std::string> files;
    const auto prefix="layers."+std::to_string(layer)+".ffn.experts."+std::to_string(expert)+".w";
    std::ifstream index(pack/"tensors.txt"); CHECK(index.good());
    while (std::getline(index,line)) {
        std::istringstream s(line); std::string kind; s>>kind;
        if (kind=="file") {
            std::size_t i; std::string name; s>>i; std::getline(s,name);
            if (!name.empty() && name[0]==' ') name.erase(0,1);
            CHECK(i<10000 && !name.empty()); if (files.size()<=i) files.resize(i+1); files[i]=name;
        } else if (kind=="t") {
            std::string name,dtype; s>>name>>dtype;
            if (name.size()!=prefix.size()+8 || name.compare(0,prefix.size(),prefix)!=0 ||
                name.substr(prefix.size()+1)!=".weight") continue;
            const int role=name[prefix.size()]-'0'; CHECK(role>=1 && role<=3);
            auto& t=tensors[role==1?0:role==3?1:2]; CHECK(t.file==-1);
            s>>t.file>>t.offset>>t.size>>t.rows>>t.cols; CHECK(!s.fail());
        }
    }
    std::array<std::vector<std::uint8_t>,3> source;
    for (int i=0;i<3;++i) {
        const auto& t=tensors[i]; CHECK(t.file>=0 && static_cast<std::size_t>(t.file)<files.size());
        CHECK(t.rows==(i==2?hidden:width) && t.cols==(i==2?width:hidden));
        CHECK(t.size>0 && t.size<=(1ull<<30)); source[i].resize(static_cast<std::size_t>(t.size));
        std::ifstream f(std::filesystem::path(dir)/files[t.file],std::ios::binary); CHECK(f.good());
        f.seekg(static_cast<std::streamoff>(t.offset));
        f.read(reinterpret_cast<char*>(source[i].data()),static_cast<std::streamsize>(t.size)); CHECK(f.good());
    }
    CHECK(tensors[0].size%width==0 && tensors[1].size%width==0 && tensors[2].size%hidden==0);
    const std::array<ConstBytes,3> src{source[0],source[1],source[2]};
    std::array<Range,2> ranges; CHECK(expert_ranges(width,256,layer,ranges,error));
    hipStream_t stream=nullptr; CHECK(hipStreamCreate(&stream)==hipSuccess);
    std::vector<ExpertSpans> plans;
    std::vector<std::vector<std::uint8_t>> expected_parts;
    for (int rank=0;rank<2;++rank) {
        ExpertSpans sp; CHECK(expert_spans(width,hidden,256,ranges[rank],tensors[0].size/width,
                                           tensors[1].size/width,tensors[2].size/hidden,sp,error));
        DeviceExpertLayout layout; CHECK(device_expert_layout(sp,layout,error));
        std::vector<std::uint8_t> expected(layout.bytes,0), actual(layout.bytes);
        const auto down=sp.down_row_bytes*sp.down_rows;
        CHECK(pack_expert(sp,src,{Bytes(expected).subspan(layout.offsets[0],sp.gate_bytes),
              Bytes(expected).subspan(layout.offsets[1],sp.up_bytes), Bytes(expected).subspan(layout.offsets[2],down)},error));
        std::uint8_t* gpu=nullptr; CHECK(hipMalloc(&gpu,layout.bytes)==hipSuccess);
        CHECK(hipMemsetAsync(gpu,0,layout.bytes,stream)==hipSuccess);
        CHECK(upload_expert(sp,src,gpu,layout.bytes,stream,layout,error));
        CHECK(hipMemcpyAsync(actual.data(),gpu,layout.bytes,hipMemcpyDeviceToHost,stream)==hipSuccess);
        CHECK(hipStreamSynchronize(stream)==hipSuccess);
        CHECK(expected==actual); CHECK(hipFree(gpu)==hipSuccess);
        plans.push_back(sp); expected_parts.push_back(std::move(expected));
        std::printf("layer=%d expert=%d rank=%d width=%d device_bytes=%zu: exact GPU slice\n",
                    layer,expert,rank,ranges[rank].count,layout.bytes);
    }
    CHECK(hipStreamDestroy(stream)==hipSuccess);
    {
        RankExpertPool too_small;
        CHECK(!too_small.allocate(0,plans,1,error));
        CHECK(!too_small.ready() && too_small.data(0)==nullptr && !too_small.failure().empty());
    }
    {
        RankExpertPool pool;
        const auto total=expected_parts[0].size()+expected_parts[1].size();
        CHECK(pool.allocate(0,plans,total,error) && pool.bytes()==total);
        CHECK(pool.staging_bytes()==std::max(expected_parts[0].size(),expected_parts[1].size()));
        CHECK(!pool.ready() && pool.data(0)==nullptr);
        CHECK(pool.upload(0,src,error) && !pool.ready() && pool.data(0)==nullptr);
        CHECK(pool.upload(1,src,error) && pool.ready());
        for (int rank=0;rank<2;++rank) {
            std::vector<std::uint8_t> got(expected_parts[rank].size());
            CHECK(hipMemcpy(got.data(),pool.data(rank),got.size(),hipMemcpyDeviceToHost)==hipSuccess);
            CHECK(got==expected_parts[rank]);
        }
        CHECK(pool.data(2)==nullptr);
        std::printf("layer=%d expert=%d: bounded GPU pool loaded both checked slices (%zu bytes)\n",layer,expert,total);
        CHECK(!pool.upload(0,src,error) && !pool.ready() && pool.data(0)==nullptr);
        const std::string first=error;
        CHECK(!pool.upload(1,src,error) && error==first);
    }
    {
        RankExpertPool poisoned;
        CHECK(poisoned.allocate(0,{plans[0]},expected_parts[0].size(),error));
        auto bad=src; bad[0]=src[0].first(0);
        CHECK(!poisoned.upload(0,bad,error) && !poisoned.ready());
        const std::string first=error;
        CHECK(!poisoned.upload(0,src,error) && error==first && poisoned.data(0)==nullptr);
    }
}
