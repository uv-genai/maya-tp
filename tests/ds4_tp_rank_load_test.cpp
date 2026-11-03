// Explicit-pack GPU rank loader test; not an unattended test or model execution.
#include "../src/ds4/ds4_tp_rank_loader.hpp"
#include "../src/ds4/ds4_tp_upload.cuh"
#include <hip/hip_runtime.h>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using namespace strata::ds4::tp2;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"line %d: %s (%s)\n",__LINE__,#x,error.c_str()); return false; } } while (0)

static bool check_entry(const std::filesystem::path& pack, const RankExpertEntry& entry, int rank,
                        std::string& error) {
    std::ifstream cfg(pack/"ds4.txt"); CHECK(cfg.good());
    std::string line, dir; int width=0, hidden=0;
    while (std::getline(cfg,line)) {
        const auto p=line.find(' '); if (p==std::string::npos) continue;
        const auto key=line.substr(0,p), value=line.substr(p+1);
        if (key=="model_dir") dir=value;
        else if (key=="moe_intermediate_size") width=std::stoi(value);
        else if (key=="hidden_size") hidden=std::stoi(value);
    }
    CHECK(width>0 && hidden>0 && !dir.empty());
    struct Tensor { int file=-1; std::uint64_t offset=0, size=0; int rows=0, cols=0; };
    std::array<Tensor,3> tensors{}; std::vector<std::string> files;
    const auto prefix="layers."+std::to_string(entry.layer)+".ffn.experts."+std::to_string(entry.expert)+".w";
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
    for (int r=0;r<3;++r) {
        const auto& t=tensors[r]; CHECK(t.file>=0 && static_cast<std::size_t>(t.file)<files.size());
        CHECK(t.rows==(r==2?hidden:width) && t.cols==(r==2?width:hidden) && t.size<=(1ull<<30));
        source[r].resize(static_cast<std::size_t>(t.size));
        std::ifstream f(std::filesystem::path(dir)/files[t.file],std::ios::binary); CHECK(f.good());
        f.seekg(static_cast<std::streamoff>(t.offset));
        f.read(reinterpret_cast<char*>(source[r].data()),static_cast<std::streamsize>(t.size)); CHECK(f.good());
    }
    CHECK(tensors[0].size%width==0 && tensors[1].size%width==0 && tensors[2].size%hidden==0);
    std::array<Range,2> ranges; CHECK(expert_ranges(width,256,entry.layer,ranges,error));
    CHECK(entry.ff_begin==ranges[rank].begin && entry.ff_count==ranges[rank].count);
    ExpertSpans sp; CHECK(expert_spans(width,hidden,256,ranges[rank],tensors[0].size/width,
                                       tensors[1].size/width,tensors[2].size/hidden,sp,error));
    DeviceExpertLayout layout; CHECK(device_expert_layout(sp,layout,error));
    CHECK(layout.bytes==entry.bytes && layout.offsets==entry.offsets && entry.device!=nullptr);
    std::vector<std::uint8_t> expected(layout.bytes,0), got(layout.bytes);
    const std::array<ConstBytes,3> src{source[0],source[1],source[2]};
    CHECK(pack_expert(sp,src,{Bytes(expected).subspan(layout.offsets[0],sp.gate_bytes),
          Bytes(expected).subspan(layout.offsets[1],sp.up_bytes),
          Bytes(expected).subspan(layout.offsets[2],sp.down_row_bytes*sp.down_rows)},error));
    CHECK(hipMemcpy(got.data(),entry.device,got.size(),hipMemcpyDeviceToHost)==hipSuccess);
    CHECK(got==expected);
    std::printf("rank=%d layer=%d expert=%d: GPU bytes match source slices\n",rank,entry.layer,entry.expert);
    return true;
}

static bool check_bad_pack(const std::filesystem::path& pack, int rank, std::string& error) {
    char name[]="/tmp/ds4-rank-bad-XXXXXX";
    char* made=::mkdtemp(name); CHECK(made!=nullptr);
    const std::filesystem::path tmp=made;
    struct Cleanup { std::filesystem::path p; ~Cleanup() { std::error_code ec; std::filesystem::remove_all(p,ec); } } cleanup{tmp};
    std::error_code ec;
    std::filesystem::copy_file(pack/"ds4.txt",tmp/"ds4.txt",ec); CHECK(!ec);
    std::filesystem::copy_file(pack/"engram.bin",tmp/"engram.bin",ec); CHECK(!ec);
    std::ifstream input(pack/"tensors.txt"); CHECK(input.good());
    std::ostringstream original; original<<input.rdbuf(); const std::string full=original.str();
    const std::string prefix="t layers.0.ffn.experts.0.w1.weight ";
    const auto at=full.find(prefix); CHECK(at!=std::string::npos);
    const auto end=full.find('\n',at); CHECK(end!=std::string::npos);
    std::istringstream row(full.substr(at,end-at));
    std::string marker,tensor,dtype,file,offset,bytes,rows,cols;
    row>>marker>>tensor>>dtype>>file>>offset>>bytes>>rows>>cols; CHECK(!row.fail());
    for (int mode=0;mode<2;++mode) {
        const std::string replacement=marker+" "+tensor+" "+(mode==0?"BAD_FORMAT":dtype)+" "+file+" "+
            (mode==1?"18446744073709551615":offset)+" "+bytes+" "+rows+" "+cols;
        std::ofstream output(tmp/"tensors.txt",std::ios::trunc); CHECK(output.good());
        output<<full.substr(0,at)<<replacement<<full.substr(end); output.close(); CHECK(output.good());
        RankExpertLoader bad;
        CHECK(!bad.load(tmp.string(),0,rank,1ull<<30,1,error));
        CHECK(bad.size()==0 && bad.entry(0)==nullptr &&
              error.find(mode==0?"unsupported or missing tensor":"outside its file")!=std::string::npos);
    }
    return true;
}

#undef CHECK
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"line %d: %s (%s)\n",__LINE__,#x,error.c_str()); return 1; } } while (0)
int ds4_tp_rank_load_test(int argc, char** argv) {
    if (argc!=6) {
        std::fprintf(stderr,"usage: ds4_tp_verify_test --rank-load rank max-experts budget-GiB pack-dir\n"); return 2;
    }
    std::string error; char* end=nullptr;
    const long rank=std::strtol(argv[2],&end,10); if (*argv[2]==0 || *end || rank<0 || rank>1) return 2;
    if (*argv[3]=='-' || *argv[4]=='-') return 2;
    const auto count=std::strtoull(argv[3],&end,10); if (*argv[3]==0 || *end) return 2;
    const auto gib=std::strtoull(argv[4],&end,10); if (*argv[4]==0 || *end || gib==0 ||
        gib>std::numeric_limits<std::size_t>::max()/(1ull<<30) || count>std::numeric_limits<std::size_t>::max()) return 2;
    const auto budget=static_cast<std::size_t>(gib)*(std::size_t{1}<<30);
    RankExpertLoader invalid;
    CHECK(!invalid.load(argv[5],0,2,budget,1,error) && invalid.size()==0 && invalid.entry(0)==nullptr);
    RankExpertLoader too_small;
    CHECK(!too_small.load(argv[5],0,static_cast<int>(rank),1,1,error) && too_small.size()==0);
    CHECK(check_bad_pack(argv[5],static_cast<int>(rank),error));
    RankExpertLoader loader;
    CHECK(loader.load(argv[5],0,static_cast<int>(rank),budget,static_cast<std::size_t>(count),error,
                      [](std::size_t n,std::size_t total) { std::fprintf(stderr,"rank load %zu/%zu\n",n,total); }));
    CHECK(loader.size()>0 && loader.bytes()<=budget && loader.staging_bytes()>0 && loader.entry(loader.size())==nullptr);
    CHECK(check_entry(argv[5],*loader.entry(0),static_cast<int>(rank),error));
    if (loader.size()>1) CHECK(check_entry(argv[5],*loader.entry(loader.size()-1),static_cast<int>(rank),error));
    std::printf("rank=%ld loaded=%zu GPU_bytes=%zu pinned_staging=%zu\n",rank,loader.size(),loader.bytes(),loader.staging_bytes());
    return 0;
}
