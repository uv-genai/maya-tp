// Explicit real-pack validation of rank-local dense matrix bytes.
#include "../src/ds4/ds4_tp_dense_loader.hpp"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace strata::ds4::tp2;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"line %d: %s (%s)\n",__LINE__,#x,error.c_str()); return 1; } } while (0)

static bool bad_metadata(const std::filesystem::path& pack,const std::string& target,
                         DenseAxis axis,int rank,std::string& error) {
    char name[]="/tmp/ds4-dense-bad-XXXXXX";
    char* made=::mkdtemp(name); if (!made) { error="mkdtemp failed"; return false; }
    const std::filesystem::path tmp=made;
    struct Cleanup { std::filesystem::path p; ~Cleanup() { std::error_code ec; std::filesystem::remove_all(p,ec); } } cleanup{tmp};
    std::error_code ec;
    std::filesystem::copy_file(pack/"ds4.txt",tmp/"ds4.txt",ec); if (ec) { error=ec.message(); return false; }
    std::filesystem::copy_file(pack/"engram.bin",tmp/"engram.bin",ec); if (ec) { error=ec.message(); return false; }
    std::ifstream index(pack/"tensors.txt"); if (!index) { error="missing index"; return false; }
    std::ostringstream original; original<<index.rdbuf(); const auto full=original.str();
    const auto prefix="t "+target+" "; const auto at=full.find(prefix);
    if (at==std::string::npos) { error="missing target"; return false; }
    const auto end=full.find('\n',at); if (end==std::string::npos) { error="truncated record"; return false; }
    std::istringstream s(full.substr(at,end-at));
    std::string marker,tensor,dtype,file,offset,bytes,rows,cols;
    s>>marker>>tensor>>dtype>>file>>offset>>bytes>>rows>>cols;
    if (s.fail()) { error="invalid source record"; return false; }
    for (int mode=0;mode<2;++mode) {
        const std::string replacement=marker+" "+tensor+" "+(mode==0?"BAD_FORMAT":dtype)+" "+file+" "+
            (mode==1?"18446744073709551615":offset)+" "+bytes+" "+rows+" "+cols;
        std::ofstream out(tmp/"tensors.txt",std::ios::trunc);
        out<<full.substr(0,at)<<replacement<<full.substr(end); out.close();
        if (!out) { error="cannot write test index"; return false; }
        RankDenseLoader bad;
        if (bad.load(tmp.string(),target,0,rank,axis,1ull<<30,error) || bad.entry()!=nullptr ||
            error.find(mode==0?"unsupported matrix type":"file bounds") == std::string::npos) return false;
    }
    return true;
}

int ds4_tp_dense_load_test(int argc,char** argv) {
    if (argc!=6) {
        std::fprintf(stderr,"usage: ds4_tp_verify_test --dense-load rank whole|output|input pack-dir tensor-name\n"); return 2;
    }
    std::string error; char* end=nullptr; const long rank=std::strtol(argv[2],&end,10);
    if (*argv[2]==0 || *end || rank<0 || rank>1) return 2;
    const std::string mode=argv[3];
    if (mode!="whole" && mode!="output" && mode!="input") return 2;
    const auto axis=mode=="whole"?DenseAxis::Whole:mode=="output"?DenseAxis::Output:DenseAxis::Input;
    const std::filesystem::path pack=argv[4]; const std::string target=argv[5];
    RankDenseLoader invalid;
    CHECK(!invalid.load(pack.string(),target,0,2,axis,1ull<<30,error) && invalid.entry()==nullptr);
    RankDenseLoader too_small;
    CHECK(!too_small.load(pack.string(),target,0,static_cast<int>(rank),axis,1,error) && too_small.entry()==nullptr);
    CHECK(bad_metadata(pack,target,axis,static_cast<int>(rank),error));
    RankDenseLoader loader;
    CHECK(loader.load(pack.string(),target,0,static_cast<int>(rank),axis,1ull<<30,error));
    const RankDenseEntry* e=loader.entry(); CHECK(e && e->device && e->bytes>0);
    std::ifstream cfg(pack/"ds4.txt"); CHECK(cfg.good()); std::string line,dir;
    while (std::getline(cfg,line)) if (line.rfind("model_dir ",0)==0) dir=line.substr(10);
    CHECK(!dir.empty());
    std::ifstream index(pack/"tensors.txt"); CHECK(index.good());
    std::vector<std::string> files; int file=-1,rows=0,cols=0;
    std::uint64_t offset=0,bytes=0;
    while (std::getline(index,line)) {
        std::istringstream s(line); std::string kind; s>>kind;
        if (kind=="file") {
            std::size_t i; std::string name; s>>i>>name; CHECK(!s.fail() && i<10000);
            if (files.size()<=i) files.resize(i+1);
            files[i]=name;
        } else if (kind=="t") {
            std::string name,dtype; s>>name>>dtype; if (name!=target) continue;
            CHECK(file==-1); s>>file>>offset>>bytes>>rows>>cols; CHECK(!s.fail());
        }
    }
    CHECK(file>=0 && static_cast<std::size_t>(file)<files.size() && rows==e->full_n && cols==e->full_k &&
          rows>0 && bytes%static_cast<std::uint64_t>(rows)==0);
    const auto source_row=static_cast<std::size_t>(bytes/static_cast<std::uint64_t>(rows));
    CHECK(e->n==(axis==DenseAxis::Output?rows/2:rows) && e->k==(axis==DenseAxis::Input?cols/2:cols));
    CHECK(e->row_begin==(axis==DenseAxis::Output?rank*e->n:0) &&
          e->column_begin==(axis==DenseAxis::Input?rank*e->k:0));
    CHECK(e->row_bytes==(axis==DenseAxis::Input?source_row/2:source_row));
    std::ifstream src(std::filesystem::path(dir)/files[static_cast<std::size_t>(file)],std::ios::binary); CHECK(src.good());
    std::vector<std::uint8_t> expected(e->bytes),got(e->bytes);
    for (int r=0;r<e->n;++r) {
        const auto at=offset+static_cast<std::uint64_t>(e->row_begin+r)*source_row+
                      (axis==DenseAxis::Input?static_cast<std::size_t>(rank)*e->row_bytes:0);
        src.seekg(static_cast<std::streamoff>(at));
        src.read(reinterpret_cast<char*>(expected.data()+static_cast<std::size_t>(r)*e->row_bytes),
                 static_cast<std::streamsize>(e->row_bytes));
        CHECK(src.good());
    }
    CHECK(hipMemcpy(got.data(),e->device,e->bytes,hipMemcpyDeviceToHost)==hipSuccess && got==expected);
    std::printf("rank=%ld axis=%s %s n=%d k=%d GPU_bytes=%zu: exact source slice\n",
                rank,mode.c_str(),target.c_str(),e->n,e->k,e->bytes);
    return 0;
}
