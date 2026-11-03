#include "../src/ds4/ds4_tp_preflight.hpp"
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <unistd.h>

using namespace strata::ds4::tp2;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"line %d: %s: %s\n",__LINE__,#x,error.c_str()); return 1; } } while (0)
int main() {
    std::string error; Identity id;
    char path[]="/tmp/ds4-tp-preflight-XXXXXX";
    CHECK(mkdtemp(path)!=nullptr);
    const std::filesystem::path root(path);
    std::filesystem::create_directory(root/"one");
    std::filesystem::create_directory(root/"two");
    std::filesystem::create_directory_symlink(root/"one",root/"link");
    const std::string a(64,'a'), b(64,'b'), c(64,'c');
    CHECK(preflight_identity((root/"one").string(),(root/"link").c_str(),a.c_str(),b.c_str(),c.c_str(),id,error));
    CHECK(id.model_digest[0]==0xaa && id.partition_digest[0]==0xbb && id.build_digest[0]==0xcc);
    const auto saved=id;
    CHECK(!preflight_identity((root/"one").string(),(root/"two").c_str(),a.c_str(),b.c_str(),c.c_str(),id,error));
    CHECK(id.model_digest==saved.model_digest);
    CHECK(!preflight_identity((root/"one").string(),nullptr,a.c_str(),b.c_str(),c.c_str(),id,error));
    CHECK(!preflight_identity((root/"one").string(),(root/"one").c_str(),a.c_str(),"invalid",c.c_str(),id,error));
    CHECK(id.partition_digest==saved.partition_digest);
    std::filesystem::remove_all(root);
    std::puts("DeepSeek TP2 preflight: identity parsing and pack binding passed");
}
