// Diagnostic: the existing one-host window and chunk paths on the same input.
#include "../src/ds4/ds4_model.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"line %d: %s (%s)\n",__LINE__,#x,error.c_str()); return 1; } } while (0)
int ds4_single_prefill_compare_test(int argc,char** argv) {
    if (argc!=4) { std::fprintf(stderr,"usage: --single-prefill-compare pack prompt-tokens\n"); return 2; }
    std::string error; std::vector<int> seed;
    { std::ifstream f(argv[3]); CHECK(f.good()); int t; while(f>>t) seed.push_back(t); CHECK(f.eof() && !seed.empty()); }
    CHECK(setenv("DS4_STREAM","0",1)==0 && setenv("DS4_FLOW","0",1)==0 &&
          setenv("DS4_YOCO","0",1)==0 && setenv("DS4_CHUNK_MIN","17",1)==0);
    strata::ds4::Options o; o.pack=argv[2]; o.max_context=32768; o.gpus={0}; o.reserve_mib=1024;
    o.tp=1; o.threads=false; o.vision=false; o.dspark=false; o.lookup=false; o.prefill_chunk=256;
    o.expert_cache=false; o.mmap_experts=1; o.profile_readonly=true; o.cpu_threads=14; o.vram_experts=1;
    strata::ds4::Model model; CHECK(model.load(o,error));
    std::vector<int> tokens;
    if (std::getenv("DS4_TP_TEST_USE_PROMPT_EXACT")) tokens=seed;
    else for (int i=0;i<512;++i) tokens.push_back(seed[static_cast<std::size_t>(i)%seed.size()]);
    const int length=static_cast<int>(tokens.size());
    CHECK(length>0 && length<=1024);
    for (int at=0;at<length;) {
        const int n=std::min(16,length-at);
        CHECK(model.feed(tokens.data()+at,n,at+n==length?1:0,error)); at+=n;
    }
    std::vector<float> window; CHECK(model.logits(window) && !window.empty());
    model.reset();
    for (int at=0;at<length;) {
        const int n=std::min(256,length-at);
        CHECK(model.feed(tokens.data()+at,n,at+n==length?1:0,error)); at+=n;
    }
    std::vector<float> chunk; CHECK(model.logits(chunk) && chunk.size()==window.size());
    float max=0; double square=0;
    for (std::size_t i=0;i<chunk.size();++i) {
        CHECK(std::isfinite(window[i]) && std::isfinite(chunk[i]));
        const float d=std::fabs(window[i]-chunk[i]); max=std::max(max,d); square+=double(d)*d;
    }
    const auto a=std::max_element(window.begin(),window.end())-window.begin();
    const auto b=std::max_element(chunk.begin(),chunk.end())-chunk.begin();
    std::printf("one-host %d tokens: window_top=%td chunk_top=%td max=%.8g rms=%.8g\n",
                length,a,b,max,std::sqrt(square/chunk.size()));
    if (const char* path=std::getenv("DS4_TP_TEST_LOGITS")) {
        std::ofstream f(path,std::ios::binary|std::ios::trunc); CHECK(f.good());
        f.write(reinterpret_cast<const char*>(window.data()),static_cast<std::streamsize>(window.size()*sizeof(float)));
        f.write(reinterpret_cast<const char*>(chunk.data()),static_cast<std::streamsize>(chunk.size()*sizeof(float)));
        CHECK(f.good());
    }
    return 0; // Diagnostic, not an assertion that the paths are equivalent.
}
