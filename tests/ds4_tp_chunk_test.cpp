#include "../src/ds4/ds4_tp_chunk.hpp"
#include <cstdio>
#include <limits>

using namespace strata::ds4::tp2;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"line %d: %s: %s\n",__LINE__,#x,error.c_str()); return 1; } } while (0)
int main() {
    std::string error; std::uint32_t rows=0;
    CHECK(chunk_rows(2048,5120,48ull<<20,rows,error) && rows==2048);
    CHECK(chunk_rows(4096,5120,48ull<<20,rows,error) && rows==2457);
    CHECK(chunk_rows(1,5120,48ull<<20,rows,error) && rows==16);
    rows=73;
    CHECK(!chunk_rows(16,5120,16ull*5120*4-1,rows,error) && rows==73);
    CHECK(!chunk_rows(0,5120,48ull<<20,rows,error) && rows==73);
    CHECK(!chunk_rows(16,0,48ull<<20,rows,error) && rows==73);
    CHECK(!chunk_rows(kControlMaxWords+1,5120,48ull<<20,rows,error) && rows==73);
    CHECK(chunk_rows(kControlMaxWords,1,std::numeric_limits<std::uint64_t>::max(),rows,error) && rows==kControlMaxWords);
    std::puts("DeepSeek TP2 row bounds: negotiated capacity and invalid inputs passed");
}
