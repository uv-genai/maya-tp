// gfx1151 integration test: real Lt tuning reads a store without writing it
// when paired execution marks it read-only, and still writes in normal mode.
#include "../src/ds4/ds4_lt.hpp"
#include <hip/hip_runtime_api.h>
#include <cstdio>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); return 1; } } while (0)

int main(int argc, char** argv) {
    if (argc != 2) { std::fprintf(stderr, "usage: ds4_lt_store_test NEW_TEMP_FILE\n"); return 2; }
    const int fd = ::open(argv[1], O_CREAT | O_EXCL | O_WRONLY, 0600);
    CHECK(fd >= 0); CHECK(::close(fd) == 0);
    CHECK(strata::ds4::lt::available());
    hipStream_t stream{}; CHECK(hipStreamCreate(&stream) == hipSuccess);
    void *w=nullptr, *x=nullptr, *y=nullptr;
    constexpr int k=512, rows=64, max_n=768;
    CHECK(hipMalloc(&w, max_n*k*sizeof(uint16_t)) == hipSuccess);
    CHECK(hipMalloc(&x, rows*k*sizeof(uint16_t)) == hipSuccess);
    CHECK(hipMalloc(&y, rows*max_n*sizeof(float)) == hipSuccess);
    CHECK(hipMemset(w, 0, max_n*k*sizeof(uint16_t)) == hipSuccess);
    CHECK(hipMemset(x, 0, rows*k*sizeof(uint16_t)) == hipSuccess);
    auto run=[&](int n) {
        return strata::ds4::lt::gemm(stream, static_cast<uint16_t*>(w), n, k,
                                      static_cast<uint16_t*>(x), k, rows,
                                      static_cast<float*>(y), n) && hipStreamSynchronize(stream)==hipSuccess;
    };
    struct stat st{};
    strata::ds4::lt::set_store(argv[1], true);
    CHECK(run(512)); CHECK(::stat(argv[1], &st)==0 && st.st_size==0);
    strata::ds4::lt::set_store(argv[1], false);
    CHECK(run(768)); CHECK(::stat(argv[1], &st)==0 && st.st_size>0);
    CHECK(hipFree(y)==hipSuccess); CHECK(hipFree(x)==hipSuccess); CHECK(hipFree(w)==hipSuccess);
    CHECK(hipStreamDestroy(stream)==hipSuccess);
    CHECK(::unlink(argv[1])==0);
    std::puts("HIP Lt read-only and writable store behavior passed");
    return 0;
}
