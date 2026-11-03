// Two-host GEMV of one real rank-sliced attention matrix; not full attention.
#include "../src/ds4/ds4_tp_dense_loader.hpp"
#include "../src/ds4/ds4_tp_channel.hpp"
#include "../src/ds4/ds4_gguf.cuh"
#include "strata/net/tp_kernels.hpp"
#include "strata/net/verbs.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace strata::ds4::tp2;
namespace gg=strata::ds4::gguf;
namespace x3=strata::kernels::exl3;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"line %d: %s (%s)\n",__LINE__,#x,error.c_str()); return 1; } } while (0)

static x3::Mat matrix(const RankDenseEntry& e) {
    x3::Mat m; m.fmt=static_cast<int>(e.encoding); m.gg_type=e.gg_type;
    m.raw=e.device; m.n=e.n; m.k=e.k;
    m.ld=e.encoding==DenseEncoding::Bf16 ? e.k : static_cast<int>(e.row_bytes);
    return m;
}

int ds4_tp_dense_rdma_test(int argc,char** argv) {
    if (argc!=9) {
        std::fprintf(stderr,"usage: ds4_tp_verify_test --rdma-dense rank host port rdma pack input|output tensor\n"); return 2;
    }
    std::string error; char* end=nullptr;
    const long rank=std::strtol(argv[2],&end,10); if (*argv[2]==0 || *end || rank<0 || rank>1) return 2;
    const long port=std::strtol(argv[4],&end,10); if (*argv[4]==0 || *end || port<=0 || port>65535) return 2;
    const std::string mode=argv[7], name=argv[8];
    if (mode!="input" && mode!="output") return 2;
    const auto axis=mode=="input"?DenseAxis::Input:DenseAxis::Output;
    RankDenseLoader loader; CHECK(loader.load(argv[6],name,0,static_cast<int>(rank),axis,1ull<<30,error));
    const RankDenseEntry* e=loader.entry(); CHECK(e && e->device);
    hipStream_t stream=nullptr; CHECK(hipStreamCreate(&stream)==hipSuccess && gg::prepare(stream));
    std::vector<float> input(static_cast<std::size_t>(e->full_k));
    for (int i=0;i<e->full_k;++i) input[static_cast<std::size_t>(i)]=(static_cast<float>(i%41)-20.0f)*0.01f;
    float *dx=nullptr,*out=nullptr;
    CHECK(hipMalloc(&dx,input.size()*sizeof(float))==hipSuccess);
    CHECK(hipMalloc(&out,static_cast<std::size_t>(e->full_n)*sizeof(float))==hipSuccess);
    CHECK(hipMemcpyAsync(dx,input.data(),input.size()*sizeof(float),hipMemcpyHostToDevice,stream)==hipSuccess);
    x3::Mat local=matrix(*e); x3::MvJob job; job.m=&local; job.xf=dx+e->column_begin; job.y=out;
    CHECK(gg::mv(&job,1,stream));
    strata::net::IbrverbsConfig net; net.rank=static_cast<std::uint32_t>(rank); net.bootstrap_host=argv[3];
    net.bootstrap_port=static_cast<std::uint16_t>(port); net.rdma_device=argv[5];
    auto comm=strata::net::CreateIbrverbsCommunicator(net,&error); CHECK(comm);
    std::uint64_t fingerprint=1469598103934665603ull;
    for (unsigned char c:name) { fingerprint^=c; fingerprint*=1099511628211ull; }
    Identity id; id.protocol=1; id.world=2; id.rank=net.rank; id.hidden=e->full_n; id.layers=1;
    id.experts=1; id.expert_width=e->full_k; id.max_context=1;
    id.feature_flags=fingerprint^(axis==DenseAxis::Input?1ull:2ull); id.max_partial_bytes=comm->MaxPartialBytes();
    id.model_digest[0]=1; id.partition_digest[0]=2; id.build_digest[0]=3; // synthetic test identity only
    Identity peer; std::uint64_t cap=0; CHECK(exchange_identity(*comm,id,peer,cap,error));
    const std::size_t bytes=static_cast<std::size_t>(e->n)*sizeof(float); CHECK(bytes<=cap);
    CHECK(comm->BeginOperation(1,&error));
    const float* peer_part=comm->ExchangePartial(out,bytes,stream,&error); CHECK(peer_part);
    if (axis==DenseAxis::Input) {
        strata::net::AddRowsBroadcast(peer_part,out,1,static_cast<std::uint32_t>(e->n),1,stream);
        peer_part=comm->ExchangePartial(out,bytes,stream,&error); CHECK(peer_part);
    }
    std::vector<float> local_out(static_cast<std::size_t>(e->n)), peer_out(local_out.size());
    CHECK(hipMemcpyAsync(local_out.data(),out,bytes,hipMemcpyDeviceToHost,stream)==hipSuccess);
    CHECK(hipMemcpyAsync(peer_out.data(),peer_part,bytes,hipMemcpyDeviceToHost,stream)==hipSuccess);
    CHECK(hipStreamSynchronize(stream)==hipSuccess && comm->CheckQueued(&error) && comm->EndOperation(1,&error));
    float pair_delta=0,ref_delta=0;
    for (int i=0;i<e->n;++i) {
        CHECK(std::isfinite(local_out[i]) && std::isfinite(peer_out[i]));
        if (axis==DenseAxis::Input) pair_delta=std::max(pair_delta,std::fabs(local_out[i]-peer_out[i]));
    }
    if (axis==DenseAxis::Input) CHECK(pair_delta==0);
    if (rank==0) {
        RankDenseLoader whole; CHECK(whole.load(argv[6],name,0,0,DenseAxis::Whole,1ull<<30,error));
        const RankDenseEntry* f=whole.entry(); CHECK(f && f->n==e->full_n && f->k==e->full_k);
        x3::Mat full=matrix(*f); x3::MvJob ref; ref.m=&full; ref.xf=dx; ref.y=out;
        CHECK(gg::mv(&ref,1,stream));
        std::vector<float> expected(static_cast<std::size_t>(f->n));
        CHECK(hipMemcpyAsync(expected.data(),out,expected.size()*sizeof(float),hipMemcpyDeviceToHost,stream)==hipSuccess);
        CHECK(hipStreamSynchronize(stream)==hipSuccess);
        for (int i=0;i<f->n;++i) {
            const float got=axis==DenseAxis::Input?local_out[i]:i<e->n?local_out[i]:peer_out[i-e->n];
            CHECK(std::isfinite(expected[i]));
            const float delta=std::fabs(expected[i]-got); ref_delta=std::max(ref_delta,delta);
            CHECK(delta<=1e-3f+1e-3f*std::fabs(expected[i]));
        }
    }
    CHECK(gg::launch_errors()==0);
    Control done;
    if (rank==0) {
        CHECK(send_message(*comm,{Command::Shutdown,2,0,{}},16,error));
        CHECK(receive_message(*comm,16,done,error));
    } else {
        CHECK(receive_message(*comm,16,done,error));
        CHECK(send_message(*comm,{Command::Shutdown,2,0,{}},16,error));
    }
    CHECK(done.command==Command::Shutdown && done.sequence==2);
    CHECK(hipFree(dx)==hipSuccess && hipFree(out)==hipSuccess && hipStreamDestroy(stream)==hipSuccess);
    if (rank==0 && axis==DenseAxis::Input)
        std::printf("rank %ld: %s %s pair max_abs=%.8g full matrix max_abs=%.8g\n",rank,mode.c_str(),name.c_str(),pair_delta,ref_delta);
    else if (rank==0)
        std::printf("rank %ld: %s %s gathered output full matrix max_abs=%.8g\n",rank,mode.c_str(),name.c_str(),ref_delta);
    else
        std::printf("rank %ld: %s %s %s (reference on rank 0)\n",rank,mode.c_str(),name.c_str(),
                    axis==DenseAxis::Input?"reduced sums matched":"output sent");
    return 0;
}
