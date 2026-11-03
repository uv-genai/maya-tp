// One real GGUF expert split over two hosts, reduced over RDMA. Not a full model.
#include "../src/ds4/ds4_tp_rank_loader.hpp"
#include "../src/ds4/ds4_tp_channel.hpp"
#include "../src/ds4/ds4_gguf.cuh"
#include "strata/net/tp_kernels.hpp"
#include "strata/net/verbs.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace strata::ds4::tp2;
namespace gg=strata::ds4::gguf;
namespace x3=strata::kernels::exl3;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"line %d: %s (%s)\n",__LINE__,#x,error.c_str()); return 1; } } while (0)

static bool full_expert(const std::filesystem::path& pack, const RankExpertEntry& part, int dim, int width,
                        std::vector<std::uint8_t>& blob, x3::ExpertLayout& layout, std::string& error) {
    std::ifstream cfg(pack/"ds4.txt"); if (!cfg) { error="missing geometry"; return false; }
    std::string line,dir;
    while (std::getline(cfg,line)) if (line.rfind("model_dir ",0)==0) dir=line.substr(10);
    if (dir.empty()) { error="missing model_dir"; return false; }
    struct Tensor { int file=-1; std::uint64_t off=0, bytes=0; int rows=0,cols=0; };
    std::array<Tensor,3> t{}; std::vector<std::string> files;
    const auto prefix="layers."+std::to_string(part.layer)+".ffn.experts."+std::to_string(part.expert)+".w";
    std::ifstream index(pack/"tensors.txt");
    if (!index) { error="missing tensor index"; return false; }
    while (std::getline(index,line)) {
        std::istringstream s(line); std::string kind; s>>kind;
        if (kind=="file") {
            std::size_t i; std::string name; s>>i>>name;
            if (s.fail() || i>=10000) { error="bad file record"; return false; }
            if (files.size()<=i) files.resize(i+1); files[i]=name;
        } else if (kind=="t") {
            std::string name,dtype; s>>name>>dtype;
            if (name.size()!=prefix.size()+8 || name.compare(0,prefix.size(),prefix)!=0 ||
                name.substr(prefix.size()+1)!=".weight") continue;
            const int role=name[prefix.size()]-'0';
            if (role<1 || role>3) { error="bad expert role"; return false; }
            const int r=role==1?0:role==3?1:2;
            if (t[r].file!=-1 || dtype!=gg::type_name(part.gg_types[r])) { error="expert type mismatch"; return false; }
            s>>t[r].file>>t[r].off>>t[r].bytes>>t[r].rows>>t[r].cols;
            if (s.fail()) { error="bad expert record"; return false; }
        }
    }
    std::size_t total=0;
    layout={}; layout.fmt=gg::kFmtMXFP4;
    for (int r=0;r<3;++r) {
        const int rows=r<2?width:dim, cols=r<2?dim:width;
        const std::size_t bytes=static_cast<std::size_t>(rows)*gg::type_row_bytes(part.gg_types[r],cols);
        if (t[r].file<0 || static_cast<std::size_t>(t[r].file)>=files.size() || t[r].rows!=rows ||
            t[r].cols!=cols || t[r].bytes!=bytes || bytes==0) { error="bad expert geometry"; return false; }
        layout.gg_type[r]=part.gg_types[r]; layout.trellis[r]=total;
        total+=(bytes+255)&~std::size_t{255};
    }
    blob.assign(total,0);
    for (int r=0;r<3;++r) {
        std::ifstream f(std::filesystem::path(dir)/files[static_cast<std::size_t>(t[r].file)],std::ios::binary);
        if (!f) { error="cannot open source expert"; return false; }
        f.seekg(static_cast<std::streamoff>(t[r].off));
        f.read(reinterpret_cast<char*>(blob.data()+layout.trellis[r]),static_cast<std::streamsize>(t[r].bytes));
        if (!f) { error="cannot read source expert"; return false; }
    }
    return true;
}

int ds4_tp_expert_rdma_test(int argc, char** argv) {
    if (argc!=7 && argc!=9) {
        std::fprintf(stderr,"usage: ds4_tp_verify_test --rdma-expert rank bootstrap-host port rdma-device pack-dir [layer expert]\n"); return 2;
    }
    std::string error; char* end=nullptr;
    const long rank=std::strtol(argv[2],&end,10); if (*argv[2]==0 || *end || rank<0 || rank>1) return 2;
    const long port=std::strtol(argv[4],&end,10); if (*argv[4]==0 || *end || port<=0 || port>65535) return 2;
    std::ifstream cfg(std::filesystem::path(argv[6])/"ds4.txt"); CHECK(cfg.good());
    int dim=0,width=0,layers=0,n_experts=0; std::string line;
    while (std::getline(cfg,line)) {
        std::istringstream s(line); std::string key; s>>key;
        if (key=="hidden_size") s>>dim;
        if (key=="moe_intermediate_size") s>>width;
        if (key=="num_hidden_layers") s>>layers;
        if (key=="n_routed_experts") s>>n_experts;
    }
    CHECK(dim>0 && width>0 && layers>0 && n_experts>0);
    long layer=0,expert=0;
    if (argc==9) {
        layer=std::strtol(argv[7],&end,10); if (*argv[7]==0 || *end || layer<0 || layer>1 || layer>=layers) return 2;
        expert=std::strtol(argv[8],&end,10); if (*argv[8]==0 || *end || expert<0 || expert>=n_experts) return 2;
    }
    const auto selected=static_cast<std::size_t>(layer)*static_cast<std::size_t>(n_experts)+
                        static_cast<std::size_t>(expert);
    RankExpertLoader loader; CHECK(loader.load(argv[6],0,static_cast<int>(rank),8ull<<30,selected+1,error));
    const RankExpertEntry* part=loader.entry(selected); CHECK(part && part->layer==layer && part->expert==expert);
    hipStream_t stream=nullptr; CHECK(hipStreamCreate(&stream)==hipSuccess && gg::prepare(stream));
    std::vector<float> input(static_cast<std::size_t>(dim));
    for (int i=0;i<dim;++i) input[static_cast<std::size_t>(i)]=(static_cast<float>(i%41)-20.0f)*0.01f;
    float *dx=nullptr,*dh=nullptr,*out=nullptr,*zero=nullptr,*weight=nullptr;
    int* flag=nullptr; unsigned long long* ptr=nullptr;
    CHECK(hipMalloc(&dx,static_cast<std::size_t>(dim)*4)==hipSuccess);
    CHECK(hipMalloc(&dh,static_cast<std::size_t>(width)*4)==hipSuccess);
    CHECK(hipMalloc(&out,static_cast<std::size_t>(dim)*4)==hipSuccess);
    CHECK(hipMalloc(&zero,static_cast<std::size_t>(dim)*4)==hipSuccess);
    CHECK(hipMalloc(&weight,sizeof(float))==hipSuccess && hipMalloc(&flag,sizeof(int))==hipSuccess);
    CHECK(hipMalloc(&ptr,sizeof(*ptr))==hipSuccess);
    const float one=1.0f;
    CHECK(hipMemcpyAsync(dx,input.data(),input.size()*4,hipMemcpyHostToDevice,stream)==hipSuccess);
    CHECK(hipMemsetAsync(zero,0,input.size()*4,stream)==hipSuccess);
    CHECK(hipMemsetAsync(flag,0,sizeof(int),stream)==hipSuccess);
    CHECK(hipMemcpyAsync(weight,&one,sizeof(one),hipMemcpyHostToDevice,stream)==hipSuccess);
    auto launch=[&](const std::uint8_t* data,const x3::ExpertLayout& layout,int ff) {
        const unsigned long long address=reinterpret_cast<unsigned long long>(data);
        if (hipMemcpyAsync(ptr,&address,sizeof(address),hipMemcpyHostToDevice,stream)!=hipSuccess ||
            hipStreamSynchronize(stream)!=hipSuccess) return false; // address must outlive async copy
        x3::MoeArgs a; a.plan_ptr=ptr; a.plan_w=weight; a.cpu_part=zero; a.cpu_flag=flag;
        a.lay=layout; a.k=1; a.n_embd=dim; a.n_ff=ff; a.limit=10.0f;
        gg::moe_gate_up(a,dx,dh,stream); gg::moe_down(a,dh,zero,out,stream);
        return hipGetLastError()==hipSuccess;
    };
    x3::ExpertLayout local{}; local.fmt=gg::kFmtMXFP4;
    for (int r=0;r<3;++r) { local.trellis[r]=part->offsets[r]; local.gg_type[r]=part->gg_types[r]; }
    CHECK(launch(part->device,local,part->ff_count));
    strata::net::IbrverbsConfig net;
    net.rank=static_cast<std::uint32_t>(rank); net.bootstrap_host=argv[3];
    net.bootstrap_port=static_cast<std::uint16_t>(port); net.rdma_device=argv[5];
    auto comm=strata::net::CreateIbrverbsCommunicator(net,&error); CHECK(comm);
    Identity id; id.protocol=1; id.world=2; id.rank=net.rank; id.hidden=dim; id.layers=layers;
    id.experts=n_experts; id.expert_width=width; id.max_context=32768;
    id.feature_flags=(static_cast<std::uint64_t>(layer)<<32)|static_cast<std::uint64_t>(expert);
    id.max_partial_bytes=comm->MaxPartialBytes(); id.model_digest[0]=1; id.partition_digest[0]=2; id.build_digest[0]=3;
    Identity other; std::uint64_t cap=0; CHECK(exchange_identity(*comm,id,other,cap,error));
    const std::size_t bytes=static_cast<std::size_t>(dim)*sizeof(float); CHECK(bytes<=cap);
    CHECK(comm->BeginOperation(1,&error));
    const float* peer=comm->ExchangePartial(out,bytes,stream,&error); CHECK(peer);
    strata::net::AddRowsBroadcast(peer,out,1,static_cast<std::uint32_t>(dim),1,stream);
    const float* peer_sum=comm->ExchangePartial(out,bytes,stream,&error); CHECK(peer_sum);
    std::vector<float> sum(static_cast<std::size_t>(dim)), other_sum(sum.size());
    CHECK(hipMemcpyAsync(sum.data(),out,bytes,hipMemcpyDeviceToHost,stream)==hipSuccess);
    CHECK(hipMemcpyAsync(other_sum.data(),peer_sum,bytes,hipMemcpyDeviceToHost,stream)==hipSuccess);
    CHECK(hipStreamSynchronize(stream)==hipSuccess && comm->CheckQueued(&error) && comm->EndOperation(1,&error));
    float pair_delta=0;
    for (int i=0;i<dim;++i) {
        CHECK(std::isfinite(sum[i]) && std::isfinite(other_sum[i]));
        pair_delta=std::max(pair_delta,std::fabs(sum[i]-other_sum[i]));
    }
    CHECK(pair_delta==0);
    float ref_delta=0;
    if (rank==0) {
        std::vector<std::uint8_t> full; x3::ExpertLayout layout{};
        CHECK(full_expert(argv[6],*part,dim,width,full,layout,error));
        std::uint8_t* d_full=nullptr; CHECK(hipMalloc(&d_full,full.size())==hipSuccess);
        CHECK(hipMemcpyAsync(d_full,full.data(),full.size(),hipMemcpyHostToDevice,stream)==hipSuccess);
        CHECK(launch(d_full,layout,width));
        std::vector<float> reference(sum.size());
        CHECK(hipMemcpyAsync(reference.data(),out,bytes,hipMemcpyDeviceToHost,stream)==hipSuccess);
        CHECK(hipStreamSynchronize(stream)==hipSuccess && hipFree(d_full)==hipSuccess);
        for (int i=0;i<dim;++i) {
            CHECK(std::isfinite(reference[i]));
            const float delta=std::fabs(reference[i]-sum[i]); ref_delta=std::max(ref_delta,delta);
            CHECK(delta<=1e-3f+1e-3f*std::fabs(reference[i]));
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
    CHECK(hipFree(dx)==hipSuccess && hipFree(dh)==hipSuccess && hipFree(out)==hipSuccess &&
          hipFree(zero)==hipSuccess && hipFree(weight)==hipSuccess && hipFree(flag)==hipSuccess &&
          hipFree(ptr)==hipSuccess && hipStreamDestroy(stream)==hipSuccess);
    if (rank==0)
        std::printf("rank %ld: layer=%ld expert=%ld FFN=%d+peer, pair max_abs=%.8g, full expert max_abs=%.8g\n",
                    rank,layer,expert,part->ff_count,pair_delta,ref_delta);
    else
        std::printf("rank %ld: layer=%ld expert=%ld FFN=%d+peer, pair max_abs=%.8g (reference on rank 0)\n",
                    rank,layer,expert,part->ff_count,pair_delta);
    return 0;
}
