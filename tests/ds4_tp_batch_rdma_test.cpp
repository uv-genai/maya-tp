// Pack-free owned protocol gate with actual queued GPU/RDMA payloads.
#include "../src/ds4/ds4_tp_controller.hpp"
#include "../src/ds4/ds4_tp_worker.hpp"
#include "strata/net/verbs.hpp"
#include "strata/net/tp_kernels.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace strata::ds4::tp2;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"line %d: %s (%s)\n",__LINE__,#x,error.c_str()); return 1; } } while(0)
int ds4_tp_batch_rdma_test(int argc,char** argv) {
    if (argc!=6) { std::fprintf(stderr,"usage: --batch-protocol rank host port rdma\n");return 2; }
    std::string error;char* end=nullptr;
    const long rank=std::strtol(argv[2],&end,10);if (!*argv[2] || *end || rank<0 || rank>1) return 2;
    const long port=std::strtol(argv[4],&end,10);if (!*argv[4] || *end || port<1 || port>65535) return 2;
    strata::net::IbrverbsConfig config;config.rank=static_cast<std::uint32_t>(rank);
    config.bootstrap_host=argv[3];config.bootstrap_port=static_cast<std::uint16_t>(port);config.rdma_device=argv[5];
    auto comm=strata::net::CreateIbrverbsCommunicator(config,&error);CHECK(comm);
    Identity id;id.world=2;id.rank=config.rank;id.hidden=5120;id.layers=40;id.experts=384;id.expert_width=2304;
    id.max_context=128;id.max_partial_bytes=comm->MaxPartialBytes();
    id.model_digest[0]=1;id.partition_digest[0]=2;id.build_digest[0]=3;
    CHECK(configure_batch_identity(id,4,error));Identity peer;std::uint64_t limit=0;
    CHECK(exchange_identity(*comm,id,peer,limit,error));CHECK(limit>=kBatchRows*sizeof(float));
    hipStream_t stream=nullptr;float* device=nullptr;
    CHECK(hipStreamCreate(&stream)==hipSuccess && hipMalloc(&device,kBatchRows*sizeof(float))==hipSuccess);
    int exchanges=0;
    const auto handle=[&](const Control& c,std::string& why) {
        if (c.command!=Command::Rows && c.command!=Command::BatchVerify) return true;
        BatchPlan plan;if (!decode_batch(c.words,{4,100,128},c.command==Command::Rows,plan,why)) return false;
        std::size_t rows=0;for (const auto& m:plan.members) rows+=m.tokens.size();
        std::vector<float> values(rows);
        for (std::size_t i=0;i<rows;++i) values[i]=static_cast<float>(rank*1000+i+1);
        if (hipMemcpyAsync(device,values.data(),rows*sizeof(float),hipMemcpyHostToDevice,stream)!=hipSuccess) return false;
        strata::net::Communicator::QueuedPartial partial;
        if (!comm->QueuePartial(rows*sizeof(float),&partial,&why)) return false;
        strata::net::StagePartial(device,partial.send,rows,partial.arrivals,partial.ready,partial.value,stream);
        // Exercise the accepted model path, including all rows. WaitAdd is
        // currently unused by the engines and has a separate row/stream bug.
        strata::net::WaitValue(partial.arrived,partial.value,partial.wait_ticks,partial.timed_out,stream);
        strata::net::AddRowsBroadcast(partial.peer,device,static_cast<std::uint32_t>(rows),1,1,stream);
        if (hipMemcpyAsync(values.data(),device,rows*sizeof(float),hipMemcpyDeviceToHost,stream)!=hipSuccess ||
            hipStreamSynchronize(stream)!=hipSuccess || !comm->CheckQueued(&why)) return false;
        for (std::size_t i=0;i<rows;++i) if (values[i]!=static_cast<float>(1000+2*(i+1))) {
            why="queued batch RDMA sum mismatch";return false;
        }
        ++exchanges;return true;
    };
    if (rank==1) {
        Worker worker(*comm,128,100,16,4);
        while (!worker.state().stopped()) CHECK(worker.run_one(handle,error));
        CHECK(worker.state().generation(2)==2 && worker.state().position(2)==1 && worker.state().position(0)==3);
    } else {
        Controller controller(*comm,128,100,16,4);
        const auto run=[&](Command command,std::uint64_t position,std::vector<std::uint32_t> words) {
            return controller.run(command,position,std::move(words),handle,error);
        };
        for (std::uint32_t slot=0;slot<4;++slot) CHECK(run(Command::BeginSlot,0,{slot,1,0}));
        BatchPlan plan{{{0,1,0,{1,2,3,4}},{1,1,0,{5,6,7}},{2,1,0,{8,9}},{3,1,0,{10}}}};
        std::vector<std::uint32_t> words;CHECK(encode_batch(plan,{4,100,128},false,words,error));
        CHECK(run(Command::BatchVerify,0,words));
        BatchCommit commit{5,{{0,1,0,true,{99}},{1,1,0,true,{6,99}},{2,1,0,true,{9,99}},{3,1,0,true,{99}}}};
        CHECK(encode_commit(commit,plan,5,{4,100,128},words,error));CHECK(run(Command::BatchCommit,0,words));
        BatchPlan ordinary;
        for (std::uint32_t slot=0;slot<4;++slot)
            ordinary.members.push_back({slot,1,controller.state().position(slot),{99}});
        CHECK(encode_batch(ordinary,{4,100,128},true,words,error));CHECK(run(Command::Rows,0,words));
        CHECK(run(Command::Bind,3,{2,1,0}));CHECK(run(Command::Reset,0,{}));CHECK(run(Command::BeginSlot,0,{2,2,0}));
        ordinary.members={{0,1,2,{9}},{2,2,0,{11}}};
        CHECK(encode_batch(ordinary,{4,100,128},true,words,error));CHECK(run(Command::Rows,0,words));
        CHECK(run(Command::Shutdown,1,{}));CHECK(controller.state().stopped());
    }
    CHECK(exchanges==3);CHECK(hipFree(device)==hipSuccess && hipStreamDestroy(stream)==hipSuccess);
    std::uint32_t done=1,other=0;
    CHECK(comm->SendControl(&done,sizeof(done),&error) && comm->RecvControl(&other,sizeof(other),&error) && other==1);
    std::printf("rank %ld: checked batch protocol, ragged commit/reuse and 3 queued RDMA sums passed\n",rank);
    return 0;
}
#undef CHECK
