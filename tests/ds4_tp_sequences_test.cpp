// Paired TP2 state/capacity gate, explicitly invoked with real XS/S packs.
#include "../src/ds4/ds4_model.hpp"
#include "../src/ds4/ds4_tp_identity.hpp"
#include "../src/ds4/ds4_tp_channel.hpp"
#include "strata/net/verbs.hpp"
#include "ds4_tp_test_identity.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"line %d: %s (%s)\n",__LINE__,#x,error.c_str()); return 1; } } while (0)
int ds4_tp_sequences_test(int argc,char** argv) {
    if (argc!=7) { std::fprintf(stderr,"usage: --sharded-sequences rank host port rdma pack\n"); return 2; }
    std::string error; char* end=nullptr;
    const long rank=std::strtol(argv[2],&end,10); if (!*argv[2] || *end || rank<0 || rank>1) return 2;
    const long port=std::strtol(argv[4],&end,10); if (!*argv[4] || *end || port<1 || port>65535) return 2;
    const char* ns=std::getenv("DS4_TP_TEST_SEQS");
    const long sequences=ns?std::strtol(ns,&end,10):4;
    if (sequences<1 || sequences>4 || (ns && (!*ns || *end))) return 2;
    std::int64_t context=0; CHECK(ds4_tp_test_context(context,error));
    // Window/compressor boundary at 512: the three-row verify crosses it,
    // while retaining only its anchor must roll the rejected rows back.
    const int prompt_length=509;
    CHECK(context>prompt_length+32);
    CHECK(setenv("DS4_STREAM","0",1)==0 && setenv("DS4_FLOW","0",1)==0 && setenv("DS4_YOCO","0",1)==0);
    strata::ds4::Options o; o.pack=argv[6]; o.max_context=context; o.gpus={0}; o.reserve_mib=1024;
    o.tp=1; o.threads=false; o.vision=false; o.dspark=rank==0; o.lookup=false;
    o.prefill_chunk=256; o.tp2_chunk_enabled=true; o.expert_cache=false; o.mmap_experts=1;
    o.profile_readonly=true; o.cpu_threads=14; o.vram_experts=rank==0?1:0;
    o.tp2_rank=static_cast<int>(rank); o.tp2_expert_budget=90ull<<30; o.tp_worker_verify=true;
    o.seqs=static_cast<int>(sequences);
    strata::ds4::Model model; CHECK(model.load(o,error));
    CHECK(model.seq_count()==sequences && model.tp2_sequence_layout(error));
    strata::ds4::Tp2Capacity capacity; CHECK(model.tp2_capacity(capacity,error));
    const auto tiers=model.tiers(); CHECK(tiers.cpu==0 && capacity.free>=(1ull<<30));
    std::fprintf(stderr,"capacity rank=%ld slots=%ld context=%lld weights=%llu caches=%llu scratch=%llu "
                        "experts=%llu extra_caches=%llu staging=%llu free=%llu total=%llu available=%llu\n",
                 rank,sequences,(long long)context,(unsigned long long)capacity.weights,
                 (unsigned long long)capacity.caches,(unsigned long long)capacity.scratch,
                 (unsigned long long)capacity.experts,(unsigned long long)capacity.extra_sequence_caches,
                 (unsigned long long)capacity.pinned_staging,(unsigned long long)capacity.free,
                 (unsigned long long)capacity.total,(unsigned long long)capacity.system_available);
    strata::net::IbrverbsConfig cfg; cfg.rank=static_cast<std::uint32_t>(rank); cfg.bootstrap_host=argv[3];
    cfg.bootstrap_port=static_cast<std::uint16_t>(port); cfg.rdma_device=argv[5];
    auto comm=strata::net::CreateIbrverbsCommunicator(cfg,&error); CHECK(comm);
    strata::ds4::tp2::Identity identity,peer; const auto g=model.tp2_geometry();
    CHECK(ds4_tp_test_identity(identity,error));
    identity.protocol=1; identity.world=2; identity.rank=cfg.rank;
    identity.hidden=g.hidden; identity.layers=g.layers; identity.experts=g.experts; identity.expert_width=g.expert_width;
    identity.max_context=context; identity.feature_flags=(static_cast<std::uint64_t>(sequences)<<48)|1ull;
    identity.max_partial_bytes=comm->MaxPartialBytes(); std::uint64_t limit=0;
    CHECK(strata::ds4::tp2::exchange_identity(*comm,identity,peer,limit,error));
    CHECK(model.tp2_attach(*comm,static_cast<std::size_t>(limit),error,false));
    std::uint64_t operation=0;
    const auto execute=[&](const auto& work) {
        const auto id=++operation;
        return comm->BeginOperation(id,&error) && work() && model.tp_worker_synchronize(error) &&
               comm->CheckQueued(&error) && comm->EndOperation(id,&error);
    };
    const auto bind=[&](int s) { model.seq_bind(s); return model.seq_bound()==s; };
    const auto feed=[&](const std::vector<int>& ts) { return model.feed(ts.data(),static_cast<int>(ts.size()),0,error); };
    const auto equal=[&](const std::vector<float>& a,const std::vector<float>& b) {
        if (a.size()!=b.size() || a.empty()) return false;
        float worst=0;
        for (std::size_t i=0;i<a.size();++i) {
            if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return false;
            worst=std::max(worst,std::abs(a[i]-b[i]));
        }
        std::fprintf(stderr,"slot logits max_abs=%g\n",worst); return worst<=1e-5f;
    };
    std::vector<int> prompt(static_cast<std::size_t>(prompt_length));
    for (int i=0;i<prompt_length;++i) prompt[static_cast<std::size_t>(i)]=5+i%3;
    std::vector<float> initial;
    for (int s=0;s<sequences;++s) {
        CHECK(bind(s)); CHECK(execute([&] { return model.tp_worker_reset(error) && feed(prompt) && model.step(8,true,error); }));
        std::vector<float> actual; CHECK(model.logits(actual));
        if (s==0) initial=actual; else CHECK(equal(initial,actual));
        CHECK(model.seq_pos(s)==prompt_length+1);
    }
    CHECK(bind(0)); CHECK(execute([&] { return model.step(9,false,error) && model.step(12,true,error); }));
    std::vector<float> reference; CHECK(model.logits(reference));
    std::vector<int> reference_drafts; std::vector<float> confidence;
    if (rank==0) CHECK(model.draft(13,reference_drafts,confidence,error) && !reference_drafts.empty());
    // Match the reference's chunk/token boundary exactly: folding token 8
    // into the final prefill chunk changes the accepted prompt arithmetic.
    CHECK(execute([&] { return model.tp_worker_reset(error) && feed(prompt) && model.step(8,false,error); }));
    for (int s=static_cast<int>(sequences)-1;s>=0;--s) {
        CHECK(bind(s)); const int tokens[]{9,10,11};
        CHECK(execute([&] { return model.tp_verify_begin(tokens,3,error); }));
        CHECK(execute([&] { return model.tp_verify_commit(0,error); }));
        CHECK(model.seq_pos(s)==prompt_length+2);
        CHECK(execute([&] { return model.step(12,true,error); }));
        std::vector<float> actual; CHECK(model.logits(actual)); CHECK(equal(reference,actual));
        if (rank==0) {
            std::vector<int> drafts; CHECK(model.draft(13,drafts,confidence,error)); CHECK(drafts==reference_drafts);
        }
        for (int other=0;other<sequences;++other)
            CHECK(model.seq_pos(other)==prompt_length+(other>=s?3:1));
    }
    CHECK(model.tp2_sequence_layout(error)); CHECK(model.tp_worker_synchronize(error));
    CHECK(comm->CheckQueued(&error));
    std::uint32_t done=1,other=0;
    CHECK(comm->SendControl(&done,sizeof(done),&error) && comm->RecvControl(&other,sizeof(other),&error) && other==1);
    std::printf("{\"passed\":true,\"rank\":%ld,\"slots\":%ld,\"context\":%lld,\"weights\":%llu,\"caches\":%llu,"
                "\"scratch\":%llu,\"experts\":%llu,\"extra_sequence_caches\":%llu,\"pinned_staging\":%llu,"
                "\"free\":%llu,\"total\":%llu,\"system_available\":%llu}\n",rank,sequences,(long long)context,
                (unsigned long long)capacity.weights,(unsigned long long)capacity.caches,
                (unsigned long long)capacity.scratch,(unsigned long long)capacity.experts,
                (unsigned long long)capacity.extra_sequence_caches,(unsigned long long)capacity.pinned_staging,
                (unsigned long long)capacity.free,(unsigned long long)capacity.total,
                (unsigned long long)capacity.system_available);
    return 0;
}
#undef CHECK
