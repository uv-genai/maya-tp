// Test-only bridge from independently checked preflight to the wire identity.
#pragma once

#include "../src/ds4/ds4_tp_identity.hpp"
#include <cstdlib>
#include <cerrno>

inline bool ds4_tp_test_identity(strata::ds4::tp2::Identity& identity, std::string& error) {
    const char* model=std::getenv("DS4_TP_TEST_MODEL_DIGEST");
    const char* partition=std::getenv("DS4_TP_TEST_PARTITION_DIGEST");
    const char* build=std::getenv("DS4_TP_TEST_BUILD_DIGEST");
    if (!model && !partition && !build) {
        identity.model_digest[0]=1; identity.partition_digest[0]=2; identity.build_digest[0]=3;
        return true; // Existing tests explicitly use synthetic identities.
    }
    if (!model || !partition || !build) { error="incomplete test preflight identities"; return false; }
    return strata::ds4::tp2::parse_digest(model,identity.model_digest,error) &&
           strata::ds4::tp2::parse_digest(partition,identity.partition_digest,error) &&
           strata::ds4::tp2::parse_digest(build,identity.build_digest,error);
}

inline bool ds4_tp_test_context(std::int64_t& context, std::string& error) {
    const char* value=std::getenv("DS4_TP_TEST_CONTEXT");
    if (!value) { context=32768; return true; }
    errno=0; char* end=nullptr; const long long n=std::strtoll(value,&end,10);
    if (!*value || !end || *end || errno || n<256 || n>1048576) {
        error="invalid test context"; return false;
    }
    context=n; return true;
}
