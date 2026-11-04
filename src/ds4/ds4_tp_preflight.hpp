// Validate the preflight identities handed to a DeepSeek rank by its launcher.
#pragma once

#include "ds4_tp_identity.hpp"
#include <cstring>
#include <filesystem>
#include <string>

namespace strata::ds4::tp2 {

inline bool preflight_chunk(const char* requested, int option, bool& enabled, std::string& error) {
    if (!requested || (std::strcmp(requested,"0")!=0 && std::strcmp(requested,"256")!=0)) {
        error="ds4 tp2: invalid launcher prefill chunk"; return false;
    }
    const int rows=std::strcmp(requested,"256")==0 ? 256 : 0;
    if (option!=rows) { error="ds4 tp2: prefill chunk differs from preflight"; return false; }
    enabled=rows!=0; error.clear(); return true;
}

inline bool preflight_lookup(const char* requested, int option, int rank, int& mode, std::string& error) {
    if (!requested || (std::strcmp(requested,"0")!=0 && std::strcmp(requested,"3")!=0) || rank<0 || rank>1) {
        error="ds4 tp2: invalid launcher prompt lookup"; return false;
    }
    const int value=std::strcmp(requested,"3")==0 ? 3 : 0;
    if (option!=(rank==0 ? value : 0)) {
        error="ds4 tp2: prompt lookup differs from preflight"; return false;
    }
    mode=value; error.clear(); return true;
}

// The launcher must check the manifest, hash the actual executable, and bind
// these values to the pack immediately before exec. This function rejects
// accidental omissions or a different pack; it is not an authentication
// mechanism and does not eliminate a filesystem race after preflight.
inline bool preflight_identity(const std::string& pack, const char* prepared_pack, const char* model,
                               const char* partition, const char* build, Identity& identity, std::string& error) {
    if (pack.empty() || !prepared_pack || !*prepared_pack || !model || !partition || !build) {
        error="ds4 tp2: missing launcher preflight identity"; return false;
    }
    std::error_code ec;
    const bool same=std::filesystem::equivalent(pack,prepared_pack,ec);
    if (ec || !same || !std::filesystem::is_directory(pack,ec) || ec) {
        error="ds4 tp2: checked pack differs from requested pack"; return false;
    }
    Identity out=identity;
    if (!parse_digest(model,out.model_digest,error) || !parse_digest(partition,out.partition_digest,error) ||
        !parse_digest(build,out.build_digest,error)) return false;
    identity=out; error.clear(); return true;
}

} // namespace strata::ds4::tp2
