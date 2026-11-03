// Validate the preflight identities handed to a DeepSeek rank by its launcher.
#pragma once

#include "ds4_tp_identity.hpp"
#include <filesystem>
#include <string>

namespace strata::ds4::tp2 {

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
