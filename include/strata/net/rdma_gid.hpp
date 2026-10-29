// include/strata/net/rdma_gid.hpp - RoCE v2 GID choice for the Thunderbolt RDMA NIC when none is configured.
//
// A RoCE port's GID table holds one entry per address family the netdev has; picking the wrong one gives a QP
// that never reaches its peer.  The rule here is the port netdev's only IPv4 address, which is what the usb4_rdma
// links are addressed with.
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>

namespace strata::net {

enum class RdmaGidType { kInfiniBand, kRoceV1, kRoceV2 };

/// One entry of an RDMA port's GID table.
struct RdmaGidEntry {
  std::uint32_t index{0};
  RdmaGidType type{RdmaGidType::kInfiniBand};
  std::array<std::uint8_t, 16> gid{};
};

/// Chooses the GID a RoCE port uses when none is configured: the RoCE v2 GID
/// of the port netdev's only IPv4 address, or else the port's only RoCE v2
/// GID. Unset (all-zero) entries are skipped. Several candidates are
/// ambiguous and fail with their indices, as does a table without one.
[[nodiscard]] std::optional<std::uint32_t> SelectRoceV2Gid(
    std::span<const RdmaGidEntry> entries, std::string* error_msg);

}  // namespace strata::net
