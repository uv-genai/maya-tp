// include/strata/net/tp_wire.hpp - the two-host TP wire identity: handshake, collective header, window header size.
//
// Byte-compatible with gufo (GUFT v7 / GUCH v5 / GURD): both ranks run the same ABI and exchange these structs
// raw over the bootstrap TCP connection and inside every RDMA write window, so the layouts, field order, and
// host-endian packing must not drift.  The only on-wire endianness conversion is the write's immediate data
// (htonl of the operation ordinal), which the verbs layer applies.
#pragma once

#include <infiniband/verbs.h>

#include <cstdint>

namespace strata::net {

inline constexpr std::uint32_t kWireMagic = 0x47554654U;  // "GUFT"
inline constexpr std::uint32_t kWireVersion = 7;
inline constexpr std::uint32_t kCollectiveMagic = 0x47554348U;  // "GUCH"
inline constexpr std::uint32_t kCollectiveVersion = 5;
inline constexpr std::uint32_t kReadyMagic = 0x47555244U;  // "GURD"
// A window starts with the writer's collective header; the partial follows at this offset.
inline constexpr std::size_t kDataOffset = 64;

/// QP metadata one rank sends the other over the bootstrap socket, then the
/// GURD readiness word that closes the handshake.
struct Wire {
  std::uint32_t magic{kWireMagic};
  std::uint32_t version{kWireVersion};
  std::uint32_t rank{0};
  std::uint32_t world_size{0};
  std::uint32_t qp_num{0};
  std::uint16_t lid{0};
  std::uint8_t mtu{0};
  std::uint8_t gid_index{0};
  std::uint8_t link_layer{0};
  std::uint16_t port_num{1};
  ibv_gid sgid{};
  std::uint64_t recv_address{0};
  std::uint32_t recv_rkey{0};
};

/// The header every exchange writes ahead of its partial; the peer validates
/// it so a collective cannot be silently associated with a different request.
struct CollectiveHeader {
  std::uint32_t magic{kCollectiveMagic};
  std::uint32_t version{kCollectiveVersion};
  std::uint64_t scope_id{0};
  std::uint64_t operation_id{0};
  std::uint64_t bytes{0};
};

static_assert(sizeof(CollectiveHeader) <= kDataOffset);

}  // namespace strata::net
