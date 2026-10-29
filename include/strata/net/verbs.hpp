// include/strata/net/verbs.hpp - two-host TP2 all-reduce over RoCE v2 RC queue pairs (Thunderbolt usb4_rdma).
//
// Ported from gufo's Ibrverbs, which measures ~37 Gb/s and ~6 µs per exchange on twin Strix Halo hosts: the
// partial travels as one RDMA write with immediate data into a fixed receive window the peer's GPU can read
// directly, and overlapped/queued exchanges keep the GPU waiting on a flag the NIC's completion updates rather
// than on a thread.  The TCP bootstrap carries only QP metadata and peer-loss detection - never the data path.
#pragma once

#include "strata/net/communicator.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace strata::net {

struct IbrverbsConfig {
  std::uint32_t rank{0};
  std::uint32_t world_size{2};
  /// TCP bootstrap address used by rank 1. Rank 0 binds to all interfaces.
  std::string bootstrap_host;
  std::uint16_t bootstrap_port{18515};
  /// HIP device index.
  std::uint32_t device_index{0};
  /// RDMA device (HCA) name; empty selects the host's only one.
  std::string rdma_device;
  std::uint32_t rdma_port{1};
  /// GID table index; unset uses 0 on InfiniBand and, on RoCE, the port's
  /// RoCE v2 GID (see SelectRoceV2Gid).
  std::optional<std::uint32_t> gid_index;
  /// Shared secret checked over the bootstrap socket before the handshake;
  /// empty (the default) skips the check and stays byte-compatible with
  /// gufo. Both ranks must set the same token or none. It gates the
  /// handshake; it does not encrypt anything.
  std::string auth_token;
};

/// Creates a two-rank RC communicator over native InfiniBand or RoCE v2; both
/// ranks must use the same link layer. Each exchange is one RDMA write with
/// immediate data of a header and the partial into the peer's next receive
/// window; completions arrive through an event-driven completion channel,
/// and the bootstrap socket stays open only to detect a lost peer. Overlapped
/// exchanges run on a background thread. Both processes map the same fixed
/// send and receive windows. Callers must bind the same control scope before
/// model execution; every collective carries and validates its operation
/// ordinal.
[[nodiscard]] std::shared_ptr<Communicator> CreateIbrverbsCommunicator(
    const IbrverbsConfig& config, std::string* error_msg);

}  // namespace strata::net
