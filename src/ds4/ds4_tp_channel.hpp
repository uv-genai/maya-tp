// Typed DeepSeek control channel over an already-connected TP communicator.
// Tensor payloads do not pass through this channel.
#pragma once

#include "ds4_tp_identity.hpp"
#include "strata/net/communicator.hpp"

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace strata::ds4::tp2 {

inline bool exchange_identity(strata::net::Communicator& comm, const Identity& local, Identity& peer,
                              std::uint64_t& partial_limit, std::string& error) {
    if (comm.world_size() != 2 || comm.rank() > 1 || local.rank != comm.rank()) {
        error = "ds4 tp2: communicator rank does not match local identity"; return false;
    }
    const auto send = encode_identity(local);
    std::array<std::uint8_t, kIdentityBytes> received{};
    // Opposite orders allow a bounded control socket to work even if a write
    // blocks; both sides always finish the exchange before validating.
    if (comm.rank() == 0) {
        if (!comm.SendControl(send.data(), send.size(), &error) ||
            !comm.RecvControl(received.data(), received.size(), &error)) return false;
    } else {
        if (!comm.RecvControl(received.data(), received.size(), &error) ||
            !comm.SendControl(send.data(), send.size(), &error)) return false;
    }
    Identity other;
    if (!decode_identity(received, other, error) || !agree_identity(local, other, partial_limit, error)) return false;
    peer = other;
    return true;
}

inline bool send_message(strata::net::Communicator& comm, const Control& message,
                         std::uint32_t max_words, std::string& error) {
    std::vector<std::uint8_t> b;
    if (!encode_control(message, max_words, b, error)) return false;
    return comm.SendControl(b.data(), b.size(), &error);
}

inline bool receive_message(strata::net::Communicator& comm, std::uint32_t max_words,
                            Control& message, std::string& error) {
    std::array<std::uint8_t, kControlHeaderBytes> header{};
    if (!comm.RecvControl(header.data(), header.size(), &error)) return false;
    std::size_t frame_bytes = 0;
    if (!control_frame_size(header, max_words, frame_bytes, error)) return false;
    std::vector<std::uint8_t> b(frame_bytes);
    std::copy(header.begin(), header.end(), b.begin());
    if (frame_bytes > header.size() &&
        !comm.RecvControl(b.data() + header.size(), frame_bytes - header.size(), &error)) return false;
    return decode_control(b, max_words, message, error);
}

}  // namespace strata::ds4::tp2
