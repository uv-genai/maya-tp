// New-mode transaction acknowledgements, bounded bootstrap/control metadata.
#pragma once
#include "ds4_tp_control.hpp"
#include "strata/net/communicator.hpp"
#include <algorithm>
#include <array>
#include <string>
#include <vector>

namespace strata::ds4::tp2 {
inline constexpr std::uint32_t kBatchAckMagic=0x4434414bu;
enum class AckStage : std::uint32_t { Accepted=1,Completed=2 };
inline bool send_batch_ack(strata::net::Communicator& comm,std::uint64_t sequence,AckStage stage,
                           const std::string& cause,std::string& error) {
    const auto count=std::min<std::size_t>(cause.size(),1024);
    std::vector<std::uint8_t> header;
    put32(header,kBatchAckMagic);put32(header,static_cast<std::uint32_t>(stage));put64(header,sequence);
    put32(header,static_cast<std::uint32_t>(count));
    return comm.SendControl(header.data(),header.size(),&error) &&
           (count==0 || comm.SendControl(cause.data(),count,&error));
}
inline bool receive_batch_ack(strata::net::Communicator& comm,std::uint64_t sequence,AckStage stage,
                              std::string& error) {
    std::array<std::uint8_t,20> header{};
    if (!comm.RecvControl(header.data(),header.size(),&error)) return false;
    if (get32(header,0)!=kBatchAckMagic || get32(header,4)!=static_cast<std::uint32_t>(stage) ||
        get64(header,8)!=sequence || get32(header,16)>1024) {
        error="ds4 tp2: invalid batch acknowledgement";return false;
    }
    const auto count=get32(header,16);
    if (count!=0) {
        std::string cause(count,'\0');
        if (!comm.RecvControl(cause.data(),count,&error)) return false;
        error=std::move(cause);return false;
    }
    error.clear();return true;
}
} // namespace strata::ds4::tp2
