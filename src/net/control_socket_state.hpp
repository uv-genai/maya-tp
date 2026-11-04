// Classify bootstrap/control socket poll events without consuming control data.
#pragma once

#include <cerrno>
#include <poll.h>
#include <sys/socket.h>

namespace strata::net {

enum class ControlSocketState { quiet, pending_data, closed };

inline short control_poll_events(bool data_pending) {
    short events = data_pending ? 0 : POLLIN;
#ifdef POLLRDHUP
    events |= POLLRDHUP;
#endif
    return events;
}

inline ControlSocketState control_socket_state(int fd, short events) {
    short dead = POLLERR | POLLHUP | POLLNVAL;
#ifdef POLLRDHUP
    dead |= POLLRDHUP;
#endif
    if (events & dead) return ControlSocketState::closed;
    if (!(events & POLLIN)) return ControlSocketState::quiet;
    char byte;
    const auto n = ::recv(fd, &byte, 1, MSG_PEEK | MSG_DONTWAIT);
    if (n > 0) return ControlSocketState::pending_data;
    if (n == 0) return ControlSocketState::closed;
    if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) return ControlSocketState::quiet;
    return ControlSocketState::closed;
}

}  // namespace strata::net
