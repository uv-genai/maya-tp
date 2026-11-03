#include "../src/net/control_socket_state.hpp"
#include <cstdio>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

using namespace strata::net;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); return 1; } } while (0)
int main() {
    int fd[2]; CHECK(::socketpair(AF_UNIX, SOCK_STREAM, 0, fd) == 0);
    CHECK(control_socket_state(fd[0], 0) == ControlSocketState::quiet);
    const char command[] = "control";
    CHECK(::send(fd[1], command, sizeof(command), 0) == sizeof(command));
    pollfd p{fd[0], control_poll_events(false), 0};
    CHECK(::poll(&p, 1, 1000) == 1 && (p.revents & POLLIN));
    CHECK(control_socket_state(fd[0], p.revents) == ControlSocketState::pending_data);
    // Detection must not consume the command or spin on already pending data.
    p = {fd[0], control_poll_events(true), 0};
    CHECK(::poll(&p, 1, 0) == 0);
    char received[sizeof(command)]{};
    CHECK(::recv(fd[0], received, sizeof(received), 0) == sizeof(received));
    for (std::size_t i = 0; i < sizeof(command); ++i) CHECK(received[i] == command[i]);
    CHECK(::shutdown(fd[1], SHUT_WR) == 0);
    p = {fd[0], control_poll_events(false), 0};
    CHECK(::poll(&p, 1, 1000) == 1);
    CHECK(control_socket_state(fd[0], p.revents) == ControlSocketState::closed);
    ::close(fd[0]); ::close(fd[1]);
    CHECK(control_socket_state(-1, POLLNVAL) == ControlSocketState::closed);
#ifdef POLLRDHUP
    // Even with an unread control frame, a half-closed peer remains detectable
    // after POLLIN was suppressed to avoid a busy loop.
    CHECK(::socketpair(AF_UNIX, SOCK_STREAM, 0, fd) == 0);
    CHECK(::send(fd[1], command, sizeof(command), 0) == sizeof(command));
    CHECK(::shutdown(fd[1], SHUT_WR) == 0);
    p = {fd[0], control_poll_events(true), 0};
    CHECK(::poll(&p, 1, 1000) == 1);
    CHECK(control_socket_state(fd[0], p.revents) == ControlSocketState::closed);
    ::close(fd[0]); ::close(fd[1]);
#endif
    std::puts("control socket: data remains readable, no spin, peer close detected");
}
