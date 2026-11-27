#include <arpa/inet.h>
#include <cerrno>
#include <csignal>
#include <cstddef>
#include <cstdlib>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <poll.h>
#include <spawn.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

extern char** environ;

namespace {
struct Fd {
    explicit Fd(int value) : fd(value) {
        if (fd < 0) throw std::runtime_error("open socket or accept failed");
    }
    ~Fd() { close(fd); }
    Fd(const Fd&) = delete;
    Fd& operator=(const Fd&) = delete;
    int fd;
};

struct Work {
    Work() {
        auto candidate = (std::filesystem::temp_directory_path() / "spec-loopback-XXXXXX").string();
        if (!mkdtemp(candidate.data())) throw std::runtime_error("create temporary directory failed");
        path = candidate;
    }
    ~Work() {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }
    std::filesystem::path path;
};

void wait_readable(int fd) {
    pollfd descriptor{fd, POLLIN, 0};
    int result;
    do { result = poll(&descriptor, 1, 5000); } while (result < 0 && errno == EINTR);
    if (result <= 0) throw std::runtime_error("loopback timed out");
}

std::string request_from(int fd) {
    std::string request;
    std::size_t needed = 0;
    while (request.size() < 16384) {
        wait_readable(fd);
        char bytes[4096];
        const auto count = recv(fd, bytes, sizeof(bytes), 0);
        if (count <= 0) throw std::runtime_error("incomplete loopback request");
        request.append(bytes, static_cast<std::size_t>(count));
        const auto header_end = request.find("\r\n\r\n");
        if (header_end == std::string::npos) continue;
        if (needed == 0) {
            const std::string_view name = "Content-Length: ";
            const auto field = request.find(name);
            if (field == std::string::npos || field >= header_end) {
                throw std::runtime_error("Content-Length header missing");
            }
            needed = header_end + 4 + std::stoul(request.substr(field + name.size()));
        }
        if (request.size() >= needed) return request;
    }
    throw std::runtime_error("loopback request exceeded fixture bound");
}

void send_response(int fd, int status, const std::string& body) {
    struct timeval timeout {5, 0};
    if (setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) != 0) {
        throw std::runtime_error("cannot set loopback send timeout");
    }
    const auto headers = "HTTP/1.1 " + std::to_string(status) + " Fixture\r\n"
        "Content-Type: application/json\r\nConnection: close\r\nContent-Length: "
        + std::to_string(body.size()) + "\r\n\r\n";
    for (auto remaining : {std::string_view(headers), std::string_view(body)}) {
        while (!remaining.empty()) {
            const auto count = send(fd, remaining.data(), remaining.size(), 0);
            if (count < 0 && errno == EINTR) continue;
            if (count <= 0) return;
            remaining.remove_prefix(static_cast<std::size_t>(count));
        }
    }
}

std::string read(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot read fixture child output");
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

void scenario(const std::string& executable, const Work& work, bool ignore_eos,
              std::optional<int> status, const std::string& body, std::string_view expected) {
    Fd listener(socket(AF_INET, SOCK_STREAM, 0));
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (bind(listener.fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0
        || listen(listener.fd, 1) != 0) throw std::runtime_error("cannot bind localhost");
    socklen_t length = sizeof(address);
    if (getsockname(listener.fd, reinterpret_cast<sockaddr*>(&address), &length) != 0) {
        throw std::runtime_error("cannot get localhost port");
    }
    // The child must not retain the listening socket.
    if (fcntl(listener.fd, F_SETFD, FD_CLOEXEC) != 0) throw std::runtime_error("fcntl failed");
    const auto port = std::to_string(ntohs(address.sin_port));
    const auto stdout_path = work.path / "stdout";
    const auto stderr_path = work.path / "stderr";
    std::vector<std::string> args{executable, "--server",
        ignore_eos ? "http://localhost:" + port + "/" : "http://127.0.0.1:" + port,
        "quote \" backslash \\ newline\n café", "64"};
    if (ignore_eos) args.emplace_back("--ignore-eos");
    std::vector<char*> argv;
    for (auto& arg : args) argv.push_back(arg.data());
    argv.push_back(nullptr);
    posix_spawn_file_actions_t actions;
    int spawn = posix_spawn_file_actions_init(&actions);
    if (spawn != 0) throw std::runtime_error("spawn actions failed");
    spawn = posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, stdout_path.c_str(),
                                           O_CREAT | O_WRONLY | O_TRUNC, 0600);
    if (spawn == 0) {
        spawn = posix_spawn_file_actions_addopen(&actions, STDERR_FILENO, stderr_path.c_str(),
                                                O_CREAT | O_WRONLY | O_TRUNC, 0600);
    }
    pid_t child = -1;
    if (spawn == 0) spawn = posix_spawn(&child, executable.c_str(), &actions, nullptr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&actions);
    if (spawn != 0) throw std::runtime_error("spawn client failed");
    try {
        wait_readable(listener.fd);
        Fd client(accept(listener.fd, nullptr, nullptr));
        const auto request = request_from(client.fd);
        if (!request.starts_with("POST /completion HTTP/")
            || request.find(R"(quote \" backslash \\ newline\u000a café)") == std::string::npos
            || request.find(R"("n_predict":64,"seed":2027,"temperature":0.7)") == std::string::npos
            || request.find(ignore_eos ? R"("ignore_eos":true)" : R"("ignore_eos":false)") == std::string::npos) {
            throw std::runtime_error("unexpected client request");
        }
        if (status) send_response(client.fd, *status, body);
    } catch (...) {
        kill(child, SIGKILL);
        while (waitpid(child, nullptr, 0) < 0 && errno == EINTR) {}
        throw;
    }
    int result = 0;
    while (waitpid(child, &result, 0) < 0) {
        if (errno != EINTR) throw std::runtime_error("waitpid failed");
    }
    const auto stdout_text = read(stdout_path);
    const auto stderr_text = read(stderr_path);
    const bool success = status == 200 && body.size() <= 8 * 1024 * 1024;
    if (!WIFEXITED(result) || WEXITSTATUS(result) != (success ? 0 : 1)) {
        throw std::runtime_error("unexpected client exit: " + stderr_text);
    }
    if (success && (stdout_text != body + '\n' || !stderr_text.starts_with("elapsed_ms="))) {
        throw std::runtime_error("client did not preserve server timings");
    }
    // libcurl versions use different wording for the same empty HTTP reply.
    const bool empty_reply = !status && (stderr_text.find("Empty reply") != std::string::npos
        || stderr_text.find("Server returned nothing") != std::string::npos);
    if (!success && (!stdout_text.empty()
        || (!empty_reply && stderr_text.find(expected) == std::string::npos))) {
        throw std::runtime_error("unexpected client failure: " + stderr_text);
    }
}
} // namespace

int main(int argc, char** argv) try {
    if (argc != 2) throw std::runtime_error("expected CLI binary argument");
    std::signal(SIGPIPE, SIG_IGN);
    Work work;
    scenario(argv[1], work, false, 200, R"({"tokens_predicted":64,"timings":{"predicted_per_second":100}})", "");
    scenario(argv[1], work, true, 200, R"({"tokens_predicted":64,"timings":{"draft_n":48,"draft_n_accepted":40}})", "");
    scenario(argv[1], work, false, 503, R"({"error":"busy"})", "llama-server HTTP 503");
    scenario(argv[1], work, false, std::nullopt, "", "Empty reply");
    scenario(argv[1], work, false, 200, std::string(8 * 1024 * 1024 + 1, 'x'), "writing");
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
