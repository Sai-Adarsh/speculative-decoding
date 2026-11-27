#include <curl/curl.h>

#include <charconv>
#include <chrono>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

std::string json_string(std::string_view text) {
    constexpr std::string_view hex = "0123456789abcdef";
    std::string escaped = "\"";
    for (const unsigned char c : text) {
        if (c == '\\' || c == '"') {
            escaped.push_back('\\');
            escaped.push_back(static_cast<char>(c));
        } else if (c < 0x20) {
            escaped += "\\u00";
            escaped.push_back(hex[c >> 4]);
            escaped.push_back(hex[c & 0x0f]);
        } else {
            escaped.push_back(static_cast<char>(c));
        }
    }
    escaped.push_back('"');
    return escaped;
}

std::size_t collect(char* bytes, std::size_t size, std::size_t count, void* output) noexcept {
    auto& response = *static_cast<std::string*>(output);
    const auto length = size * count;
    if (length > 8 * 1024 * 1024 - response.size()) {
        return 0;
    }
    try {
        response.append(bytes, length);
        return length;
    } catch (...) {
        return 0;
    }
}

bool local_server(std::string_view url) {
    for (const auto prefix : {std::string_view("http://127.0.0.1:"),
                              std::string_view("http://localhost:")}) {
        if (!url.starts_with(prefix)) {
            continue;
        }
        url.remove_prefix(prefix.size());
        if (url.ends_with('/')) {
            url.remove_suffix(1);
        }
        unsigned port = 0;
        const auto result = std::from_chars(url.data(), url.data() + url.size(), port);
        return result.ec == std::errc{} && result.ptr == url.data() + url.size() &&
               port > 0 && port <= 65535;
    }
    return false;
}

} // namespace

void llama_completion(std::string_view server, std::string_view prompt, std::size_t tokens) {
    if (!local_server(server)) {
        throw std::runtime_error("use a local llama-server URL such as http://127.0.0.1:8080");
    }
    const std::string url = std::string(server) + (server.ends_with('/') ? "completion" : "/completion");
    const std::string payload = "{\"prompt\":" + json_string(prompt)
        + ",\"n_predict\":" + std::to_string(tokens)
        + ",\"seed\":2027,\"temperature\":0.7,\"cache_prompt\":false,\"stream\":false}";
    const std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> curl(curl_easy_init(), curl_easy_cleanup);
    const std::unique_ptr<curl_slist, decltype(&curl_slist_free_all)> headers(
        curl_slist_append(nullptr, "Content-Type: application/json"), curl_slist_free_all);
    if (!curl || !headers) {
        throw std::runtime_error("cannot initialize libcurl");
    }

    std::string response;
    curl_easy_setopt(curl.get(), CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl.get(), CURLOPT_NOPROXY, "*");
    curl_easy_setopt(curl.get(), CURLOPT_HTTPHEADER, headers.get());
    curl_easy_setopt(curl.get(), CURLOPT_POSTFIELDS, payload.c_str());
    curl_easy_setopt(curl.get(), CURLOPT_WRITEFUNCTION, collect);
    curl_easy_setopt(curl.get(), CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl.get(), CURLOPT_CONNECTTIMEOUT, 10L);
    curl_easy_setopt(curl.get(), CURLOPT_TIMEOUT, 600L);
    const auto start = std::chrono::steady_clock::now();
    const auto result = curl_easy_perform(curl.get());
    const auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    if (result != CURLE_OK) {
        throw std::runtime_error(curl_easy_strerror(result));
    }
    long status = 0;
    curl_easy_getinfo(curl.get(), CURLINFO_RESPONSE_CODE, &status);
    if (status < 200 || status >= 300) {
        throw std::runtime_error("llama-server HTTP " + std::to_string(status) + ": " + response);
    }
    std::cerr << "elapsed_ms=" << elapsed << '\n';
    std::cout << response << '\n';
}
