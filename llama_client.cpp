#include <curl/curl.h>
#include <nlohmann/json.hpp>

#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {
using Json = nlohmann::json;

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

unsigned local_port(std::string_view url) {
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
        if (result.ec == std::errc{} && result.ptr == url.data() + url.size() && port <= 65535) {
            return port;
        }
    }
    return 0;
}

struct Completion {
    std::string body;
    double elapsed_ms;
};

enum class Sampling { stochastic, greedy };

Completion request_completion(std::string_view server, std::string_view prompt, std::size_t tokens,
                              bool ignore_eos, Sampling sampling) {
    if (local_port(server) == 0) {
        throw std::runtime_error("use a local llama-server URL such as http://127.0.0.1:8080");
    }
    const std::string url = std::string(server) + (server.ends_with('/') ? "completion" : "/completion");
    const std::string payload = "{\"prompt\":" + json_string(prompt)
        + ",\"n_predict\":" + std::to_string(tokens)
        + ",\"seed\":2027,\"temperature\":" + (sampling == Sampling::greedy ? "0" : "0.7")
        + ",\"cache_prompt\":false,\"stream\":false,\"ignore_eos\":"
        + (ignore_eos ? "true}" : "false}");
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
    return {std::move(response), elapsed};
}

struct Timings {
    std::uint64_t prompt_tokens;
    double prompt_ms;
    double decode_ms;
    double decode_tokens_per_s;
    double client_ms;
    std::uint64_t drafted;
    std::uint64_t accepted;
};

struct Measurement {
    Timings timings;
    std::string content;
};

std::uint64_t count(const Json& timings, const char* key) {
    const auto value = timings.find(key);
    if (value != timings.end() && value->is_number_unsigned()) {
        return value->get<std::uint64_t>();
    }
    if (value != timings.end() && value->is_number_integer() && value->get<std::int64_t>() >= 0) {
        return static_cast<std::uint64_t>(value->get<std::int64_t>());
    }
    throw std::runtime_error(std::string("expected a nonnegative completion count: ") + key);
}

double positive_timing(const Json& timings, const char* key) {
    const auto value = timings.find(key);
    if (value != timings.end() && value->is_number()) {
        const double measurement = value->get<double>();
        if (std::isfinite(measurement) && measurement > 0) return measurement;
    }
    throw std::runtime_error(std::string("expected a positive completion timing: ") + key);
}

Measurement read_measurement(const Completion& response, std::size_t tokens, bool speculative) {
    const auto payload = Json::parse(response.body, nullptr, false);
    const auto timings = payload.find("timings");
    if (timings == payload.end() || !timings->is_object()) {
        throw std::runtime_error("completion has no timings object");
    }
    if (count(*timings, "predicted_n") != tokens) {
        throw std::runtime_error("completion did not reach the requested token count");
    }
    if (timings->contains("cache_n") && count(*timings, "cache_n") != 0) {
        throw std::runtime_error("completion reused cached prompt tokens");
    }
    const auto drafted = timings->contains("draft_n") ? count(*timings, "draft_n") : 0;
    const auto accepted = timings->contains("draft_n_accepted") ? count(*timings, "draft_n_accepted") : 0;
    if (!speculative && (drafted != 0 || accepted != 0)) {
        throw std::runtime_error("baseline endpoint reported draft activity");
    }
    if (speculative && (drafted == 0 || !timings->contains("draft_n_accepted") || accepted > drafted)) {
        throw std::runtime_error("speculative endpoint did not report valid draft activity");
    }
    const auto content = payload.find("content");
    if (content == payload.end() || !content->is_string()) {
        throw std::runtime_error("completion has no string content");
    }
    return {{count(*timings, "prompt_n"), positive_timing(*timings, "prompt_ms"),
             positive_timing(*timings, "predicted_ms"), positive_timing(*timings, "predicted_per_second"),
             response.elapsed_ms, drafted, accepted}, content->get<std::string>()};
}

} // namespace

void llama_completion(std::string_view server, std::string_view prompt, std::size_t tokens,
                      bool ignore_eos) {
    const auto completion = request_completion(server, prompt, tokens, ignore_eos, Sampling::stochastic);
    std::cerr << "elapsed_ms=" << completion.elapsed_ms << '\n';
    std::cout << completion.body << '\n';
}

void llama_compare(std::string_view baseline, std::string_view speculative, std::string_view prompt,
                   std::size_t tokens, std::size_t pairs) {
    if (local_port(baseline) == 0 || local_port(speculative) == 0 ||
        local_port(baseline) == local_port(speculative)) {
        throw std::runtime_error("paired comparison requires two distinct loopback server ports");
    }
    const auto measure = [&](std::string_view server, bool with_draft) {
        return read_measurement(request_completion(server, prompt, tokens, true, Sampling::greedy),
                                tokens, with_draft);
    };
    const auto baseline_warmup = measure(baseline, false);
    const auto speculative_warmup = measure(speculative, true);
    const auto prompt_tokens = baseline_warmup.timings.prompt_tokens;
    if (prompt_tokens == 0 || speculative_warmup.timings.prompt_tokens != prompt_tokens) {
        throw std::runtime_error("baseline and speculative prompt token counts differ or are empty");
    }
    if (baseline_warmup.content != speculative_warmup.content) {
        throw std::runtime_error("completion content differs in warmup");
    }

    std::vector<std::pair<Timings, Timings>> results;
    results.reserve(pairs);
    for (std::size_t pair = 0; pair < pairs; ++pair) {
        Measurement base{};
        Measurement spec{};
        if (pair % 2 == 0) {
            base = measure(baseline, false);
            spec = measure(speculative, true);
        } else {
            spec = measure(speculative, true);
            base = measure(baseline, false);
        }
        if (base.timings.prompt_tokens != prompt_tokens || spec.timings.prompt_tokens != prompt_tokens) {
            throw std::runtime_error("prompt token counts changed after warmup");
        }
        if (base.content != spec.content) {
            throw std::runtime_error("completion content differs in pair " + std::to_string(pair + 1));
        }
        results.emplace_back(base.timings, spec.timings);
    }

    std::cout << "pair,first,prompt_tokens,generated_tokens,baseline_prompt_ms,speculative_prompt_ms,"
                 "baseline_decode_ms,speculative_decode_ms,baseline_elapsed_ms,speculative_elapsed_ms,"
                 "baseline_decode_tokens_per_s,speculative_decode_tokens_per_s,decode_speedup,"
                 "speculative_drafted,speculative_accepted\n" << std::setprecision(12);
    for (std::size_t pair = 0; pair < results.size(); ++pair) {
        const auto& [base, spec] = results[pair];
        std::cout << pair + 1 << ',' << (pair % 2 == 0 ? "baseline" : "speculative") << ','
                  << prompt_tokens << ',' << tokens << ',' << base.prompt_ms << ',' << spec.prompt_ms << ','
                  << base.decode_ms << ',' << spec.decode_ms << ',' << base.client_ms << ',' << spec.client_ms
                  << ',' << base.decode_tokens_per_s << ',' << spec.decode_tokens_per_s << ','
                  << spec.decode_tokens_per_s / base.decode_tokens_per_s << ',' << spec.drafted << ','
                  << spec.accepted << '\n';
    }
}
