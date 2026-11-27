#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#ifdef ENABLE_LLAMA_SERVER
void llama_completion(std::string_view server, std::string_view prompt, std::size_t tokens,
                      bool ignore_eos);
void llama_compare(std::string_view baseline, std::string_view speculative, std::string_view prompt,
                   std::size_t tokens, std::size_t pairs);
#endif

namespace {

constexpr std::string_view vocabulary = "abcd";
constexpr auto vocab_size = vocabulary.size();
constexpr auto context_count = vocab_size * vocab_size;
constexpr auto joint_count = context_count * vocab_size;
using Token = std::size_t;
using Distribution = std::array<double, vocab_size>;
using Random = std::mt19937_64;
using Clock = std::chrono::steady_clock;

class FixtureModel {
public:
    explicit FixtureModel(bool draft) : draft_(draft) {}

    Distribution next(std::span<const Token> context) const {
        const Token last = context.empty() ? 3 : context.back();
        const Token penultimate = context.size() < 2 ? 2 : context[context.size() - 2];
        const Token preferred = (last + (last == penultimate ? 2 : 1)) % vocabulary.size();
        Distribution result{};
        const Distribution masses = draft_
            ? Distribution{0.60, 0.24, 0.10, 0.06}
            : Distribution{0.68, 0.13, 0.14, 0.05};
        for (Token i = 0; i < result.size(); ++i) {
            result[(preferred + i) % result.size()] = masses[i];
        }
        return result;
    }

    // This fixture evaluates one distribution at a time. A real transformer can
    // obtain the same logical columns in one multi-position forward pass.
    std::vector<Distribution> verify(std::vector<Token> context,
                                     std::span<const Token> proposals) const {
        std::vector<Distribution> columns;
        columns.reserve(proposals.size() + 1);
        for (const auto token : proposals) {
            columns.push_back(next(context));
            context.push_back(token);
        }
        columns.push_back(next(context));
        return columns;
    }

private:
    bool draft_;
};

Token sample(const Distribution& probabilities, Random& rng) {
    return std::discrete_distribution<Token>(probabilities.begin(), probabilities.end())(rng);
}

struct Proposal {
    Token token;
    Distribution probabilities;
};

struct Stats {
    std::size_t proposed = 0;
    std::size_t accepted = 0;
    std::size_t rejected = 0;
    std::size_t target_requests = 0;
    std::size_t target_positions = 0;
    double elapsed_ms = 0;
    std::array<std::size_t, joint_count> prefix_histogram{};
    std::array<std::size_t, joint_count> history_histogram{};

    void observe(std::span<const Token> sequence) {
        std::size_t prefix = 0;
        for (std::size_t i = 0; i < std::min<std::size_t>(3, sequence.size()); ++i) {
            prefix = prefix * vocab_size + sequence[i];
        }
        ++prefix_histogram[prefix];
        for (std::size_t i = 2; i < sequence.size(); ++i) {
            const auto index = (sequence[i - 2] * vocab_size + sequence[i - 1]) * vocab_size + sequence[i];
            ++history_histogram[index];
        }
    }
};

std::vector<Token> speculative(const FixtureModel& draft, const FixtureModel& target,
                               std::size_t length, std::size_t lookahead, Random& rng,
                               Stats& stats) {
    std::vector<Token> output;
    output.reserve(length);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    while (output.size() < length) {
        const auto capacity = std::min(lookahead, length - output.size() - 1);
        auto draft_context = output;
        std::vector<Proposal> proposals;
        std::vector<Token> draft_ids;
        proposals.reserve(capacity);
        draft_ids.reserve(capacity);
        for (std::size_t i = 0; i < capacity; ++i) {
            const auto probabilities = draft.next(draft_context);
            const auto token = sample(probabilities, rng);
            proposals.push_back({token, probabilities});
            draft_ids.push_back(token);
            draft_context.push_back(token);
        }

        const auto target_columns = target.verify(output, draft_ids);
        ++stats.target_requests;
        stats.target_positions += target_columns.size();
        stats.proposed += proposals.size();
        bool rejected = false;
        for (std::size_t i = 0; i < proposals.size(); ++i) {
            const auto& [token, q] = proposals[i];
            const auto& p = target_columns[i];
            const auto acceptance = std::min(1.0, p[token] / q[token]);
            if (uniform(rng) < acceptance) {
                output.push_back(token);
                ++stats.accepted;
                continue;
            }

            Distribution residual{};
            for (Token id = 0; id < residual.size(); ++id) {
                residual[id] = std::max(0.0, p[id] - q[id]);
            }
            output.push_back(sample(residual, rng));
            ++stats.rejected;
            rejected = true;
            break;
        }

        if (!rejected) {
            output.push_back(sample(target_columns.back(), rng));
        }
    }
    return output;
}

std::vector<Token> autoregressive(const FixtureModel& target, std::size_t length, Random& rng,
                                 Stats& stats) {
    std::vector<Token> output;
    output.reserve(length);
    while (output.size() < length) {
        output.push_back(sample(target.next(output), rng));
        ++stats.target_requests;
        ++stats.target_positions;
    }
    return output;
}

std::size_t positive(std::string_view input, std::size_t limit) {
    std::size_t result = 0;
    const auto [end, error] = std::from_chars(input.data(), input.data() + input.size(), result);
    if (error != std::errc{} || end != input.data() + input.size() || result == 0 || result > limit) {
        throw std::runtime_error("integer out of range: " + std::string(input));
    }
    return result;
}

double prefix_tv(const Stats& stats, const FixtureModel& target, std::size_t trials, std::size_t depth) {
    std::size_t leaves = 1;
    for (std::size_t i = 0; i < depth; ++i) leaves *= vocab_size;
    double l1 = 0;
    for (std::size_t leaf = 0; leaf < leaves; ++leaf) {
        std::array<Token, 3> prefix{};
        auto index = leaf;
        for (std::size_t i = depth; i > 0; --i) {
            prefix[i - 1] = index % vocab_size;
            index /= vocab_size;
        }
        double expected = 1;
        for (std::size_t i = 0; i < depth; ++i) {
            expected *= target.next(std::span(prefix).first(i))[prefix[i]];
        }
        l1 += std::abs(static_cast<double>(stats.prefix_histogram[leaf]) / trials - expected);
    }
    return l1 / 2;
}

double conditional_tv(const Stats& stats, const FixtureModel& target, std::size_t windows) {
    double l1 = 0;
    for (std::size_t history = 0; history < context_count; ++history) {
        const std::array<Token, 2> context{history / vocab_size, history % vocab_size};
        const auto expected = target.next(context);
        const auto offset = history * vocab_size;
        std::size_t count = 0;
        for (std::size_t t = 0; t < vocab_size; ++t) count += stats.history_histogram[offset + t];
        for (std::size_t t = 0; t < vocab_size; ++t) {
            l1 += std::abs(static_cast<double>(stats.history_histogram[offset + t]) - count * expected[t]);
        }
    }
    return l1 / (2 * windows);
}

void print_stats(std::string_view mode, const Stats& stats, const FixtureModel& target,
                 std::size_t trials, std::size_t length, std::size_t lookahead) {
    const auto windows = trials * (length > 2 ? length - 2 : 0);
    const auto depth = std::min<std::size_t>(3, length);
    std::cout << mode << ',' << trials << ',' << length << ',' << lookahead << ','
              << stats.target_requests << ',' << stats.target_positions << ',' << stats.proposed
              << ',' << stats.accepted << ',' << stats.rejected << ',' << std::fixed << std::setprecision(6);
    if (stats.proposed > 0) {
        std::cout << static_cast<double>(stats.accepted) / static_cast<double>(stats.proposed);
    }
    std::cout << ',' << stats.elapsed_ms << ',';
    if (stats.elapsed_ms > 0) std::cout << 1000.0 * trials * length / stats.elapsed_ms;
    std::cout << ',' << depth << ',' << prefix_tv(stats, target, trials, depth) << ',' << windows << ',';
    if (windows > 0) std::cout << conditional_tv(stats, target, windows);
    std::cout << '\n';
}

} // namespace

int main(int argc, const char* const* argv) {
    try {
        if (argc > 1 && std::string_view(argv[1]) == "--server") {
#ifdef ENABLE_LLAMA_SERVER
            const bool ignore_eos = argc == 6 && std::string_view(argv[5]) == "--ignore-eos";
            if (argc != 5 && !ignore_eos) {
                throw std::runtime_error("usage: speculative-decoding --server URL PROMPT TOKENS [--ignore-eos]");
            }
            llama_completion(argv[2], argv[3], positive(argv[4], 16'384), ignore_eos);
            return 0;
#else
            throw std::runtime_error("rebuild with -DENABLE_LLAMA_SERVER=ON to use --server");
#endif
        }
        if (argc > 1 && std::string_view(argv[1]) == "--compare") {
#ifdef ENABLE_LLAMA_SERVER
            if (argc != 7) {
                throw std::runtime_error("usage: speculative-decoding --compare BASE_URL SPEC_URL PROMPT TOKENS PAIRS");
            }
            llama_compare(argv[2], argv[3], argv[4], positive(argv[5], 16'384), positive(argv[6], 500));
            return 0;
#else
            throw std::runtime_error("rebuild with -DENABLE_LLAMA_SERVER=ON to use --compare");
#endif
        }
        if (argc > 4 || (argc > 1 && std::string_view(argv[1]) == "--help")) {
            std::cerr << "usage: speculative-decoding [trials=1000] [tokens=128] [lookahead=4]\n";
#ifdef ENABLE_LLAMA_SERVER
            std::cerr << "       speculative-decoding --server URL PROMPT TOKENS [--ignore-eos]\n"
                         "       speculative-decoding --compare BASE_URL SPEC_URL PROMPT TOKENS PAIRS\n";
#endif
            return argc > 4 ? 1 : 0;
        }

        const auto trials = argc > 1 ? positive(argv[1], 100'000) : 1'000;
        const auto length = argc > 2 ? positive(argv[2], 4'096) : 128;
        const auto lookahead = argc > 3 ? positive(argv[3], 64) : 4;
        if (trials * length > 1'000'000) {
            throw std::runtime_error("trial count times sequence length must be at most 1000000");
        }
        FixtureModel draft(true);
        FixtureModel target(false);
        Random baseline_rng(2027);
        Random speculative_rng(2028);
        Stats baseline_stats;
        Stats speculative_stats;

        const auto run = [](Stats& stats, const auto& decode) {
            const auto start = Clock::now();
            const auto sequence = decode();
            stats.elapsed_ms += std::chrono::duration<double, std::milli>(Clock::now() - start).count();
            stats.observe(sequence);
        };
        const auto run_baseline = [&] {
            run(baseline_stats, [&] { return autoregressive(target, length, baseline_rng, baseline_stats); });
        };
        const auto run_speculative = [&] {
            run(speculative_stats, [&] { return speculative(draft, target, length, lookahead, speculative_rng, speculative_stats); });
        };
        for (std::size_t trial = 0; trial < trials; ++trial) {
            if (trial % 2 == 0) {
                run_baseline();
                run_speculative();
            } else {
                run_speculative();
                run_baseline();
            }
        }

        std::cout << "mode,trials,tokens_per_trial,lookahead,target_requests,serial_target_positions,"
                     "proposed,accepted,rejections,acceptance_rate,cpu_elapsed_ms,cpu_tokens_per_s,"
                     "joint_prefix_depth,joint_prefix_tv,conditional_windows,conditional_tv\n";
        print_stats("autoregressive_fixture", baseline_stats, target, trials, length, 0);
        print_stats("speculative_fixture", speculative_stats, target, trials, length, lookahead);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
