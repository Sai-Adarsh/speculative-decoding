#include <algorithm>
#include <array>
#include <charconv>
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
void llama_completion(std::string_view server, std::string_view prompt, std::size_t tokens);
#endif

namespace {

constexpr std::string_view vocabulary = "abcd";
using Token = std::size_t;
using Distribution = std::array<double, vocabulary.size()>;
using Random = std::mt19937_64;

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

    // Columns represent next-token distributions before each proposal and after the last one.
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
    std::size_t target_passes = 0;
    std::array<std::size_t, vocabulary.size()> token_histogram{};
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
        ++stats.target_passes;
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
        ++stats.target_passes;
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

void print_stats(std::string_view mode, const Stats& stats, std::size_t total) {
    std::cout << mode << ',' << stats.target_passes << ',' << stats.proposed << ',' << stats.accepted
              << ',' << std::fixed << std::setprecision(4);
    if (stats.proposed > 0) {
        std::cout << static_cast<double>(stats.accepted) / static_cast<double>(stats.proposed);
    }
    for (const auto count : stats.token_histogram) {
        std::cout << ',' << static_cast<double>(count) / static_cast<double>(total);
    }
    std::cout << '\n';
}

} // namespace

int main(int argc, const char* const* argv) {
    try {
        if (argc > 1 && std::string_view(argv[1]) == "--server") {
#ifdef ENABLE_LLAMA_SERVER
            if (argc != 5) {
                throw std::runtime_error("usage: speculative-decoding --server URL PROMPT TOKENS");
            }
            llama_completion(argv[2], argv[3], positive(argv[4], 16'384));
            return 0;
#else
            throw std::runtime_error("rebuild with -DENABLE_LLAMA_SERVER=ON to use --server");
#endif
        }
        if (argc > 4 || (argc > 1 && std::string_view(argv[1]) == "--help")) {
            std::cerr << "usage: speculative-decoding [trials=1000] [tokens=128] [lookahead=4]\n";
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

        for (std::size_t trial = 0; trial < trials; ++trial) {
            for (const auto token : autoregressive(target, length, baseline_rng, baseline_stats)) {
                ++baseline_stats.token_histogram[token];
            }
            for (const auto token : speculative(draft, target, length, lookahead, speculative_rng,
                                                speculative_stats)) {
                ++speculative_stats.token_histogram[token];
            }
        }

        std::cout << "mode,target_calls,proposed,accepted,acceptance_rate,p_a,p_b,p_c,p_d\n";
        print_stats("autoregressive_fixture", baseline_stats, trials * length);
        print_stats("speculative_fixture", speculative_stats, trials * length);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
