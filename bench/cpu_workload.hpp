#pragma once

#include <batchfit/cpu/batch.hpp>
#include <batchfit/cpu/initial_guess.hpp>
#include <batchfit/synthetic.hpp>

#include <bit>
#include <cstdint>
#include <stdexcept>

namespace batchfit_bench {
namespace cpu = batchfit::cpu;
namespace models = batchfit::models;
namespace synthetic = batchfit::synthetic;

template <template <typename> class Model> struct Workload {
    static constexpr unsigned p = Model<float>::parameter_count;
    std::size_t fits, fallback_guesses = 0;
    std::vector<float> x, y, initial;
    cpu::SolverOptions<float, p> options;

    Workload(std::size_t count, std::size_t points) : fits(count) {
        if (count == 0 || points < 2 || count > (std::size_t(1) << 30) / sizeof(float) / points)
            throw std::invalid_argument("benchmark observations must fit within 1 GiB");
        synthetic::Options data;
        if constexpr (std::is_same_v<Model<float>, models::Gaussian<float>>)
            data.model = synthetic::Model::gaussian;
        else if constexpr (std::is_same_v<Model<float>, models::Lorentzian<float>>)
            data.model = synthetic::Model::lorentzian;
        else if constexpr (p == 4)
            data.model = synthetic::Model::pseudo_voigt;
        else
            data.model = synthetic::Model::two_lorentzian;
        data.parameters = synthetic::default_ranges(data.model);
        data.parameters[2] = p == 4 ? synthetic::Range{1.4F, 2.0F} : synthetic::Range{0.6F, 1.0F};
        if constexpr (p == 8) {
            data.parameters[1] = {-2, -1};
            data.parameters[4] = {1, 2};
            data.parameters[2] = data.parameters[5] = {0.5F, 0.8F};
            data.parameters[6] = {0.3F, 0.5F};
            data.parameters[7] = {-0.02F, 0.02F};
        }
        data.fits = count;
        data.points = points;
        data.chunk_fits = 1024;
        data.noise = synthetic::Noise::gaussian;
        data.snr = 20;
        data.seed = 20261107 + 137 * std::uint64_t(data.model) + points;
        y.resize(count * points);
        initial.resize(count * p);
        synthetic::generate(data, [&](std::span<const float> grid, const synthetic::Chunk &chunk) {
            if (x.empty())
                x.assign(grid.begin(), grid.end());
            std::copy(chunk.y.begin(), chunk.y.end(), y.begin() + chunk.first_fit * points);
            for (std::size_t i = 0; i < chunk.fits; ++i) {
                const auto guess =
                    cpu::initial_guess<float, Model>(x, chunk.y.subspan(i * points, points));
                auto parameters = guess.parameters;
                if (guess.status != cpu::GuessStatus::success) {
                    ++fallback_guesses;
                    if constexpr (p == 8)
                        parameters = {2, -1.5F, 0.65F, 2, 1.5F, 0.65F, 0.4F, 0};
                    else if constexpr (p == 4)
                        parameters = {2, 0, 1.7F, 0.5F};
                    else
                        parameters = {2, 0, 0.8F};
                }
                std::copy(parameters.begin(), parameters.end(),
                          initial.begin() + (chunk.first_fit + i) * p);
            }
        });
        options.lower[0] = 0;
        options.lower[1] = x.front();
        options.upper[1] = x.back();
        options.lower[2] = (x[1] - x[0]) / 8;
        options.upper[2] = x.back() - x.front();
        if constexpr (p == 4) {
            options.lower[3] = 0;
            options.upper[3] = 1;
        }
        if constexpr (p == 8) {
            options.lower[3] = 0;
            options.lower[4] = x.front();
            options.upper[4] = x.back();
            options.lower[5] = options.lower[2];
            options.upper[5] = options.upper[2];
        }
    }

    void warmup(cpu::ThreadPool &pool) const {
        const auto count = std::min(fits, std::size_t(256));
        cpu::fit_batch<float, Model>(x, std::span(y).first(count * x.size()),
                                     std::span(initial).first(count * p), pool, options);
    }
    cpu::BatchResult<float> run(cpu::ThreadPool &pool) const {
        return cpu::fit_batch<float, Model>(x, y, initial, pool, options);
    }
};

struct Summary {
    std::size_t converged = 0;
    double mean_iterations = 0;
    std::uint64_t hash = 14695981039346656037ULL;
};

inline Summary summarize(const cpu::BatchResult<float> &result) {
    Summary summary;
    const auto mix = [&](std::uint32_t value) {
        summary.hash = (summary.hash ^ value) * 1099511628211ULL;
    };
    for (float value : result.parameters)
        mix(std::bit_cast<std::uint32_t>(value));
    for (std::size_t i = 0; i < result.status.size(); ++i) {
        summary.converged += result.status[i] == batchfit::FitStatus::converged;
        summary.mean_iterations += result.iterations[i];
        mix(std::uint32_t(result.status[i]));
        mix(result.iterations[i]);
        mix(std::bit_cast<std::uint32_t>(result.chi_squared[i]));
    }
    if (!result.status.empty())
        summary.mean_iterations /= double(result.status.size());
    return summary;
}

inline unsigned all_cores() { return std::max(1U, std::thread::hardware_concurrency()); }
} // namespace batchfit_bench
