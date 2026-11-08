#pragma once

#include <batchfit/cpu/initial_guess.hpp>
#include <batchfit/cpu/solver.hpp>
#include <batchfit/synthetic.hpp>

#include "model_checks.hpp"

#include <array>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace initial_guess_checks {
namespace cpu = batchfit::cpu;
namespace models = batchfit::models;
namespace synthetic = batchfit::synthetic;
using model_checks::near;
using model_checks::require;

template <typename T> void features() {
    const double tolerance = std::is_same_v<T, float> ? 2e-6 : 1e-12;
    std::array<T, 9> x, y{T(2), T(2), T(2), T(4), T(6), T(4), T(2), T(2), T(2)};
    for (unsigned j = 0; j < x.size(); ++j)
        x[j] = T(j);
    const auto before_x = x, before_y = y;
    const auto gaussian = cpu::initial_guess<T, models::Gaussian>(x, y);
    const auto lorentzian = cpu::initial_guess<T, models::Lorentzian>(x, y);
    const auto voigt = cpu::initial_guess<T, models::PseudoVoigt>(x, y);
    require(gaussian.status == cpu::GuessStatus::success &&
                lorentzian.status == cpu::GuessStatus::success &&
                voigt.status == cpu::GuessStatus::success,
            "hand-computed peak found");
    near(gaussian.baseline, 2, tolerance, 0, "edge median baseline");
    near(gaussian.parameters[0], 4, tolerance, 0, "height above baseline");
    near(gaussian.parameters[1], 4, tolerance, 0, "symmetric centroid");
    near(gaussian.parameters[2], 2 / 2.354820045030949382, tolerance, 0, "FWHM to sigma");
    near(lorentzian.parameters[2], 1, tolerance, 0, "FWHM to HWHM");
    near(voigt.parameters[2], 2, tolerance, 0, "pseudo-Voigt common FWHM");
    near(voigt.parameters[3], 0.5, tolerance, 0, "neutral mixture seed");
    require(x == before_x && y == before_y, "guessing must not mutate observations");
    y = {T(1), T(1), T(1), T(4), T(5), T(3.5), T(1), T(1), T(1)};
    const auto centroid = cpu::initial_guess<T, models::Lorentzian>(x, y);
    require(centroid.status == cpu::GuessStatus::success, "asymmetric peak found");
    near(centroid.parameters[1], 37.5 / 9.5, tolerance, 0, "local weighted centroid, not argmax");
    near(centroid.parameters[2], (5.2 - 8.0 / 3) / 2, tolerance, 0,
         "interpolated half-height crossings");
    std::array<T, 6> even{T(100), T(2), T(1), T(4), T(3), T(-100)};
    near(cpu::detail::median<T>(even), 2.5, tolerance, 0, "even median resists edge outliers");
    std::array<T, 5> odd{T(100), T(2), T(1), T(3), T(-100)};
    near(cpu::detail::median<T>(odd), 2, tolerance, 0, "odd median");

    // Translation and nonuniform spacing must preserve physical x units.
    x = {T(10), T(11), T(12), T(14), T(15), T(16), T(18), T(19), T(20)};
    y = {T(2), T(2), T(2), T(4), T(6), T(4), T(2), T(2), T(2)};
    const auto nonuniform = cpu::initial_guess<T, models::Lorentzian>(x, y);
    require(nonuniform.status == cpu::GuessStatus::success, "nonuniform grid");
    near(nonuniform.parameters[1], 15, tolerance, 0, "physical center");
    near(nonuniform.parameters[2], 1, tolerance, 0, "physical width");
}

template <typename T> void paired_features() {
    std::array<T, 201> x, y;
    for (bool reverse_heights : {false, true}) {
        const std::array<T, 8> truth{T(reverse_heights ? 2 : 3),
                                     T(-1.5),
                                     T(0.4),
                                     T(reverse_heights ? 3 : 2),
                                     T(1.5),
                                     T(0.5),
                                     T(0.3),
                                     T(-0.04)};
        for (unsigned j = 0; j < x.size(); ++j) {
            x[j] = T(-5) + T(10) * T(j) / T(x.size() - 1);
            y[j] = models::TwoLorentzian<T>::eval(x[j], truth.data());
        }
        const auto guess = cpu::initial_guess<T, models::TwoLorentzian>(x, y);
        require(guess.status == cpu::GuessStatus::success, "two separated peaks found");
        require(guess.parameters[1] < guess.parameters[4],
                "paired peaks ordered by center, not height");
        for (unsigned k = 0; k < 6; ++k)
            near(guess.parameters[k], truth[k], k % 3 == 0 ? 0.15 : 0.06, 0, "paired peak seed");
        near(guess.parameters[6], truth[6], 0.12, 0, "edge-median linear intercept seed");
        near(guess.parameters[7], truth[7], 0.02, 0, "edge-median linear slope seed");
    }
    const std::array<T, 3> single{T(2), T(0), T(0.5)};
    for (unsigned j = 0; j < x.size(); ++j)
        y[j] = T(0.2) + models::Lorentzian<T>::eval(x[j], single.data());
    require(cpu::initial_guess<T, models::TwoLorentzian>(x, y).status ==
                cpu::GuessStatus::insufficient_peaks,
            "must not invent a second peak on a monotonic wing");
    const std::array<T, 8> weak{T(100), T(-2), T(0.2), T(1), T(2), T(0.2), T(0), T(0)};
    for (unsigned j = 0; j < x.size(); ++j) {
        x[j] = T(-10) + T(20) * T(j) / T(x.size() - 1);
        y[j] = models::TwoLorentzian<T>::eval(x[j], weak.data());
    }
    const auto weak_guess = cpu::initial_guess<T, models::TwoLorentzian>(x, y);
    require(weak_guess.status == cpu::GuessStatus::success,
            "resolved weak secondary peak is not excluded by a height-ratio floor");
    near(weak_guess.parameters[1], -2, 0.06, 0, "strong peak center");
    near(weak_guess.parameters[4], 2, 0.06, 0, "weak peak center");
    const std::array<T, 3> a{T(3), T(-3), T(0.3)}, b{T(2), T(0), T(0.3)}, c{T(1), T(3), T(0.3)};
    for (unsigned j = 0; j < x.size(); ++j)
        y[j] = models::Lorentzian<T>::eval(x[j], a.data()) +
               models::Lorentzian<T>::eval(x[j], b.data()) +
               models::Lorentzian<T>::eval(x[j], c.data());
    const auto largest = cpu::initial_guess<T, models::TwoLorentzian>(x, y);
    require(largest.status == cpu::GuessStatus::success, "two largest of three separated peaks");
    near(largest.parameters[1], -3, 0.06, 0, "largest peak selected");
    near(largest.parameters[4], 0, 0.06, 0, "second-largest peak selected");
}

template <typename T> void invalid_and_unresolved() {
    std::array<T, 9> x, y{};
    for (unsigned j = 0; j < x.size(); ++j)
        x[j] = T(j);
    const auto flat = cpu::initial_guess<T, models::Gaussian>(x, y);
    require(flat.status == cpu::GuessStatus::no_peak, "zero signal has no peak");
    require(std::all_of(flat.parameters.begin(), flat.parameters.end(),
                        [](T value) { return std::isnan(value); }),
            "failed guess must not expose usable initial parameters");
    for (unsigned choice = 0; choice < 5; ++choice) {
        auto bad_x = x, bad_y = y;
        if (choice == 0)
            bad_x[2] = bad_x[1];
        if (choice == 1)
            bad_x[2] = T(-1);
        if (choice == 2)
            bad_x[2] = std::numeric_limits<T>::infinity();
        if (choice == 3)
            bad_y[2] = std::numeric_limits<T>::quiet_NaN();
        if (choice == 4) {
            bad_x.front() = -std::numeric_limits<T>::max();
            bad_x.back() = std::numeric_limits<T>::max();
        }
        require(cpu::initial_guess<T, models::Gaussian>(bad_x, bad_y).status ==
                    cpu::GuessStatus::invalid_input,
                "invalid grid/data rejected");
    }
    require(cpu::initial_guess<T, models::Gaussian>({}, {}).status ==
                cpu::GuessStatus::invalid_input,
            "empty data");
    require(cpu::initial_guess<T, models::Gaussian>(x, {}).status ==
                cpu::GuessStatus::invalid_input,
            "shape mismatch");
    require(cpu::initial_guess<T, models::Gaussian>(std::span(x).first(4), std::span(y).first(4))
                    .status == cpu::GuessStatus::invalid_input,
            "insufficient samples");
    const std::array<T, 3> clipped_peak{T(2), T(0), T(1)};
    for (unsigned j = 0; j < x.size(); ++j)
        y[j] = models::Gaussian<T>::eval(x[j], clipped_peak.data());
    require(cpu::initial_guess<T, models::Gaussian>(x, y).status == cpu::GuessStatus::no_peak,
            "missing half-height crossing must not fabricate a full width");
    const std::array<T, 3> absorption{T(-2), T(4), T(1)};
    for (unsigned j = 0; j < x.size(); ++j)
        y[j] = models::Gaussian<T>::eval(x[j], absorption.data());
    require(cpu::initial_guess<T, models::Gaussian>(x, y).status == cpu::GuessStatus::no_peak,
            "automatic guesses target emission peaks, not absorption dips");
}

struct Counts {
    std::size_t total = 0, guessed = 0, converged = 0, recovered = 0;
    void add(const Counts &other) {
        total += other.total;
        guessed += other.guessed;
        converged += other.converged;
        recovered += other.recovered;
    }
};

template <typename T, template <typename> class Model>
Counts convergence(synthetic::Model kind, const char *name) {
    constexpr unsigned p = Model<T>::parameter_count;
    Counts total;
    for (auto noise : {synthetic::Noise::gaussian, synthetic::Noise::poisson}) {
        for (double snr : {10.0, 20.0, 40.0}) {
            synthetic::Options data;
            data.model = kind;
            data.parameters = synthetic::default_ranges(kind);
            if constexpr (p == 8) {
                data.parameters[6] = {0.3F, 0.5F};
                data.parameters[7] = {-0.04F, 0.04F};
            }
            data.fits = 500;
            data.points = 129;
            data.chunk_fits = 64;
            data.snr = snr;
            data.noise = noise;
            data.seed = 20261107 + std::uint64_t(kind) * 137 + std::uint64_t(snr) * 31 +
                        std::uint64_t(noise) * 11;
            Counts block;
            synthetic::generate(
                data, [&](std::span<const float> grid, const synthetic::Chunk &chunk) {
                    const std::vector<T> x(grid.begin(), grid.end());
                    std::vector<T> y(grid.size());
                    cpu::SolverOptions<T, p> options;
                    const T minimum_width = (x[1] - x[0]) / T(8);
                    options.lower[0] = T(0);
                    options.lower[1] = x.front();
                    options.upper[1] = x.back();
                    options.lower[2] = minimum_width;
                    options.upper[2] = x.back() - x.front();
                    if constexpr (p == 4) {
                        options.lower[3] = T(0);
                        options.upper[3] = T(1);
                    }
                    if constexpr (p == 8) {
                        options.lower[3] = T(0);
                        options.lower[4] = x.front();
                        options.upper[4] = x.back();
                        options.lower[5] = minimum_width;
                        options.upper[5] = x.back() - x.front();
                    }
                    for (std::size_t i = 0; i < chunk.fits; ++i) {
                        ++block.total;
                        std::copy_n(chunk.y.begin() + i * y.size(), y.size(), y.begin());
                        const auto guess = cpu::initial_guess<T, Model>(x, y);
                        if (guess.status != cpu::GuessStatus::success)
                            continue;
                        ++block.guessed;
                        const auto fit = cpu::fit_single<T, Model>(x, y, guess.parameters, options);
                        if (fit.status != batchfit::FitStatus::converged)
                            continue;
                        ++block.converged;
                        std::array<T, p> truth;
                        for (unsigned k = 0; k < p; ++k)
                            truth[k] = T(chunk.true_parameters[i * p + k]);
                        double error = 0, signal = 0;
                        for (std::size_t j = 0; j < x.size(); ++j) {
                            const double clean = double(Model<T>::eval(x[j], truth.data()));
                            const double residual =
                                double(Model<T>::eval(x[j], fit.parameters.data())) - clean;
                            error += residual * residual;
                            signal += clean * clean;
                        }
                        if (std::isfinite(fit.chi_squared) && error <= 4 * signal / (snr * snr))
                            ++block.recovered;
                    }
                });
            if constexpr (p != 8) {
                require(block.recovered * 100 > block.total * 99,
                        std::string(name) + (std::is_same_v<T, float> ? " float" : " double") +
                            " noise=" + std::to_string(int(noise)) + " SNR=" + std::to_string(snr) +
                            " guessed=" + std::to_string(block.guessed) +
                            " converged=" + std::to_string(block.converged) + " recovered=" +
                            std::to_string(block.recovered) + "/" + std::to_string(block.total));
            }
            total.add(block);
        }
    }
    std::cout << (std::is_same_v<T, float> ? "float  " : "double ") << name
              << ": total=" << total.total << " guesses=" << total.guessed
              << " status_converged=" << total.converged << " signal_recovered=" << total.recovered
              << " (" << std::fixed << std::setprecision(3)
              << 100.0 * double(total.recovered) / double(total.total) << "%)\n";
    return total;
}

template <typename T> void all() {
    features<T>();
    paired_features<T>();
    invalid_and_unresolved<T>();
    convergence<T, models::Gaussian>(synthetic::Model::gaussian, "Gaussian");
    convergence<T, models::Lorentzian>(synthetic::Model::lorentzian, "Lorentzian");
    convergence<T, models::PseudoVoigt>(synthetic::Model::pseudo_voigt, "PseudoVoigt");
    convergence<T, models::TwoLorentzian>(synthetic::Model::two_lorentzian, "TwoLorentzian");
}
} // namespace initial_guess_checks
