#pragma once

#include <batchfit/cpu/solver.hpp>
#include <batchfit/models.hpp>
#include <batchfit/synthetic.hpp>

#include "model_checks.hpp"

#include <array>
#include <limits>
#include <string>
#include <vector>

namespace solver_checks {
using batchfit::FitStatus;
using model_checks::near;
using model_checks::require;
namespace cpu = batchfit::cpu;
namespace models = batchfit::models;
namespace synthetic = batchfit::synthetic;

template <typename T> struct Constant {
    enum : unsigned { parameter_count = 1 };
    static T eval(T, const T *p) { return p[0]; }
    static void jacobian(T, const T *, T *out) { out[0] = T(1); }
};
template <typename T> struct Exponential {
    enum : unsigned { parameter_count = 1 };
    static T eval(T x, const T *p) { return std::exp(x * p[0]); }
    static void jacobian(T x, const T *p, T *out) { out[0] = x * eval(x, p); }
};
template <typename T> struct Affine {
    enum : unsigned { parameter_count = 2 };
    static T eval(T x, const T *p) { return p[0] + p[1] * x; }
    static void jacobian(T x, const T *, T *out) {
        out[0] = T(1);
        out[1] = x;
    }
};
template <typename T> struct BadJacobian : Constant<T> {
    static void jacobian(T, const T *, T *out) { out[0] = std::numeric_limits<T>::quiet_NaN(); }
};

template <typename T> void cholesky() {
    cpu::detail::Matrix<T, 3> a{{{4, 12, -16}, {12, 37, -43}, {-16, -43, 98}}};
    std::array<T, 3> b{-20, -43, 192};
    require(cpu::detail::cholesky_solve<T, 3>(a, b), "known SPD factorization");
    for (unsigned k = 0; k < 3; ++k)
        near(b[k], k + 1, 1e-6, 0, "known SPD solution");
    cpu::detail::Matrix<T, 2> indefinite{{{1, 2}, {2, 1}}};
    std::array<T, 2> rhs{1, 2};
    require(!cpu::detail::cholesky_solve<T, 2>(indefinite, rhs), "indefinite matrix rejected");
    cpu::detail::Matrix<T, 2> singular{{{1, 1}, {1, 1}}};
    require(!cpu::detail::cholesky_solve<T, 2>(singular, rhs), "singular matrix rejected");
}

template <typename T, template <typename> class Model> void recover(synthetic::Model kind) {
    constexpr unsigned p = Model<T>::parameter_count;
    for (auto noise :
         {synthetic::Noise::none, synthetic::Noise::gaussian, synthetic::Noise::poisson}) {
        synthetic::Options data;
        data.model = kind;
        data.parameters = synthetic::default_ranges(kind);
        data.fits = 32;
        data.points = 129;
        data.chunk_fits = 7;
        data.noise = noise;
        data.snr = 80;
        data.seed = 20261107;
        synthetic::generate(data, [&](std::span<const float> grid, const synthetic::Chunk &chunk) {
            std::vector<T> x(grid.begin(), grid.end()), y(grid.size());
            for (std::size_t i = 0; i < chunk.fits; ++i) {
                std::array<T, p> truth, initial;
                cpu::SolverOptions<T, p> options;
                for (unsigned k = 0; k < p; ++k) {
                    truth[k] = T(chunk.true_parameters[i * p + k]);
                    initial[k] =
                        truth[k] + T(k % 2 == 0 ? 0.08 : -0.08) * (T(0.5) + std::abs(truth[k]));
                    options.lower[k] = T(-5);
                    options.upper[k] = T(5);
                }
                options.lower[0] = T(0);
                options.lower[2] = T(0.05);
                options.upper[2] = T(3);
                if constexpr (p == 4) {
                    options.lower[3] = T(0);
                    options.upper[3] = T(1);
                }
                if constexpr (p == 8) {
                    options.upper[1] = T(-0.1);
                    options.lower[3] = T(0);
                    options.lower[4] = T(0.1);
                    options.lower[5] = T(0.05);
                    options.upper[5] = T(3);
                }
                double signal_squares = 0, signal_sum = 0;
                std::array<double, p> true_double;
                for (unsigned k = 0; k < p; ++k)
                    true_double[k] = double(truth[k]);
                cpu::detail::Matrix<double, p> fisher{};
                for (std::size_t j = 0; j < x.size(); ++j) {
                    const double clean = Model<double>::eval(double(x[j]), true_double.data());
                    signal_squares += clean * clean;
                    signal_sum += clean;
                    double jacobian[p];
                    Model<double>::jacobian(double(x[j]), true_double.data(), jacobian);
                    for (unsigned k = 0; k < p; ++k) {
                        for (unsigned l = 0; l < p; ++l)
                            fisher[k][l] += jacobian[k] * jacobian[l];
                    }
                    // Generate exact scalar-type noiseless data; noisy observations
                    // come from the streaming generator, including its float32 I/O precision.
                    y[j] = noise == synthetic::Noise::none ? Model<T>::eval(x[j], truth.data())
                                                           : T(chunk.y[i * x.size() + j]);
                }
                const double sigma = noise == synthetic::Noise::none
                                         ? 0
                                         : std::sqrt(signal_squares / double(x.size())) / data.snr;
                const double gain = data.snr * data.snr * signal_sum / signal_squares;
                const auto x_before = x, y_before = y;
                const auto initial_before = initial;
                const auto result = cpu::fit_single<T, Model>(x, y, initial, options);
                const auto context = std::string(std::is_same_v<T, float> ? "float" : "double") +
                                     " model=" + std::to_string(int(kind)) +
                                     " noise=" + std::to_string(int(noise)) +
                                     " fit=" + std::to_string(chunk.first_fit + i);
                require(result.status == FitStatus::converged,
                        context + " status=" + std::to_string(int(result.status)));
                require(result.iterations > 0 && result.iterations <= options.max_iterations,
                        context + " iterations");
                require(x == x_before && y == y_before && initial == initial_before,
                        "fit mutated borrowed inputs");
                for (unsigned k = 0; k < p; ++k) {
                    auto matrix = fisher;
                    std::array<double, p> unit{};
                    unit[k] = 1;
                    require(cpu::detail::cholesky_solve<double, p>(matrix, unit),
                            "identifiable recovery fixture");
                    const double floor =
                        (std::is_same_v<T, float> ? 5e-4 : 1e-8) * (1 + std::abs(double(truth[k])));
                    double variance = sigma * sigma * unit[k];
                    if (noise == synthetic::Noise::poisson) {
                        variance = 0;
                        for (std::size_t j = 0; j < x.size(); ++j) {
                            double jacobian[p], weight = 0;
                            Model<double>::jacobian(double(x[j]), true_double.data(), jacobian);
                            for (unsigned l = 0; l < p; ++l)
                                weight += unit[l] * jacobian[l];
                            variance += weight * weight *
                                        Model<double>::eval(double(x[j]), true_double.data()) /
                                        gain;
                        }
                    }
                    near(result.parameters[k], truth[k], floor + 8 * std::sqrt(variance), 0,
                         context + " recovered parameter=" + std::to_string(k));
                    require(result.parameters[k] >= options.lower[k] &&
                                result.parameters[k] <= options.upper[k],
                            "feasible output");
                }
                double chi_squared = 0;
                for (std::size_t j = 0; j < x.size(); ++j) {
                    const double residual =
                        double(y[j]) - double(Model<T>::eval(x[j], result.parameters.data()));
                    chi_squared += residual * residual;
                }
                near(result.chi_squared, chi_squared, 1e-12,
                     std::is_same_v<T, float> ? 2e-5 : 1e-12, context + " final chi squared");
                require(result.chi_squared <= cpu::detail::cost<T, Model<T>>(x, y, initial),
                        "accepted objective must not increase");
            }
        });
    }
}

template <typename T> void bounds_and_stopping() {
    const std::array<T, 3> x{T(-1), T(0), T(1)}, y{T(3), T(3), T(3)};
    const std::array<T, 1> initial{T(0)};
    cpu::SolverOptions<T, 1> options;
    options.lower[0] = T(0);
    options.upper[0] = T(2);
    auto result = cpu::fit_single<T, Constant>(x, y, initial, options);
    require(result.status == FitStatus::converged && result.parameters[0] == T(2),
            "upper-bound constrained solution");
    near(result.chi_squared, 3, 1e-6, 0, "upper-bound objective");
    options.lower[0] = options.upper[0] = T(1);
    const std::array<T, 1> outside{T(99)};
    result = cpu::fit_single<T, Constant>(x, y, outside, options);
    require(result.status == FitStatus::converged && result.parameters[0] == T(1) &&
                result.iterations == 0,
            "fixed parameter and initial projection");
    near(result.chi_squared, 12, 1e-6, 0, "fixed-parameter objective");
    const std::array<T, 3> negative_y{T(-3), T(-3), T(-3)};
    options.lower[0] = T(-2);
    options.upper[0] = T(2);
    result = cpu::fit_single<T, Constant>(x, negative_y, initial, options);
    require(result.status == FitStatus::converged && result.parameters[0] == T(-2),
            "lower-bound constrained solution");
    const std::array<T, 3> affine_x{T(0), T(1), T(2)}, affine_y{T(3), T(5), T(7)};
    const std::array<T, 2> affine_initial{T(0), T(0)};
    cpu::SolverOptions<T, 2> affine_options;
    affine_options.lower[0] = T(0);
    affine_options.upper[0] = T(2);
    const auto affine =
        cpu::fit_single<T, Affine>(affine_x, affine_y, affine_initial, affine_options);
    require(affine.status == FitStatus::converged && affine.parameters[0] == T(2),
            "mixed active/free bounds");
    near(affine.parameters[1], 2.6, 2e-5, 0, "free slope at constrained optimum");
    near(affine.chi_squared, 1.2, 2e-6, 0, "constrained affine objective");
    const std::array<T, 3> peak_initial{T(1), T(0), T(1)};
    const std::array<T, 3> identical_x{T(0), T(0), T(0)};
    cpu::SolverOptions<T, 3> fixed_peak;
    fixed_peak.lower = {T(0), T(0), T(1)};
    fixed_peak.upper = {T(10), T(0), T(1)};
    const auto identified =
        cpu::fit_single<T, models::Lorentzian>(identical_x, y, peak_initial, fixed_peak);
    require(identified.status == FitStatus::converged, "fixed unidentifiable columns are excluded");
    near(identified.parameters[0], 3, 1e-5, 0, "remaining free amplitude");
    std::array<T, 129> peak_x, peak_y;
    const std::array<T, 3> peak_truth{T(2), T(0.2), T(0.7)},
        negative_width{T(1.5), T(-0.1), T(-0.5)};
    for (unsigned j = 0; j < peak_x.size(); ++j) {
        peak_x[j] = T(-5) + T(10) * T(j) / T(peak_x.size() - 1);
        peak_y[j] = models::Gaussian<T>::eval(peak_x[j], peak_truth.data());
    }
    cpu::SolverOptions<T, 3> positive_width;
    positive_width.lower[2] = T(0.05);
    positive_width.upper[2] = T(2);
    const auto peak =
        cpu::fit_single<T, models::Gaussian>(peak_x, peak_y, negative_width, positive_width);
    require(peak.status == FitStatus::converged && peak.parameters[2] > T(0),
            "initial width is clamped before evaluation");
    for (unsigned k = 0; k < 3; ++k)
        near(peak.parameters[k], peak_truth[k], 1e-5, 0, "positive-width recovery");
    options = {};
    options.gradient_tolerance = T(10);
    result = cpu::fit_single<T, Constant>(x, y, initial, options);
    require(result.status == FitStatus::converged && result.iterations == 0 &&
                result.parameters[0] == T(0),
            "gradient-only stop");
    options = {};
    options.gradient_tolerance = options.cost_tolerance = T(0);
    options.step_tolerance = T(10);
    result = cpu::fit_single<T, Constant>(x, y, initial, options);
    require(result.status == FitStatus::converged && result.iterations == 1, "step-only stop");
    options = {};
    options.gradient_tolerance = options.step_tolerance = T(0);
    options.cost_tolerance = T(1);
    result = cpu::fit_single<T, Constant>(x, y, initial, options);
    require(result.status == FitStatus::converged && result.iterations == 1,
            "relative-cost-only stop");
    options = {};
    options.initial_lambda = T(16) / std::numeric_limits<T>::epsilon();
    options.max_iterations = 3;
    result = cpu::fit_single<T, Constant>(x, y, initial, options);
    require(result.status == FitStatus::max_iterations && result.iterations == 3,
            "large damping alone must not manufacture convergence");
}

template <typename T> void failure_statuses() {
    const std::array<T, 3> x{T(1), T(1), T(1)}, y{T(2), T(2), T(2)};
    const std::array<T, 1> initial{T(0)};
    cpu::SolverOptions<T, 1> options;
    options.max_iterations = 0;
    auto result = cpu::fit_single<T, Constant>(x, y, initial, options);
    require(result.status == FitStatus::max_iterations && result.iterations == 0 &&
                result.chi_squared == T(12),
            "zero iteration budget");
    const std::array<T, 1> exponential_initial{T(-2)};
    const std::array<T, 3> exponential_y{T(std::exp(2.0)), T(std::exp(2.0)), T(std::exp(2.0))};
    options.max_iterations = 1;
    auto rejected = cpu::fit_single<T, Exponential>(x, exponential_y, exponential_initial, options);
    require(rejected.status == FitStatus::max_iterations && rejected.iterations == 1 &&
                rejected.parameters == exponential_initial,
            "rejected step must preserve accepted parameters");
    near(rejected.chi_squared,
         cpu::detail::cost<T, Exponential<T>>(x, exponential_y, exponential_initial), 0, 0,
         "rejected step preserves cost");
    options.max_iterations = 200;
    auto recovered =
        cpu::fit_single<T, Exponential>(x, exponential_y, exponential_initial, options);
    require(recovered.status == FitStatus::converged && recovered.iterations > 1,
            "damping recovers after rejection");
    near(recovered.parameters[0], 2, 1e-5, 0, "recovered exponential parameter");
    const std::array<T, 3> identical_x{T(0), T(0), T(0)}, peak_initial{T(1), T(0), T(1)};
    auto singular = cpu::fit_single<T, models::Lorentzian>(identical_x, y, peak_initial);
    require(singular.status == FitStatus::singular_matrix && singular.iterations == 1 &&
                singular.parameters == peak_initial && singular.chi_squared == T(3),
            "zero Jacobian columns trigger singular status");
    for (unsigned choice = 0; choice < 11; ++choice) {
        auto bad_x = x, bad_y = y;
        auto bad_initial = initial;
        cpu::SolverOptions<T, 1> bad;
        switch (choice) {
        case 0:
            bad_x[0] = std::numeric_limits<T>::quiet_NaN();
            break;
        case 1:
            bad_y[0] = std::numeric_limits<T>::infinity();
            break;
        case 2:
            bad_initial[0] = std::numeric_limits<T>::quiet_NaN();
            break;
        case 3:
            bad.initial_lambda = T(0);
            break;
        case 4:
            bad.gradient_tolerance = T(-1);
            break;
        case 5:
            bad.step_tolerance = std::numeric_limits<T>::infinity();
            break;
        case 6:
            bad.cost_tolerance = std::numeric_limits<T>::quiet_NaN();
            break;
        case 7:
            bad.lower[0] = T(2);
            bad.upper[0] = T(1);
            break;
        case 8:
            bad.lower[0] = std::numeric_limits<T>::quiet_NaN();
            break;
        case 9:
            bad.lower[0] = std::numeric_limits<T>::infinity();
            break;
        case 10:
            bad.upper[0] = -std::numeric_limits<T>::infinity();
            break;
        }
        const auto invalid = cpu::fit_single<T, Constant>(bad_x, bad_y, bad_initial, bad);
        require(invalid.status == FitStatus::invalid_input && invalid.iterations == 0 &&
                    std::isnan(invalid.chi_squared),
                "invalid input status");
    }
    require(cpu::fit_single<T, Constant>({}, {}, initial).status == FitStatus::invalid_input,
            "empty data");
    require(cpu::fit_single<T, Constant>(x, {}, initial).status == FitStatus::invalid_input,
            "mismatched data");
    require(cpu::fit_single<T, Constant>(x, y, {}).status == FitStatus::invalid_input,
            "mismatched initial parameters");
    require(cpu::fit_single<T, BadJacobian>(x, y, initial).status == FitStatus::invalid_input,
            "non-finite analytic Jacobian");
    const std::array<T, 3> zero_width{T(1), T(0), T(0)};
    require(cpu::fit_single<T, models::Gaussian>(identical_x, y, zero_width).status ==
                FitStatus::invalid_input,
            "non-finite initial model");
}

template <typename T> void all() {
    cholesky<T>();
    recover<T, models::Gaussian>(synthetic::Model::gaussian);
    recover<T, models::Lorentzian>(synthetic::Model::lorentzian);
    recover<T, models::PseudoVoigt>(synthetic::Model::pseudo_voigt);
    recover<T, models::TwoLorentzian>(synthetic::Model::two_lorentzian);
    bounds_and_stopping<T>();
    failure_statuses<T>();
}
} // namespace solver_checks
