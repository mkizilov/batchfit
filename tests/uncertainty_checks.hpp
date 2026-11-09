#pragma once
#include "model_checks.hpp"
#include <batchfit/batchfit.hpp>
#include <cstring>

namespace uncertainty_checks {
using model_checks::require;
template <typename T> struct Line {
    static constexpr unsigned parameter_count = 2;
    static T eval(T x, const T *p) { return p[0] + p[1] * x; }
    static void jacobian(T x, const T *, T *out) {
        out[0] = T(1);
        out[1] = x;
    }
};
template <typename T> struct Duplicate : Line<T> {
    static void jacobian(T, const T *, T *out) { out[0] = out[1] = T(1); }
};

template <typename T> void algebra() {
    const std::array<T, 5> x{T(-2), T(-1), T(0), T(1), T(2)};
    batchfit::cpu::FitResult<T, 2> fit{{T(1), T(2)}, batchfit::FitStatus::converged, 0, T(6)};
    auto covariance = batchfit::cpu::estimate_uncertainty<T, Line>(x, fit);
    require(covariance.status == batchfit::UncertaintyStatus::available,
            "covariance of full-rank linear model");
    model_checks::near(covariance.covariance[0], .4, 1e-7, 1e-6,
                       "SSE/(M-P) inverse information intercept");
    model_checks::near(covariance.covariance[3], .2, 1e-7, 1e-6,
                       "SSE/(M-P) inverse information slope");
    require(covariance.covariance[1] == T(0) && covariance.covariance[2] == T(0),
            "symmetric independent columns");
    batchfit::cpu::SolverOptions<T, 2> options;
    options.lower[0] = options.upper[0] = T(1);
    covariance = batchfit::cpu::estimate_uncertainty<T, Line>(x, fit, options);
    require(covariance.status == batchfit::UncertaintyStatus::available &&
                covariance.standard_errors[0] == T(0),
            "fixed parameter has zero conditional uncertainty");
    model_checks::near(covariance.covariance[3], .15, 1e-7, 1e-6,
                       "fixed parameter excluded from DOF");
    options.lower[1] = options.upper[1] = T(2);
    covariance = batchfit::cpu::estimate_uncertainty<T, Line>(x, fit, options);
    require(covariance.status == batchfit::UncertaintyStatus::available &&
                covariance.standard_errors[1] == T(0),
            "all-fixed covariance zero");
    covariance = batchfit::cpu::estimate_uncertainty<T, Line>(std::span<const T>(x).first(2), fit);
    require(covariance.status == batchfit::UncertaintyStatus::insufficient_dof &&
                std::isnan(covariance.covariance[0]),
            "no fabricated covariance without residual DOF");
    covariance = batchfit::cpu::estimate_uncertainty<T, Duplicate>(x, fit);
    require(covariance.status == batchfit::UncertaintyStatus::singular_matrix &&
                std::isnan(covariance.standard_errors[0]),
            "rank-deficient information is rejected");
    options = {};
    options.lower[0] = T(1);
    covariance = batchfit::cpu::estimate_uncertainty<T, Line>(x, fit, options);
    require(covariance.status == batchfit::UncertaintyStatus::active_bound,
            "active inequality bound is not Gaussian");
    fit.status = batchfit::FitStatus::max_iterations;
    covariance = batchfit::cpu::estimate_uncertainty<T, Line>(x, fit);
    require(covariance.status == batchfit::UncertaintyStatus::not_converged,
            "fit status is preserved separately");
    fit.status = batchfit::FitStatus::converged;
    fit.chi_squared = std::numeric_limits<T>::quiet_NaN();
    covariance = batchfit::cpu::estimate_uncertainty<T, Line>(x, fit);
    require(covariance.status == batchfit::UncertaintyStatus::invalid_input,
            "nonfinite accepted objective cannot scale covariance");
    fit.chi_squared = T(0);
    covariance = batchfit::cpu::estimate_uncertainty<T, Line>(x, fit);
    require(covariance.status == batchfit::UncertaintyStatus::available &&
                covariance.standard_errors[0] == T(0),
            "noiseless identifiable solution has zero estimated variance");
}
inline void cpu() {
    algebra<float>();
    algebra<double>();
    batchfit::Coordinates frame{5, 2, 3};
    std::array<double, 64> identity{};
    for (unsigned i = 0; i < 8; ++i)
        identity[i * 8 + i] = 1;
    identity[6 * 8 + 7] = identity[7 * 8 + 6] = .4;
    const auto physical = frame.physical_covariance<batchfit::models::TwoLorentzian>(identity);
    model_checks::near(physical[6 * 8 + 6], 47.25, 1e-12, 1e-12,
                       "baseline covariance transforms full affine Jacobian");
    model_checks::near(physical[6 * 8 + 7], -9.45, 1e-12, 1e-12,
                       "baseline/slope correlation retained");
    model_checks::near(physical[7 * 8 + 7], 2.25, 1e-12, 1e-12, "slope variance unit conversion");
    std::array<double, 5> x{-2, -1, 0, 1, 2}, y{-2, -1, 1, 3, 5};
    std::array<double, 2> initial{0, 1};
    batchfit::cpu::ThreadPool pool(2);
    const auto basic = batchfit::cpu::fit_batch<double, Line>(x, y, initial, pool);
    const auto extended = batchfit::cpu::fit_batch<double, Line>(x, y, initial, pool, {}, true);
    require(basic.covariance.empty() && basic.standard_errors.empty() &&
                basic.uncertainty_status.empty(),
            "disabled CPU covariance allocates no output vectors");
    require(extended.covariance.size() == 4 && extended.standard_errors.size() == 2 &&
                extended.uncertainty_status[0] == batchfit::UncertaintyStatus::available,
            "requested CPU batch output");
    require(basic.parameters == extended.parameters && basic.status == extended.status &&
                basic.iterations == extended.iterations &&
                basic.chi_squared == extended.chi_squared,
            "uncertainty must not change fitted state");
}

template <template <typename> class Model> void gpu_model(batchfit::metal::Context &context) {
    constexpr unsigned p = Model<float>::parameter_count;
    constexpr std::size_t n = 35, m = 129;
    std::array<float, p> truth;
    if constexpr (p == 8)
        truth = {2, -.4F, .12F, 1.5F, .4F, .15F, .2F, .01F};
    else if constexpr (p == 4)
        truth = {2, 0, .2F, .4F};
    else
        truth = {2, 0, .2F};
    auto basic = context.make_batch<Model>(n, m);
    auto extended = context.make_batch<Model>(n, m, true);
    for (unsigned j = 0; j < m; ++j) {
        basic.x()[j] = extended.x()[j] = -1 + float(j) / 64;
        for (unsigned i = 0; i < n; ++i)
            basic.y()[j * n + i] = extended.y()[j * n + i] =
                Model<float>::eval(basic.x()[j], truth.data()) +
                .01F * std::sin(float(7 * j + 13 * i));
    }
    for (unsigned i = 0; i < n; ++i)
        for (unsigned k = 0; k < p; ++k)
            basic.initial()[i * p + k] = extended.initial()[i * p + k] = truth[k] * 1.02F + .005F;
    batchfit::cpu::SolverOptions<float, p> options;
    options.lower[0] = 0;
    options.lower[2] = 1e-5F;
    if constexpr (p == 4) {
        options.lower[3] = 0;
        options.upper[3] = 1;
    }
    if constexpr (p == 8) {
        options.lower[3] = 0;
        options.lower[5] = 1e-5F;
    }
    for (auto variant :
         {batchfit::metal::KernelVariant::ThreadPerFit, batchfit::metal::KernelVariant::SimdPerFit,
          batchfit::metal::KernelVariant::ThreadgroupPerFit,
          batchfit::metal::KernelVariant::Automatic}) {
        const auto a = batchfit::fit_batch<Model>(basic, batchfit::Device::Metal, options, variant);
        const auto b =
            batchfit::fit_batch<Model>(extended, batchfit::Device::Metal, options, variant);
        require(!a.has_uncertainties && a.covariance.empty() && a.standard_errors.empty() &&
                    a.uncertainty_status.empty(),
                "disabled Metal outputs have no storage");
        require(b.has_uncertainties && b.covariance.size() == n * p * p &&
                    b.standard_errors.size() == n * p,
                "enabled Metal output shapes");
        require(std::memcmp(a.parameters.data(), b.parameters.data(), n * p * sizeof(float)) == 0 &&
                    std::memcmp(a.chi_squared.data(), b.chi_squared.data(), n * sizeof(float)) == 0,
                "optional GPU pass leaves fit bytes unchanged");
        for (unsigned i = 0; i < n; ++i) {
            batchfit::cpu::FitResult<float, p> fit;
            std::copy_n(b.parameters.data() + i * p, p, fit.parameters.begin());
            fit.status = b.status[i];
            fit.chi_squared = b.chi_squared[i];
            const auto expected =
                batchfit::cpu::estimate_uncertainty<float, Model>(basic.x(), fit, options);
            require(b.uncertainty_status[i] == expected.status &&
                        expected.status == batchfit::UncertaintyStatus::available,
                    "GPU covariance status matches CPU");
            for (unsigned k = 0; k < p * p; ++k)
                model_checks::near(b.covariance[i * p * p + k], expected.covariance[k], 1e-9, 2e-4,
                                   "GPU information inverse matches accepted-state CPU reference");
            for (unsigned k = 0; k < p; ++k)
                model_checks::near(b.standard_errors[i * p + k], expected.standard_errors[k], 1e-8,
                                   2e-4, "GPU standard-error parity");
        }
    }
    options.lower[2] = options.upper[2] = truth[2];
    const auto fixed = batchfit::fit_batch<Model>(extended, batchfit::Device::Metal, options);
    for (unsigned i = 0; i < n; ++i) {
        require(fixed.uncertainty_status[i] == batchfit::UncertaintyStatus::available,
                "fixed-width GPU covariance remains identifiable");
        for (unsigned j = 0; j < p; ++j)
            require(fixed.covariance[i * p * p + 2 * p + j] == 0 &&
                        fixed.covariance[i * p * p + j * p + 2] == 0,
                    "fixed-width covariance row and column are zero");
    }
    const auto async = batchfit::submit_batch<Model>(extended, options).get();
    require(async.uncertainty_status[0] == batchfit::UncertaintyStatus::available,
            "async completion includes uncertainty pass");
    extended.resize_fits(0);
    const auto empty = batchfit::submit_batch<Model>(extended).get();
    require(empty.has_uncertainties && empty.covariance.empty(),
            "empty requested covariance remains an empty view");
    extended.resize_fits(n);
    require(extended.capacity() == n, "uncertainty buffers retain capacity across reuse");
}
inline void gpu() {
    batchfit::metal::Context context;
    gpu_model<batchfit::models::Gaussian>(context);
    gpu_model<batchfit::models::Lorentzian>(context);
    gpu_model<batchfit::models::PseudoVoigt>(context);
    gpu_model<batchfit::models::TwoLorentzian>(context);
}
} // namespace uncertainty_checks
