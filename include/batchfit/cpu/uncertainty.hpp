#pragma once
#include <batchfit/cpu/solver.hpp>
#include <batchfit/uncertainty_types.hpp>

namespace batchfit::cpu {
template <typename T, std::size_t P> struct ParameterUncertainty {
    std::array<T, P * P> covariance;
    std::array<T, P> standard_errors;
    UncertaintyStatus status = UncertaintyStatus::not_converged;
};

// Re-linearize the accepted solution, never invert the damped LM trial system.
template <typename T, template <typename> class Model>
ParameterUncertainty<T, Model<T>::parameter_count>
estimate_uncertainty(std::span<const T> x, const FitResult<T, Model<T>::parameter_count> &fit,
                     const SolverOptions<T, Model<T>::parameter_count> &options = {}) {
    constexpr unsigned p = Model<T>::parameter_count;
    static_assert(p > 0 && p <= 8 && (std::is_same_v<T, float> || std::is_same_v<T, double>));
    ParameterUncertainty<T, p> output;
    const T nan = std::numeric_limits<T>::quiet_NaN();
    output.covariance.fill(nan);
    output.standard_errors.fill(nan);
    if (fit.status != FitStatus::converged)
        return output;
    output.status = UncertaintyStatus::invalid_input;
    if (!std::isfinite(fit.chi_squared) || fit.chi_squared < T(0))
        return output;
    std::array<bool, p> fixed{};
    unsigned free = 0;
    for (unsigned k = 0; k < p; ++k) {
        if (!std::isfinite(fit.parameters[k]) || std::isnan(options.lower[k]) ||
            std::isnan(options.upper[k]) || options.lower[k] > options.upper[k] ||
            fit.parameters[k] < options.lower[k] || fit.parameters[k] > options.upper[k])
            return output;
        fixed[k] = options.lower[k] == options.upper[k];
        free += !fixed[k];
        if (!fixed[k] &&
            (fit.parameters[k] == options.lower[k] || fit.parameters[k] == options.upper[k])) {
            output.status = UncertaintyStatus::active_bound;
            return output;
        }
    }
    if (x.size() <= free) {
        output.status = UncertaintyStatus::insufficient_dof;
        return output;
    }
    detail::Matrix<T, p> information{}, correction{};
    for (T point : x) {
        if (!std::isfinite(point))
            return output;
        T jacobian[p];
        Model<T>::jacobian(point, fit.parameters.data(), jacobian);
        for (unsigned i = 0; i < p; ++i) {
            if (!std::isfinite(jacobian[i]))
                return output;
            for (unsigned j = 0; j <= i; ++j)
                numerics::add(jacobian[i] * jacobian[j], information[i][j], correction[i][j]);
        }
    }
    for (unsigned i = 0; i < p; ++i)
        for (unsigned j = 0; j <= i; ++j) {
            if (!std::isfinite(information[i][j]))
                return output;
            information[j][i] = information[i][j];
        }
    for (unsigned i = 0; i < p; ++i)
        if (fixed[i]) {
            for (unsigned j = 0; j < p; ++j)
                information[i][j] = information[j][i] = T(0);
            information[i][i] = T(1);
        }
    const T variance = fit.chi_squared / T(x.size() - free);
    if (!std::isfinite(variance))
        return output;
    std::array<T, std::size_t(p) * p> covariance{};
    for (unsigned column = 0; column < p; ++column) {
        if (fixed[column])
            continue;
        auto factor = information;
        std::array<T, p> rhs{};
        rhs[column] = T(1);
        if (!detail::cholesky_solve<T, p>(factor, rhs, true)) {
            output.status = UncertaintyStatus::singular_matrix;
            return output;
        }
        for (unsigned i = 0; i < p; ++i) {
            if (!fixed[i] &&
                factor[i][i] * factor[i][i] <= T(64) * std::numeric_limits<T>::epsilon()) {
                output.status = UncertaintyStatus::singular_matrix;
                return output;
            }
            covariance[i * p + column] = fixed[i] ? T(0) : rhs[i] * variance;
            if (!std::isfinite(covariance[i * p + column]))
                return output;
        }
    }
    for (unsigned i = 0; i < p; ++i) {
        if (covariance[i * p + i] < T(0))
            return output;
        for (unsigned j = 0; j < i; ++j) {
            const T symmetric = covariance[i * p + j] / T(2) + covariance[j * p + i] / T(2);
            covariance[i * p + j] = covariance[j * p + i] = symmetric;
        }
    }
    output.covariance = covariance;
    for (unsigned i = 0; i < p; ++i)
        output.standard_errors[i] = std::sqrt(covariance[i * p + i]);
    output.status = UncertaintyStatus::available;
    return output;
}
} // namespace batchfit::cpu
