#pragma once

#include <batchfit/models.hpp>
#include <batchfit/numerics.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <span>
#include <type_traits>

namespace batchfit {
enum class FitStatus {
    converged,
    max_iterations,
    singular_matrix,
    invalid_input,
    degenerate_data,
    unresolved_peak
};

namespace cpu {
template <typename T, std::size_t P> struct SolverOptions {
    unsigned max_iterations = 100;
    T initial_lambda = T(1e-3);
    T gradient_tolerance = std::is_same_v<T, float> ? T(1e-5) : T(1e-10);
    T step_tolerance = std::is_same_v<T, float> ? T(1e-6) : T(1e-10);
    T cost_tolerance = std::is_same_v<T, float> ? T(1e-6) : T(1e-12);
    bool compensated_sum = false;
    bool scaled_cholesky = false;
    std::array<T, P> lower;
    std::array<T, P> upper;

    SolverOptions() {
        lower.fill(-std::numeric_limits<T>::infinity());
        upper.fill(std::numeric_limits<T>::infinity());
    }
};

template <typename T, std::size_t P> struct FitResult {
    std::array<T, P> parameters;
    FitStatus status = FitStatus::invalid_input;
    unsigned iterations = 0;
    T chi_squared = std::numeric_limits<T>::quiet_NaN();
};

namespace detail {
template <typename T, std::size_t P> using Matrix = std::array<std::array<T, P>, P>;

template <typename T, typename Model>
inline constexpr bool spectral_model =
    std::is_same_v<Model, models::Gaussian<T>> || std::is_same_v<Model, models::Lorentzian<T>> ||
    std::is_same_v<Model, models::PseudoVoigt<T>> ||
    std::is_same_v<Model, models::TwoLorentzian<T>>;
template <typename T> T grid_spacing(std::span<const T> x) {
    T spacing = std::numeric_limits<T>::infinity();
    bool up = true, down = true;
    for (std::size_t j = 1; j < x.size(); ++j) {
        const T difference = x[j] - x[j - 1];
        up = up && difference >= T(0);
        down = down && difference <= T(0);
        if (difference != T(0))
            spacing = std::min(spacing, std::abs(difference));
    }
    return (up || down) && std::isfinite(spacing) ? spacing : T(0);
}
template <typename T, typename Model>
bool free_shape(const SolverOptions<T, Model::parameter_count> &options) {
    if constexpr (spectral_model<T, Model>) {
        bool free = options.lower[1] != options.upper[1] || options.lower[2] != options.upper[2];
        if constexpr (Model::parameter_count == 4)
            free = free || options.lower[3] != options.upper[3];
        if constexpr (Model::parameter_count == 8)
            free = free || options.lower[4] != options.upper[4] ||
                   options.lower[5] != options.upper[5];
        return free;
    } else
        return false;
}
template <typename T, typename Model>
FitStatus spectral_success(const std::array<T, Model::parameter_count> &parameters, T spacing) {
    if constexpr (spectral_model<T, Model>) {
        const T factor = std::is_same_v<Model, models::Gaussian<T>>      ? T(2.354820045030949382)
                         : std::is_same_v<Model, models::PseudoVoigt<T>> ? T(1)
                                                                         : T(2);
        if (spacing > T(0) && std::abs(parameters[2]) * factor < spacing)
            return FitStatus::unresolved_peak;
        if constexpr (Model::parameter_count == 8)
            if (spacing > T(0) && std::abs(parameters[5]) * factor < spacing)
                return FitStatus::unresolved_peak;
    }
    return FitStatus::converged;
}

// The lower triangle becomes the Cholesky factor; rhs becomes the solution.
template <typename T, std::size_t P>
bool cholesky_solve(Matrix<T, P> &matrix, std::array<T, P> &rhs, bool scaled = false) {
    std::array<T, P> scale;
    if (scaled) {
        for (unsigned i = 0; i < P; ++i) {
            if (!std::isfinite(matrix[i][i]) || matrix[i][i] <= T(0))
                return false;
            scale[i] = std::sqrt(matrix[i][i]);
        }
        for (unsigned i = 0; i < P; ++i) {
            rhs[i] /= scale[i];
            for (unsigned j = 0; j < P; ++j)
                matrix[i][j] = (matrix[i][j] / scale[i]) / scale[j];
        }
    }
    for (unsigned i = 0; i < P; ++i) {
        for (unsigned j = 0; j <= i; ++j) {
            T value = matrix[i][j];
            for (unsigned k = 0; k < j; ++k)
                value -= matrix[i][k] * matrix[j][k];
            if (!std::isfinite(value) || (i == j && value <= T(0)))
                return false;
            matrix[i][j] = i == j ? std::sqrt(value) : value / matrix[j][j];
        }
        T value = rhs[i];
        for (unsigned j = 0; j < i; ++j)
            value -= matrix[i][j] * rhs[j];
        rhs[i] = value / matrix[i][i];
    }
    for (unsigned i = P; i-- > 0;) {
        T value = rhs[i];
        for (unsigned j = i + 1; j < P; ++j)
            value -= matrix[j][i] * rhs[j];
        rhs[i] = value / matrix[i][i];
        if (!std::isfinite(rhs[i]))
            return false;
    }
    if (scaled)
        for (unsigned i = 0; i < P; ++i) {
            rhs[i] /= scale[i];
            if (!std::isfinite(rhs[i]))
                return false;
        }
    return true;
}

template <typename T, typename Model, std::size_t P>
T cost(std::span<const T> x, std::span<const T> y, const std::array<T, P> &params,
       bool compensated = false) {
    T value = 0, correction = 0;
    for (std::size_t j = 0; j < x.size(); ++j) {
        const T residual = y[j] - Model::eval(x[j], params.data());
        if (compensated)
            numerics::add(residual * residual, value, correction);
        else
            value += residual * residual;
    }
    return value;
}

template <typename T, typename Model, std::size_t P>
bool linearize(std::span<const T> x, std::span<const T> y, const std::array<T, P> &params,
               Matrix<T, P> &matrix, std::array<T, P> &gradient, bool compensated = false) {
    matrix = {};
    gradient.fill(T(0));
    Matrix<T, P> matrix_correction{};
    std::array<T, P> gradient_correction{};
    for (std::size_t j = 0; j < x.size(); ++j) {
        const T residual = y[j] - Model::eval(x[j], params.data());
        T jacobian[P];
        Model::jacobian(x[j], params.data(), jacobian);
        for (unsigned k = 0; k < P; ++k) {
            if (compensated)
                numerics::add(jacobian[k] * residual, gradient[k], gradient_correction[k]);
            else
                gradient[k] += jacobian[k] * residual;
            for (unsigned l = 0; l <= k; ++l)
                if (compensated)
                    numerics::add(jacobian[k] * jacobian[l], matrix[k][l], matrix_correction[k][l]);
                else
                    matrix[k][l] += jacobian[k] * jacobian[l];
        }
    }
    for (unsigned k = 0; k < P; ++k) {
        if (!std::isfinite(gradient[k]))
            return false;
        for (unsigned l = 0; l <= k; ++l) {
            if (!std::isfinite(matrix[k][l]))
                return false;
            matrix[l][k] = matrix[k][l];
        }
    }
    return true;
}
} // namespace detail

// Model<T> supplies parameter_count, eval(x, params), and jacobian(x, params, out).
template <typename T, template <typename> class Model>
FitResult<T, Model<T>::parameter_count>
fit_single(std::span<const T> x, std::span<const T> y, std::span<const T> initial,
           const SolverOptions<T, Model<T>::parameter_count> &options = {}) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
    constexpr unsigned p = Model<T>::parameter_count;
    static_assert(p > 0 && p <= 8);
    FitResult<T, p> result;
    result.parameters.fill(std::numeric_limits<T>::quiet_NaN());
    if (initial.size() != p)
        return result;
    std::copy(initial.begin(), initial.end(), result.parameters.begin());
    if (x.empty() || x.size() != y.size() || !std::isfinite(options.initial_lambda) ||
        options.initial_lambda <= T(0))
        return result;
    for (const T tolerance :
         {options.gradient_tolerance, options.step_tolerance, options.cost_tolerance}) {
        if (!std::isfinite(tolerance) || tolerance < T(0))
            return result;
    }
    for (unsigned k = 0; k < p; ++k) {
        if (!std::isfinite(initial[k]) || std::isnan(options.lower[k]) ||
            std::isnan(options.upper[k]) || options.lower[k] > options.upper[k] ||
            options.lower[k] == std::numeric_limits<T>::infinity() ||
            options.upper[k] == -std::numeric_limits<T>::infinity())
            return result;
    }
    for (std::size_t j = 0; j < x.size(); ++j) {
        if (!std::isfinite(x[j]) || !std::isfinite(y[j]))
            return result;
    }
    for (unsigned k = 0; k < p; ++k) {
        result.parameters[k] = std::clamp(initial[k], options.lower[k], options.upper[k]);
    }
    result.chi_squared =
        detail::cost<T, Model<T>>(x, y, result.parameters, options.compensated_sum);
    if (!std::isfinite(result.chi_squared)) {
        result.chi_squared = std::numeric_limits<T>::quiet_NaN();
        return result;
    }
    const T spacing = detail::grid_spacing<T>(x);
    if constexpr (detail::spectral_model<T, Model<T>>) {
        const bool varied_x =
            std::any_of(x.begin(), x.end(), [&](T value) { return value != x[0]; });
        const bool flat = std::all_of(y.begin(), y.end(), [&](T value) { return value == y[0]; });
        if (varied_x && flat && detail::free_shape<T, Model<T>>(options)) {
            result.status = FitStatus::degenerate_data;
            return result;
        }
    }
    const auto success = [&] {
        result.status = detail::spectral_success<T, Model<T>>(result.parameters, spacing);
        return result;
    };
    result.status = FitStatus::max_iterations;
    T lambda = options.initial_lambda;
    const T maximum_lambda =
        std::max(options.initial_lambda, T(1) / std::numeric_limits<T>::epsilon());
    for (unsigned iteration = 0; iteration < options.max_iterations; ++iteration) {
        detail::Matrix<T, p> matrix;
        std::array<T, p> gradient;
        if (!detail::linearize<T, Model<T>>(x, y, result.parameters, matrix, gradient,
                                            options.compensated_sum)) {
            result.status = FitStatus::invalid_input;
            return result;
        }
        std::array<bool, p> active;
        T gradient_norm = 0;
        for (unsigned k = 0; k < p; ++k) {
            // Outward directions at a bound are not feasible descent directions.
            active[k] = options.lower[k] == options.upper[k] ||
                        (result.parameters[k] == options.lower[k] && gradient[k] < T(0)) ||
                        (result.parameters[k] == options.upper[k] && gradient[k] > T(0));
            if (!active[k])
                gradient_norm = std::max(gradient_norm, std::abs(gradient[k]));
        }
        if (gradient_norm <= options.gradient_tolerance) {
            return success();
        }
        auto damped = matrix;
        auto delta = gradient;
        for (unsigned k = 0; k < p; ++k) {
            if (active[k]) {
                for (unsigned l = 0; l < p; ++l)
                    damped[k][l] = damped[l][k] = T(0);
                damped[k][k] = T(1);
                delta[k] = T(0);
            } else {
                damped[k][k] += lambda * matrix[k][k];
            }
        }
        result.iterations = iteration + 1;
        if (!detail::cholesky_solve<T, p>(damped, delta, options.scaled_cholesky)) {
            result.status = FitStatus::singular_matrix;
            return result;
        }
        auto trial = result.parameters;
        T step_norm = 0, parameter_norm = 0, predicted_reduction = 0;
        bool finite_trial = true;
        for (unsigned k = 0; k < p; ++k) {
            step_norm = std::max(step_norm, std::abs(delta[k]));
            trial[k] =
                std::clamp(result.parameters[k] + delta[k], options.lower[k], options.upper[k]);
            finite_trial = finite_trial && std::isfinite(trial[k]);
            delta[k] = trial[k] - result.parameters[k];
            parameter_norm = std::max(parameter_norm, std::abs(result.parameters[k]));
        }
        // Remove damping's artificial shrinkage before declaring a small step.
        if (step_norm <= options.step_tolerance * (parameter_norm + options.step_tolerance) /
                             std::max(T(1), lambda)) {
            return success();
        }
        for (unsigned k = 0; k < p; ++k) {
            predicted_reduction += T(2) * gradient[k] * delta[k];
            for (unsigned l = 0; l < p; ++l)
                predicted_reduction -= delta[k] * matrix[k][l] * delta[l];
        }
        const T trial_cost = finite_trial
                                 ? detail::cost<T, Model<T>>(x, y, trial, options.compensated_sum)
                                 : std::numeric_limits<T>::quiet_NaN();
        const T reduction = result.chi_squared - trial_cost;
        const T cost_threshold = options.cost_tolerance * result.chi_squared;
        const bool small_cost_change =
            std::isfinite(trial_cost) && std::isfinite(predicted_reduction) &&
            predicted_reduction >= T(0) && std::abs(reduction) <= cost_threshold &&
            predicted_reduction <= cost_threshold / std::max(T(1), lambda);
        if (std::isfinite(trial_cost) && reduction > T(0) && std::isfinite(predicted_reduction) &&
            predicted_reduction > T(0)) {
            result.parameters = trial;
            result.chi_squared = trial_cost;
            const T ratio = reduction / predicted_reduction;
            const T factor =
                ratio >= T(1)
                    ? T(1) / T(3)
                    : std::max(T(1) / T(3), T(1) - (T(2) * ratio - T(1)) * (T(2) * ratio - T(1)) *
                                                       (T(2) * ratio - T(1)));
            lambda = std::clamp(lambda * factor, std::numeric_limits<T>::epsilon(), maximum_lambda);
        } else {
            // Cap damping to avoid manufacturing non-finite systems on repeated rejection.
            lambda = std::min(maximum_lambda, lambda * T(10));
        }
        if (small_cost_change) {
            return success();
        }
    }
    return result;
}
} // namespace cpu
} // namespace batchfit
