#include <batchfit/metal/types.hpp>
#include <batchfit/models.hpp>
#include <metal_stdlib>
#pragma METAL fp contract(on)
using namespace metal;
using batchfit::metal::detail::Config;

enum : uint { converged, max_iterations, singular_matrix, invalid_input };
template <uint P> struct LMResult {
    float parameters[P];
    uint status, iterations;
    float chi_squared;
};

template <typename Model>
float cost(device const float *x, device const float *y, constant Config &cfg, uint fit,
           thread const float *params) {
    float value = 0;
    for (uint j = 0; j < cfg.points; ++j) {
        const float residual = y[j * cfg.fits + fit] - Model::eval(x[j], params);
        value += residual * residual;
    }
    return value;
}

template <uint P> bool cholesky(thread float matrix[P][P], thread float rhs[P]) {
    for (uint i = 0; i < P; ++i) {
        for (uint j = 0; j <= i; ++j) {
            float value = matrix[i][j];
            for (uint k = 0; k < j; ++k)
                value -= matrix[i][k] * matrix[j][k];
            if (!isfinite(value) || (i == j && value <= 0))
                return false;
            matrix[i][j] = i == j ? sqrt(value) : value / matrix[j][j];
        }
        float value = rhs[i];
        for (uint j = 0; j < i; ++j)
            value -= matrix[i][j] * rhs[j];
        rhs[i] = value / matrix[i][i];
    }
    for (uint i = P; i-- > 0;) {
        float value = rhs[i];
        for (uint j = i + 1; j < P; ++j)
            value -= matrix[j][i] * rhs[j];
        rhs[i] = value / matrix[i][i];
        if (!isfinite(rhs[i]))
            return false;
    }
    return true;
}

template <typename Model>
LMResult<Model::parameter_count> solve(device const float *x, device const float *y,
                                       device const float *initial, constant Config &cfg,
                                       uint fit) {
    constexpr uint P = Model::parameter_count;
    LMResult<P> result;
    result.status = invalid_input;
    result.iterations = 0;
    result.chi_squared = NAN;
    for (uint k = 0; k < P; ++k)
        result.parameters[k] = initial[fit * P + k];
    if (!isfinite(cfg.initial_lambda) || cfg.initial_lambda <= 0)
        return result;
    const float tolerances[3] = {cfg.gradient_tolerance, cfg.step_tolerance, cfg.cost_tolerance};
    for (uint k = 0; k < 3; ++k)
        if (!isfinite(tolerances[k]) || tolerances[k] < 0)
            return result;
    for (uint k = 0; k < P; ++k)
        if (!isfinite(result.parameters[k]) || isnan(cfg.lower[k]) || isnan(cfg.upper[k]) ||
            cfg.lower[k] > cfg.upper[k] || cfg.lower[k] == INFINITY || cfg.upper[k] == -INFINITY)
            return result;
    for (uint j = 0; j < cfg.points; ++j)
        if (!isfinite(x[j]) || !isfinite(y[j * cfg.fits + fit]))
            return result;
    for (uint k = 0; k < P; ++k)
        result.parameters[k] = clamp(result.parameters[k], cfg.lower[k], cfg.upper[k]);
    result.chi_squared = cost<Model>(x, y, cfg, fit, result.parameters);
    if (!isfinite(result.chi_squared)) {
        result.chi_squared = NAN;
        return result;
    }
    result.status = max_iterations;
    constexpr float epsilon = 0x1p-23f;
    float lambda = cfg.initial_lambda;
    const float maximum_lambda = max(cfg.initial_lambda, 1.0f / epsilon);
    for (uint iteration = 0; iteration < cfg.max_iterations; ++iteration) {
        float matrix[P][P], gradient[P];
        for (uint k = 0; k < P; ++k) {
            gradient[k] = 0;
            for (uint l = 0; l < P; ++l)
                matrix[k][l] = 0;
        }
        for (uint j = 0; j < cfg.points; ++j) {
            const float residual = y[j * cfg.fits + fit] - Model::eval(x[j], result.parameters);
            float jacobian[P];
            Model::jacobian(x[j], result.parameters, jacobian);
            for (uint k = 0; k < P; ++k) {
                gradient[k] += jacobian[k] * residual;
                for (uint l = 0; l <= k; ++l)
                    matrix[k][l] += jacobian[k] * jacobian[l];
            }
        }
        for (uint k = 0; k < P; ++k) {
            if (!isfinite(gradient[k])) {
                result.status = invalid_input;
                return result;
            }
            for (uint l = 0; l <= k; ++l) {
                if (!isfinite(matrix[k][l])) {
                    result.status = invalid_input;
                    return result;
                }
                matrix[l][k] = matrix[k][l];
            }
        }
        bool active[P];
        float gradient_norm = 0;
        for (uint k = 0; k < P; ++k) {
            active[k] = cfg.lower[k] == cfg.upper[k] ||
                        (result.parameters[k] == cfg.lower[k] && gradient[k] < 0) ||
                        (result.parameters[k] == cfg.upper[k] && gradient[k] > 0);
            if (!active[k])
                gradient_norm = max(gradient_norm, abs(gradient[k]));
        }
        if (gradient_norm <= cfg.gradient_tolerance) {
            result.status = converged;
            return result;
        }
        float damped[P][P], delta[P];
        for (uint k = 0; k < P; ++k) {
            delta[k] = active[k] ? 0 : gradient[k];
            for (uint l = 0; l < P; ++l)
                damped[k][l] = active[k] || active[l] ? (k == l ? 1.0f : 0.0f) : matrix[k][l];
            if (!active[k])
                damped[k][k] += lambda * matrix[k][k];
        }
        result.iterations = iteration + 1;
        if (!cholesky<P>(damped, delta)) {
            result.status = singular_matrix;
            return result;
        }
        float trial[P], step_norm = 0, parameter_norm = 0, predicted_reduction = 0;
        bool finite_trial = true;
        for (uint k = 0; k < P; ++k) {
            step_norm = max(step_norm, abs(delta[k]));
            trial[k] = clamp(result.parameters[k] + delta[k], cfg.lower[k], cfg.upper[k]);
            finite_trial = finite_trial && isfinite(trial[k]);
            delta[k] = trial[k] - result.parameters[k];
            parameter_norm = max(parameter_norm, abs(result.parameters[k]));
        }
        if (step_norm <=
            cfg.step_tolerance * (parameter_norm + cfg.step_tolerance) / max(1.0f, lambda)) {
            result.status = converged;
            return result;
        }
        for (uint k = 0; k < P; ++k) {
            predicted_reduction += 2.0f * gradient[k] * delta[k];
            for (uint l = 0; l < P; ++l)
                predicted_reduction -= delta[k] * matrix[k][l] * delta[l];
        }
        const float trial_cost = finite_trial ? cost<Model>(x, y, cfg, fit, trial) : NAN;
        const float reduction = result.chi_squared - trial_cost;
        const float threshold = cfg.cost_tolerance * result.chi_squared;
        const bool small_cost_change = isfinite(trial_cost) && isfinite(predicted_reduction) &&
                                       predicted_reduction >= 0 && abs(reduction) <= threshold &&
                                       predicted_reduction <= threshold / max(1.0f, lambda);
        if (isfinite(trial_cost) && reduction > 0 && isfinite(predicted_reduction) &&
            predicted_reduction > 0) {
            for (uint k = 0; k < P; ++k)
                result.parameters[k] = trial[k];
            result.chi_squared = trial_cost;
            const float ratio = reduction / predicted_reduction;
            const float factor =
                ratio >= 1 ? 1.0f / 3
                           : max(1.0f / 3, 1.0f - (2.0f * ratio - 1) * (2.0f * ratio - 1) *
                                                      (2.0f * ratio - 1));
            lambda = clamp(lambda * factor, epsilon, maximum_lambda);
        } else
            lambda = min(maximum_lambda, lambda * 10.0f);
        if (small_cost_change) {
            result.status = converged;
            return result;
        }
    }
    return result;
}

template <typename Model>
void run(device const float *x, device const float *y, device const float *initial,
         constant Config &cfg, device float *parameters, device uint *status,
         device uint *iterations, device float *chi, uint fit) {
    if (fit >= cfg.fits)
        return;
    const auto result = solve<Model>(x, y, initial, cfg, fit);
    for (uint k = 0; k < Model::parameter_count; ++k)
        parameters[fit * Model::parameter_count + k] = result.parameters[k];
    status[fit] = result.status;
    iterations[fit] = result.iterations;
    chi[fit] = result.chi_squared;
}

#define BATCHFIT_KERNEL(NAME, MODEL)                                                               \
    kernel void NAME(device const float *x [[buffer(0)]], device const float *y [[buffer(1)]],     \
                     device const float *initial [[buffer(2)]],                                    \
                     constant Config &cfg [[buffer(3)]], device float *parameters [[buffer(4)]],   \
                     device uint *status [[buffer(5)]], device uint *iterations [[buffer(6)]],     \
                     device float *chi [[buffer(7)]], uint fit [[thread_position_in_grid]]) {      \
        run<batchfit::models::MODEL<float>>(x, y, initial, cfg, parameters, status, iterations,    \
                                            chi, fit);                                             \
    }
BATCHFIT_KERNEL(fit_gaussian, Gaussian)
BATCHFIT_KERNEL(fit_lorentzian, Lorentzian)
BATCHFIT_KERNEL(fit_pseudo_voigt, PseudoVoigt)
BATCHFIT_KERNEL(fit_two_lorentzian, TwoLorentzian)
