#include <batchfit/metal/types.hpp>
#include <batchfit/models.hpp>
#include <batchfit/numerics.hpp>
#include <metal_stdlib>
#pragma METAL fp contract(on)
using namespace metal;
using batchfit::metal::detail::Config;
constant uint requested_numerics [[function_constant(0)]];
constant uint numerical_mode =
    is_function_constant_defined(requested_numerics) ? requested_numerics : 0u;

enum : uint {
    converged,
    max_iterations,
    singular_matrix,
    invalid_input,
    degenerate_data,
    unresolved_peak
};
bool finite_value(float value) { return (as_type<uint>(value) & 0x7f800000u) != 0x7f800000u; }
bool nan_value(float value) { return (as_type<uint>(value) & 0x7fffffffu) > 0x7f800000u; }
template <uint P> bool free_shape(constant Config &cfg) {
    bool free = cfg.lower[1] != cfg.upper[1] || cfg.lower[2] != cfg.upper[2];
    if (P == 4)
        free = free || cfg.lower[3] != cfg.upper[3];
    if (P == 8)
        free = free || cfg.lower[4] != cfg.upper[4] || cfg.lower[5] != cfg.upper[5];
    return free;
}
uint resolved_status(float first, float second, uint p, float spacing, float factor) {
    return spacing > 0 &&
                   (abs(first) * factor < spacing || (p == 8 && abs(second) * factor < spacing))
               ? unresolved_peak
               : converged;
}
template <uint P> struct LMResult {
    float parameters[P];
    uint status, iterations;
    float chi_squared;
};

template <typename Sample, bool PointMajor>
float observation(device const Sample *y, constant Config &cfg, uint fit, uint point) {
    return float(y[PointMajor ? point * cfg.fits + fit : fit * cfg.points + point]);
}

template <typename Model, typename Sample = float, bool PointMajor = true>
float cost(device const float *x, device const Sample *y, constant Config &cfg, uint fit,
           thread const float *params) {
    float value = 0, correction = 0;
    for (uint j = 0; j < cfg.points; ++j) {
        const float residual =
            observation<Sample, PointMajor>(y, cfg, fit, j) - Model::eval(x[j], params);
        if (numerical_mode & 1u)
            batchfit::numerics::add(residual * residual, value, correction);
        else
            value += residual * residual;
    }
    return value;
}

template <uint P>
bool cholesky(thread float matrix[P][P], thread float rhs[P], bool scaled = false) {
    float scale[P];
    if (scaled) {
        for (uint i = 0; i < P; ++i) {
            if (!finite_value(matrix[i][i]) || matrix[i][i] <= 0)
                return false;
            scale[i] = sqrt(matrix[i][i]);
        }
        for (uint i = 0; i < P; ++i) {
            rhs[i] /= scale[i];
            for (uint j = 0; j < P; ++j)
                matrix[i][j] = (matrix[i][j] / scale[i]) / scale[j];
        }
    }
    for (uint i = 0; i < P; ++i) {
        for (uint j = 0; j <= i; ++j) {
            float value = matrix[i][j];
            for (uint k = 0; k < j; ++k)
                value -= matrix[i][k] * matrix[j][k];
            if (!finite_value(value) || (i == j && value <= 0))
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
        if (!finite_value(rhs[i]))
            return false;
    }
    if (scaled)
        for (uint i = 0; i < P; ++i) {
            rhs[i] /= scale[i];
            if (!finite_value(rhs[i]))
                return false;
        }
    return true;
}

template <typename Model, typename Sample = float, bool PointMajor = true>
LMResult<Model::parameter_count> solve(device const float *x, device const Sample *y,
                                       device const float *initial, constant Config &cfg,
                                       uint fit) {
    constexpr uint P = Model::parameter_count;
    LMResult<P> result;
    result.status = invalid_input;
    result.iterations = 0;
    result.chi_squared = NAN;
    for (uint k = 0; k < P; ++k)
        result.parameters[k] = initial[fit * P + k];
    if (cfg.numerics != numerical_mode)
        return result;
    if (!finite_value(cfg.initial_lambda) || cfg.initial_lambda <= 0)
        return result;
    const float tolerances[3] = {cfg.gradient_tolerance, cfg.step_tolerance, cfg.cost_tolerance};
    for (uint k = 0; k < 3; ++k)
        if (!finite_value(tolerances[k]) || tolerances[k] < 0)
            return result;
    for (uint k = 0; k < P; ++k)
        if (!finite_value(result.parameters[k]) || nan_value(cfg.lower[k]) ||
            nan_value(cfg.upper[k]) || cfg.lower[k] > cfg.upper[k] ||
            as_type<uint>(cfg.lower[k]) == 0x7f800000u ||
            as_type<uint>(cfg.upper[k]) == 0xff800000u)
            return result;
    bool varied_y = false;
    for (uint j = 0; j < cfg.points; ++j) {
        if (!finite_value(x[j]) || !finite_value(observation<Sample, PointMajor>(y, cfg, fit, j)))
            return result;
        varied_y = varied_y || observation<Sample, PointMajor>(y, cfg, fit, j) !=
                                   observation<Sample, PointMajor>(y, cfg, fit, 0);
    }
    for (uint k = 0; k < P; ++k)
        result.parameters[k] = clamp(result.parameters[k], cfg.lower[k], cfg.upper[k]);
    result.chi_squared = cost<Model, Sample, PointMajor>(x, y, cfg, fit, result.parameters);
    if (!finite_value(result.chi_squared)) {
        result.chi_squared = NAN;
        return result;
    }
    if (!varied_y && cfg.varied_x && free_shape<P>(cfg)) {
        result.status = degenerate_data;
        return result;
    }
    result.status = max_iterations;
    constexpr float epsilon = 0x1p-23f;
    float lambda = cfg.initial_lambda;
    const float maximum_lambda = max(cfg.initial_lambda, 1.0f / epsilon);
    for (uint iteration = 0; iteration < cfg.max_iterations; ++iteration) {
        float matrix[P][P], gradient[P], matrix_correction[P][P], gradient_correction[P];
        for (uint k = 0; k < P; ++k) {
            gradient[k] = 0;
            gradient_correction[k] = 0;
            for (uint l = 0; l < P; ++l) {
                matrix[k][l] = 0;
                matrix_correction[k][l] = 0;
            }
        }
        for (uint j = 0; j < cfg.points; ++j) {
            const float residual = observation<Sample, PointMajor>(y, cfg, fit, j) -
                                   Model::eval(x[j], result.parameters);
            float jacobian[P];
            Model::jacobian(x[j], result.parameters, jacobian);
            for (uint k = 0; k < P; ++k) {
                if (numerical_mode & 1u)
                    batchfit::numerics::add(jacobian[k] * residual, gradient[k],
                                            gradient_correction[k]);
                else
                    gradient[k] += jacobian[k] * residual;
                for (uint l = 0; l <= k; ++l)
                    if (numerical_mode & 1u)
                        batchfit::numerics::add(jacobian[k] * jacobian[l], matrix[k][l],
                                                matrix_correction[k][l]);
                    else
                        matrix[k][l] += jacobian[k] * jacobian[l];
            }
        }
        for (uint k = 0; k < P; ++k) {
            if (!finite_value(gradient[k])) {
                result.status = invalid_input;
                return result;
            }
            for (uint l = 0; l <= k; ++l) {
                if (!finite_value(matrix[k][l])) {
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
            result.status = resolved_status(result.parameters[2], result.parameters[min(P - 1, 5u)],
                                            P, cfg.minimum_spacing, cfg.width_factor);
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
        if (!cholesky<P>(damped, delta, (numerical_mode & 2u) != 0)) {
            result.status = singular_matrix;
            return result;
        }
        float trial[P], step_norm = 0, parameter_norm = 0, predicted_reduction = 0;
        bool finite_trial = true;
        for (uint k = 0; k < P; ++k) {
            step_norm = max(step_norm, abs(delta[k]));
            trial[k] = clamp(result.parameters[k] + delta[k], cfg.lower[k], cfg.upper[k]);
            finite_trial = finite_trial && finite_value(trial[k]);
            delta[k] = trial[k] - result.parameters[k];
            parameter_norm = max(parameter_norm, abs(result.parameters[k]));
        }
        if (step_norm <=
            cfg.step_tolerance * (parameter_norm + cfg.step_tolerance) / max(1.0f, lambda)) {
            result.status = resolved_status(result.parameters[2], result.parameters[min(P - 1, 5u)],
                                            P, cfg.minimum_spacing, cfg.width_factor);
            return result;
        }
        for (uint k = 0; k < P; ++k) {
            predicted_reduction += 2.0f * gradient[k] * delta[k];
            for (uint l = 0; l < P; ++l)
                predicted_reduction -= delta[k] * matrix[k][l] * delta[l];
        }
        const float trial_cost =
            finite_trial ? cost<Model, Sample, PointMajor>(x, y, cfg, fit, trial) : NAN;
        const float reduction = result.chi_squared - trial_cost;
        const float threshold = cfg.cost_tolerance * result.chi_squared;
        const bool small_cost_change = finite_value(trial_cost) &&
                                       finite_value(predicted_reduction) &&
                                       predicted_reduction >= 0 && abs(reduction) <= threshold &&
                                       predicted_reduction <= threshold / max(1.0f, lambda);
        if (finite_value(trial_cost) && reduction > 0 && finite_value(predicted_reduction) &&
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
            result.status = resolved_status(result.parameters[2], result.parameters[min(P - 1, 5u)],
                                            P, cfg.minimum_spacing, cfg.width_factor);
            return result;
        }
    }
    return result;
}

template <typename Model, typename Sample = float, bool PointMajor = true>
void run(device const float *x, device const Sample *y, device const float *initial,
         constant Config &cfg, device float *parameters, device uint *status,
         device uint *iterations, device float *chi, uint fit) {
    if (fit >= cfg.fits)
        return;
    const auto result = solve<Model, Sample, PointMajor>(x, y, initial, cfg, fit);
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
