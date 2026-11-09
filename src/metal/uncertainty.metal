#include <batchfit/uncertainty_types.hpp>
using batchfit::UncertaintyStatus;

template <typename Model>
void uncertainty_run(device const float *x, constant Config &cfg, device const float *parameters,
                     device const uint *status, device const float *chi, device float *covariance,
                     device float *errors, device uint *uncertainty_status, uint fit) {
    constexpr uint P = Model::parameter_count;
    if (fit >= cfg.fits)
        return;
    for (uint i = 0; i < P * P; ++i)
        covariance[fit * P * P + i] = NAN;
    for (uint i = 0; i < P; ++i)
        errors[fit * P + i] = NAN;
    uncertainty_status[fit] = uint(UncertaintyStatus::not_converged);
    if (status[fit] != converged)
        return;
    uncertainty_status[fit] = uint(UncertaintyStatus::invalid_input);
    if (!finite_value(chi[fit]) || chi[fit] < 0)
        return;
    float params[P];
    bool fixed[P];
    uint free = 0;
    for (uint i = 0; i < P; ++i) {
        params[i] = parameters[fit * P + i];
        if (!finite_value(params[i]) || nan_value(cfg.lower[i]) || nan_value(cfg.upper[i]) ||
            cfg.lower[i] > cfg.upper[i] || params[i] < cfg.lower[i] || params[i] > cfg.upper[i])
            return;
        fixed[i] = cfg.lower[i] == cfg.upper[i];
        free += !fixed[i];
        if (!fixed[i] && (params[i] == cfg.lower[i] || params[i] == cfg.upper[i])) {
            uncertainty_status[fit] = uint(UncertaintyStatus::active_bound);
            return;
        }
    }
    if (cfg.points <= free) {
        uncertainty_status[fit] = uint(UncertaintyStatus::insufficient_dof);
        return;
    }
    float information[P][P], correction[P][P];
    for (uint i = 0; i < P; ++i)
        for (uint j = 0; j < P; ++j) {
            information[i][j] = 0;
            correction[i][j] = 0;
        }
    for (uint point = 0; point < cfg.points; ++point) {
        if (!finite_value(x[point]))
            return;
        float jacobian[P];
        Model::jacobian(x[point], params, jacobian);
        for (uint i = 0; i < P; ++i) {
            if (!finite_value(jacobian[i]))
                return;
            for (uint j = 0; j <= i; ++j)
                batchfit::numerics::add(jacobian[i] * jacobian[j], information[i][j],
                                        correction[i][j]);
        }
    }
    // Cholesky reads only the lower triangle; do not mirror private arrays in place.
    for (uint i = 0; i < P; ++i)
        for (uint j = 0; j <= i; ++j) {
            if (!finite_value(information[i][j]))
                return;
        }
    const float variance = chi[fit] / float(cfg.points - free);
    if (!finite_value(variance))
        return;
    float cov[P][P];
    for (uint i = 0; i < P; ++i)
        for (uint j = 0; j < P; ++j)
            cov[i][j] = 0;
    for (uint column = 0; column < P; ++column) {
        if (fixed[column])
            continue;
        float factor[P][P], rhs[P];
        for (uint i = 0; i < P; ++i) {
            rhs[i] = i == column ? 1.0f : 0.0f;
            for (uint j = 0; j < P; ++j)
                factor[i][j] = fixed[i] || fixed[j] ? (i == j ? 1.0f : 0.0f) : information[i][j];
        }
        if (!cholesky<P>(factor, rhs, true)) {
            uncertainty_status[fit] = uint(UncertaintyStatus::singular_matrix);
            return;
        }
        for (uint i = 0; i < P; ++i) {
            if (!fixed[i] && factor[i][i] * factor[i][i] <= 64.0f * 0x1p-23f) {
                uncertainty_status[fit] = uint(UncertaintyStatus::singular_matrix);
                return;
            }
            cov[i][column] = fixed[i] ? 0.0f : rhs[i] * variance;
            if (!finite_value(cov[i][column]))
                return;
        }
    }
    for (uint i = 0; i < P; ++i) {
        if (cov[i][i] < 0)
            return;
    }
    for (uint i = 0; i < P; ++i) {
        errors[fit * P + i] = sqrt(cov[i][i]);
        for (uint j = 0; j < P; ++j)
            covariance[fit * P * P + i * P + j] = cov[i][j] / 2.0f + cov[j][i] / 2.0f;
    }
    uncertainty_status[fit] = uint(UncertaintyStatus::available);
}

#define UNCERTAINTY_KERNEL(NAME, MODEL)                                                            \
    kernel void NAME(                                                                              \
        device const float *x [[buffer(0)]], constant Config &cfg [[buffer(3)]],                   \
        device const float *parameters [[buffer(4)]], device const uint *status [[buffer(5)]],     \
        device const float *chi [[buffer(7)]], device float *covariance [[buffer(8)]],             \
        device float *errors [[buffer(9)]], device uint *uncertainty_status [[buffer(10)]],        \
        uint fit [[thread_position_in_grid]]) {                                                    \
        uncertainty_run<batchfit::models::MODEL<float>>(                                           \
            x, cfg, parameters, status, chi, covariance, errors, uncertainty_status, fit);         \
    }
UNCERTAINTY_KERNEL(uncertainty_gaussian, Gaussian)
UNCERTAINTY_KERNEL(uncertainty_lorentzian, Lorentzian)
UNCERTAINTY_KERNEL(uncertainty_pseudo_voigt, PseudoVoigt)
UNCERTAINTY_KERNEL(uncertainty_two_lorentzian, TwoLorentzian)
