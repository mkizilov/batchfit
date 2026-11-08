// Cooperative variants retain v1's control policy, but reduce point sums in parallel.
enum : uint { coop_running = 6 };
template <uint P> struct CooperativeState {
    float parameters[P], matrix[P][P], damped[P][P], gradient[P], delta[P], trial[P];
    float chi, lambda, predicted, spacing, width_factor;
    uint status, iterations, finite_trial, flat;
};

template <bool WholeGroup> void cooperative_barrier() {
    if (WholeGroup)
        threadgroup_barrier(mem_flags::mem_threadgroup);
    else
        simdgroup_barrier(mem_flags::mem_threadgroup);
}

template <bool WholeGroup>
float cooperative_sum(float value, threadgroup float *sums, uint lane, uint simd, uint groups) {
    const float warp = simd_sum(value);
    if (!WholeGroup)
        return warp;
    if (lane == 0)
        sums[simd] = warp;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0 && lane == 0) {
        float total = 0;
        for (uint i = 0; i < groups; ++i)
            total += sums[i];
        sums[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return sums[0];
}

template <bool WholeGroup>
uint cooperative_flags(uint value, threadgroup float *sums, uint lane, uint simd, uint groups) {
    const uint warp = simd_or(value);
    if (!WholeGroup)
        return warp;
    if (lane == 0)
        sums[simd] = float(warp);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0 && lane == 0) {
        uint total = 0;
        for (uint i = 0; i < groups; ++i)
            total |= uint(sums[i]);
        sums[0] = float(total);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return uint(sums[0]);
}

template <uint P>
bool cooperative_cholesky(threadgroup float matrix[P][P], threadgroup float rhs[P],
                          bool scaled = false) {
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

template <uint P>
void cooperative_initialize(threadgroup CooperativeState<P> &state, device const float *initial,
                            constant Config &cfg, uint fit) {
    state.status = invalid_input;
    state.iterations = 0;
    state.chi = NAN;
    state.spacing = cfg.minimum_spacing;
    state.width_factor = cfg.width_factor;
    state.flat = 0;
    for (uint k = 0; k < P; ++k)
        state.parameters[k] = initial[fit * P + k];
    if (cfg.numerics != numerical_mode)
        return;
    if (!finite_value(cfg.initial_lambda) || cfg.initial_lambda <= 0)
        return;
    const float tolerances[3] = {cfg.gradient_tolerance, cfg.step_tolerance, cfg.cost_tolerance};
    for (uint k = 0; k < 3; ++k)
        if (!finite_value(tolerances[k]) || tolerances[k] < 0)
            return;
    for (uint k = 0; k < P; ++k)
        if (!finite_value(state.parameters[k]) || nan_value(cfg.lower[k]) ||
            nan_value(cfg.upper[k]) || cfg.lower[k] > cfg.upper[k] ||
            as_type<uint>(cfg.lower[k]) == 0x7f800000u ||
            as_type<uint>(cfg.upper[k]) == 0xff800000u)
            return;
    // Projection occurs after shared-data validation, as in the CPU reference.
    state.lambda = cfg.initial_lambda;
    state.status = coop_running;
}

template <uint P>
void cooperative_step(threadgroup CooperativeState<P> &state, constant Config &cfg,
                      uint iteration) {
    for (uint k = 0; k < P; ++k) {
        if (!finite_value(state.gradient[k])) {
            state.status = invalid_input;
            return;
        }
        for (uint l = 0; l <= k; ++l) {
            if (!finite_value(state.matrix[k][l])) {
                state.status = invalid_input;
                return;
            }
            state.matrix[l][k] = state.matrix[k][l];
        }
    }
    bool active[P];
    float norm = 0;
    for (uint k = 0; k < P; ++k) {
        active[k] = cfg.lower[k] == cfg.upper[k] ||
                    (state.parameters[k] == cfg.lower[k] && state.gradient[k] < 0) ||
                    (state.parameters[k] == cfg.upper[k] && state.gradient[k] > 0);
        if (!active[k])
            norm = max(norm, abs(state.gradient[k]));
    }
    if (norm <= cfg.gradient_tolerance) {
        state.status = converged;
        return;
    }
    for (uint k = 0; k < P; ++k) {
        state.delta[k] = active[k] ? 0 : state.gradient[k];
        for (uint l = 0; l < P; ++l)
            state.damped[k][l] =
                active[k] || active[l] ? (k == l ? 1.0f : 0.0f) : state.matrix[k][l];
        if (!active[k])
            state.damped[k][k] += state.lambda * state.matrix[k][k];
    }
    state.iterations = iteration + 1;
    if (!cooperative_cholesky<P>(state.damped, state.delta, (numerical_mode & 2u) != 0)) {
        state.status = singular_matrix;
        return;
    }
    float step_norm = 0, parameter_norm = 0;
    state.finite_trial = 1;
    for (uint k = 0; k < P; ++k) {
        step_norm = max(step_norm, abs(state.delta[k]));
        parameter_norm = max(parameter_norm, abs(state.parameters[k]));
        state.trial[k] = clamp(state.parameters[k] + state.delta[k], cfg.lower[k], cfg.upper[k]);
        state.finite_trial = state.finite_trial && finite_value(state.trial[k]);
    }
    if (step_norm <=
        cfg.step_tolerance * (parameter_norm + cfg.step_tolerance) / max(1.0f, state.lambda)) {
        state.status = converged;
        return;
    }
    state.predicted = 0;
    for (uint k = 0; k < P; ++k) {
        const float delta = state.trial[k] - state.parameters[k];
        state.predicted += 2.0f * state.gradient[k] * delta;
        for (uint l = 0; l < P; ++l)
            state.predicted -= delta * state.matrix[k][l] * (state.trial[l] - state.parameters[l]);
    }
}

template <uint P>
void cooperative_accept(threadgroup CooperativeState<P> &state, constant Config &cfg,
                        float trial_cost) {
    const float reduction = state.chi - trial_cost, threshold = cfg.cost_tolerance * state.chi;
    const bool small = finite_value(trial_cost) && finite_value(state.predicted) &&
                       state.predicted >= 0 && abs(reduction) <= threshold &&
                       state.predicted <= threshold / max(1.0f, state.lambda);
    constexpr float epsilon = 0x1p-23f;
    const float maximum = max(cfg.initial_lambda, 1.0f / epsilon);
    if (finite_value(trial_cost) && reduction > 0 && finite_value(state.predicted) &&
        state.predicted > 0) {
        for (uint k = 0; k < P; ++k)
            state.parameters[k] = state.trial[k];
        state.chi = trial_cost;
        const float ratio = reduction / state.predicted;
        const float factor = ratio >= 1
                                 ? 1.0f / 3
                                 : max(1.0f / 3, 1.0f - (2.0f * ratio - 1) * (2.0f * ratio - 1) *
                                                            (2.0f * ratio - 1));
        state.lambda = clamp(state.lambda * factor, epsilon, maximum);
    } else
        state.lambda = min(maximum, state.lambda * 10.0f);
    if (small)
        state.status = converged;
}

template <uint P>
void cooperative_store(threadgroup CooperativeState<P> &state, device float *parameters,
                       device uint *status, device uint *iterations, device float *chi, uint fit) {
    for (uint k = 0; k < P; ++k)
        parameters[fit * P + k] = state.parameters[k];
    status[fit] = state.status == converged
                      ? resolved_status(state.parameters[2], state.parameters[min(P - 1, 5u)], P,
                                        state.spacing, state.width_factor)
                      : state.status;
    iterations[fit] = state.iterations;
    chi[fit] = state.chi;
}

template <typename Model, bool WholeGroup>
void cooperative_run(device const float *x, device const float *y, device const float *initial,
                     constant Config &cfg, device float *parameters, device uint *status,
                     device uint *iterations, device float *chi,
                     threadgroup CooperativeState<Model::parameter_count> *states,
                     threadgroup float *sums, uint group_id, uint thread_index, uint lane,
                     uint simd, uint group_size) {
    constexpr uint P = Model::parameter_count, triangle = P * (P + 1) / 2, terms = triangle + P;
    const uint groups = group_size / 32;
    const uint fit = WholeGroup ? group_id : group_id * groups + simd;
    if (fit >= cfg.fits)
        return; // uniform for the participating fit's collective
    threadgroup CooperativeState<P> &state = states[WholeGroup ? 0 : simd];
    const bool leader = WholeGroup ? thread_index == 0 : lane == 0;
    const uint point_lane = WholeGroup ? thread_index : lane, stride = WholeGroup ? group_size : 32;
    if (leader)
        cooperative_initialize<P>(state, initial, cfg, fit);
    cooperative_barrier<WholeGroup>();
    if (state.status != coop_running) {
        if (leader)
            cooperative_store<P>(state, parameters, status, iterations, chi, fit);
        return;
    }
    uint input_flags = 0;
    for (uint j = point_lane; j < cfg.points; j += stride) {
        if (!finite_value(x[j]) || !finite_value(y[j * cfg.fits + fit]))
            input_flags |= 1u;
        if (y[j * cfg.fits + fit] != y[fit])
            input_flags |= 2u;
    }
    input_flags = cooperative_flags<WholeGroup>(input_flags, sums, lane, simd, groups);
    if (leader) {
        if (input_flags & 1u)
            state.status = invalid_input;
        else {
            state.flat = !(input_flags & 2u) && cfg.varied_x && free_shape<P>(cfg);
            for (uint k = 0; k < P; ++k)
                state.parameters[k] = clamp(state.parameters[k], cfg.lower[k], cfg.upper[k]);
        }
    }
    cooperative_barrier<WholeGroup>();
    if (state.status != coop_running) {
        if (leader)
            cooperative_store<P>(state, parameters, status, iterations, chi, fit);
        return;
    }
    float params[P], local_cost = 0, cost_correction = 0;
    for (uint k = 0; k < P; ++k)
        params[k] = simd_broadcast(lane == 0 ? state.parameters[k] : 0.0f, 0);
    for (uint j = point_lane; j < cfg.points; j += stride) {
        const float r = y[j * cfg.fits + fit] - Model::eval(x[j], params);
        if (numerical_mode & 1u)
            batchfit::numerics::add(r * r, local_cost, cost_correction);
        else
            local_cost += r * r;
    }
    const float initial_cost = cooperative_sum<WholeGroup>(local_cost, sums, lane, simd, groups);
    if (leader) {
        state.chi = finite_value(initial_cost) ? initial_cost : NAN;
        if (!finite_value(initial_cost))
            state.status = invalid_input;
        else if (state.flat)
            state.status = degenerate_data;
        else if (cfg.max_iterations == 0)
            state.status = max_iterations;
    }
    cooperative_barrier<WholeGroup>();
    if (state.status != coop_running) {
        if (leader)
            cooperative_store<P>(state, parameters, status, iterations, chi, fit);
        return;
    }
    for (uint iteration = 0; iteration < cfg.max_iterations; ++iteration) {
        for (uint k = 0; k < P; ++k)
            params[k] = simd_broadcast(lane == 0 ? state.parameters[k] : 0.0f, 0);
        float partial[terms], corrections[terms];
        for (uint q = 0; q < terms; ++q) {
            partial[q] = 0;
            corrections[q] = 0;
        }
        for (uint j = point_lane; j < cfg.points; j += stride) {
            const float residual = y[j * cfg.fits + fit] - Model::eval(x[j], params);
            float jacobian[P];
            Model::jacobian(x[j], params, jacobian);
            for (uint k = 0; k < P; ++k) {
                if (numerical_mode & 1u)
                    batchfit::numerics::add(jacobian[k] * residual, partial[triangle + k],
                                            corrections[triangle + k]);
                else
                    partial[triangle + k] += jacobian[k] * residual;
                for (uint l = 0; l <= k; ++l)
                    if (numerical_mode & 1u)
                        batchfit::numerics::add(jacobian[k] * jacobian[l],
                                                partial[k * (k + 1) / 2 + l],
                                                corrections[k * (k + 1) / 2 + l]);
                    else
                        partial[k * (k + 1) / 2 + l] += jacobian[k] * jacobian[l];
            }
        }
        for (uint q = 0; q < terms; ++q) {
            const float value = simd_sum(partial[q]);
            if (lane == 0)
                sums[simd * terms + q] = value;
        }
        cooperative_barrier<WholeGroup>();
        if (leader) {
            for (uint k = 0; k < P; ++k) {
                float g = 0;
                for (uint s = WholeGroup ? 0 : simd; s < (WholeGroup ? groups : simd + 1); ++s)
                    g += sums[s * terms + triangle + k];
                state.gradient[k] = g;
                for (uint l = 0; l <= k; ++l) {
                    float a = 0;
                    for (uint s = WholeGroup ? 0 : simd; s < (WholeGroup ? groups : simd + 1); ++s)
                        a += sums[s * terms + k * (k + 1) / 2 + l];
                    state.matrix[k][l] = a;
                }
            }
            cooperative_step<P>(state, cfg, iteration);
        }
        cooperative_barrier<WholeGroup>();
        if (state.status != coop_running) {
            if (leader)
                cooperative_store<P>(state, parameters, status, iterations, chi, fit);
            return;
        }
        float trial[P];
        for (uint k = 0; k < P; ++k) {
            const float step = simd_broadcast(lane == 0 ? state.delta[k] : 0.0f, 0);
            trial[k] = clamp(params[k] + step, cfg.lower[k], cfg.upper[k]);
        }
        local_cost = 0;
        cost_correction = 0;
        if (state.finite_trial)
            for (uint j = point_lane; j < cfg.points; j += stride) {
                const float residual = y[j * cfg.fits + fit] - Model::eval(x[j], trial);
                if (numerical_mode & 1u)
                    batchfit::numerics::add(residual * residual, local_cost, cost_correction);
                else
                    local_cost += residual * residual;
            }
        const float trial_cost = cooperative_sum<WholeGroup>(state.finite_trial ? local_cost : NAN,
                                                             sums, lane, simd, groups);
        if (leader)
            cooperative_accept<P>(state, cfg, trial_cost);
        cooperative_barrier<WholeGroup>();
        if (state.status != coop_running) {
            if (leader)
                cooperative_store<P>(state, parameters, status, iterations, chi, fit);
            return;
        }
    }
    if (leader) {
        state.status = max_iterations;
        cooperative_store<P>(state, parameters, status, iterations, chi, fit);
    }
}

#define COOPERATIVE_KERNEL(NAME, MODEL, WHOLE, STATES)                                             \
    kernel void NAME(                                                                              \
        device const float *x [[buffer(0)]], device const float *y [[buffer(1)]],                  \
        device const float *initial [[buffer(2)]], constant Config &cfg [[buffer(3)]],             \
        device float *parameters [[buffer(4)]], device uint *status [[buffer(5)]],                 \
        device uint *iterations [[buffer(6)]], device float *chi [[buffer(7)]],                    \
        uint3 group_id [[threadgroup_position_in_grid]],                                           \
        uint thread_index [[thread_index_in_threadgroup]],                                         \
        uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]],     \
        uint3 size [[threads_per_threadgroup]]) {                                                  \
        constexpr uint P = batchfit::models::MODEL<float>::parameter_count;                        \
        threadgroup CooperativeState<P> state[STATES];                                             \
        threadgroup float sums[8 * (P * (P + 1) / 2 + P)];                                         \
        cooperative_run<batchfit::models::MODEL<float>, WHOLE>(                                    \
            x, y, initial, cfg, parameters, status, iterations, chi, state, sums, group_id.x,      \
            thread_index, lane, simd, size.x);                                                     \
    }
#define COOPERATIVE_MODEL(PREFIX, MODEL)                                                           \
    COOPERATIVE_KERNEL(PREFIX##_simd, MODEL, false, 8)                                             \
    COOPERATIVE_KERNEL(PREFIX##_group, MODEL, true, 1)
COOPERATIVE_MODEL(fit_gaussian, Gaussian)
COOPERATIVE_MODEL(fit_lorentzian, Lorentzian)
COOPERATIVE_MODEL(fit_pseudo_voigt, PseudoVoigt)
COOPERATIVE_MODEL(fit_two_lorentzian, TwoLorentzian)
