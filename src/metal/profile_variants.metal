// Profiling-only instantiations. x/parameters/LM arithmetic remain float32;
// Sample changes observation storage only. Production defaults are unchanged.
#define PROFILE_KERNEL(NAME, MODEL, SAMPLE, POINT_MAJOR)                                           \
    kernel void NAME(device const float *x [[buffer(0)]], device const SAMPLE *y [[buffer(1)]],    \
                     device const float *initial [[buffer(2)]],                                    \
                     constant Config &cfg [[buffer(3)]], device float *parameters [[buffer(4)]],   \
                     device uint *status [[buffer(5)]], device uint *iterations [[buffer(6)]],     \
                     device float *chi [[buffer(7)]], uint fit [[thread_position_in_grid]]) {      \
        run<batchfit::models::MODEL<float>, SAMPLE, POINT_MAJOR>(x, y, initial, cfg, parameters,   \
                                                                 status, iterations, chi, fit);    \
    }
#define PROFILE_MODEL(PREFIX, MODEL)                                                               \
    PROFILE_KERNEL(PREFIX##_fit_float, MODEL, float, false)                                        \
    PROFILE_KERNEL(PREFIX##_point_half, MODEL, half, true)                                         \
    PROFILE_KERNEL(PREFIX##_fit_half, MODEL, half, false)
PROFILE_MODEL(fit_gaussian, Gaussian)
PROFILE_MODEL(fit_lorentzian, Lorentzian)
PROFILE_MODEL(fit_pseudo_voigt, PseudoVoigt)
PROFILE_MODEL(fit_two_lorentzian, TwoLorentzian)
