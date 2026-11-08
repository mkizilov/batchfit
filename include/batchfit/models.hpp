#pragma once

#if defined(__METAL_VERSION__)
#include <metal_math>
#define BATCHFIT_MODEL_THREAD thread
#define BATCHFIT_MODEL_EXP(value) metal::exp(value)
#define BATCHFIT_MODEL_LN2 0.69314718055994530942f
#else
#define BATCHFIT_MODEL_THREAD
#define BATCHFIT_MODEL_EXP(value) ::batchfit::models::detail::exp(value)
#define BATCHFIT_MODEL_LN2 0.693147180559945309417232121458176568
#endif

namespace batchfit {
namespace models {

enum class ParameterName {
    amplitude,
    center,
    sigma,
    hwhm,
    fwhm,
    mixing,
    amplitude1,
    center1,
    hwhm1,
    amplitude2,
    center2,
    hwhm2,
    intercept,
    slope,
    unknown
};

#if !defined(__METAL_VERSION__)
namespace detail {
// Compiler intrinsics keep this shared header free of C/C++ library includes.
inline float exp(float value) { return __builtin_expf(value); }
inline double exp(double value) { return __builtin_exp(value); }
} // namespace detail

inline const char *parameter_name_text(ParameterName name) {
    switch (name) {
    case ParameterName::amplitude:
        return "amplitude";
    case ParameterName::center:
        return "center";
    case ParameterName::sigma:
        return "sigma";
    case ParameterName::hwhm:
        return "hwhm";
    case ParameterName::fwhm:
        return "fwhm";
    case ParameterName::mixing:
        return "mixing";
    case ParameterName::amplitude1:
        return "amplitude1";
    case ParameterName::center1:
        return "center1";
    case ParameterName::hwhm1:
        return "hwhm1";
    case ParameterName::amplitude2:
        return "amplitude2";
    case ParameterName::center2:
        return "center2";
    case ParameterName::hwhm2:
        return "hwhm2";
    case ParameterName::intercept:
        return "intercept";
    case ParameterName::slope:
        return "slope";
    default:
        return "unknown";
    }
}
#endif

template <typename T> struct Gaussian {
    enum : unsigned { parameter_count = 3 };

    static constexpr ParameterName parameter_name(unsigned index) {
        return index == 0   ? ParameterName::amplitude
               : index == 1 ? ParameterName::center
               : index == 2 ? ParameterName::sigma
                            : ParameterName::unknown;
    }

    static T eval(T x, BATCHFIT_MODEL_THREAD const T *params) {
        const T z = (x - params[1]) / params[2];
        return params[0] * BATCHFIT_MODEL_EXP(-T(0.5f) * z * z);
    }

    static void jacobian(T x, BATCHFIT_MODEL_THREAD const T *params, BATCHFIT_MODEL_THREAD T *out) {
        const T z = (x - params[1]) / params[2];
        const T shape = BATCHFIT_MODEL_EXP(-T(0.5f) * z * z);
        out[0] = shape;
        out[1] = params[0] * shape * z / params[2];
        out[2] = params[0] * shape * z * z / params[2];
    }
};

template <typename T> struct Lorentzian {
    enum : unsigned { parameter_count = 3 };

    static constexpr ParameterName parameter_name(unsigned index) {
        return index == 0   ? ParameterName::amplitude
               : index == 1 ? ParameterName::center
               : index == 2 ? ParameterName::hwhm
                            : ParameterName::unknown;
    }

    static T eval(T x, BATCHFIT_MODEL_THREAD const T *params) {
        const T z = (x - params[1]) / params[2];
        return params[0] / (T(1) + z * z);
    }

    static void jacobian(T x, BATCHFIT_MODEL_THREAD const T *params, BATCHFIT_MODEL_THREAD T *out) {
        const T z = (x - params[1]) / params[2];
        const T shape = T(1) / (T(1) + z * z);
        const T scale = T(2) * params[0] * shape * shape / params[2];
        out[0] = shape;
        out[1] = scale * z;
        out[2] = scale * z * z;
    }
};

template <typename T> struct PseudoVoigt {
    enum : unsigned { parameter_count = 4 };

    static constexpr ParameterName parameter_name(unsigned index) {
        return index == 0   ? ParameterName::amplitude
               : index == 1 ? ParameterName::center
               : index == 2 ? ParameterName::fwhm
               : index == 3 ? ParameterName::mixing
                            : ParameterName::unknown;
    }

    static T eval(T x, BATCHFIT_MODEL_THREAD const T *params) {
        const T z = (x - params[1]) / params[2];
        const T gaussian = BATCHFIT_MODEL_EXP(-T(4) * T(BATCHFIT_MODEL_LN2) * z * z);
        const T lorentzian = T(1) / (T(1) + T(4) * z * z);
        return params[0] * ((T(1) - params[3]) * gaussian + params[3] * lorentzian);
    }

    static void jacobian(T x, BATCHFIT_MODEL_THREAD const T *params, BATCHFIT_MODEL_THREAD T *out) {
        const T z = (x - params[1]) / params[2];
        const T gaussian = BATCHFIT_MODEL_EXP(-T(4) * T(BATCHFIT_MODEL_LN2) * z * z);
        const T lorentzian = T(1) / (T(1) + T(4) * z * z);
        const T scale = params[0] * T(8) / params[2] *
                        ((T(1) - params[3]) * T(BATCHFIT_MODEL_LN2) * gaussian +
                         params[3] * lorentzian * lorentzian);
        out[0] = (T(1) - params[3]) * gaussian + params[3] * lorentzian;
        out[1] = scale * z;
        out[2] = scale * z * z;
        out[3] = params[0] * (lorentzian - gaussian);
    }
};

template <typename T> struct TwoLorentzian {
    enum : unsigned { parameter_count = 8 };

    static constexpr ParameterName parameter_name(unsigned index) {
        return index == 0   ? ParameterName::amplitude1
               : index == 1 ? ParameterName::center1
               : index == 2 ? ParameterName::hwhm1
               : index == 3 ? ParameterName::amplitude2
               : index == 4 ? ParameterName::center2
               : index == 5 ? ParameterName::hwhm2
               : index == 6 ? ParameterName::intercept
               : index == 7 ? ParameterName::slope
                            : ParameterName::unknown;
    }

    static T eval(T x, BATCHFIT_MODEL_THREAD const T *params) {
        return Lorentzian<T>::eval(x, params) + Lorentzian<T>::eval(x, params + 3) + params[6] +
               params[7] * x;
    }

    static void jacobian(T x, BATCHFIT_MODEL_THREAD const T *params, BATCHFIT_MODEL_THREAD T *out) {
        Lorentzian<T>::jacobian(x, params, out);
        Lorentzian<T>::jacobian(x, params + 3, out + 3);
        out[6] = T(1);
        out[7] = x;
    }
};

} // namespace models
} // namespace batchfit

#undef BATCHFIT_MODEL_THREAD
#undef BATCHFIT_MODEL_EXP
#undef BATCHFIT_MODEL_LN2
