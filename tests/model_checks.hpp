#pragma once

#include <batchfit/models.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace model_checks {
using namespace batchfit::models;

inline void require(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

inline void near(double actual, double expected, double absolute, double relative,
                 const std::string &context) {
    require(std::isfinite(actual) && std::isfinite(expected) &&
                std::abs(actual - expected) <= absolute + relative * std::abs(expected),
            context + ": actual=" + std::to_string(actual) +
                ", expected=" + std::to_string(expected));
}

template <typename T> struct Sample {
    T x;
    T params[8];
};

template <typename T> Sample<T> random_sample(std::mt19937 &rng, bool paired) {
    const auto draw = [&rng](double low, double high) {
        return T(std::uniform_real_distribution<double>(low, high)(rng));
    };
    Sample<T> sample{};
    sample.x = draw(-6, 6);
    sample.params[0] = draw(-5, 5);
    sample.params[1] = draw(-4, 4);
    sample.params[2] = draw(0.3, 3);
    sample.params[3] = paired ? draw(-5, 5) : draw(0, 1);
    sample.params[4] = draw(-4, 4);
    sample.params[5] = draw(0.3, 3);
    sample.params[6] = draw(-2, 2);
    sample.params[7] = draw(-1, 1);
    return sample;
}

template <typename T> void hand_values() {
    constexpr double tolerance = std::is_same_v<T, float> ? 2e-6 : 1e-14;
    const T single[3] = {T(4), T(1), T(2)};
    near(Gaussian<T>::eval(T(1), single), 4, tolerance, tolerance, "Gaussian center");
    near(Gaussian<T>::eval(T(3), single), 4 * std::exp(-0.5), tolerance, tolerance,
         "Gaussian one sigma");
    near(Lorentzian<T>::eval(T(1), single), 4, tolerance, tolerance, "Lorentzian center");
    near(Lorentzian<T>::eval(T(3), single), 2, tolerance, tolerance, "Lorentzian HWHM");
    T voigt[4] = {T(4), T(1), T(2), T(0.25)};
    near(PseudoVoigt<T>::eval(T(1), voigt), 4, tolerance, tolerance, "pseudo-Voigt center");
    near(PseudoVoigt<T>::eval(T(2), voigt), 2, tolerance, tolerance, "pseudo-Voigt HWHM");
    near(PseudoVoigt<T>::eval(T(3), voigt), 0.3875, tolerance, tolerance,
         "pseudo-Voigt off-center mixture");
    const T pair[8] = {T(4), T(1), T(2), T(3), T(-1), T(1), T(0.5), T(-0.25)};
    near(TwoLorentzian<T>::eval(T(1), pair), 4.85, tolerance, tolerance,
         "paired peaks plus nonzero baseline");
    near(TwoLorentzian<T>::eval(T(-1), pair), 5.75, tolerance, tolerance,
         "paired peaks at second center");

    // Mixing endpoints must be the common-FWHM Gaussian and Lorentzian, not
    // a mixture of components with inconsistent width conventions.
    for (const T x : {T(-3), T(1), T(2), T(5)}) {
        const T equivalent_gaussian[3] = {T(4), T(1), T(2) / T(std::sqrt(8 * std::log(2)))};
        const T equivalent_lorentzian[3] = {T(4), T(1), T(1)};
        voigt[3] = T(0);
        near(PseudoVoigt<T>::eval(x, voigt), Gaussian<T>::eval(x, equivalent_gaussian), tolerance,
             tolerance, "pseudo-Voigt pure Gaussian");
        voigt[3] = T(1);
        near(PseudoVoigt<T>::eval(x, voigt), Lorentzian<T>::eval(x, equivalent_lorentzian),
             tolerance, tolerance, "pseudo-Voigt pure Lorentzian");
    }
}

template <typename Model> void metadata(std::initializer_list<const char *> expected) {
    static_assert(std::is_empty_v<Model> && std::is_standard_layout_v<Model>);
    static_assert(Model::parameter_count <= 8);
    require(expected.size() == Model::parameter_count, "parameter count");
    unsigned index = 0;
    for (const auto name : expected) {
        require(std::string(parameter_name_text(Model::parameter_name(index++))) == name,
                "parameter name/order");
    }
    require(Model::parameter_name(Model::parameter_count) == ParameterName::unknown &&
                Model::parameter_name(999) == ParameterName::unknown,
            "out-of-range parameter name");
}

template <typename T, typename Model> void finite_differences(bool paired) {
    std::mt19937 rng(0xBA7CF17U);
    constexpr bool single = std::is_same_v<T, float>;
    constexpr T step = single ? T(0.002) : T(1e-5);
    constexpr double absolute = single ? 3e-4 : 2e-8;
    constexpr double relative = single ? 2e-3 : 2e-7;
    for (unsigned trial = 0; trial < 2000; ++trial) {
        auto sample = random_sample<T>(rng, paired);
        // Explicitly cover zero heights, centers, and mixing endpoints as well
        // as the random interior, without dividing a derivative by amplitude.
        if (trial % 10 == 0) {
            sample.params[0] = T(0);
            if (paired) {
                sample.params[3] = T(0);
            }
        }
        if (trial % 11 == 0) {
            sample.x = sample.params[1];
        }
        if (!paired && trial % 13 == 0) {
            sample.params[3] = T(trial % 2);
        }
        T out[Model::parameter_count + 2];
        std::fill_n(out, Model::parameter_count + 2, std::numeric_limits<T>::quiet_NaN());
        out[0] = T(123);
        out[Model::parameter_count + 1] = T(456);
        Model::jacobian(sample.x, sample.params, out + 1);
        require(out[0] == T(123) && out[Model::parameter_count + 1] == T(456),
                "Jacobian wrote outside its fixed-size output");
        for (unsigned k = 0; k < Model::parameter_count; ++k) {
            auto plus = sample;
            auto minus = sample;
            const bool peak_position = k == 1 || k == 2 || (paired && (k == 4 || k == 5));
            const T scale = peak_position ? sample.params[k < 3 ? 2 : 5]
                                          : std::max(T(1), std::abs(sample.params[k]));
            plus.params[k] += step * scale;
            minus.params[k] -= step * scale;
            const T numeric =
                (Model::eval(plus.x, plus.params) - Model::eval(minus.x, minus.params)) /
                (plus.params[k] - minus.params[k]);
            near(out[k + 1], numeric, absolute, relative,
                 std::string(single ? "float" : "double") + " trial=" + std::to_string(trial) +
                     " param=" + parameter_name_text(Model::parameter_name(k)));
        }
    }
}

template <typename T> void all() {
    metadata<Gaussian<T>>({"amplitude", "center", "sigma"});
    metadata<Lorentzian<T>>({"amplitude", "center", "hwhm"});
    metadata<PseudoVoigt<T>>({"amplitude", "center", "fwhm", "mixing"});
    metadata<TwoLorentzian<T>>(
        {"amplitude1", "center1", "hwhm1", "amplitude2", "center2", "hwhm2", "intercept", "slope"});
    hand_values<T>();
    finite_differences<T, Gaussian<T>>(false);
    finite_differences<T, Lorentzian<T>>(false);
    finite_differences<T, PseudoVoigt<T>>(false);
    finite_differences<T, TwoLorentzian<T>>(true);
}
} // namespace model_checks
