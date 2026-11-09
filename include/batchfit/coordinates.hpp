#pragma once

#include <algorithm>
#include <array>
#include <batchfit/models.hpp>
#include <cmath>
#include <numeric>
#include <span>
#include <stdexcept>
#include <type_traits>

namespace batchfit {
// Prepare in double BEFORE conversion to float. Physical-unit reconstruction is
// host arithmetic, not shader float64. Bounds must be specified in fitted units.
struct Coordinates {
    double origin, x_scale, y_scale;
    static Coordinates from(std::span<const double> x, std::span<const double> y) {
        if (x.empty() || x.size() != y.size())
            throw std::invalid_argument("normalization requires matching nonempty x/y");
        double lo = x[0], hi = x[0], signal = 0;
        for (std::size_t j = 0; j < x.size(); ++j) {
            if (!std::isfinite(x[j]) || !std::isfinite(y[j]))
                throw std::invalid_argument("normalization requires finite inputs");
            lo = std::min(lo, x[j]);
            hi = std::max(hi, x[j]);
            signal = std::max(signal, std::abs(y[j]));
        }
        const double half = hi / 2 - lo / 2;
        if (!(half > 0) || !std::isfinite(half))
            throw std::invalid_argument("normalization requires a resolved x range");
        return {std::midpoint(lo, hi), half, signal > 0 ? signal : 1};
    }
    double x(double value) const { return (value - origin) / x_scale; }
    double y(double value) const { return value / y_scale; }
    double physical_chi_squared(double value) const { return (value * y_scale) * y_scale; }
    template <template <typename> class Model>
    std::array<double, std::size_t(Model<double>::parameter_count) * Model<double>::parameter_count>
    physical_covariance(std::span<const double> covariance) const {
        constexpr std::size_t p = Model<double>::parameter_count;
        static_assert(std::is_same_v<Model<double>, models::Gaussian<double>> ||
                      std::is_same_v<Model<double>, models::Lorentzian<double>> ||
                      std::is_same_v<Model<double>, models::PseudoVoigt<double>> ||
                      std::is_same_v<Model<double>, models::TwoLorentzian<double>>);
        if (covariance.size() != p * p)
            throw std::invalid_argument("covariance shape mismatch");
        std::array<double, std::size_t(p) * p> transform{}, result{};
        for (unsigned i = 0; i < p; ++i)
            transform[i * p + i] = i == 0 ? y_scale : i == 1 || i == 2 ? x_scale : 1;
        if constexpr (p == 8) {
            transform[3 * p + 3] = y_scale;
            transform[4 * p + 4] = transform[5 * p + 5] = x_scale;
            transform[6 * p + 6] = y_scale;
            transform[6 * p + 7] = -origin * y_scale / x_scale;
            transform[7 * p + 7] = y_scale / x_scale;
        }
        for (unsigned i = 0; i < p; ++i)
            for (unsigned j = 0; j < p; ++j)
                for (unsigned k = 0; k < p; ++k)
                    for (unsigned l = 0; l < p; ++l)
                        result[i * p + j] +=
                            transform[i * p + k] * covariance[k * p + l] * transform[j * p + l];
        return result;
    }

    template <template <typename> class Model>
    std::array<double, Model<double>::parameter_count>
    normalize(std::span<const double> parameters) const {
        constexpr unsigned p = Model<double>::parameter_count;
        static_assert(std::is_same_v<Model<double>, models::Gaussian<double>> ||
                      std::is_same_v<Model<double>, models::Lorentzian<double>> ||
                      std::is_same_v<Model<double>, models::PseudoVoigt<double>> ||
                      std::is_same_v<Model<double>, models::TwoLorentzian<double>>);
        if (parameters.size() != p)
            throw std::invalid_argument("normalization parameter count mismatch");
        std::array<double, p> q;
        std::copy(parameters.begin(), parameters.end(), q.begin());
        q[0] /= y_scale;
        q[1] = x(q[1]);
        q[2] /= x_scale;
        if constexpr (p == 8) {
            q[3] /= y_scale;
            q[4] = x(q[4]);
            q[5] /= x_scale;
            q[6] = std::fma(parameters[7], origin, parameters[6]) / y_scale;
            q[7] = parameters[7] * x_scale / y_scale;
        }
        return q;
    }
    template <template <typename> class Model>
    std::array<double, Model<double>::parameter_count>
    physical(std::span<const double> parameters) const {
        constexpr unsigned p = Model<double>::parameter_count;
        static_assert(std::is_same_v<Model<double>, models::Gaussian<double>> ||
                      std::is_same_v<Model<double>, models::Lorentzian<double>> ||
                      std::is_same_v<Model<double>, models::PseudoVoigt<double>> ||
                      std::is_same_v<Model<double>, models::TwoLorentzian<double>>);
        if (parameters.size() != p)
            throw std::invalid_argument("physical parameter count mismatch");
        std::array<double, p> result;
        std::copy(parameters.begin(), parameters.end(), result.begin());
        result[0] *= y_scale;
        result[1] = std::fma(result[1], x_scale, origin);
        result[2] *= x_scale;
        if constexpr (p == 8) {
            result[3] *= y_scale;
            result[4] = std::fma(result[4], x_scale, origin);
            result[5] *= x_scale;
            result[7] = parameters[7] * y_scale / x_scale;
            result[6] = std::fma(-result[7], origin, parameters[6] * y_scale);
        }
        return result;
    }
};
} // namespace batchfit
