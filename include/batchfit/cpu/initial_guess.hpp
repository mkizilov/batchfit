#pragma once

#include <batchfit/models.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <span>
#include <type_traits>
#include <vector>

namespace batchfit::cpu {
enum class GuessStatus { success, invalid_input, no_peak, insufficient_peaks };

template <typename T, std::size_t P> struct InitialGuess {
    std::array<T, P> parameters;
    T baseline = std::numeric_limits<T>::quiet_NaN();
    GuessStatus status = GuessStatus::invalid_input;
};

namespace detail {
template <typename T> T median(std::span<T> values) {
    const auto middle = values.begin() + values.size() / 2;
    std::nth_element(values.begin(), middle, values.end());
    return values.size() % 2 ? *middle
                             : std::midpoint(*std::max_element(values.begin(), middle), *middle);
}

template <typename T> T smooth(std::span<const T> signal, std::size_t i) {
    if (i == 0 || i + 1 == signal.size())
        return signal[i];
    return T(0.25) * signal[i - 1] + T(0.5) * signal[i] + T(0.25) * signal[i + 1];
}

template <typename T> T prominence(std::span<const T> signal, std::size_t index) {
    const T height = smooth(signal, index);
    T left = height, right = height;
    for (std::size_t j = index; j > 0;) {
        const T value = smooth(signal, --j);
        if (value > height)
            break;
        left = std::min(left, value);
    }
    for (std::size_t j = index + 1; j < signal.size(); ++j) {
        const T value = smooth(signal, j);
        if (value > height)
            break;
        right = std::min(right, value);
    }
    return height - std::max(left, right);
}

template <typename T> struct Peak {
    T amplitude = 0, center = 0, width = 0, left_radius = 0, right_radius = 0;
    bool valid = false;
};

template <typename T>
Peak<T> peak(std::span<const T> x, std::span<const T> signal, std::size_t index, std::size_t begin,
             std::size_t end, bool mirror_missing) {
    Peak<T> result;
    result.amplitude = signal[index];
    if (!(result.amplitude > T(0)))
        return result;
    const T half = result.amplitude / T(2);
    auto left = index, right = index;
    while (left > begin && signal[left] > half)
        --left;
    while (right < end && signal[right] > half)
        ++right;
    const bool has_left = signal[left] <= half, has_right = signal[right] <= half;
    if ((!has_left && (!mirror_missing || begin == 0)) ||
        (!has_right && (!mirror_missing || end + 1 == x.size())) || (!has_left && !has_right))
        return result;
    auto low = index > 2 ? index - 2 : 0;
    auto high = std::min(index + 2, end);
    low = std::max(low, has_left ? left + 1 : begin);
    high = std::min(high, has_right ? right - 1 : end);
    T total = 0;
    result.center = x[index];
    for (auto j = low; j <= high; ++j) {
        const T weight = std::max(T(0), signal[j]) / result.amplitude;
        if (weight > T(0)) {
            total += weight;
            result.center = std::lerp(result.center, x[j], weight / total);
        }
    }
    const auto crossing = [&](std::size_t a, std::size_t b) {
        const T denominator = signal[b] - signal[a];
        if (!std::isfinite(denominator) || denominator == T(0))
            return std::numeric_limits<T>::quiet_NaN();
        return std::lerp(x[a], x[b], (half - signal[a]) / denominator);
    };
    if (has_left)
        result.left_radius = result.center - crossing(left, left + 1);
    if (has_right)
        result.right_radius = crossing(right - 1, right) - result.center;
    if (!has_left)
        result.left_radius = result.right_radius;
    if (!has_right)
        result.right_radius = result.left_radius;
    result.width = result.left_radius + result.right_radius;
    result.valid = std::isfinite(result.center) && std::isfinite(result.width) &&
                   result.left_radius > T(0) && result.right_radius > T(0);
    return result;
}

template <typename T> std::size_t raw_maximum(std::span<const T> signal, std::size_t index) {
    const auto low = index - 1, high = index + 1;
    return std::size_t(std::max_element(signal.begin() + low, signal.begin() + high + 1) -
                       signal.begin());
}
} // namespace detail

// Positive emission peaks on a finite, strictly increasing grid. Baseline is
// used for feature extraction; this does not alter the observations sent to LM.
template <typename T, template <typename> class Model>
InitialGuess<T, Model<T>::parameter_count> initial_guess(std::span<const T> x,
                                                         std::span<const T> y) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
    static_assert(std::is_same_v<Model<T>, models::Gaussian<T>> ||
                  std::is_same_v<Model<T>, models::Lorentzian<T>> ||
                  std::is_same_v<Model<T>, models::PseudoVoigt<T>> ||
                  std::is_same_v<Model<T>, models::TwoLorentzian<T>>);
    constexpr bool paired = std::is_same_v<Model<T>, models::TwoLorentzian<T>>;
    InitialGuess<T, Model<T>::parameter_count> result;
    result.parameters.fill(std::numeric_limits<T>::quiet_NaN());
    if (x.size() < 5 || x.size() != y.size())
        return result;
    T spacing = std::numeric_limits<T>::max();
    for (std::size_t j = 0; j < x.size(); ++j) {
        if (!std::isfinite(x[j]) || !std::isfinite(y[j]) || (j > 0 && x[j] <= x[j - 1]))
            return result;
        if (j > 0)
            spacing = std::min(spacing, x[j] - x[j - 1]);
    }
    if (!std::isfinite(x.back() - x.front()))
        return result;
    const auto count = std::max(std::size_t(1), x.size() / 10);
    std::vector<T> edges(2 * count), signal(x.size());
    for (std::size_t j = 0; j < count; ++j) {
        edges[j] = y[j];
        edges[count + j] = y[y.size() - count + j];
    }
    const T left_level = detail::median<T>(std::span(edges).first(count));
    const T right_level = detail::median<T>(std::span(edges).subspan(count));
    result.baseline = detail::median<T>(edges);
    const T left_x = std::midpoint(x[(count - 1) / 2], x[count / 2]);
    const T right_x =
        std::midpoint(x[x.size() - count + (count - 1) / 2], x[x.size() - count + count / 2]);
    const T slope = paired ? (right_level - left_level) / (right_x - left_x) : T(0);
    const T level = paired ? left_level : result.baseline;
    if (!std::isfinite(slope))
        return result;
    for (std::size_t j = 0; j < x.size(); ++j) {
        signal[j] = y[j] - (level + slope * (x[j] - left_x));
        if (!std::isfinite(signal[j]))
            return result;
    }
    for (std::size_t j = 0; j < count; ++j) {
        edges[j] = std::abs(y[j] - left_level);
        edges[count + j] = std::abs(y[y.size() - count + j] - right_level);
    }
    const T noise = T(1.4826) * detail::median<T>(edges);
    if (!std::isfinite(noise))
        return result;
    std::size_t first = x.size();
    T height = 0;
    for (std::size_t j = 1; j + 1 < x.size(); ++j) {
        const T value = detail::smooth<T>(signal, j);
        if (value > height && value > detail::smooth<T>(signal, j - 1) &&
            value >= detail::smooth<T>(signal, j + 1)) {
            first = j;
            height = value;
        }
    }
    result.status = GuessStatus::no_peak;
    if (first == x.size() || height <= T(3) * noise)
        return result;
    first = detail::raw_maximum<T>(signal, first);
    auto primary = detail::peak<T>(x, signal, first, 0, x.size() - 1, false);
    if (!primary.valid)
        return result;
    if constexpr (!paired) {
        result.parameters[0] = primary.amplitude;
        result.parameters[1] = primary.center;
        if constexpr (std::is_same_v<Model<T>, models::Gaussian<T>>)
            result.parameters[2] = primary.width / T(2.354820045030949382);
        else if constexpr (std::is_same_v<Model<T>, models::Lorentzian<T>>)
            result.parameters[2] = primary.width / T(2);
        else {
            result.parameters[2] = primary.width;
            result.parameters[3] = T(0.5);
        }
    } else {
        result.status = GuessStatus::insufficient_peaks;
        // A half-width admits noise maxima on the same peak's shoulder at dense M.
        const T separation =
            std::max(T(3) * spacing, T(2) * std::min(primary.left_radius, primary.right_radius));
        std::size_t second = x.size();
        T second_height = 0;
        for (std::size_t j = 1; j + 1 < x.size(); ++j) {
            const T value = detail::smooth<T>(signal, j);
            if (value <= second_height || value <= detail::smooth<T>(signal, j - 1) ||
                value < detail::smooth<T>(signal, j + 1))
                continue;
            const auto index = detail::raw_maximum<T>(signal, j);
            const auto bins = index > first ? index - first : first - index;
            if (bins < 3 || std::abs(x[index] - primary.center) < separation ||
                detail::prominence<T>(signal, j) <= T(3) * noise)
                continue;
            second = index;
            second_height = value;
        }
        if (second == x.size())
            return result;
        if (first > second)
            std::swap(first, second);
        const auto valley = std::size_t(
            std::min_element(signal.begin() + first + 1, signal.begin() + second) - signal.begin());
        primary = detail::peak<T>(x, signal, first, 0, valley, true);
        const auto secondary = detail::peak<T>(x, signal, second, valley, x.size() - 1, true);
        if (!primary.valid || !secondary.valid || !(primary.center < secondary.center))
            return result;
        result.parameters = {primary.amplitude,
                             primary.center,
                             primary.width / T(2),
                             secondary.amplitude,
                             secondary.center,
                             secondary.width / T(2),
                             left_level - slope * left_x,
                             slope};
    }
    for (const T value : result.parameters) {
        if (!std::isfinite(value)) {
            result.status = GuessStatus::invalid_input;
            return result;
        }
    }
    if (!(result.parameters[2] > T(0)) || (paired && !(result.parameters[5] > T(0)))) {
        result.status = GuessStatus::invalid_input;
        return result;
    }
    result.status = GuessStatus::success;
    return result;
}
} // namespace batchfit::cpu
