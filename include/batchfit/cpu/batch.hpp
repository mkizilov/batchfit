#pragma once

#include <batchfit/cpu/solver.hpp>
#include <batchfit/cpu/thread_pool.hpp>
#include <batchfit/cpu/uncertainty.hpp>

#include <stdexcept>
#include <vector>

namespace batchfit::cpu {
template <typename T> struct BatchResult {
    std::vector<T> parameters;
    std::vector<FitStatus> status;
    std::vector<unsigned> iterations;
    std::vector<T> chi_squared;
    std::vector<T> covariance{};
    std::vector<T> standard_errors{};
    std::vector<UncertaintyStatus> uncertainty_status{};
};

// Shared x[M], fit-major y[N*M] and initial[N*P]. Each row is the unchanged
// single-fit reference, including its per-fit invalid-input/failure behavior.
template <typename T, template <typename> class Model>
BatchResult<T> fit_batch(std::span<const T> x, std::span<const T> y, std::span<const T> initial,
                         ThreadPool &pool,
                         const SolverOptions<T, Model<T>::parameter_count> &options = {},
                         bool uncertainties = false) {
    constexpr unsigned p = Model<T>::parameter_count;
    static_assert(p > 0 && p <= 8);
    if (x.empty() || y.size() % x.size() != 0)
        throw std::invalid_argument("fit_batch requires nonempty x[M] and fit-major y[N*M]");
    const auto count = y.size() / x.size();
    // Division-based shape checks avoid overflowing N*P or N*M.
    if (initial.size() % p != 0 || initial.size() / p != count)
        throw std::invalid_argument("fit_batch initial parameters must have shape [N,P]");
    if (uncertainties && count > std::numeric_limits<std::size_t>::max() / p / p)
        throw std::invalid_argument("covariance shape overflows size_t");
    BatchResult<T> result{std::vector<T>(initial.size()), std::vector<FitStatus>(count),
                          std::vector<unsigned>(count), std::vector<T>(count)};
    if (uncertainties) {
        result.covariance.resize(count * p * p);
        result.standard_errors.resize(count * p);
        result.uncertainty_status.resize(count);
    }
    pool.parallel_for(count, [&](std::size_t begin, std::size_t end) {
        for (auto i = begin; i < end; ++i) {
            const auto fit = fit_single<T, Model>(x, y.subspan(i * x.size(), x.size()),
                                                  initial.subspan(i * p, p), options);
            std::copy(fit.parameters.begin(), fit.parameters.end(),
                      result.parameters.begin() + i * p);
            result.status[i] = fit.status;
            result.iterations[i] = fit.iterations;
            result.chi_squared[i] = fit.chi_squared;
            if (uncertainties) {
                const auto estimate = estimate_uncertainty<T, Model>(x, fit, options);
                std::copy(estimate.covariance.begin(), estimate.covariance.end(),
                          result.covariance.data() + i * p * p);
                std::copy(estimate.standard_errors.begin(), estimate.standard_errors.end(),
                          result.standard_errors.data() + i * p);
                result.uncertainty_status[i] = estimate.status;
            }
        }
    });
    return result;
}
} // namespace batchfit::cpu
