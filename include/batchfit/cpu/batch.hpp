#pragma once

#include <batchfit/cpu/solver.hpp>
#include <batchfit/cpu/thread_pool.hpp>

#include <stdexcept>
#include <vector>

namespace batchfit::cpu {
template <typename T> struct BatchResult {
    std::vector<T> parameters;
    std::vector<FitStatus> status;
    std::vector<unsigned> iterations;
    std::vector<T> chi_squared;
};

// Shared x[M], fit-major y[N*M] and initial[N*P]. Each row is the unchanged
// single-fit reference, including its per-fit invalid-input/failure behavior.
template <typename T, template <typename> class Model>
BatchResult<T> fit_batch(std::span<const T> x, std::span<const T> y, std::span<const T> initial,
                         ThreadPool &pool,
                         const SolverOptions<T, Model<T>::parameter_count> &options = {}) {
    constexpr unsigned p = Model<T>::parameter_count;
    static_assert(p > 0 && p <= 8);
    if (x.empty() || y.size() % x.size() != 0)
        throw std::invalid_argument("fit_batch requires nonempty x[M] and fit-major y[N*M]");
    const auto count = y.size() / x.size();
    // Division-based shape checks avoid overflowing N*P or N*M.
    if (initial.size() % p != 0 || initial.size() / p != count)
        throw std::invalid_argument("fit_batch initial parameters must have shape [N,P]");
    BatchResult<T> result{std::vector<T>(initial.size()), std::vector<FitStatus>(count),
                          std::vector<unsigned>(count), std::vector<T>(count)};
    pool.parallel_for(count, [&](std::size_t begin, std::size_t end) {
        for (auto i = begin; i < end; ++i) {
            const auto fit = fit_single<T, Model>(x, y.subspan(i * x.size(), x.size()),
                                                  initial.subspan(i * p, p), options);
            std::copy(fit.parameters.begin(), fit.parameters.end(),
                      result.parameters.begin() + i * p);
            result.status[i] = fit.status;
            result.iterations[i] = fit.iterations;
            result.chi_squared[i] = fit.chi_squared;
        }
    });
    return result;
}
} // namespace batchfit::cpu
