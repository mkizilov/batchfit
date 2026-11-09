#include <batchfit/cpu/solver.hpp>
#include <batchfit/cpu/uncertainty.hpp>
#include <batchfit/models.hpp>

#include <array>
#include <cstdlib>
#include <iostream>
#include <new>
#include <stdexcept>

namespace {
bool tracking = false;
std::size_t allocations = 0;
} // namespace

void *operator new(std::size_t bytes) {
    if (tracking)
        ++allocations;
    if (void *pointer = std::malloc(bytes == 0 ? 1 : bytes))
        return pointer;
    throw std::bad_alloc();
}
void *operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete(void *pointer) noexcept { std::free(pointer); }
void operator delete[](void *pointer) noexcept { std::free(pointer); }
void operator delete(void *pointer, std::size_t) noexcept { std::free(pointer); }
void operator delete[](void *pointer, std::size_t) noexcept { std::free(pointer); }

template <typename T, template <typename> class Model> void check() {
    constexpr unsigned p = Model<T>::parameter_count;
    std::array<T, p> truth, initial;
    if constexpr (p == 8)
        truth = {T(2), T(-1.5), T(0.5), T(1.5), T(1.5), T(0.7), T(0.1), T(0.02)};
    else if constexpr (p == 4)
        truth = {T(2), T(0.2), T(0.7), T(0.4)};
    else
        truth = {T(2), T(0.2), T(0.7)};
    for (unsigned k = 0; k < p; ++k)
        initial[k] = truth[k] * T(1.05) + T(0.01);
    std::array<T, 129> x, y;
    for (unsigned j = 0; j < x.size(); ++j) {
        x[j] = T(-5) + T(10) * T(j) / T(x.size() - 1);
        y[j] = Model<T>::eval(x[j], truth.data());
    }
    batchfit::cpu::SolverOptions<T, p> options;
    options.lower[2] = T(0.05);
    if constexpr (p == 8)
        options.lower[5] = T(0.05);
    if constexpr (p == 4) {
        options.lower[3] = T(0);
        options.upper[3] = T(1);
    }
    for (unsigned flags = 0; flags < 4; ++flags) {
        options.compensated_sum = (flags & 1U) != 0;
        options.scaled_cholesky = (flags & 2U) != 0;
        allocations = 0;
        tracking = true;
        const auto result = batchfit::cpu::fit_single<T, Model>(x, y, initial, options);
        const auto uncertainty = batchfit::cpu::estimate_uncertainty<T, Model>(x, result, options);
        tracking = false;
        if (allocations != 0 || result.status != batchfit::FitStatus::converged ||
            uncertainty.status != batchfit::UncertaintyStatus::available)
            throw std::runtime_error(
                "solver allocated or failed during allocation instrumentation");
    }
}

int main() {
    using namespace batchfit::models;
    check<float, Gaussian>();
    check<float, Lorentzian>();
    check<float, PseudoVoigt>();
    check<float, TwoLorentzian>();
    check<double, Gaussian>();
    check<double, Lorentzian>();
    check<double, PseudoVoigt>();
    check<double, TwoLorentzian>();
    std::cout
        << "All 32 model/scalar/numerical-option fits converged with zero C++ heap allocations\n";
}
