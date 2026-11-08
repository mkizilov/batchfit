#include <batchfit/batchfit.hpp>

#include <iostream>
#include <stdexcept>
#include <string>

namespace {
template <typename T> struct ThrowingModel {
    enum : unsigned { parameter_count = 1 };
    static T eval(T, const T *) { throw std::runtime_error("model failure"); }
    static void jacobian(T, const T *, T *out) { out[0] = T(1); }
};

template <typename T> void check() {
    batchfit::cpu::ThreadPool pool(2);
    const std::array<T, 3> x{T(0), T(0), T(0)};
    const std::array<T, 9> y{T(2), T(2), T(2), T(2), T(2), T(2), T(2), T(2), T(2)};
    const std::array<T, 9> initial{T(1), T(0), T(1), T(2), T(0), T(1), T(1), T(0), T(0)};
    const auto mixed =
        batchfit::cpu::fit_batch<T, batchfit::models::Lorentzian>(x, y, initial, pool);
    const std::array expected{batchfit::FitStatus::singular_matrix, batchfit::FitStatus::converged,
                              batchfit::FitStatus::invalid_input};
    if (!std::equal(expected.begin(), expected.end(), mixed.status.begin()))
        throw std::runtime_error("batch mixed failure statuses changed");
    auto invalid_x = x;
    invalid_x[0] = std::numeric_limits<T>::quiet_NaN();
    const auto invalid =
        batchfit::cpu::fit_batch<T, batchfit::models::Lorentzian>(invalid_x, y, initial, pool);
    for (auto status : invalid.status)
        if (status != batchfit::FitStatus::invalid_input)
            throw std::runtime_error("shared invalid x not isolated");
    const std::array<T, 1> one{T(1)};
    bool original_error = false;
    try {
        batchfit::cpu::fit_batch<T, ThrowingModel>(one, one, one, pool);
    } catch (const std::runtime_error &error) {
        original_error = std::string(error.what()) == "model failure";
    }
    if (!original_error)
        throw std::runtime_error("model exception was not propagated");
    const auto again =
        batchfit::cpu::fit_batch<T, batchfit::models::Lorentzian>(x, y, initial, pool);
    if (again.status != mixed.status)
        throw std::runtime_error("pool did not recover after model exception");
}
} // namespace

int main() {
    try {
        check<float>();
        check<double>();
        std::cout << "Batch mixed failures, shared invalid x, model exceptions and reuse passed\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
