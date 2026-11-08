#include <batchfit/batchfit.hpp>

#include "residuals.hpp"

#include <array>
#include <iostream>
#include <stdexcept>

int main() {
    static_assert(batchfit::max_parameters == 8);
    const std::array predicted{1.0F, -2.0F, 3.5F};
    const std::array observed{0.0F, 2.0F, 3.0F};
    const auto result = batchfit::detail::squared_residuals_cpu(predicted, observed);
    if (result != std::vector<float>{1.0F, 16.0F, 0.25F}) {
        throw std::runtime_error("CPU squared residuals disagree with analytic reference");
    }
    if (!batchfit::detail::squared_residuals_cpu({}, {}).empty()) {
        throw std::runtime_error("CPU empty input must return empty output");
    }
    try {
        batchfit::detail::squared_residuals_cpu(predicted, {});
    } catch (const std::invalid_argument &) {
        std::cout << "batchfit " << batchfit::version << ": CPU smoke checks passed\n";
        return 0;
    }
    throw std::runtime_error("CPU mismatched lengths must throw invalid_argument");
}
