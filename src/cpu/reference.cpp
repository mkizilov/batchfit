#include "residuals.hpp"

#include <stdexcept>

namespace batchfit::detail {
std::vector<float> squared_residuals_cpu(std::span<const float> predicted,
                                         std::span<const float> observed) {
    if (predicted.size() != observed.size()) {
        throw std::invalid_argument("squared_residuals_cpu: predicted/observed length mismatch");
    }
    std::vector<float> result(predicted.size());
    for (std::size_t i = 0; i < result.size(); ++i) {
        const float residual = predicted[i] - observed[i];
        result[i] = residual * residual;
    }
    return result;
}
} // namespace batchfit::detail
