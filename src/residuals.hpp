#pragma once

#include <span>
#include <vector>

namespace batchfit::detail {
// Internal compute smoke primitive, not yet a nonlinear fitting API.
std::vector<float> squared_residuals_cpu(std::span<const float> predicted,
                                         std::span<const float> observed);
#if BATCHFIT_HAS_METAL
std::vector<float> squared_residuals_metal(std::span<const float> predicted,
                                           std::span<const float> observed);
#endif
} // namespace batchfit::detail
