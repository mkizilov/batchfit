#pragma once

namespace batchfit {
namespace metal {
namespace detail {
// Identical four-byte scalar layout in C++ and MSL. No pointers or STL types.
struct Config {
    unsigned fits, points, max_iterations;
    float initial_lambda, gradient_tolerance, step_tolerance, cost_tolerance;
    float lower[8], upper[8];
};
} // namespace detail
} // namespace metal
} // namespace batchfit
