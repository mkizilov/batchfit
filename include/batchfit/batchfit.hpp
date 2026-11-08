#pragma once

#include <batchfit/cpu/batch.hpp>
#include <batchfit/cpu/initial_guess.hpp>
#include <batchfit/cpu/solver.hpp>
#include <batchfit/metal/batch.hpp>
#include <batchfit/metal/pipeline.hpp>
#include <batchfit/models.hpp>
#include <batchfit/synthetic.hpp>

#include <cstddef>
#include <string_view>

namespace batchfit {
inline constexpr std::size_t max_parameters = 8;
inline constexpr std::string_view version = "0.1.0";
enum class Backend { cpu, metal };
} // namespace batchfit
