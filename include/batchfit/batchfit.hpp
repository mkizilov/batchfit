#pragma once

#include <cstddef>
#include <string_view>

namespace batchfit {
inline constexpr std::size_t max_parameters = 8;
inline constexpr std::string_view version = "0.1.0";
enum class Backend { cpu, metal };
} // namespace batchfit
