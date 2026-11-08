#include "residuals.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <stdexcept>

TEST(CpuReference, AnalyticResiduals) {
    const std::array predicted{1.0F, -2.0F, 3.5F};
    const std::array observed{0.0F, 2.0F, 3.0F};
    EXPECT_EQ(batchfit::detail::squared_residuals_cpu(predicted, observed),
              (std::vector<float>{1.0F, 16.0F, 0.25F}));
    EXPECT_TRUE(batchfit::detail::squared_residuals_cpu({}, {}).empty());
    EXPECT_THROW(batchfit::detail::squared_residuals_cpu(predicted, {}), std::invalid_argument);
}

#if BATCHFIT_HAS_METAL
TEST(MetalBackend, MatchesCpuAcrossSimdAndThreadgroupBoundaries) {
    for (const auto count : {0, 1, 31, 32, 33, 255, 256, 257, 1025}) {
        SCOPED_TRACE(count);
        std::vector<float> predicted(count), observed(count);
        for (int i = 0; i < count; ++i) {
            predicted[i] = std::sin(static_cast<float>(i));
            observed[i] = static_cast<float>(i % 17) / 7.0F;
        }
        const auto cpu = batchfit::detail::squared_residuals_cpu(predicted, observed);
        const auto gpu = batchfit::detail::squared_residuals_metal(predicted, observed);
        ASSERT_EQ(cpu.size(), gpu.size());
        for (std::size_t i = 0; i < cpu.size(); ++i) {
            EXPECT_NEAR(cpu[i], gpu[i], 1e-6F + 1e-5F * std::abs(cpu[i]));
        }
    }
}

TEST(MetalBackend, RejectsMismatchedLengths) {
    const std::array input{1.0F};
    EXPECT_THROW(batchfit::detail::squared_residuals_metal(input, {}), std::invalid_argument);
}
#endif
