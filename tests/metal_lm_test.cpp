#include "metal_lm_checks.hpp"

#include <gtest/gtest.h>

TEST(MetalLM, MatchesFloatCpuReference) { EXPECT_NO_THROW(metal_lm_checks::all()); }
