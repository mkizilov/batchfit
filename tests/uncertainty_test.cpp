#include "uncertainty_checks.hpp"
#include <gtest/gtest.h>
TEST(Uncertainty, CpuAlgebraAndOptionalStorage) { EXPECT_NO_THROW(uncertainty_checks::cpu()); }
#if BATCHFIT_HAS_METAL
TEST(Uncertainty, AllMetalVariantsAndCompletion) { EXPECT_NO_THROW(uncertainty_checks::gpu()); }
#endif
