#include "numerics_checks.hpp"
#include <gtest/gtest.h>
TEST(Numerics, CpuPreparationAndPathologies) { EXPECT_NO_THROW(numerics_checks::cpu()); }
#if BATCHFIT_HAS_METAL
TEST(Numerics, SafeAndFastMetalGuards) { EXPECT_NO_THROW(numerics_checks::gpu()); }
#endif
