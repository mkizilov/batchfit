#include "batch_checks.hpp"

#include <gtest/gtest.h>

TEST(CpuBatch, PoolLifecycleAndExceptions) { EXPECT_NO_THROW(batch_checks::pool_contract()); }
TEST(CpuBatch, FloatMatchesReference) { EXPECT_NO_THROW(batch_checks::models<float>()); }
TEST(CpuBatch, DoubleMatchesReference) { EXPECT_NO_THROW(batch_checks::models<double>()); }
