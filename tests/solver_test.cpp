#include "solver_checks.hpp"

#include <gtest/gtest.h>

TEST(CpuSolver, FloatRecoveryAndStatuses) { EXPECT_NO_THROW(solver_checks::all<float>()); }
TEST(CpuSolver, DoubleRecoveryAndStatuses) { EXPECT_NO_THROW(solver_checks::all<double>()); }
