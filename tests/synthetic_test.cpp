#include "synthetic_checks.hpp"

#include <gtest/gtest.h>

TEST(SyntheticData, StatisticsStreamingAndNpy) { EXPECT_NO_THROW(synthetic_checks::all()); }
