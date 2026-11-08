#include "cooperative_checks.hpp"
#include <gtest/gtest.h>
TEST(MetalCooperative, AllVariantsMatchCpu) { EXPECT_NO_THROW(cooperative_checks::all()); }
