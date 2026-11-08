#include "model_checks.hpp"

#include <gtest/gtest.h>

TEST(Models, FloatEvaluationAndJacobians) { EXPECT_NO_THROW(model_checks::all<float>()); }
TEST(Models, DoubleEvaluationAndJacobians) { EXPECT_NO_THROW(model_checks::all<double>()); }
