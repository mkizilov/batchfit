#include "initial_guess_checks.hpp"

#include <gtest/gtest.h>

TEST(InitialGuess, FloatFeaturesAndConvergence) {
    EXPECT_NO_THROW(initial_guess_checks::all<float>());
}
TEST(InitialGuess, DoubleFeaturesAndConvergence) {
    EXPECT_NO_THROW(initial_guess_checks::all<double>());
}
