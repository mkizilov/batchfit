#include "pipeline_checks.hpp"
#include <gtest/gtest.h>
TEST(MetalPipeline, MappedChunksAndCompletionSafety) { EXPECT_NO_THROW(pipeline_checks::all()); }
