#pragma once
#include <Metal/Metal.hpp>

namespace batchfit::metal::detail {
void register_completion(MTL::CommandBuffer *command, const MTL::HandlerFunction &function);
}
