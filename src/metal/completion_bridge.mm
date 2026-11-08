#include "completion_bridge.hpp"
#include <mutex>

namespace batchfit::metal::detail {
namespace {
std::mutex publication;
}
void register_completion(MTL::CommandBuffer *command, const MTL::HandlerFunction &function) {
    // Metal's driver is uninstrumented. Explicitly publish immutable callback
    // storage across its thread handoff; avoid metal-cpp's mutable __block shim.
    std::lock_guard registration(publication);
    const auto handler = function;
    command->addCompletedHandler(^(MTL::CommandBuffer *finished) {
      std::lock_guard invocation(publication);
      handler(finished);
    });
}
} // namespace batchfit::metal::detail
