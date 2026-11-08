#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "residuals.hpp"
#include "shader_source.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace batchfit::detail {
namespace {
void require(bool success, const char *context, NS::Error *error = nullptr) {
    if (!success) {
        const auto message =
            error ? error->localizedDescription()->utf8String() : "no error detail";
        throw std::runtime_error(std::string("Metal squared_residuals: ") + context + ": " +
                                 message);
    }
}
} // namespace

std::vector<float> squared_residuals_metal(std::span<const float> predicted,
                                           std::span<const float> observed) {
    if (predicted.size() != observed.size()) {
        throw std::invalid_argument("squared_residuals_metal: predicted/observed length mismatch");
    }
    if (predicted.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("squared_residuals_metal: grid exceeds uint32 index range");
    }
    if (predicted.empty()) {
        return {};
    }
    const auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    const auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
    require(device.get() != nullptr, "create default device");
    const auto options = NS::TransferPtr(MTL::CompileOptions::alloc()->init());
    options->setFastMathEnabled(false);
    NS::Error *error = nullptr;
    const auto library = NS::TransferPtr(device->newLibrary(
        NS::String::string(residuals_shader, NS::UTF8StringEncoding), options.get(), &error));
    require(library.get() != nullptr, "compile residuals.metal (fast math disabled)", error);
    const auto function = NS::TransferPtr(
        library->newFunction(NS::String::string("squared_residuals", NS::UTF8StringEncoding)));
    require(function.get() != nullptr, "find kernel squared_residuals in residuals.metal");
    error = nullptr;
    const auto pipeline = NS::TransferPtr(device->newComputePipelineState(function.get(), &error));
    require(pipeline.get() != nullptr, "create squared_residuals pipeline", error);
    const auto queue = NS::TransferPtr(device->newCommandQueue());
    require(queue.get() != nullptr, "create command queue");
    const auto bytes = predicted.size_bytes();
    const auto input =
        NS::TransferPtr(device->newBuffer(predicted.data(), bytes, MTL::ResourceStorageModeShared));
    const auto target =
        NS::TransferPtr(device->newBuffer(observed.data(), bytes, MTL::ResourceStorageModeShared));
    const auto output = NS::TransferPtr(device->newBuffer(bytes, MTL::ResourceStorageModeShared));
    require(input.get() && target.get() && output.get(), "allocate shared residual buffers");
    auto *command = queue->commandBuffer();
    require(command != nullptr, "create command buffer");
    auto *encoder = command->computeCommandEncoder();
    require(encoder != nullptr, "create compute encoder");
    encoder->setComputePipelineState(pipeline.get());
    encoder->setBuffer(input.get(), 0, 0);
    encoder->setBuffer(target.get(), 0, 1);
    encoder->setBuffer(output.get(), 0, 2);
    const auto width = std::min(pipeline->maxTotalThreadsPerThreadgroup(), NS::UInteger(256));
    encoder->dispatchThreads(MTL::Size(predicted.size(), 1, 1), MTL::Size(width, 1, 1));
    encoder->endEncoding();
    command->commit();
    command->waitUntilCompleted();
    require(command->status() == MTL::CommandBufferStatusCompleted,
            "execute squared_residuals command buffer", command->error());
    std::vector<float> result(predicted.size());
    std::memcpy(result.data(), output->contents(), bytes);
    return result;
}
} // namespace batchfit::detail
