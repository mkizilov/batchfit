#include <batchfit/metal/batch.hpp>

#if BATCHFIT_HAS_METAL
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "lm_source.hpp"
#include "storage.hpp"

#include <mutex>
#include <string>

namespace batchfit::metal {
namespace detail {
void require(bool success, const std::string &operation, NS::Error *error = nullptr) {
    if (success)
        return;
    const auto *description = error ? error->localizedDescription() : nullptr;
    const char *message = description ? description->utf8String() : nullptr;
    throw std::runtime_error("Metal LM: " + operation + ": " +
                             (message ? message : "no error detail"));
}
struct Runtime {
    NS::SharedPtr<MTL::Device> device;
    NS::SharedPtr<MTL::CommandQueue> queue;
    NS::SharedPtr<MTL::Library> library;
    std::array<NS::SharedPtr<MTL::ComputePipelineState>, 4> pipelines;
    std::mutex mutex;
};
struct Resources {
    std::shared_ptr<Runtime> runtime;
    std::array<NS::SharedPtr<MTL::Buffer>, 8> buffers;
};
} // namespace detail

Context::Context() : runtime_(std::make_shared<detail::Runtime>()) {
    const auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    runtime_->device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
    detail::require(runtime_->device.get() != nullptr, "create default device");
    runtime_->queue = NS::TransferPtr(runtime_->device->newCommandQueue());
    detail::require(runtime_->queue.get() != nullptr, "create command queue");
    const auto options = NS::TransferPtr(MTL::CompileOptions::alloc()->init());
    detail::require(options.get() != nullptr, "allocate compile options");
    detail::require(NS::ProcessInfo::processInfo()->operatingSystemVersion().majorVersion >= 15,
                    "safe/precise Metal LM requires macOS 15 or newer");
    options->setMathMode(MTL::MathModeSafe);
    options->setMathFloatingPointFunctions(MTL::MathFloatingPointFunctionsPrecise);
    NS::Error *error = nullptr;
    runtime_->library = NS::TransferPtr(runtime_->device->newLibrary(
        NS::String::string(detail::shader_source, NS::UTF8StringEncoding), options.get(), &error));
    detail::require(runtime_->library.get() != nullptr,
                    "compile src/metal/lm.metal and embedded shared headers (#line preserved)",
                    error);
    for (unsigned k = 0; k < 4; ++k) {
        const auto function = NS::TransferPtr(runtime_->library->newFunction(
            NS::String::string(detail::kernel_names[k], NS::UTF8StringEncoding)));
        detail::require(function.get() != nullptr,
                        std::string("find kernel ") + detail::kernel_names[k]);
        error = nullptr;
        runtime_->pipelines[k] =
            NS::TransferPtr(runtime_->device->newComputePipelineState(function.get(), &error));
        detail::require(runtime_->pipelines[k].get() != nullptr,
                        std::string("create pipeline ") + detail::kernel_names[k], error);
    }
}

SharedBatch Context::allocate(std::size_t n, std::size_t m, unsigned p) const {
    const auto counts = detail::buffer_counts(n, m, p);
    const auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    auto resources = std::make_shared<detail::Resources>();
    resources->runtime = runtime_;
    std::array<void *, 8> pointers{};
    for (unsigned k = 0; k < 8; ++k) {
        if (counts[k] == 0)
            continue;
        const auto bytes = counts[k] * sizeof(float);
        detail::require(bytes <= runtime_->device->maxBufferLength(),
                        "buffer " + std::to_string(k) + " exceeds maxBufferLength");
        resources->buffers[k] =
            NS::TransferPtr(runtime_->device->newBuffer(bytes, MTL::ResourceStorageModeShared));
        detail::require(resources->buffers[k].get() != nullptr,
                        "allocate shared buffer " + std::to_string(k));
        pointers[k] = resources->buffers[k]->contents();
        detail::require(pointers[k] != nullptr, "map shared buffer " + std::to_string(k));
    }
    return SharedBatch(detail::make_storage(n, m, p, pointers, resources));
}

Timings detail::run_gpu(Storage &storage, unsigned kernel) {
    const auto start = std::chrono::steady_clock::now();
    const auto resources = std::static_pointer_cast<Resources>(storage.resources);
    auto &runtime = *resources->runtime;
    std::lock_guard lock(runtime.mutex);
    const auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    const auto &pipeline = runtime.pipelines.at(kernel);
    const auto width = pipeline->threadExecutionWidth();
    const auto maximum = pipeline->maxTotalThreadsPerThreadgroup();
    require(width > 0 && maximum >= width,
            "invalid thread limits for " + std::string(kernel_names[kernel]));
    const auto group = width * std::min(NS::UInteger(8), maximum / width);
    if (storage.fits == 0)
        return {0, std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(),
                width, group};
    const auto command = NS::RetainPtr(runtime.queue->commandBuffer());
    require(command.get() != nullptr, "create command buffer");
    const auto encoder = NS::RetainPtr(command->computeCommandEncoder());
    require(encoder.get() != nullptr, "create compute encoder");
    encoder->setComputePipelineState(pipeline.get());
    for (unsigned k = 0; k < 8; ++k)
        encoder->setBuffer(resources->buffers[k].get(), 0, k);
    encoder->dispatchThreads(MTL::Size(storage.fits, 1, 1), MTL::Size(group, 1, 1));
    encoder->endEncoding();
    command->commit();
    command->waitUntilCompleted();
    const auto context = std::string(kernel_names[kernel]) + " N=" + std::to_string(storage.fits) +
                         " M=" + std::to_string(storage.points);
    require(command->status() == MTL::CommandBufferStatusCompleted,
            context + " command status=" + std::to_string(command->status()), command->error());
    require(command->error() == nullptr, context + " command error", command->error());
    const double begin = command->GPUStartTime(), end = command->GPUEndTime();
    require(std::isfinite(begin) && std::isfinite(end) && end > begin,
            context + " invalid GPU timestamps");
    return {end - begin,
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(), width,
            group};
}
} // namespace batchfit::metal
#else
namespace batchfit::metal {
Context::Context() {
    throw std::runtime_error(
        "batchfit was built without BATCHFIT_METAL; Device::Metal is unavailable");
}
SharedBatch Context::allocate(std::size_t, std::size_t, unsigned) const {
    throw std::runtime_error("Device::Metal is unavailable");
}
Timings detail::run_gpu(Storage &, unsigned) {
    throw std::runtime_error("Device::Metal is unavailable");
}
} // namespace batchfit::metal
#endif
