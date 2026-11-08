#include <batchfit/metal/batch.hpp>

#if BATCHFIT_HAS_METAL
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "completion.hpp"
#include "completion_bridge.hpp"
#include "lm_source.hpp"
#include "storage.hpp"

#include <iostream>
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
    std::array<NS::SharedPtr<MTL::ComputePipelineState>, 12> pipelines;
    std::mutex mutex;
};
struct Resources {
    std::shared_ptr<Runtime> runtime;
    std::array<NS::SharedPtr<MTL::Buffer>, 8> buffers;
};
struct AsyncResources {
    NS::SharedPtr<MTL::CommandBuffer> command;
    std::shared_ptr<Resources> buffers;
};
} // namespace detail

} // namespace batchfit::metal
#include "capture_cpp.hpp"
namespace batchfit::metal {

Context::Context() : runtime_(std::make_shared<detail::Runtime>()) {
    detail::prepare_capture_environment();
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
    for (unsigned k = 0; k < 12; ++k) {
        const auto function = NS::TransferPtr(runtime_->library->newFunction(
            NS::String::string(detail::variant_kernel_names[k], NS::UTF8StringEncoding)));
        detail::require(function.get() != nullptr,
                        std::string("find kernel ") + detail::variant_kernel_names[k]);
        error = nullptr;
        runtime_->pipelines[k] =
            NS::TransferPtr(runtime_->device->newComputePipelineState(function.get(), &error));
        detail::require(runtime_->pipelines[k].get() != nullptr,
                        std::string("create pipeline ") + detail::variant_kernel_names[k], error);
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

DeviceLimits Context::limits() const {
    return {runtime_->device->maxBufferLength(), runtime_->device->recommendedMaxWorkingSetSize()};
}

PendingBatch detail::submit_gpu(const std::shared_ptr<Storage> &storage, unsigned kernel,
                                KernelVariant variant) {
    const auto start = std::chrono::steady_clock::now();
    const auto resources = std::static_pointer_cast<Resources>(storage->resources);
    auto &runtime = *resources->runtime;
    std::lock_guard lock(runtime.mutex);
    const auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    if (variant == KernelVariant::Automatic && kernel < 4 &&
        runtime.pipelines[kernel + 4]->threadExecutionWidth() != 32)
        variant = KernelVariant::ThreadPerFit;
    const auto selected = pipeline_index(kernel, storage->points, variant);
    const auto &pipeline = runtime.pipelines.at(selected);
    const auto width = pipeline->threadExecutionWidth(),
               maximum = pipeline->maxTotalThreadsPerThreadgroup();
    require(width > 0 && maximum >= width, "async pipeline thread limits");
    require(variant == KernelVariant::ThreadPerFit || width == 32,
            "async cooperative kernel SIMD width");
    const auto group = width * std::min(NS::UInteger(8), maximum / width);
    auto completion = std::make_shared<Completion>();
    completion->timings = {0, 0, width, group, variant};
    if (storage->fits == 0) {
        completion->done = true;
        return PendingBatch(storage, completion);
    }
    Capture capture(runtime.queue.get());
    const auto command = NS::RetainPtr(runtime.queue->commandBuffer());
    require(command.get() != nullptr, "create async command buffer");
    const auto encoder = NS::RetainPtr(command->computeCommandEncoder());
    require(encoder.get() != nullptr, "create async compute encoder");
    const auto context = std::make_shared<const std::string>(
        std::string(variant_kernel_names[selected]) + " N=" + std::to_string(storage->fits) +
        " M=" + std::to_string(storage->points));
    command->setLabel(NS::String::string(context->c_str(), NS::UTF8StringEncoding));
    encoder->setLabel(NS::String::string(variant_kernel_names[selected], NS::UTF8StringEncoding));
    encoder->setComputePipelineState(pipeline.get());
    for (unsigned k = 0; k < 8; ++k)
        encoder->setBuffer(resources->buffers[k].get(), 0, k);
    if (variant == KernelVariant::ThreadPerFit)
        encoder->dispatchThreads(MTL::Size(storage->fits, 1, 1), MTL::Size(group, 1, 1));
    else {
        const auto per_group = variant == KernelVariant::SimdPerFit ? group / 32 : 1;
        encoder->dispatchThreadgroups(MTL::Size((storage->fits - 1) / per_group + 1, 1, 1),
                                      MTL::Size(group, 1, 1));
    }
    encoder->endEncoding();
    completion->backend = std::make_shared<AsyncResources>(AsyncResources{command, resources});
    std::weak_ptr<Completion> weak = completion;
    MTL::HandlerFunction handler = [weak, start, context](MTL::CommandBuffer *finished) {
        auto state = weak.lock();
        if (!state) {
            std::cerr << "orphan Metal completion: " << *context << '\n';
            return;
        }
        std::exception_ptr failure;
        Timings timings = state->timings;
        try {
            const auto callback_pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
            require(finished->status() == MTL::CommandBufferStatusCompleted,
                    *context + " async status=" + std::to_string(finished->status()),
                    finished->error());
            require(finished->error() == nullptr, *context + " async command error",
                    finished->error());
            timings.gpu_start_seconds = finished->GPUStartTime();
            timings.gpu_end_seconds = finished->GPUEndTime();
            require(std::isfinite(timings.gpu_start_seconds) &&
                        std::isfinite(timings.gpu_end_seconds) &&
                        timings.gpu_end_seconds > timings.gpu_start_seconds,
                    *context + " async GPU timestamps");
            timings.gpu_seconds = timings.gpu_end_seconds - timings.gpu_start_seconds;
            timings.wall_seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        } catch (...) {
            failure = std::current_exception();
        }
        {
            std::lock_guard done_lock(state->mutex);
            state->failure = failure;
            state->timings = timings;
            state->done = true;
        }
        state->changed.notify_all();
    };
    register_completion(command.get(), handler);
    command->commit();
    // Capture stays first-dispatch-only; normal async submission never waits here.
    if (capture.enabled()) {
        command->waitUntilCompleted();
        std::unique_lock done_lock(completion->mutex);
        completion->changed.wait(done_lock, [&] { return completion->done; });
        done_lock.unlock();
        capture.finish();
    }
    return PendingBatch(storage, completion);
}

Timings detail::run_gpu(Storage &storage, unsigned kernel, KernelVariant variant) {
    const auto start = std::chrono::steady_clock::now();
    const auto resources = std::static_pointer_cast<Resources>(storage.resources);
    auto &runtime = *resources->runtime;
    std::lock_guard lock(runtime.mutex);
    const auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    if (variant == KernelVariant::Automatic && kernel < 4 &&
        runtime.pipelines[kernel + 4]->threadExecutionWidth() != 32)
        variant = KernelVariant::ThreadPerFit;
    const auto selected = pipeline_index(kernel, storage.points, variant);
    const auto &pipeline = runtime.pipelines.at(selected);
    const auto width = pipeline->threadExecutionWidth();
    const auto maximum = pipeline->maxTotalThreadsPerThreadgroup();
    require(width > 0 && maximum >= width,
            "invalid thread limits for " + std::string(variant_kernel_names[selected]));
    require(variant == KernelVariant::ThreadPerFit || width == 32,
            "cooperative kernels require SIMD width 32");
    const auto group = width * std::min(NS::UInteger(8), maximum / width);
    if (storage.fits == 0)
        return {0, std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(),
                width, group, variant};
    Capture capture(runtime.queue.get());
    const auto command = NS::RetainPtr(runtime.queue->commandBuffer());
    require(command.get() != nullptr, "create command buffer");
    command->setLabel(NS::String::string(variant_kernel_names[selected], NS::UTF8StringEncoding));
    const auto encoder = NS::RetainPtr(command->computeCommandEncoder());
    require(encoder.get() != nullptr, "create compute encoder");
    encoder->setLabel(NS::String::string(variant_kernel_names[selected], NS::UTF8StringEncoding));
    encoder->setComputePipelineState(pipeline.get());
    for (unsigned k = 0; k < 8; ++k)
        encoder->setBuffer(resources->buffers[k].get(), 0, k);
    if (variant == KernelVariant::ThreadPerFit)
        encoder->dispatchThreads(MTL::Size(storage.fits, 1, 1), MTL::Size(group, 1, 1));
    else {
        const auto per_group = variant == KernelVariant::SimdPerFit ? group / 32 : 1;
        encoder->dispatchThreadgroups(MTL::Size((storage.fits - 1) / per_group + 1, 1, 1),
                                      MTL::Size(group, 1, 1));
    }
    encoder->endEncoding();
    command->commit();
    command->waitUntilCompleted();
    const auto context = std::string(variant_kernel_names[selected]) +
                         " N=" + std::to_string(storage.fits) +
                         " M=" + std::to_string(storage.points);
    require(command->status() == MTL::CommandBufferStatusCompleted,
            context + " command status=" + std::to_string(command->status()), command->error());
    require(command->error() == nullptr, context + " command error", command->error());
    const double begin = command->GPUStartTime(), end = command->GPUEndTime();
    require(std::isfinite(begin) && std::isfinite(end) && end > begin,
            context + " invalid GPU timestamps");
    capture.finish();
    return {end - begin,
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(), width,
            group, variant};
}
} // namespace batchfit::metal
#else
namespace batchfit::metal {
Context::Context() {
    throw std::runtime_error(
        "batchfit was built without BATCHFIT_METAL; Device::Metal is unavailable");
}
DeviceLimits Context::limits() const { throw std::runtime_error("Device::Metal is unavailable"); }
SharedBatch Context::allocate(std::size_t, std::size_t, unsigned) const {
    throw std::runtime_error("Device::Metal is unavailable");
}
Timings detail::run_gpu(Storage &, unsigned, KernelVariant) {
    throw std::runtime_error("Device::Metal is unavailable");
}
PendingBatch detail::submit_gpu(const std::shared_ptr<Storage> &, unsigned, KernelVariant) {
    throw std::runtime_error("Device::Metal is unavailable");
}
} // namespace batchfit::metal
#endif
