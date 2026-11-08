// Test-only SDK driver for the production embedded shader/public API when
// metal-cpp downloads are unavailable. Never linked into the production library.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "lm_source.hpp"
#include "storage.hpp"

#include <mutex>
#include <string>

namespace batchfit::metal {
namespace detail {
void require(bool success, const std::string &operation, NSError *error = nil) {
    if (!success)
        throw std::runtime_error(
            "Native Metal LM: " + operation + ": " +
            (error ? error.localizedDescription.UTF8String : "no error detail"));
}
struct Runtime {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    id<MTLComputePipelineState> pipelines[4];
    std::mutex mutex;
};
struct Resources {
    std::shared_ptr<Runtime> runtime;
    id<MTLBuffer> buffers[8];
};
} // namespace detail

Context::Context() : runtime_(std::make_shared<detail::Runtime>()) {
    @autoreleasepool {
        runtime_->device = MTLCreateSystemDefaultDevice();
        detail::require(runtime_->device != nil, "create default device");
        runtime_->queue = [runtime_->device newCommandQueue];
        detail::require(runtime_->queue != nil, "create command queue");
        MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
        if (@available(macOS 15.0, *)) {
            options.mathMode = MTLMathModeSafe;
            options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
        } else {
            throw std::runtime_error("safe/precise Metal LM requires macOS 15 or newer");
        }
#else
        throw std::runtime_error("safe/precise Metal LM requires a macOS 15 or newer SDK");
#endif
        NSError *error = nil;
        runtime_->library = [runtime_->device
            newLibraryWithSource:[NSString stringWithUTF8String:detail::shader_source]
                         options:options
                           error:&error];
        detail::require(runtime_->library != nil,
                        "compile src/metal/lm.metal with #line shared headers", error);
        for (unsigned k = 0; k < 4; ++k) {
            id<MTLFunction> function = [runtime_->library
                newFunctionWithName:[NSString stringWithUTF8String:detail::kernel_names[k]]];
            detail::require(function != nil, std::string("find ") + detail::kernel_names[k]);
            error = nil;
            runtime_->pipelines[k] = [runtime_->device newComputePipelineStateWithFunction:function
                                                                                     error:&error];
            detail::require(runtime_->pipelines[k] != nil,
                            std::string("pipeline ") + detail::kernel_names[k], error);
        }
    }
}

SharedBatch Context::allocate(std::size_t n, std::size_t m, unsigned p) const {
    const auto counts = detail::buffer_counts(n, m, p);
    @autoreleasepool {
        auto resources = std::make_shared<detail::Resources>();
        resources->runtime = runtime_;
        std::array<void *, 8> pointers{};
        for (unsigned k = 0; k < 8; ++k) {
            if (counts[k] == 0)
                continue;
            detail::require(counts[k] * 4 <= runtime_->device.maxBufferLength,
                            "buffer exceeds maxBufferLength");
            resources->buffers[k] =
                [runtime_->device newBufferWithLength:counts[k] * 4
                                              options:MTLResourceStorageModeShared];
            detail::require(resources->buffers[k] != nil, "allocate shared buffer");
            pointers[k] = resources->buffers[k].contents;
            detail::require(pointers[k] != nullptr, "map shared buffer");
        }
        return SharedBatch(detail::make_storage(n, m, p, pointers, resources));
    }
}

Timings detail::run_gpu(Storage &storage, unsigned kernel) {
    const auto start = std::chrono::steady_clock::now();
    const auto resources = std::static_pointer_cast<Resources>(storage.resources);
    auto &runtime = *resources->runtime;
    std::lock_guard lock(runtime.mutex);
    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = runtime.pipelines[kernel];
        const auto width = pipeline.threadExecutionWidth,
                   maximum = pipeline.maxTotalThreadsPerThreadgroup;
        require(width > 0 && maximum >= width, "pipeline thread limits");
        const auto group = width * std::min(NSUInteger(8), maximum / width);
        if (storage.fits == 0)
            return {0,
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(),
                    width, group};
        id<MTLCommandBuffer> command = [runtime.queue commandBuffer];
        require(command != nil, "command buffer creation");
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        require(encoder != nil, "encoder creation");
        [encoder setComputePipelineState:pipeline];
        for (unsigned k = 0; k < 8; ++k)
            [encoder setBuffer:resources->buffers[k] offset:0 atIndex:k];
        [encoder dispatchThreads:MTLSizeMake(storage.fits, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        const auto context = std::string(kernel_names[kernel]) +
                             " N=" + std::to_string(storage.fits) +
                             " M=" + std::to_string(storage.points);
        require(command.status == MTLCommandBufferStatusCompleted,
                context + " command status=" + std::to_string(command.status), command.error);
        require(command.error == nil, context + " command error", command.error);
        const double begin = command.GPUStartTime, end = command.GPUEndTime;
        require(std::isfinite(begin) && std::isfinite(end) && end > begin,
                context + " GPU timestamps");
        return {end - begin,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(),
                width, group};
    }
}
} // namespace batchfit::metal

void verify_native_shader_diagnostics() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        batchfit::metal::detail::require(device != nil, "diagnostic test device");
        NSError *error = nil;
        id<MTLLibrary> invalid =
            [device newLibraryWithSource:
                        @"#line 71 \"src/metal/diagnostic_test.metal\"\nthis_is_not_valid_msl;\n"
                                 options:nil
                                   error:&error];
        const std::string text = error ? error.localizedDescription.UTF8String : "";
        batchfit::metal::detail::require(invalid == nil && error != nil &&
                                             text.find("src/metal/diagnostic_test.metal:71") !=
                                                 std::string::npos,
                                         "shader compiler must report file and line", error);
    }
    __weak id<MTLBuffer> released_buffer = nil;
    @autoreleasepool {
        batchfit::metal::Context context;
        auto batch = context.make_batch<batchfit::models::Gaussian>(3, 7);
        auto resources = std::static_pointer_cast<batchfit::metal::detail::Resources>(
            batchfit::metal::detail::Access::get(batch).resources);
        released_buffer = resources->buffers[1];
        batchfit::metal::detail::require(released_buffer != nil, "weak buffer ownership probe");
    }
    batchfit::metal::detail::require(released_buffer == nil,
                                     "shared buffer released after last owner");
}
