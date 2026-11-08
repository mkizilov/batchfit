// A standalone compiler/compute oracle: no downloaded headers are required to
// validate the shared header with Apple's real MSL compiler.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "model_checks.hpp"

#include <iostream>
#include <vector>

namespace {
void metal_require(bool success, const std::string &context, NSError *error = nil) {
    model_checks::require(
        success,
        context + ": " + (error ? error.localizedDescription.UTF8String : "no Metal error detail"));
}

template <typename Model, typename Reference>
void check(id<MTLDevice> device, id<MTLLibrary> library, const char *name, bool paired) {
    constexpr unsigned count = 257;
    std::vector<float> input(count * 9);
    std::mt19937 rng(0xBA7CF17U);
    for (unsigned i = 0; i < count; ++i) {
        auto sample = model_checks::random_sample<float>(rng, paired);
        input[i * 9] = sample.x;
        std::copy_n(sample.params, 8, input.data() + i * 9 + 1);
    }
    NSString *kernel = [NSString stringWithUTF8String:name];
    id<MTLFunction> function = [library newFunctionWithName:kernel];
    metal_require(function != nil, std::string(name) + " function lookup");
    NSError *error = nil;
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function
                                                                                 error:&error];
    metal_require(pipeline != nil, std::string(name) + " pipeline creation", error);
    const auto bytes = input.size() * sizeof(float);
    id<MTLBuffer> in = [device newBufferWithBytes:input.data()
                                           length:bytes
                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> out = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    metal_require(in != nil && out != nil, std::string(name) + " buffer allocation");
    id<MTLCommandQueue> queue = [device newCommandQueue];
    metal_require(queue != nil, std::string(name) + " command queue creation");
    id<MTLCommandBuffer> command = [queue commandBuffer];
    metal_require(command != nil, std::string(name) + " command buffer creation");
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    metal_require(encoder != nil, std::string(name) + " compute encoder creation");
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:in offset:0 atIndex:0];
    [encoder setBuffer:out offset:0 atIndex:1];
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(pipeline.threadExecutionWidth, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    metal_require(command.status == MTLCommandBufferStatusCompleted,
                  std::string(name) + " command buffer completion", command.error);
    const float *actual = static_cast<const float *>(out.contents);
    for (unsigned i = 0; i < count; ++i) {
        const float *params = input.data() + i * 9 + 1;
        const float x = input[i * 9];
        double reference_params[8];
        std::copy_n(params, 8, reference_params);
        double jacobian[Reference::parameter_count];
        Reference::jacobian(double(x), reference_params, jacobian);
        const auto context = std::string(name) + " sample=" + std::to_string(i);
        model_checks::near(actual[i * 9], Reference::eval(double(x), reference_params), 2e-5, 3e-5,
                           context + " eval vs CPU double");
        float cpu_jacobian[Model::parameter_count];
        Model::jacobian(x, params, cpu_jacobian);
        for (unsigned k = 0; k < Model::parameter_count; ++k) {
            model_checks::near(actual[i * 9 + k + 1], jacobian[k], 2e-5, 3e-5,
                               context + " Jacobian vs CPU double param=" + std::to_string(k));
            model_checks::near(actual[i * 9 + k + 1], cpu_jacobian[k], 2e-5, 3e-5,
                               context + " Jacobian vs CPU float param=" + std::to_string(k));
        }
    }
}
} // namespace

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        model_checks::require(argc == 2, "usage: models_metal <repository-root>");
        NSString *root = [NSString stringWithUTF8String:argv[1]];
        NSError *error = nil;
        NSString *header =
            [NSString stringWithContentsOfFile:
                          [root stringByAppendingPathComponent:@"include/batchfit/models.hpp"]
                                      encoding:NSUTF8StringEncoding
                                         error:&error];
        metal_require(header != nil, "read models.hpp", error);
        error = nil;
        NSString *source = [NSString
            stringWithContentsOfFile:[root
                                         stringByAppendingPathComponent:@"tests/models_check.metal"]
                            encoding:NSUTF8StringEncoding
                               error:&error];
        metal_require(source != nil, "read models_check.metal", error);
        // Runtime Metal compilation has no filesystem include search-path API.
        source = [source stringByReplacingOccurrencesOfString:@"#include <batchfit/models.hpp>"
                                                   withString:header];
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        metal_require(device != nil, "create default Metal device");
        MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
        if (@available(macOS 15.0, *)) {
            options.mathMode = MTLMathModeSafe;
        } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            options.fastMathEnabled = NO;
#pragma clang diagnostic pop
        }
#else
        options.fastMathEnabled = NO;
#endif
        error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
        metal_require(library != nil, "compile models_check.metal with shared models.hpp", error);
        using namespace batchfit::models;
        check<Gaussian<float>, Gaussian<double>>(device, library, "gaussian", false);
        check<Lorentzian<float>, Lorentzian<double>>(device, library, "lorentzian", false);
        check<PseudoVoigt<float>, PseudoVoigt<double>>(device, library, "pseudo_voigt", false);
        check<TwoLorentzian<float>, TwoLorentzian<double>>(device, library, "two_lorentzian", true);
        std::cout << "MSL header compilation and 1028 GPU model/Jacobian comparisons passed\n";
    }
}
