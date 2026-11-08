// Installed-SDK profiler for the embedded production algorithm and opt-in variants.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "capture_native.hpp"
#include "cpu_workload.hpp"
#include "profile_source.hpp"
#include "storage.hpp"

#include <iomanip>
#include <iostream>
#include <numeric>
#include <sys/resource.h>

namespace {
namespace detail = batchfit::metal::detail;
void check(bool valid, const std::string &context, NSError *error = nil) {
    if (!valid)
        throw std::runtime_error(
            context + ": " + (error ? error.localizedDescription.UTF8String : "no error detail"));
}
struct Profiler {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    id<MTLCounterSet> timestamps = nil;
    bool parity_failed = false;

    Profiler() {
        detail::prepare_capture_environment();
        device = MTLCreateSystemDefaultDevice();
        check(device != nil, "create profiling device");
        queue = [device newCommandQueue];
        check(queue != nil, "create profiling queue");
        MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
        if (@available(macOS 15.0, *)) {
            options.mathMode = MTLMathModeSafe;
            options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
        } else {
            throw std::runtime_error("Metal profiling requires macOS 15 or newer");
        }
        NSError *error = nil;
        library = [device
            newLibraryWithSource:[NSString stringWithUTF8String:detail::profile_shader_source]
                         options:options
                           error:&error];
        check(library != nil, "compile embedded production/profile kernels", error);
        std::cerr << "device=" << device.name.UTF8String << " counter_stage_sampling=" <<
            [device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]
                  << " counter_dispatch_sampling=" <<
            [device supportsCounterSampling:MTLCounterSamplingPointAtDispatchBoundary] << '\n';
        for (id<MTLCounterSet> set in device.counterSets) {
            std::cerr << "counter_set=" << set.name.UTF8String;
            for (id<MTLCounter> counter in set.counters)
                std::cerr << " " << counter.name.UTF8String;
            std::cerr << '\n';
            if ([set.name isEqualToString:MTLCommonCounterSetTimestamp])
                timestamps = set;
        }
    }

    template <template <typename> class Model>
    void model(const char *name, std::size_t n, std::size_t m, bool single, bool smoke) {
        constexpr unsigned p = Model<float>::parameter_count;
        batchfit_bench::Workload<Model> data(n, m);
        batchfit::cpu::ThreadPool pool(2);
        const auto full_reference = data.run(pool);
        batchfit::cpu::BatchResult<float> half_reference{
            std::vector<float>(n * p), std::vector<batchfit::FitStatus>(n),
            std::vector<unsigned>(n), std::vector<float>(n)};
        double input_squared_error = 0, input_max_error = 0;
        for (const float value : data.y) {
            const double error = double(float(_Float16(value))) - value;
            input_squared_error += error * error;
            input_max_error = std::max(input_max_error, std::abs(error));
        }
        pool.parallel_for(n, [&](std::size_t begin, std::size_t end) {
            std::vector<float> row(m);
            for (auto i = begin; i < end; ++i) {
                for (std::size_t j = 0; j < m; ++j)
                    row[j] = float(_Float16(data.y[i * m + j]));
                const auto fit = batchfit::cpu::fit_single<float, Model>(
                    data.x, row, std::span<const float>(data.initial).subspan(i * p, p),
                    data.options);
                std::copy(fit.parameters.begin(), fit.parameters.end(),
                          half_reference.parameters.begin() + i * p);
                half_reference.status[i] = fit.status;
                half_reference.iterations[i] = fit.iterations;
                half_reference.chi_squared[i] = fit.chi_squared;
            }
        });
        auto counts = detail::buffer_counts(n, m, p);
        id<MTLBuffer> buffers[8];
        for (unsigned k = 0; k < 8; ++k) {
            if (k == 1)
                continue;
            buffers[k] = [device newBufferWithLength:counts[k] * 4
                                             options:MTLResourceStorageModeShared];
            check(buffers[k] != nil, "allocate profile buffer");
        }
        std::copy(data.x.begin(), data.x.end(), static_cast<float *>(buffers[0].contents));
        auto *config = static_cast<detail::Config *>(buffers[3].contents);
        *config = {};
        config->fits = unsigned(n);
        config->points = unsigned(m);
        config->max_iterations = data.options.max_iterations;
        config->initial_lambda = data.options.initial_lambda;
        config->gradient_tolerance = data.options.gradient_tolerance;
        config->step_tolerance = data.options.step_tolerance;
        config->cost_tolerance = data.options.cost_tolerance;
        for (unsigned k = 0; k < p; ++k) {
            config->lower[k] = data.options.lower[k];
            config->upper[k] = data.options.upper[k];
        }

        const auto experiment = [&](bool point_major, bool half, bool sorted) {
            @autoreleasepool {
                std::vector<std::size_t> order(n);
                std::iota(order.begin(), order.end(), 0);
                const auto &reference = half ? half_reference : full_reference;
                if (sorted)
                    std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) {
                        return reference.iterations[a] < reference.iterations[b];
                    });
                const std::size_t bytes = n * m * (half ? 2 : 4);
                buffers[1] = [device newBufferWithLength:bytes
                                                 options:MTLResourceStorageModeShared];
                check(buffers[1] != nil, "allocate observation variant");
                auto *initial = static_cast<float *>(buffers[2].contents);
                for (std::size_t i = 0; i < n; ++i) {
                    const auto original = order[i];
                    std::copy_n(data.initial.begin() + original * p, p, initial + i * p);
                    for (std::size_t j = 0; j < m; ++j) {
                        const auto index = point_major ? j * n + i : i * m + j;
                        if (half)
                            static_cast<_Float16 *>(buffers[1].contents)[index] =
                                _Float16(data.y[original * m + j]);
                        else
                            static_cast<float *>(buffers[1].contents)[index] =
                                data.y[original * m + j];
                    }
                }
                const std::string base = detail::kernel_names[detail::kernel_index<Model>()];
                const std::string kernel = point_major && !half
                                               ? base
                                               : base + (point_major ? "_point_half"
                                                         : half      ? "_fit_half"
                                                                     : "_fit_float");
                id<MTLFunction> function =
                    [library newFunctionWithName:[NSString stringWithUTF8String:kernel.c_str()]];
                check(function != nil, "find " + kernel);
                NSError *error = nil;
                id<MTLComputePipelineState> pipeline =
                    [device newComputePipelineStateWithFunction:function error:&error];
                check(pipeline != nil, "pipeline " + kernel, error);
                std::cerr << "pipeline=" << kernel << " simd=" << pipeline.threadExecutionWidth
                          << " max_group=" << pipeline.maxTotalThreadsPerThreadgroup
                          << " static_threadgroup_bytes=" << pipeline.staticThreadgroupMemoryLength
                          << '\n';
                const auto width = pipeline.threadExecutionWidth,
                           maximum = pipeline.maxTotalThreadsPerThreadgroup;
                std::vector<unsigned> groups =
                    (single || sorted) ? std::vector<unsigned>{256}
                    : smoke            ? std::vector<unsigned>{32, 256}
                                       : std::vector<unsigned>{32, 64, 128, 256, 512, 1024};
                for (const auto group : groups) {
                    if (group > maximum || group % width)
                        continue;
                    id<MTLCounterSampleBuffer> sample = nil;
                    if (timestamps &&
                        [device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) {
                        MTLCounterSampleBufferDescriptor *descriptor =
                            [[MTLCounterSampleBufferDescriptor alloc] init];
                        descriptor.counterSet = timestamps;
                        descriptor.storageMode = MTLStorageModeShared;
                        descriptor.sampleCount = 2;
                        error = nil;
                        sample = [device newCounterSampleBufferWithDescriptor:descriptor
                                                                        error:&error];
                        check(sample != nil, "allocate timestamp counter samples", error);
                    }
                    std::array<double, 3> gpu_times{}, wall_times{}, counter_times{};
                    const unsigned repetitions = single || smoke ? 1 : 3;
                    for (unsigned repeat = 0; repeat < repetitions; ++repeat) {
                        const auto start = std::chrono::steady_clock::now();
                        detail::NativeCapture capture(queue);
                        id<MTLCommandBuffer> command = [queue commandBuffer];
                        check(command != nil, "profile command buffer");
                        command.label = [NSString stringWithUTF8String:kernel.c_str()];
                        MTLComputePassDescriptor *pass =
                            [MTLComputePassDescriptor computePassDescriptor];
                        if (sample) {
                            pass.sampleBufferAttachments[0].sampleBuffer = sample;
                            pass.sampleBufferAttachments[0].startOfEncoderSampleIndex = 0;
                            pass.sampleBufferAttachments[0].endOfEncoderSampleIndex = 1;
                        }
                        id<MTLComputeCommandEncoder> encoder =
                            [command computeCommandEncoderWithDescriptor:pass];
                        check(encoder != nil, "profile encoder");
                        encoder.label = command.label;
                        [encoder setComputePipelineState:pipeline];
                        for (unsigned k = 0; k < 8; ++k)
                            [encoder setBuffer:buffers[k] offset:0 atIndex:k];
                        [encoder dispatchThreads:MTLSizeMake(n, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
                        [encoder endEncoding];
                        [command commit];
                        [command waitUntilCompleted];
                        check(command.status == MTLCommandBufferStatusCompleted,
                              "profile command status", command.error);
                        check(command.error == nil, "profile command error", command.error);
                        gpu_times[repeat] = command.GPUEndTime - command.GPUStartTime;
                        check(std::isfinite(gpu_times[repeat]) && gpu_times[repeat] > 0,
                              "profile GPU timestamps");
                        capture.finish();
                        wall_times[repeat] =
                            std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
                                .count();
                        if (sample) {
                            NSData *resolved = [sample resolveCounterRange:NSMakeRange(0, 2)];
                            check(resolved.length >= 2 * sizeof(MTLCounterResultTimestamp),
                                  "resolve timestamp samples");
                            const auto *values =
                                static_cast<const MTLCounterResultTimestamp *>(resolved.bytes);
                            check(values[1].timestamp > values[0].timestamp,
                                  "timestamp counter order");
                            counter_times[repeat] =
                                double(values[1].timestamp - values[0].timestamp) * 1e-9;
                        }
                    }
                    std::size_t statuses = 0, parameters = 0, costs = 0, converged = 0,
                                quantization_parameter_rows = 0;
                    double max_absolute = 0;
                    std::uint64_t hash = 14695981039346656037ULL;
                    const auto *output = static_cast<const float *>(buffers[4].contents);
                    const auto *status =
                        static_cast<const batchfit::FitStatus *>(buffers[5].contents);
                    const auto *iterations = static_cast<const unsigned *>(buffers[6].contents);
                    const auto *chi = static_cast<const float *>(buffers[7].contents);
                    for (std::size_t i = 0; i < n; ++i) {
                        const auto original = order[i];
                        bool bad = false, quantized = false;
                        statuses += status[i] != reference.status[original];
                        converged += status[i] == batchfit::FitStatus::converged;
                        costs += !(std::abs(chi[i] - reference.chi_squared[original]) <=
                                   1e-6F + 2e-5F * std::abs(reference.chi_squared[original]));
                        for (unsigned k = 0; k < p; ++k) {
                            const auto value = output[i * p + k],
                                       expected = reference.parameters[original * p + k];
                            const double difference = std::abs(double(value) - expected);
                            max_absolute = std::max(max_absolute, difference);
                            if (reference.status[original] == batchfit::FitStatus::converged)
                                bad =
                                    bad || !(difference <= 5e-4 * (1 + std::abs(double(expected))));
                            quantized =
                                quantized ||
                                !(std::abs(value - full_reference.parameters[original * p + k]) <=
                                  2e-4F + 2e-4F * std::abs(
                                                      full_reference.parameters[original * p + k]));
                            hash = (hash ^ std::bit_cast<std::uint32_t>(value)) * 1099511628211ULL;
                        }
                        parameters += bad;
                        quantization_parameter_rows += quantized;
                    }
                    parity_failed = parity_failed || statuses || parameters || costs;
                    std::uint64_t useful = 0, scheduled = 0;
                    unsigned minimum = 100, maximum_iterations = 0;
                    std::vector<unsigned> distribution(iterations, iterations + n);
                    for (std::size_t begin = 0; begin < n; begin += width) {
                        unsigned longest = 0;
                        for (auto i = begin; i < std::min(n, begin + width); ++i) {
                            useful += iterations[i];
                            longest = std::max(longest, iterations[i]);
                            minimum = std::min(minimum, iterations[i]);
                            maximum_iterations = std::max(maximum_iterations, iterations[i]);
                        }
                        scheduled += width * longest;
                    }
                    std::sort(distribution.begin(), distribution.end());
                    std::sort(gpu_times.begin(), gpu_times.begin() + repetitions);
                    std::sort(wall_times.begin(), wall_times.begin() + repetitions);
                    std::sort(counter_times.begin(), counter_times.begin() + repetitions);
                    const auto middle = repetitions / 2;
                    const auto gpu = gpu_times[middle];
                    std::cout << name << ',' << n << ',' << m << ',' << group << ',' << width << ','
                              << maximum << ',' << (point_major ? "point" : "fit") << ','
                              << (half ? "half" : "float") << ','
                              << (sorted ? "oracle" : "original") << ',' << std::setprecision(9)
                              << gpu << ',' << wall_times[middle] << ',' << double(n) / gpu << ','
                              << counter_times[middle] << ',' << converged << ',' << statuses << ','
                              << parameters << ',' << costs << ',' << max_absolute << ','
                              << (scheduled ? double(useful) / double(scheduled) : 1) << ','
                              << minimum << ',' << distribution[n / 2] << ','
                              << distribution[n * 9 / 10] << ',' << distribution[n * 99 / 100]
                              << ',' << maximum_iterations << ',' << (half ? input_max_error : 0)
                              << ','
                              << (half ? std::sqrt(input_squared_error / double(data.y.size())) : 0)
                              << ',' << quantization_parameter_rows << ',' << hash << std::endl;
                }
                buffers[1] = nil;
            }
        };
        experiment(true, false, false);
        if (!single) {
            experiment(false, false, false);
            experiment(true, true, false);
            experiment(false, true, false);
            experiment(true, false, true);
        }
    }
};
} // namespace

int main(int argc, char **argv) {
    try {
        @autoreleasepool {
            const bool single = argc == 2 && std::string_view(argv[1]) == "--single";
            const bool smoke = argc == 2 && std::string_view(argv[1]) == "--smoke";
            if (argc != 1 && !single && !smoke)
                throw std::invalid_argument("usage: batchfit_metal_profile [--single|--smoke]");
            Profiler profiler;
            std::cout
                << "model,N,M,group,simd,max_group,layout,storage,order,gpu_seconds,wall_seconds,"
                   "gpu_fits_per_second,counter_seconds,converged,status_errors,parameter_errors,"
                   "cost_errors,max_parameter_abs,iteration_lane_efficiency,min_iterations,p50_"
                   "iterations,p90_iterations,p99_iterations,max_iterations,input_max_quantization,"
                   "input_rms_quantization,strict_rows_vs_float_cpu,parameter_hash\n";
            const auto n = single ? 4096U : smoke ? 257U : 100000U;
            const auto m = smoke ? 32U : 512U;
            if (!single) {
                profiler.model<batchfit::models::Gaussian>("Gaussian", n, m, false, smoke);
                profiler.model<batchfit::models::Lorentzian>("Lorentzian", n, m, false, smoke);
                profiler.model<batchfit::models::PseudoVoigt>("PseudoVoigt", n, m, false, smoke);
            }
            profiler.model<batchfit::models::TwoLorentzian>("TwoLorentzian", n, m, single, smoke);
            rusage usage{};
            if (getrusage(RUSAGE_SELF, &usage) == 0)
                std::cerr << "peak_rss_bytes=" << usage.ru_maxrss << '\n';
            if (profiler.parity_failed)
                throw std::runtime_error("profiling parity failures retained in output");
        }
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
