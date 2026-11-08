#import <Metal/Metal.h>
#include <batchfit/batchfit.hpp>
#include <batchfit/cpu/initial_guess.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

namespace {
void require(bool value, const std::string &message) {
    if (!value)
        throw std::runtime_error(message);
}
std::string metal_error(NSError *error) {
    const char *text = error.localizedDescription.UTF8String;
    return text ? text : "no error detail";
}
struct Regime {
    const char *name;
    double origin, half_range, amplitude;
};
constexpr Regime regimes[] = {{"ordinary", 0, 1, 1},
                              {"offset", 1e6, 5, 1},
                              {"small_units", 1e3, .005, 1e-6},
                              {"large_units", -1e5, 500, 1e6}};
template <typename T, std::size_t P>
batchfit::cpu::SolverOptions<T, P> options(double origin, double scale, std::size_t m) {
    batchfit::cpu::SolverOptions<T, P> result;
    result.lower[0] = 0;
    result.lower[1] = T(origin - scale);
    result.upper[1] = T(origin + scale);
    result.lower[2] = T(scale * .2 / double(m - 1));
    result.upper[2] = T(2 * scale);
    if constexpr (P == 4) {
        result.lower[3] = 0;
        result.upper[3] = 1;
    }
    if constexpr (P == 8) {
        result.lower[3] = 0;
        result.lower[4] = T(origin - scale);
        result.upper[4] = T(origin + scale);
        result.lower[5] = result.lower[2];
        result.upper[5] = result.upper[2];
    }
    return result;
}
template <template <typename> class Model> struct Cases {
    static constexpr unsigned p = Model<double>::parameter_count;
    std::size_t n, m;
    std::vector<double> x, y, physical_initial, normalized_x, normalized_y, normalized_initial;
    std::vector<batchfit::Coordinates> frames;
    std::vector<batchfit::cpu::FitResult<double, p>> references;
    std::vector<std::array<double, p>> physical_references;
    double raw_quantization_max = 0, normalized_quantization_max = 0;
    double reference_truth_squared = 0, reference_truth_max = 0;
    std::size_t reference_truth_terms = 0;
    unsigned guessed_failures = 0, reference_failures = 0, quantization_failures = 0;
    Cases(Regime regime, double snr, std::size_t points, std::size_t count)
        : n(count), m(points), x(m), y(n * m), physical_initial(n * p), normalized_x(m),
          normalized_y(n * m), normalized_initial(n * p) {
        std::mt19937_64 random(20261108 + std::uint64_t(p) * 7919 + points);
        std::uniform_real_distribution<double> uniform(0, 1);
        std::normal_distribution<double> normal;
        for (std::size_t j = 0; j < m; ++j)
            x[j] = regime.origin + regime.half_range * (-1 + 2 * double(j) / double(m - 1));
        for (std::size_t i = 0; i < n; ++i) {
            std::array<double, p> truth;
            if constexpr (p == 8)
                truth = {1.5 + uniform(random),
                         -.55 + .15 * uniform(random),
                         .06 + .06 * uniform(random),
                         1.5 + uniform(random),
                         .4 + .15 * uniform(random),
                         .06 + .06 * uniform(random),
                         .15,
                         .02 * (uniform(random) - .5)};
            else if constexpr (p == 4)
                truth = {1.5 + uniform(random), -.2 + .4 * uniform(random),
                         .15 + .2 * uniform(random), .2 + .6 * uniform(random)};
            else
                truth = {1.5 + uniform(random), -.2 + .4 * uniform(random),
                         .12 + .15 * uniform(random)};
            const batchfit::Coordinates base{regime.origin, regime.half_range, regime.amplitude};
            const auto physical = base.physical<Model>(truth);
            auto row = std::span<double>(y).subspan(i * m, m);
            double squared = 0;
            for (std::size_t j = 0; j < m; ++j) {
                row[j] = Model<double>::eval(x[j], physical.data());
                squared += row[j] * row[j];
            }
            const auto deviation = snr > 0 ? std::sqrt(squared / double(m)) / snr : 0;
            for (auto &value : row)
                value += deviation * normal(random);
            const auto frame = batchfit::Coordinates::from(x, row);
            frames.push_back(frame);
            std::vector<double> u(m), v(m), raw_x(m), raw_y(m), rounded_u(m), rounded_v(m);
            for (std::size_t j = 0; j < m; ++j) {
                u[j] = frame.x(x[j]);
                v[j] = frame.y(row[j]);
                normalized_x[j] = u[j];
                normalized_y[i * m + j] = v[j];
                raw_x[j] = double(float(x[j]));
                raw_y[j] = double(float(row[j]));
                rounded_u[j] = double(float(u[j]));
                rounded_v[j] = double(float(v[j]));
            }
            const auto guess = batchfit::cpu::initial_guess<double, Model>(u, v);
            guessed_failures += guess.status != batchfit::cpu::GuessStatus::success;
            std::copy(guess.parameters.begin(), guess.parameters.end(),
                      normalized_initial.data() + i * p);
            const auto seed = frame.template physical<Model>(guess.parameters);
            std::copy(seed.begin(), seed.end(), physical_initial.data() + i * p);
            auto opts = options<double, p>(0, 1, m);
            opts.compensated_sum = true;
            opts.scaled_cholesky = true;
            const auto reference =
                batchfit::cpu::fit_single<double, Model>(u, v, guess.parameters, opts);
            references.push_back(reference);
            physical_references.push_back(frame.template physical<Model>(reference.parameters));
            if (reference.status != batchfit::FitStatus::converged) {
                ++reference_failures;
                continue;
            }
            const auto truth_units = frame.template normalize<Model>(physical);
            for (unsigned k = 0; k < p; ++k) {
                const auto error = std::abs(reference.parameters[k] - truth_units[k]);
                reference_truth_max = std::max(reference_truth_max, error);
                reference_truth_squared += error * error;
                ++reference_truth_terms;
            }
            const auto raw_frame = batchfit::Coordinates::from(raw_x, raw_y);
            std::vector<double> qu(m), qv(m);
            for (std::size_t j = 0; j < m; ++j) {
                qu[j] = raw_frame.x(raw_x[j]);
                qv[j] = raw_frame.y(raw_y[j]);
            }
            const auto raw_seed = raw_frame.normalize<Model>(seed);
            const auto raw_floor = batchfit::cpu::fit_single<double, Model>(qu, qv, raw_seed, opts);
            const auto floor_units =
                frame.template normalize<Model>(raw_frame.physical<Model>(raw_floor.parameters));
            const auto normalized_floor = batchfit::cpu::fit_single<double, Model>(
                rounded_u, rounded_v, guess.parameters, opts);
            if (raw_floor.status != batchfit::FitStatus::converged ||
                normalized_floor.status != batchfit::FitStatus::converged) {
                ++quantization_failures;
                continue;
            }
            for (unsigned k = 0; k < p; ++k) {
                raw_quantization_max = std::max(raw_quantization_max,
                                                std::abs(floor_units[k] - reference.parameters[k]));
                normalized_quantization_max =
                    std::max(normalized_quantization_max,
                             std::abs(normalized_floor.parameters[k] - reference.parameters[k]));
            }
        }
    }
};

template <template <typename> class Model>
void measure(std::ostream &output, batchfit::metal::Context &safe, batchfit::metal::Context &fast,
             Regime regime, double snr, std::size_t m, std::size_t n) {
    constexpr unsigned p = Model<double>::parameter_count;
    const char *name =
        std::is_same_v<Model<double>, batchfit::models::Gaussian<double>>     ? "Gaussian"
        : std::is_same_v<Model<double>, batchfit::models::Lorentzian<double>> ? "Lorentzian"
        : p == 4                                                              ? "PseudoVoigt"
                                                                              : "TwoLorentzian";
    Cases<Model> data(regime, snr, m, n);
    for (bool normalized : {false, true})
        for (unsigned flags = 0; flags < 4; ++flags)
            for (unsigned math = 0; math < 2; ++math) {
                auto batch = safe.make_batch<Model>(n, m);
                // A batch belongs to its compiling context, including its math mode.
                if (math)
                    batch = fast.make_batch<Model>(n, m);
                for (std::size_t j = 0; j < m; ++j)
                    batch.x()[j] = float(normalized ? data.normalized_x[j] : data.x[j]);
                for (std::size_t i = 0; i < n; ++i) {
                    for (std::size_t j = 0; j < m; ++j)
                        batch.y()[j * n + i] =
                            float(normalized ? data.normalized_y[i * m + j] : data.y[i * m + j]);
                    for (unsigned k = 0; k < p; ++k)
                        batch.initial()[i * p + k] =
                            float(normalized ? data.normalized_initial[i * p + k]
                                             : data.physical_initial[i * p + k]);
                }
                auto opts = options<float, p>(normalized ? 0 : regime.origin,
                                              normalized ? 1 : regime.half_range, m);
                opts.compensated_sum = flags & 1;
                opts.scaled_cholesky = flags & 2;
                for (auto variant : {batchfit::metal::KernelVariant::ThreadPerFit,
                                     batchfit::metal::KernelVariant::SimdPerFit,
                                     batchfit::metal::KernelVariant::ThreadgroupPerFit}) {
                    const auto result =
                        batchfit::fit_batch<Model>(batch, batchfit::Device::Metal, opts, variant);
                    unsigned converged = 0, status_mismatches = 0, nonfinite = 0, accuracy_rows = 0;
                    double squared = 0, maximum = 0;
                    std::vector<double> errors;
                    std::array<double, 8> parameter_max{};
                    for (std::size_t i = 0; i < n; ++i) {
                        converged += result.status[i] == batchfit::FitStatus::converged;
                        status_mismatches += result.status[i] != data.references[i].status;
                        std::array<double, p> stored;
                        for (unsigned k = 0; k < p; ++k)
                            stored[k] = result.parameters[i * p + k];
                        const auto physical =
                            normalized ? data.frames[i].template physical<Model>(stored) : stored;
                        const auto units = normalized
                                               ? stored
                                               : data.frames[i].template normalize<Model>(physical);
                        bool finite = true;
                        for (auto value : units)
                            finite = finite && std::isfinite(value);
                        if (!finite) {
                            ++nonfinite;
                            continue;
                        }
                        if (data.references[i].status != batchfit::FitStatus::converged)
                            continue;
                        ++accuracy_rows;
                        for (unsigned k = 0; k < p; ++k) {
                            const auto error =
                                std::abs(units[k] - data.references[i].parameters[k]);
                            errors.push_back(error);
                            maximum = std::max(maximum, error);
                            squared += error * error;
                            parameter_max[k] =
                                std::max(parameter_max[k],
                                         std::abs(physical[k] - data.physical_references[i][k]));
                        }
                    }
                    std::sort(errors.begin(), errors.end());
                    const auto p95 = errors.empty() ? std::numeric_limits<double>::quiet_NaN()
                                                    : errors[(errors.size() - 1) * 95 / 100];
                    output << std::setprecision(12) << name << ',' << p << ',' << regime.name << ','
                           << snr << ',' << m << ',' << n << ',' << normalized << ',' << flags
                           << ',' << math << ',' << unsigned(variant) << ','
                           << data.guessed_failures << ',' << data.reference_failures << ','
                           << data.quantization_failures << ',' << converged << ','
                           << status_mismatches << ',' << nonfinite << ',' << accuracy_rows << ','
                           << maximum << ','
                           << (errors.empty() ? std::numeric_limits<double>::quiet_NaN()
                                              : std::sqrt(squared / double(errors.size())))
                           << ',' << p95 << ',' << data.raw_quantization_max << ','
                           << data.normalized_quantization_max << ','
                           << (data.reference_truth_terms
                                   ? std::sqrt(data.reference_truth_squared /
                                               double(data.reference_truth_terms))
                                   : std::numeric_limits<double>::quiet_NaN())
                           << ',' << data.reference_truth_max << ',' << result.timings.gpu_seconds;
                    for (auto value : parameter_max)
                        output << ',' << value;
                    output << '\n';
                    require(bool(output), "write numerical study CSV");
                }
            }
}

void probe(std::ostream &output) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        require(device != nil, "numerics probe device");
        id<MTLCommandQueue> queue = [device newCommandQueue];
        require(queue != nil, "numerics probe queue");
        NSString *source =
            @"#include <metal_stdlib>\nusing namespace metal;\nkernel void probe(device const "
            @"float* v [[buffer(0)]],device uint* out [[buffer(1)]],uint i "
            @"[[thread_position_in_grid]]){out[4*i]=isfinite(v[i]);out[4*i+1]=isnan(v[i]);out[4*i+"
            @"2]=(as_type<uint>(v[i])&0x7f800000u)!=0x7f800000u;out[4*i+3]=as_type<uint>(v[i]*2.0f)"
            @";}";
        output << "fast,input,builtin_finite,builtin_nan,bit_finite,twice_bits\n";
        for (bool fast : {false, true}) {
            MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
            require(options != nil, "allocate math probe compiler options");
            options.mathMode = fast ? MTLMathModeFast : MTLMathModeSafe;
            options.mathFloatingPointFunctions =
                fast ? MTLMathFloatingPointFunctionsFast : MTLMathFloatingPointFunctionsPrecise;
            NSError *error = nil;
            id<MTLLibrary> library = [device newLibraryWithSource:source
                                                          options:options
                                                            error:&error];
            require(library != nil, "compile math probe: " + metal_error(error));
            id<MTLFunction> function = [library newFunctionWithName:@"probe"];
            require(function != nil, "math probe kernel");
            id<MTLComputePipelineState> pipeline =
                [device newComputePipelineStateWithFunction:function error:&error];
            require(pipeline != nil, "math probe pipeline: " + metal_error(error));
            const std::array<float, 5> values{
                1, std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
                -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::max()};
            id<MTLBuffer> input = [device newBufferWithBytes:values.data()
                                                      length:sizeof(values)
                                                     options:MTLResourceStorageModeShared];
            id<MTLBuffer> result =
                [device newBufferWithLength:std::size_t(20) * sizeof(std::uint32_t)
                                    options:MTLResourceStorageModeShared];
            require(input != nil && result != nil, "math probe buffers");
            id<MTLCommandBuffer> command = [queue commandBuffer];
            require(command != nil, "math probe command");
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            require(encoder != nil, "math probe encoder");
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:input offset:0 atIndex:0];
            [encoder setBuffer:result offset:0 atIndex:1];
            [encoder dispatchThreads:MTLSizeMake(5, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(5, 1, 1)];
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
            require(command.status == MTLCommandBufferStatusCompleted && command.error == nil,
                    "math probe completion: " + metal_error(command.error));
            const auto *bits = static_cast<const std::uint32_t *>(result.contents);
            for (std::size_t i = 0; i < 5; ++i)
                output << fast << ',' << i << ',' << bits[4 * i] << ',' << bits[4 * i + 1] << ','
                       << bits[4 * i + 2] << ',' << bits[4 * i + 3] << '\n';
        }
    }
}
} // namespace
int main(int argc, char **argv) {
    try {
        bool smoke = false;
        std::filesystem::path output_path, probe_path;
        for (int i = 1; i < argc; ++i) {
            std::string_view argument = argv[i];
            if (argument == "--smoke")
                smoke = true;
            else if (argument == "--output" && i + 1 < argc)
                output_path = argv[++i];
            else if (argument == "--probe" && i + 1 < argc)
                probe_path = argv[++i];
            else
                throw std::invalid_argument(
                    "usage: batchfit_numerics_study [--smoke] [--output CSV] [--probe CSV]");
        }
        std::ofstream file, probe_file;
        std::ostream *output = &std::cout;
        if (!output_path.empty()) {
            require(!std::filesystem::exists(output_path), "refusing existing numerical results");
            file.open(output_path);
            require(bool(file), "open numerical CSV");
            output = &file;
        }
        if (!probe_path.empty()) {
            require(!std::filesystem::exists(probe_path), "refusing existing probe CSV");
            probe_file.open(probe_path);
            require(bool(probe_file), "open probe CSV");
            probe(probe_file);
            probe_file.close();
            require(bool(probe_file), "close probe CSV");
        }
        *output << "model,parameters,regime,snr,points,fits,normalized,flags,fast,kernel,guess_"
                   "failures,"
                   "reference_failures,quantization_reference_failures,converged,status_mismatches,"
                   "nonfinite_rows,accuracy_rows,max_unit_error,rms_unit_error,p95_unit_error,raw_"
                   "input_quantization_max,normalized_input_quantization_max,reference_truth_rms,"
                   "reference_truth_max,gpu_seconds,p0_max_abs,p1_"
                   "max_abs,p2_max_abs,p3_max_abs,p4_max_abs,p5_max_abs,p6_max_abs,p7_max_abs\n";
        batchfit::metal::Context safe, fast(batchfit::metal::MathMode::Fast);
        for (const auto regime : regimes) {
            if (smoke && std::string_view(regime.name) != "ordinary")
                continue;
            for (double snr : {0., 10., 30., 100., 1000.}) {
                if (smoke && snr != 30)
                    continue;
                for (std::size_t m : {65U, 2049U}) {
                    if (smoke && m != 65)
                        continue;
                    const std::size_t n = smoke ? 4 : 16;
                    measure<batchfit::models::Gaussian>(*output, safe, fast, regime, snr, m, n);
                    measure<batchfit::models::Lorentzian>(*output, safe, fast, regime, snr, m, n);
                    measure<batchfit::models::PseudoVoigt>(*output, safe, fast, regime, snr, m, n);
                    measure<batchfit::models::TwoLorentzian>(*output, safe, fast, regime, snr, m,
                                                             n);
                }
            }
        }
        if (file.is_open()) {
            file.close();
            require(bool(file), "close numerical CSV");
        }
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
