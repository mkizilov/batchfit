#include "metal_workload.hpp"

#include <iomanip>
#include <iostream>
#include <sys/resource.h>

namespace {
template <template <typename> class Model>
void measure(batchfit::metal::Context &context, const char *name, std::size_t n, std::size_t m,
             bool smoke) {
    constexpr unsigned p = Model<float>::parameter_count;
    batchfit_bench::MetalWorkload<Model> data(context, n, m);
    const auto &cpu = data.reference;
    const auto fallbacks = data.fallback_guesses;
    data.warmup(context);
    const unsigned repeats = smoke ? 1 : 3;
    std::array<double, 3> gpu_times{}, wall_times{};
    std::size_t converged = 0, status_differences = 0, parameter_differences = 0,
                chi_differences = 0;
    double max_parameter_absolute = 0, max_parameter_scaled = 0;
    std::size_t converged_strict_differences = 0, reference_floor_differences = 0;
    bool parity_failed = false;
    batchfit::metal::Timings geometry;
    for (unsigned repeat = 0; repeat < repeats; ++repeat) {
        const auto start = std::chrono::steady_clock::now();
        const auto result = data.run();
        wall_times[repeat] =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        gpu_times[repeat] = result.timings.gpu_seconds;
        geometry = result.timings;
        converged = status_differences = parameter_differences = chi_differences = 0;
        max_parameter_absolute = max_parameter_scaled = 0;
        converged_strict_differences = reference_floor_differences = 0;
        for (std::size_t i = 0; i < n; ++i) {
            converged += result.status[i] == batchfit::FitStatus::converged;
            status_differences += result.status[i] != cpu.status[i];
            bool mismatch = false;
            bool floor_mismatch = false;
            for (unsigned k = 0; k < p; ++k) {
                const double error =
                    std::abs(double(result.parameters[i * p + k]) - cpu.parameters[i * p + k]);
                const double tolerance = 2e-4 + 2e-4 * std::abs(double(cpu.parameters[i * p + k]));
                max_parameter_absolute = std::max(max_parameter_absolute, error);
                max_parameter_scaled = std::max(max_parameter_scaled, error / tolerance);
                mismatch = mismatch || !(error <= tolerance);
                floor_mismatch =
                    floor_mismatch ||
                    !(error <= 5e-4 * (1 + std::abs(double(cpu.parameters[i * p + k]))));
            }
            parameter_differences += mismatch;
            if (cpu.status[i] == batchfit::FitStatus::converged) {
                converged_strict_differences += mismatch;
                reference_floor_differences += floor_mismatch;
                if (floor_mismatch && repeat == 0 && reference_floor_differences <= 3) {
                    std::cerr << name << " M=" << m << " fit=" << i
                              << " iterations cpu=" << cpu.iterations[i]
                              << " gpu=" << result.iterations[i]
                              << " chi cpu=" << cpu.chi_squared[i]
                              << " gpu=" << result.chi_squared[i] << '\n';
                    for (unsigned k = 0; k < p; ++k)
                        std::cerr << "  p" << k << " cpu=" << cpu.parameters[i * p + k]
                                  << " gpu=" << result.parameters[i * p + k] << '\n';
                }
            }
            chi_differences += !(std::abs(result.chi_squared[i] - cpu.chi_squared[i]) <=
                                 1e-6F + 2e-5F * std::abs(cpu.chi_squared[i]));
        }
        parity_failed =
            parity_failed || status_differences || chi_differences || reference_floor_differences;
    }
    std::sort(gpu_times.begin(), gpu_times.begin() + repeats);
    std::sort(wall_times.begin(), wall_times.begin() + repeats);
    const double gpu = gpu_times[repeats / 2], wall = wall_times[repeats / 2];
    std::cout << name << ',' << n << ',' << m << ',' << repeats << ',' << std::setprecision(9)
              << gpu << ',' << wall << ',' << double(n) / gpu << ',' << double(n) / wall << ','
              << geometry.simd_width << ',' << geometry.threads_per_threadgroup << ',' << converged
              << ',' << status_differences << ',' << parameter_differences << ',' << chi_differences
              << ',' << fallbacks << ',' << max_parameter_absolute << ',' << max_parameter_scaled
              << ',' << converged_strict_differences << ',' << reference_floor_differences
              << std::endl;
    if (parity_failed)
        throw std::runtime_error(
            std::string(name) + " M=" + std::to_string(m) +
            " GPU parity failed; all rows remain in timing and mismatch counts above");
}
} // namespace

int main(int argc, char **argv) {
    try {
        const bool smoke = argc == 2 && std::string_view(argv[1]) == "--smoke";
        if (argc != 1 && !smoke)
            throw std::invalid_argument("usage: batchfit_metal_baseline [--smoke]");
        batchfit::metal::Context context;
        std::cout << "model,N,M,repeats,gpu_seconds,wall_seconds,gpu_fits_per_second,wall_fits_per_"
                     "second,simd_width,threads_per_group,converged,status_differences,parameter_"
                     "differences,chi_differences,fallbacks,max_parameter_absolute,max_parameter_"
                     "scaled,converged_strict_differences,reference_floor_differences\n";
        for (auto m :
             (smoke ? std::vector<unsigned>{32} : std::vector<unsigned>{32, 128, 512, 2048})) {
            const auto n = smoke ? 257U : 100000U;
            measure<batchfit::models::Gaussian>(context, "Gaussian", n, m, smoke);
            measure<batchfit::models::Lorentzian>(context, "Lorentzian", n, m, smoke);
            measure<batchfit::models::PseudoVoigt>(context, "PseudoVoigt", n, m, smoke);
            measure<batchfit::models::TwoLorentzian>(context, "TwoLorentzian", n, m, smoke);
        }
        rusage usage{};
        if (getrusage(RUSAGE_SELF, &usage) != 0)
            throw std::runtime_error("getrusage failed");
#if defined(__APPLE__)
        std::cout << "peak_rss_bytes=" << usage.ru_maxrss << '\n';
#else
        std::cout << "peak_rss_bytes=" << usage.ru_maxrss * 1024ULL << '\n';
#endif
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        rusage usage{};
        if (getrusage(RUSAGE_SELF, &usage) == 0) {
#if defined(__APPLE__)
            std::cout << "peak_rss_bytes=" << usage.ru_maxrss << '\n';
#else
            std::cout << "peak_rss_bytes=" << usage.ru_maxrss * 1024ULL << '\n';
#endif
        } else
            std::cerr << "getrusage failed during error reporting\n";
        return 1;
    }
    return 0;
}
