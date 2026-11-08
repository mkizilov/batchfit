#include "metal_workload.hpp"
#include <iomanip>
#include <iostream>
#include <sys/resource.h>

namespace {
using Variant = batchfit::metal::KernelVariant;
template <template <typename> class Model>
void measure(batchfit::metal::Context &context, const char *name, std::size_t n, std::size_t m,
             bool smoke) {
    constexpr unsigned p = Model<float>::parameter_count;
    batchfit_bench::MetalWorkload<Model> data(context, n, m);
    const std::array<Variant, 4> variants{Variant::ThreadPerFit, Variant::SimdPerFit,
                                          Variant::ThreadgroupPerFit, Variant::Automatic};
    const unsigned repeats = smoke ? 1 : 5;
    std::array<std::array<double, 5>, 4> gpu{}, wall{};
    std::array<std::size_t, 4> statuses{}, parameters{}, costs{}, converged{};
    std::array<batchfit::metal::Timings, 4> info{};
    // Warm full batches and rotate order to reduce clock/cache/order bias.
    for (auto variant : variants)
        data.run(variant);
    for (unsigned repeat = 0; repeat < repeats; ++repeat) {
        for (unsigned offset = 0; offset < 4; ++offset) {
            const auto index = (repeat + offset) % 4;
            const auto variant = variants[index];
            const auto start = std::chrono::steady_clock::now();
            const auto result = data.run(variant);
            wall[index][repeat] =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            gpu[index][repeat] = result.timings.gpu_seconds;
            info[index] = result.timings;
            std::size_t status_count = 0, parameter_count = 0, cost_count = 0, converged_count = 0;
            for (std::size_t i = 0; i < n; ++i) {
                status_count += result.status[i] != data.reference.status[i];
                converged_count += result.status[i] == batchfit::FitStatus::converged;
                cost_count += !(std::abs(result.chi_squared[i] - data.reference.chi_squared[i]) <=
                                1e-6F + 2e-5F * std::abs(data.reference.chi_squared[i]));
                bool bad = false;
                for (unsigned k = 0; k < p; ++k)
                    if (data.reference.status[i] == batchfit::FitStatus::converged)
                        bad =
                            bad || !(std::abs(result.parameters[i * p + k] -
                                              data.reference.parameters[i * p + k]) <=
                                     5e-4F * (1 + std::abs(data.reference.parameters[i * p + k])));
                parameter_count += bad;
            }
            statuses[index] = std::max(statuses[index], status_count);
            parameters[index] = std::max(parameters[index], parameter_count);
            costs[index] = std::max(costs[index], cost_count);
            converged[index] = converged_count;
        }
    }
    for (unsigned index = 0; index < 4; ++index) {
        std::sort(gpu[index].begin(), gpu[index].begin() + repeats);
        std::sort(wall[index].begin(), wall[index].begin() + repeats);
    }
    const auto mid = repeats / 2;
    unsigned best = 0;
    for (unsigned i = 1; i < 3; ++i)
        if (gpu[i][mid] < gpu[best][mid])
            best = i;
    for (unsigned index = 0; index < 4; ++index) {
        const auto mid = repeats / 2;
        std::cout << name << ',' << n << ',' << m << ',' << unsigned(variants[index]) << ','
                  << unsigned(info[index].variant) << ',' << std::setprecision(9) << gpu[index][mid]
                  << ',' << wall[index][mid] << ',' << double(n) / gpu[index][mid] << ','
                  << double(n) / wall[index][mid] << ',' << info[index].threads_per_threadgroup
                  << ',' << converged[index] << ',' << statuses[index] << ',' << parameters[index]
                  << ',' << costs[index] << ',' << best + 1 << ','
                  << 100 * (gpu[index][mid] / gpu[best][mid] - 1)
                  << std::endl; // NOLINT(performance-avoid-endl): preserve completed measurements
                                // on GPU failure
    }
    for (unsigned index = 0; index < 4; ++index)
        if (statuses[index] || parameters[index] || costs[index])
            throw std::runtime_error("cooperative benchmark CPU parity failure");
    const auto chosen = unsigned(info[3].variant) - 1;
    if (!smoke && gpu[chosen][mid] > gpu[best][mid] * 1.05)
        throw std::runtime_error(
            "dispatcher selected a kernel more than 5% slower than measured best");
}
} // namespace
int main(int argc, char **argv) {
    try {
        const bool smoke = argc == 2 && std::string_view(argv[1]) == "--smoke";
        if (argc != 1 && !smoke)
            throw std::invalid_argument("usage: cooperative_baseline [--smoke]");
        batchfit::metal::Context context;
        std::cout << "model,N,M,requested,selected,gpu_seconds,wall_seconds,gpu_fits_per_second,"
                     "wall_fits_per_second,threads_per_group,converged,status_errors,parameter_"
                     "errors,cost_errors,best_measured,regret_percent\n";
        for (unsigned m :
             (smoke ? std::vector<unsigned>{32, 512}
                    : std::vector<unsigned>{32, 64, 128, 256, 512, 1024, 2048, 4096})) {
            const unsigned n = smoke ? 257 : 16384;
            measure<batchfit::models::Gaussian>(context, "Gaussian", n, m, smoke);
            measure<batchfit::models::Lorentzian>(context, "Lorentzian", n, m, smoke);
            measure<batchfit::models::PseudoVoigt>(context, "PseudoVoigt", n, m, smoke);
            measure<batchfit::models::TwoLorentzian>(context, "TwoLorentzian", n, m, smoke);
        }
        rusage usage{};
        if (getrusage(RUSAGE_SELF, &usage) == 0)
            std::cerr << "peak_rss_bytes=" << usage.ru_maxrss << '\n';
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
    return 0;
}
