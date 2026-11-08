#include "metal_workload.hpp"

#include <benchmark/benchmark.h>

#include <optional>

namespace {
template <template <typename> class Model> void metal_lm(benchmark::State &state) {
    const auto n = std::size_t(state.range(0)), m = std::size_t(state.range(1));
    batchfit::metal::Context context;
    batchfit_bench::MetalWorkload<Model> data(context, n, m);
    data.warmup(context);
    double gpu_total = 0, wall_total = 0;
    std::optional<batchfit::metal::Result> result;
    for (auto _ : state) {
        const auto start = std::chrono::steady_clock::now();
        result.emplace(data.run());
        wall_total +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        gpu_total += result->timings.gpu_seconds;
        state.SetIterationTime(result->timings.gpu_seconds);
        benchmark::DoNotOptimize(result->parameters.data());
    }
    if (!result) {
        state.SkipWithError("Metal benchmark did not execute an iteration");
        return;
    }
    std::size_t status_errors = 0, parameter_errors = 0, cost_errors = 0;
    constexpr unsigned p = Model<float>::parameter_count;
    for (std::size_t i = 0; i < n; ++i) {
        status_errors += result->status[i] != data.reference.status[i];
        cost_errors += !(std::abs(result->chi_squared[i] - data.reference.chi_squared[i]) <=
                         1e-6F + 2e-5F * std::abs(data.reference.chi_squared[i]));
        for (unsigned k = 0; k < p; ++k)
            if (data.reference.status[i] == batchfit::FitStatus::converged)
                parameter_errors += !(std::abs(result->parameters[i * p + k] -
                                               data.reference.parameters[i * p + k]) <=
                                      5e-4F * (1 + std::abs(data.reference.parameters[i * p + k])));
    }
    state.counters["gpu_fits_per_second"] = double(n) * double(state.iterations()) / gpu_total;
    state.counters["wall_fits_per_second"] = double(n) * double(state.iterations()) / wall_total;
    state.counters["status_errors"] = double(status_errors);
    state.counters["parameter_errors"] = double(parameter_errors);
    state.counters["cost_errors"] = double(cost_errors);
    if (status_errors || parameter_errors || cost_errors)
        state.SkipWithError("GPU/CPU parity failure (see counters)");
}
void matrix(benchmark::internal::Benchmark *benchmark) {
    for (int points : {32, 128, 512, 2048})
        benchmark->Args({100000, points});
}
BENCHMARK_TEMPLATE(metal_lm, batchfit::models::Gaussian)
    ->Name("metal_lm_gaussian")
    ->Apply(matrix)
    ->UseManualTime()
    ->Iterations(1);
BENCHMARK_TEMPLATE(metal_lm, batchfit::models::Lorentzian)
    ->Name("metal_lm_lorentzian")
    ->Apply(matrix)
    ->UseManualTime()
    ->Iterations(1);
BENCHMARK_TEMPLATE(metal_lm, batchfit::models::PseudoVoigt)
    ->Name("metal_lm_pseudo_voigt")
    ->Apply(matrix)
    ->UseManualTime()
    ->Iterations(1);
BENCHMARK_TEMPLATE(metal_lm, batchfit::models::TwoLorentzian)
    ->Name("metal_lm_two_lorentzian")
    ->Apply(matrix)
    ->UseManualTime()
    ->Iterations(1);
BENCHMARK_TEMPLATE(metal_lm, batchfit::models::Gaussian)
    ->Name("metal_lm_smoke")
    ->Args({257, 32})
    ->UseManualTime()
    ->Iterations(1);
} // namespace
