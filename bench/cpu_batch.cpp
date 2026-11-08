#include "cpu_workload.hpp"

#include <benchmark/benchmark.h>

namespace {
template <template <typename> class Model> void cpu_batch(benchmark::State &state) {
    batchfit_bench::Workload<Model> data(std::size_t(state.range(0)), std::size_t(state.range(1)));
    const auto workers =
        state.range(2) == 0 ? batchfit_bench::all_cores() : unsigned(state.range(2));
    batchfit::cpu::ThreadPool pool(workers);
    data.warmup(pool);
    batchfit::cpu::BatchResult<float> result;
    for (auto _ : state) {
        result = data.run(pool);
        benchmark::DoNotOptimize(result.parameters.data());
        benchmark::ClobberMemory();
    }
    const auto summary = batchfit_bench::summarize(result);
    state.SetItemsProcessed(state.iterations() * std::int64_t(data.fits));
    state.counters["fits_per_second"] = benchmark::Counter(
        double(state.iterations()) * double(data.fits), benchmark::Counter::kIsRate);
    state.counters["workers"] = workers;
    state.counters["converged"] = double(summary.converged);
    state.counters["fallbacks"] = double(data.fallback_guesses);
    state.counters["mean_iterations"] = summary.mean_iterations;
}

void matrix(benchmark::internal::Benchmark *benchmark) {
    for (int points : {32, 128, 512, 2048})
        for (int workers : {1, 0})
            benchmark->Args({100000, points, workers});
}
BENCHMARK_TEMPLATE(cpu_batch, batchfit::models::Gaussian)
    ->Name("cpu_batch_gaussian")
    ->Apply(matrix)
    ->UseRealTime()
    ->Iterations(1);
BENCHMARK_TEMPLATE(cpu_batch, batchfit::models::Lorentzian)
    ->Name("cpu_batch_lorentzian")
    ->Apply(matrix)
    ->UseRealTime()
    ->Iterations(1);
BENCHMARK_TEMPLATE(cpu_batch, batchfit::models::PseudoVoigt)
    ->Name("cpu_batch_pseudo_voigt")
    ->Apply(matrix)
    ->UseRealTime()
    ->Iterations(1);
BENCHMARK_TEMPLATE(cpu_batch, batchfit::models::TwoLorentzian)
    ->Name("cpu_batch_two_lorentzian")
    ->Apply(matrix)
    ->UseRealTime()
    ->Iterations(1);
BENCHMARK_TEMPLATE(cpu_batch, batchfit::models::Gaussian)
    ->Name("cpu_batch_smoke")
    ->Args({128, 32, 2})
    ->UseRealTime()
    ->Iterations(1);
} // namespace
