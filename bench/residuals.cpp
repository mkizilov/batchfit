#include "residuals.hpp"

#include <benchmark/benchmark.h>

static void cpu_residuals(benchmark::State &state) {
    const std::vector<float> predicted(state.range(0), 2.0F);
    const std::vector<float> observed(state.range(0), 1.0F);
    for (auto _ : state) {
        auto result = batchfit::detail::squared_residuals_cpu(predicted, observed);
        benchmark::DoNotOptimize(result.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * state.range(0));
}
BENCHMARK(cpu_residuals)->Arg(1024)->Arg(65536);
