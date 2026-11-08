#pragma once

#include "cpu_workload.hpp"
#include <batchfit/metal/batch.hpp>

namespace batchfit_bench {
template <template <typename> class Model> struct MetalWorkload {
    static constexpr unsigned p = Model<float>::parameter_count;
    batchfit::metal::SharedBatch shared;
    cpu::BatchResult<float> reference;
    cpu::SolverOptions<float, p> options;
    std::size_t fallback_guesses = 0;

    MetalWorkload(batchfit::metal::Context &context, std::size_t n, std::size_t m)
        : shared(context.make_batch<Model>(n, m)) {
        Workload<Model> source(n, m);
        options = source.options;
        fallback_guesses = source.fallback_guesses;
        cpu::ThreadPool pool(2);
        reference = source.run(pool);
        std::copy(source.x.begin(), source.x.end(), shared.x().begin());
        std::copy(source.initial.begin(), source.initial.end(), shared.initial().begin());
        for (std::size_t j = 0; j < m; ++j)
            for (std::size_t i = 0; i < n; ++i)
                shared.y()[j * n + i] = source.y[i * m + j];
    }
    void
    warmup(batchfit::metal::Context &context,
           batchfit::metal::KernelVariant variant = batchfit::metal::KernelVariant::ThreadPerFit) {
        const auto n = shared.fits(), m = shared.points(), count = std::min(n, std::size_t(256));
        auto warm = context.make_batch<Model>(count, m);
        std::copy(shared.x().begin(), shared.x().end(), warm.x().begin());
        std::copy_n(shared.initial().begin(), count * p, warm.initial().begin());
        for (std::size_t j = 0; j < m; ++j)
            for (std::size_t i = 0; i < count; ++i)
                warm.y()[j * count + i] = shared.y()[j * n + i];
        batchfit::fit_batch<Model>(warm, batchfit::Device::Metal, options, variant);
    }
    batchfit::metal::Result
    run(batchfit::metal::KernelVariant variant = batchfit::metal::KernelVariant::ThreadPerFit) {
        return batchfit::fit_batch<Model>(shared, batchfit::Device::Metal, options, variant);
    }
};
} // namespace batchfit_bench
