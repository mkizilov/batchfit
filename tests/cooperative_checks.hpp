#pragma once
#include "metal_lm_checks.hpp"

namespace cooperative_checks {
using model_checks::near;
using model_checks::require;
using Variant = batchfit::metal::KernelVariant;

template <template <typename> class Model> void compare(batchfit::metal::Context &context) {
    constexpr unsigned p = Model<float>::parameter_count;
    for (unsigned n : {1, 31, 33, 257})
        for (unsigned m : {32, 128, 512, 4096}) {
            batchfit_bench::Workload<Model> source(n, m);
            auto batch = context.make_batch<Model>(n, m);
            std::copy(source.x.begin(), source.x.end(), batch.x().begin());
            std::copy(source.initial.begin(), source.initial.end(), batch.initial().begin());
            for (unsigned j = 0; j < m; ++j)
                for (unsigned i = 0; i < n; ++i)
                    batch.y()[j * n + i] = source.y[i * m + j];
            std::vector<batchfit::cpu::FitResult<float, p>> reference;
            for (unsigned i = 0; i < n; ++i)
                reference.push_back(batchfit::cpu::fit_single<float, Model>(
                    source.x, std::span<const float>(source.y).subspan(i * m, m),
                    std::span<const float>(source.initial).subspan(i * p, p), source.options));
            for (auto variant : {Variant::ThreadPerFit, Variant::SimdPerFit,
                                 Variant::ThreadgroupPerFit, Variant::Automatic}) {
                const auto result = batchfit::fit_batch<Model>(batch, batchfit::Device::Metal,
                                                               source.options, variant);
                const auto selected = variant == Variant::Automatic
                                          ? batchfit::metal::dispatch_variant(
                                                batchfit::metal::detail::kernel_index<Model>(), m)
                                          : variant;
                require(result.timings.variant == selected, "forced/automatic dispatch");
                for (unsigned i = 0; i < n; ++i) {
                    const auto label =
                        std::string("variant=") + std::to_string(unsigned(variant)) +
                        " model=" + std::to_string(batchfit::metal::detail::kernel_index<Model>()) +
                        " N=" + std::to_string(n) + " M=" + std::to_string(m) +
                        " fit=" + std::to_string(i);
                    require(result.status[i] == reference[i].status, label + " status");
                    for (unsigned k = 0; k < p; ++k)
                        near(result.parameters[i * p + k], reference[i].parameters[k], 2e-4, 2e-4,
                             label + " parameter=" + std::to_string(k));
                    near(result.chi_squared[i], reference[i].chi_squared, 1e-6, 2e-5,
                         label + " cost");
                    require(result.iterations[i] <= source.options.max_iterations,
                            label + " budget");
                }
            }
        }
}
inline void failures(batchfit::metal::Context &context) {
    auto batch = context.make_batch<batchfit::models::Lorentzian>(3, 3);
    std::fill(batch.x().begin(), batch.x().end(), 0);
    std::fill(batch.y().begin(), batch.y().end(), 2);
    const std::array<float, 9> initial{1, 0, 1, 2, 0, 1, 1, 0, 0};
    std::copy(initial.begin(), initial.end(), batch.initial().begin());
    for (auto variant : {Variant::SimdPerFit, Variant::ThreadgroupPerFit}) {
        batch.y()[4] = std::numeric_limits<float>::quiet_NaN();
        const auto bad_row = batchfit::fit_batch<batchfit::models::Lorentzian>(
            batch, batchfit::Device::Metal, {}, variant);
        require(bad_row.status[0] == batchfit::FitStatus::singular_matrix &&
                    bad_row.status[1] == batchfit::FitStatus::invalid_input,
                "collective nonfinite observation is isolated to its fit");
        batch.y()[4] = 2;
        batch.x()[2] = std::numeric_limits<float>::quiet_NaN();
        const auto bad_x = batchfit::fit_batch<batchfit::models::Lorentzian>(
            batch, batchfit::Device::Metal, {}, variant);
        for (auto status : bad_x.status)
            require(status == batchfit::FitStatus::invalid_input, "collective shared nonfinite x");
        batch.x()[2] = 0;
        const auto result = batchfit::fit_batch<batchfit::models::Lorentzian>(
            batch, batchfit::Device::Metal, {}, variant);
        require(result.status[0] == batchfit::FitStatus::singular_matrix &&
                    result.status[1] == batchfit::FitStatus::converged &&
                    result.status[2] == batchfit::FitStatus::invalid_input,
                "uniform isolated failures");
        require(result.iterations[0] == 1 && result.iterations[1] == 0 && result.iterations[2] == 0,
                "failure iterations");
        batchfit::cpu::SolverOptions<float, 3> options;
        options.max_iterations = 0;
        const auto zero = batchfit::fit_batch<batchfit::models::Lorentzian>(
            batch, batchfit::Device::Metal, options, variant);
        require(zero.status[0] == batchfit::FitStatus::max_iterations && zero.chi_squared[0] == 3,
                "zero budget");
        options = {};
        options.lower = {0, 0, 1};
        options.upper = {1.5F, 0, 1};
        const auto fixed = batchfit::fit_batch<batchfit::models::Lorentzian>(
            batch, batchfit::Device::Metal, options, variant);
        for (unsigned i = 0; i < 3; ++i)
            require(fixed.status[i] == batchfit::FitStatus::converged &&
                        fixed.parameters[i * 3] == 1.5F,
                    "fixed/active coordinates");
        options.initial_lambda = 0;
        const auto invalid = batchfit::fit_batch<batchfit::models::Lorentzian>(
            batch, batchfit::Device::Metal, options, variant);
        for (auto status : invalid.status)
            require(status == batchfit::FitStatus::invalid_input, "invalid options");
        auto empty = context.make_batch<batchfit::models::Gaussian>(0, 32);
        require(batchfit::fit_batch<batchfit::models::Gaussian>(empty, batchfit::Device::Metal, {},
                                                                variant)
                    .status.empty(),
                "empty collective batch");
    }
}
inline void all() {
    const unsigned sizes[8] = {32, 64, 128, 256, 512, 1024, 2048, 4096};
    const unsigned expected[4][8] = {{1, 1, 1, 1, 1, 2, 3, 3},
                                     {1, 1, 1, 1, 2, 2, 3, 3},
                                     {1, 1, 1, 1, 1, 2, 1, 3},
                                     {1, 1, 1, 1, 2, 2, 2, 3}};
    for (unsigned model = 0; model < 4; ++model)
        for (unsigned i = 0; i < 8; ++i)
            require(unsigned(batchfit::metal::dispatch_variant(model, sizes[i])) ==
                        expected[model][i],
                    "measured dispatch policy table");
    for (unsigned model = 0; model < 4; ++model)
        require(batchfit::metal::dispatch_variant(model, 4097) == Variant::ThreadPerFit,
                "unmeasured M uses conservative v1");
    batchfit::metal::Context context;
    compare<batchfit::models::Gaussian>(context);
    compare<batchfit::models::Lorentzian>(context);
    compare<batchfit::models::PseudoVoigt>(context);
    compare<batchfit::models::TwoLorentzian>(context);
    failures(context);
}
} // namespace cooperative_checks
