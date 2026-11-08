#pragma once

#include "../bench/cpu_workload.hpp"
#include "model_checks.hpp"
#include <batchfit/batchfit.hpp>

namespace metal_lm_checks {
using model_checks::near;
using model_checks::require;

template <template <typename> class Model> void compare(batchfit::metal::Context &context) {
    constexpr unsigned p = Model<float>::parameter_count;
    for (unsigned n : {1, 31, 32, 33, 257}) {
        for (unsigned m : {32, 129}) {
            batchfit_bench::Workload<Model> data(n, m);
            auto shared = context.make_batch<Model>(n, m);
            std::copy(data.x.begin(), data.x.end(), shared.x().begin());
            std::copy(data.initial.begin(), data.initial.end(), shared.initial().begin());
            for (unsigned j = 0; j < m; ++j)
                for (unsigned i = 0; i < n; ++i)
                    shared.y()[j * n + i] = data.y[i * m + j];
            const auto *x_pointer = shared.x().data();
            const auto *y_pointer = shared.y().data();
            const auto result =
                batchfit::fit_batch<Model>(shared, batchfit::Device::Metal, data.options);
            require(result.timings.gpu_seconds > 0 && result.timings.wall_seconds > 0,
                    "GPU and wall timestamps");
            require(result.timings.simd_width > 0 &&
                        result.timings.threads_per_threadgroup % result.timings.simd_width == 0,
                    "pipeline-derived threadgroup geometry");
            require(shared.x().data() == x_pointer && shared.y().data() == y_pointer,
                    "shared buffers must not be replaced or staged");
            for (unsigned i = 0; i < n; ++i) {
                const auto cpu = batchfit::cpu::fit_single<float, Model>(
                    data.x, std::span<const float>(data.y).subspan(i * m, m),
                    std::span<const float>(data.initial).subspan(i * p, p), data.options);
                const auto label = std::string("kernel=") +
                                   std::to_string(batchfit::metal::detail::kernel_index<Model>()) +
                                   " N=" + std::to_string(n) + " M=" + std::to_string(m) +
                                   " fit=" + std::to_string(i);
                require(result.status[i] == cpu.status, label + " status");
                for (unsigned k = 0; k < p; ++k)
                    near(result.parameters[i * p + k], cpu.parameters[k], 2e-4, 2e-4,
                         label + " parameter=" + std::to_string(k));
                near(result.chi_squared[i], cpu.chi_squared, 1e-6, 2e-5, label + " chi squared");
                require(result.iterations[i] <= data.options.max_iterations,
                        label + " iteration budget");
            }
            const std::vector<float> saved(result.parameters.begin(), result.parameters.end());
            const auto point_major_cpu =
                batchfit::fit_batch<Model>(shared, batchfit::Device::Cpu, data.options);
            for (std::size_t k = 0; k < saved.size(); ++k)
                near(saved[k], point_major_cpu.parameters[k], 2e-4, 2e-4,
                     "same-buffer CPU reference path");
        }
    }
}

inline void failures(batchfit::metal::Context &context) {
    auto batch = context.make_batch<batchfit::models::Lorentzian>(3, 3);
    std::fill(batch.x().begin(), batch.x().end(), 0);
    std::fill(batch.y().begin(), batch.y().end(), 2);
    const std::array<float, 9> initial{1, 0, 1, 2, 0, 1, 1, 0, 0};
    std::copy(initial.begin(), initial.end(), batch.initial().begin());
    const auto result =
        batchfit::fit_batch<batchfit::models::Lorentzian>(batch, batchfit::Device::Metal);
    require(result.status[0] == batchfit::FitStatus::singular_matrix &&
                result.status[1] == batchfit::FitStatus::converged &&
                result.status[2] == batchfit::FitStatus::invalid_input,
            "GPU singular/converged/invalid isolation");
    require(result.iterations[0] == 1 && result.iterations[1] == 0 && result.iterations[2] == 0,
            "failure iteration counts");
    batchfit::cpu::SolverOptions<float, 3> options;
    options.max_iterations = 0;
    auto zero =
        batchfit::fit_batch<batchfit::models::Lorentzian>(batch, batchfit::Device::Metal, options);
    require(zero.status[0] == batchfit::FitStatus::max_iterations && zero.chi_squared[0] == 3,
            "GPU zero budget");
    options = {};
    options.lower[1] = options.upper[1] = 0;
    options.lower[2] = options.upper[2] = 1;
    auto fixed =
        batchfit::fit_batch<batchfit::models::Lorentzian>(batch, batchfit::Device::Metal, options);
    require(fixed.status[0] == batchfit::FitStatus::converged && fixed.parameters[1] == 0 &&
                fixed.parameters[2] == 1,
            "GPU fixed coordinates");
    options.upper[0] = 1.5F;
    const auto clamped =
        batchfit::fit_batch<batchfit::models::Lorentzian>(batch, batchfit::Device::Metal, options);
    for (unsigned i = 0; i < 3; ++i)
        require(clamped.status[i] == batchfit::FitStatus::converged &&
                    clamped.parameters[i * 3] == 1.5F,
                "GPU active upper bound and projected initial width");
    options.initial_lambda = 0;
    const auto bad_options =
        batchfit::fit_batch<batchfit::models::Lorentzian>(batch, batchfit::Device::Metal, options);
    for (auto status : bad_options.status)
        require(status == batchfit::FitStatus::invalid_input, "GPU invalid options");
    batch.x()[0] = std::numeric_limits<float>::quiet_NaN();
    auto invalid =
        batchfit::fit_batch<batchfit::models::Lorentzian>(batch, batchfit::Device::Metal);
    for (auto status : invalid.status)
        require(status == batchfit::FitStatus::invalid_input, "shared bad x");
    auto empty = context.make_batch<batchfit::models::Gaussian>(0, 3);
    auto no_fits = batchfit::fit_batch<batchfit::models::Gaussian>(empty, batchfit::Device::Metal);
    require(no_fits.parameters.empty() && no_fits.timings.gpu_seconds == 0, "empty GPU batch");
    bool rejected = false;
    try {
        context.make_batch<batchfit::models::Gaussian>(1, 0);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    require(rejected, "zero M rejected before allocation");
    rejected = false;
    try {
        context.make_batch<batchfit::models::Gaussian>(std::numeric_limits<std::size_t>::max(), 3);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    require(rejected, "GPU uint32 index overflow rejected");
    rejected = false;
    try {
        batchfit::fit_batch<batchfit::models::PseudoVoigt>(batch, batchfit::Device::Metal);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    require(rejected, "model/buffer shape mismatch rejected");
}

inline void all() {
    batchfit::metal::Context context;
    compare<batchfit::models::Gaussian>(context);
    compare<batchfit::models::Lorentzian>(context);
    compare<batchfit::models::PseudoVoigt>(context);
    compare<batchfit::models::TwoLorentzian>(context);
    failures(context);
    const auto retained = [] {
        batchfit::metal::Context temporary;
        auto batch = temporary.make_batch<batchfit::models::Gaussian>(1, 3);
        const float parameters[3] = {2, 0, 1};
        for (unsigned j = 0; j < 3; ++j) {
            batch.x()[j] = float(j) - 1;
            batch.y()[j] = batchfit::models::Gaussian<float>::eval(batch.x()[j], parameters);
            batch.initial()[j] = parameters[j];
        }
        return batchfit::fit_batch<batchfit::models::Gaussian>(batch, batchfit::Device::Metal);
    }();
    require(retained.status[0] == batchfit::FitStatus::converged && retained.parameters[0] == 2,
            "result retains buffers and runtime after context/batch destruction");
}
} // namespace metal_lm_checks
