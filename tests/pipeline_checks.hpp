#pragma once
#include "../bench/cpu_workload.hpp"
#include "model_checks.hpp"
#include <batchfit/metal/pipeline.hpp>
#include <chrono>

namespace pipeline_checks {
using model_checks::require;
template <typename F> void rejects(F function) {
    bool caught = false;
    try {
        function();
    } catch (const std::exception &) {
        caught = true;
    }
    require(caught, "pipeline operation must reject invalid use");
}

template <template <typename> class Model>
void model(batchfit::metal::Context &context, const std::filesystem::path &directory) {
    constexpr unsigned p = Model<float>::parameter_count;
    batchfit_bench::Workload<Model> source(257, 65);
    std::vector<double> y(source.y.begin(), source.y.end()),
        initial(source.initial.begin(), source.initial.end());
    for (auto &value : y)
        value += 1e-10;
    std::transform(y.begin(), y.end(), source.y.begin(), [](double value) { return float(value); });
    {
        batchfit::npy::Writer writer(directory / "x.npy", {65});
        writer.append(source.x);
        writer.finish();
    }
    {
        batchfit::npy::Writer writer(directory / "y.npy", {257, 65}, batchfit::npy::Dtype::float64);
        writer.append(y);
        writer.finish();
    }
    {
        batchfit::npy::Writer writer(directory / "initial.npy", {257, p},
                                     batchfit::npy::Dtype::float64);
        writer.append(initial);
        writer.finish();
    }
    const batchfit::metal::NpyInput files{directory / "x.npy", directory / "y.npy",
                                          directory / "initial.npy"};
    batchfit::cpu::BatchResult<float> expected;
    {
        batchfit::cpu::ThreadPool pool(2);
        expected = source.run(pool);
    }
    for (auto variant :
         {batchfit::metal::KernelVariant::ThreadPerFit, batchfit::metal::KernelVariant::SimdPerFit,
          batchfit::metal::KernelVariant::ThreadgroupPerFit,
          batchfit::metal::KernelVariant::Automatic}) {
        for (auto depth : {1U, 3U}) {
            std::size_t next = 0, events = 0;
            const auto stats = batchfit::metal::fit_npy<Model>(
                context, files,
                [&](std::size_t first, const batchfit::metal::Result &result) {
                    require(first == next, "pipeline results remain ordered");
                    for (std::size_t i = 0; i < result.status.size(); ++i) {
                        const auto row = first + i;
                        require(result.status[i] == expected.status[row],
                                "streaming CPU status parity");
                        model_checks::near(result.chi_squared[i], expected.chi_squared[row], 1e-6,
                                           2e-5, "streaming cost parity");
                        for (unsigned k = 0; k < p; ++k)
                            model_checks::near(result.parameters[i * p + k],
                                               expected.parameters[row * p + k], 2e-4, 2e-4,
                                               "streaming parameter parity");
                    }
                    next += result.status.size();
                },
                source.options, {31, depth, variant},
                [&](const batchfit::metal::PipelineEvent &event) {
                    require(event.prepare_end >= event.prepare_begin &&
                                event.submit_begin >= event.prepare_end &&
                                event.gpu_end > event.gpu_begin &&
                                event.readback_begin >= event.gpu_end,
                            "coherent CPU/GPU event clock/order");
                    ++events;
                });
            require(next == 257 && stats.chunks == 9 && events == 9 && stats.max_in_flight <= depth,
                    "uneven chunk coverage/depth");
            require(stats.max_mapped_bytes < 31 * 65 * 8 + 32768, "mapping stays chunk-bounded");
        }
    }
    rejects([&] {
        batchfit::metal::fit_npy<Model>(
            context, files,
            [](auto, const auto &) { throw std::runtime_error("consumer failure"); },
            source.options, {17, 3});
    });
    std::size_t recovered = 0;
    batchfit::metal::fit_npy<Model>(
        context, files, [&](auto, const auto &result) { recovered += result.status.size(); },
        source.options, {32, 3});
    require(recovered == 257, "context reusable after consumer error and in-flight drain");
    std::size_t automatic = 0;
    batchfit::metal::fit_npy<Model>(
        context, {files.x, files.y, {}},
        [&](std::size_t first, const auto &result) {
            for (std::size_t i = 0; i < result.status.size(); ++i) {
                const auto row = std::span<const float>(source.y).subspan((first + i) * 65, 65);
                const auto guess = batchfit::cpu::initial_guess<float, Model>(source.x, row);
                const auto reference = batchfit::cpu::fit_single<float, Model>(
                    source.x, row, guess.parameters, source.options);
                require(result.status[i] == reference.status,
                        "automatic pipeline CPU status parity");
                if (std::isnan(reference.chi_squared)) {
                    require(std::isnan(result.chi_squared[i]),
                            "failed guess remains an invalid row");
                    continue;
                }
                model_checks::near(result.chi_squared[i], reference.chi_squared, 1e-6, 2e-5,
                                   "automatic pipeline cost parity");
                for (unsigned k = 0; k < p; ++k)
                    model_checks::near(result.parameters[i * p + k], reference.parameters[k], 2e-4,
                                       2e-4, "automatic pipeline parameter parity");
            }
            automatic += result.status.size();
        },
        source.options, {64, 3});
    require(automatic == 257, "automatic initial preparation preserves every row");
    batchfit::metal::PipelineOptions schedule{31, 3};
    schedule.uncertainties = true;
    std::size_t uncertainty_rows = 0;
    const auto stats = batchfit::metal::fit_npy<Model>(
        context, files,
        [&](std::size_t, const batchfit::metal::Result &result) {
            require(result.has_uncertainties &&
                        result.covariance.size() == result.status.size() * p * p,
                    "pipeline optional covariance/tail shape");
            for (std::size_t i = 0; i < result.status.size(); ++i) {
                batchfit::cpu::FitResult<float, p> fit;
                std::copy_n(result.parameters.data() + i * p, p, fit.parameters.begin());
                fit.status = result.status[i];
                fit.chi_squared = result.chi_squared[i];
                const auto estimate = batchfit::cpu::estimate_uncertainty<float, Model>(
                    source.x, fit, source.options);
                require(result.uncertainty_status[i] == estimate.status,
                        "streamed covariance status parity");
                if (estimate.status == batchfit::UncertaintyStatus::available)
                    for (unsigned k = 0; k < p * p; ++k)
                        model_checks::near(result.covariance[i * p * p + k], estimate.covariance[k],
                                           1e-8, 1e-3, "streamed accepted-state covariance parity");
                else
                    for (unsigned k = 0; k < p * p; ++k)
                        require(std::isnan(result.covariance[i * p * p + k]),
                                "unavailable streamed covariance is NaN");
            }
            uncertainty_rows += result.status.size();
        },
        source.options, schedule);
    require(uncertainty_rows == 257 && stats.chunks == 9,
            "pipeline uncertainty reuse retains every row");
}

inline void pending(batchfit::metal::Context &context) {
    auto batch = context.make_batch<batchfit::models::Gaussian>(33, 65);
    const float truth[3] = {2, 0, 1};
    for (unsigned m = 0; m < 65; ++m) {
        batch.x()[m] = -5 + 10 * float(m) / 64;
        for (unsigned i = 0; i < 33; ++i)
            batch.y()[m * 33 + i] = batchfit::models::Gaussian<float>::eval(batch.x()[m], truth);
    }
    for (unsigned i = 0; i < 33; ++i)
        std::copy_n(truth, 3, batch.initial().begin() + i * 3);
    auto first = batchfit::submit_batch<batchfit::models::Gaussian>(batch);
    rejects([&] { batch.resize_fits(3); });
    rejects([&] { batch.y(); });
    rejects([&] { batchfit::submit_batch<batchfit::models::Gaussian>(batch); });
    require(first.get().status[0] == batchfit::FitStatus::converged, "async get");
    auto second = batchfit::submit_batch<batchfit::models::Gaussian>(batch);
    first.get(); // A consumed older handle must not clear a newer submission's lease.
    rejects([&] { batch.resize_fits(3); });
    second.get();
    batch.resize_fits(0);
    require(batchfit::submit_batch<batchfit::models::Gaussian>(batch).get().status.empty(),
            "empty async batch");
    batch.resize_fits(33);
    require(batch.y().size() == 33 * 65, "capacity-preserving resize grows again");
    { auto dropped = batchfit::submit_batch<batchfit::models::Gaussian>(batch); }
    batch.resize_fits(1);
    rejects([&] { batch.resize_fits(34); });
}
inline void all() {
    using namespace batchfit::metal;
    PipelineOptions covariance_limits{2, 3};
    covariance_limits.uncertainties = true;
    rejects([&] { choose_chunk_fits({256, 64U << 20}, 100, 1, 8, 4, 4, covariance_limits); });
    require(choose_chunk_fits({8192, 64U << 20}, 100, 512, 8, 8, 4, {4, 3}) == 4,
            "explicit chunk respects maxBufferLength");
    rejects([] { choose_chunk_fits({8192, 64U << 20}, 100, 512, 8, 8, 4, {5, 3}); });
    rejects([] { choose_chunk_fits({8192, 64U << 20}, 100, 512, 8, 8, 4, {1, 4}); });
    const auto directory =
        std::filesystem::path(".agent-local") /
        ("pipeline-test-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    require(std::filesystem::create_directory(directory), "create owned pipeline fixture");
    try {
        Context context;
        pending(context);
        model<batchfit::models::Gaussian>(context, directory);
        model<batchfit::models::Lorentzian>(context, directory);
        model<batchfit::models::PseudoVoigt>(context, directory);
        model<batchfit::models::TwoLorentzian>(context, directory);
        std::filesystem::remove_all(directory);
    } catch (...) {
        std::filesystem::remove_all(directory);
        throw;
    }
}
} // namespace pipeline_checks
