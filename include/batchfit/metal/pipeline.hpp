#pragma once

#include <batchfit/cpu/initial_guess.hpp>
#include <batchfit/metal/batch.hpp>
#include <batchfit/npy.hpp>

#include <functional>

namespace batchfit::metal {
struct NpyInput {
    std::filesystem::path x, y, initial;
}; // initial empty => data-only guesses
struct PipelineOptions {
    std::size_t chunk_fits = 0;
    unsigned in_flight = 3;
    KernelVariant variant = KernelVariant::Automatic;
};
struct PipelineEvent {
    std::size_t first_fit = 0, fits = 0;
    double prepare_begin = 0, prepare_end = 0, submit_begin = 0, submit_end = 0, gpu_begin = 0,
           gpu_end = 0, readback_begin = 0, readback_end = 0;
    double convert_end = 0, initial_end = 0;
    double prefetch_begin = 0, prefetch_end = 0;
};
struct PipelineStats {
    std::size_t fits = 0, chunks = 0, chunk_fits = 0, failed_guesses = 0, max_in_flight = 0,
                max_mapped_bytes = 0, shared_payload_bytes = 0;
    double wall_seconds = 0, prepare_seconds = 0, readback_seconds = 0, gpu_seconds = 0;
    double convert_seconds = 0, initial_seconds = 0, transpose_seconds = 0;
};
using ChunkConsumer = std::function<void(std::size_t first_fit, const Result &result)>;
using PipelineObserver = std::function<void(const PipelineEvent &event)>;
double pipeline_clock_seconds(); // Mach-time seconds on Metal platforms, matching GPU timestamps
std::size_t choose_chunk_fits(DeviceLimits limits, std::size_t fits, std::size_t points,
                              unsigned parameters, std::size_t input_bytes,
                              std::size_t initial_bytes, PipelineOptions options);
namespace detail {
using BatchFactory = std::function<SharedBatch(std::size_t, std::size_t)>;
using InitialGuesser =
    std::function<bool(std::span<const float>, std::span<const float>, std::span<float>)>;
using BatchSubmitter = std::function<PendingBatch(SharedBatch &)>;
PipelineStats stream_npy(Context &context, const NpyInput &input, unsigned p,
                         PipelineOptions schedule, const ChunkConsumer &consume,
                         const PipelineObserver &observe, const BatchFactory &factory,
                         const InitialGuesser &guess, const BatchSubmitter &submit);
} // namespace detail

template <template <typename> class Model>
PipelineStats fit_npy(Context &context, const NpyInput &input, const ChunkConsumer &consume,
                      const cpu::SolverOptions<float, Model<float>::parameter_count> &options = {},
                      PipelineOptions schedule = {}, const PipelineObserver &observe = {}) {
    constexpr unsigned p = Model<float>::parameter_count;
    return detail::stream_npy(
        context, input, p, schedule, consume, observe,
        [&](std::size_t n, std::size_t m) { return context.make_batch<Model>(n, m); },
        [](std::span<const float> x, std::span<const float> y, std::span<float> initial) {
            const auto result = cpu::initial_guess<float, Model>(x, y);
            std::copy(result.parameters.begin(), result.parameters.end(), initial.begin());
            return result.status == cpu::GuessStatus::success;
        },
        [&](SharedBatch &batch) {
            return batchfit::submit_batch<Model>(batch, options, schedule.variant);
        });
}
} // namespace batchfit::metal
