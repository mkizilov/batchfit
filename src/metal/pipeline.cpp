#include <batchfit/metal/pipeline.hpp>

#include <condition_variable>
#include <deque>
#include <limits>
#include <mutex>
#include <optional>
#include <thread>
#if defined(__APPLE__)
#include <mach/mach_time.h>
#endif

namespace batchfit::metal {
double pipeline_clock_seconds() {
#if defined(__APPLE__)
    static const auto scale = [] {
        mach_timebase_info_data_t info{};
        if (mach_timebase_info(&info) != KERN_SUCCESS)
            throw std::runtime_error("Mach timebase unavailable");
        return double(info.numer) / double(info.denom) * 1e-9;
    }();
    return double(mach_absolute_time()) * scale;
#else
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
#endif
}
std::size_t choose_chunk_fits(DeviceLimits limits, std::size_t fits, std::size_t points, unsigned p,
                              std::size_t input_bytes, std::size_t initial_bytes,
                              PipelineOptions options) {
    if (points == 0 || p == 0 || p > 8 || options.in_flight == 0 || options.in_flight > 3 ||
        (input_bytes != 4 && input_bytes != 8) ||
        (initial_bytes != 0 && initial_bytes != 4 && initial_bytes != 8))
        throw std::invalid_argument("invalid pipeline dimensions/dtype/depth");
    if (points > limits.max_buffer_bytes / 4 || sizeof(detail::Config) > limits.max_buffer_bytes)
        throw std::invalid_argument("pipeline x/config exceeds maxBufferLength");
    const auto index_limit = std::numeric_limits<std::uint32_t>::max();
    const auto maximum =
        std::min({limits.max_buffer_bytes / 4 / points, limits.max_buffer_bytes / 4 / p,
                  std::size_t(index_limit) / points, std::size_t(index_limit) / p, fits});
    if (fits == 0)
        return 0;
    if (maximum == 0)
        throw std::invalid_argument("one fit exceeds Metal buffer/index limits");
    if (options.chunk_fits) {
        const auto requested = std::min(options.chunk_fits, fits);
        if (requested > maximum)
            throw std::invalid_argument("requested chunk exceeds maxBufferLength/index limit");
        return requested;
    }
    const auto budget =
        limits.recommended_working_set_bytes
            ? std::min(std::size_t(256) << 20, limits.recommended_working_set_bytes / 8)
            : std::size_t(64) << 20;
    const auto fixed = options.in_flight * (points * 4 + sizeof(detail::Config));
    const auto per_fit = options.in_flight * (points * 4 + std::size_t(p) * 8 + 12) +
                         points * (4 + input_bytes) + p * initial_bytes;
    if (budget <= fixed || per_fit > budget - fixed)
        throw std::invalid_argument("device working-set budget cannot hold one pipeline fit");
    auto chunk = std::min(maximum, (budget - fixed) / per_fit);
    if (chunk >= 256)
        chunk = chunk / 256 * 256;
    return std::max(std::size_t(1), chunk);
}

PipelineStats detail::stream_npy(Context &context, const NpyInput &input, unsigned p,
                                 PipelineOptions schedule, const ChunkConsumer &consume,
                                 const PipelineObserver &observe, const BatchFactory &factory,
                                 const InitialGuesser &guess, const BatchSubmitter &submit) {
    const auto start = pipeline_clock_seconds();
    if (!consume)
        throw std::invalid_argument("pipeline requires a result consumer");
    npy::MappedReader x_file(input.x), y_file(input.y);
    const auto &shape = y_file.header().shape;
    if (shape.size() != 2 || shape[1] == 0 ||
        x_file.header().shape != std::vector<std::size_t>{shape[1]})
        throw std::invalid_argument("pipeline requires x[M], y[N,M] NPY arrays");
    const auto n = shape[0], m = shape[1];
    std::unique_ptr<npy::MappedReader> initial_file;
    if (!input.initial.empty()) {
        initial_file = std::make_unique<npy::MappedReader>(input.initial);
        if (initial_file->header().shape != std::vector<std::size_t>{n, p})
            throw std::invalid_argument("pipeline initial NPY must be [N,P]");
    }
    PipelineStats stats;
    stats.fits = n;
    stats.chunk_fits =
        choose_chunk_fits(context.limits(), n, m, p, y_file.header().item_bytes,
                          initial_file ? initial_file->header().item_bytes : 0, schedule);
    if (n == 0) {
        stats.wall_seconds = pipeline_clock_seconds() - start;
        return stats;
    }
    const auto capacity = stats.chunk_fits;
    struct Slot {
        explicit Slot(SharedBatch value) : batch(std::move(value)) {}
        SharedBatch batch;
        std::optional<PendingBatch> pending;
        PipelineEvent event;
        bool free = true;
    };
    std::vector<Slot> slots;
    slots.reserve(schedule.in_flight);
    std::vector<float> grid(m), staging(capacity * m);
    x_file.window(0, m).copy_to(grid);
    for (unsigned i = 0; i < schedule.in_flight; ++i) {
        slots.emplace_back(factory(capacity, m));
        std::copy(grid.begin(), grid.end(), slots.back().batch.x().begin());
    }
    stats.shared_payload_bytes =
        schedule.in_flight *
        (capacity * (m * 4 + std::size_t(p) * 8 + 12) + m * 4 + sizeof(Config));
    std::mutex mutex;
    std::condition_variable changed;
    std::deque<std::size_t> queued;
    std::exception_ptr failure;
    bool producer_done = false;
    const auto fail = [&](std::exception_ptr error) {
        std::lock_guard lock(mutex);
        if (!failure)
            failure = std::move(error);
        changed.notify_all();
    };
    std::jthread reader([&] {
        for (;;) {
            std::size_t index;
            {
                std::unique_lock lock(mutex);
                changed.wait(lock, [&] { return !queued.empty() || producer_done; });
                if (queued.empty())
                    return;
                index = queued.front();
                queued.pop_front();
            }
            auto &slot = slots[index];
            try {
                const auto result = slot.pending->get();
                slot.event.gpu_begin = result.timings.gpu_start_seconds;
                slot.event.gpu_end = result.timings.gpu_end_seconds;
                slot.event.readback_begin = pipeline_clock_seconds();
                bool cancelled;
                {
                    std::lock_guard lock(mutex);
                    cancelled = bool(failure);
                }
                if (!cancelled)
                    consume(slot.event.first_fit, result);
                slot.event.readback_end = pipeline_clock_seconds();
                stats.readback_seconds += slot.event.readback_end - slot.event.readback_begin;
                stats.gpu_seconds += result.timings.gpu_seconds;
                if (observe)
                    observe(slot.event);
            } catch (...) {
                fail(std::current_exception());
            }
            {
                std::lock_guard lock(mutex);
                slot.pending.reset();
                slot.free = true;
                changed.notify_all();
            }
        }
    });
    std::size_t current = 0;
    double prefetch_begin = 0, prefetch_end = 0;
    try {
        prefetch_begin = pipeline_clock_seconds();
        y_file.prefetch(0, std::min(capacity, n) * m);
        prefetch_end = pipeline_clock_seconds();
        for (std::size_t first = 0; first < n;) {
            current = stats.chunks % schedule.in_flight;
            auto &slot = slots[current];
            {
                std::unique_lock lock(mutex);
                changed.wait(lock, [&] { return slot.free || failure; });
                if (failure)
                    break;
                slot.free = false;
            }
            const auto count = std::min(capacity, n - first);
            slot.event = {};
            slot.event.first_fit = first;
            slot.event.fits = count;
            slot.event.prefetch_begin = prefetch_begin;
            slot.event.prefetch_end = prefetch_end;
            slot.event.prepare_begin = pipeline_clock_seconds();
            slot.batch.resize_fits(count);
            {
                auto mapping = y_file.window(first * m, count * m);
                stats.max_mapped_bytes = std::max(stats.max_mapped_bytes, mapping.mapped_bytes());
                mapping.copy_to(std::span(staging).first(count * m));
            }
            auto initial = slot.batch.initial();
            slot.event.convert_end = pipeline_clock_seconds();
            stats.convert_seconds += slot.event.convert_end - slot.event.prepare_begin;
            if (initial_file) {
                auto mapping = initial_file->window(first * p, count * p);
                stats.max_mapped_bytes = std::max(stats.max_mapped_bytes, mapping.mapped_bytes());
                mapping.copy_to(initial);
            } else
                for (std::size_t i = 0; i < count; ++i)
                    if (!guess(grid, std::span<const float>(staging).subspan(i * m, m),
                               initial.subspan(i * p, p)))
                        ++stats.failed_guesses;
            auto y = slot.batch.y();
            slot.event.initial_end = pipeline_clock_seconds();
            stats.initial_seconds += slot.event.initial_end - slot.event.convert_end;
            // Source-contiguous 16x16 tiles avoid power-of-two row-stride cache
            // conflicts while retaining a small working set of destination lines.
            for (std::size_t i0 = 0; i0 < count; i0 += 16)
                for (std::size_t j0 = 0; j0 < m; j0 += 16)
                    for (auto i = i0; i < std::min(count, i0 + 16); ++i)
                        for (auto j = j0; j < std::min(m, j0 + 16); ++j)
                            y[j * count + i] = staging[i * m + j];
            slot.event.prepare_end = pipeline_clock_seconds();
            stats.transpose_seconds += slot.event.prepare_end - slot.event.initial_end;
            stats.prepare_seconds += slot.event.prepare_end - slot.event.prepare_begin;
            {
                std::lock_guard lock(mutex);
                if (failure) {
                    slot.free = true;
                    break;
                }
            }
            slot.event.submit_begin = pipeline_clock_seconds();
            slot.pending.emplace(submit(slot.batch));
            slot.event.submit_end = pipeline_clock_seconds();
            {
                std::lock_guard lock(mutex);
                queued.push_back(current);
                std::size_t active = 0;
                for (const auto &candidate : slots)
                    if (candidate.pending && !candidate.pending->ready())
                        ++active;
                stats.max_in_flight = std::max(stats.max_in_flight, active);
            }
            changed.notify_all();
            ++stats.chunks;
            first += count;
            if (first < n) {
                prefetch_begin = pipeline_clock_seconds();
                y_file.prefetch(first * m, std::min(capacity, n - first) * m);
                prefetch_end = pipeline_clock_seconds();
            }
        }
    } catch (...) {
        fail(std::current_exception());
    }
    {
        std::lock_guard lock(mutex);
        producer_done = true;
        changed.notify_all();
    }
    reader.join(); // callbacks/errors drain before mappings, slots or consumer references die
    if (failure)
        std::rethrow_exception(failure);
    stats.wall_seconds = pipeline_clock_seconds() - start;
    return stats;
}
} // namespace batchfit::metal
