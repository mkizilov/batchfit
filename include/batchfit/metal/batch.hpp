#pragma once

#include <batchfit/cpu/solver.hpp>
#include <batchfit/metal/types.hpp>
#include <batchfit/models.hpp>

#include <atomic>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <vector>

namespace batchfit {
enum class Device { Cpu, Metal };
namespace metal {
bool available();
class PendingBatch;
enum class KernelVariant { Automatic, ThreadPerFit, SimdPerFit, ThreadgroupPerFit };
enum class MathMode { Safe, Fast };
inline KernelVariant dispatch_variant(unsigned model, std::size_t points) {
    if (model >= 4)
        throw std::invalid_argument("unknown Metal model");
    // M4, point-major float32, large-batch measurements; see DESIGN.md.
    if (points < 32 || points > 4096)
        return KernelVariant::ThreadPerFit;
    switch (model) {
    case 0:
    case 1:
        if (points < (model == 0 ? 1024U : 512U))
            return KernelVariant::ThreadPerFit;
        if (points < 2048)
            return KernelVariant::SimdPerFit;
        return KernelVariant::ThreadgroupPerFit;
    case 2:
        if (points < 1024)
            return KernelVariant::ThreadPerFit;
        if (points < 2048)
            return KernelVariant::SimdPerFit;
        if (points < 4096)
            return KernelVariant::ThreadPerFit;
        return KernelVariant::ThreadgroupPerFit;
    default:
        if (points < 512)
            return KernelVariant::ThreadPerFit;
        if (points < 4096)
            return KernelVariant::SimdPerFit;
        return KernelVariant::ThreadgroupPerFit;
    }
}
struct Timings {
    double gpu_seconds = 0, wall_seconds = 0;
    std::size_t simd_width = 0, threads_per_threadgroup = 0;
    KernelVariant variant = KernelVariant::ThreadPerFit;
    double gpu_start_seconds = 0, gpu_end_seconds = 0;
};
struct DeviceLimits {
    std::size_t max_buffer_bytes, recommended_working_set_bytes;
};
namespace detail {
struct Runtime;
struct Completion;
struct Access;
struct Storage {
    std::size_t fits, capacity, points, p;
    std::span<float> x, y, initial, parameters, chi_squared;
    std::span<FitStatus> status;
    std::span<unsigned> iterations;
    Config *config;
    std::shared_ptr<void> resources;
    std::atomic<bool> busy{false};
};
struct BusyLease {
    explicit BusyLease(Storage &value) : storage(value) {
        if (storage.busy.exchange(true))
            throw std::logic_error("Metal batch is already in flight");
    }
    ~BusyLease() {
        if (active)
            storage.busy.store(false);
    }
    BusyLease(const BusyLease &) = delete;
    BusyLease &operator=(const BusyLease &) = delete;
    Storage &storage;
    bool active = true;
};
Timings run_gpu(Storage &storage, unsigned kernel, KernelVariant variant);
PendingBatch submit_gpu(const std::shared_ptr<Storage> &storage, unsigned kernel,
                        KernelVariant variant);
template <template <typename> class Model> constexpr unsigned kernel_index() {
    if constexpr (std::is_same_v<Model<float>, models::Gaussian<float>>)
        return 0;
    else if constexpr (std::is_same_v<Model<float>, models::Lorentzian<float>>)
        return 1;
    else if constexpr (std::is_same_v<Model<float>, models::PseudoVoigt<float>>)
        return 2;
    else {
        static_assert(std::is_same_v<Model<float>, models::TwoLorentzian<float>>);
        return 3;
    }
}
} // namespace detail

class SharedBatch {
  public:
    std::size_t fits() const { return storage_->fits; }
    std::size_t points() const { return storage_->points; }
    std::size_t capacity() const { return storage_->capacity; }
    std::span<float> x() {
        ensure_idle();
        return storage_->x;
    }
    std::span<float> y() {
        ensure_idle();
        return storage_->y;
    } // point-major: y[m*N+i]
    std::span<float> initial() {
        ensure_idle();
        return storage_->initial;
    } // fit-major: initial[i*P+k]
    void resize_fits(std::size_t fits) {
        detail::BusyLease lease(*storage_);
        if (fits > storage_->capacity)
            throw std::out_of_range("Metal active fits exceed buffer capacity");
        auto &s = *storage_;
        s.fits = fits;
        s.y = {s.y.data(), fits * s.points};
        s.initial = {s.initial.data(), fits * s.p};
        s.parameters = {s.parameters.data(), fits * s.p};
        s.status = {s.status.data(), fits};
        s.iterations = {s.iterations.data(), fits};
        s.chi_squared = {s.chi_squared.data(), fits};
    }

  private:
    void ensure_idle() const {
        if (storage_->busy.load())
            throw std::logic_error("Metal input access while batch is in flight");
    }
    explicit SharedBatch(std::shared_ptr<detail::Storage> storage) : storage_(std::move(storage)) {}
    std::shared_ptr<detail::Storage> storage_;
    friend class Context;
    friend struct Result;
    friend struct detail::Access;
};

struct detail::Access {
    static Storage &get(SharedBatch &batch) { return *batch.storage_; }
    static std::shared_ptr<Storage> owner(SharedBatch &batch) { return batch.storage_; }
};

class Context {
  public:
    Context();
    explicit Context(MathMode mode);
    DeviceLimits limits() const;
    template <template <typename> class Model>
    SharedBatch make_batch(std::size_t n, std::size_t m) const {
        (void)detail::kernel_index<Model>();
        return allocate(n, m, Model<float>::parameter_count);
    }

  private:
    SharedBatch allocate(std::size_t n, std::size_t m, unsigned p) const;
    std::shared_ptr<detail::Runtime> runtime_;
};

struct Result {
    std::span<const float> parameters;
    std::span<const FitStatus> status;
    std::span<const unsigned> iterations;
    std::span<const float> chi_squared;
    Timings timings;
    Result(const SharedBatch &batch, Timings times) : Result(batch.storage_, times) {}

  private:
    Result(std::shared_ptr<detail::Storage> storage, Timings times)
        : parameters(storage->parameters), status(storage->status), iterations(storage->iterations),
          chi_squared(storage->chi_squared), timings(times), owner_(std::move(storage)) {}
    std::shared_ptr<detail::Storage> owner_;
    friend class PendingBatch;
};

class PendingBatch {
  public:
    PendingBatch(PendingBatch &&) noexcept = default;
    PendingBatch &operator=(PendingBatch &&other) noexcept;
    PendingBatch(const PendingBatch &) = delete;
    PendingBatch &operator=(const PendingBatch &) = delete;
    ~PendingBatch();
    bool ready() const;
    Result get();

  private:
    PendingBatch(std::shared_ptr<detail::Storage> owner,
                 std::shared_ptr<detail::Completion> completion)
        : owner_(std::move(owner)), completion_(std::move(completion)) {}
    void drain() noexcept;
    std::shared_ptr<detail::Storage> owner_;
    std::shared_ptr<detail::Completion> completion_;
    friend PendingBatch detail::submit_gpu(const std::shared_ptr<detail::Storage> &, unsigned,
                                           KernelVariant);
};

namespace detail {
template <template <typename> class Model>
void configure(Storage &data,
               const cpu::SolverOptions<float, Model<float>::parameter_count> &options) {
    constexpr unsigned p = Model<float>::parameter_count;
    if (data.p != p)
        throw std::invalid_argument("Metal batch/model parameter count mismatch");
    auto &config = *data.config;
    config = {};
    config.fits = static_cast<unsigned>(data.fits);
    config.points = static_cast<unsigned>(data.points);
    config.max_iterations = options.max_iterations;
    config.initial_lambda = options.initial_lambda;
    config.gradient_tolerance = options.gradient_tolerance;
    config.step_tolerance = options.step_tolerance;
    config.cost_tolerance = options.cost_tolerance;
    config.numerics = (options.compensated_sum ? 1U : 0U) | (options.scaled_cholesky ? 2U : 0U);
    if (data.fits > 0) {
        config.minimum_spacing = cpu::detail::grid_spacing<float>(data.x);
        config.varied_x = std::any_of(data.x.begin(), data.x.end(),
                                      [&](float value) { return value != data.x[0]; });
    }
    config.width_factor = std::is_same_v<Model<float>, models::Gaussian<float>>
                              ? 2.354820045030949382F
                          : std::is_same_v<Model<float>, models::PseudoVoigt<float>> ? 1.F
                                                                                     : 2.F;
    for (unsigned k = 0; k < p; ++k) {
        config.lower[k] = options.lower[k];
        config.upper[k] = options.upper[k];
    }
}
} // namespace detail
} // namespace metal

template <template <typename> class Model>
metal::PendingBatch
submit_batch(metal::SharedBatch &batch,
             const cpu::SolverOptions<float, Model<float>::parameter_count> &options = {},
             metal::KernelVariant variant = metal::KernelVariant::Automatic) {
    auto owner = metal::detail::Access::owner(batch);
    metal::detail::BusyLease lease(*owner);
    metal::detail::configure<Model>(*owner, options);
    auto pending = metal::detail::submit_gpu(owner, metal::detail::kernel_index<Model>(), variant);
    lease.active = false;
    return pending;
}

// Metal owns the shared arrays. Device::Cpu is a point-major reference path on
// the same buffers; the existing cpu::fit_batch remains the pooled fit-major API.
template <template <typename> class Model>
metal::Result
fit_batch(metal::SharedBatch &batch, Device device,
          const cpu::SolverOptions<float, Model<float>::parameter_count> &options = {},
          metal::KernelVariant variant = metal::KernelVariant::Automatic) {
    constexpr unsigned p = Model<float>::parameter_count;
    constexpr unsigned kernel = metal::detail::kernel_index<Model>();
    const auto start = std::chrono::steady_clock::now();
    auto &data = metal::detail::Access::get(batch);
    metal::detail::BusyLease lease(data);
    metal::detail::configure<Model>(data, options);
    if (device == Device::Metal) {
        auto timings = metal::detail::run_gpu(data, kernel, variant);
        timings.wall_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        return {batch, timings};
    }
    if (device != Device::Cpu)
        throw std::invalid_argument("unknown batchfit device");
    std::vector<float> row(data.points);
    for (std::size_t i = 0; i < data.fits; ++i) {
        for (std::size_t m = 0; m < data.points; ++m)
            row[m] = data.y[m * data.fits + i];
        const auto result =
            cpu::fit_single<float, Model>(data.x, row, data.initial.subspan(i * p, p), options);
        std::copy(result.parameters.begin(), result.parameters.end(),
                  data.parameters.begin() + i * p);
        data.status[i] = result.status;
        data.iterations[i] = result.iterations;
        data.chi_squared[i] = result.chi_squared;
    }
    return {
        batch,
        {0, std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(), 0, 0}};
}
} // namespace batchfit
