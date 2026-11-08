#pragma once

#include <batchfit/cpu/solver.hpp>
#include <batchfit/metal/types.hpp>
#include <batchfit/models.hpp>

#include <chrono>
#include <memory>
#include <stdexcept>
#include <vector>

namespace batchfit {
enum class Device { Cpu, Metal };
namespace metal {
struct Timings {
    double gpu_seconds = 0, wall_seconds = 0;
    std::size_t simd_width = 0, threads_per_threadgroup = 0;
};
namespace detail {
struct Runtime;
struct Access;
struct Storage {
    std::size_t fits, points, p;
    std::span<float> x, y, initial, parameters, chi_squared;
    std::span<FitStatus> status;
    std::span<unsigned> iterations;
    Config *config;
    std::shared_ptr<void> resources;
};
Timings run_gpu(Storage &storage, unsigned kernel);
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
    std::span<float> x() { return storage_->x; }
    std::span<float> y() { return storage_->y; }             // point-major: y[m*N+i]
    std::span<float> initial() { return storage_->initial; } // fit-major: initial[i*P+k]
  private:
    explicit SharedBatch(std::shared_ptr<detail::Storage> storage) : storage_(std::move(storage)) {}
    std::shared_ptr<detail::Storage> storage_;
    friend class Context;
    friend struct Result;
    friend struct detail::Access;
};

struct detail::Access {
    static Storage &get(SharedBatch &batch) { return *batch.storage_; }
};

class Context {
  public:
    Context();
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
    Result(const SharedBatch &batch, Timings times)
        : parameters(batch.storage_->parameters), status(batch.storage_->status),
          iterations(batch.storage_->iterations), chi_squared(batch.storage_->chi_squared),
          timings(times), owner_(batch.storage_) {}

  private:
    std::shared_ptr<detail::Storage> owner_;
};
} // namespace metal

// Metal owns the shared arrays. Device::Cpu is a point-major reference path on
// the same buffers; the existing cpu::fit_batch remains the pooled fit-major API.
template <template <typename> class Model>
metal::Result
fit_batch(metal::SharedBatch &batch, Device device,
          const cpu::SolverOptions<float, Model<float>::parameter_count> &options = {}) {
    constexpr unsigned p = Model<float>::parameter_count;
    constexpr unsigned kernel = metal::detail::kernel_index<Model>();
    const auto start = std::chrono::steady_clock::now();
    auto &data = metal::detail::Access::get(batch);
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
    for (unsigned k = 0; k < p; ++k) {
        config.lower[k] = options.lower[k];
        config.upper[k] = options.upper[k];
    }
    if (device == Device::Metal) {
        auto timings = metal::detail::run_gpu(data, kernel);
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
