#include "cpu_workload.hpp"
#include <batchfit/metal/batch.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <chrono>
#include <memory>
#include <optional>

namespace py = pybind11;
namespace {
using Clock = std::chrono::steady_clock;
using Variant = batchfit::metal::KernelVariant;
using FloatArray = py::array_t<float, py::array::c_style>;
using IntArray = py::array_t<std::int32_t, py::array::c_style>;
double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
template <typename T>
py::array_t<T> copy(std::span<const T> data, const std::vector<py::ssize_t> &shape) {
    py::array_t<T> result(shape);
    std::copy(data.begin(), data.end(), result.mutable_data());
    return result;
}
template <template <typename> class Model> py::dict inputs(std::size_t m, std::size_t patterns) {
    const auto start = Clock::now();
    batchfit_bench::Workload<Model> data(patterns, m);
    batchfit::cpu::ThreadPool pool(2);
    const auto reference = data.run(pool);
    constexpr unsigned p = Model<float>::parameter_count;
    const auto k = static_cast<py::ssize_t>(patterns), points = static_cast<py::ssize_t>(m);
    py::dict result;
    result["x"] = copy<float>(data.x, {points});
    result["y"] = copy<float>(data.y, {k, points});
    result["p0"] = copy<float>(data.initial, {k, p});
    result["lower"] = copy<float>(data.options.lower, {p});
    result["upper"] = copy<float>(data.options.upper, {p});
    result["parameters"] = copy<float>(reference.parameters, {k, p});
    result["chi2"] = copy<float>(reference.chi_squared, {k});
    py::array_t<std::int32_t> status(k);
    for (std::size_t i = 0; i < patterns; ++i)
        status.mutable_data()[i] = static_cast<std::int32_t>(reference.status[i]);
    result["status"] = status;
    result["fallback_guesses"] = data.fallback_guesses;
    result["dataset_setup_seconds"] = seconds(start);
    return result;
}
struct Data {
    FloatArray x, y, initial, lower, upper, parameters, chi;
    IntArray status;
    explicit Data(const py::dict &values)
        : x(values["x"].cast<FloatArray>()), y(values["y"].cast<FloatArray>()),
          initial(values["p0"].cast<FloatArray>()), lower(values["lower"].cast<FloatArray>()),
          upper(values["upper"].cast<FloatArray>()),
          parameters(values["parameters"].cast<FloatArray>()),
          chi(values["chi2"].cast<FloatArray>()), status(values["status"].cast<IntArray>()) {}
};
struct Measurement {
    double setup = 0, allocation = 0, prepare = 0, fit = 0, readback = 0, gpu = 0, wall = 0,
           validation = 0;
    std::size_t completed = 0, converged = 0, status_errors = 0, parameter_errors = 0,
                cost_errors = 0, chunks = 0, payload = 0;
    double iterations = 0;
    unsigned selected = 0;
};
template <template <typename> class Model>
Measurement run(const Data &data, std::size_t n, const std::string &backend, unsigned workers,
                std::size_t chunk) {
    constexpr unsigned p = Model<float>::parameter_count;
    if (data.x.ndim() != 1 || data.x.size() == 0 || data.y.ndim() != 2 ||
        data.y.shape(1) != data.x.size() || data.y.shape(0) == 0 || data.initial.ndim() != 2 ||
        data.initial.shape(0) != data.y.shape(0) || data.initial.shape(1) != p ||
        data.parameters.ndim() != 2 || data.parameters.shape(0) != data.y.shape(0) ||
        data.parameters.shape(1) != p || data.status.size() != data.y.shape(0) ||
        data.chi.size() != data.y.shape(0) || data.lower.size() != p || data.upper.size() != p)
        throw std::invalid_argument("benchmark inputs have inconsistent shapes");
    if (n == 0 || workers == 0 || chunk == 0)
        throw std::invalid_argument("N, workers and chunk must be positive");
    const auto m = static_cast<std::size_t>(data.x.size()),
               patterns = static_cast<std::size_t>(data.y.shape(0)), capacity = std::min(n, chunk);
    if (capacity > std::numeric_limits<std::size_t>::max() / m ||
        capacity * m > (std::size_t(512) << 20) / sizeof(float))
        throw std::invalid_argument("benchmark chunk must fit within 512 MiB");
    Variant variant = Variant::Automatic;
    const bool metal = backend.rfind("metal_", 0) == 0;
    if (backend == "metal_v1")
        variant = Variant::ThreadPerFit;
    else if (backend == "metal_v2_simd")
        variant = Variant::SimdPerFit;
    else if (backend == "metal_v2_group")
        variant = Variant::ThreadgroupPerFit;
    else if (backend != "metal_auto" && backend != "cpu_single" && backend != "cpu_multi")
        throw std::invalid_argument("unknown benchmark backend");
    // Keep Python owners above the GIL-release scope; only native buffers below it.
    const std::span<const float> x(data.x.data(), m);
    const auto *observations = data.y.data();
    const auto *initial = data.initial.data();
    const auto *reference = data.parameters.data();
    const auto *chi = data.chi.data();
    const auto *status = data.status.data();
    batchfit::cpu::SolverOptions<float, p> options;
    std::copy_n(data.lower.data(), p, options.lower.begin());
    std::copy_n(data.upper.data(), p, options.upper.begin());
    Measurement out;
    py::gil_scoped_release release;
    const auto setup = Clock::now();
    std::unique_ptr<batchfit::cpu::ThreadPool> pool;
    static std::unique_ptr<batchfit::metal::Context> context;
    if (metal) {
        if (!context)
            context = std::make_unique<batchfit::metal::Context>();
    } else
        pool = std::make_unique<batchfit::cpu::ThreadPool>(backend == "cpu_single" ? 1 : workers);
    out.setup = seconds(setup);
    const auto allocated = Clock::now();
    std::optional<batchfit::metal::SharedBatch> shared;
    std::vector<float> rows, starts;
    if (metal)
        shared = context->make_batch<Model>(capacity, m);
    else {
        rows.resize(capacity * m);
        starts.resize(capacity * p);
    }
    out.allocation = seconds(allocated);
    out.payload = sizeof(float) * (capacity * (m + std::size_t(p) * 2 + 3) + m);
    auto pack = [&](std::size_t first, std::size_t count) {
        if (metal) {
            shared->resize_fits(count);
            std::copy(x.begin(), x.end(), shared->x().begin());
        }
        auto *destination = metal ? shared->y().data() : rows.data();
        auto *seeds = metal ? shared->initial().data() : starts.data();
        for (std::size_t i = 0; i < count; ++i)
            std::copy_n(initial + ((first + i) % patterns) * p, p, seeds + i * p);
        if (!metal) {
            for (std::size_t i = 0; i < count; ++i)
                std::copy_n(observations + ((first + i) % patterns) * m, m, destination + i * m);
        } else
            for (std::size_t j0 = 0; j0 < m; j0 += 32)
                for (std::size_t i0 = 0; i0 < count; i0 += 32)
                    for (auto j = j0; j < std::min(m, j0 + 32); ++j)
                        for (auto i = i0; i < std::min(count, i0 + 32); ++i)
                            destination[j * count + i] =
                                observations[((first + i) % patterns) * m + j];
    };
    const auto warm = Clock::now();
    const auto warm_count = std::min(capacity, std::size_t(256));
    pack(0, warm_count);
    if (metal)
        batchfit::fit_batch<Model>(*shared, batchfit::Device::Metal, options, variant);
    else
        batchfit::cpu::fit_batch<float, Model>(x, std::span(rows).first(warm_count * m),
                                               std::span(starts).first(warm_count * p), *pool,
                                               options);
    out.setup += seconds(warm);
    const auto start = Clock::now();
    for (std::size_t first = 0; first < n; first += capacity) {
        const auto count = std::min(capacity, n - first);
        const auto prepare = Clock::now();
        pack(first, count);
        out.prepare += seconds(prepare);
        batchfit::cpu::BatchResult<float> result;
        const auto fit = Clock::now();
        if (metal) {
            const auto value =
                batchfit::fit_batch<Model>(*shared, batchfit::Device::Metal, options, variant);
            out.fit += seconds(fit);
            out.gpu += value.timings.gpu_seconds;
            out.selected = static_cast<unsigned>(value.timings.variant);
            const auto readback = Clock::now();
            result.parameters.assign(value.parameters.begin(), value.parameters.end());
            result.status.assign(value.status.begin(), value.status.end());
            result.iterations.assign(value.iterations.begin(), value.iterations.end());
            result.chi_squared.assign(value.chi_squared.begin(), value.chi_squared.end());
            out.readback += seconds(readback);
        } else {
            result = batchfit::cpu::fit_batch<float, Model>(x, std::span(rows).first(count * m),
                                                            std::span(starts).first(count * p),
                                                            *pool, options);
            out.fit += seconds(fit);
        }
        const auto checked = Clock::now();
        for (std::size_t i = 0; i < count; ++i) {
            const auto pattern = (first + i) % patterns;
            out.converged += result.status[i] == batchfit::FitStatus::converged;
            out.iterations += result.iterations[i];
            out.status_errors += static_cast<std::int32_t>(result.status[i]) != status[pattern];
            const auto value = result.chi_squared[i], expected = chi[pattern];
            out.cost_errors += !(std::isnan(value) && std::isnan(expected)) &&
                               !(std::abs(value - expected) <= 1e-6F + 2e-5F * std::abs(expected));
            bool bad = false;
            for (unsigned k = 0; k < p; ++k)
                if (status[pattern] == 0)
                    bad = bad ||
                          !(std::abs(result.parameters[i * p + k] - reference[pattern * p + k]) <=
                            2e-4F * (1 + std::abs(reference[pattern * p + k])));
            out.parameter_errors += bad;
        }
        out.validation += seconds(checked);
        out.completed += count;
        ++out.chunks;
    }
    out.wall = out.allocation + seconds(start) - out.validation;
    return out;
}
py::dict as_dict(const Measurement &value) {
    py::dict out;
    out["setup_seconds"] = value.setup;
    out["allocation_seconds"] = value.allocation;
    out["prepare_seconds"] = value.prepare;
    out["fit_wall_seconds"] = value.fit;
    out["readback_seconds"] = value.readback;
    out["gpu_seconds"] = value.gpu;
    out["wall_seconds"] = value.wall;
    out["validation_seconds"] = value.validation;
    out["completed"] = value.completed;
    out["converged"] = value.converged;
    out["status_errors"] = value.status_errors;
    out["parameter_errors"] = value.parameter_errors;
    out["cost_errors"] = value.cost_errors;
    out["chunks"] = value.chunks;
    out["shared_or_staging_bytes"] = value.payload;
    out["mean_iterations"] = value.iterations / double(value.completed);
    out["selected"] = value.selected;
    return out;
}
py::dict make_inputs(const std::string &model, std::size_t points, std::size_t patterns) {
    if (model == "gaussian")
        return inputs<batchfit::models::Gaussian>(points, patterns);
    if (model == "lorentzian")
        return inputs<batchfit::models::Lorentzian>(points, patterns);
    if (model == "pseudo_voigt")
        return inputs<batchfit::models::PseudoVoigt>(points, patterns);
    if (model == "two_lorentzian")
        return inputs<batchfit::models::TwoLorentzian>(points, patterns);
    throw std::invalid_argument("unknown model");
}
py::dict measure(const std::string &model, const py::dict &values, std::size_t n,
                 const std::string &backend, unsigned workers, std::size_t chunk) {
    const Data data(values);
    if (model == "gaussian")
        return as_dict(run<batchfit::models::Gaussian>(data, n, backend, workers, chunk));
    if (model == "lorentzian")
        return as_dict(run<batchfit::models::Lorentzian>(data, n, backend, workers, chunk));
    if (model == "pseudo_voigt")
        return as_dict(run<batchfit::models::PseudoVoigt>(data, n, backend, workers, chunk));
    if (model == "two_lorentzian")
        return as_dict(run<batchfit::models::TwoLorentzian>(data, n, backend, workers, chunk));
    throw std::invalid_argument("unknown model");
}
} // namespace
PYBIND11_MODULE(_benchmark, module) {
    module.def("inputs", &make_inputs);
    module.def("run", &measure);
}
