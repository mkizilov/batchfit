#include <batchfit/batchfit.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>

namespace py = pybind11;
namespace {
// Bound total native fitting workers/resources even for concurrent Python calls.
std::mutex fitting;
batchfit::cpu::ThreadPool &pool() {
    static batchfit::cpu::ThreadPool workers(2);
    return workers;
}
batchfit::metal::Context &context() {
    static batchfit::metal::Context device;
    return device;
}
template <typename T> using Array = py::array_t<T, py::array::c_style>;

template <typename T> void aligned(const Array<T> &array) {
    if (!(array.flags() & py::array::c_style) ||
        reinterpret_cast<std::uintptr_t>(array.data()) % alignof(T) != 0)
        throw std::invalid_argument("native inputs must be aligned C-contiguous arrays");
}
template <typename T>
py::tuple arrays(batchfit::cpu::BatchResult<T> result, py::ssize_t n, py::ssize_t p,
                 bool uncertainties) {
    static_assert(sizeof(batchfit::FitStatus) == sizeof(std::int32_t) &&
                  sizeof(unsigned) == sizeof(std::uint32_t) &&
                  sizeof(batchfit::UncertaintyStatus) == sizeof(std::int32_t));
    auto data = std::make_unique<batchfit::cpu::BatchResult<T>>(std::move(result));
    py::capsule owner(data.get(), [](void *pointer) {
        delete static_cast<batchfit::cpu::BatchResult<T> *>(pointer);
    });
    const auto *storage = data.release();
    const py::ssize_t scalar = sizeof(T), integer = sizeof(std::uint32_t);
    auto basic = py::make_tuple(
        py::array(py::dtype::of<T>(), {n, p}, {p * scalar, scalar}, storage->parameters.data(),
                  owner),
        py::array(py::dtype::of<std::int32_t>(), {n}, {integer}, storage->status.data(), owner),
        py::array(py::dtype::of<std::uint32_t>(), {n}, {integer}, storage->iterations.data(),
                  owner),
        py::array(py::dtype::of<T>(), {n}, {scalar}, storage->chi_squared.data(), owner));
    if (!uncertainties)
        return basic;
    return py::make_tuple(basic[0], basic[1], basic[2], basic[3],
                          py::array(py::dtype::of<T>(), {n, p, p},
                                    {p * p * scalar, p * scalar, scalar},
                                    storage->covariance.data(), owner),
                          py::array(py::dtype::of<T>(), {n, p}, {p * scalar, scalar},
                                    storage->standard_errors.data(), owner),
                          py::array(py::dtype::of<std::int32_t>(), {n}, {integer},
                                    storage->uncertainty_status.data(), owner));
}

template <typename Out, typename In, template <typename> class Model>
batchfit::cpu::BatchResult<Out>
compute(std::span<const In> x, std::span<const In> y, std::span<const In> provided, std::size_t n,
        const batchfit::cpu::SolverOptions<In, Model<In>::parameter_count> &options, bool metal,
        bool uncertainties) {
    constexpr unsigned p = Model<In>::parameter_count;
    std::lock_guard lock(fitting);
    std::vector<In> generated;
    auto initial = provided;
    if (provided.empty() && n) {
        generated.resize(n * p);
        pool().parallel_for(n, [&](std::size_t begin, std::size_t end) {
            for (auto i = begin; i < end; ++i) {
                const auto guess =
                    batchfit::cpu::initial_guess<In, Model>(x, y.subspan(i * x.size(), x.size()));
                std::copy(guess.parameters.begin(), guess.parameters.end(),
                          generated.data() + i * p);
            }
        });
        initial = generated;
    }
    if (!metal) {
        if constexpr (std::is_same_v<Out, In>)
            return batchfit::cpu::fit_batch<In, Model>(x, y, initial, pool(), options,
                                                       uncertainties);
        else
            throw std::logic_error("CPU dtype dispatch mismatch");
    }
    if (!batchfit::metal::available())
        throw std::runtime_error("Metal fitting is unavailable in this build or on this device");
    auto shared = context().make_batch<Model>(n, x.size(), uncertainties);
    std::transform(x.begin(), x.end(), shared.x().begin(), [](In value) { return float(value); });
    std::transform(initial.begin(), initial.end(), shared.initial().begin(),
                   [](In value) { return float(value); });
    auto point_major = shared.y();
    for (std::size_t i0 = 0; i0 < n; i0 += 16)
        for (std::size_t j0 = 0; j0 < x.size(); j0 += 16)
            for (auto i = i0; i < std::min(n, i0 + 16); ++i)
                for (auto j = j0; j < std::min(x.size(), j0 + 16); ++j)
                    point_major[j * n + i] = float(y[i * x.size() + j]);
    batchfit::cpu::SolverOptions<float, p> gpu_options;
    for (unsigned k = 0; k < p; ++k) {
        gpu_options.lower[k] = float(options.lower[k]);
        gpu_options.upper[k] = float(options.upper[k]);
        if ((std::isfinite(options.lower[k]) && !std::isfinite(gpu_options.lower[k])) ||
            (std::isfinite(options.upper[k]) && !std::isfinite(gpu_options.upper[k])))
            throw std::invalid_argument("bounds are not representable in Metal float32");
    }
    const auto result = batchfit::fit_batch<Model>(shared, batchfit::Device::Metal, gpu_options);
    // Retain only small output vectors, not the transposed observation buffers.
    batchfit::cpu::BatchResult<Out> output{{result.parameters.begin(), result.parameters.end()},
                                           {result.status.begin(), result.status.end()},
                                           {result.iterations.begin(), result.iterations.end()},
                                           {result.chi_squared.begin(), result.chi_squared.end()}};
    if (uncertainties) {
        output.covariance.assign(result.covariance.begin(), result.covariance.end());
        output.standard_errors.assign(result.standard_errors.begin(), result.standard_errors.end());
        output.uncertainty_status.assign(result.uncertainty_status.begin(),
                                         result.uncertainty_status.end());
    }
    return output;
}

template <typename T, template <typename> class Model>
py::tuple fit_model(const Array<T> &x, const Array<T> &y, const py::object &p0,
                    const Array<T> &lower, const Array<T> &upper, bool metal, bool uncertainties) {
    constexpr unsigned p = Model<T>::parameter_count;
    aligned(x);
    aligned(y);
    aligned(lower);
    aligned(upper);
    if (x.ndim() != 1 || x.shape(0) == 0 || y.ndim() != 2 || y.shape(1) != x.shape(0))
        throw std::invalid_argument("expected x[M] and y[N,M] with M>0");
    if (lower.ndim() != 1 || upper.ndim() != 1 || lower.size() != p || upper.size() != p)
        throw std::invalid_argument("bounds must have shape [P]");
    const auto n = y.shape(0);
    if (n > std::numeric_limits<py::ssize_t>::max() / p)
        throw std::invalid_argument("result shape exceeds addressable NumPy size");
    if (uncertainties && n > std::numeric_limits<py::ssize_t>::max() / p / p)
        throw std::invalid_argument("covariance shape exceeds addressable NumPy size");
    std::optional<Array<T>> initial_owner;
    std::span<const T> initial;
    if (!p0.is_none()) {
        if (!py::isinstance<Array<T>>(p0))
            throw std::invalid_argument("initial dtype must match data");
        initial_owner = py::reinterpret_borrow<Array<T>>(p0);
        aligned(*initial_owner);
        if (initial_owner->ndim() != 2 || initial_owner->shape(0) != n ||
            initial_owner->shape(1) != p)
            throw std::invalid_argument("initial parameters must have shape [N,P]");
        initial = {initial_owner->data(), static_cast<std::size_t>(initial_owner->size())};
    }
    batchfit::cpu::SolverOptions<T, p> options;
    for (unsigned k = 0; k < p; ++k) {
        options.lower[k] = lower.data()[k];
        options.upper[k] = upper.data()[k];
        if (std::isnan(options.lower[k]) || std::isnan(options.upper[k]) ||
            options.lower[k] > options.upper[k] ||
            options.lower[k] == std::numeric_limits<T>::infinity() ||
            options.upper[k] == -std::numeric_limits<T>::infinity())
            throw std::invalid_argument("invalid parameter bounds");
    }
    const std::span<const T> samples(x.data(), static_cast<std::size_t>(x.size()));
    const std::span<const T> observations(y.data(), static_cast<std::size_t>(y.size()));
    if (metal) {
        batchfit::cpu::BatchResult<float> result;
        {
            py::gil_scoped_release release;
            result =
                compute<float, T, Model>(samples, observations, initial,
                                         static_cast<std::size_t>(n), options, true, uncertainties);
        }
        return arrays(std::move(result), n, p, uncertainties);
    }
    batchfit::cpu::BatchResult<T> result;
    {
        py::gil_scoped_release release;
        result = compute<T, T, Model>(samples, observations, initial, static_cast<std::size_t>(n),
                                      options, false, uncertainties);
    }
    return arrays(std::move(result), n, p, uncertainties);
}
template <typename T>
py::tuple fit(const Array<T> &x, const Array<T> &y, const py::object &p0, const Array<T> &lower,
              const Array<T> &upper, const std::string &model, bool metal, bool uncertainties) {
    if (model == "gaussian")
        return fit_model<T, batchfit::models::Gaussian>(x, y, p0, lower, upper, metal,
                                                        uncertainties);
    if (model == "lorentzian")
        return fit_model<T, batchfit::models::Lorentzian>(x, y, p0, lower, upper, metal,
                                                          uncertainties);
    if (model == "pseudo_voigt")
        return fit_model<T, batchfit::models::PseudoVoigt>(x, y, p0, lower, upper, metal,
                                                           uncertainties);
    if (model == "two_lorentzian")
        return fit_model<T, batchfit::models::TwoLorentzian>(x, y, p0, lower, upper, metal,
                                                             uncertainties);
    throw std::invalid_argument("unknown fit model");
}
} // namespace
PYBIND11_MODULE(_core, module) {
    module.doc() = "Native batchfit buffer interface; use batchfit.fit for array preparation.";
    module.def("fit", &fit<float>, py::arg("x").noconvert(), py::arg("y").noconvert(),
               py::arg("p0"), py::arg("lower").noconvert(), py::arg("upper").noconvert(),
               py::arg("model"), py::arg("metal"), py::arg("uncertainties") = false);
    module.def("fit", &fit<double>, py::arg("x").noconvert(), py::arg("y").noconvert(),
               py::arg("p0"), py::arg("lower").noconvert(), py::arg("upper").noconvert(),
               py::arg("model"), py::arg("metal"), py::arg("uncertainties") = false);
    module.def("metal_available", [] {
        py::gil_scoped_release release;
        return batchfit::metal::available();
    });
    module.attr("__version__") = py::str(std::string(batchfit::version));
}
