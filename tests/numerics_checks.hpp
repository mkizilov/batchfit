#pragma once
#include "model_checks.hpp"
#include <batchfit/batchfit.hpp>

namespace numerics_checks {
using model_checks::require;
template <typename T, template <typename> class Model>
std::array<T, Model<T>::parameter_count> parameters() {
    if constexpr (Model<T>::parameter_count == 8)
        return {T(2), T(-.4), T(.12), T(1.5), T(.4), T(.15), T(.2), T(.01)};
    else if constexpr (Model<T>::parameter_count == 4)
        return {T(2), T(0), T(.2), T(.4)};
    else
        return {T(2), T(0), T(.2)};
}
template <typename T, template <typename> class Model> void cpu_paths() {
    constexpr unsigned p = Model<T>::parameter_count;
    std::array<T, 129> x{}, y{};
    for (unsigned j = 0; j < x.size(); ++j)
        x[j] = T(-1) + T(j) / T(64);
    const auto initial = parameters<T, Model>();
    for (unsigned flags = 0; flags < 4; ++flags) {
        batchfit::cpu::SolverOptions<T, p> options;
        options.compensated_sum = flags & 1;
        options.scaled_cholesky = flags & 2;
        y.fill(T(0));
        require(batchfit::cpu::fit_single<T, Model>(x, y, initial, options).status ==
                    batchfit::FitStatus::degenerate_data,
                "zero spectra diagnosed");
        y.fill(T(3));
        require(batchfit::cpu::fit_single<T, Model>(x, y, initial, options).status ==
                    batchfit::FitStatus::degenerate_data,
                "flat spectra diagnosed");
        for (unsigned j = 0; j < x.size(); ++j)
            y[j] = Model<T>::eval(x[j], initial.data());
        for (T bad : {std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::infinity(),
                      -std::numeric_limits<T>::infinity()}) {
            auto damaged = y;
            damaged[17] = bad;
            const auto result = batchfit::cpu::fit_single<T, Model>(x, damaged, initial, options);
            require(result.status == batchfit::FitStatus::invalid_input &&
                        std::isnan(result.chi_squared),
                    "nonfinite points rejected");
        }
        auto edge = initial;
        edge[1] = T(-1);
        for (unsigned j = 0; j < x.size(); ++j)
            y[j] = Model<T>::eval(x[j], edge.data());
        require(batchfit::cpu::fit_single<T, Model>(x, y, edge, options).status ==
                    batchfit::FitStatus::converged,
                "resolved edge peak remains valid");
        auto narrow = initial;
        narrow[2] = T(1e-4);
        if constexpr (p == 8)
            narrow[5] = T(1e-4);
        for (unsigned j = 0; j < x.size(); ++j)
            y[j] = Model<T>::eval(x[j], narrow.data());
        require(batchfit::cpu::fit_single<T, Model>(x, y, narrow, options).status ==
                    batchfit::FitStatus::unresolved_peak,
                "sub-spacing FWHM is not reported as resolved convergence");
    }
}
template <template <typename> class Model> void coordinates() {
    constexpr unsigned p = Model<double>::parameter_count;
    std::array<double, 129> x{}, y{};
    auto physical = parameters<double, Model>();
    constexpr double offset = 1e6, scale = .01, signal = 1e-6;
    physical[0] *= signal;
    physical[1] = std::fma(physical[1], scale, offset);
    physical[2] *= scale;
    if constexpr (p == 8) {
        physical[3] *= signal;
        physical[4] = std::fma(physical[4], scale, offset);
        physical[5] *= scale;
        physical[7] *= signal / scale;
        physical[6] = std::fma(-physical[7], offset, physical[6] * signal);
    }
    for (unsigned j = 0; j < x.size(); ++j) {
        x[j] = offset + scale * (-1 + double(j) / 64);
        y[j] = Model<double>::eval(x[j], physical.data());
    }
    const auto frame = batchfit::Coordinates::from(x, y);
    const auto normalized = frame.normalize<Model>(physical);
    const auto back = frame.physical<Model>(normalized);
    for (unsigned k = 0; k < p; ++k)
        model_checks::near(back[k], physical[k], 1e-12, 1e-14, "coordinate parameter roundtrip");
    for (unsigned j = 0; j < x.size(); ++j)
        model_checks::near(Model<double>::eval(frame.x(x[j]), normalized.data()), frame.y(y[j]),
                           1e-7, 1e-10,
                           "coordinate model equivalence including centered linear baseline");
    require(std::abs(frame.x(x.front()) + 1) < 1e-7 && std::abs(frame.x(x.back()) - 1) < 1e-7,
            "centered grid endpoints");
}
inline void cpu() {
    float sum = 1e8F, correction = 0;
    for (unsigned i = 0; i < 1000; ++i)
        batchfit::numerics::add(1.F, sum, correction);
    require(sum == 100001000.F, "compensated small terms retained");
    for (bool scaled : {false, true}) {
        batchfit::cpu::detail::Matrix<double, 3> matrix{{{4, 1, 0}, {1, 3, 1}, {0, 1, 2}}};
        std::array<double, 3> rhs{6, 10, 8};
        require(batchfit::cpu::detail::cholesky_solve<double, 3>(matrix, rhs, scaled),
                "checked equilibrated Cholesky");
        for (unsigned k = 0; k < 3; ++k)
            model_checks::near(rhs[k], double(k + 1), 1e-14, 1e-14, "scaled Cholesky algebra");
    }
    cpu_paths<float, batchfit::models::Gaussian>();
    cpu_paths<double, batchfit::models::Gaussian>();
    cpu_paths<float, batchfit::models::Lorentzian>();
    cpu_paths<double, batchfit::models::Lorentzian>();
    cpu_paths<float, batchfit::models::PseudoVoigt>();
    cpu_paths<double, batchfit::models::PseudoVoigt>();
    cpu_paths<float, batchfit::models::TwoLorentzian>();
    cpu_paths<double, batchfit::models::TwoLorentzian>();
    coordinates<batchfit::models::Gaussian>();
    coordinates<batchfit::models::Lorentzian>();
    coordinates<batchfit::models::PseudoVoigt>();
    coordinates<batchfit::models::TwoLorentzian>();
}
template <template <typename> class Model> void gpu_model(batchfit::metal::Context &context) {
    constexpr unsigned p = Model<float>::parameter_count;
    constexpr std::size_t n = 9, m = 129;
    auto batch = context.make_batch<Model>(n, m);
    auto base = parameters<float, Model>();
    for (unsigned j = 0; j < m; ++j)
        batch.x()[j] = -1 + float(j) / 64;
    for (unsigned i = 0; i < n; ++i) {
        auto initial = base;
        if (i == 5)
            initial[1] = -1;
        if (i == 6) {
            initial[2] = 1e-4F;
            if constexpr (p == 8)
                initial[5] = 1e-4F;
        }
        if (i == 8) {
            initial[0] *= 1e30F;
            if constexpr (p == 8)
                initial[3] *= 1e30F;
        }
        std::copy(initial.begin(), initial.end(), batch.initial().begin() + i * p);
        for (unsigned j = 0; j < m; ++j)
            batch.y()[j * n + i] = i == 0   ? 0
                                   : i == 1 ? 3
                                            : Model<float>::eval(batch.x()[j], initial.data());
    }
    batch.y()[17 * n + 2] = std::numeric_limits<float>::quiet_NaN();
    batch.y()[17 * n + 3] = std::numeric_limits<float>::infinity();
    batch.y()[17 * n + 4] = -std::numeric_limits<float>::infinity();
    for (unsigned flags = 0; flags < 4; ++flags)
        for (auto variant : {batchfit::metal::KernelVariant::ThreadPerFit,
                             batchfit::metal::KernelVariant::SimdPerFit,
                             batchfit::metal::KernelVariant::ThreadgroupPerFit,
                             batchfit::metal::KernelVariant::Automatic}) {
            batchfit::cpu::SolverOptions<float, p> options;
            options.compensated_sum = flags & 1;
            options.scaled_cholesky = flags & 2;
            const auto result =
                batchfit::fit_batch<Model>(batch, batchfit::Device::Metal, options, variant);
            require(result.status[0] == batchfit::FitStatus::degenerate_data &&
                        result.status[1] == batchfit::FitStatus::degenerate_data,
                    "GPU flat/zero diagnostic");
            for (unsigned i = 2; i <= 4; ++i)
                require(result.status[i] == batchfit::FitStatus::invalid_input &&
                            std::isnan(result.chi_squared[i]),
                        "GPU nonfinite guard survives fast math");
            require(result.status[5] == batchfit::FitStatus::converged &&
                        result.status[6] == batchfit::FitStatus::unresolved_peak &&
                        result.status[7] == batchfit::FitStatus::converged,
                    "GPU edge/subspacing/ordinary isolation");
            require(result.status[8] == batchfit::FitStatus::invalid_input,
                    "finite-input overflow is diagnosed");
            for (unsigned i : {0U, 1U, 5U, 6U, 7U}) {
                require(std::isfinite(result.chi_squared[i]),
                        "diagnosed finite accepted objective");
                for (unsigned k = 0; k < p; ++k)
                    require(std::isfinite(result.parameters[i * p + k]),
                            "finite accepted parameters");
            }
        }
    batch.x()[17] = std::numeric_limits<float>::quiet_NaN();
    for (auto variant :
         {batchfit::metal::KernelVariant::ThreadPerFit, batchfit::metal::KernelVariant::SimdPerFit,
          batchfit::metal::KernelVariant::ThreadgroupPerFit}) {
        const auto result = batchfit::fit_batch<Model>(batch, batchfit::Device::Metal, {}, variant);
        for (auto status : result.status)
            require(status == batchfit::FitStatus::invalid_input,
                    "nonfinite shared x is rejected in both math modes");
    }
}
template <template <typename> class Model> void accuracy(batchfit::metal::Context &context) {
    constexpr unsigned p = Model<double>::parameter_count;
    std::array<double, 257> x{}, y{}, u{}, v{};
    const batchfit::Coordinates base{1e6, .005, 1e-6};
    const auto physical = base.physical<Model>(parameters<double, Model>());
    for (unsigned j = 0; j < x.size(); ++j) {
        x[j] = base.origin + base.x_scale * (-1 + double(j) / 128);
        y[j] = Model<double>::eval(x[j], physical.data());
    }
    const auto frame = batchfit::Coordinates::from(x, y);
    for (unsigned j = 0; j < x.size(); ++j) {
        u[j] = frame.x(x[j]);
        v[j] = frame.y(y[j]);
    }
    const auto guess = batchfit::cpu::initial_guess<double, Model>(u, v);
    require(guess.status == batchfit::cpu::GuessStatus::success, "data-only normalized seed");
    batchfit::cpu::SolverOptions<double, p> reference_options;
    reference_options.compensated_sum = true;
    reference_options.scaled_cholesky = true;
    reference_options.lower[0] = 0;
    reference_options.lower[2] = 1e-5;
    if constexpr (p == 4) {
        reference_options.lower[3] = 0;
        reference_options.upper[3] = 1;
    }
    if constexpr (p == 8) {
        reference_options.lower[3] = 0;
        reference_options.lower[5] = 1e-5;
    }
    const auto reference =
        batchfit::cpu::fit_single<double, Model>(u, v, guess.parameters, reference_options);
    require(reference.status == batchfit::FitStatus::converged, "float64 normalized reference");
    batchfit::cpu::SolverOptions<float, p> options;
    options.compensated_sum = true;
    options.scaled_cholesky = true;
    for (unsigned k = 0; k < p; ++k) {
        options.lower[k] = float(reference_options.lower[k]);
        options.upper[k] = float(reference_options.upper[k]);
    }
    auto batch = context.make_batch<Model>(1, x.size());
    for (unsigned j = 0; j < x.size(); ++j) {
        batch.x()[j] = float(u[j]);
        batch.y()[j] = float(v[j]);
    }
    for (unsigned k = 0; k < p; ++k)
        batch.initial()[k] = float(guess.parameters[k]);
    for (auto variant :
         {batchfit::metal::KernelVariant::ThreadPerFit, batchfit::metal::KernelVariant::SimdPerFit,
          batchfit::metal::KernelVariant::ThreadgroupPerFit}) {
        const auto result =
            batchfit::fit_batch<Model>(batch, batchfit::Device::Metal, options, variant);
        require(result.status[0] == reference.status, "robust float64-reference status");
        std::array<double, p> q{};
        for (unsigned k = 0; k < p; ++k) {
            q[k] = result.parameters[k];
            model_checks::near(q[k], reference.parameters[k], 2e-4, 2e-4,
                               "robust unit-parameter CPU64 parity");
        }
        const auto recovered = frame.physical<Model>(q);
        require(std::abs(recovered[1] - physical[1]) < 1e-6,
                "host reconstruction beats raw float32 coordinate ULP");
    }
}
inline void gpu() {
    for (auto mode : {batchfit::metal::MathMode::Safe, batchfit::metal::MathMode::Fast}) {
        batchfit::metal::Context context(mode);
        gpu_model<batchfit::models::Gaussian>(context);
        gpu_model<batchfit::models::Lorentzian>(context);
        gpu_model<batchfit::models::PseudoVoigt>(context);
        gpu_model<batchfit::models::TwoLorentzian>(context);
        if (mode == batchfit::metal::MathMode::Safe) {
            accuracy<batchfit::models::Gaussian>(context);
            accuracy<batchfit::models::Lorentzian>(context);
            accuracy<batchfit::models::PseudoVoigt>(context);
            accuracy<batchfit::models::TwoLorentzian>(context);
        }
    }
}
} // namespace numerics_checks
