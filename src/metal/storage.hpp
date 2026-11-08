#pragma once

#include <batchfit/metal/batch.hpp>

#include <array>
#include <cstdint>

namespace batchfit::metal::detail {
static_assert(sizeof(Config) == 92 && sizeof(float) == 4 && sizeof(unsigned) == 4 &&
              sizeof(FitStatus) == 4);
static_assert(unsigned(FitStatus::converged) == 0 && unsigned(FitStatus::max_iterations) == 1 &&
              unsigned(FitStatus::singular_matrix) == 2 && unsigned(FitStatus::invalid_input) == 3);

inline std::array<std::size_t, 8> buffer_counts(std::size_t n, std::size_t m, unsigned p) {
    constexpr auto limit = std::numeric_limits<std::uint32_t>::max();
    if (m == 0 || m > limit || p == 0 || p > 8 || n > limit / m || n > limit / p)
        throw std::invalid_argument(
            "Metal dimensions require M>0, 1<=P<=8 and uint32-addressable arrays");
    return {m, n * m, n * p, sizeof(Config) / 4, n * p, n, n, n};
}

inline std::shared_ptr<Storage> make_storage(std::size_t n, std::size_t m, unsigned p,
                                             const std::array<void *, 8> &pointers,
                                             std::shared_ptr<void> resources) {
    return std::shared_ptr<Storage>(new Storage{n,
                                                n,
                                                m,
                                                p,
                                                {static_cast<float *>(pointers[0]), m},
                                                {static_cast<float *>(pointers[1]), n * m},
                                                {static_cast<float *>(pointers[2]), n * p},
                                                {static_cast<float *>(pointers[4]), n * p},
                                                {static_cast<float *>(pointers[7]), n},
                                                {static_cast<FitStatus *>(pointers[5]), n},
                                                {static_cast<unsigned *>(pointers[6]), n},
                                                static_cast<Config *>(pointers[3]),
                                                std::move(resources)});
}
inline constexpr const char *kernel_names[] = {"fit_gaussian", "fit_lorentzian", "fit_pseudo_voigt",
                                               "fit_two_lorentzian"};
inline constexpr const char *variant_kernel_names[] = {
    "fit_gaussian",          "fit_lorentzian",          "fit_pseudo_voigt",
    "fit_two_lorentzian",    "fit_gaussian_simd",       "fit_lorentzian_simd",
    "fit_pseudo_voigt_simd", "fit_two_lorentzian_simd", "fit_gaussian_group",
    "fit_lorentzian_group",  "fit_pseudo_voigt_group",  "fit_two_lorentzian_group"};
inline unsigned pipeline_index(unsigned model, std::size_t points, KernelVariant &variant) {
    if (variant == KernelVariant::Automatic)
        variant = dispatch_variant(model, points);
    const auto value = unsigned(variant);
    if (model >= 4 || value < 1 || value > 3)
        throw std::invalid_argument("unknown Metal kernel variant/model");
    return (value - 1) * 4 + model;
}
} // namespace batchfit::metal::detail
