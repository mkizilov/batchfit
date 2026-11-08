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
    return std::make_shared<Storage>(Storage{n,
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
} // namespace batchfit::metal::detail
