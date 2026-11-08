#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <span>

namespace batchfit::synthetic {
enum class Model { gaussian, lorentzian, pseudo_voigt, two_lorentzian };
enum class Noise { none, gaussian, poisson };
struct Range {
    float lower;
    float upper;
};
std::array<Range, 8> default_ranges(Model model);
std::size_t parameter_count(Model model);

struct Options {
    Model model = Model::gaussian;
    std::size_t fits = 1000;
    std::size_t points = 64;
    float x_min = -5;
    float x_max = 5;
    std::array<Range, 8> parameters = default_ranges(Model::gaussian);
    Noise noise = Noise::gaussian;
    double snr = 20;
    std::uint64_t seed = 42;
    std::size_t chunk_fits = 4096;
};

struct Chunk {
    std::size_t first_fit;
    std::size_t fits;
    std::size_t parameters_per_fit;
    std::span<const float> y;
    std::span<const float> true_parameters;
};
using Consumer = std::function<void(std::span<const float> x, const Chunk &chunk)>;

// Borrowed spans are valid only during the callback. Chunk size does not affect RNG output.
void generate(const Options &options, const Consumer &consume);
// Output directory must not already exist. Writes x.npy, y.npy, true_params.npy.
void save_dataset(const std::filesystem::path &directory, const Options &options);
// Streams the same callback layout from disk, validating all three shapes first.
void load_dataset(const std::filesystem::path &directory, std::size_t chunk_fits,
                  const Consumer &consume);
} // namespace batchfit::synthetic
