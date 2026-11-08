#include <batchfit/models.hpp>
#include <batchfit/npy.hpp>
#include <batchfit/synthetic.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace batchfit::synthetic {
namespace {
void require(bool valid, const char *message) {
    if (!valid) {
        throw std::invalid_argument(message);
    }
}

std::size_t product(std::size_t a, std::size_t b) {
    require(b == 0 || a <= std::numeric_limits<std::size_t>::max() / b / sizeof(float),
            "synthetic array dimensions overflow");
    return a * b;
}

void validate(const Options &options) {
    const auto p = parameter_count(options.model);
    require(options.fits > 0 && options.points >= 2 && options.chunk_fits > 0,
            "fits/chunk_fits must be positive and points must be at least 2");
    product(options.fits, options.points);
    product(options.fits, p);
    require(std::isfinite(options.x_min) && std::isfinite(options.x_max) &&
                options.x_min < options.x_max,
            "x_min and x_max must be finite and increasing");
    require(std::isfinite(options.snr) && options.snr > 0, "snr must be finite and positive");
    require(options.noise == Noise::none || options.noise == Noise::gaussian ||
                options.noise == Noise::poisson,
            "unknown noise type");
    for (std::size_t k = 0; k < p; ++k) {
        const auto range = options.parameters[k];
        require(std::isfinite(range.lower) && std::isfinite(range.upper) &&
                    range.lower <= range.upper &&
                    double(range.upper) - range.lower <= std::numeric_limits<float>::max(),
                "parameter ranges must be finite and ordered");
        if (k == 2 || (options.model == Model::two_lorentzian && k == 5)) {
            require(range.lower > 0, "width ranges must be strictly positive");
        }
    }
    if (options.model == Model::pseudo_voigt) {
        require(options.parameters[3].lower >= 0 && options.parameters[3].upper <= 1,
                "pseudo-Voigt mixing range must lie in [0,1]");
    }
}

template <typename FitModel> void generate_model(const Options &options, const Consumer &consume) {
    constexpr auto p = FitModel::parameter_count;
    std::vector<float> x(options.points);
    for (std::size_t j = 0; j < x.size(); ++j) {
        x[j] = static_cast<float>(double(options.x_min) +
                                  (double(options.x_max) - options.x_min) *
                                      (double(j) / double(options.points - 1)));
    }
    const auto capacity = std::min(options.fits, options.chunk_fits);
    std::vector<float> y(product(capacity, options.points)), params(product(capacity, p));
    std::mt19937_64 parameter_rng(options.seed);
    std::mt19937_64 noise_rng(options.seed ^ 0x9E3779B97F4A7C15ULL);
    std::normal_distribution<float> normal;
    for (std::size_t first = 0; first < options.fits;) {
        const auto count = std::min(capacity, options.fits - first);
        for (std::size_t i = 0; i < count; ++i) {
            float *parameters = params.data() + i * p;
            for (unsigned k = 0; k < p; ++k) {
                const auto range = options.parameters[k];
                parameters[k] =
                    std::uniform_real_distribution<float>(range.lower, range.upper)(parameter_rng);
            }
            float *row = y.data() + i * options.points;
            double sum = 0, squares = 0;
            for (std::size_t j = 0; j < options.points; ++j) {
                row[j] = FitModel::eval(x[j], parameters);
                require(std::isfinite(row[j]), "model produced a non-finite signal");
                require(options.noise != Noise::poisson || row[j] >= 0,
                        "Poisson noise requires a nonnegative signal at every point");
                sum += row[j];
                squares += double(row[j]) * row[j];
            }
            if (options.noise == Noise::gaussian) {
                const auto sigma = std::sqrt(squares / double(options.points)) / options.snr;
                for (std::size_t j = 0; j < options.points; ++j) {
                    row[j] = static_cast<float>(row[j] + sigma * normal(noise_rng));
                    require(std::isfinite(row[j]), "Gaussian noise exceeds float32 range");
                }
            } else if (options.noise == Noise::poisson && squares > 0) {
                const double gain = options.snr * options.snr * sum / squares;
                require(std::isfinite(gain) && gain > 0, "Poisson count scaling is out of range");
                for (std::size_t j = 0; j < options.points; ++j) {
                    const double mean = gain * row[j];
                    require(std::isfinite(mean) && mean <= 1e12,
                            "Poisson expected count exceeds supported range [0,1e12]");
                    if (mean == 0) {
                        row[j] = 0;
                        continue;
                    }
                    const auto counts = std::poisson_distribution<std::uint64_t>(mean)(noise_rng);
                    row[j] = static_cast<float>(double(counts) / gain);
                    require(std::isfinite(row[j]), "Poisson noise exceeds float32 range");
                }
            }
        }
        consume(x, {first, count, p, std::span(y).first(count * options.points),
                    std::span(params).first(count * p)});
        first += count;
    }
}
} // namespace

std::size_t parameter_count(Model model) {
    switch (model) {
    case Model::gaussian:
        return models::Gaussian<float>::parameter_count;
    case Model::lorentzian:
        return models::Lorentzian<float>::parameter_count;
    case Model::pseudo_voigt:
        return models::PseudoVoigt<float>::parameter_count;
    case Model::two_lorentzian:
        return models::TwoLorentzian<float>::parameter_count;
    }
    throw std::invalid_argument("unknown synthetic model");
}

std::array<Range, 8> default_ranges(Model model) {
    parameter_count(model);
    std::array<Range, 8> ranges{
        {{1, 3}, {-1, 1}, {0.3F, 1}, {0, 1}, {0, 0}, {0, 0}, {0, 0}, {0, 0}}};
    if (model == Model::two_lorentzian) {
        ranges = {
            {{1, 3}, {-2, -0.5F}, {0.3F, 1}, {1, 3}, {0.5F, 2}, {0.3F, 1}, {0, 0.2F}, {0, 0}}};
    }
    return ranges;
}

void generate(const Options &options, const Consumer &consume) {
    validate(options);
    require(bool(consume), "a dataset consumer is required");
    switch (options.model) {
    case Model::gaussian:
        return generate_model<models::Gaussian<float>>(options, consume);
    case Model::lorentzian:
        return generate_model<models::Lorentzian<float>>(options, consume);
    case Model::pseudo_voigt:
        return generate_model<models::PseudoVoigt<float>>(options, consume);
    case Model::two_lorentzian:
        return generate_model<models::TwoLorentzian<float>>(options, consume);
    }
}

void save_dataset(const std::filesystem::path &directory, const Options &options) {
    validate(options);
    require(std::filesystem::create_directory(directory),
            "dataset output directory already exists");
    try {
        npy::Writer x_file(directory / "x.npy", {options.points});
        npy::Writer y_file(directory / "y.npy", {options.fits, options.points});
        npy::Writer params_file(directory / "true_params.npy",
                                {options.fits, parameter_count(options.model)});
        generate(options, [&](std::span<const float> x, const Chunk &chunk) {
            if (chunk.first_fit == 0) {
                x_file.append(x);
            }
            y_file.append(chunk.y);
            params_file.append(chunk.true_parameters);
        });
        x_file.finish();
        y_file.finish();
        params_file.finish();
    } catch (...) {
        // This call exclusively created the directory; never remove preexisting data.
        std::filesystem::remove_all(directory);
        throw;
    }
}

void load_dataset(const std::filesystem::path &directory, std::size_t chunk_fits,
                  const Consumer &consume) {
    require(chunk_fits > 0 && bool(consume), "positive chunk_fits and a consumer are required");
    npy::Reader x_file(directory / "x.npy"), y_file(directory / "y.npy"),
        params_file(directory / "true_params.npy");
    const auto &xs = x_file.shape();
    const auto &ys = y_file.shape();
    const auto &ps = params_file.shape();
    require(xs.size() == 1 && xs[0] >= 2 && ys.size() == 2 && ps.size() == 2 && ys[0] > 0 &&
                ys[1] == xs[0] && ps[0] == ys[0] && ps[1] > 0 && ps[1] <= 8,
            "inconsistent dataset x/y/true_params shapes");
    const auto capacity = std::min(chunk_fits, ys[0]);
    std::vector<float> x(xs[0]), y(product(capacity, ys[1])), params(product(capacity, ps[1]));
    x_file.read(x);
    for (std::size_t first = 0; first < ys[0];) {
        const auto count = std::min(capacity, ys[0] - first);
        auto data = std::span(y).first(count * ys[1]);
        auto truth = std::span(params).first(count * ps[1]);
        y_file.read(data);
        params_file.read(truth);
        consume(x, {first, count, ps[1], data, truth});
        first += count;
    }
}
} // namespace batchfit::synthetic
