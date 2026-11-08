#pragma once

#include <batchfit/models.hpp>
#include <batchfit/npy.hpp>
#include <batchfit/synthetic.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace synthetic_checks {
using namespace batchfit::synthetic;
namespace fs = std::filesystem;

inline void require(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}
template <typename Function> void rejects(Function function) {
    bool rejected = false;
    try {
        function();
    } catch (const std::exception &) {
        rejected = true;
    }
    require(rejected, "invalid input was not rejected");
}
struct Scratch {
    fs::path path = ".agent-local/synthetic-test-" +
                    std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    Scratch() {
        fs::create_directories(path.parent_path());
        require(fs::create_directory(path), "scratch directory collision");
    }
    ~Scratch() { fs::remove_all(path); }
};
struct Data {
    std::vector<float> x, y, params;
    Consumer sink() {
        return [this](std::span<const float> grid, const Chunk &chunk) {
            require(chunk.first_fit * grid.size() == y.size(), "chunk offset/order");
            require(chunk.y.size() == chunk.fits * grid.size() &&
                        chunk.true_parameters.size() == chunk.fits * chunk.parameters_per_fit,
                    "chunk shape");
            if (x.empty())
                x.assign(grid.begin(), grid.end());
            require(std::equal(x.begin(), x.end(), grid.begin(), grid.end()), "shared x grid");
            y.insert(y.end(), chunk.y.begin(), chunk.y.end());
            params.insert(params.end(), chunk.true_parameters.begin(), chunk.true_parameters.end());
        };
    }
};

inline void reproducibility_and_roundtrip(const fs::path &root) {
    for (auto model :
         {Model::gaussian, Model::lorentzian, Model::pseudo_voigt, Model::two_lorentzian}) {
        for (auto noise : {Noise::none, Noise::gaussian, Noise::poisson}) {
            Options options;
            options.model = model;
            options.parameters = default_ranges(model);
            options.noise = noise;
            options.fits = 127;
            options.points = 17;
            options.chunk_fits = 1;
            Data first, second, loaded;
            generate(options, first.sink());
            require(first.x.front() == options.x_min && first.x.back() == options.x_max,
                    "grid endpoints");
            options.chunk_fits = 19;
            generate(options, second.sink());
            require(first.y == second.y && first.params == second.params, "chunk-invariant RNG");
            const auto output =
                root / (std::to_string(int(model)) + "-" + std::to_string(int(noise)));
            save_dataset(output, options);
            load_dataset(output, 7, loaded.sink());
            require(first.x == loaded.x && first.y == loaded.y && first.params == loaded.params,
                    "dataset NPY roundtrip");
            rejects([&] { save_dataset(output, options); });
            options.seed += 1;
            Data changed;
            generate(options, changed.sink());
            require(first.params != changed.params && first.y != changed.y, "seed changes data");
            const auto p = parameter_count(model);
            for (std::size_t i = 0; i < options.fits; ++i) {
                for (std::size_t k = 0; k < p; ++k) {
                    const float value = first.params[i * p + k];
                    require(value >= options.parameters[k].lower &&
                                value <= options.parameters[k].upper,
                            "parameters outside configured range");
                }
            }
        }
    }
    Options fixed;
    fixed.fits = 9;
    fixed.points = 13;
    fixed.noise = Noise::none;
    fixed.parameters[0] = {4, 4};
    fixed.parameters[1] = {1, 1};
    fixed.parameters[2] = {2, 2};
    generate(fixed, [&](std::span<const float> x, const Chunk &chunk) {
        for (std::size_t i = 0; i < chunk.fits; ++i) {
            for (std::size_t j = 0; j < x.size(); ++j) {
                const double expected = 4 * std::exp(-0.5 * std::pow((double(x[j]) - 1) / 2, 2));
                require(std::abs(chunk.y[i * x.size() + j] - expected) < 1e-6,
                        "noiseless generator vs independent Gaussian equation");
            }
        }
    });
    Options independent;
    independent.fits = 31;
    independent.chunk_fits = 7;
    independent.noise = Noise::none;
    Data clean;
    generate(independent, clean.sink());
    for (auto noise : {Noise::gaussian, Noise::poisson}) {
        independent.noise = noise;
        independent.snr = 7;
        Data noisy;
        generate(independent, noisy.sink());
        require(clean.params == noisy.params, "noise choice must not change ground truth");
        independent.snr = 23;
        Data changed_snr;
        generate(independent, changed_snr.sink());
        require(clean.params == changed_snr.params && noisy.y != changed_snr.y,
                "SNR changes noise but not ground truth");
    }
}

inline void noise_statistics() {
    for (auto noise : {Noise::gaussian, Noise::poisson}) {
        Options options;
        options.fits = 20000;
        options.points = 33;
        options.x_min = -4;
        options.x_max = 4;
        options.snr = 10;
        options.noise = noise;
        options.parameters[0] = {2, 2};
        options.parameters[1] = {0, 0};
        options.parameters[2] = {1, 1};
        double residual_sum = 0, residual_squares = 0, normalized_sum = 0, normalized_squares = 0;
        double fourth = 0, expected_squares = 0;
        std::size_t count = 0;
        generate(options, [&](std::span<const float> x, const Chunk &chunk) {
            std::vector<float> clean(x.size());
            double sum = 0, squares = 0;
            const float truth[] = {2, 0, 1};
            for (std::size_t j = 0; j < x.size(); ++j) {
                clean[j] = batchfit::models::Gaussian<float>::eval(x[j], truth);
                sum += clean[j];
                squares += double(clean[j]) * clean[j];
            }
            const double sigma = std::sqrt(squares / double(x.size())) / options.snr;
            const double gain = options.snr * options.snr * sum / squares;
            expected_squares += double(chunk.fits) * squares / (options.snr * options.snr);
            for (std::size_t i = 0; i < chunk.fits; ++i) {
                for (std::size_t j = 0; j < x.size(); ++j) {
                    const double observed = chunk.y[i * x.size() + j];
                    const double residual = observed - clean[j];
                    const double z =
                        residual / (noise == Noise::gaussian ? sigma : std::sqrt(clean[j] / gain));
                    residual_sum += residual;
                    residual_squares += residual * residual;
                    normalized_sum += z;
                    normalized_squares += z * z;
                    fourth += z * z * z * z;
                    ++count;
                    if (noise == Noise::poisson) {
                        require(observed >= 0 &&
                                    std::abs(observed * gain - std::round(observed * gain)) < 0.001,
                                "Poisson samples are nonnegative scaled integer counts");
                    }
                }
            }
        });
        require(std::abs(residual_sum / double(count)) < 0.001, "noise is unbiased");
        require(std::abs(normalized_sum / double(count)) < 0.01, "normalized noise mean");
        require(std::abs(normalized_squares / double(count) - 1) < 0.03, "noise variance");
        require(std::abs(residual_squares / expected_squares - 1) < 0.02, "requested RMS SNR");
        if (noise == Noise::gaussian) {
            require(std::abs(fourth / double(count) - 3) < 0.06, "Gaussian fourth moment");
        }
        options.fits = 3;
        options.parameters[0] = {0, 0};
        generate(options, [](std::span<const float>, const Chunk &chunk) {
            require(std::all_of(chunk.y.begin(), chunk.y.end(), [](float y) { return y == 0; }),
                    "zero signal produces zero noise");
        });
    }
}

inline void fixture(const fs::path &path, std::string dictionary,
                    const std::string &payload =
                        std::string("\x00\x00\x80\x3f\x00\x00\x00\xc0\x00\x00\x00\x3f", 12)) {
    dictionary.append((16 - (10 + dictionary.size() + 1) % 16) % 16, ' ');
    dictionary += '\n';
    std::ofstream file(path, std::ios::binary);
    file.write("\x93NUMPY\x01\x00", 8);
    const char length[] = {char(dictionary.size() & 255), char(dictionary.size() >> 8)};
    file.write(length, 2);
    file << dictionary << payload;
    require(bool(file), "fixture write");
}

inline void npy_checks(const fs::path &root) {
    const auto path = root / "fixture.npy";
    const std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (3,), }";
    fixture(path, header);
    batchfit::npy::Reader reader(path);
    std::vector<float> values(3);
    reader.read(values);
    require(reader.shape() == std::vector<std::size_t>{3} &&
                values == std::vector<float>{1, -2, 0.5F},
            "independent NPY fixture");
    rejects([&] { reader.read(values); });
    for (const auto &invalid :
         {std::string("{'descr': '>f4', 'fortran_order': False, 'shape': (3,), }"),
          std::string("{'descr': '<f8', 'fortran_order': False, 'shape': (3,), }"),
          std::string("{'descr': '<f4', 'fortran_order': True, 'shape': (3,), }"),
          std::string("{'descr': '<f4', 'fortran_order': False, 'shape': (3,,), }"),
          std::string("{'descr': '<f4', 'fortran_order': False, 'shape': "
                      "(999999999999999999999999,), }")}) {
        fixture(path, invalid);
        rejects([&] { batchfit::npy::Reader bad(path); });
    }
    fixture(path, header, "short");
    rejects([&] { batchfit::npy::Reader bad(path); });
    fixture(path, header, std::string(13, 'x'));
    rejects([&] { batchfit::npy::Reader bad(path); });
    fixture(path, header);
    fs::resize_file(path, 9);
    rejects([&] { batchfit::npy::Reader bad(path); });
    const auto written = root / "written.npy";
    {
        batchfit::npy::Writer writer(written, {3});
        writer.append(std::span(values).first(1));
        rejects([&] { writer.finish(); });
        writer.append(std::span(values).subspan(1));
        rejects([&] { writer.append(values); });
        writer.finish();
        rejects([&] { writer.finish(); });
    }
    std::ifstream raw(written, std::ios::binary);
    const std::string bytes((std::istreambuf_iterator<char>(raw)), {});
    require(bytes.substr(0, 8) == std::string("\x93NUMPY\x01\x00", 8), "NPY magic/version bytes");
    const auto length =
        static_cast<unsigned char>(bytes[8]) + 256 * static_cast<unsigned char>(bytes[9]);
    require((10 + length) % 64 == 0 && bytes[9 + length] == '\n', "NPY padding/newline");
    require(bytes.substr(10 + length) ==
                std::string("\x00\x00\x80\x3f\x00\x00\x00\xc0\x00\x00\x00\x3f", 12),
            "NPY independent little-endian float32 payload");
    {
        batchfit::npy::Writer empty(root / "empty.npy", {0, 3});
        empty.finish();
    }
    batchfit::npy::Reader empty(root / "empty.npy");
    empty.read({});
    rejects([&] {
        batchfit::npy::Writer bad(root / "bad.npy", {std::numeric_limits<std::size_t>::max(), 2});
    });
}

inline void invalid_options(const fs::path &root) {
    const auto consumer = [](std::span<const float>, const Chunk &) {};
    Options options;
    options.fits = 1;
    for (int choice = 0; choice < 13; ++choice) {
        auto bad = options;
        switch (choice) {
        case 0:
            bad.fits = 0;
            break;
        case 1:
            bad.points = 1;
            break;
        case 2:
            bad.chunk_fits = 0;
            break;
        case 3:
            bad.snr = 0;
            break;
        case 4:
            bad.x_max = bad.x_min;
            break;
        case 5:
            bad.parameters[2] = {0, 1};
            break;
        case 6:
            bad.parameters[0] = {2, 1};
            break;
        case 7:
            bad.fits = std::numeric_limits<std::size_t>::max();
            break;
        case 8:
            bad.parameters[0].upper = std::numeric_limits<float>::infinity();
            break;
        case 9:
            bad.snr = std::numeric_limits<double>::quiet_NaN();
            break;
        case 10:
            bad.noise = static_cast<Noise>(999);
            break;
        case 11:
            bad.model = static_cast<Model>(999);
            break;
        case 12:
            bad.model = Model::pseudo_voigt;
            bad.parameters[3] = {-1, 1};
            break;
        }
        rejects([&] { generate(bad, consumer); });
    }
    auto bad = options;
    bad.noise = Noise::poisson;
    bad.parameters[0] = {-2, -1};
    const auto failed = root / "failed";
    rejects([&] { save_dataset(failed, bad); });
    require(!fs::exists(failed), "failed save must roll back its own output");
    rejects([&] { generate(options, {}); });
    rejects([&] {
        generate(options, [](std::span<const float>, const Chunk &) {
            throw std::runtime_error("sink error");
        });
    });
    const auto output = root / "mismatched";
    save_dataset(output, options);
    {
        batchfit::npy::Writer writer(output / "true_params.npy", {2, 3});
        const std::vector<float> wrong(6);
        writer.append(wrong);
        writer.finish();
    }
    rejects([&] { load_dataset(output, 2, consumer); });
    rejects([&] { load_dataset(output, 0, consumer); });
}

inline void all() {
    Scratch scratch;
    reproducibility_and_roundtrip(scratch.path);
    noise_statistics();
    npy_checks(scratch.path);
    invalid_options(scratch.path);
}
} // namespace synthetic_checks
