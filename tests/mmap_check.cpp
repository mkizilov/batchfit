#include "model_checks.hpp"
#include <batchfit/npy.hpp>
#include <chrono>
#include <iostream>

int main() {
    std::filesystem::path directory;
    bool owned = false;
    try {
        directory = std::filesystem::path(".agent-local") /
                    ("mmap-test-" +
                     std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        owned = std::filesystem::create_directory(directory);
        model_checks::require(owned, "create owned mmap fixture");
        std::vector<double> values(5003);
        for (std::size_t i = 0; i < values.size(); ++i)
            values[i] = double(i) / 7;
        {
            batchfit::npy::Writer writer(directory / "f8.npy", {values.size()},
                                         batchfit::npy::Dtype::float64);
            writer.append(values);
            writer.finish();
        }
        batchfit::npy::MappedReader file(directory / "f8.npy");
        for (std::size_t first : {0U, 1U, 2047U, 4095U}) {
            auto mapping = file.window(first, 17);
            std::array<float, 17> output{};
            mapping.copy_to(output);
            for (unsigned j = 0; j < 17; ++j)
                model_checks::require(output[j] == float(values[first + j]),
                                      "mapped double conversion/window alignment");
            auto moved = std::move(mapping);
            moved.copy_to(output);
        }
        file.window(values.size(), 0).copy_to({});
        bool rejected = false;
        try {
            file.window(values.size(), 1);
        } catch (const std::out_of_range &) {
            rejected = true;
        }
        model_checks::require(rejected, "mapped bounds checked");
        rejected = false;
        try {
            batchfit::npy::Reader legacy(directory / "f8.npy");
        } catch (const std::exception &) {
            rejected = true;
        }
        model_checks::require(rejected, "legacy Reader does not implicitly convert");
        {
            const std::array<std::uint32_t, 3> data{0, 2, 100};
            batchfit::npy::Writer writer(directory / "u4.npy", {3}, batchfit::npy::Dtype::uint32);
            writer.append(data);
            writer.finish();
        }
        model_checks::require(batchfit::npy::inspect(directory / "u4.npy").dtype ==
                                  batchfit::npy::Dtype::uint32,
                              "explicit integer output dtype");
        rejected = false;
        try {
            batchfit::npy::MappedReader unsupported(directory / "u4.npy");
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        model_checks::require(rejected, "mapped input dtype restricted");
        {
            batchfit::npy::Writer writer(directory / "f4.npy", {2});
            const std::array<float, 2> data{1.25F, -2.5F};
            writer.append(data);
            writer.finish();
        }
        batchfit::npy::MappedReader floats(directory / "f4.npy");
        std::array<float, 2> output{};
        floats.window(0, 2).copy_to(output);
        model_checks::require(output[0] == 1.25F && output[1] == -2.5F, "mapped float input");
        std::filesystem::resize_file(directory / "f4.npy",
                                     batchfit::npy::inspect(directory / "f4.npy").offset);
        rejected = false;
        try {
            floats.window(0, 2);
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        model_checks::require(rejected, "truncation rejected before mapping");
        std::filesystem::remove_all(directory);
        std::cout
            << "Windowed NPY mapping/conversion, typed output and malformed/bounds checks passed\n";
    } catch (const std::exception &error) {
        if (owned) {
            std::error_code cleanup;
            std::filesystem::remove_all(directory, cleanup);
            if (cleanup)
                std::cerr << "fixture cleanup failed: " << cleanup.message() << '\n';
        }
        std::cerr << error.what() << '\n';
        return 1;
    }
}
