#include "model_checks.hpp"
#include "timeline.hpp"
#include <batchfit/batchfit.hpp>
#include <batchfit/npy.hpp>
#include <iomanip>
#include <iostream>
#include <sys/resource.h>
#include <sys/stat.h>

namespace {
void check_near(double actual, double expected, double absolute, double relative,
                const char *message) {
    if (!std::isfinite(actual) || !std::isfinite(expected) ||
        std::abs(actual - expected) > absolute + relative * std::abs(expected))
        model_checks::near(actual, expected, absolute, relative, message);
}
struct Dataset {
    std::filesystem::path directory;
    std::size_t n, m, item_bytes;
    std::vector<batchfit::cpu::FitResult<float, 8>> reference;
    batchfit::cpu::SolverOptions<float, 8> options;
    explicit Dataset(bool smoke, bool float32)
        : directory(std::filesystem::path(".agent-local") /
                    ("pipeline-benchmark-" +
                     std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))),
          n(smoke     ? 257
            : float32 ? 5000000
                      : 312500),
          m(smoke     ? 128
            : float32 ? 512
                      : 4096),
          item_bytes(float32 ? 4 : 8) {
        if (!std::filesystem::create_directory(directory))
            throw std::runtime_error("benchmark directory collision");
        try {
            if (std::filesystem::space(directory).available <
                n * m * item_bytes + (std::size_t(1) << 30))
                throw std::runtime_error("insufficient local space for dense dataset");
            const std::size_t patterns = 64;
            std::vector<float> x(m), initial(patterns * 8);
            std::vector<double> rows(patterns * m);
            for (std::size_t j = 0; j < m; ++j)
                x[j] = -5 + 10 * float(j) / float(m - 1);
            options.lower[0] = options.lower[3] = 0;
            options.lower[1] = options.lower[4] = -5;
            options.upper[1] = options.upper[4] = 5;
            options.lower[2] = options.lower[5] = (x[1] - x[0]) / 8;
            options.upper[2] = options.upper[5] = 10;
            for (std::size_t k = 0; k < patterns; ++k) {
                const double t = double(k) / double(patterns - 1);
                const double params[8] = {1.5 + t,       -1.9 + 0.7 * t, 0.5 + 0.2 * t,
                                          2.4 - 0.8 * t, 1.2 + 0.7 * t,  0.7 - 0.15 * t,
                                          0.4,           0.02 * t - 0.01};
                std::vector<float> row(m);
                for (std::size_t j = 0; j < m; ++j) {
                    rows[k * m + j] =
                        batchfit::models::TwoLorentzian<double>::eval(double(x[j]), params);
                    row[j] = float(rows[k * m + j]);
                }
                const auto guessed =
                    batchfit::cpu::initial_guess<float, batchfit::models::TwoLorentzian>(x, row);
                if (guessed.status != batchfit::cpu::GuessStatus::success)
                    throw std::runtime_error(
                        "benchmark pattern needs an observable two-peak guess");
                std::copy(guessed.parameters.begin(), guessed.parameters.end(),
                          initial.data() + k * 8);
                reference.push_back(
                    batchfit::cpu::fit_single<float, batchfit::models::TwoLorentzian>(
                        x, row, guessed.parameters, options));
                if (reference.back().status != batchfit::FitStatus::converged)
                    throw std::runtime_error("CPU benchmark pattern did not converge");
            }
            {
                batchfit::npy::Writer writer(directory / "x.npy", {m});
                writer.append(x);
                writer.finish();
            }
            batchfit::npy::Writer y(directory / "y.npy", {n, m},
                                    float32 ? batchfit::npy::Dtype::float32
                                            : batchfit::npy::Dtype::float64);
            batchfit::npy::Writer init(directory / "initial.npy", {n, 8});
            const std::size_t block = 128;
            std::vector<double> y_block(block * m);
            std::vector<float> float_block(float32 ? block * m : 0);
            std::vector<float> init_block(block * 8);
            for (std::size_t first = 0; first < n;) {
                const auto count = std::min(block, n - first);
                for (std::size_t i = 0; i < count; ++i) {
                    const auto pattern = (first + i) % patterns;
                    std::copy_n(rows.data() + pattern * m, m, y_block.data() + i * m);
                    std::copy_n(initial.data() + pattern * 8, 8, init_block.data() + i * 8);
                }
                if (float32) {
                    std::transform(y_block.data(), y_block.data() + count * m, float_block.begin(),
                                   [](double value) { return float(value); });
                    y.append(std::span<const float>(float_block).first(count * m));
                } else
                    y.append(std::span<const double>(y_block).first(count * m));
                init.append(std::span<const float>(init_block).first(count * 8));
                first += count;
            }
            y.finish();
            init.finish();
            struct stat status {};
            if (stat((directory / "y.npy").c_str(), &status) != 0)
                throw std::runtime_error("cannot stat benchmark file");
            std::cout << "dataset_bytes=" << std::filesystem::file_size(directory / "y.npy")
                      << " allocated_disk_bytes=" << std::uintmax_t(status.st_blocks) * 512
                      << " patterns=64 dtype=" << (float32 ? "float32" : "float64") << " N=" << n
                      << " M=" << m << '\n';
        } catch (...) {
            std::filesystem::remove_all(directory);
            throw;
        }
    }
    ~Dataset() {
        std::error_code error;
        std::filesystem::remove_all(directory, error);
        if (error)
            std::cerr << "benchmark cleanup failed: " << error.message() << '\n';
    }
};
} // namespace
int main(int argc, char **argv) {
    try {
        bool smoke = false, float32 = false;
        for (int i = 1; i < argc; ++i) {
            if (std::string_view(argv[i]) == "--smoke")
                smoke = true;
            else if (std::string_view(argv[i]) == "--float32")
                float32 = true;
            else
                throw std::invalid_argument(
                    "usage: batchfit_pipeline_baseline [--smoke] [--float32]");
        }
        const auto generate_begin = std::chrono::steady_clock::now();
        Dataset dataset(smoke, float32);
        std::cout << "generation_seconds="
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                                   generate_begin)
                         .count()
                  << '\n';
        batchfit::metal::Context context;
        const auto limits = context.limits();
        std::cout << "maxBufferLength=" << limits.max_buffer_bytes
                  << " recommendedWorkingSet=" << limits.recommended_working_set_bytes << '\n';
        const auto chunk = smoke
                               ? 31
                               : batchfit::metal::choose_chunk_fits(limits, dataset.n, dataset.m, 8,
                                                                    dataset.item_bytes, 4, {0, 3});
        for (unsigned depth : {1, 3}) {
            const std::filesystem::path trace =
                depth == 3 ? (smoke     ? ".agent-local/pipeline-smoke.json"
                              : float32 ? "profiling/pipeline_float32_events.json"
                                        : "profiling/pipeline_events.json")
                           : "";
            batchfit::metal::detail::Timeline timeline(trace);
            std::size_t checked = 0;
            rusage usage_begin{};
            if (getrusage(RUSAGE_SELF, &usage_begin) != 0)
                throw std::runtime_error("getrusage before run failed");
            const auto start = std::chrono::steady_clock::now();
            const auto stats = batchfit::metal::fit_npy<batchfit::models::TwoLorentzian>(
                context,
                {dataset.directory / "x.npy", dataset.directory / "y.npy",
                 dataset.directory / "initial.npy"},
                [&](std::size_t first, const batchfit::metal::Result &result) {
                    for (std::size_t i = 0; i < result.status.size(); ++i) {
                        const auto &expected =
                            dataset.reference[(first + i) % dataset.reference.size()];
                        if (result.status[i] != expected.status)
                            throw std::runtime_error("large-file status parity");
                        for (unsigned k = 0; k < 8; ++k)
                            check_near(result.parameters[i * 8 + k], expected.parameters[k], 2e-4,
                                       2e-4, "large-file parameter parity");
                        check_near(result.chi_squared[i], expected.chi_squared, 1e-6, 2e-5,
                                   "large-file cost parity");
                    }
                    checked += result.status.size();
                },
                dataset.options, {chunk, depth},
                [&](const auto &event) { timeline.append(event); });
            timeline.finish();
            const auto wall =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            rusage usage_end{};
            if (getrusage(RUSAGE_SELF, &usage_end) != 0)
                throw std::runtime_error("getrusage after run failed");
            if (checked != dataset.n)
                throw std::runtime_error("large-file output coverage mismatch");
            std::cout << std::setprecision(9) << "depth=" << depth << " fits=" << stats.fits
                      << " chunks=" << stats.chunks << " chunk_fits=" << stats.chunk_fits
                      << " max_in_flight=" << stats.max_in_flight
                      << " max_mapping_bytes=" << stats.max_mapped_bytes
                      << " shared_payload_bytes=" << stats.shared_payload_bytes
                      << " wall_seconds=" << wall
                      << " fits_per_second=" << double(stats.fits) / wall << " input_GB_per_second="
                      << double(dataset.n * dataset.m * dataset.item_bytes) / wall / 1e9
                      << " prepare_seconds=" << stats.prepare_seconds
                      << " convert_seconds=" << stats.convert_seconds
                      << " initial_seconds=" << stats.initial_seconds
                      << " transpose_seconds=" << stats.transpose_seconds
                      << " readback_seconds=" << stats.readback_seconds
                      << " gpu_sum_seconds=" << stats.gpu_seconds << '\n';
            std::cout << "cpu_user_seconds="
                      << double(usage_end.ru_utime.tv_sec - usage_begin.ru_utime.tv_sec) +
                             double(usage_end.ru_utime.tv_usec - usage_begin.ru_utime.tv_usec) *
                                 1e-6
                      << " cpu_system_seconds="
                      << double(usage_end.ru_stime.tv_sec - usage_begin.ru_stime.tv_sec) +
                             double(usage_end.ru_stime.tv_usec - usage_begin.ru_stime.tv_usec) *
                                 1e-6
                      << " minor_faults=" << usage_end.ru_minflt - usage_begin.ru_minflt
                      << " major_faults=" << usage_end.ru_majflt - usage_begin.ru_majflt << '\n';
            if (smoke && std::filesystem::exists(trace))
                std::filesystem::remove(trace);
        }
        rusage usage{};
        if (getrusage(RUSAGE_SELF, &usage) != 0)
            throw std::runtime_error("getrusage failed");
        std::cout << "peak_rss_bytes=" << usage.ru_maxrss << '\n';
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
