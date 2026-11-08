#include "cpu_workload.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <sys/resource.h>

namespace {
template <template <typename> class Model>
void measure(const char *name, std::size_t count, std::size_t points, bool smoke) {
    batchfit_bench::Workload<Model> data(count, points);
    std::uint64_t serial_hash = 0;
    const unsigned repetitions = smoke ? 1 : 3;
    for (const unsigned workers : {1U, smoke ? 2U : batchfit_bench::all_cores()}) {
        batchfit::cpu::ThreadPool pool(workers);
        data.warmup(pool);
        std::array<double, 3> times{};
        batchfit_bench::Summary summary;
        for (unsigned repetition = 0; repetition < repetitions; ++repetition) {
            const auto start = std::chrono::steady_clock::now();
            const auto result = data.run(pool);
            times[repetition] =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            summary = batchfit_bench::summarize(result);
            if (workers == 1 && repetition == 0)
                serial_hash = summary.hash;
            else if (summary.hash != serial_hash)
                throw std::runtime_error("serial/parallel/repetition result hash mismatch");
        }
        std::sort(times.begin(), times.begin() + repetitions);
        const double seconds = times[repetitions / 2];
        std::cout
            << name << ',' << count << ',' << points << ',' << workers << ',' << repetitions << ','
            << std::setprecision(9) << times.front() << ',' << seconds << ','
            << times[repetitions - 1] << ',' << double(count) / seconds << ',' << summary.converged
            << ',' << data.fallback_guesses << ',' << summary.mean_iterations << ',' << summary.hash
            << std::endl; // NOLINT(performance-avoid-endl): stream untimed completed measurements
    }
}
} // namespace

int main(int argc, char **argv) {
    try {
        const bool smoke = argc == 2 && std::string_view(argv[1]) == "--smoke";
        if (argc != 1 && !smoke)
            throw std::invalid_argument("usage: batchfit_cpu_baseline [--smoke]");
        std::cout << "model,N,M,workers,repetitions,seconds_min,seconds_median,seconds_max,fits_"
                     "per_second_median,converged,fallbacks,mean_"
                     "iterations,result_hash\n";
        for (auto points :
             (smoke ? std::vector<unsigned>{32} : std::vector<unsigned>{32, 128, 512, 2048})) {
            const auto n = smoke ? 128U : 100000U;
            measure<batchfit::models::Gaussian>("Gaussian", n, points, smoke);
            measure<batchfit::models::Lorentzian>("Lorentzian", n, points, smoke);
            measure<batchfit::models::PseudoVoigt>("PseudoVoigt", n, points, smoke);
            measure<batchfit::models::TwoLorentzian>("TwoLorentzian", n, points, smoke);
        }
        rusage usage{};
        if (getrusage(RUSAGE_SELF, &usage) != 0)
            throw std::runtime_error("getrusage failed");
#if defined(__APPLE__)
        std::cout << "peak_rss_bytes=" << usage.ru_maxrss << '\n';
#else
        std::cout << "peak_rss_bytes=" << usage.ru_maxrss * 1024ULL << '\n';
#endif
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
