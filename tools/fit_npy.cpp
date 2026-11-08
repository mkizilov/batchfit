#include "timeline.hpp"
#include <batchfit/metal/pipeline.hpp>
#include <charconv>
#include <iostream>

namespace {
struct Arguments {
    std::string model = "gaussian";
    batchfit::metal::NpyInput files;
    std::filesystem::path output, timeline;
    batchfit::metal::PipelineOptions pipeline;
};
std::size_t integer(std::string_view text) {
    std::size_t value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
        throw std::invalid_argument("invalid integer");
    return value;
}
template <template <typename> class Model> void run(const Arguments &args) {
    constexpr unsigned p = Model<float>::parameter_count;
    const auto header = batchfit::npy::inspect(args.files.y);
    if (header.shape.size() != 2)
        throw std::invalid_argument("y must have shape [N,M]");
    batchfit::metal::Context context;
    if (!std::filesystem::create_directory(args.output))
        throw std::runtime_error("output directory must not already exist");
    try {
        const auto begin = std::chrono::steady_clock::now();
        batchfit::npy::Writer parameters(args.output / "parameters.npy", {header.shape[0], p});
        batchfit::npy::Writer chi(args.output / "chi_squared.npy", {header.shape[0]});
        batchfit::npy::Writer status(args.output / "status.npy", {header.shape[0]},
                                     batchfit::npy::Dtype::uint32);
        batchfit::npy::Writer iterations(args.output / "iterations.npy", {header.shape[0]},
                                         batchfit::npy::Dtype::uint32);
        batchfit::metal::detail::Timeline timeline(args.timeline);
        const auto stats = batchfit::metal::fit_npy<Model>(
            context, args.files,
            [&](std::size_t, const batchfit::metal::Result &result) {
                parameters.append(result.parameters);
                chi.append(result.chi_squared);
                iterations.append(result.iterations);
                std::vector<std::uint32_t> codes(result.status.size());
                std::transform(result.status.begin(), result.status.end(), codes.begin(),
                               [](auto value) { return std::uint32_t(value); });
                status.append(codes);
            },
            {}, args.pipeline, [&](const auto &event) { timeline.append(event); });
        parameters.finish();
        chi.finish();
        status.finish();
        iterations.finish();
        timeline.finish();
        const auto wall =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        std::cout << "fits=" << stats.fits << " chunks=" << stats.chunks
                  << " chunk_fits=" << stats.chunk_fits << " peak_pending=" << stats.max_in_flight
                  << " failed_guesses=" << stats.failed_guesses
                  << " pipeline_seconds=" << stats.wall_seconds << " wall_seconds=" << wall
                  << " fits_per_second=" << double(stats.fits) / wall << '\n';
    } catch (...) {
        std::filesystem::remove_all(args.output);
        throw;
    }
}
} // namespace
int main(int argc, char **argv) {
    try {
        Arguments args;
        for (int i = 1; i < argc; ++i) {
            const std::string_view key = argv[i];
            if (key == "--help") {
                std::cout
                    << "batchfit_fit_npy --x x.npy --y y.npy [--initial initial.npy] --output "
                       "NEW_DIR [--model gaussian|lorentzian|pseudo_voigt|two_lorentzian] "
                       "[--chunk-fits 0] [--in-flight 3] [--timeline profiling/events.json]\n";
                return 0;
            }
            if (i + 1 == argc)
                throw std::invalid_argument("missing option value");
            const std::string value = argv[++i];
            if (key == "--x")
                args.files.x = value;
            else if (key == "--y")
                args.files.y = value;
            else if (key == "--initial")
                args.files.initial = value;
            else if (key == "--output")
                args.output = value;
            else if (key == "--model")
                args.model = value;
            else if (key == "--chunk-fits")
                args.pipeline.chunk_fits = integer(value);
            else if (key == "--in-flight") {
                const auto depth = integer(value);
                if (depth < 1 || depth > 3)
                    throw std::invalid_argument("in-flight must be 1..3");
                args.pipeline.in_flight = unsigned(depth);
            } else if (key == "--timeline")
                args.timeline = value;
            else
                throw std::invalid_argument("unknown option");
        }
        if (args.files.x.empty() || args.files.y.empty() || args.output.empty())
            throw std::invalid_argument("--x, --y and --output are required");
        if (args.model == "gaussian")
            run<batchfit::models::Gaussian>(args);
        else if (args.model == "lorentzian")
            run<batchfit::models::Lorentzian>(args);
        else if (args.model == "pseudo_voigt")
            run<batchfit::models::PseudoVoigt>(args);
        else if (args.model == "two_lorentzian")
            run<batchfit::models::TwoLorentzian>(args);
        else
            throw std::invalid_argument("unknown model");
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
