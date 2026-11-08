#include <batchfit/models.hpp>
#include <batchfit/synthetic.hpp>

#include <charconv>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using namespace batchfit::synthetic;

template <typename T> T number(std::string_view value) {
    T result{};
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()) {
        throw std::invalid_argument("invalid numeric argument: " + std::string(value));
    }
    return result;
}

Model model(std::string_view value) {
    if (value == "gaussian")
        return Model::gaussian;
    if (value == "lorentzian")
        return Model::lorentzian;
    if (value == "pseudo_voigt")
        return Model::pseudo_voigt;
    if (value == "two_lorentzian")
        return Model::two_lorentzian;
    throw std::invalid_argument("unknown model: " + std::string(value));
}

batchfit::models::ParameterName name(Model model, unsigned index) {
    using namespace batchfit::models;
    switch (model) {
    case Model::gaussian:
        return Gaussian<float>::parameter_name(index);
    case Model::lorentzian:
        return Lorentzian<float>::parameter_name(index);
    case Model::pseudo_voigt:
        return PseudoVoigt<float>::parameter_name(index);
    case Model::two_lorentzian:
        return TwoLorentzian<float>::parameter_name(index);
    }
    throw std::invalid_argument("unknown model");
}

void range(Options &options, const std::string &text) {
    const auto equal = text.find('=');
    const auto colon = text.find(':', equal == std::string::npos ? 0 : equal);
    if (equal == std::string::npos || colon == std::string::npos) {
        throw std::invalid_argument("range must be parameter=lower:upper: " + text);
    }
    for (unsigned k = 0; k < parameter_count(options.model); ++k) {
        if (text.substr(0, equal) ==
            batchfit::models::parameter_name_text(name(options.model, k))) {
            options.parameters[k] = {
                number<float>(std::string_view(text).substr(equal + 1, colon - equal - 1)),
                number<float>(std::string_view(text).substr(colon + 1))};
            return;
        }
    }
    throw std::invalid_argument("unknown parameter in range: " + text);
}
} // namespace

int main(int argc, const char *argv[]) {
    try {
        Options options;
        std::filesystem::path output;
        std::vector<std::string> ranges;
        for (int i = 1; i < argc; ++i) {
            const std::string_view flag(argv[i]);
            if (flag == "--help") {
                std::cout << "batchfit_generate --output DIRECTORY [--model "
                             "gaussian|lorentzian|pseudo_voigt|two_lorentzian]\n"
                             "  [--fits N] [--points M] [--noise none|gaussian|poisson] [--snr "
                             "LINEAR_RMS_RATIO]\n"
                             "  [--seed UINT64] [--chunk-fits N] [--x-min X] [--x-max X]\n"
                             "  [--range parameter=lower:upper] (repeatable; fixed values use "
                             "equal bounds)\n";
                return 0;
            }
            if (++i == argc)
                throw std::invalid_argument("missing value for " + std::string(flag));
            const std::string_view value(argv[i]);
            if (flag == "--output")
                output = value;
            else if (flag == "--model")
                options.model = model(value);
            else if (flag == "--fits")
                options.fits = number<std::size_t>(value);
            else if (flag == "--points")
                options.points = number<std::size_t>(value);
            else if (flag == "--snr")
                options.snr = number<double>(value);
            else if (flag == "--seed")
                options.seed = number<std::uint64_t>(value);
            else if (flag == "--chunk-fits")
                options.chunk_fits = number<std::size_t>(value);
            else if (flag == "--x-min")
                options.x_min = number<float>(value);
            else if (flag == "--x-max")
                options.x_max = number<float>(value);
            else if (flag == "--range")
                ranges.emplace_back(value);
            else if (flag == "--noise") {
                if (value == "none")
                    options.noise = Noise::none;
                else if (value == "gaussian")
                    options.noise = Noise::gaussian;
                else if (value == "poisson")
                    options.noise = Noise::poisson;
                else
                    throw std::invalid_argument("unknown noise: " + std::string(value));
            } else
                throw std::invalid_argument("unknown option: " + std::string(flag));
        }
        if (output.empty())
            throw std::invalid_argument("--output DIRECTORY is required");
        options.parameters = default_ranges(options.model);
        for (const auto &text : ranges)
            range(options, text);
        const auto start = std::chrono::steady_clock::now();
        save_dataset(output, options);
        const auto elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::cout << "Generated " << options.fits << " fits x " << options.points << " points in "
                  << elapsed << " s (" << double(options.fits) / elapsed
                  << " fits/s), seed=" << options.seed << ", chunk_fits=" << options.chunk_fits
                  << "\n";
    } catch (const std::exception &error) {
        std::cerr << "batchfit_generate: " << error.what() << '\n';
        return 2;
    }
}
