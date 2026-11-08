#include "initial_guess_checks.hpp"

int main() {
    try {
        initial_guess_checks::all<float>();
        initial_guess_checks::all<double>();
        std::cout
            << "Initial-guess checks passed: median/centroid/FWHM, paired peaks, input validation, "
               "24000 noisy guess-to-fit cases\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
