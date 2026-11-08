#include "synthetic_checks.hpp"

int main() {
    try {
        synthetic_checks::all();
        std::cout
            << "Synthetic checks passed: 12 model/noise roundtrips, chunk/seed reproducibility, "
               "Gaussian/Poisson moments and SNR, independent NPY bytes, invalid inputs\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
