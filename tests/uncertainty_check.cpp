#include "uncertainty_checks.hpp"
#include <iostream>
int main() {
    try {
        uncertainty_checks::cpu();
#if BATCHFIT_HAS_METAL
        uncertainty_checks::gpu();
#endif
        std::cout
            << "Optional covariance algebra, DOF, diagnostics, storage and CPU/GPU parity passed\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
