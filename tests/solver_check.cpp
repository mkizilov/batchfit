#include "solver_checks.hpp"

#include <iostream>

int main() {
    try {
        solver_checks::all<float>();
        solver_checks::all<double>();
        std::cout << "CPU LM checks passed: 768 float/double synthetic fits, all models/noises, "
                     "Cholesky, bounds, stopping criteria, rejection and every status\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
