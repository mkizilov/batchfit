#include "batch_checks.hpp"

#include <iostream>

int main() {
    try {
        batch_checks::pool_contract();
        batch_checks::models<float>();
        batch_checks::models<double>();
        std::cout
            << "Batch checks passed: concurrent jthreads, lifecycle/errors/reuse, and bitwise "
               "single-fit equivalence for all models/scalars with bad-row isolation\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
