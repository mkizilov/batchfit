#include "model_checks.hpp"

#include <iostream>

int main() {
    try {
        model_checks::all<float>();
        model_checks::all<double>();
        std::cout
            << "Model checks passed: hand values/metadata and 72000 finite-difference derivatives "
               "across 16000 float/double parameter sets\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
