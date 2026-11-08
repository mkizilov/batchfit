#include "model_checks.hpp"

#include <iostream>

int main() {
    model_checks::all<float>();
    model_checks::all<double>();
    std::cout
        << "Model checks passed: hand values/metadata and 72000 finite-difference derivatives "
           "across 16000 float/double parameter sets\n";
}
