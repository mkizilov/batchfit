#include "metal_lm_checks.hpp"

#include <iostream>

#ifdef BATCHFIT_NATIVE_METAL_TESTS
void verify_native_shader_diagnostics();
#endif

int main() {
    try {
        metal_lm_checks::all();
#ifdef BATCHFIT_NATIVE_METAL_TESTS
        verify_native_shader_diagnostics();
#endif
        std::cout << "Metal LM checks passed: 2832 CPU/GPU fits, all models, SIMD tails, "
                     "point-major layout and failure statuses\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
