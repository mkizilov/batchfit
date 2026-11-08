#include "numerics_checks.hpp"
#include <iostream>
int main() {
    try {
        numerics_checks::cpu();
        std::cout << "Numerics: coordinates, compensation, scaled solve and CPU pathology statuses "
                     "passed\n";
#if BATCHFIT_HAS_METAL
        numerics_checks::gpu();
        std::cout << "Numerics: safe/fast Metal, every kernel and option combination passed "
                     "pathology guards\n";
#endif
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
