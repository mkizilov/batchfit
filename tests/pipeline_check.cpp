#include "pipeline_checks.hpp"
#include <iostream>
int main() {
    try {
        pipeline_checks::all();
        std::cout << "Mapped streaming async/ring pipeline matches CPU for all "
                     "models/variants/depths; errors drain safely\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
