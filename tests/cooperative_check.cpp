#include "cooperative_checks.hpp"
#include <iostream>
int main() {
    try {
        cooperative_checks::all();
        std::cout
            << "Cooperative variants/dispatcher pass CPU parity, tails, bounds and failures\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
