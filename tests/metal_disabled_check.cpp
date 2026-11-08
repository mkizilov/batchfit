#include <batchfit/batchfit.hpp>

#include <iostream>
#include <string>

int main() {
    try {
        batchfit::metal::Context context;
    } catch (const std::runtime_error &error) {
        if (std::string(error.what()).find("without BATCHFIT_METAL") != std::string::npos) {
            std::cout << "Disabled Metal reports unavailability without a CPU fallback\n";
            return 0;
        }
    }
    return 1;
}
