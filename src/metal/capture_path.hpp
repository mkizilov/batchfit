#pragma once

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace batchfit::metal::detail {
inline bool capture_requested() {
    const char *value = std::getenv("BATCHFIT_GPU_CAPTURE");
    if (!value || std::string_view(value).empty() || std::string_view(value) == "0")
        return false;
    if (std::string_view(value) != "1")
        throw std::invalid_argument("BATCHFIT_GPU_CAPTURE must be 0 or 1");
    return true;
}

inline void prepare_capture_environment() {
    if (!capture_requested())
        return;
    const char *existing = std::getenv("MTL_CAPTURE_ENABLED");
    if (existing && std::string_view(existing) != "1")
        throw std::runtime_error(
            "BATCHFIT_GPU_CAPTURE requires MTL_CAPTURE_ENABLED=1 before Metal initialization");
    if (!existing && setenv("MTL_CAPTURE_ENABLED", "1", 0) != 0)
        throw std::runtime_error("could not enable process-local Metal capture");
}

class CapturePath {
  public:
    CapturePath() {
        if (!capture_requested())
            return;
        bool expected = false;
        if (!claimed_.compare_exchange_strong(expected, true))
            return;
        try {
            path_ = std::filesystem::current_path() / "profiling/kernel-v1.gputrace";
            if (std::filesystem::exists(path_))
                throw std::runtime_error("refusing to overwrite GPU trace: " + path_.string());
            std::filesystem::create_directories(path_.parent_path());
        } catch (...) {
            claimed_ = false;
            throw;
        }
    }
    CapturePath(const CapturePath &) = delete;
    CapturePath &operator=(const CapturePath &) = delete;
    ~CapturePath() {
        if (path_.empty() || committed_)
            return;
        std::error_code error;
        std::filesystem::remove_all(path_, error);
        if (error)
            std::cerr << "failed to remove incomplete GPU trace " << path_ << ": "
                      << error.message() << '\n';
        claimed_ = false;
    }
    bool enabled() const { return !path_.empty(); }
    const std::filesystem::path &path() const { return path_; }
    void commit() {
        if (!std::filesystem::exists(path_))
            throw std::runtime_error("Metal capture did not produce " + path_.string());
        committed_ = true;
    }

  private:
    inline static std::atomic<bool> claimed_{false};
    std::filesystem::path path_;
    bool committed_ = false;
};
} // namespace batchfit::metal::detail
