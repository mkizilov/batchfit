#pragma once
#include <batchfit/metal/batch.hpp>
#include <condition_variable>
#include <exception>
#include <mutex>

namespace batchfit::metal::detail {
struct Completion {
    std::mutex mutex;
    std::condition_variable changed;
    bool done = false, observed = false, released = false;
    std::exception_ptr failure;
    Timings timings;
    std::shared_ptr<void> backend;
};
} // namespace batchfit::metal::detail
