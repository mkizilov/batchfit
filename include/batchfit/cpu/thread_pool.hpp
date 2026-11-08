#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <stop_token>
#include <thread>
#include <vector>

namespace batchfit::cpu {
class ThreadPool {
  public:
    explicit ThreadPool(std::size_t workers = 2);
    ~ThreadPool();
    ThreadPool(const ThreadPool &) = delete;
    ThreadPool &operator=(const ThreadPool &) = delete;
    std::size_t size() const noexcept { return workers_.size(); }

    // Synchronous, disjoint [begin,end) ranges. Concurrent submissions serialize.
    // Direct recursive submission on this pool is rejected, not deadlocked.
    void parallel_for(std::size_t count, const std::function<void(std::size_t, std::size_t)> &work);

  private:
    void worker(const std::stop_token &stop);
    std::mutex submission_, mutex_;
    std::condition_variable_any ready_;
    std::condition_variable done_;
    std::function<void(std::size_t, std::size_t)> work_;
    std::atomic<std::size_t> next_chunk_{0};
    std::atomic<bool> cancelled_{false};
    std::size_t count_ = 0, chunks_ = 0, generation_ = 0, remaining_ = 0;
    std::exception_ptr failure_;
    // Destroy/join workers before the synchronization objects, including on a
    // partially failed constructor. stop-aware waits make jthread cleanup safe.
    std::vector<std::jthread> workers_;
};
} // namespace batchfit::cpu
