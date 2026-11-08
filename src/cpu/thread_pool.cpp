#include <batchfit/cpu/thread_pool.hpp>

#include <algorithm>
#include <stdexcept>

namespace batchfit::cpu {
namespace {
thread_local ThreadPool *current_pool = nullptr;
constexpr std::size_t grain = 64;
} // namespace

ThreadPool::ThreadPool(std::size_t workers) {
    if (workers == 0)
        throw std::invalid_argument("ThreadPool requires at least one worker");
    workers_.reserve(workers);
    for (std::size_t i = 0; i < workers; ++i)
        workers_.emplace_back([this](std::stop_token stop) { worker(stop); });
}

ThreadPool::~ThreadPool() {
    for (auto &thread : workers_)
        thread.request_stop();
    ready_.notify_all();
}

void ThreadPool::parallel_for(std::size_t count,
                              const std::function<void(std::size_t, std::size_t)> &work) {
    if (current_pool == this)
        throw std::logic_error("recursive submission on the same ThreadPool");
    if (!work)
        throw std::invalid_argument("ThreadPool requires a callback");
    if (count == 0)
        return;
    std::lock_guard submission(submission_);
    std::unique_lock lock(mutex_);
    work_ = work;
    count_ = count;
    chunks_ = (count - 1) / grain + 1;
    next_chunk_.store(0, std::memory_order_relaxed);
    cancelled_.store(false, std::memory_order_relaxed);
    failure_ = nullptr;
    remaining_ = workers_.size();
    ++generation_;
    ready_.notify_all();
    done_.wait(lock, [this] { return remaining_ == 0; });
    work_ = {};
    if (failure_)
        std::rethrow_exception(failure_);
}

void ThreadPool::worker(std::stop_token stop) {
    current_pool = this;
    std::size_t observed = 0;
    std::unique_lock lock(mutex_);
    while (ready_.wait(lock, stop, [&] { return generation_ != observed; })) {
        if (stop.stop_requested())
            return;
        observed = generation_;
        lock.unlock();
        try {
            while (!cancelled_.load(std::memory_order_relaxed)) {
                const auto chunk = next_chunk_.fetch_add(1, std::memory_order_relaxed);
                if (chunk >= chunks_)
                    break;
                const auto begin = chunk * grain;
                work_(begin, begin + std::min(grain, count_ - begin));
            }
        } catch (...) {
            std::lock_guard failure_lock(mutex_);
            if (!failure_)
                failure_ = std::current_exception();
            cancelled_.store(true, std::memory_order_relaxed);
        }
        lock.lock();
        if (--remaining_ == 0)
            done_.notify_one();
    }
}
} // namespace batchfit::cpu
