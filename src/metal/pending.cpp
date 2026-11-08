#include "completion.hpp"
#include <iostream>

namespace batchfit::metal {
bool PendingBatch::ready() const {
    if (!completion_)
        return false;
    std::lock_guard lock(completion_->mutex);
    return completion_->done;
}
Result PendingBatch::get() {
    if (!completion_)
        throw std::logic_error("empty Metal pending batch");
    std::unique_lock lock(completion_->mutex);
    completion_->changed.wait(lock, [&] { return completion_->done; });
    completion_->observed = true;
    if (!completion_->released) {
        owner_->busy.store(false);
        completion_->released = true;
    }
    if (completion_->failure)
        std::rethrow_exception(completion_->failure);
    return Result(owner_, completion_->timings);
}
void PendingBatch::drain() noexcept {
    if (!completion_)
        return;
    try {
        std::unique_lock lock(completion_->mutex);
        completion_->changed.wait(lock, [&] { return completion_->done; });
        if (!completion_->released) {
            owner_->busy.store(false);
            completion_->released = true;
        }
        if (completion_->failure && !completion_->observed) {
            completion_->observed = true;
            auto error = completion_->failure;
            lock.unlock();
            std::rethrow_exception(error);
        }
    } catch (const std::exception &error) {
        std::cerr << "unobserved Metal submission error: " << error.what() << '\n';
    } catch (...) {
        std::cerr << "unobserved nonstandard Metal submission error\n";
    }
}
PendingBatch::~PendingBatch() { drain(); }
PendingBatch &PendingBatch::operator=(PendingBatch &&other) noexcept {
    if (this != &other) {
        drain();
        owner_ = std::move(other.owner_);
        completion_ = std::move(other.completion_);
    }
    return *this;
}
} // namespace batchfit::metal
