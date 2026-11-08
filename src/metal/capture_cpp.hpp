#pragma once

#include "capture_path.hpp"

namespace batchfit::metal::detail {
class Capture {
  public:
    explicit Capture(MTL::CommandQueue *queue) {
        if (!path_.enabled())
            return;
        manager_ = NS::RetainPtr(MTL::CaptureManager::sharedCaptureManager());
        require(manager_.get() != nullptr, "get capture manager");
        require(manager_->supportsDestination(MTL::CaptureDestinationGPUTraceDocument),
                "GPU trace document capture unsupported (set MTL_CAPTURE_ENABLED=1 before launch)");
        require(!manager_->isCapturing(), "another GPU capture is already active");
        const auto descriptor = NS::TransferPtr(MTL::CaptureDescriptor::alloc()->init());
        require(descriptor.get() != nullptr, "allocate capture descriptor");
        descriptor->setCaptureObject(queue);
        descriptor->setDestination(MTL::CaptureDestinationGPUTraceDocument);
        descriptor->setOutputURL(NS::URL::fileURLWithPath(
            NS::String::string(path_.path().c_str(), NS::UTF8StringEncoding)));
        NS::Error *error = nullptr;
        require(manager_->startCapture(descriptor.get(), &error),
                "start GPU capture " + path_.path().string(), error);
        active_ = true;
    }
    ~Capture() {
        if (active_)
            manager_->stopCapture();
    }
    bool enabled() const { return active_; }
    void finish() {
        if (!active_)
            return;
        manager_->stopCapture();
        require(!manager_->isCapturing(), "stop GPU capture");
        active_ = false;
        path_.commit();
    }

  private:
    CapturePath path_;
    NS::SharedPtr<MTL::CaptureManager> manager_;
    bool active_ = false;
};
} // namespace batchfit::metal::detail
