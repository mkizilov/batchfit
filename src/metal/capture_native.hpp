#pragma once

#include "capture_path.hpp"

namespace batchfit::metal::detail {
// Test/profile SDK driver counterpart. Production uses MTL::CaptureManager.
class NativeCapture {
  public:
    explicit NativeCapture(id<MTLCommandQueue> queue) {
        if (!path_.enabled())
            return;
        manager_ = [MTLCaptureManager sharedCaptureManager];
        if (!manager_ || ![manager_ supportsDestination:MTLCaptureDestinationGPUTraceDocument])
            throw std::runtime_error(
                "GPU trace document capture unsupported; set MTL_CAPTURE_ENABLED=1 before launch");
        if (manager_.isCapturing)
            throw std::runtime_error("another GPU capture is already active");
        MTLCaptureDescriptor *descriptor = [[MTLCaptureDescriptor alloc] init];
        descriptor.captureObject = queue;
        descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
        descriptor.outputURL =
            [NSURL fileURLWithPath:[NSString stringWithUTF8String:path_.path().c_str()]];
        NSError *error = nil;
        if (![manager_ startCaptureWithDescriptor:descriptor error:&error])
            throw std::runtime_error(
                "start GPU capture " + path_.path().string() + ": " +
                (error ? error.localizedDescription.UTF8String : "no error detail"));
        active_ = true;
    }
    ~NativeCapture() {
        if (active_)
            [manager_ stopCapture];
    }
    void finish() {
        if (!active_)
            return;
        [manager_ stopCapture];
        if (manager_.isCapturing)
            throw std::runtime_error("stop GPU capture failed");
        active_ = false;
        path_.commit();
    }

  private:
    CapturePath path_;
    MTLCaptureManager *manager_ = nil;
    bool active_ = false;
};
} // namespace batchfit::metal::detail
