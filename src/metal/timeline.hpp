#pragma once
#include <batchfit/metal/pipeline.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace batchfit::metal::detail {
// Stream Chrome/Perfetto events with a documented common Mach clock. This is
// application instrumentation, never a substitute labeled as Metal System Trace.
class Timeline {
  public:
    explicit Timeline(std::filesystem::path path) : path_(std::move(path)) {
        if (path_.empty())
            return;
        if (std::filesystem::exists(path_))
            throw std::runtime_error("refusing to overwrite timeline: " + path_.string());
        if (!path_.parent_path().empty())
            std::filesystem::create_directories(path_.parent_path());
        stream_.open(path_);
        if (!stream_)
            throw std::runtime_error("cannot create timeline " + path_.string());
        created_ = true;
        stream_ << "{\"displayTimeUnit\":\"ms\",\"provenance\":\"batchfit application/Mach and "
                   "command-buffer timestamps; NOT Metal System Trace\",\"traceEvents\":[\n";
    }
    ~Timeline() {
        if (created_ && !finished_) {
            stream_.close();
            std::error_code error;
            std::filesystem::remove(path_, error);
            if (error)
                std::cerr << "timeline cleanup failed: " << error.message() << '\n';
        }
    }
    void append(const PipelineEvent &event) {
        if (!created_)
            return;
        if (first_)
            origin_ = event.prefetch_begin;
        emit("CPU read-ahead request", 1, event.prefetch_begin, event.prefetch_end, event);
        emit("CPU preparation", 1, event.prepare_begin, event.prepare_end, event);
        emit("CPU map/convert", 1, event.prepare_begin, event.convert_end, event);
        emit("CPU initial", 1, event.convert_end, event.initial_end, event);
        emit("CPU transpose", 1, event.initial_end, event.prepare_end, event);
        emit("CPU submit", 1, event.submit_begin, event.submit_end, event);
        emit("Metal GPU", 3, event.gpu_begin, event.gpu_end, event);
        emit("CPU readback/consumer", 2, event.readback_begin, event.readback_end, event);
    }
    void finish() {
        if (!created_)
            return;
        stream_ << "\n]}\n";
        stream_.close();
        if (!stream_)
            throw std::runtime_error("timeline flush failed: " + path_.string());
        finished_ = true;
    }

  private:
    void emit(const char *name, unsigned thread, double begin, double end,
              const PipelineEvent &event) {
        if (!first_)
            stream_ << ",\n";
        first_ = false;
        stream_ << std::setprecision(15) << "{\"name\":\"" << name
                << "\",\"ph\":\"X\",\"pid\":1,\"tid\":" << thread
                << ",\"ts\":" << (begin - origin_) * 1e6 << ",\"dur\":" << (end - begin) * 1e6
                << ",\"args\":{\"first_fit\":" << event.first_fit << ",\"fits\":" << event.fits
                << "}}";
        if (!stream_)
            throw std::runtime_error("timeline write failed");
    }
    std::filesystem::path path_;
    std::ofstream stream_;
    double origin_ = 0;
    bool created_ = false, finished_ = false, first_ = true;
};
} // namespace batchfit::metal::detail
