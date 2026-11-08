#pragma once

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <span>
#include <vector>

namespace batchfit::npy {
// NPY v1.0, little-endian float32, C order, rank 1 or 2 only.
class Writer {
  public:
    Writer(const std::filesystem::path &path, const std::vector<std::size_t> &shape);
    void append(std::span<const float> values);
    void finish();

  private:
    std::filesystem::path path_;
    std::ofstream stream_;
    std::size_t remaining_;
};

class Reader {
  public:
    explicit Reader(const std::filesystem::path &path);
    const std::vector<std::size_t> &shape() const { return shape_; }
    void read(std::span<float> values);

  private:
    std::filesystem::path path_;
    std::ifstream stream_;
    std::vector<std::size_t> shape_;
    std::size_t remaining_;
};
} // namespace batchfit::npy
