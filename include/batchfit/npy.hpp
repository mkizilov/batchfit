#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <vector>

namespace batchfit::npy {
enum class Dtype { float32, float64, uint32 };
struct Header {
    std::vector<std::size_t> shape;
    Dtype dtype;
    std::size_t offset, elements, item_bytes;
};
Header inspect(const std::filesystem::path &path);

// NPY v1.0, little-endian C order, rank 1 or 2. Default API remains float32.
class Writer {
  public:
    Writer(const std::filesystem::path &path, const std::vector<std::size_t> &shape,
           Dtype dtype = Dtype::float32);
    void append(std::span<const float> values);
    void append(std::span<const double> values);
    void append(std::span<const std::uint32_t> values);
    void finish();

  private:
    std::filesystem::path path_;
    std::ofstream stream_;
    std::size_t remaining_;
    Dtype dtype_;
    template <typename T> void append_values(std::span<const T> values, Dtype dtype);
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

// Immutable float32/float64 inputs. Map only a requested element window; returned
// windows own their mapping and must not outlive/truncate the underlying file.
class Mapping {
  public:
    Mapping(Mapping &&other) noexcept;
    Mapping &operator=(Mapping &&other) noexcept;
    ~Mapping();
    Mapping(const Mapping &) = delete;
    Mapping &operator=(const Mapping &) = delete;
    void copy_to(std::span<float> output) const;
    std::size_t mapped_bytes() const { return bytes_; }

  private:
    Mapping(void *base, std::size_t bytes, std::size_t delta, std::size_t count, Dtype dtype);
    void reset() noexcept;
    void *base_;
    std::size_t bytes_, delta_, count_;
    Dtype dtype_;
    friend class MappedReader;
};
class MappedReader {
  public:
    explicit MappedReader(const std::filesystem::path &path);
    ~MappedReader();
    MappedReader(const MappedReader &) = delete;
    MappedReader &operator=(const MappedReader &) = delete;
    const Header &header() const { return header_; }
    Mapping window(std::size_t first, std::size_t count) const;
    void prefetch(std::size_t first, std::size_t count) const;

  private:
    std::filesystem::path path_;
    Header header_;
    int fd_ = -1;
};
} // namespace batchfit::npy
