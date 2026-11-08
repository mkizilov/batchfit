#include <batchfit/npy.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace batchfit::npy {
Mapping::Mapping(void *base, std::size_t bytes, std::size_t delta, std::size_t count, Dtype dtype)
    : base_(base), bytes_(bytes), delta_(delta), count_(count), dtype_(dtype) {}
Mapping::Mapping(Mapping &&other) noexcept
    : base_(std::exchange(other.base_, nullptr)), bytes_(std::exchange(other.bytes_, 0)),
      delta_(std::exchange(other.delta_, 0)), count_(std::exchange(other.count_, 0)),
      dtype_(other.dtype_) {}
Mapping &Mapping::operator=(Mapping &&other) noexcept {
    if (this != &other) {
        reset();
        base_ = std::exchange(other.base_, nullptr);
        bytes_ = std::exchange(other.bytes_, 0);
        delta_ = std::exchange(other.delta_, 0);
        count_ = std::exchange(other.count_, 0);
        dtype_ = other.dtype_;
    }
    return *this;
}
Mapping::~Mapping() { reset(); }
void Mapping::reset() noexcept {
#if defined(__unix__) || defined(__APPLE__)
    if (base_ && munmap(base_, bytes_) != 0)
        std::cerr << "NPY munmap failed\n";
#endif
    base_ = nullptr;
    bytes_ = 0;
}
void Mapping::copy_to(std::span<float> output) const {
    if (output.size() != count_)
        throw std::invalid_argument("mapped NPY output length mismatch");
    if (count_ == 0)
        return;
    const auto *data = static_cast<const std::byte *>(base_) + delta_;
    for (std::size_t i = 0; i < count_; ++i) {
        if (dtype_ == Dtype::float32) {
            float value;
            std::memcpy(&value, data + i * 4, 4);
            if constexpr (std::endian::native != std::endian::little) {
                auto bytes = std::bit_cast<std::array<std::byte, 4>>(value);
                std::reverse(bytes.begin(), bytes.end());
                value = std::bit_cast<float>(bytes);
            }
            output[i] = value;
        } else {
            double value;
            std::memcpy(&value, data + i * 8, 8);
            if constexpr (std::endian::native != std::endian::little) {
                auto bytes = std::bit_cast<std::array<std::byte, 8>>(value);
                std::reverse(bytes.begin(), bytes.end());
                value = std::bit_cast<double>(bytes);
            }
            output[i] = static_cast<float>(value);
        }
    }
}
MappedReader::MappedReader(const std::filesystem::path &path)
    : path_(path), header_(inspect(path)) {
    if (header_.dtype != Dtype::float32 && header_.dtype != Dtype::float64)
        throw std::invalid_argument("mapped NPY inputs require float32/float64");
#if defined(__unix__) || defined(__APPLE__)
    fd_ = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0)
        throw std::runtime_error("NPY mmap open failed: " + path.string());
#else
    throw std::runtime_error("NPY mapping requires a POSIX platform");
#endif
}
MappedReader::~MappedReader() {
#if defined(__unix__) || defined(__APPLE__)
    if (fd_ >= 0 && close(fd_) != 0)
        std::cerr << "NPY mapped file close failed: " << path_ << '\n';
#endif
}
Mapping MappedReader::window(std::size_t first, std::size_t count) const {
    if (first > header_.elements || count > header_.elements - first)
        throw std::out_of_range("NPY mapped window exceeds shape");
    if (count == 0)
        return Mapping(nullptr, 0, 0, 0, header_.dtype);
#if defined(__unix__) || defined(__APPLE__)
    struct stat status {};
    const auto expected = header_.offset + header_.elements * header_.item_bytes;
    if (fstat(fd_, &status) != 0 || status.st_size < 0 ||
        std::uintmax_t(status.st_size) != expected)
        throw std::runtime_error("NPY mapped file changed size: " + path_.string());
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0)
        throw std::runtime_error("NPY mmap cannot query page size");
    const auto offset = header_.offset + first * header_.item_bytes;
    const auto aligned = offset / std::size_t(page_size) * std::size_t(page_size),
               delta = offset - aligned;
    const auto bytes = count * header_.item_bytes + delta;
    void *base = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd_, static_cast<off_t>(aligned));
    if (base == MAP_FAILED)
        throw std::runtime_error("NPY mmap failed: " + path_.string());
    if (madvise(base, bytes, MADV_SEQUENTIAL) != 0 || madvise(base, bytes, MADV_WILLNEED) != 0) {
        const auto message = "NPY sequential mapping advice failed: " + path_.string();
        if (munmap(base, bytes) != 0)
            std::cerr << "NPY mapping cleanup failed\n";
        throw std::runtime_error(message);
    }
    return Mapping(base, bytes, delta, count, header_.dtype);
#else
    throw std::runtime_error("NPY mapping requires POSIX");
#endif
}
void MappedReader::prefetch(std::size_t first, std::size_t count) const {
    if (first > header_.elements || count > header_.elements - first)
        throw std::out_of_range("NPY prefetch exceeds shape");
    if (count == 0)
        return;
    const auto offset = header_.offset + first * header_.item_bytes;
    const auto bytes = count * header_.item_bytes;
#if defined(__APPLE__)
    radvisory advice{};
    advice.ra_offset = static_cast<off_t>(offset);
    advice.ra_count =
        static_cast<int>(std::min(bytes, std::size_t(std::numeric_limits<int>::max())));
    if (fcntl(fd_, F_RDADVISE, &advice) != 0)
        throw std::runtime_error("NPY read-ahead advice failed: " + path_.string());
#elif defined(__unix__)
    const int error = posix_fadvise(fd_, static_cast<off_t>(offset), static_cast<off_t>(bytes),
                                    POSIX_FADV_WILLNEED);
    if (error != 0)
        throw std::runtime_error("NPY read-ahead advice failed: " + path_.string());
#else
    throw std::runtime_error("NPY prefetch requires POSIX");
#endif
}
} // namespace batchfit::npy
