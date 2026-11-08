#include <batchfit/npy.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cstdint>
#include <cstring>
#include <limits>
#include <regex>
#include <stdexcept>
#include <string>

namespace batchfit::npy {
namespace {
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);

void require(bool valid, const std::filesystem::path &path, const char *operation) {
    if (!valid) {
        throw std::runtime_error("NPY " + path.string() + ": " + operation);
    }
}

std::size_t count(const std::vector<std::size_t> &shape, const std::filesystem::path &path,
                  std::size_t width) {
    require(shape.size() == 1 || shape.size() == 2, path, "only rank 1/2 arrays are supported");
    std::size_t result = 1;
    for (auto dimension : shape) {
        require(dimension == 0 || result <= std::numeric_limits<std::size_t>::max() / dimension,
                path, "shape overflows size_t");
        result *= dimension;
    }
    require(result <= static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()) / width,
            path, "array byte count overflows streamsize");
    return result;
}

std::size_t dimension(const std::string &text, const std::filesystem::path &path) {
    std::size_t result = 0;
    const auto conversion = std::from_chars(text.data(), text.data() + text.size(), result);
    require(conversion.ec == std::errc{} && conversion.ptr == text.data() + text.size(), path,
            "invalid shape dimension");
    return result;
}

template <typename T> T swap_scalar(T value) {
    auto bytes = std::bit_cast<std::array<std::byte, sizeof(T)>>(value);
    std::reverse(bytes.begin(), bytes.end());
    return std::bit_cast<T>(bytes);
}
Header read_header(std::ifstream &stream, const std::filesystem::path &path) {
    require(bool(stream), path, "open for reading failed");
    unsigned char prefix[10]{};
    stream.read(reinterpret_cast<char *>(prefix), sizeof(prefix));
    require(bool(stream) && std::memcmp(prefix, "\x93NUMPY\x01\x00", 8) == 0, path,
            "expected NPY v1.0 magic/version");
    const auto length = std::size_t(prefix[8]) + std::size_t(256) * prefix[9];
    require(length > 0 && length <= 4096 && (10 + length) % 16 == 0, path,
            "invalid or oversized header length/alignment");
    std::string text(length, '\0');
    stream.read(text.data(), static_cast<std::streamsize>(length));
    require(bool(stream), path, "truncated header");
    static const std::regex format(
        R"(\{\s*['"]descr['"]\s*:\s*['"]<(f4|f8|u4)['"]\s*,\s*['"]fortran_order['"]\s*:\s*False\s*,\s*['"]shape['"]\s*:\s*\(\s*([0-9]+)\s*,\s*(?:([0-9]+)\s*,?\s*)?\)\s*,?\s*\}\s*\n)");
    std::smatch match;
    require(std::regex_match(text, match, format), path,
            "unsupported header (requires <f4/<f8/<u4, C order, rank 1/2, alphabetical keys)");
    Header header{{dimension(match[2].str(), path)},
                  match[1] == "f4"   ? Dtype::float32
                  : match[1] == "f8" ? Dtype::float64
                                     : Dtype::uint32,
                  10 + length,
                  0,
                  match[1] == "f8" ? 8U : 4U};
    if (match[3].matched)
        header.shape.push_back(dimension(match[3].str(), path));
    header.elements = count(header.shape, path, header.item_bytes);
    const auto payload = header.elements * header.item_bytes;
    require(payload <= std::numeric_limits<std::size_t>::max() - header.offset, path,
            "file byte count overflows size_t");
    require(std::filesystem::file_size(path) == header.offset + payload, path,
            "payload length does not match shape");
    return header;
}
} // namespace

Header inspect(const std::filesystem::path &path) {
    std::ifstream stream(path, std::ios::binary);
    return read_header(stream, path);
}

Writer::Writer(const std::filesystem::path &path, const std::vector<std::size_t> &shape,
               Dtype dtype)
    : path_(path), remaining_(count(shape, path, dtype == Dtype::float64 ? 8 : 4)), dtype_(dtype) {
    const char *descr = dtype == Dtype::float32   ? "<f4"
                        : dtype == Dtype::float64 ? "<f8"
                        : dtype == Dtype::uint32  ? "<u4"
                                                  : nullptr;
    require(descr != nullptr, path, "unsupported writer dtype");
    std::string header =
        std::string("{'descr': '") + descr + "', 'fortran_order': False, 'shape': (";
    for (auto dimension : shape) {
        header += std::to_string(dimension) + ", ";
    }
    header += "), }";
    header.append((64 - (10 + header.size() + 1) % 64) % 64, ' ');
    header += '\n';
    require(header.size() <= 65535, path_, "v1 header too large");
    stream_.open(path_, std::ios::binary | std::ios::trunc);
    require(bool(stream_), path_, "open for writing failed");
    const unsigned char prefix[] = {0x93,
                                    'N',
                                    'U',
                                    'M',
                                    'P',
                                    'Y',
                                    1,
                                    0,
                                    static_cast<unsigned char>(header.size() & 255),
                                    static_cast<unsigned char>(header.size() >> 8)};
    stream_.write(reinterpret_cast<const char *>(prefix), sizeof(prefix));
    stream_.write(header.data(), static_cast<std::streamsize>(header.size()));
    require(bool(stream_), path_, "write header failed");
}

template <typename T> void Writer::append_values(std::span<const T> values, Dtype dtype) {
    require(dtype_ == dtype, path_, "append dtype does not match writer");
    require(stream_.is_open() && values.size() <= remaining_, path_, "append exceeds array shape");
    if constexpr (std::endian::native == std::endian::little) {
        if (!values.empty()) {
            stream_.write(reinterpret_cast<const char *>(values.data()),
                          static_cast<std::streamsize>(values.size_bytes()));
        }
    } else {
        for (auto value : values) {
            const auto little = swap_scalar(value);
            stream_.write(reinterpret_cast<const char *>(&little), sizeof(little));
        }
    }
    require(bool(stream_), path_, "write payload failed");
    remaining_ -= values.size();
}
void Writer::append(std::span<const float> values) { append_values(values, Dtype::float32); }
void Writer::append(std::span<const double> values) { append_values(values, Dtype::float64); }
void Writer::append(std::span<const std::uint32_t> values) { append_values(values, Dtype::uint32); }

void Writer::finish() {
    require(stream_.is_open() && remaining_ == 0, path_, "array is incomplete or already closed");
    stream_.close();
    require(bool(stream_), path_, "close/flush failed");
}

Reader::Reader(const std::filesystem::path &path) : path_(path), stream_(path, std::ios::binary) {
    auto header = read_header(stream_, path_);
    require(header.dtype == Dtype::float32, path_,
            "Reader requires float32; use MappedReader for explicit conversion");
    shape_ = std::move(header.shape);
    remaining_ = header.elements;
}

void Reader::read(std::span<float> values) {
    require(values.size() <= remaining_, path_, "read exceeds array shape");
    if (!values.empty()) {
        stream_.read(reinterpret_cast<char *>(values.data()),
                     static_cast<std::streamsize>(values.size_bytes()));
    }
    require(bool(stream_), path_, "read payload failed");
    if constexpr (std::endian::native != std::endian::little) {
        for (auto &value : values) {
            value = swap_scalar(value);
        }
    }
    remaining_ -= values.size();
}
} // namespace batchfit::npy
