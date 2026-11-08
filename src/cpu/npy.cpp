#include <batchfit/npy.hpp>

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

std::size_t count(const std::vector<std::size_t> &shape, const std::filesystem::path &path) {
    require(shape.size() == 1 || shape.size() == 2, path, "only rank 1/2 arrays are supported");
    std::size_t result = 1;
    for (auto dimension : shape) {
        require(dimension == 0 || result <= std::numeric_limits<std::size_t>::max() / dimension,
                path, "shape overflows size_t");
        result *= dimension;
    }
    require(result <= static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()) / 4,
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

[[maybe_unused]] float swap_float(float value) {
    auto bits = std::bit_cast<std::uint32_t>(value);
    bits = (bits >> 24) | ((bits >> 8) & 0xFF00U) | ((bits << 8) & 0xFF0000U) | (bits << 24);
    return std::bit_cast<float>(bits);
}
} // namespace

Writer::Writer(const std::filesystem::path &path, const std::vector<std::size_t> &shape)
    : path_(path), remaining_(count(shape, path)) {
    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (";
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

void Writer::append(std::span<const float> values) {
    require(stream_.is_open() && values.size() <= remaining_, path_, "append exceeds array shape");
    if constexpr (std::endian::native == std::endian::little) {
        if (!values.empty()) {
            stream_.write(reinterpret_cast<const char *>(values.data()),
                          static_cast<std::streamsize>(values.size_bytes()));
        }
    } else {
        for (auto value : values) {
            const auto little = swap_float(value);
            stream_.write(reinterpret_cast<const char *>(&little), sizeof(little));
        }
    }
    require(bool(stream_), path_, "write payload failed");
    remaining_ -= values.size();
}

void Writer::finish() {
    require(stream_.is_open() && remaining_ == 0, path_, "array is incomplete or already closed");
    stream_.close();
    require(bool(stream_), path_, "close/flush failed");
}

Reader::Reader(const std::filesystem::path &path) : path_(path), stream_(path, std::ios::binary) {
    require(bool(stream_), path_, "open for reading failed");
    unsigned char prefix[10]{};
    stream_.read(reinterpret_cast<char *>(prefix), sizeof(prefix));
    require(bool(stream_) && std::memcmp(prefix, "\x93NUMPY\x01\x00", 8) == 0, path_,
            "expected NPY v1.0 magic/version");
    const auto length = static_cast<std::size_t>(prefix[8]) + std::size_t(256) * prefix[9];
    require(length > 0 && length <= 4096 && (10 + length) % 16 == 0, path_,
            "invalid or oversized header length/alignment");
    std::string header(length, '\0');
    stream_.read(header.data(), static_cast<std::streamsize>(length));
    require(bool(stream_), path_, "truncated header");
    // Restrict this parser to NumPy's ordinary, alphabetical-key float32 header.
    // No Python expression evaluation, object arrays, or implicit dtype conversion.
    static const std::regex format(
        R"(\{\s*['"]descr['"]\s*:\s*['"]<f4['"]\s*,\s*['"]fortran_order['"]\s*:\s*False\s*,\s*['"]shape['"]\s*:\s*\(\s*([0-9]+)\s*,\s*(?:([0-9]+)\s*,?\s*)?\)\s*,?\s*\}\s*\n)");
    std::smatch match;
    require(std::regex_match(header, match, format), path_,
            "unsupported header (requires <f4, C order, rank 1/2, alphabetical keys)");
    shape_.push_back(dimension(match[1].str(), path_));
    if (match[2].matched) {
        shape_.push_back(dimension(match[2].str(), path_));
    }
    remaining_ = count(shape_, path_);
    require(std::filesystem::file_size(path_) == 10 + length + remaining_ * 4, path_,
            "payload length does not match shape");
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
            value = swap_float(value);
        }
    }
    remaining_ -= values.size();
}
} // namespace batchfit::npy
