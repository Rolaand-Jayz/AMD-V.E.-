// ─────────────────────────────────────────────────────────────────
// frame_io.cpp — Frame I/O helper implementation
// ─────────────────────────────────────────────────────────────────
#include "ave/frame_io.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

namespace ave {
namespace frame_io {

// ─── Helper: quote a shell argument ──────────────────────────────
namespace {

std::string quoteArg(const std::string& value) {
    std::string out = "\"";
    for (const char ch : value) {
        if (ch == '\\' || ch == '"') out.push_back('\\');
        out.push_back(ch);
    }
    out.push_back('"');
    return out;
}

}  // namespace

// ─────────────────────────────────────────────────────────────────
// readPngDimensions — parse IHDR directly (no process spawn)
// ─────────────────────────────────────────────────────────────────
bool readPngDimensions(const std::string& path, int& width, int& height) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;

    // PNG layout: 8-byte signature, then IHDR chunk:
    //   4 bytes chunk length  |  4 bytes "IHDR"
    //   4 bytes width (big-endian u32)
    //   4 bytes height (big-endian u32)
    // Total header bytes we need: 8 (sig) + 8 (chunk hdr) + 8 (w+h) = 24
    std::uint8_t buf[24];
    f.read(reinterpret_cast<char*>(buf), sizeof(buf));
    if (f.gcount() < 24) return false;

    // Verify PNG signature (first 4 bytes: 0x89 'P' 'N' 'G')
    if (buf[0] != 0x89 || buf[1] != 'P' || buf[2] != 'N' || buf[3] != 'G')
        return false;

    // Width at byte offset 16, Height at 20 (big-endian u32)
    width  = static_cast<int>((static_cast<unsigned>(buf[16]) << 24u) |
                              (static_cast<unsigned>(buf[17]) << 16u) |
                              (static_cast<unsigned>(buf[18]) <<  8u) |
                               static_cast<unsigned>(buf[19]));
    height = static_cast<int>((static_cast<unsigned>(buf[20]) << 24u) |
                              (static_cast<unsigned>(buf[21]) << 16u) |
                              (static_cast<unsigned>(buf[22]) <<  8u) |
                               static_cast<unsigned>(buf[23]));
    return width > 0 && height > 0;
}

// ─────────────────────────────────────────────────────────────────
// loadPngRgb24 — load via FFmpeg raw pipe
// ─────────────────────────────────────────────────────────────────
bool loadPngRgb24(const std::string& path,
                  int& width, int& height,
                  std::vector<std::uint8_t>& data,
                  std::string& error) {
    // Step 1: get dimensions from the PNG header.
    if (!readPngDimensions(path, width, height)) {
        error = "Cannot read PNG dimensions from " + path;
        return false;
    }

    // Step 2: decode to raw RGB24 via FFmpeg pipe.
    const std::size_t expected = static_cast<std::size_t>(width) *
                                 static_cast<std::size_t>(height) * 3u;
    data.resize(expected);

    const std::string cmd =
        "ffmpeg -hide_banner -loglevel quiet -i " + quoteArg(path) +
        " -f rawvideo -pix_fmt rgb24 -frames:v 1 pipe:1 2>/dev/null";

    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        error = "Failed to start FFmpeg for frame loading: " + path;
        return false;
    }

    std::size_t totalRead = 0;
    while (totalRead < expected) {
        const std::size_t n = std::fread(data.data() + totalRead, 1,
                                         expected - totalRead, pipe);
        if (n == 0) break;
        totalRead += n;
    }
    const int ret = pclose(pipe);

    if (ret != 0 || totalRead != expected) {
        error = "FFmpeg frame load incomplete for " + path +
                " (expected " + std::to_string(expected) +
                " bytes, got " + std::to_string(totalRead) + ")";
        return false;
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────
// saveRgb24ToPng — write via FFmpeg pipe
// ─────────────────────────────────────────────────────────────────
bool saveRgb24ToPng(const std::string& path,
                    int width, int height,
                    const std::uint8_t* data,
                    std::string& error) {
    const std::size_t bytes = static_cast<std::size_t>(width) *
                              static_cast<std::size_t>(height) * 3u;

    const std::string cmd =
        "ffmpeg -y -hide_banner -loglevel quiet "
        "-f rawvideo -pix_fmt rgb24 -s " +
        std::to_string(width) + "x" + std::to_string(height) +
        " -i pipe:0 -frames:v 1 " + quoteArg(path) + " 2>/dev/null";

    FILE* pipe = popen(cmd.c_str(), "w");
    if (!pipe) {
        error = "Failed to start FFmpeg for frame saving: " + path;
        return false;
    }

    const std::size_t written = std::fwrite(data, 1, bytes, pipe);
    const int ret = pclose(pipe);

    if (ret != 0 || written != bytes) {
        error = "FFmpeg frame save failed for " + path;
        return false;
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────
// rgb24ToNchwFp32 — RGB24 uint8 [H,W,3] → fp32 NCHW [1,3,H,W]
// ─────────────────────────────────────────────────────────────────
void rgb24ToNchwFp32(const std::uint8_t* rgb,
                     int width, int height,
                     std::vector<float>& tensor) {
    const std::size_t hw = static_cast<std::size_t>(height) *
                           static_cast<std::size_t>(width);
    tensor.resize(3u * hw);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const std::size_t pixIdx = static_cast<std::size_t>(y) *
                                       static_cast<std::size_t>(width) +
                                       static_cast<std::size_t>(x);
            const std::size_t rgbIdx = pixIdx * 3u;
            tensor[0u * hw + pixIdx] = static_cast<float>(rgb[rgbIdx + 0]) / 255.0f;
            tensor[1u * hw + pixIdx] = static_cast<float>(rgb[rgbIdx + 1]) / 255.0f;
            tensor[2u * hw + pixIdx] = static_cast<float>(rgb[rgbIdx + 2]) / 255.0f;
        }
    }
}

// ─────────────────────────────────────────────────────────────────
// nchwFp32ToRgb24 — fp32 NCHW [1,C,H,W] → RGB24 uint8 [H,W,3]
// ─────────────────────────────────────────────────────────────────
void nchwFp32ToRgb24(const float* tensor,
                     int channels, int width, int height,
                     std::vector<std::uint8_t>& rgb) {
    const std::size_t hw = static_cast<std::size_t>(height) *
                           static_cast<std::size_t>(width);
    rgb.resize(hw * 3u);

    const int usedChannels = std::min(channels, 3);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const std::size_t pixIdx = static_cast<std::size_t>(y) *
                                       static_cast<std::size_t>(width) +
                                       static_cast<std::size_t>(x);
            for (int c = 0; c < usedChannels; ++c) {
                float val = tensor[static_cast<std::size_t>(c) * hw + pixIdx] * 255.0f;
                val = std::max(0.0f, std::min(255.0f, val));
                rgb[pixIdx * 3u + static_cast<std::size_t>(c)] =
                    static_cast<std::uint8_t>(val + 0.5f);
            }
            // Fill remaining channels with 0 if the tensor has < 3 channels.
            for (int c = usedChannels; c < 3; ++c) {
                rgb[pixIdx * 3u + static_cast<std::size_t>(c)] = 0;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────
// listPngFramesSorted
// ─────────────────────────────────────────────────────────────────
std::vector<std::filesystem::path> listPngFramesSorted(const std::string& dir) {
    std::vector<std::filesystem::path> result;
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
        if (ec) break;
        if (entry.is_regular_file() && entry.path().extension() == ".png")
            result.push_back(entry.path());
    }
    std::sort(result.begin(), result.end());
    return result;
}

}  // namespace frame_io
}  // namespace ave
