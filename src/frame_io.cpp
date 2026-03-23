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
#include <thread>
#include <unordered_map>

extern "C" {
#include <libswscale/swscale.h>
}

namespace ave {
namespace frame_io {

// ─── SwsContext cache to avoid per-frame allocation ───────────────────────
// Thread-local cache for zero-contention performance
namespace {

struct SwsKey {
    int srcW;
    int srcH;
    int srcFmt;
    int dstW;
    int dstH;
    int dstFmt;

    bool operator==(const SwsKey& other) const noexcept {
        return srcW == other.srcW && srcH == other.srcH && srcFmt == other.srcFmt &&
               dstW == other.dstW && dstH == other.dstH && dstFmt == other.dstFmt;
    }
};

struct SwsKeyHash {
    std::size_t operator()(const SwsKey& k) const noexcept {
        std::size_t h = static_cast<std::size_t>(k.srcW);
        h = h * 31 + static_cast<std::size_t>(k.srcH);
        h = h * 31 + static_cast<std::size_t>(k.srcFmt);
        h = h * 31 + static_cast<std::size_t>(k.dstW);
        h = h * 31 + static_cast<std::size_t>(k.dstH);
        h = h * 31 + static_cast<std::size_t>(k.dstFmt);
        return h;
    }
};

using SwsCache = std::unordered_map<SwsKey, SwsContext*, SwsKeyHash>;

// Thread-local cache: each thread has its own cache, eliminating all contention
SwsCache& getThreadLocalSwsCache() {
    thread_local SwsCache cache;
    return cache;
}

// RAII cleaner for thread-local cache (cleanup when thread exits)
struct ThreadLocalSwsCacheCleaner {
    ~ThreadLocalSwsCacheCleaner() {
        SwsCache& cache = getThreadLocalSwsCache();
        for (auto& pair : cache) {
            sws_freeContext(pair.second);
        }
        cache.clear();
    }
};

// One cleaner per thread, automatically frees contexts when thread exits
ThreadLocalSwsCacheCleaner& getThreadLocalCleaner() {
    thread_local ThreadLocalSwsCacheCleaner cleaner;
    return cleaner;
}

SwsContext* getOrCreateSwsContext(const SwsKey& key) {
    // Initialize thread-local cleaner (ensures cleanup on thread exit)
    (void)getThreadLocalCleaner();
    
    // Access thread-local cache (no synchronization needed)
    SwsCache& cache = getThreadLocalSwsCache();
    
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;  // Cache hit - zero overhead
    }

    // Cache miss - create new context
    SwsContext* sws = sws_getContext(
        key.srcW, key.srcH, static_cast<AVPixelFormat>(key.srcFmt),
        key.dstW, key.dstH, static_cast<AVPixelFormat>(key.dstFmt),
        SWS_BILINEAR, nullptr, nullptr, nullptr);

    if (sws) {
        cache[key] = sws;
    }
    return sws;
}

// Global cleanup (for compatibility, now just clears thread-local cache)
void clearSwsCache() {
    SwsCache& cache = getThreadLocalSwsCache();
    for (auto& pair : cache) {
        sws_freeContext(pair.second);
    }
    cache.clear();
}

// ─── Helper: quote a shell argument ──────────────────────────────

std::string quoteArg(const std::string& value) {
    std::string out = "\"";
    for (const char ch : value) {
        if (ch == '\\' || ch == '"') out.push_back('\\');
        out.push_back(ch);
    }
    out.push_back('"');
    return out;
}

template <typename Fn>
void parallelForPixels(std::size_t pixelCount, Fn&& fn) {
    constexpr std::size_t kMinPixelsPerWorker = 262144u;
    unsigned workerCap = std::thread::hardware_concurrency();
    if (workerCap == 0u) {
        workerCap = 1u;
    }
    workerCap = std::min(workerCap, 8u);
    const std::size_t maxWorkersByProblem =
        (pixelCount + kMinPixelsPerWorker - 1u) / kMinPixelsPerWorker;
    const unsigned workers = static_cast<unsigned>(
        std::min<std::size_t>(static_cast<std::size_t>(workerCap), maxWorkersByProblem));
    if (workers <= 1u) {
        fn(0u, pixelCount);
        return;
    }

    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(workers - 1u));
    for (unsigned worker = 1u; worker < workers; ++worker) {
        const std::size_t begin =
            pixelCount * static_cast<std::size_t>(worker) / static_cast<std::size_t>(workers);
        const std::size_t end =
            pixelCount * static_cast<std::size_t>(worker + 1u) / static_cast<std::size_t>(workers);
        threads.emplace_back([begin, end, &fn]() { fn(begin, end); });
    }

    fn(0u, pixelCount / static_cast<std::size_t>(workers));
    for (auto& thread : threads) {
        thread.join();
    }
}

inline std::uint8_t floatToRgbByte(float value) {
    value = std::clamp(value * 255.0f, 0.0f, 255.0f);
    return static_cast<std::uint8_t>(value + 0.5f);
}

inline std::uint16_t floatToHalfBits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    const std::uint32_t sign = (bits >> 16u) & 0x8000u;
    std::uint32_t mantissa = bits & 0x007fffffu;
    int exponent = static_cast<int>((bits >> 23u) & 0xffu) - 127 + 15;

    if (exponent <= 0) {
        if (exponent < -10) {
            return static_cast<std::uint16_t>(sign);
        }
        mantissa = (mantissa | 0x00800000u) >> static_cast<unsigned>(1 - exponent);
        return static_cast<std::uint16_t>(sign | ((mantissa + 0x00001000u) >> 13u));
    }

    if (exponent >= 31) {
        if (((bits >> 23u) & 0xffu) == 0xffu && mantissa != 0u) {
            return static_cast<std::uint16_t>(sign | 0x7c00u | (mantissa >> 13u) | 1u);
        }
        return static_cast<std::uint16_t>(sign | 0x7c00u);
    }

    return static_cast<std::uint16_t>(
        sign | (static_cast<std::uint32_t>(exponent) << 10u)
             | ((mantissa + 0x00001000u) >> 13u));
}

inline float halfBitsToFloat(std::uint16_t bits) {
    const std::uint32_t sign = (static_cast<std::uint32_t>(bits & 0x8000u)) << 16u;
    std::uint32_t exponent = (static_cast<std::uint32_t>(bits) >> 10u) & 0x1fu;
    std::uint32_t mantissa = static_cast<std::uint32_t>(bits & 0x03ffu);
    std::uint32_t outBits = 0;

    if (exponent == 0u) {
        if (mantissa == 0u) {
            outBits = sign;
        } else {
            exponent = 127u - 15u + 1u;
            while ((mantissa & 0x0400u) == 0u) {
                mantissa <<= 1u;
                --exponent;
            }
            mantissa &= 0x03ffu;
            outBits = sign | (exponent << 23u) | (mantissa << 13u);
        }
    } else if (exponent == 0x1fu) {
        outBits = sign | 0x7f800000u | (mantissa << 13u);
    } else {
        exponent = exponent + (127u - 15u);
        outBits = sign | (exponent << 23u) | (mantissa << 13u);
    }

    float out = 0.0f;
    std::memcpy(&out, &outBits, sizeof(out));
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
    float* const rPlane = tensor.data();
    float* const gPlane = rPlane + hw;
    float* const bPlane = gPlane + hw;
    constexpr float kInv255 = 1.0f / 255.0f;

    parallelForPixels(hw, [&](std::size_t begin, std::size_t end) {
        std::size_t rgbIdx = begin * 3u;
        for (std::size_t pixIdx = begin; pixIdx < end; ++pixIdx, rgbIdx += 3u) {
            rPlane[pixIdx] = static_cast<float>(rgb[rgbIdx + 0u]) * kInv255;
            gPlane[pixIdx] = static_cast<float>(rgb[rgbIdx + 1u]) * kInv255;
            bPlane[pixIdx] = static_cast<float>(rgb[rgbIdx + 2u]) * kInv255;
        }
    });
}

// ─────────────────────────────────────────────────────────────────
// rgb24ToNchwFp16 — RGB24 uint8 [H,W,3] → fp16 NCHW [1,3,H,W]
// ─────────────────────────────────────────────────────────────────
void rgb24ToNchwFp16(const std::uint8_t* rgb,
                     int width, int height,
                     std::vector<std::uint16_t>& tensor) {
    const std::size_t hw = static_cast<std::size_t>(height) *
                           static_cast<std::size_t>(width);
    tensor.resize(3u * hw);
    std::uint16_t* const rPlane = tensor.data();
    std::uint16_t* const gPlane = rPlane + hw;
    std::uint16_t* const bPlane = gPlane + hw;
    constexpr float kInv255 = 1.0f / 255.0f;

    parallelForPixels(hw, [&](std::size_t begin, std::size_t end) {
        std::size_t rgbIdx = begin * 3u;
        for (std::size_t pixIdx = begin; pixIdx < end; ++pixIdx, rgbIdx += 3u) {
            rPlane[pixIdx] = floatToHalfBits(static_cast<float>(rgb[rgbIdx + 0u]) * kInv255);
            gPlane[pixIdx] = floatToHalfBits(static_cast<float>(rgb[rgbIdx + 1u]) * kInv255);
            bPlane[pixIdx] = floatToHalfBits(static_cast<float>(rgb[rgbIdx + 2u]) * kInv255);
        }
    });
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
    const float* const rPlane = channels > 0 ? tensor : nullptr;
    const float* const gPlane = channels > 1 ? tensor + hw : nullptr;
    const float* const bPlane = channels > 2 ? tensor + (2u * hw) : nullptr;

    if (channels >= 3) {
        parallelForPixels(hw, [&](std::size_t begin, std::size_t end) {
            std::size_t rgbIdx = begin * 3u;
            for (std::size_t pixIdx = begin; pixIdx < end; ++pixIdx, rgbIdx += 3u) {
                rgb[rgbIdx + 0u] = floatToRgbByte(rPlane[pixIdx]);
                rgb[rgbIdx + 1u] = floatToRgbByte(gPlane[pixIdx]);
                rgb[rgbIdx + 2u] = floatToRgbByte(bPlane[pixIdx]);
            }
        });
        return;
    }

    if (channels == 2) {
        parallelForPixels(hw, [&](std::size_t begin, std::size_t end) {
            std::size_t rgbIdx = begin * 3u;
            for (std::size_t pixIdx = begin; pixIdx < end; ++pixIdx, rgbIdx += 3u) {
                rgb[rgbIdx + 0u] = floatToRgbByte(rPlane[pixIdx]);
                rgb[rgbIdx + 1u] = floatToRgbByte(gPlane[pixIdx]);
                rgb[rgbIdx + 2u] = 0u;
            }
        });
        return;
    }

    if (channels == 1) {
        parallelForPixels(hw, [&](std::size_t begin, std::size_t end) {
            std::size_t rgbIdx = begin * 3u;
            for (std::size_t pixIdx = begin; pixIdx < end; ++pixIdx, rgbIdx += 3u) {
                rgb[rgbIdx + 0u] = floatToRgbByte(rPlane[pixIdx]);
                rgb[rgbIdx + 1u] = 0u;
                rgb[rgbIdx + 2u] = 0u;
            }
        });
        return;
    }

    std::fill(rgb.begin(), rgb.end(), static_cast<std::uint8_t>(0));
}

// ─────────────────────────────────────────────────────────────────
// nchwFp16ToRgb24 — fp16 NCHW [1,C,H,W] → RGB24 uint8 [H,W,3]
// ─────────────────────────────────────────────────────────────────
void nchwFp16ToRgb24(const std::uint16_t* tensor,
                     int channels, int width, int height,
                     std::vector<std::uint8_t>& rgb) {
    const std::size_t hw = static_cast<std::size_t>(height) *
                           static_cast<std::size_t>(width);
    rgb.resize(hw * 3u);
    const std::uint16_t* const rPlane = channels > 0 ? tensor : nullptr;
    const std::uint16_t* const gPlane = channels > 1 ? tensor + hw : nullptr;
    const std::uint16_t* const bPlane = channels > 2 ? tensor + (2u * hw) : nullptr;

    if (channels >= 3) {
        parallelForPixels(hw, [&](std::size_t begin, std::size_t end) {
            std::size_t rgbIdx = begin * 3u;
            for (std::size_t pixIdx = begin; pixIdx < end; ++pixIdx, rgbIdx += 3u) {
                rgb[rgbIdx + 0u] = floatToRgbByte(halfBitsToFloat(rPlane[pixIdx]));
                rgb[rgbIdx + 1u] = floatToRgbByte(halfBitsToFloat(gPlane[pixIdx]));
                rgb[rgbIdx + 2u] = floatToRgbByte(halfBitsToFloat(bPlane[pixIdx]));
            }
        });
        return;
    }

    if (channels == 2) {
        parallelForPixels(hw, [&](std::size_t begin, std::size_t end) {
            std::size_t rgbIdx = begin * 3u;
            for (std::size_t pixIdx = begin; pixIdx < end; ++pixIdx, rgbIdx += 3u) {
                rgb[rgbIdx + 0u] = floatToRgbByte(halfBitsToFloat(rPlane[pixIdx]));
                rgb[rgbIdx + 1u] = floatToRgbByte(halfBitsToFloat(gPlane[pixIdx]));
                rgb[rgbIdx + 2u] = 0u;
            }
        });
        return;
    }

    if (channels == 1) {
        parallelForPixels(hw, [&](std::size_t begin, std::size_t end) {
            std::size_t rgbIdx = begin * 3u;
            for (std::size_t pixIdx = begin; pixIdx < end; ++pixIdx, rgbIdx += 3u) {
                rgb[rgbIdx + 0u] = floatToRgbByte(halfBitsToFloat(rPlane[pixIdx]));
                rgb[rgbIdx + 1u] = 0u;
                rgb[rgbIdx + 2u] = 0u;
            }
        });
        return;
    }

    std::fill(rgb.begin(), rgb.end(), static_cast<std::uint8_t>(0));
}

// ─────────────────────────────────────────────────────────────────
// avFrameToRgb24 — convert any AVFrame to RGB24 using swscale
// ─────────────────────────────────────────────────────────────────
bool avFrameToRgb24(const AVFrame* frame,
                    int width, int height,
                    std::vector<std::uint8_t>& rgb,
                    std::string& error) {
    if (!frame) {
        error = "avFrameToRgb24: null frame";
        return false;
    }

    const std::size_t rgbSize = static_cast<std::size_t>(width) *
                                 static_cast<std::size_t>(height) * 3u;
    rgb.resize(rgbSize);

    AVPixelFormat srcFmt = static_cast<AVPixelFormat>(frame->format);
    if (srcFmt == AV_PIX_FMT_VULKAN) {
        srcFmt = AV_PIX_FMT_YUV420P;  // Vulkan frames are typically YUV420P
    }

    SwsKey key{frame->width, frame->height, static_cast<int>(srcFmt),
               width, height, static_cast<int>(AV_PIX_FMT_RGB24)};
    SwsContext* sws = getOrCreateSwsContext(key);

    if (!sws) {
        error = "avFrameToRgb24: sws_getContext failed";
        return false;
    }

    uint8_t* dst[4] = { rgb.data(), nullptr, nullptr, nullptr };
    int dstLinesize[4] = { width * 3, 0, 0, 0 };

    const uint8_t* src[4] = {
        frame->data[0], frame->data[1], frame->data[2], nullptr
    };
    const int srcLinesize[4] = {
        frame->linesize[0], frame->linesize[1], frame->linesize[2], 0
    };

    sws_scale(sws, src, srcLinesize, 0, frame->height, dst, dstLinesize);

    return true;
}

// ─────────────────────────────────────────────────────────────────
// rgb24ToAvFrame — create YUV420P AVFrame from RGB24 bytes
// ─────────────────────────────────────────────────────────────────
AVFrame* rgb24ToAvFrame(const std::uint8_t* rgb,
                        int width, int height,
                        std::string& error) {
    AVFrame* frame = av_frame_alloc();
    if (!frame) {
        error = "rgb24ToAvFrame: av_frame_alloc failed";
        return nullptr;
    }

    frame->format = AV_PIX_FMT_YUV420P;
    frame->width = width;
    frame->height = height;

    if (av_frame_get_buffer(frame, 0) < 0) {
        error = "rgb24ToAvFrame: av_frame_get_buffer failed";
        av_frame_free(&frame);
        return nullptr;
    }

    SwsKey key{width, height, static_cast<int>(AV_PIX_FMT_RGB24),
               width, height, static_cast<int>(AV_PIX_FMT_YUV420P)};
    SwsContext* sws = getOrCreateSwsContext(key);

    if (!sws) {
        error = "rgb24ToAvFrame: sws_getContext failed";
        av_frame_free(&frame);
        return nullptr;
    }

    const uint8_t* src[4] = { rgb, nullptr, nullptr, nullptr };
    const int srcLinesize[4] = { width * 3, 0, 0, 0 };

    uint8_t* dst[4] = {
        frame->data[0], frame->data[1], frame->data[2], nullptr
    };
    int dstLinesize[4] = {
        frame->linesize[0], frame->linesize[1], frame->linesize[2], 0
    };

    sws_scale(sws, src, srcLinesize, 0, height, dst, dstLinesize);

    frame->pts = 0;
    return frame;
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
