// ─────────────────────────────────────────────────────────────────
// frame_io.cpp — Frame I/O helper implementation
// ─────────────────────────────────────────────────────────────────
#include "ave/frame_io.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <unordered_map>

#if defined(__linux__)
#  include <fcntl.h>
#  include <unistd.h>
#endif

extern "C" {
#include <libswscale/swscale.h>
}

#include "ave/video_probe.hpp"

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

struct ThreadLocalSwsCache {
    SwsCache cache;

    ~ThreadLocalSwsCache() {
        for (auto& pair : cache) {
            sws_freeContext(pair.second);
        }
        cache.clear();
    }
};

// Thread-local cache: each thread has its own cache, eliminating all contention
ThreadLocalSwsCache& getThreadLocalSwsCache() {
    thread_local ThreadLocalSwsCache cache;
    return cache;
}

SwsContext* getOrCreateSwsContext(const SwsKey& key) {
    // Access thread-local cache (no synchronization needed)
    SwsCache& cache = getThreadLocalSwsCache().cache;
    
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

void tunePipeIo(FILE* pipe, std::size_t stdioBufferBytes, int pipeBytes) {
    if (pipe == nullptr) {
        return;
    }
    if (stdioBufferBytes > 0u) {
        setvbuf(pipe, nullptr, _IOFBF, stdioBufferBytes);
    }
#if defined(__linux__)
    const int fd = fileno(pipe);
    if (fd >= 0 && pipeBytes > 0) {
        (void)fcntl(fd, F_SETPIPE_SZ, pipeBytes);
    }
#else
    (void)pipeBytes;
#endif
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

enum class DirectTensorPackingMode {
    Unsupported,
    PackedRgb,
    Yuv420Planar,
    Nv12,
    P010,
};

struct PackedRgbFrameView {
    const std::uint8_t* data = nullptr;
    int linesize = 0;
    int width = 0;
    int height = 0;
    int pixelStride = 0;
    int redOffset = 0;
    int greenOffset = 1;
    int blueOffset = 2;
};

bool makePackedRgbFrameView(const AVFrame* frame, PackedRgbFrameView& view) {
    view = {};
    if (frame == nullptr || frame->data[0] == nullptr || frame->width <= 0 ||
        frame->height <= 0 || frame->linesize[0] <= 0) {
        return false;
    }

    view.data = frame->data[0];
    view.linesize = frame->linesize[0];
    view.width = frame->width;
    view.height = frame->height;

    switch (static_cast<AVPixelFormat>(frame->format)) {
        case AV_PIX_FMT_RGB24:
            view.pixelStride = 3;
            view.redOffset = 0;
            view.greenOffset = 1;
            view.blueOffset = 2;
            return true;
        case AV_PIX_FMT_BGR24:
            view.pixelStride = 3;
            view.redOffset = 2;
            view.greenOffset = 1;
            view.blueOffset = 0;
            return true;
        case AV_PIX_FMT_RGBA:
            view.pixelStride = 4;
            view.redOffset = 0;
            view.greenOffset = 1;
            view.blueOffset = 2;
            return true;
        case AV_PIX_FMT_BGRA:
            view.pixelStride = 4;
            view.redOffset = 2;
            view.greenOffset = 1;
            view.blueOffset = 0;
            return true;
        case AV_PIX_FMT_ARGB:
            view.pixelStride = 4;
            view.redOffset = 1;
            view.greenOffset = 2;
            view.blueOffset = 3;
            return true;
        case AV_PIX_FMT_ABGR:
            view.pixelStride = 4;
            view.redOffset = 3;
            view.greenOffset = 2;
            view.blueOffset = 1;
            return true;
        default:
            return false;
    }
}

struct YuvCoefficients {
    float kr = 0.299f;
    float kb = 0.114f;
};

DirectTensorPackingMode detectDirectTensorPackingMode(const AVFrame* frame,
                                                      PackedRgbFrameView* packedRgbView = nullptr) {
    if (frame == nullptr || isVulkanHardwareFrame(frame) || frame->width <= 0 ||
        frame->height <= 0) {
        return DirectTensorPackingMode::Unsupported;
    }

    PackedRgbFrameView localView;
    if (makePackedRgbFrameView(frame, localView)) {
        if (packedRgbView != nullptr) {
            *packedRgbView = localView;
        }
        return DirectTensorPackingMode::PackedRgb;
    }

    switch (static_cast<AVPixelFormat>(frame->format)) {
        case AV_PIX_FMT_YUV420P:
        case AV_PIX_FMT_YUVJ420P:
            return frame->data[0] != nullptr && frame->data[1] != nullptr && frame->data[2] != nullptr
                ? DirectTensorPackingMode::Yuv420Planar
                : DirectTensorPackingMode::Unsupported;
        case AV_PIX_FMT_NV12:
            return frame->data[0] != nullptr && frame->data[1] != nullptr
                ? DirectTensorPackingMode::Nv12
                : DirectTensorPackingMode::Unsupported;
        case AV_PIX_FMT_P010LE:
            return frame->data[0] != nullptr && frame->data[1] != nullptr
                ? DirectTensorPackingMode::P010
                : DirectTensorPackingMode::Unsupported;
        default:
            return DirectTensorPackingMode::Unsupported;
    }
}

bool frameUsesFullRange(const AVFrame* frame) {
    if (frame == nullptr) {
        return false;
    }
    if (frame->color_range == AVCOL_RANGE_JPEG) {
        return true;
    }
    return static_cast<AVPixelFormat>(frame->format) == AV_PIX_FMT_YUVJ420P;
}

YuvCoefficients yuvCoefficientsForFrame(const AVFrame* frame) {
    if (frame == nullptr) {
        return {};
    }
    switch (frame->colorspace) {
        case AVCOL_SPC_BT709:
            return {0.2126f, 0.0722f};
        case AVCOL_SPC_BT2020_NCL:
        case AVCOL_SPC_BT2020_CL:
            return {0.2627f, 0.0593f};
        default:
            return {};
    }
}

float normalizeLuma(float sample, float maxValue, bool fullRange) {
    if (fullRange) {
        return std::clamp(sample / maxValue, 0.0f, 1.0f);
    }
    const float black = maxValue > 255.0f ? 64.0f : 16.0f;
    const float scale = maxValue > 255.0f ? 876.0f : 219.0f;
    return std::clamp((sample - black) / scale, 0.0f, 1.0f);
}

float normalizeChroma(float sample, float maxValue, bool fullRange) {
    if (fullRange) {
        return std::clamp((sample / maxValue) - 0.5f, -0.5f, 0.5f);
    }
    const float center = maxValue > 255.0f ? 512.0f : 128.0f;
    const float scale = maxValue > 255.0f ? 896.0f : 224.0f;
    return std::clamp((sample - center) / scale, -0.5f, 0.5f);
}

void yuvToRgb(const float y,
              const float cb,
              const float cr,
              const YuvCoefficients& coeffs,
              float& red,
              float& green,
              float& blue) {
    const float kg = 1.0f - coeffs.kr - coeffs.kb;
    red = std::clamp(y + (2.0f - 2.0f * coeffs.kr) * cr, 0.0f, 1.0f);
    blue = std::clamp(y + (2.0f - 2.0f * coeffs.kb) * cb, 0.0f, 1.0f);
    green = std::clamp(
        y - ((coeffs.kb * (2.0f - 2.0f * coeffs.kb)) / kg) * cb
          - ((coeffs.kr * (2.0f - 2.0f * coeffs.kr)) / kg) * cr,
        0.0f,
        1.0f);
}

std::uint16_t readLe16(const std::uint8_t* ptr) {
    std::uint16_t value = 0u;
    std::memcpy(&value, ptr, sizeof(value));
    return value;
}

bool sampleDirectFrameRgb(const AVFrame* frame,
                          DirectTensorPackingMode mode,
                          const PackedRgbFrameView& packedRgbView,
                          int srcX,
                          int srcY,
                          float& red,
                          float& green,
                          float& blue) {
    srcX = std::clamp(srcX, 0, frame->width - 1);
    srcY = std::clamp(srcY, 0, frame->height - 1);

    if (mode == DirectTensorPackingMode::PackedRgb) {
        constexpr float kInv255 = 1.0f / 255.0f;
        const auto* srcRow =
            packedRgbView.data + static_cast<std::ptrdiff_t>(srcY) * packedRgbView.linesize;
        const auto* pixel = srcRow + static_cast<std::ptrdiff_t>(srcX) * packedRgbView.pixelStride;
        red = static_cast<float>(pixel[packedRgbView.redOffset]) * kInv255;
        green = static_cast<float>(pixel[packedRgbView.greenOffset]) * kInv255;
        blue = static_cast<float>(pixel[packedRgbView.blueOffset]) * kInv255;
        return true;
    }

    const bool fullRange = frameUsesFullRange(frame);
    const YuvCoefficients coeffs = yuvCoefficientsForFrame(frame);

    if (mode == DirectTensorPackingMode::Yuv420Planar) {
        const int chromaX = srcX / 2;
        const int chromaY = srcY / 2;
        const float y = normalizeLuma(
            static_cast<float>(frame->data[0][srcY * frame->linesize[0] + srcX]),
            255.0f,
            fullRange);
        const float cb = normalizeChroma(
            static_cast<float>(frame->data[1][chromaY * frame->linesize[1] + chromaX]),
            255.0f,
            fullRange);
        const float cr = normalizeChroma(
            static_cast<float>(frame->data[2][chromaY * frame->linesize[2] + chromaX]),
            255.0f,
            fullRange);
        yuvToRgb(y, cb, cr, coeffs, red, green, blue);
        return true;
    }

    if (mode == DirectTensorPackingMode::Nv12) {
        const int chromaX = srcX / 2;
        const int chromaY = srcY / 2;
        const auto* uvRow =
            frame->data[1] + static_cast<std::ptrdiff_t>(chromaY) * frame->linesize[1];
        const float y = normalizeLuma(
            static_cast<float>(frame->data[0][srcY * frame->linesize[0] + srcX]),
            255.0f,
            fullRange);
        const float cb = normalizeChroma(
            static_cast<float>(uvRow[chromaX * 2]),
            255.0f,
            fullRange);
        const float cr = normalizeChroma(
            static_cast<float>(uvRow[chromaX * 2 + 1]),
            255.0f,
            fullRange);
        yuvToRgb(y, cb, cr, coeffs, red, green, blue);
        return true;
    }

    if (mode == DirectTensorPackingMode::P010) {
        const int chromaX = srcX / 2;
        const int chromaY = srcY / 2;
        const auto* yPtr = frame->data[0]
            + static_cast<std::ptrdiff_t>(srcY) * frame->linesize[0]
            + static_cast<std::ptrdiff_t>(srcX) * 2;
        const auto* uvPtr = frame->data[1]
            + static_cast<std::ptrdiff_t>(chromaY) * frame->linesize[1]
            + static_cast<std::ptrdiff_t>(chromaX) * 4;
        const float y = normalizeLuma(
            static_cast<float>(readLe16(yPtr) >> 6u),
            1023.0f,
            fullRange);
        const float cb = normalizeChroma(
            static_cast<float>(readLe16(uvPtr) >> 6u),
            1023.0f,
            fullRange);
        const float cr = normalizeChroma(
            static_cast<float>(readLe16(uvPtr + 2) >> 6u),
            1023.0f,
            fullRange);
        yuvToRgb(y, cb, cr, coeffs, red, green, blue);
        return true;
    }

    return false;
}

template <typename StorePixel>
bool packDirectFrameTileGeneric(const AVFrame* frame,
                                int tileX,
                                int tileY,
                                int tileWidth,
                                int tileHeight,
                                StorePixel&& storePixel,
                                std::string& error) {
    PackedRgbFrameView packedRgbView;
    const DirectTensorPackingMode mode = detectDirectTensorPackingMode(frame, &packedRgbView);
    if (mode == DirectTensorPackingMode::Unsupported) {
        error = "Direct tensor packing does not support AVFrame format "
              + std::to_string(frame != nullptr ? frame->format : -1) + '.';
        return false;
    }

    const std::size_t hw = static_cast<std::size_t>(tileWidth) *
                           static_cast<std::size_t>(tileHeight);
    parallelForPixels(hw, [&](std::size_t begin, std::size_t end) {
        for (std::size_t pixelIndex = begin; pixelIndex < end; ++pixelIndex) {
            const int localY = static_cast<int>(pixelIndex / static_cast<std::size_t>(tileWidth));
            const int localX = static_cast<int>(pixelIndex % static_cast<std::size_t>(tileWidth));
            float red = 0.0f;
            float green = 0.0f;
            float blue = 0.0f;
            if (!sampleDirectFrameRgb(frame,
                                      mode,
                                      packedRgbView,
                                      tileX + localX,
                                      tileY + localY,
                                      red,
                                      green,
                                      blue)) {
                red = 0.0f;
                green = 0.0f;
                blue = 0.0f;
            }
            storePixel(pixelIndex, red, green, blue, hw);
        }
    });
    return true;
}

bool cloneFrameReference(const AVFrame* source,
                         AvFramePtr& destination,
                         std::string& error) {
    destination.reset();
    if (source == nullptr) {
        return true;
    }

    AVFrame* cloned = av_frame_alloc();
    if (cloned == nullptr) {
        error = "Failed to allocate AVFrame for decoded frame packet.";
        return false;
    }
    if (av_frame_ref(cloned, source) < 0) {
        av_frame_free(&cloned);
        error = "Failed to retain decoded AVFrame for frame packet.";
        return false;
    }
    destination.reset(cloned);
    return true;
}

}  // namespace

struct RgbVideoPipeSource::Impl {
    FILE* pipe = nullptr;
    std::size_t frameBytes = 0u;
};

struct VideoFrameSource::Impl {
    RgbVideoSourceMode mode = RgbVideoSourceMode::RawPipe;
    int width = 0;
    int height = 0;
    RgbVideoPipeSource pipeSource;
    VulkanVideoReader vulkanReader;
};

struct RgbVideoFrameSource::Impl {
    VideoFrameSource source;
};

struct RgbVideoFrameWriter::Impl {
    VulkanVideoWriter writer;
    int width = 0;
    int height = 0;
    std::size_t expectedBytes = 0u;
};

struct RgbVideoSession::Impl {
    RgbVideoFrameSource source;
    RgbVideoFrameWriter writer;
};

struct AsyncRgbVideoPipeEncoder::Impl {
    struct QueuedFrame {
        int frameIdx = 0;
        std::vector<std::uint8_t> pixels;
    };

    FILE* pipe = nullptr;
    std::size_t frameBytes = 0u;
    std::size_t queueDepth = 1u;
    std::filesystem::path logPath;

    mutable std::mutex mutex;
    std::condition_variable cv;
    std::deque<QueuedFrame> queue;
    std::deque<std::vector<std::uint8_t>> freeBuffers;
    bool inputClosed = false;
    bool writerFailed = false;
    std::string writerError;
    std::thread writerThread;
    std::atomic<std::int64_t> writeTimeNs{0};
};

void AvFrameDeleter::operator()(AVFrame* frame) const {
    if (frame != nullptr) {
        av_frame_free(&frame);
    }
}

bool VideoFramePacket::hasRgb24() const {
    return !rgb24.empty();
}

bool VideoFramePacket::hasFrame() const {
    return static_cast<bool>(frame);
}

bool VideoFramePacket::isHardwareFrame() const {
    return frame_io::isVulkanHardwareFrame(frame.get());
}

void VideoFramePacket::reset() {
    width = 0;
    height = 0;
    sourceMode = RgbVideoSourceMode::RawPipe;
    rgb24.clear();
    frame.reset();
}

bool VideoFramePacketMaterializer::resolveSoftwareFrame(const VideoFramePacket& packet,
                                                        AVFrame*& frameOut,
                                                        std::string& error) {
    frameOut = nullptr;
    lastResolveMode_ = SoftwareFrameResolveMode::None;
    if (!packet.hasFrame()) {
        error = "VideoFramePacketMaterializer: packet contains no AVFrame payload.";
        return false;
    }

    if (!packet.isHardwareFrame()) {
        frameOut = packet.frame.get();
        lastResolveMode_ = SoftwareFrameResolveMode::PacketFrame;
        return true;
    }

    auto ensureMappedFrame = [&]() -> bool {
        if (mappedFrame_ == nullptr) {
            mappedFrame_.reset(av_frame_alloc());
            if (mappedFrame_ == nullptr) {
                error = "VideoFramePacketMaterializer: failed to allocate hardware-map frame.";
                return false;
            }
        } else {
            av_frame_unref(mappedFrame_.get());
        }
        return true;
    };

    if (!ensureMappedFrame()) {
        return false;
    }
    int mapStatus = av_hwframe_map(mappedFrame_.get(),
                                   packet.frame.get(),
                                   AV_HWFRAME_MAP_READ | AV_HWFRAME_MAP_DIRECT);
    if (mapStatus >= 0 && mappedFrame_->format != AV_PIX_FMT_VULKAN) {
        frameOut = mappedFrame_.get();
        lastResolveMode_ = SoftwareFrameResolveMode::DirectMap;
        return true;
    }
    av_frame_unref(mappedFrame_.get());

    if (!ensureMappedFrame()) {
        return false;
    }
    mapStatus = av_hwframe_map(mappedFrame_.get(),
                               packet.frame.get(),
                               AV_HWFRAME_MAP_READ);
    if (mapStatus >= 0 && mappedFrame_->format != AV_PIX_FMT_VULKAN) {
        frameOut = mappedFrame_.get();
        lastResolveMode_ = SoftwareFrameResolveMode::Mapped;
        return true;
    }
    av_frame_unref(mappedFrame_.get());

    if (softwareFrame_ == nullptr) {
        softwareFrame_.reset(av_frame_alloc());
        if (softwareFrame_ == nullptr) {
            error = "VideoFramePacketMaterializer: failed to allocate hardware-transfer frame.";
            return false;
        }
    } else {
        av_frame_unref(softwareFrame_.get());
    }

    if (av_hwframe_transfer_data(softwareFrame_.get(), packet.frame.get(), 0) < 0) {
        error = "VideoFramePacketMaterializer: failed to transfer Vulkan frame to host memory.";
        return false;
    }

    frameOut = softwareFrame_.get();
    lastResolveMode_ = SoftwareFrameResolveMode::Transfer;
    return true;
}

bool VideoFramePacketMaterializer::resolveRgb24(const VideoFramePacket& packet,
                                                const std::vector<std::uint8_t>*& rgbOut,
                                                std::string& error) {
    rgbOut = nullptr;
    if (packet.hasRgb24()) {
        rgbOut = &packet.rgb24;
        return true;
    }

    AVFrame* softwareFrame = nullptr;
    if (!resolveSoftwareFrame(packet, softwareFrame, error)) {
        return false;
    }
    if (!avFrameToRgb24(softwareFrame,
                        packet.width,
                        packet.height,
                        rgbScratch_,
                        error)) {
        return false;
    }
    rgbOut = &rgbScratch_;
    return true;
}

SoftwareFrameResolveMode VideoFramePacketMaterializer::lastSoftwareFrameResolveMode() const {
    return lastResolveMode_;
}

const std::vector<std::uint8_t>& VideoFramePacketMaterializer::rgbScratch() const {
    return rgbScratch_;
}

RgbVideoPipeSource::RgbVideoPipeSource()
    : impl_(std::make_unique<Impl>()) {}

RgbVideoPipeSource::~RgbVideoPipeSource() {
    close();
}

bool RgbVideoPipeSource::open(const std::string& path,
                              const int width,
                              const int height,
                              const RgbVideoPipeSourceOptions& options,
                              std::string& error) {
    close();
    if (width <= 0 || height <= 0) {
        error = "RgbVideoPipeSource: invalid frame dimensions.";
        return false;
    }

    std::ostringstream cmd;
    cmd << "ffmpeg -hide_banner -loglevel error ";
    if (options.durationSec > 0.0) {
        cmd << "-t " << options.durationSec << ' ';
    }
    cmd << "-i " << quoteArg(path)
        << " -f rawvideo -pix_fmt rgb24 pipe:1 2>/dev/null";

    impl_->pipe = popen(cmd.str().c_str(), "r");
    if (impl_->pipe == nullptr) {
        error = "Failed to open FFmpeg decode pipe for " + path;
        return false;
    }

    impl_->frameBytes =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3u;
    tunePipeIo(impl_->pipe, options.stdioBufferBytes, options.pipeBytes);
    return true;
}

RgbVideoFrameReadStatus RgbVideoPipeSource::readFrame(std::vector<std::uint8_t>& rgb,
                                                      std::string& error) {
    if (impl_->pipe == nullptr) {
        error = "RgbVideoPipeSource: source not open.";
        return RgbVideoFrameReadStatus::Error;
    }

    rgb.resize(impl_->frameBytes);
    std::size_t totalRead = 0u;
    while (totalRead < impl_->frameBytes) {
        const std::size_t count = std::fread(
            rgb.data() + totalRead, 1, impl_->frameBytes - totalRead, impl_->pipe);
        if (count == 0u) {
            break;
        }
        totalRead += count;
    }

    if (totalRead == 0u) {
        return RgbVideoFrameReadStatus::EndOfStream;
    }
    if (totalRead != impl_->frameBytes) {
        error = "RgbVideoPipeSource: partial frame read (" + std::to_string(totalRead)
              + "/" + std::to_string(impl_->frameBytes) + " bytes).";
        return RgbVideoFrameReadStatus::Error;
    }
    return RgbVideoFrameReadStatus::FrameReady;
}

void RgbVideoPipeSource::close() {
    if (impl_->pipe != nullptr) {
        (void)pclose(impl_->pipe);
        impl_->pipe = nullptr;
    }
    impl_->frameBytes = 0u;
}

std::size_t RgbVideoPipeSource::frameBytes() const {
    return impl_->frameBytes;
}

VideoFrameSource::VideoFrameSource()
    : impl_(std::make_unique<Impl>()) {}

VideoFrameSource::~VideoFrameSource() {
    close();
}

bool VideoFrameSource::open(const std::string& path,
                            const int width,
                            const int height,
                            const RgbVideoFrameSourceOptions& options,
                            std::string& error) {
    close();
    impl_->width = width;
    impl_->height = height;
    impl_->mode = RgbVideoSourceMode::RawPipe;

    auto openPipeFallback = [&](std::string& fallbackError) -> bool {
        impl_->mode = RgbVideoSourceMode::RawPipe;
        return impl_->pipeSource.open(path, width, height, options.pipe, fallbackError);
    };

    if (options.preferredMode == RgbVideoSourceMode::VulkanTransfer ||
        options.preferredMode == RgbVideoSourceMode::VulkanHardware) {
        std::string vulkanError;
        if (impl_->vulkanReader.open(path, vulkanError)) {
            const int readerWidth = impl_->vulkanReader.width();
            const int readerHeight = impl_->vulkanReader.height();
            if (readerWidth > 0 && readerHeight > 0 &&
                readerWidth == width && readerHeight == height) {
                impl_->width = readerWidth;
                impl_->height = readerHeight;
                impl_->mode = options.preferredMode;
                error.clear();
                return true;
            }
            std::ostringstream mismatch;
            mismatch << "VideoFrameSource: Vulkan decode dimensions "
                     << readerWidth << "x" << readerHeight
                     << " did not match the expected " << width << "x" << height << '.';
            vulkanError = mismatch.str();
        }
        impl_->vulkanReader.close();
        if (!options.allowFallback) {
            error = vulkanError;
            return false;
        }
        if (openPipeFallback(error)) {
            return true;
        }
        if (!vulkanError.empty()) {
            error += "\nFallback reason: " + vulkanError;
        }
        return false;
    }

    return openPipeFallback(error);
}

RgbVideoFrameReadStatus VideoFrameSource::readFrame(VideoFramePacket& packet,
                                                    std::string& error) {
    packet.reset();
    packet.width = impl_->width;
    packet.height = impl_->height;
    packet.sourceMode = impl_->mode;

    if (impl_->mode == RgbVideoSourceMode::RawPipe) {
        return impl_->pipeSource.readFrame(packet.rgb24, error);
    }

    AVFrame* frame = nullptr;
    const VulkanFrameReadMode readMode =
        impl_->mode == RgbVideoSourceMode::VulkanHardware
            ? VulkanFrameReadMode::PreserveHardware
            : VulkanFrameReadMode::TransferToHost;
    if (!impl_->vulkanReader.readFrame(frame, readMode, error)) {
        return RgbVideoFrameReadStatus::Error;
    }
    if (frame == nullptr) {
        return RgbVideoFrameReadStatus::EndOfStream;
    }
    if (!cloneFrameReference(frame, packet.frame, error)) {
        packet.reset();
        return RgbVideoFrameReadStatus::Error;
    }
    packet.width = frame->width;
    packet.height = frame->height;
    return RgbVideoFrameReadStatus::FrameReady;
}

void VideoFrameSource::close() {
    impl_->pipeSource.close();
    impl_->vulkanReader.close();
    impl_->width = 0;
    impl_->height = 0;
    impl_->mode = RgbVideoSourceMode::RawPipe;
}

RgbVideoSourceMode VideoFrameSource::activeMode() const {
    return impl_->mode;
}

RgbVideoFrameSource::RgbVideoFrameSource()
    : impl_(std::make_unique<Impl>()) {}

RgbVideoFrameSource::~RgbVideoFrameSource() {
    close();
}

bool RgbVideoFrameSource::open(const std::string& path,
                               const int width,
                               const int height,
                               const RgbVideoFrameSourceOptions& options,
                               std::string& error) {
    return impl_->source.open(path, width, height, options, error);
}

RgbVideoFrameReadStatus RgbVideoFrameSource::readFrame(std::vector<std::uint8_t>& rgb,
                                                       std::string& error) {
    VideoFramePacket packet;
    const auto status = impl_->source.readFrame(packet, error);
    if (status != RgbVideoFrameReadStatus::FrameReady) {
        return status;
    }
    if (packet.hasRgb24()) {
        rgb = std::move(packet.rgb24);
        return RgbVideoFrameReadStatus::FrameReady;
    }
    if (!videoFramePacketToRgb24(packet, rgb, error)) {
        return RgbVideoFrameReadStatus::Error;
    }
    return RgbVideoFrameReadStatus::FrameReady;
}

void RgbVideoFrameSource::close() {
    impl_->source.close();
}

RgbVideoSourceMode RgbVideoFrameSource::activeMode() const {
    return impl_->source.activeMode();
}

RgbVideoFrameWriter::RgbVideoFrameWriter()
    : impl_(std::make_unique<Impl>()) {}

RgbVideoFrameWriter::~RgbVideoFrameWriter() {
    close();
}

bool RgbVideoFrameWriter::open(const std::string& path,
                               const int width,
                               const int height,
                               const AVRational fps,
                               std::string& error) {
    close();
    if (width <= 0 || height <= 0) {
        error = "RgbVideoFrameWriter: invalid frame dimensions.";
        return false;
    }
    if (!impl_->writer.open(path, width, height, fps, error)) {
        return false;
    }
    impl_->width = width;
    impl_->height = height;
    impl_->expectedBytes = static_cast<std::size_t>(width) *
                           static_cast<std::size_t>(height) * 3u;
    return true;
}

bool RgbVideoFrameWriter::writeFrame(const std::uint8_t* rgb,
                                     std::string& error) {
    if (rgb == nullptr) {
        error = "RgbVideoFrameWriter: null RGB frame.";
        return false;
    }
    if (impl_->expectedBytes == 0u || impl_->width <= 0 || impl_->height <= 0) {
        error = "RgbVideoFrameWriter: writer not open.";
        return false;
    }

    AVFrame* frame = rgb24ToAvFrame(rgb, impl_->width, impl_->height, error);
    if (frame == nullptr) {
        return false;
    }
    const bool ok = impl_->writer.writeFrame(frame, error);
    av_frame_free(&frame);
    return ok;
}

void RgbVideoFrameWriter::close() {
    impl_->writer.close();
    impl_->width = 0;
    impl_->height = 0;
    impl_->expectedBytes = 0u;
}

RgbVideoSession::RgbVideoSession()
    : impl_(std::make_unique<Impl>()) {}

RgbVideoSession::~RgbVideoSession() {
    close();
}

bool RgbVideoSession::open(const RgbVideoSessionOptions& options,
                           std::string& error) {
    close();
    RgbVideoFrameSourceOptions sourceOptions;
    sourceOptions.preferredMode = options.preferredSourceMode;
    sourceOptions.allowFallback = options.allowSourceFallback;
    sourceOptions.pipe = options.sourcePipe;
    if (!impl_->source.open(options.inputVideo,
                            options.inputWidth,
                            options.inputHeight,
                            sourceOptions,
                            error)) {
        return false;
    }
    if (!impl_->writer.open(options.outputVideo,
                            options.outputWidth,
                            options.outputHeight,
                            options.fps,
                            error)) {
        impl_->source.close();
        return false;
    }
    return true;
}

RgbVideoFrameReadStatus RgbVideoSession::readFrame(std::vector<std::uint8_t>& rgb,
                                                   std::string& error) {
    return impl_->source.readFrame(rgb, error);
}

bool RgbVideoSession::writeFrame(const std::uint8_t* rgb,
                                 std::string& error) {
    return impl_->writer.writeFrame(rgb, error);
}

void RgbVideoSession::close() {
    impl_->writer.close();
    impl_->source.close();
}

RgbVideoSourceMode RgbVideoSession::activeSourceMode() const {
    return impl_->source.activeMode();
}

SyncRgbVideoFrameSink::SyncRgbVideoFrameSink(RgbVideoFrameWriter& writer,
                                             const std::size_t frameBytes)
    : writer_(&writer),
      frameBytes_(frameBytes) {}

bool SyncRgbVideoFrameSink::acquireBuffer(std::vector<std::uint8_t>& buffer,
                                          std::string& error) {
    if (writer_ == nullptr || frameBytes_ == 0u) {
        error = "SyncRgbVideoFrameSink: sink not initialized.";
        return false;
    }
    if (writerFailed_) {
        error = writerError_.empty()
            ? "SyncRgbVideoFrameSink: writer has already failed."
            : writerError_;
        return false;
    }
    if (buffer.size() != frameBytes_) {
        buffer.resize(frameBytes_);
    }
    return true;
}

bool SyncRgbVideoFrameSink::submitFrame(const int /*frameIdx*/,
                                        std::vector<std::uint8_t>&& buffer,
                                        std::string& error) {
    if (writer_ == nullptr) {
        error = "SyncRgbVideoFrameSink: sink not initialized.";
        return false;
    }

    const auto writeStart = std::chrono::steady_clock::now();
    const bool ok = writer_->writeFrame(buffer.data(), error);
    totalWriteTime_ += std::chrono::steady_clock::now() - writeStart;
    if (!ok) {
        writerFailed_ = true;
        writerError_ = error;
        return false;
    }
    return true;
}

bool SyncRgbVideoFrameSink::writerFailed() const {
    return writerFailed_;
}

std::string SyncRgbVideoFrameSink::writerFailure() const {
    return writerError_;
}

std::chrono::nanoseconds SyncRgbVideoFrameSink::totalWriteTime() const {
    return totalWriteTime_;
}

AsyncRgbVideoPipeEncoder::AsyncRgbVideoPipeEncoder()
    : impl_(std::make_unique<Impl>()) {}

AsyncRgbVideoPipeEncoder::~AsyncRgbVideoPipeEncoder() {
    (void)finish(true);
}

bool AsyncRgbVideoPipeEncoder::open(const RgbVideoPipeEncoderOptions& options,
                                    std::string& error) {
    (void)finish(true);

    if (options.width <= 0 || options.height <= 0 || options.fps <= 0.0) {
        error = "AsyncRgbVideoPipeEncoder: invalid output geometry or fps.";
        return false;
    }
    if (options.inputVideo.empty() || options.outputVideo.empty()) {
        error = "AsyncRgbVideoPipeEncoder: input/output paths are required.";
        return false;
    }

    impl_->frameBytes = static_cast<std::size_t>(options.width)
                      * static_cast<std::size_t>(options.height) * 3u;
    impl_->queueDepth = std::max<std::size_t>(options.queueDepth, 1u);
    impl_->logPath = options.stderrLogPath;
    impl_->writeTimeNs.store(0, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lk(impl_->mutex);
        impl_->queue.clear();
        impl_->freeBuffers.clear();
        impl_->inputClosed = false;
        impl_->writerFailed = false;
        impl_->writerError.clear();
    }

    std::ostringstream encodeCmd;
    encodeCmd << "ffmpeg -y -hide_banner -loglevel error "
              << "-f rawvideo -pix_fmt rgb24 "
              << "-s " << options.width << "x" << options.height << " "
              << "-r " << options.fps << " "
              << "-i pipe:0 "
              << "-i " << quoteArg(options.inputVideo) << " "
              << "-map 0:v:0 -map 1:a? ";
    if (options.directOutputEncode) {
        const std::string codec = options.outputCodec.empty() ? "libx264" : options.outputCodec;
        encodeCmd << "-vf " << quoteArg("format=yuv420p") << ' '
                  << "-c:v " << codec << ' ';
        if (!options.outputProfile.empty()) {
            encodeCmd << "-profile:v " << options.outputProfile << ' ';
        }
        if (options.outputThreads > 0) {
            encodeCmd << "-threads " << options.outputThreads << ' ';
        }
        const std::string preset = options.outputPreset.empty() ? "medium" : options.outputPreset;
        encodeCmd << "-crf " << options.outputCrf << ' '
                  << "-preset " << preset << ' '
                  << "-c:a copy -shortest ";
    } else {
        encodeCmd << "-c:v ffv1 -level 3 -slicecrc 1 -c:a copy ";
    }
    encodeCmd << quoteArg(options.outputVideo) << ' ';
    if (impl_->logPath.empty()) {
        encodeCmd << "2>/dev/null";
    } else {
        encodeCmd << "2> " << quoteArg(impl_->logPath.string());
    }

    impl_->pipe = popen(encodeCmd.str().c_str(), "w");
    if (impl_->pipe == nullptr) {
        error = "Failed to open FFmpeg encode pipe for " + options.outputVideo;
        return false;
    }
    tunePipeIo(impl_->pipe, options.stdioBufferBytes, options.pipeBytes);

    {
        std::lock_guard<std::mutex> lk(impl_->mutex);
        for (std::size_t i = 0; i < impl_->queueDepth + 1u; ++i) {
            impl_->freeBuffers.emplace_back(impl_->frameBytes);
        }
    }

    impl_->writerThread = std::thread([this]() {
        using Clock = std::chrono::steady_clock;
        while (true) {
            Impl::QueuedFrame frame;
            {
                std::unique_lock<std::mutex> lk(impl_->mutex);
                impl_->cv.wait(lk, [&]() {
                    return !impl_->queue.empty() || impl_->inputClosed;
                });
                if (impl_->queue.empty()) {
                    break;
                }
                frame = std::move(impl_->queue.front());
                impl_->queue.pop_front();
            }
            impl_->cv.notify_all();

            const auto writeStart = Clock::now();
            const std::size_t written =
                std::fwrite(frame.pixels.data(), 1, impl_->frameBytes, impl_->pipe);
            const auto elapsedNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
                Clock::now() - writeStart).count();
            impl_->writeTimeNs.fetch_add(elapsedNs, std::memory_order_relaxed);
            if (written != impl_->frameBytes) {
                std::lock_guard<std::mutex> lk(impl_->mutex);
                impl_->writerFailed = true;
                impl_->writerError =
                    "Encode pipe write failed at frame " + std::to_string(frame.frameIdx)
                    + "; the FFmpeg encoder exited early.";
                impl_->inputClosed = true;
                impl_->queue.clear();
                impl_->cv.notify_all();
                break;
            }

            {
                std::lock_guard<std::mutex> lk(impl_->mutex);
                impl_->freeBuffers.push_back(std::move(frame.pixels));
            }
            impl_->cv.notify_all();
        }
    });

    return true;
}

bool AsyncRgbVideoPipeEncoder::acquireBuffer(std::vector<std::uint8_t>& buffer,
                                             std::string& error) {
    std::unique_lock<std::mutex> lk(impl_->mutex);
    impl_->cv.wait(lk, [&]() {
        return !impl_->freeBuffers.empty() || impl_->writerFailed;
    });
    if (impl_->writerFailed) {
        error = impl_->writerError.empty()
            ? "Encode pipe failed while acquiring an output buffer."
            : impl_->writerError;
        return false;
    }
    buffer = std::move(impl_->freeBuffers.front());
    impl_->freeBuffers.pop_front();
    lk.unlock();
    if (buffer.size() != impl_->frameBytes) {
        buffer.resize(impl_->frameBytes);
    }
    return true;
}

bool AsyncRgbVideoPipeEncoder::submitFrame(int frameIdx,
                                           std::vector<std::uint8_t>&& buffer,
                                           std::string& error) {
    std::unique_lock<std::mutex> lk(impl_->mutex);
    impl_->cv.wait(lk, [&]() {
        return impl_->queue.size() < impl_->queueDepth || impl_->writerFailed;
    });
    if (impl_->writerFailed) {
        error = impl_->writerError.empty()
            ? "Encode pipe failed while submitting a frame."
            : impl_->writerError;
        return false;
    }
    impl_->queue.push_back(Impl::QueuedFrame{frameIdx, std::move(buffer)});
    lk.unlock();
    impl_->cv.notify_all();
    return true;
}

int AsyncRgbVideoPipeEncoder::finish(const bool discardPending) {
    int status = 0;
    {
        std::lock_guard<std::mutex> lk(impl_->mutex);
        if (impl_->pipe == nullptr) {
            return 0;
        }
        if (discardPending) {
            impl_->queue.clear();
        }
        impl_->inputClosed = true;
    }
    impl_->cv.notify_all();
    if (impl_->writerThread.joinable()) {
        impl_->writerThread.join();
    }
    status = pclose(impl_->pipe);
    impl_->pipe = nullptr;
    return status;
}

bool AsyncRgbVideoPipeEncoder::writerFailed() const {
    std::lock_guard<std::mutex> lk(impl_->mutex);
    return impl_->writerFailed;
}

std::string AsyncRgbVideoPipeEncoder::writerFailure() const {
    std::lock_guard<std::mutex> lk(impl_->mutex);
    return impl_->writerError;
}

std::chrono::nanoseconds AsyncRgbVideoPipeEncoder::totalWriteTime() const {
    return std::chrono::nanoseconds{
        impl_->writeTimeNs.load(std::memory_order_relaxed)};
}

const std::filesystem::path& AsyncRgbVideoPipeEncoder::stderrLogPath() const {
    return impl_->logPath;
}

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

bool extractVideoFrameRgb24(const std::string& path,
                            double timeSec,
                            int& width,
                            int& height,
                            std::vector<std::uint8_t>& data,
                            std::string& error) {
    const auto probe = probeVideoStream(path, error);
    if (!probe.has_value()) {
        return false;
    }

    width = static_cast<int>(probe->width);
    height = static_cast<int>(probe->height);
    if (width <= 0 || height <= 0) {
        error = "Video probe returned invalid frame dimensions.";
        return false;
    }

    const std::size_t expected = static_cast<std::size_t>(width) *
                                 static_cast<std::size_t>(height) * 3u;
    data.resize(expected);

    std::ostringstream cmd;
    cmd << "ffmpeg -hide_banner -loglevel quiet ";
    if (timeSec > 0.0) {
        cmd << "-ss " << timeSec << ' ';
    }
    cmd << "-i " << quoteArg(path)
        << " -frames:v 1 -f rawvideo -pix_fmt rgb24 pipe:1 2>/dev/null";

    FILE* pipe = popen(cmd.str().c_str(), "r");
    if (pipe == nullptr) {
        error = "Failed to start FFmpeg while extracting a preview frame.";
        return false;
    }

    std::size_t totalRead = 0;
    while (totalRead < expected) {
        const std::size_t bytes = std::fread(data.data() + totalRead,
                                             1,
                                             expected - totalRead,
                                             pipe);
        if (bytes == 0) {
            break;
        }
        totalRead += bytes;
    }

    const int exitCode = pclose(pipe);
    if (exitCode != 0 || totalRead != expected) {
        error = "FFmpeg could not extract a full preview frame from: " + path;
        return false;
    }

    error.clear();
    return true;
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

    const AVFrame* sourceFrame = frame;
    AvFramePtr transferredFrame;
    if (isVulkanHardwareFrame(frame)) {
        AVFrame* softwareFrame = av_frame_alloc();
        if (softwareFrame == nullptr) {
            error = "avFrameToRgb24: failed to allocate transfer frame";
            return false;
        }
        if (av_hwframe_transfer_data(softwareFrame, const_cast<AVFrame*>(frame), 0) < 0) {
            av_frame_free(&softwareFrame);
            error = "avFrameToRgb24: failed to transfer Vulkan frame to host memory";
            return false;
        }
        transferredFrame.reset(softwareFrame);
        sourceFrame = transferredFrame.get();
    }

    AVPixelFormat srcFmt = static_cast<AVPixelFormat>(sourceFrame->format);

    SwsKey key{sourceFrame->width, sourceFrame->height, static_cast<int>(srcFmt),
               width, height, static_cast<int>(AV_PIX_FMT_RGB24)};
    SwsContext* sws = getOrCreateSwsContext(key);

    if (!sws) {
        error = "avFrameToRgb24: sws_getContext failed";
        return false;
    }

    uint8_t* dst[4] = { rgb.data(), nullptr, nullptr, nullptr };
    int dstLinesize[4] = { width * 3, 0, 0, 0 };

    const uint8_t* src[4] = {
        sourceFrame->data[0], sourceFrame->data[1], sourceFrame->data[2], nullptr
    };
    const int srcLinesize[4] = {
        sourceFrame->linesize[0], sourceFrame->linesize[1], sourceFrame->linesize[2], 0
    };

    sws_scale(sws, src, srcLinesize, 0, sourceFrame->height, dst, dstLinesize);

    return true;
}

bool avFrameSupportsDirectTensorPacking(const AVFrame* frame) {
    return detectDirectTensorPackingMode(frame) != DirectTensorPackingMode::Unsupported;
}

bool avFrameTileToNchwFp32(const AVFrame* frame,
                           int tileX,
                           int tileY,
                           int tileWidth,
                           int tileHeight,
                           float* tensor,
                           std::string& error) {
    if (tensor == nullptr) {
        error = "avFrameTileToNchwFp32: null tensor";
        return false;
    }
    return packDirectFrameTileGeneric(
        frame,
        tileX,
        tileY,
        tileWidth,
        tileHeight,
        [&](const std::size_t dstOffset,
            const float red,
            const float green,
            const float blue,
            const std::size_t hw) {
            tensor[dstOffset] = red;
            tensor[hw + dstOffset] = green;
            tensor[(2u * hw) + dstOffset] = blue;
        },
        error);
}

bool avFrameTileToNchwFp16(const AVFrame* frame,
                           int tileX,
                           int tileY,
                           int tileWidth,
                           int tileHeight,
                           std::uint16_t* tensor,
                           std::string& error) {
    if (tensor == nullptr) {
        error = "avFrameTileToNchwFp16: null tensor";
        return false;
    }
    return packDirectFrameTileGeneric(
        frame,
        tileX,
        tileY,
        tileWidth,
        tileHeight,
        [&](const std::size_t dstOffset,
            const float red,
            const float green,
            const float blue,
            const std::size_t hw) {
            tensor[dstOffset] = floatToHalfBits(red);
            tensor[hw + dstOffset] = floatToHalfBits(green);
            tensor[(2u * hw) + dstOffset] = floatToHalfBits(blue);
        },
        error);
}

bool videoFramePacketToRgb24(const VideoFramePacket& packet,
                             std::vector<std::uint8_t>& rgb,
                             std::string& error) {
    VideoFramePacketMaterializer materializer;
    const std::vector<std::uint8_t>* rgbView = nullptr;
    if (!materializer.resolveRgb24(packet, rgbView, error) || rgbView == nullptr) {
        if (error.empty()) {
            error = "videoFramePacketToRgb24: packet contains no RGB data or AVFrame payload";
        }
        return false;
    }
    rgb = *rgbView;
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
