#pragma once
// ─────────────────────────────────────────────────────────────────
// frame_io.hpp — Frame I/O utilities for AI inference pipelines
//
// Provides functions to load/save PNG frames as raw RGB24 data
// and convert between RGB24 pixel buffers and fp32 NCHW tensors.
//
// Also provides VulkanVideoReader and VulkanVideoWriter for direct
// FFmpeg Vulkan hardware frame (AVVkFrame) extraction and encoding.
// ─────────────────────────────────────────────────────────────────

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>
#include <memory>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/pixdesc.h>

struct AVVulkanDeviceContext;
}

namespace ave {

class VulkanRuntime;

namespace frame_io {

// ── Vulkan Hardware Frame I/O ───────────────────────────────────

enum class VulkanFrameReadMode {
    TransferToHost,
    PreserveHardware,
};

enum class SoftwareFrameResolveMode {
    None,
    PacketFrame,
    DirectMap,
    Mapped,
    Transfer,
};

// Returns true when the frame still wraps a Vulkan hardware surface rather
// than a transferred software frame.
bool isVulkanHardwareFrame(const AVFrame* frame);

class VulkanVideoReader {
public:
    VulkanVideoReader();
    ~VulkanVideoReader();

    bool open(const std::string& path, std::string& error);
    // Legacy behavior: decode and transfer Vulkan frames to host-visible
    // software frames before returning them.
    bool readFrame(AVFrame*& outFrame, std::string& error);
    // Explicit transfer policy. PreserveHardware keeps AV_PIX_FMT_VULKAN
    // frames intact for future interop callers; TransferToHost preserves the
    // current behavior for existing callers.
    bool readFrame(AVFrame*& outFrame, VulkanFrameReadMode mode, std::string& error);
    void close();

    int width() const;
    int height() const;
    AVRational frameRate() const;
    const AVBufferRef* hwDeviceContext() const;
    const AVVulkanDeviceContext* vulkanDeviceContext() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class VulkanHardwareFrameStager {
public:
    VulkanHardwareFrameStager();
    ~VulkanHardwareFrameStager();

    bool stageToSoftwareFrame(const AVFrame* hardwareFrame,
                              VulkanRuntime& runtime,
                              AVFrame*& frameOut,
                              std::string& error);
    void reset();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class VulkanVideoWriter {
public:
    VulkanVideoWriter();
    ~VulkanVideoWriter();

    bool open(const std::string& path, int width, int height, AVRational fps, std::string& error);
    bool writeFrame(AVFrame* frame, std::string& error);
    void close();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class RgbVideoFrameWriter {
public:
    RgbVideoFrameWriter();
    ~RgbVideoFrameWriter();

    bool open(const std::string& path, int width, int height, AVRational fps, std::string& error);
    bool writeFrame(const std::uint8_t* rgb, std::string& error);
    void close();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

enum class RgbVideoFrameReadStatus {
    FrameReady,
    EndOfStream,
    Error,
};

struct RgbVideoPipeSourceOptions {
    double durationSec = 0.0;
    std::size_t stdioBufferBytes = 0u;
    int pipeBytes = 0;
};

class RgbVideoPipeSource {
public:
    RgbVideoPipeSource();
    ~RgbVideoPipeSource();

    bool open(const std::string& path,
              int width,
              int height,
              const RgbVideoPipeSourceOptions& options,
              std::string& error);
    RgbVideoFrameReadStatus readFrame(std::vector<std::uint8_t>& rgb,
                                      std::string& error);
    void close();

    std::size_t frameBytes() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

enum class RgbVideoSourceMode {
    RawPipe,
    VulkanTransfer,
    VulkanHardware,
};

struct RgbVideoFrameSourceOptions {
    RgbVideoSourceMode preferredMode = RgbVideoSourceMode::RawPipe;
    bool allowFallback = true;
    RgbVideoPipeSourceOptions pipe;
};

struct AvFrameDeleter {
    void operator()(AVFrame* frame) const;
};

using AvFramePtr = std::unique_ptr<AVFrame, AvFrameDeleter>;

struct VideoFramePacket {
    int width = 0;
    int height = 0;
    RgbVideoSourceMode sourceMode = RgbVideoSourceMode::RawPipe;
    std::vector<std::uint8_t> rgb24;
    AvFramePtr frame;

    bool hasRgb24() const;
    bool hasFrame() const;
    bool isHardwareFrame() const;
    void reset();
};

// Reusable packet materializer for backends and loop wrappers. It keeps
// hardware-frame transfer scratch and RGB scratch local to the caller so
// packet-to-CPU conversion does not allocate fresh AVFrames every frame.
class VideoFramePacketMaterializer {
public:
    VideoFramePacketMaterializer() = default;
    VideoFramePacketMaterializer(const VideoFramePacketMaterializer&) = delete;
    VideoFramePacketMaterializer& operator=(const VideoFramePacketMaterializer&) = delete;
    VideoFramePacketMaterializer(VideoFramePacketMaterializer&&) noexcept = default;
    VideoFramePacketMaterializer& operator=(VideoFramePacketMaterializer&&) noexcept = default;
    ~VideoFramePacketMaterializer() = default;

    // Resolve the packet to a CPU-accessible AVFrame. RGB-only packets fail.
    bool resolveSoftwareFrame(const VideoFramePacket& packet,
                              AVFrame*& frameOut,
                              std::string& error);

    // Resolve the packet to RGB24 bytes. Embedded RGB24 is reused directly;
    // AVFrame payloads are converted into the internal RGB scratch.
    bool resolveRgb24(const VideoFramePacket& packet,
                      const std::vector<std::uint8_t>*& rgbOut,
                      std::string& error);

    SoftwareFrameResolveMode lastSoftwareFrameResolveMode() const;

    const std::vector<std::uint8_t>& rgbScratch() const;

private:
    std::vector<std::uint8_t> rgbScratch_;
    AvFramePtr mappedFrame_;
    AvFramePtr softwareFrame_;
    SoftwareFrameResolveMode lastResolveMode_ = SoftwareFrameResolveMode::None;
};

class VideoFrameSource {
public:
    VideoFrameSource();
    ~VideoFrameSource();

    bool open(const std::string& path,
              int width,
              int height,
              const RgbVideoFrameSourceOptions& options,
              std::string& error);
    RgbVideoFrameReadStatus readFrame(VideoFramePacket& packet,
                                      std::string& error);
    void close();

    RgbVideoSourceMode activeMode() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class RgbVideoFrameSource {
public:
    RgbVideoFrameSource();
    ~RgbVideoFrameSource();

    bool open(const std::string& path,
              int width,
              int height,
              const RgbVideoFrameSourceOptions& options,
              std::string& error);
    RgbVideoFrameReadStatus readFrame(std::vector<std::uint8_t>& rgb,
                                      std::string& error);
    void close();

    RgbVideoSourceMode activeMode() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct RgbVideoSessionOptions {
    std::string inputVideo;
    std::string outputVideo;
    int inputWidth = 0;
    int inputHeight = 0;
    int outputWidth = 0;
    int outputHeight = 0;
    AVRational fps{30, 1};
    RgbVideoSourceMode preferredSourceMode = RgbVideoSourceMode::RawPipe;
    bool allowSourceFallback = true;
    RgbVideoPipeSourceOptions sourcePipe;
};

class RgbVideoSession {
public:
    RgbVideoSession();
    ~RgbVideoSession();

    bool open(const RgbVideoSessionOptions& options, std::string& error);
    RgbVideoFrameReadStatus readFrame(std::vector<std::uint8_t>& rgb,
                                      std::string& error);
    bool writeFrame(const std::uint8_t* rgb, std::string& error);
    void close();

    RgbVideoSourceMode activeSourceMode() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class IRgbVideoFrameSink {
public:
    virtual ~IRgbVideoFrameSink() = default;

    virtual bool acquireBuffer(std::vector<std::uint8_t>& buffer,
                               std::string& error) = 0;
    virtual bool submitFrame(int frameIdx,
                             std::vector<std::uint8_t>&& buffer,
                             std::string& error) = 0;

    virtual bool writerFailed() const = 0;
    virtual std::string writerFailure() const = 0;
    virtual std::chrono::nanoseconds totalWriteTime() const = 0;
};

class SyncRgbVideoFrameSink : public IRgbVideoFrameSink {
public:
    SyncRgbVideoFrameSink(RgbVideoFrameWriter& writer, std::size_t frameBytes);

    bool acquireBuffer(std::vector<std::uint8_t>& buffer,
                       std::string& error) override;
    bool submitFrame(int frameIdx,
                     std::vector<std::uint8_t>&& buffer,
                     std::string& error) override;

    bool writerFailed() const override;
    std::string writerFailure() const override;
    std::chrono::nanoseconds totalWriteTime() const override;

private:
    RgbVideoFrameWriter* writer_ = nullptr;
    std::size_t frameBytes_ = 0u;
    bool writerFailed_ = false;
    std::string writerError_;
    std::chrono::nanoseconds totalWriteTime_{0};
};

struct RgbVideoPipeEncoderOptions {
    std::string inputVideo;
    std::string outputVideo;
    int width = 0;
    int height = 0;
    double fps = 0.0;
    bool directOutputEncode = false;
    std::string outputCodec;
    std::string outputProfile;
    std::string outputPreset;
    int outputCrf = 18;
    int outputThreads = 0;
    std::size_t stdioBufferBytes = 0u;
    int pipeBytes = 0;
    std::size_t queueDepth = 1u;
    std::filesystem::path stderrLogPath;
};

class AsyncRgbVideoPipeEncoder : public IRgbVideoFrameSink {
public:
    AsyncRgbVideoPipeEncoder();
    ~AsyncRgbVideoPipeEncoder();

    bool open(const RgbVideoPipeEncoderOptions& options, std::string& error);
    bool acquireBuffer(std::vector<std::uint8_t>& buffer, std::string& error) override;
    bool submitFrame(int frameIdx,
                     std::vector<std::uint8_t>&& buffer,
                     std::string& error) override;
    int finish(bool discardPending);

    bool writerFailed() const override;
    std::string writerFailure() const override;
    std::chrono::nanoseconds totalWriteTime() const override;
    const std::filesystem::path& stderrLogPath() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ── PNG dimension probe ─────────────────────────────────────────
// Reads width/height directly from the PNG IHDR chunk (no process
// spawn).  Returns false if the file cannot be opened or the IHDR
// is malformed.
bool readPngDimensions(const std::string& path, int& width, int& height);

// Extract a single RGB24 frame from a video using FFmpeg.
bool extractVideoFrameRgb24(const std::string& path,
                            double timeSec,
                            int& width,
                            int& height,
                            std::vector<std::uint8_t>& data,
                            std::string& error);

// Load/save PNGs via FFmpeg raw RGB pipes.
bool loadPngRgb24(const std::string& path,
                  int& width,
                  int& height,
                  std::vector<std::uint8_t>& data,
                  std::string& error);
bool saveRgb24ToPng(const std::string& path,
                    int width,
                    int height,
                    const std::uint8_t* data,
                    std::string& error);

// ── RGB24 → fp32 NCHW tensor ────────────────────────────────────
// Converts interleaved RGB24 uint8 [H,W,3] to a contiguous fp32
// tensor in NCHW layout [1,3,H,W] normalised to [0,1].
void rgb24ToNchwFp32(const std::uint8_t* rgb,
                     int width, int height,
                     std::vector<float>& tensor);

// ── RGB24 → fp16 NCHW tensor ────────────────────────────────────
// Converts interleaved RGB24 uint8 [H,W,3] to IEEE-754 binary16 bits
// in NCHW layout [1,3,H,W] normalised to [0,1].
void rgb24ToNchwFp16(const std::uint8_t* rgb,
                     int width, int height,
                     std::vector<std::uint16_t>& tensor);

// ── fp32 NCHW tensor → RGB24 ────────────────────────────────────
// Converts a contiguous fp32 NCHW tensor [1,C,H,W] (C≥3, values
// in [0,1]) back to interleaved RGB24 uint8.  Clamps to [0,255].
void nchwFp32ToRgb24(const float* tensor,
                     int channels, int width, int height,
                     std::vector<std::uint8_t>& rgb);

// ── fp16 NCHW tensor → RGB24 ────────────────────────────────────
// Converts IEEE-754 binary16 NCHW [1,C,H,W] (C≥3, values in [0,1]) back
// to interleaved RGB24 uint8. Clamps to [0,255].
void nchwFp16ToRgb24(const std::uint16_t* tensor,
                     int channels, int width, int height,
                     std::vector<std::uint8_t>& rgb);

// ── AVFrame conversion utilities ─────────────────────────────────
// Convert any AVFrame to RGB24 bytes using swscale.
// Returns false on conversion failure.
bool avFrameToRgb24(const AVFrame* frame,
                    int width, int height,
                    std::vector<std::uint8_t>& rgb,
                    std::string& error);

// Fast-path tensor packing helpers for software AVFrames whose storage layout
// can be converted directly without routing through swscale first.
bool avFrameSupportsDirectTensorPacking(const AVFrame* frame);
bool avFrameTileToNchwFp32(const AVFrame* frame,
                           int tileX,
                           int tileY,
                           int tileWidth,
                           int tileHeight,
                           float* tensor,
                           std::string& error);
bool avFrameTileToNchwFp16(const AVFrame* frame,
                           int tileX,
                           int tileY,
                           int tileWidth,
                           int tileHeight,
                           std::uint16_t* tensor,
                           std::string& error);

// Convert a decoded frame packet into RGB24 bytes. Reuses embedded RGB24 data
// when available and otherwise converts the owned AVFrame payload.
bool videoFramePacketToRgb24(const VideoFramePacket& packet,
                             std::vector<std::uint8_t>& rgb,
                             std::string& error);

// Create a new AVFrame (YUV420P) from RGB24 bytes.
// Caller owns the returned frame (must call av_frame_free).
AVFrame* rgb24ToAvFrame(const std::uint8_t* rgb,
                        int width, int height,
                        std::string& error);

// ── Directory listing ───────────────────────────────────────────
// Returns a sorted vector of all .png file paths in a directory.
std::vector<std::filesystem::path> listPngFramesSorted(const std::string& dir);

}  // namespace frame_io
}  // namespace ave
