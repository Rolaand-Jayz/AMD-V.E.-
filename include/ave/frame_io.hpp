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
}

namespace ave {
namespace frame_io {

// ── Vulkan Hardware Frame I/O ───────────────────────────────────

class VulkanVideoReader {
public:
    VulkanVideoReader();
    ~VulkanVideoReader();

    bool open(const std::string& path, std::string& error);
    bool readFrame(AVFrame*& outFrame, std::string& error);
    void close();

    int width() const;
    int height() const;
    AVRational frameRate() const;

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

// ── PNG dimension probe ─────────────────────────────────────────
// Reads width/height directly from the PNG IHDR chunk (no process
// spawn).  Returns false if the file cannot be opened or the IHDR
// is malformed.
bool readPngDimensions(const std::string& path, int& width, int& height);

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
