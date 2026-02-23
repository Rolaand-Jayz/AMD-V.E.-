#pragma once
// ─────────────────────────────────────────────────────────────────
// frame_io.hpp — Frame I/O utilities for AI inference pipelines
//
// Provides functions to load/save PNG frames as raw RGB24 data
// and convert between RGB24 pixel buffers and fp32 NCHW tensors.
//
// PNG I/O uses FFmpeg CLI (always available at runtime) so no
// additional image-loading library is required.
// ─────────────────────────────────────────────────────────────────

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace ave {
namespace frame_io {

// ── PNG dimension probe ─────────────────────────────────────────
// Reads width/height directly from the PNG IHDR chunk (no process
// spawn).  Returns false if the file cannot be opened or the IHDR
// is malformed.
bool readPngDimensions(const std::string& path, int& width, int& height);

// ── PNG → raw RGB24 ─────────────────────────────────────────────
// Loads a PNG file to an interleaved RGB24 byte buffer using FFmpeg.
// Sets width/height and fills data (size = width * height * 3).
bool loadPngRgb24(const std::string& path,
                  int& width, int& height,
                  std::vector<std::uint8_t>& data,
                  std::string& error);

// ── raw RGB24 → PNG ─────────────────────────────────────────────
// Saves an interleaved RGB24 byte buffer as a PNG file using FFmpeg.
bool saveRgb24ToPng(const std::string& path,
                    int width, int height,
                    const std::uint8_t* data,
                    std::string& error);

// ── RGB24 → fp32 NCHW tensor ────────────────────────────────────
// Converts interleaved RGB24 uint8 [H,W,3] to a contiguous fp32
// tensor in NCHW layout [1,3,H,W] normalised to [0,1].
void rgb24ToNchwFp32(const std::uint8_t* rgb,
                     int width, int height,
                     std::vector<float>& tensor);

// ── fp32 NCHW tensor → RGB24 ────────────────────────────────────
// Converts a contiguous fp32 NCHW tensor [1,C,H,W] (C≥3, values
// in [0,1]) back to interleaved RGB24 uint8.  Clamps to [0,255].
void nchwFp32ToRgb24(const float* tensor,
                     int channels, int width, int height,
                     std::vector<std::uint8_t>& rgb);

// ── Directory listing ───────────────────────────────────────────
// Returns a sorted vector of all .png file paths in a directory.
std::vector<std::filesystem::path> listPngFramesSorted(const std::string& dir);

}  // namespace frame_io
}  // namespace ave
