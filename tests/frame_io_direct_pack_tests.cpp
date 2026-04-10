#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

extern "C" {
#include <libavutil/frame.h>
}

#include "ave/frame_io.hpp"

namespace {

void check(const bool condition, const char* message) {
    if (condition) {
        return;
    }
    std::cerr << "frame_io_direct_pack_tests failed: " << message << '\n';
    std::abort();
}

bool nearlyEqual(const float lhs, const float rhs, const float epsilon = 0.01f) {
    return std::fabs(lhs - rhs) <= epsilon;
}

AVFrame* allocateFrame(const AVPixelFormat format, const int width, const int height) {
    AVFrame* frame = av_frame_alloc();
    check(frame != nullptr, "av_frame_alloc should succeed");
    frame->format = static_cast<int>(format);
    frame->width = width;
    frame->height = height;
    check(av_frame_get_buffer(frame, 32) >= 0, "av_frame_get_buffer should succeed");
    check(av_frame_make_writable(frame) >= 0, "av_frame_make_writable should succeed");
    return frame;
}

void testPackedRgbaDirectPack() {
    AVFrame* frame = allocateFrame(AV_PIX_FMT_RGBA, 2, 1);
    frame->data[0][0] = 255;
    frame->data[0][1] = 0;
    frame->data[0][2] = 0;
    frame->data[0][3] = 0;
    frame->data[0][4] = 0;
    frame->data[0][5] = 128;
    frame->data[0][6] = 255;
    frame->data[0][7] = 0;

    check(ave::frame_io::avFrameSupportsDirectTensorPacking(frame),
          "RGBA frame should support direct tensor packing");

    std::vector<float> tensor(6u, 0.0f);
    std::string error;
    check(ave::frame_io::avFrameTileToNchwFp32(frame, 0, 0, 2, 1, tensor.data(), error),
          "RGBA direct tensor packing should succeed");
    check(error.empty(), "RGBA direct tensor packing should not return an error");

    check(nearlyEqual(tensor[0], 1.0f), "First pixel red channel should be 1.0");
    check(nearlyEqual(tensor[1], 0.0f), "Second pixel red channel should be 0.0");
    check(nearlyEqual(tensor[2], 0.0f), "First pixel green channel should be 0.0");
    check(nearlyEqual(tensor[3], 128.0f / 255.0f), "Second pixel green channel should match source");
    check(nearlyEqual(tensor[4], 0.0f), "First pixel blue channel should be 0.0");
    check(nearlyEqual(tensor[5], 1.0f), "Second pixel blue channel should be 1.0");

    av_frame_free(&frame);
}

void testNv12NeutralChromaDirectPack() {
    AVFrame* frame = allocateFrame(AV_PIX_FMT_NV12, 2, 2);
    frame->color_range = AVCOL_RANGE_JPEG;

    frame->data[0][0] = 0;
    frame->data[0][1] = 64;
    frame->data[0][frame->linesize[0]] = 128;
    frame->data[0][frame->linesize[0] + 1] = 255;

    frame->data[1][0] = 128;
    frame->data[1][1] = 128;

    check(ave::frame_io::avFrameSupportsDirectTensorPacking(frame),
          "NV12 frame should support direct tensor packing");

    std::vector<float> tensor(12u, 0.0f);
    std::string error;
    check(ave::frame_io::avFrameTileToNchwFp32(frame, 0, 0, 2, 2, tensor.data(), error),
          "NV12 direct tensor packing should succeed");
    check(error.empty(), "NV12 direct tensor packing should not return an error");

    for (std::size_t i = 0; i < 4u; ++i) {
        check(nearlyEqual(tensor[i], tensor[4u + i]), "Neutral chroma should keep R and G aligned");
        check(nearlyEqual(tensor[i], tensor[8u + i]), "Neutral chroma should keep R and B aligned");
    }
    check(nearlyEqual(tensor[0], 0.0f), "Top-left luma should map to 0.0");
    check(nearlyEqual(tensor[1], 64.0f / 255.0f), "Top-right luma should map correctly");
    check(nearlyEqual(tensor[2], 128.0f / 255.0f), "Bottom-left luma should map correctly");
    check(nearlyEqual(tensor[3], 1.0f), "Bottom-right luma should map to 1.0");

    av_frame_free(&frame);
}

void testFp16PackedRgbaDirectPack() {
    AVFrame* frame = allocateFrame(AV_PIX_FMT_RGBA, 1, 1);
    frame->data[0][0] = 255;
    frame->data[0][1] = 0;
    frame->data[0][2] = 0;
    frame->data[0][3] = 0;

    std::vector<std::uint16_t> tensor(3u, 0u);
    std::string error;
    check(ave::frame_io::avFrameTileToNchwFp16(frame, 0, 0, 1, 1, tensor.data(), error),
          "RGBA fp16 tensor packing should succeed");
    check(error.empty(), "RGBA fp16 tensor packing should not return an error");
    check(tensor[0] == 0x3c00u, "Red channel should encode as fp16 1.0");
    check(tensor[1] == 0x0000u, "Green channel should encode as fp16 0.0");
    check(tensor[2] == 0x0000u, "Blue channel should encode as fp16 0.0");

    av_frame_free(&frame);
}

}  // namespace

int main() {
    testPackedRgbaDirectPack();
    testNv12NeutralChromaDirectPack();
    testFp16PackedRgbaDirectPack();
    return 0;
}
