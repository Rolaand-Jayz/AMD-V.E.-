#include "ave/frame_io.hpp"
#include <iostream>

namespace ave {
namespace frame_io {

struct VulkanVideoReader::Impl {
    AVFormatContext* fmt_ctx = nullptr;
    AVCodecContext* codec_ctx = nullptr;
    AVBufferRef* hw_device_ctx = nullptr;
    int video_stream_idx = -1;
    AVPacket* pkt = nullptr;
    AVFrame* frame = nullptr;
    AVFrame* sw_frame = nullptr;
};

static enum AVPixelFormat get_hw_format(AVCodecContext */*ctx*/, const enum AVPixelFormat *pix_fmts) {
    const enum AVPixelFormat *p;
    for (p = pix_fmts; *p != -1; p++) {
        if (*p == AV_PIX_FMT_VULKAN)
            return *p;
    }
    std::cerr << "Failed to get HW surface format." << std::endl;
    return AV_PIX_FMT_NONE;
}

VulkanVideoReader::VulkanVideoReader() : impl_(std::make_unique<Impl>()) {
    impl_->pkt = av_packet_alloc();
    impl_->frame = av_frame_alloc();
    impl_->sw_frame = av_frame_alloc();
}

VulkanVideoReader::~VulkanVideoReader() {
    close();
}

bool VulkanVideoReader::open(const std::string& path, std::string& error) {
    if (avformat_open_input(&impl_->fmt_ctx, path.c_str(), nullptr, nullptr) < 0) {
        error = "Could not open input file: " + path;
        return false;
    }

    if (avformat_find_stream_info(impl_->fmt_ctx, nullptr) < 0) {
        error = "Could not find stream information";
        return false;
    }

    const AVCodec* codec = nullptr;
    impl_->video_stream_idx = av_find_best_stream(impl_->fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0);
    if (impl_->video_stream_idx < 0) {
        error = "Could not find video stream";
        return false;
    }

    if (av_hwdevice_ctx_create(&impl_->hw_device_ctx, AV_HWDEVICE_TYPE_VULKAN, nullptr, nullptr, 0) < 0) {
        error = "Failed to create Vulkan hardware device context";
        return false;
    }

    impl_->codec_ctx = avcodec_alloc_context3(codec);
    if (!impl_->codec_ctx) {
        error = "Failed to allocate codec context";
        return false;
    }

    avcodec_parameters_to_context(impl_->codec_ctx, impl_->fmt_ctx->streams[impl_->video_stream_idx]->codecpar);
    impl_->codec_ctx->hw_device_ctx = av_buffer_ref(impl_->hw_device_ctx);
    impl_->codec_ctx->get_format = get_hw_format;

    if (avcodec_open2(impl_->codec_ctx, codec, nullptr) < 0) {
        error = "Failed to open codec";
        return false;
    }

    return true;
}

bool VulkanVideoReader::readFrame(AVFrame*& outFrame, std::string& error) {
    while (true) {
        int ret = av_read_frame(impl_->fmt_ctx, impl_->pkt);
        if (ret < 0) {
            if (ret == AVERROR_EOF) {
                outFrame = nullptr;
                return true; // EOF
            }
            error = "Error reading frame";
            return false;
        }

        if (impl_->pkt->stream_index == impl_->video_stream_idx) {
            ret = avcodec_send_packet(impl_->codec_ctx, impl_->pkt);
            if (ret < 0) {
                error = "Error sending packet to decoder";
                av_packet_unref(impl_->pkt);
                return false;
            }

            while (ret >= 0) {
                ret = avcodec_receive_frame(impl_->codec_ctx, impl_->frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    break;
                } else if (ret < 0) {
                    error = "Error receiving frame from decoder";
                    av_packet_unref(impl_->pkt);
                    return false;
                }

                if (impl_->frame->format == AV_PIX_FMT_VULKAN) {
                    outFrame = impl_->frame;
                    av_packet_unref(impl_->pkt);
                    return true;
                } else {
                    // Fallback: transfer to Vulkan
                    if (av_hwframe_transfer_data(impl_->sw_frame, impl_->frame, 0) < 0) {
                        error = "Error transferring frame to Vulkan";
                        av_packet_unref(impl_->pkt);
                        return false;
                    }
                    outFrame = impl_->sw_frame;
                    av_packet_unref(impl_->pkt);
                    return true;
                }
            }
        }
        av_packet_unref(impl_->pkt);
    }
}

void VulkanVideoReader::close() {
    if (impl_->codec_ctx) avcodec_free_context(&impl_->codec_ctx);
    if (impl_->fmt_ctx) avformat_close_input(&impl_->fmt_ctx);
    if (impl_->hw_device_ctx) av_buffer_unref(&impl_->hw_device_ctx);
    if (impl_->pkt) av_packet_free(&impl_->pkt);
    if (impl_->frame) av_frame_free(&impl_->frame);
    if (impl_->sw_frame) av_frame_free(&impl_->sw_frame);
}

int VulkanVideoReader::width() const { return impl_->codec_ctx ? impl_->codec_ctx->width : 0; }
int VulkanVideoReader::height() const { return impl_->codec_ctx ? impl_->codec_ctx->height : 0; }
AVRational VulkanVideoReader::frameRate() const { return impl_->fmt_ctx ? impl_->fmt_ctx->streams[impl_->video_stream_idx]->avg_frame_rate : AVRational{0, 1}; }

struct VulkanVideoWriter::Impl {
    AVFormatContext* fmt_ctx = nullptr;
    AVCodecContext* codec_ctx = nullptr;
    AVStream* video_stream = nullptr;
    AVPacket* pkt = nullptr;
    AVFrame* sw_frame = nullptr;
    int frame_count = 0;
};

VulkanVideoWriter::VulkanVideoWriter() : impl_(std::make_unique<Impl>()) {
    impl_->pkt = av_packet_alloc();
    impl_->sw_frame = av_frame_alloc();
}

VulkanVideoWriter::~VulkanVideoWriter() {
    close();
}

bool VulkanVideoWriter::open(const std::string& path, int width, int height, AVRational fps, std::string& error) {
    if (avformat_alloc_output_context2(&impl_->fmt_ctx, nullptr, nullptr, path.c_str()) < 0) {
        error = "Could not allocate output context for " + path;
        return false;
    }

    const AVCodec* codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    if (!codec) {
        error = "H.264 encoder not found";
        return false;
    }

    impl_->video_stream = avformat_new_stream(impl_->fmt_ctx, codec);
    if (!impl_->video_stream) {
        error = "Failed to create new stream";
        return false;
    }

    impl_->codec_ctx = avcodec_alloc_context3(codec);
    if (!impl_->codec_ctx) {
        error = "Failed to allocate codec context";
        return false;
    }

    impl_->codec_ctx->width = width;
    impl_->codec_ctx->height = height;
    impl_->codec_ctx->time_base = av_inv_q(fps);
    impl_->codec_ctx->framerate = fps;
    impl_->codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P; // Fallback to software encoding for now
    impl_->video_stream->time_base = impl_->codec_ctx->time_base;

    if (impl_->fmt_ctx->oformat->flags & AVFMT_GLOBALHEADER) {
        impl_->codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    if (avcodec_open2(impl_->codec_ctx, codec, nullptr) < 0) {
        error = "Failed to open codec";
        return false;
    }

    if (avcodec_parameters_from_context(impl_->video_stream->codecpar, impl_->codec_ctx) < 0) {
        error = "Failed to copy codec parameters";
        return false;
    }

    if (!(impl_->fmt_ctx->oformat->flags & AVFMT_NOFILE)) {
        if (avio_open(&impl_->fmt_ctx->pb, path.c_str(), AVIO_FLAG_WRITE) < 0) {
            error = "Could not open output file " + path;
            return false;
        }
    }

    if (avformat_write_header(impl_->fmt_ctx, nullptr) < 0) {
        error = "Error occurred when opening output file";
        return false;
    }

    impl_->sw_frame->format = impl_->codec_ctx->pix_fmt;
    impl_->sw_frame->width = impl_->codec_ctx->width;
    impl_->sw_frame->height = impl_->codec_ctx->height;
    if (av_frame_get_buffer(impl_->sw_frame, 0) < 0) {
        error = "Could not allocate frame data";
        return false;
    }

    return true;
}

bool VulkanVideoWriter::writeFrame(AVFrame* frame, std::string& error) {
    AVFrame* encode_frame = frame;

    // If frame is Vulkan hardware frame, transfer to software frame
    if (frame->format == AV_PIX_FMT_VULKAN) {
        if (av_hwframe_transfer_data(impl_->sw_frame, frame, 0) < 0) {
            error = "Error transferring frame from Vulkan";
            return false;
        }
        encode_frame = impl_->sw_frame;
    }

    encode_frame->pts = impl_->frame_count++;

    int ret = avcodec_send_frame(impl_->codec_ctx, encode_frame);
    if (ret < 0) {
        error = "Error sending a frame for encoding";
        return false;
    }

    while (ret >= 0) {
        ret = avcodec_receive_packet(impl_->codec_ctx, impl_->pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            break;
        } else if (ret < 0) {
            error = "Error during encoding";
            return false;
        }

        av_packet_rescale_ts(impl_->pkt, impl_->codec_ctx->time_base, impl_->video_stream->time_base);
        impl_->pkt->stream_index = impl_->video_stream->index;

        ret = av_interleaved_write_frame(impl_->fmt_ctx, impl_->pkt);
        av_packet_unref(impl_->pkt);
        if (ret < 0) {
            error = "Error while writing output packet";
            return false;
        }
    }

    return true;
}

void VulkanVideoWriter::close() {
    if (impl_->fmt_ctx) {
        av_write_trailer(impl_->fmt_ctx);
        if (!(impl_->fmt_ctx->oformat->flags & AVFMT_NOFILE)) {
            avio_closep(&impl_->fmt_ctx->pb);
        }
        avformat_free_context(impl_->fmt_ctx);
        impl_->fmt_ctx = nullptr;
    }
    if (impl_->codec_ctx) {
        avcodec_free_context(&impl_->codec_ctx);
        impl_->codec_ctx = nullptr;
    }
    if (impl_->pkt) {
        av_packet_free(&impl_->pkt);
        impl_->pkt = nullptr;
    }
    if (impl_->sw_frame) {
        av_frame_free(&impl_->sw_frame);
        impl_->sw_frame = nullptr;
    }
}

} // namespace frame_io
} // namespace ave
