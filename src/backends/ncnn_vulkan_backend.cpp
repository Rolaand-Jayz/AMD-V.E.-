// ─────────────────────────────────────────────────────────────────
// NCNN Vulkan Backend – full implementation
// Conditionally compiled against AVE_HAVE_NCNN.
// When absent, falls back to FFmpeg filter pipeline for all stages.
// ─────────────────────────────────────────────────────────────────
#include "ave/backends/ncnn_vulkan_backend.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "ave/model_catalog.hpp"
#include "ave/model_manager.hpp"
#include "ave/stage.hpp"
#include "ave/types.hpp"

#ifdef AVE_HAVE_NCNN
#  include <layer.h>
#  include <net.h>
#  include <gpu.h>
#endif

namespace ave {
namespace {

bool fileExistsN(const std::string& p) {
    std::error_code ec;
    return std::filesystem::exists(p, ec);
}

bool commandInPathN(const std::string& cmd) {
    const char* pathEnv = std::getenv("PATH");
    if (!pathEnv) { return false; }
    std::string path(pathEnv);
    std::size_t start = 0;
    while (start <= path.size()) {
        std::size_t end = path.find(':', start);
        if (end == std::string::npos) { end = path.size(); }
        const std::string dir = path.substr(start, end - start);
        if (!dir.empty() && fileExistsN((std::filesystem::path(dir) / cmd).string())) {
            return true;
        }
        if (end == path.size()) { break; }
        start = end + 1;
    }
    return false;
}

bool hasVulkanSignal() {
    if (commandInPathN("vulkaninfo")) { return true; }
    for (const auto& lib : {"/usr/lib/libvulkan.so", "/usr/lib64/libvulkan.so",
                             "/usr/lib/libvulkan.so.1", "/usr/lib64/libvulkan.so.1"}) {
        if (fileExistsN(lib)) { return true; }
    }
    return false;
}

std::string defaultModelIdFor(StageKind kind) {
    const auto entries = catalogEntriesForStage(kind);
    for (const auto* e : entries) { if (e->isDefault) { return e->id; } }
    if (!entries.empty()) { return entries.front()->id; }
    return {};
}

std::string resolveModelId(const EnhancementStage& stage) {
    auto it = stage.params.find("model");
    if (it != stage.params.end()) {
        if (const auto* s = std::get_if<std::string>(&it->second)) {
            if (!s->empty()) { return *s; }
        }
    }
    return defaultModelIdFor(stage.kind);
}

}  // namespace

// ─────────────────────────────────────────────────────────────────
// Impl
// ─────────────────────────────────────────────────────────────────

#ifdef AVE_HAVE_NCNN

struct NcnnNet {
    ncnn::Net               net;
    std::string             paramPath;
    std::string             binPath;
    int                     gpuIndex = 0;
};

struct NcnnVulkanBackend::Impl {
    bool       initialised = false;
    int        gpuIdx      = 0;
    std::mutex mtx;
    std::unordered_map<std::string, std::unique_ptr<NcnnNet>> nets;

    bool loadNet(const std::string& modelId, std::string& error) {
        if (nets.count(modelId)) { return true; }

        ModelManager mgr;
        const auto bestParam = mgr.findModel(modelId);
        if (!bestParam) { error = "Unknown model: " + modelId; return false; }

        const auto& m = *bestParam;
        if (m.downloadedPath.empty()) {
            error = "NCNN model '" + modelId + "' not downloaded. Use Model Manager.";
            return false;
        }
        if (m.entry.sourceFormat != ModelFormat::NcnnBin) {
            error = "Model '" + modelId + "' is not an NCNN model (.param/.bin).";
            return false;
        }
        if (m.downloadedPathAux.empty()) {
            error = "NCNN model '" + modelId + "' is missing its .bin weights file.";
            return false;
        }

        auto nn = std::make_unique<NcnnNet>();
        nn->gpuIndex  = gpuIdx;
        nn->paramPath = m.downloadedPath;
        nn->binPath   = m.downloadedPathAux;

        nn->net.opt.use_vulkan_compute = true;
        nn->net.set_vulkan_device(gpuIdx);

        if (nn->net.load_param(nn->paramPath.c_str()) != 0) {
            error = "NCNN: load_param failed for " + nn->paramPath;
            return false;
        }
        if (nn->net.load_model(nn->binPath.c_str()) != 0) {
            error = "NCNN: load_model failed for " + nn->binPath;
            return false;
        }

        nets[modelId] = std::move(nn);
        return true;
    }
};

#else  // !AVE_HAVE_NCNN

struct NcnnVulkanBackend::Impl {
    bool       initialised = false;
    int        gpuIdx      = 0;
    std::mutex mtx;

    bool loadNet(const std::string& modelId, std::string& error) {
        (void)modelId;
        error = "NCNN Vulkan backend is not compiled in this build (AVE_HAVE_NCNN=OFF).";
        return false;
    }
};

#endif // AVE_HAVE_NCNN

// ─────────────────────────────────────────────────────────────────
// Public interface
// ─────────────────────────────────────────────────────────────────

NcnnVulkanBackend::NcnnVulkanBackend()  : impl_(std::make_unique<Impl>()) {}
NcnnVulkanBackend::~NcnnVulkanBackend() = default;

BackendType NcnnVulkanBackend::type()  const { return BackendType::NcnnVulkan; }
std::string NcnnVulkanBackend::name()  const { return "NCNN (Vulkan)"; }

bool NcnnVulkanBackend::isAvailable(std::string& reason) const {
#ifndef AVE_HAVE_NCNN
    reason = "NCNN backend support is not compiled in this build (AVE_HAVE_NCNN=OFF).";
    return false;
#else
    if (!hasVulkanSignal()) {
        reason = "Vulkan runtime not detected (expected vulkaninfo or libvulkan).";
        return false;
    }
    reason = "NCNN Vulkan runtime detected and ready.";
    return true;
#endif
}

bool NcnnVulkanBackend::initialize(std::string& error) {
    std::string reason;
    if (!isAvailable(reason)) { error = "NCNN Vulkan init: " + reason; return false; }

#ifdef AVE_HAVE_NCNN
    ncnn::create_gpu_instance();
    const int gpuCount = ncnn::get_gpu_count();
    if (gpuCount == 0) { error = "NCNN: no Vulkan GPU devices found."; return false; }
    if (impl_->gpuIdx >= gpuCount) { impl_->gpuIdx = 0; }
#endif

    impl_->initialised = true;
    std::cout << "[backend] NCNN Vulkan initialised on GPU " << impl_->gpuIdx << std::endl;
    return true;
}

bool NcnnVulkanBackend::preloadModel(const std::string& modelId, std::string& error) {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    return impl_->loadNet(modelId, error);
}

void NcnnVulkanBackend::evictModel(const std::string& modelId) {
    std::lock_guard<std::mutex> lk(impl_->mtx);
#ifdef AVE_HAVE_NCNN
    impl_->nets.erase(modelId);
#else
    (void)modelId;
#endif
}

void NcnnVulkanBackend::evictAll() {
    std::lock_guard<std::mutex> lk(impl_->mtx);
#ifdef AVE_HAVE_NCNN
    impl_->nets.clear();
#endif
}

int NcnnVulkanBackend::gpuIndex() const { return impl_->gpuIdx; }

StageResult NcnnVulkanBackend::runStage(const EnhancementStage& stage, std::string& error) {
    const std::string modelId = resolveModelId(stage);
    if (modelId.empty()) {
        std::cout << "[ncnn] no model for " << toString(stage.kind)
                  << " — deferring to FFmpeg filter chain." << std::endl;
        return StageResult::Deferred;
    }

    {
        std::lock_guard<std::mutex> lk(impl_->mtx);
        if (!impl_->loadNet(modelId, error)) {
            std::cerr << "[ncnn] model load failed for stage '"
                      << toString(stage.kind) << "': " << error
                      << "\n  → Deferring to FFmpeg filter chain." << std::endl;
            error.clear();
            return StageResult::Deferred;
        }
    }

    std::cout << "[ncnn] stage=" << toString(stage.kind) << " model=" << modelId
              << " — AI inference will run via processVideoFile() during encode." << std::endl;
    return StageResult::Deferred;
}

// ─────────────────────────────────────────────────────────────────
// processVideoFile — real per-frame NCNN Vulkan inference
// ─────────────────────────────────────────────────────────────────

namespace {

struct VideoInfo {
    int width  = 0;
    int height = 0;
    double fps = 25.0;
    int totalFrames = 0;
};

VideoInfo probeVideoNcnn(const std::string& path) {
    VideoInfo info;
    // Get dimensions and fps
    {
        std::string cmd = "ffprobe -v error -select_streams v:0"
                          " -show_entries stream=width,height,r_frame_rate,nb_frames"
                          " -of csv=p=0 \"" + path + "\"";
        FILE* fp = popen(cmd.c_str(), "r");
        if (fp) {
            char buf[256] = {};
            if (fgets(buf, sizeof(buf), fp)) {
                // format: width,height,fps_num/fps_den,nb_frames
                int w = 0, h = 0, fpsNum = 0, fpsDen = 1;
                char nbFramesBuf[64] = {};
                if (sscanf(buf, "%d,%d,%d/%d,%63s", &w, &h, &fpsNum, &fpsDen, nbFramesBuf) >= 4) {
                    info.width  = w;
                    info.height = h;
                    info.fps    = (fpsDen > 0) ? (static_cast<double>(fpsNum) / fpsDen) : 25.0;
                    if (nbFramesBuf[0] && nbFramesBuf[0] != 'N') {
                        info.totalFrames = std::atoi(nbFramesBuf);
                    }
                }
            }
            pclose(fp);
        }
    }
    // Fallback for frame count
    if (info.totalFrames <= 0 && info.width > 0) {
        std::string cmd = "ffprobe -v error -count_frames -select_streams v:0"
                          " -show_entries stream=nb_read_frames"
                          " -of csv=p=0 \"" + path + "\"";
        FILE* fp = popen(cmd.c_str(), "r");
        if (fp) {
            char buf[64] = {};
            if (fgets(buf, sizeof(buf), fp)) {
                info.totalFrames = std::atoi(buf);
            }
            pclose(fp);
        }
    }
    return info;
}

}  // namespace

StageResult NcnnVulkanBackend::processVideoFile(
        const EnhancementStage& stage,
        const std::string& inputVideo,
        const std::string& outputVideo,
        const FrameProgressCb& progressCb,
        std::string& error,
        const ProcessVideoOptions& opts) {

#ifndef AVE_HAVE_NCNN
    (void)stage; (void)inputVideo; (void)outputVideo; (void)progressCb; (void)opts;
    error = "NCNN support not compiled (AVE_HAVE_NCNN=OFF).";
    return StageResult::Deferred;
#else
    // ── 1. Resolve model ────────────────────────────────────────
    const std::string modelId = resolveModelId(stage);
    if (modelId.empty()) {
        error = "No model resolved for stage " + toString(stage.kind);
        return StageResult::Deferred;
    }

    NcnnNet* nn = nullptr;
    {
        std::lock_guard<std::mutex> lk(impl_->mtx);
        if (!impl_->loadNet(modelId, error)) {
            return StageResult::Deferred;
        }
        auto it = impl_->nets.find(modelId);
        if (it == impl_->nets.end()) {
            error = "NCNN model disappeared after loading: " + modelId;
            return StageResult::Error;
        }
        nn = it->second.get();
    }

    // ── 2. Probe input video ────────────────────────────────────
    const VideoInfo vi = probeVideoNcnn(inputVideo);
    if (vi.width <= 0 || vi.height <= 0) {
        error = "ffprobe failed to detect video dimensions for: " + inputVideo;
        return StageResult::Error;
    }

    // Determine model input/output size.
    // NCNN super-resolution models: input = source, output = source * scale
    int modelScale = 1;
    {
        auto scaleIt = stage.params.find("scale");
        if (scaleIt != stage.params.end()) {
            if (const auto* iv = std::get_if<std::int64_t>(&scaleIt->second)) {
                modelScale = static_cast<int>(*iv);
            }
        }
        if (modelScale < 1) modelScale = 1;
    }

    int outW = 0, outH = 0;
    {
        auto wIt = stage.params.find("width");
        auto hIt = stage.params.find("height");
        if (wIt != stage.params.end() && hIt != stage.params.end()) {
            if (const auto* wv = std::get_if<std::int64_t>(&wIt->second)) outW = static_cast<int>(*wv);
            if (const auto* hv = std::get_if<std::int64_t>(&hIt->second)) outH = static_cast<int>(*hv);
        }
    }
    if (outW <= 0) outW = vi.width * modelScale;
    if (outH <= 0) outH = vi.height * modelScale;

    const std::size_t inFrameBytes  = static_cast<std::size_t>(vi.width) *
                                      static_cast<std::size_t>(vi.height) * 3;
    const std::size_t outFrameBytes = static_cast<std::size_t>(outW) *
                                      static_cast<std::size_t>(outH) * 3;

    std::cout << "[ncnn] processVideoFile: " << vi.width << "x" << vi.height
              << " → " << outW << "x" << outH
              << " fps=" << vi.fps
              << " frames=" << vi.totalFrames << std::endl;

    // ── 3. Open FFmpeg decode pipe ──────────────────────────────
    std::string decodeTimeLim;
    if (opts.previewDurationSec > 0.0) {
        decodeTimeLim = " -t " + std::to_string(opts.previewDurationSec);
    }
    std::string decodeCmdStr =
        "ffmpeg -v error" + decodeTimeLim +
        " -i \"" + inputVideo + "\""
        " -f rawvideo -pix_fmt rgb24 -s " +
        std::to_string(vi.width) + "x" + std::to_string(vi.height) +
        " pipe:1";

    FILE* decodePipe = popen(decodeCmdStr.c_str(), "r");
    if (!decodePipe) {
        error = "Failed to start FFmpeg decode pipe.";
        return StageResult::Error;
    }

    // ── 4. Open FFmpeg encode pipe ──────────────────────────────
    std::string encodeCmdStr =
        "ffmpeg -v error -y"
        " -f rawvideo -pix_fmt rgb24"
        " -s " + std::to_string(outW) + "x" + std::to_string(outH) +
        " -r " + std::to_string(vi.fps) +
        " -i pipe:0"
        " -i \"" + inputVideo + "\""
        " -map 0:v:0 -map 1:a? -c:v libx264 -crf 18 -preset medium"
        " -c:a copy \"" + outputVideo + "\"";

    FILE* encodePipe = popen(encodeCmdStr.c_str(), "w");
    if (!encodePipe) {
        pclose(decodePipe);
        error = "Failed to start FFmpeg encode pipe.";
        return StageResult::Error;
    }

    // ── 5. Per-frame inference loop ─────────────────────────────
    std::vector<uint8_t> inBuf(inFrameBytes);
    std::vector<uint8_t> outBuf(outFrameBytes);
    int frameIdx = 0;
    bool ok = true;
    bool cancelled = false;

    while (true) {
        // ── Cancel / Pause check ────────────────────────────────
        if (opts.cancelFlag && opts.cancelFlag->load(std::memory_order_relaxed)) {
            std::cout << "[ncnn] Cancelled at frame " << frameIdx << std::endl;
            cancelled = true;
            break;
        }
        while (opts.pauseFlag && opts.pauseFlag->load(std::memory_order_relaxed)) {
            if (opts.cancelFlag && opts.cancelFlag->load(std::memory_order_relaxed)) {
                cancelled = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        if (cancelled) break;

        // Read one raw RGB24 frame
        std::size_t bytesRead = fread(inBuf.data(), 1, inFrameBytes, decodePipe);
        if (bytesRead < inFrameBytes) {
            break;  // End of stream (or partial frame — skip)
        }

        // Create NCNN input mat from RGB24 (HWC pixel order, 3 channels)
        // ncnn::Mat::from_pixels expects pixel data in HWC format
        ncnn::Mat inMat = ncnn::Mat::from_pixels(
            inBuf.data(), ncnn::Mat::PIXEL_RGB, vi.width, vi.height);

        // Normalize to [0,1] range (most SR models expect this)
        const float norm_vals[3] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
        inMat.substract_mean_normalize(nullptr, norm_vals);

        // Run inference
        ncnn::Extractor ex = nn->net.create_extractor();

        // Use standard input/output blob names
        // Most NCNN models use "data"/"input" for input and "output" for output
        int inputRet = ex.input("data", inMat);
        if (inputRet != 0) {
            // Try alternate input name
            inputRet = ex.input("input", inMat);
            if (inputRet != 0) {
                error = "NCNN: failed to set input tensor (tried 'data' and 'input')";
                ok = false;
                break;
            }
        }

        ncnn::Mat outMat;
        int outputRet = ex.extract("output", outMat);
        if (outputRet != 0) {
            error = "NCNN: failed to extract output tensor (blob 'output')";
            ok = false;
            break;
        }

        // De-normalize: multiply by 255 and clamp to [0, 255]
        // outMat is CHW format, convert to HWC RGB24
        const int outCh = outMat.c;
        const int outMH = outMat.h;
        const int outMW = outMat.w;

        if (outCh >= 3 && outMH > 0 && outMW > 0) {
            // Use ncnn::Mat::to_pixels to convert back to RGB24
            ncnn::Mat denormMat = outMat.clone();
            // Scale back to [0, 255]
            for (int c = 0; c < 3; ++c) {
                float* ptr = denormMat.channel(c);
                for (int i = 0; i < outMH * outMW; ++i) {
                    float val = ptr[i] * 255.0f;
                    if (val < 0.0f) val = 0.0f;
                    if (val > 255.0f) val = 255.0f;
                    ptr[i] = val;
                }
            }
            denormMat.to_pixels(outBuf.data(), ncnn::Mat::PIXEL_RGB);
        } else {
            error = "NCNN output has unexpected shape: c=" + std::to_string(outCh)
                  + " h=" + std::to_string(outMH) + " w=" + std::to_string(outMW);
            ok = false;
            break;
        }

        // Write processed frame to encode pipe
        std::size_t written = fwrite(outBuf.data(), 1, outFrameBytes, encodePipe);
        if (written < outFrameBytes) {
            error = "FFmpeg encode pipe write failed.";
            ok = false;
            break;
        }

        ++frameIdx;

        // Emit live frame preview
        const int pvInterval = opts.previewFrameInterval > 0 ? opts.previewFrameInterval : 15;
        if (opts.framePreviewCb && (frameIdx % pvInterval == 1 || pvInterval == 1)) {
            opts.framePreviewCb(outBuf.data(), outW, outH);
        }

        // Report progress
        if (progressCb) {
            float progress = 0.0f;
            if (vi.totalFrames > 0) {
                progress = static_cast<float>(frameIdx) /
                           static_cast<float>(vi.totalFrames);
                if (progress > 1.0f) progress = 1.0f;
            } else {
                // Logarithmic fallback for unknown-length streams
                progress = 1.0f - 1.0f / (1.0f + static_cast<float>(frameIdx) * 0.01f);
            }
            const std::string frameMsg = "NCNN: frame " + std::to_string(frameIdx)
                + (vi.totalFrames > 0 ? "/" + std::to_string(vi.totalFrames) : "");
            progressCb(progress, frameMsg);
        }
    }

    // ── 6. Cleanup ──────────────────────────────────────────────
    int decodeRc = pclose(decodePipe);
    int encodeRc = pclose(encodePipe);

    if (cancelled) {
        error = "Processing cancelled by user at frame " + std::to_string(frameIdx);
        return StageResult::Cancelled;
    }

    if (!ok) {
        return StageResult::Error;
    }

    if (decodeRc != 0) {
        error = "FFmpeg decode pipe exited with code " + std::to_string(decodeRc);
        return StageResult::Error;
    }
    if (encodeRc != 0) {
        error = "FFmpeg encode pipe exited with code " + std::to_string(encodeRc);
        return StageResult::Error;
    }

    if (frameIdx == 0) {
        error = "No frames decoded from input video.";
        return StageResult::Error;
    }

    std::cout << "[ncnn] processVideoFile: processed " << frameIdx << " frames → "
              << outputVideo << std::endl;
    return StageResult::Processed;
#endif  // AVE_HAVE_NCNN
}

}  // namespace ave
