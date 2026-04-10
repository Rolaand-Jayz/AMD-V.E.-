// ─────────────────────────────────────────────────────────────────
// NCNN Vulkan Backend – full implementation
// Conditionally compiled against AVE_HAVE_NCNN.
// When absent, falls back to FFmpeg filter pipeline for all stages.
// ─────────────────────────────────────────────────────────────────
#include "ave/backends/ncnn_vulkan_backend.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "ave/frame_io.hpp"
#include "ave/model_catalog.hpp"
#include "ave/model_manager.hpp"
#include "ave/process_observer.hpp"
#include "ave/rgb_video_loop.hpp"
#include "ave/stage.hpp"
#include "ave/types.hpp"
#include "ave/video_probe.hpp"

#ifdef AVE_HAVE_NCNN
#  include <layer.h>
#  include <net.h>
#  include <gpu.h>
#endif

namespace ave {
namespace {

bool hasVulkanSignal() {
    if (process_observer::commandInPath("vulkaninfo")) { return true; }
    for (const auto& lib : {"/usr/lib/libvulkan.so", "/usr/lib64/libvulkan.so",
                             "/usr/lib/libvulkan.so.1", "/usr/lib64/libvulkan.so.1"}) {
        if (process_observer::fileExists(lib)) { return true; }
    }
    return false;
}

std::optional<int> readNonNegativeEnvIntN(const char* name) {
    if (name == nullptr) {
        return std::nullopt;
    }
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') {
        return std::nullopt;
    }
    char* end = nullptr;
    const long value = std::strtol(raw, &end, 10);
    if (end == raw || *end != '\0' || value < 0L) {
        return std::nullopt;
    }
    return static_cast<int>(value);
}

std::optional<int> preferredNcnnGpuIndexFromEnv() {
    if (const auto explicitNcnn = readNonNegativeEnvIntN("AVE_NCNN_GPU_INDEX"); explicitNcnn.has_value()) {
        return explicitNcnn;
    }
    return readNonNegativeEnvIntN("AVE_GPU_INDEX");
}

#ifdef AVE_HAVE_NCNN
std::mutex& ncnnGpuInstanceMutex() {
    static std::mutex mutex;
    return mutex;
}

int& ncnnGpuInstanceRefCount() {
    static int refCount = 0;
    return refCount;
}

void retainNcnnGpuInstance() {
    std::lock_guard<std::mutex> lk(ncnnGpuInstanceMutex());
    int& refCount = ncnnGpuInstanceRefCount();
    if (refCount == 0) {
        ncnn::create_gpu_instance();
    }
    ++refCount;
}

void releaseNcnnGpuInstance() {
    std::lock_guard<std::mutex> lk(ncnnGpuInstanceMutex());
    int& refCount = ncnnGpuInstanceRefCount();
    if (refCount <= 0) {
        return;
    }
    --refCount;
    if (refCount == 0) {
        ncnn::destroy_gpu_instance();
    }
}
#endif

std::string defaultModelIdFor(StageKind kind) {
    if (const auto* preferred = preferredBackendModelForStage(kind, BackendType::NcnnVulkan);
        preferred != nullptr) {
        return preferred->id;
    }
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
NcnnVulkanBackend::~NcnnVulkanBackend() {
#ifdef AVE_HAVE_NCNN
    if (impl_) {
        std::lock_guard<std::mutex> lk(impl_->mtx);
        impl_->nets.clear();
        impl_->initialised = false;
    }
#endif
}

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
    retainNcnnGpuInstance();
    const int gpuCount = ncnn::get_gpu_count();
    if (gpuCount == 0) {
        releaseNcnnGpuInstance();
        error = "NCNN: no Vulkan GPU devices found.";
        return false;
    }
    if (const auto preferredGpu = preferredNcnnGpuIndexFromEnv(); preferredGpu.has_value()) {
        if (*preferredGpu >= gpuCount) {
            releaseNcnnGpuInstance();
            error = "NCNN: requested GPU index " + std::to_string(*preferredGpu)
                  + " is out of range for " + std::to_string(gpuCount) + " detected device(s).";
            return false;
        }
        impl_->gpuIdx = *preferredGpu;
    }
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
    const auto probe = probeVideoStream(inputVideo, error);
    if (!probe.has_value()) {
        return StageResult::Error;
    }
    const int inputWidth = static_cast<int>(probe->width);
    const int inputHeight = static_cast<int>(probe->height);
    const double inputFps = probe->effectiveFrameRate(25.0);
    const int totalFrames = static_cast<int>(probe->estimatedFrameCount());

    // Determine model input/output size.
    // NCNN super-resolution models: input = source, output = source * scale
    int modelScale = 0;
    {
        std::int64_t scaleValue = 0;
        if (tryGetInt(stage, StageKind::Upscale, "scale", scaleValue)) {
            modelScale = static_cast<int>(scaleValue);
        }
        if (modelScale < 1) {
            if (const ModelEntry* entry = catalogEntryById(modelId);
                entry != nullptr && entry->scale > 0) {
                modelScale = entry->scale;
            }
        }
        if (modelScale < 1) modelScale = 1;
    }

    int outW = 0, outH = 0;
    {
        std::int64_t widthValue = 0;
        std::int64_t heightValue = 0;
        if (tryGetInt(stage, StageKind::Upscale, "width", widthValue) &&
            tryGetInt(stage, StageKind::Upscale, "height", heightValue)) {
            outW = static_cast<int>(widthValue);
            outH = static_cast<int>(heightValue);
        }
    }
    if (outW <= 0) outW = inputWidth * modelScale;
    if (outH <= 0) outH = inputHeight * modelScale;

    const std::size_t outFrameBytes = static_cast<std::size_t>(outW) *
                                      static_cast<std::size_t>(outH) * 3;

    std::cout << "[ncnn] processVideoFile: " << inputWidth << "x" << inputHeight
              << " → " << outW << "x" << outH
              << " fps=" << inputFps
              << " frames=" << totalFrames << std::endl;

    RgbVideoLoopOptions loopOptions;
    loopOptions.inputVideo = inputVideo;
    loopOptions.outputVideo = outputVideo;
    loopOptions.inputWidth = inputWidth;
    loopOptions.inputHeight = inputHeight;
    loopOptions.outputWidth = outW;
    loopOptions.outputHeight = outH;
    loopOptions.fps = inputFps;
    loopOptions.fallbackFps = 25.0;
    loopOptions.totalFrames = totalFrames;
    loopOptions.backendTag = "ncnn";
    loopOptions.progressLabel = "NCNN";
    loopOptions.noFramesError = "No frames decoded from input video.";
    loopOptions.preferredSourceMode = frame_io::RgbVideoSourceMode::VulkanTransfer;
    loopOptions.allowSourceFallback = true;
    loopOptions.processOptions = opts;

    const auto loopResult = runRgbVideoLoop(
        loopOptions,
        [&](const std::vector<std::uint8_t>& inBuf,
            std::vector<std::uint8_t>& outputRgb,
            const int /*frameIdx*/,
            std::string& loopError) {
            ncnn::Mat inMat = ncnn::Mat::from_pixels(
                inBuf.data(), ncnn::Mat::PIXEL_RGB, inputWidth, inputHeight);

            const float normVals[3] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
            inMat.substract_mean_normalize(nullptr, normVals);

            ncnn::Extractor ex = nn->net.create_extractor();
            int inputRet = ex.input("data", inMat);
            if (inputRet != 0) {
                inputRet = ex.input("input", inMat);
                if (inputRet != 0) {
                    loopError = "NCNN: failed to set input tensor (tried 'data' and 'input')";
                    return false;
                }
            }

            ncnn::Mat outMat;
            const int outputRet = ex.extract("output", outMat);
            if (outputRet != 0) {
                loopError = "NCNN: failed to extract output tensor (blob 'output')";
                return false;
            }

            const int outCh = outMat.c;
            const int outMH = outMat.h;
            const int outMW = outMat.w;
            if (outMW != outW || outMH != outH) {
                loopError = "NCNN output dimensions do not match the configured frame size: expected "
                          + std::to_string(outW) + "x" + std::to_string(outH)
                          + ", got " + std::to_string(outMW) + "x" + std::to_string(outMH);
                return false;
            }
            if (outCh < 3 || outMH <= 0 || outMW <= 0) {
                loopError = "NCNN output has unexpected shape: c=" + std::to_string(outCh)
                          + " h=" + std::to_string(outMH) + " w=" + std::to_string(outMW);
                return false;
            }

            ncnn::Mat denormMat = outMat.clone();
            for (int c = 0; c < 3; ++c) {
                float* ptr = denormMat.channel(c);
                for (int i = 0; i < outMH * outMW; ++i) {
                    float val = ptr[i] * 255.0f;
                    if (val < 0.0f) val = 0.0f;
                    if (val > 255.0f) val = 255.0f;
                    ptr[i] = val;
                }
            }

            if (outputRgb.size() != outFrameBytes) {
                outputRgb.resize(outFrameBytes);
            }
            denormMat.to_pixels(outputRgb.data(), ncnn::Mat::PIXEL_RGB);
            return true;
        },
        progressCb,
        error);

    if (loopResult.stageResult != StageResult::Processed) {
        return loopResult.stageResult;
    }

    {
        std::lock_guard<std::mutex> lk(impl_->mtx);
        impl_->nets.clear();
    }

    std::cout << "[ncnn] processVideoFile: processed " << loopResult.frameCount << " frames → "
              << outputVideo << std::endl;
    return loopResult.stageResult;
#endif  // AVE_HAVE_NCNN
}

}  // namespace ave
