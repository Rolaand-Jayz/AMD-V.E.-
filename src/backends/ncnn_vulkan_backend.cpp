// ─────────────────────────────────────────────────────────────────
// NCNN Vulkan Backend – full implementation
// Conditionally compiled against AVE_HAVE_NCNN.
// When absent, falls back to FFmpeg filter pipeline for all stages.
// ─────────────────────────────────────────────────────────────────
#include "ave/backends/ncnn_vulkan_backend.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "ave/frame_io.hpp"
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

        auto nn = std::make_unique<NcnnNet>();
        nn->gpuIndex  = gpuIdx;
        nn->paramPath = m.downloadedPath;
        nn->binPath   = m.downloadedPathAux.empty() ? "" : m.downloadedPathAux;

        nn->net.opt.use_vulkan_compute = true;
        nn->net.set_vulkan_device(gpuIdx);

        if (nn->net.load_param(nn->paramPath.c_str()) != 0) {
            error = "NCNN: load_param failed for " + nn->paramPath;
            return false;
        }
        if (!nn->binPath.empty() && nn->net.load_model(nn->binPath.c_str()) != 0) {
            error = "NCNN: load_model failed for " + nn->binPath;
            return false;
        }

        nets[modelId] = std::move(nn);
        return true;
    }
};

#else  // !AVE_HAVE_NCNN – stub

struct NcnnVulkanBackend::Impl {
    bool       initialised = false;
    int        gpuIdx      = 0;
    std::mutex mtx;
    std::unordered_map<std::string, bool> loaded;

    bool loadNet(const std::string& modelId, std::string& error) {
        if (loaded.count(modelId)) { return true; }
        ModelManager mgr;
        const auto best = mgr.bestPathForModel(modelId);
        if (!best) {
            error = "No file for NCNN model '" + modelId + "'. Use Model Manager.";
            return false;
        }
        std::cout << "[ncnn-fallback] validated model path: " << *best << std::endl;
        loaded[modelId] = true;
        return true;
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
    if (!hasVulkanSignal()) {
        reason = "Vulkan runtime not detected (expected vulkaninfo or libvulkan).";
        return false;
    }
    reason = "Vulkan runtime detected for NCNN backend.";
    return true;
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
    impl_->loaded.erase(modelId);
#endif
}

void NcnnVulkanBackend::evictAll() {
    std::lock_guard<std::mutex> lk(impl_->mtx);
#ifdef AVE_HAVE_NCNN
    impl_->nets.clear();
#else
    impl_->loaded.clear();
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

    // Model is loaded but per-frame inference loop is not yet
    // implemented.  Defer to FFmpeg filters for actual processing.
    //
    // TODO: Implement extract→infer→encode loop:
    //   1. Extract frames via FFmpeg to temp PNGs.
    //   2. For each frame: load → ncnn::Mat → Extractor → output.
    //   3. Re-encode processed frames.
    //   Once implemented, return StageResult::Processed here instead.
    std::cout << "[ncnn] stage=" << toString(stage.kind) << " model=" << modelId
              << " — model validated, deferring frame processing to FFmpeg." << std::endl;
    return StageResult::Deferred;
}

// ─────────────────────────────────────────────────────────────────
// processFrameDir — per-frame AI inference using NCNN Vulkan
// ─────────────────────────────────────────────────────────────────
StageResult NcnnVulkanBackend::processFrameDir(
        const EnhancementStage& stage,
        const std::string& inputDir,
        const std::string& outputDir,
        const FrameProgressCb& progressCb,
        std::string& error) {
#ifdef AVE_HAVE_NCNN
    const std::string modelId = resolveModelId(stage);
    if (modelId.empty()) {
        std::cout << "[ncnn] processFrameDir: no model for "
                  << toString(stage.kind) << " — deferring." << std::endl;
        return StageResult::Deferred;
    }

    // Ensure model is loaded.
    {
        std::lock_guard<std::mutex> lk(impl_->mtx);
        if (!impl_->loadNet(modelId, error)) {
            std::cerr << "[ncnn] processFrameDir: load failed: "
                      << error << "\n  → Deferring to FFmpeg." << std::endl;
            error.clear();
            return StageResult::Deferred;
        }
    }

    int scale = 1;
    const auto* catalogEntry = catalogEntryById(modelId);
    if (catalogEntry) scale = catalogEntry->scale;
    if (scale < 1) scale = 1;

    const auto frames = frame_io::listPngFramesSorted(inputDir);
    if (frames.empty()) {
        error = "No PNG frames found in " + inputDir;
        return StageResult::Error;
    }

    std::cout << "[ncnn] processFrameDir: model='" << modelId
              << "' scale=" << scale
              << " frames=" << frames.size()
              << " stage=" << toString(stage.kind) << std::endl;

    auto& nn = *impl_->nets[modelId];

    for (std::size_t i = 0; i < frames.size(); ++i) {
        // 1. Load PNG → raw RGB24.
        int w = 0, h = 0;
        std::vector<std::uint8_t> rgb;
        if (!frame_io::loadPngRgb24(frames[i].string(), w, h, rgb, error))
            return StageResult::Error;

        // 2. Create ncnn::Mat from raw pixels (stores as fp32 CHW internally).
        //    from_pixels does uint8→fp32 conversion; values stay in [0,255].
        ncnn::Mat in = ncnn::Mat::from_pixels(rgb.data(), ncnn::Mat::PIXEL_RGB, w, h);

        // Normalise to [0,1] range (most ONNX-converted ESRGAN models expect this).
        const float mean_vals[3] = {0.f, 0.f, 0.f};
        const float norm_vals[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
        in.substract_mean_normalize(mean_vals, norm_vals);

        // 3. Run inference via NCNN Extractor.
        ncnn::Extractor ex = nn.net.create_extractor();

        // Use blob indices for robustness with varying model architectures.
        const auto& inputIdxs  = nn.net.input_indexes();
        const auto& outputIdxs = nn.net.output_indexes();

        if (inputIdxs.empty() || outputIdxs.empty()) {
            error = "NCNN model '" + modelId + "' has no input or output blobs.";
            return StageResult::Error;
        }

        if (ex.input(inputIdxs[0], in) != 0) {
            error = "NCNN Extractor input failed for frame " + std::to_string(i);
            return StageResult::Error;
        }

        ncnn::Mat out;
        if (ex.extract(outputIdxs[0], out) != 0) {
            error = "NCNN Extractor extract failed for frame " + std::to_string(i);
            return StageResult::Error;
        }

        // 4. Determine output dimensions.
        const int outW = out.w;
        const int outH = out.h;

        // Denormalise from [0,1] back to [0,255] for to_pixels().
        const float denorm_mean[3] = {0.f, 0.f, 0.f};
        const float denorm_norm[3] = {255.f, 255.f, 255.f};
        out.substract_mean_normalize(denorm_mean, denorm_norm);

        // 5. Convert ncnn::Mat back to interleaved RGB24 uint8.
        std::vector<std::uint8_t> outRgb(static_cast<std::size_t>(outW) *
                                          static_cast<std::size_t>(outH) * 3u);
        out.to_pixels(outRgb.data(), ncnn::Mat::PIXEL_RGB);

        // 6. Save processed frame.
        const auto outPath = std::filesystem::path(outputDir) / frames[i].filename();
        if (!frame_io::saveRgb24ToPng(outPath.string(), outW, outH,
                                       outRgb.data(), error))
            return StageResult::Error;

        // 7. Report progress.
        if (progressCb) {
            const float frac = static_cast<float>(i + 1) /
                               static_cast<float>(frames.size());
            progressCb(frac, "NCNN AI: frame " + std::to_string(i + 1) +
                             "/" + std::to_string(frames.size()));
        }
    }

    std::cout << "[ncnn] processFrameDir: completed " << frames.size()
              << " frames for stage=" << toString(stage.kind) << std::endl;
    return StageResult::Processed;

#else  // !AVE_HAVE_NCNN
    (void)stage; (void)inputDir; (void)outputDir; (void)progressCb;
    std::cout << "[ncnn-stub] processFrameDir: NCNN not compiled in; "
                 "deferring to FFmpeg." << std::endl;
    return StageResult::Deferred;
#endif
}

}  // namespace ave
