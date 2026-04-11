// ─────────────────────────────────────────────────────────────────
// Vulkan Compute Backend — GPU image processing via compute shaders
// ─────────────────────────────────────────────────────────────────
#include "ave/backends/vulkan_compute_backend.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "ave/frame_io.hpp"
#include "ave/process_observer.hpp"
#include "ave/rgb_video_loop.hpp"
#include "ave/stage.hpp"
#include "ave/types.hpp"
#include "ave/video_probe.hpp"

#ifdef AVE_HAVE_VULKAN
#include <vulkan/vulkan.h>
#endif

namespace ave {

// ─────────────────────────────────────────────────────────────────
// Embedded GLSL compute shader sources
// ─────────────────────────────────────────────────────────────────
namespace {

std::optional<int> readNonNegativeEnvIntVk(const char* name) {
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

std::optional<int> preferredVulkanDeviceIndexFromEnv() {
    if (const auto explicitVk = readNonNegativeEnvIntVk("AVE_VULKAN_DEVICE_INDEX"); explicitVk.has_value()) {
        return explicitVk;
    }
    return readNonNegativeEnvIntVk("AVE_GPU_INDEX");
}

// All shaders operate on RGB float buffers in SSBO layout:
//   Buffer[pixelIndex * 3 + 0] = R
//   Buffer[pixelIndex * 3 + 1] = G
//   Buffer[pixelIndex * 3 + 2] = B
// Values normalised to [0, 1].

const char* kSharpenGlsl = R"(
#version 450
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer Input  { float data[]; } inBuf;
layout(std430, binding = 1) writeonly buffer Output { float data[]; } outBuf;

layout(push_constant) uniform Params {
    int width;
    int height;
    float strength;
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint totalPixels = uint(width) * uint(height);
    if (idx >= totalPixels) return;

    int x = int(idx % uint(width));
    int y = int(idx / uint(width));

    // Unsharp mask: sharp = original + strength * (original - blur)
    // 3x3 Gaussian-weighted blur
    vec3 blur = vec3(0.0);
    float totalW = 0.0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int sx = clamp(x + dx, 0, width - 1);
            int sy = clamp(y + dy, 0, height - 1);
            uint si = uint(sy) * uint(width) + uint(sx);
            // Gaussian-ish weights: center=4, edge=2, corner=1
            float w = (dx == 0 && dy == 0) ? 4.0 :
                      (abs(dx) + abs(dy) == 1) ? 2.0 : 1.0;
            blur.r += inBuf.data[si * 3u + 0u] * w;
            blur.g += inBuf.data[si * 3u + 1u] * w;
            blur.b += inBuf.data[si * 3u + 2u] * w;
            totalW += w;
        }
    }
    blur /= totalW;

    vec3 orig = vec3(
        inBuf.data[idx * 3u + 0u],
        inBuf.data[idx * 3u + 1u],
        inBuf.data[idx * 3u + 2u]
    );
    vec3 sharp = orig + strength * (orig - blur);
    sharp = clamp(sharp, 0.0, 1.0);

    outBuf.data[idx * 3u + 0u] = sharp.r;
    outBuf.data[idx * 3u + 1u] = sharp.g;
    outBuf.data[idx * 3u + 2u] = sharp.b;
}
)";

const char* kDenoiseGlsl = R"(
#version 450
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer Input  { float data[]; } inBuf;
layout(std430, binding = 1) writeonly buffer Output { float data[]; } outBuf;

layout(push_constant) uniform Params {
    int width;
    int height;
    float strength; // spatial sigma ∈ [0.5, 5.0], higher = more blur
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint totalPixels = uint(width) * uint(height);
    if (idx >= totalPixels) return;

    int x = int(idx % uint(width));
    int y = int(idx / uint(width));

    vec3 center = vec3(
        inBuf.data[idx * 3u + 0u],
        inBuf.data[idx * 3u + 1u],
        inBuf.data[idx * 3u + 2u]
    );

    // Bilateral filter: weight by spatial distance AND intensity distance
    float sigmaSpatial = max(strength, 0.5);
    float sigmaRange   = 0.1; // intensity sigma
    float invSpatial2  = -0.5 / (sigmaSpatial * sigmaSpatial);
    float invRange2    = -0.5 / (sigmaRange * sigmaRange);

    int radius = int(ceil(sigmaSpatial * 2.0));
    radius = clamp(radius, 1, 5);

    vec3 accum = vec3(0.0);
    float totalW = 0.0;

    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            int sx = clamp(x + dx, 0, width - 1);
            int sy = clamp(y + dy, 0, height - 1);
            uint si = uint(sy) * uint(width) + uint(sx);

            vec3 sample_c = vec3(
                inBuf.data[si * 3u + 0u],
                inBuf.data[si * 3u + 1u],
                inBuf.data[si * 3u + 2u]
            );

            float spatialDist2 = float(dx * dx + dy * dy);
            vec3 diff = sample_c - center;
            float rangeDist2 = dot(diff, diff);

            float w = exp(spatialDist2 * invSpatial2 + rangeDist2 * invRange2);
            accum += sample_c * w;
            totalW += w;
        }
    }

    vec3 result = accum / max(totalW, 1e-6);
    outBuf.data[idx * 3u + 0u] = result.r;
    outBuf.data[idx * 3u + 1u] = result.g;
    outBuf.data[idx * 3u + 2u] = result.b;
}
)";

const char* kDehaloGlsl = R"(
#version 450
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer Input  { float data[]; } inBuf;
layout(std430, binding = 1) writeonly buffer Output { float data[]; } outBuf;

layout(push_constant) uniform Params {
    int width;
    int height;
    float strength; // dehalo strength [0.0, 1.0]
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint totalPixels = uint(width) * uint(height);
    if (idx >= totalPixels) return;

    int x = int(idx % uint(width));
    int y = int(idx / uint(width));

    vec3 center = vec3(
        inBuf.data[idx * 3u + 0u],
        inBuf.data[idx * 3u + 1u],
        inBuf.data[idx * 3u + 2u]
    );

    // Dehalo: reduce overshoot around edges.
    // Compute local min/max in 3x3 neighbourhood, then clamp center
    // towards the min/max range.  This removes bright/dark halos.
    vec3 localMin = vec3(1.0);
    vec3 localMax = vec3(0.0);
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            int sx = clamp(x + dx, 0, width - 1);
            int sy = clamp(y + dy, 0, height - 1);
            uint si = uint(sy) * uint(width) + uint(sx);
            vec3 s = vec3(
                inBuf.data[si * 3u + 0u],
                inBuf.data[si * 3u + 1u],
                inBuf.data[si * 3u + 2u]
            );
            localMin = min(localMin, s);
            localMax = max(localMax, s);
        }
    }

    // Soft clamp towards local range
    vec3 clamped = clamp(center, localMin, localMax);
    vec3 result = mix(center, clamped, strength);

    outBuf.data[idx * 3u + 0u] = result.r;
    outBuf.data[idx * 3u + 1u] = result.g;
    outBuf.data[idx * 3u + 2u] = result.b;
}
)";

const char* kDeblurGlsl = R"(
#version 450
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer Input  { float data[]; } inBuf;
layout(std430, binding = 1) writeonly buffer Output { float data[]; } outBuf;

layout(push_constant) uniform Params {
    int width;
    int height;
    float strength; // deblur strength [0.5, 3.0]
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint totalPixels = uint(width) * uint(height);
    if (idx >= totalPixels) return;

    int x = int(idx % uint(width));
    int y = int(idx / uint(width));

    vec3 center = vec3(
        inBuf.data[idx * 3u + 0u],
        inBuf.data[idx * 3u + 1u],
        inBuf.data[idx * 3u + 2u]
    );

    // Wiener-inspired deconvolution approximation:
    // Enhance edges using Laplacian sharpening.
    // Laplacian kernel: [0 -1 0; -1 4 -1; 0 -1 0]
    vec3 laplacian = center * 4.0;
    int offsets[4] = int[](1, -1, 0, 0);
    int offsets2[4] = int[](0, 0, 1, -1);
    for (int i = 0; i < 4; i++) {
        int sx = clamp(x + offsets[i], 0, width - 1);
        int sy = clamp(y + offsets2[i], 0, height - 1);
        uint si = uint(sy) * uint(width) + uint(sx);
        laplacian -= vec3(
            inBuf.data[si * 3u + 0u],
            inBuf.data[si * 3u + 1u],
            inBuf.data[si * 3u + 2u]
        );
    }

    // Add scaled Laplacian back for edge enhancement
    vec3 result = center + strength * 0.25 * laplacian;
    result = clamp(result, 0.0, 1.0);

    outBuf.data[idx * 3u + 0u] = result.r;
    outBuf.data[idx * 3u + 1u] = result.g;
    outBuf.data[idx * 3u + 2u] = result.b;
}
)";

// ─── Upscale shader: bilinear interpolation with Lanczos-weighted ───
const char* kUpscaleGlsl = R"(
#version 450
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer Input  { float data[]; } inBuf;
layout(std430, binding = 1) writeonly buffer Output { float data[]; } outBuf;

layout(push_constant) uniform Params {
    int inWidth;
    int inHeight;
    float scaleX; // outWidth / inWidth
};

// Push constant struct size limited to 128 bytes.
// outWidth = int(round(inWidth * scaleX))
// outHeight = int(round(inHeight * scaleY))
// We derive scaleY = scaleX for uniform scaling.

float lanczos2(float x) {
    if (abs(x) < 1e-6) return 1.0;
    if (abs(x) >= 2.0) return 0.0;
    float px = 3.14159265 * x;
    return (sin(px) / px) * (sin(px / 2.0) / (px / 2.0));
}

void main() {
    int outWidth  = int(round(float(inWidth) * scaleX));
    int outHeight = int(round(float(inHeight) * scaleX));
    uint totalOut = uint(outWidth) * uint(outHeight);
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= totalOut) return;

    int ox = int(idx % uint(outWidth));
    int oy = int(idx / uint(outWidth));

    // Map output pixel to input coordinates
    float srcX = (float(ox) + 0.5) / scaleX - 0.5;
    float srcY = (float(oy) + 0.5) / scaleX - 0.5;

    // Lanczos-2 interpolation (4x4 neighbourhood)
    vec3 accum = vec3(0.0);
    float totalW = 0.0;

    int x0 = int(floor(srcX)) - 1;
    int y0 = int(floor(srcY)) - 1;

    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < 4; i++) {
            int sx = clamp(x0 + i, 0, inWidth - 1);
            int sy = clamp(y0 + j, 0, inHeight - 1);
            uint si = uint(sy) * uint(inWidth) + uint(sx);

            float wx = lanczos2(srcX - float(x0 + i));
            float wy = lanczos2(srcY - float(y0 + j));
            float w = wx * wy;

            accum += vec3(
                inBuf.data[si * 3u + 0u],
                inBuf.data[si * 3u + 1u],
                inBuf.data[si * 3u + 2u]
            ) * w;
            totalW += w;
        }
    }

    vec3 result = accum / max(totalW, 1e-6);
    result = clamp(result, 0.0, 1.0);

    outBuf.data[idx * 3u + 0u] = result.r;
    outBuf.data[idx * 3u + 1u] = result.g;
    outBuf.data[idx * 3u + 2u] = result.b;
}
)";

// ─── Utility helpers ────────────────────────────────────────────

std::string spirvCacheDir() {
    const char* home = std::getenv("HOME");
    if (!home) return "/tmp/ave-spirv";
    return std::string(home) + "/.local/share/ave/spirv_cache";
}

// ─── SPIR-V compilation via glslc ──────────────────────────────

bool compileGlslToSpirv(const std::string& glslSource,
                        const std::string& cacheKey,
                        std::vector<uint32_t>& spirv,
                        std::string& error) {
    const std::string cacheDir  = spirvCacheDir();
    const std::string cachePath = cacheDir + "/" + cacheKey + ".spv";

    // Check cache first
    if (process_observer::fileExists(cachePath)) {
        std::ifstream f(cachePath, std::ios::binary | std::ios::ate);
        if (f.is_open()) {
            const auto size = f.tellg();
            if (size > 0 && size % 4 == 0) {
                spirv.resize(static_cast<std::size_t>(size) / 4);
                f.seekg(0);
                f.read(reinterpret_cast<char*>(spirv.data()), size);
                if (f.good()) return true;
            }
        }
    }

    // Create cache directory
    std::filesystem::create_directories(cacheDir);

    // Write GLSL to temp file
    const std::string tmpGlsl = cacheDir + "/" + cacheKey + ".comp";
    {
        std::ofstream f(tmpGlsl);
        if (!f.is_open()) {
            error = "Cannot write temp GLSL file: " + tmpGlsl;
            return false;
        }
        f << glslSource;
    }

    // Compile via glslc
    const std::string cmd = "glslc -fshader-stage=compute -o \""
        + cachePath + "\" \"" + tmpGlsl + "\" 2>&1";
    int rc = 0;
    const auto output = process_observer::captureCommandStdout(cmd, rc);
    if (!output.has_value()) {
        error = "Failed to start glslc";
        return false;
    }
    if (rc != 0) {
        error = "glslc compilation failed: " + *output;
        std::filesystem::remove(cachePath);
        return false;
    }

    // Read compiled SPIR-V
    std::ifstream f(cachePath, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        error = "Cannot read compiled SPIR-V: " + cachePath;
        return false;
    }
    const auto size = f.tellg();
    if (size <= 0 || size % 4 != 0) {
        error = "Invalid SPIR-V size: " + std::to_string(static_cast<long>(size));
        return false;
    }
    spirv.resize(static_cast<std::size_t>(size) / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(spirv.data()), size);
    if (!f.good()) {
        error = "Failed to read SPIR-V data";
        return false;
    }

    // Clean up temp GLSL
    std::filesystem::remove(tmpGlsl);
    return true;
}

// ─── Video probe ────────────────────────────────────────────────

// ─── Map StageKind to shader ────────────────────────────────────

const char* shaderSourceForStage(StageKind kind) {
    switch (kind) {
        case StageKind::Sharpen:            return kSharpenGlsl;
        case StageKind::Denoise:            return kDenoiseGlsl;
        case StageKind::Dehalo:             return kDehaloGlsl;
        case StageKind::Deblur:             return kDeblurGlsl;
        case StageKind::Upscale:            return kUpscaleGlsl;
        default:                            return nullptr;
    }
}

std::string shaderCacheKey(StageKind kind) {
    return "vk_" + toString(kind);
}

float defaultStrength(StageKind kind) {
    switch (kind) {
        case StageKind::Sharpen:   return 1.0f;
        case StageKind::Denoise:   return 1.5f;
        case StageKind::Dehalo:    return 0.6f;
        case StageKind::Deblur:    return 1.0f;
        case StageKind::Upscale:   return 2.0f; // scale factor
        default:                   return 1.0f;
    }
}

}  // anonymous namespace

// ─────────────────────────────────────────────────────────────────
// Impl — Vulkan pipeline management
// ─────────────────────────────────────────────────────────────────
struct VulkanComputeBackend::Impl {
    bool available = false;
    bool initialized = false;
    std::mutex mtx;

#ifdef AVE_HAVE_VULKAN
    VkInstance       instance       = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice         device         = VK_NULL_HANDLE;
    VkQueue          computeQueue   = VK_NULL_HANDLE;
    uint32_t         queueFamily    = 0;
    VkCommandPool    commandPool    = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;

    struct Pipeline {
        VkShaderModule       shaderModule  = VK_NULL_HANDLE;
        VkDescriptorSetLayout setLayout    = VK_NULL_HANDLE;
        VkPipelineLayout     pipeLayout    = VK_NULL_HANDLE;
        VkPipeline           pipeline      = VK_NULL_HANDLE;
    };
    std::unordered_map<std::string, Pipeline> pipelines;

    bool createDevice(std::string& error);
    void destroyDevice();

    bool getOrCreatePipeline(StageKind kind, Pipeline& out, std::string& error);
    void destroyPipeline(Pipeline& p);

    bool processFrame(
        Pipeline& pipeline,
        const float* inputData, std::size_t inputFloats,
        float* outputData, std::size_t outputFloats,
        int width, int height, float param3,
        std::string& error);
#endif
};

// ─────────────────────────────────────────────────────────────────
VulkanComputeBackend::VulkanComputeBackend()
    : impl_(std::make_unique<Impl>()) {}

VulkanComputeBackend::~VulkanComputeBackend() {
#ifdef AVE_HAVE_VULKAN
    std::lock_guard<std::mutex> lk(impl_->mtx);
    for (auto& [key, p] : impl_->pipelines)
        impl_->destroyPipeline(p);
    impl_->pipelines.clear();
    impl_->destroyDevice();
#endif
}

BackendType VulkanComputeBackend::type() const { return BackendType::VulkanCompute; }
std::string VulkanComputeBackend::name() const { return "Vulkan Compute"; }

bool VulkanComputeBackend::isAvailable(std::string& reason) const {
#ifdef AVE_HAVE_VULKAN
    if (!process_observer::commandInPath("glslc")) {
        reason = "glslc (Vulkan SDK shader compiler) not found in PATH.";
        return false;
    }
    // Quick check: can we even load Vulkan?
    uint32_t count = 0;
    if (vkEnumerateInstanceLayerProperties(&count, nullptr) != VK_SUCCESS) {
        reason = "Vulkan loader not functional.";
        return false;
    }
    return true;
#else
    reason = "Vulkan support not compiled in (AVE_HAVE_VULKAN=OFF).";
    return false;
#endif
}

bool VulkanComputeBackend::initialize(std::string& error) {
#ifdef AVE_HAVE_VULKAN
    std::lock_guard<std::mutex> lk(impl_->mtx);
    if (impl_->initialized) return true;

    if (!impl_->createDevice(error)) return false;

    impl_->initialized = true;
    impl_->available = true;

    std::cout << "[vulkan-compute] Initialised Vulkan compute backend." << std::endl;
    return true;
#else
    error = "Vulkan support not compiled in.";
    return false;
#endif
}

StageResult VulkanComputeBackend::runStage(
        const EnhancementStage& stage, std::string& error) {
#ifdef AVE_HAVE_VULKAN
    // Check if we have a shader for this stage
    if (!shaderSourceForStage(stage.kind)) {
        // No shader for this stage — defer to FFmpeg
        return StageResult::Deferred;
    }

    std::lock_guard<std::mutex> lk(impl_->mtx);
    if (!impl_->initialized) {
        error = "Vulkan compute backend not initialised.";
        return StageResult::Error;
    }

    // Pre-compile the shader so processVideoFile doesn't pay the cost
    Impl::Pipeline pipeline;
    if (!impl_->getOrCreatePipeline(stage.kind, pipeline, error)) {
        std::cerr << "[vulkan-compute] Shader compilation failed for "
                  << toString(stage.kind) << ": " << error
                  << "\n  → Deferring to FFmpeg." << std::endl;
        error.clear();
        return StageResult::Deferred;
    }

    std::cout << "[vulkan-compute] Pipeline ready for "
              << toString(stage.kind) << std::endl;
    return StageResult::Deferred; // Actual work done in processVideoFile
#else
    (void)stage;
    error = "Vulkan not compiled in.";
    return StageResult::Deferred;
#endif
}

StageResult VulkanComputeBackend::processVideoFile(
        const EnhancementStage& stage,
        const std::string& inputVideo,
        const std::string& outputVideo,
        const FrameProgressCb& progressCb,
        std::string& error,
        const ProcessVideoOptions& opts) {
#ifdef AVE_HAVE_VULKAN
    if (!shaderSourceForStage(stage.kind)) {
        return StageResult::Deferred;
    }

    if (!impl_->initialized) {
        error = "Vulkan compute backend not initialised.";
        return StageResult::Error;
    }

    // Get or create pipeline
    Impl::Pipeline pipeline;
    {
        std::lock_guard<std::mutex> lk(impl_->mtx);
        if (!impl_->getOrCreatePipeline(stage.kind, pipeline, error)) {
            std::cerr << "[vulkan-compute] Pipeline creation failed: " << error
                      << " — deferring." << std::endl;
            error.clear();
            return StageResult::Deferred;
        }
    }

    // Get strength/scale parameter
    float param = defaultStrength(stage.kind);
    {
        double value = static_cast<double>(param);
        if (tryGetDouble(stage, stage.kind, "strength", value)) {
            param = static_cast<float>(value);
        }
        if (stage.kind == StageKind::Upscale) {
            if (tryGetDouble(stage, StageKind::Upscale, "scale", value)) {
                param = static_cast<float>(value);
            }
        }
    }

    // Probe input video
    const auto probe = probeVideoStream(inputVideo, error);
    if (!probe.has_value()) {
        return StageResult::Error;
    }

    const int inW = static_cast<int>(probe->width);
    const int inH = static_cast<int>(probe->height);
    const int64_t totalFrames = probe->estimatedFrameCount();
    const double inputFps = probe->effectiveFrameRate(30.0);

    int outW = inW;
    int outH = inH;
    if (stage.kind == StageKind::Upscale) {
        // Check for explicit width/height params
        double widthValue = static_cast<double>(outW);
        double heightValue = static_cast<double>(outH);
        if (tryGetDouble(stage, StageKind::Upscale, "width", widthValue) &&
            tryGetDouble(stage, StageKind::Upscale, "height", heightValue)) {
            outW = static_cast<int>(widthValue);
            outH = static_cast<int>(heightValue);
            if (outW > 0 && outH > 0) {
                param = static_cast<float>(outW) / static_cast<float>(inW);
            }
        } else {
            const int scale = static_cast<int>(std::round(param));
            outW = inW * scale;
            outH = inH * scale;
        }
    }

    std::cout << "[vulkan-compute] Processing " << toString(stage.kind)
              << ": " << inW << "x" << inH << " → " << outW << "x" << outH
              << " param=" << param << std::endl;

    const std::size_t inFrameBytes  = static_cast<std::size_t>(inW) * static_cast<std::size_t>(inH) * 3u;
    const std::size_t outFrameBytes = static_cast<std::size_t>(outW) * static_cast<std::size_t>(outH) * 3u;
    const std::size_t inFloats  = static_cast<std::size_t>(inW) * static_cast<std::size_t>(inH) * 3u;
    const std::size_t outFloats = static_cast<std::size_t>(outW) * static_cast<std::size_t>(outH) * 3u;
    std::vector<float> tensorIn(inFloats);
    std::vector<float> tensorOut(outFloats);
    RgbVideoLoopOptions loopOptions;
    loopOptions.inputVideo = inputVideo;
    loopOptions.outputVideo = outputVideo;
    loopOptions.inputWidth = inW;
    loopOptions.inputHeight = inH;
    loopOptions.outputWidth = outW;
    loopOptions.outputHeight = outH;
    loopOptions.fps = inputFps;
    loopOptions.fallbackFps = 30.0;
    loopOptions.totalFrames = totalFrames;
    loopOptions.backendTag = "vulkan-compute";
    loopOptions.progressLabel = "Vulkan compute";
    loopOptions.noFramesError = "No frames decoded from " + inputVideo;
    loopOptions.preferredSourceMode = frame_io::RgbVideoSourceMode::VulkanTransfer;
    loopOptions.allowSourceFallback = true;
    loopOptions.processOptions = opts;

    const auto loopResult = runRgbVideoLoop(
        loopOptions,
        [&](const std::vector<std::uint8_t>& rgbIn,
            std::vector<std::uint8_t>& rgbOut,
            const int frameIdx,
            std::string& loopError) {
            for (std::size_t i = 0; i < inFrameBytes; ++i) {
                tensorIn[i] = static_cast<float>(rgbIn[i]) / 255.0f;
            }

            {
                std::lock_guard<std::mutex> lk(impl_->mtx);
                if (!impl_->processFrame(pipeline,
                                         tensorIn.data(), inFloats,
                                         tensorOut.data(), outFloats,
                                         inW, inH, param, loopError)) {
                    std::cerr << "[vulkan-compute] Frame " << frameIdx
                              << " compute FAILED: " << loopError << std::endl;
                    return false;
                }
            }

            if (rgbOut.size() != outFrameBytes) {
                rgbOut.resize(outFrameBytes);
            }
            for (std::size_t i = 0; i < outFrameBytes; ++i) {
                const float v = std::clamp(tensorOut[i], 0.0f, 1.0f);
                rgbOut[i] = static_cast<uint8_t>(std::round(v * 255.0f));
            }
            return true;
        },
        progressCb,
        error);

    if (loopResult.stageResult != StageResult::Processed) {
        return loopResult.stageResult;
    }

    std::cout << "[vulkan-compute] Complete: " << loopResult.frameCount
              << " frames via Vulkan compute for " << toString(stage.kind)
              << std::endl;
    return loopResult.stageResult;
#else
    (void)stage; (void)inputVideo; (void)outputVideo; (void)progressCb; (void)opts;
    error = "Vulkan not compiled in.";
    return StageResult::Deferred;
#endif
}

// ─────────────────────────────────────────────────────────────────
// Vulkan pipeline implementation
// ─────────────────────────────────────────────────────────────────
#ifdef AVE_HAVE_VULKAN

bool VulkanComputeBackend::Impl::createDevice(std::string& error) {
    // Create instance
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "AVE Vulkan Compute";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "AVE";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
        error = "Failed to create Vulkan instance.";
        return false;
    }

    // Pick physical device (prefer AMD)
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        error = "No Vulkan-capable GPUs found.";
        return false;
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    if (const auto preferredDevice = preferredVulkanDeviceIndexFromEnv(); preferredDevice.has_value()) {
        if (*preferredDevice >= 0 && static_cast<uint32_t>(*preferredDevice) < deviceCount) {
            physicalDevice = devices[static_cast<std::size_t>(*preferredDevice)];
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(physicalDevice, &props);
            std::cout << "[vulkan-compute] Using explicit Vulkan device index "
                      << *preferredDevice << ": " << props.deviceName << std::endl;
        } else {
            error = "Requested Vulkan device index " + std::to_string(*preferredDevice)
                  + " is out of range for " + std::to_string(deviceCount) + " detected device(s).";
            return false;
        }
    } else {
        // Prefer AMD discrete GPU when available.
        physicalDevice = devices[0];
        bool foundAmdDiscrete = false;
        for (auto& dev : devices) {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(dev, &props);
            if (props.vendorID == 0x1002 && props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                physicalDevice = dev;
                foundAmdDiscrete = true;
                std::cout << "[vulkan-compute] Selected discrete AMD GPU: "
                          << props.deviceName << std::endl;
                break;
            }
        }
        if (!foundAmdDiscrete) {
            for (auto& dev : devices) {
                VkPhysicalDeviceProperties props;
                vkGetPhysicalDeviceProperties(dev, &props);
                if (props.vendorID == 0x1002) {
                    physicalDevice = dev;
                    std::cout << "[vulkan-compute] Selected AMD GPU: "
                              << props.deviceName << std::endl;
                    break;
                }
            }
        }
    }

    {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(physicalDevice, &props);
        std::cout << "[vulkan-compute] GPU: " << props.deviceName
                  << " (vendor 0x" << std::hex << props.vendorID
                  << std::dec << ")" << std::endl;
    }

    // Find compute queue family
    uint32_t qfCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &qfCount, nullptr);
    std::vector<VkQueueFamilyProperties> qfProps(qfCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &qfCount, qfProps.data());

    bool found = false;
    for (uint32_t i = 0; i < qfCount; i++) {
        if (qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            queueFamily = i;
            found = true;
            break;
        }
    }
    if (!found) {
        error = "No compute queue family found.";
        return false;
    }

    // Create logical device
    float priority = 1.0f;
    VkDeviceQueueCreateInfo queueCI{};
    queueCI.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCI.queueFamilyIndex = queueFamily;
    queueCI.queueCount = 1;
    queueCI.pQueuePriorities = &priority;

    VkDeviceCreateInfo deviceCI{};
    deviceCI.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCI.pQueueCreateInfos = &queueCI;
    deviceCI.queueCreateInfoCount = 1;

    if (vkCreateDevice(physicalDevice, &deviceCI, nullptr, &device) != VK_SUCCESS) {
        error = "Failed to create Vulkan logical device.";
        return false;
    }

    vkGetDeviceQueue(device, queueFamily, 0, &computeQueue);

    // Create command pool
    VkCommandPoolCreateInfo poolCI{};
    poolCI.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolCI.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolCI.queueFamilyIndex = queueFamily;

    if (vkCreateCommandPool(device, &poolCI, nullptr, &commandPool) != VK_SUCCESS) {
        error = "Failed to create command pool.";
        return false;
    }

    // Create descriptor pool (generous: 32 sets × 2 storage buffers each)
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 64;

    VkDescriptorPoolCreateInfo dpCI{};
    dpCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpCI.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpCI.maxSets = 32;
    dpCI.poolSizeCount = 1;
    dpCI.pPoolSizes = &poolSize;

    if (vkCreateDescriptorPool(device, &dpCI, nullptr, &descriptorPool) != VK_SUCCESS) {
        error = "Failed to create descriptor pool.";
        return false;
    }

    available = true;
    return true;
}

void VulkanComputeBackend::Impl::destroyDevice() {
    if (descriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        descriptorPool = VK_NULL_HANDLE;
    }
    if (commandPool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device, commandPool, nullptr);
        commandPool = VK_NULL_HANDLE;
    }
    if (device != VK_NULL_HANDLE) {
        vkDestroyDevice(device, nullptr);
        device = VK_NULL_HANDLE;
    }
    if (instance != VK_NULL_HANDLE) {
        vkDestroyInstance(instance, nullptr);
        instance = VK_NULL_HANDLE;
    }
    available = false;
}

bool VulkanComputeBackend::Impl::getOrCreatePipeline(
        StageKind kind, Pipeline& out, std::string& error) {
    const std::string key = shaderCacheKey(kind);
    auto it = pipelines.find(key);
    if (it != pipelines.end()) {
        out = it->second;
        return true;
    }

    const char* src = shaderSourceForStage(kind);
    if (!src) {
        error = "No shader for stage " + toString(kind);
        return false;
    }

    // Compile GLSL → SPIR-V
    std::vector<uint32_t> spirv;
    if (!compileGlslToSpirv(src, key, spirv, error)) {
        return false;
    }

    Pipeline p{};

    // Create shader module
    VkShaderModuleCreateInfo smCI{};
    smCI.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smCI.codeSize = spirv.size() * sizeof(uint32_t);
    smCI.pCode = spirv.data();
    if (vkCreateShaderModule(device, &smCI, nullptr, &p.shaderModule) != VK_SUCCESS) {
        error = "Failed to create shader module for " + key;
        return false;
    }

    // Descriptor set layout: binding 0 = input SSBO, binding 1 = output SSBO
    std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo dslCI{};
    dslCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslCI.bindingCount = static_cast<uint32_t>(bindings.size());
    dslCI.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(device, &dslCI, nullptr, &p.setLayout) != VK_SUCCESS) {
        vkDestroyShaderModule(device, p.shaderModule, nullptr);
        error = "Failed to create descriptor set layout";
        return false;
    }

    // Push constant range: {int width, int height, float param}
    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcRange.offset = 0;
    pcRange.size = 12; // 3 × 4 bytes

    VkPipelineLayoutCreateInfo plCI{};
    plCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plCI.setLayoutCount = 1;
    plCI.pSetLayouts = &p.setLayout;
    plCI.pushConstantRangeCount = 1;
    plCI.pPushConstantRanges = &pcRange;
    if (vkCreatePipelineLayout(device, &plCI, nullptr, &p.pipeLayout) != VK_SUCCESS) {
        vkDestroyDescriptorSetLayout(device, p.setLayout, nullptr);
        vkDestroyShaderModule(device, p.shaderModule, nullptr);
        error = "Failed to create pipeline layout";
        return false;
    }

    // Compute pipeline
    VkComputePipelineCreateInfo cpCI{};
    cpCI.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpCI.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpCI.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpCI.stage.module = p.shaderModule;
    cpCI.stage.pName = "main";
    cpCI.layout = p.pipeLayout;

    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpCI, nullptr, &p.pipeline) != VK_SUCCESS) {
        vkDestroyPipelineLayout(device, p.pipeLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, p.setLayout, nullptr);
        vkDestroyShaderModule(device, p.shaderModule, nullptr);
        error = "Failed to create compute pipeline for " + key;
        return false;
    }

    pipelines[key] = p;
    out = p;
    return true;
}

void VulkanComputeBackend::Impl::destroyPipeline(Pipeline& p) {
    if (p.pipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(device, p.pipeline, nullptr);
    if (p.pipeLayout != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device, p.pipeLayout, nullptr);
    if (p.setLayout != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device, p.setLayout, nullptr);
    if (p.shaderModule != VK_NULL_HANDLE)
        vkDestroyShaderModule(device, p.shaderModule, nullptr);
    p = {};
}

bool VulkanComputeBackend::Impl::processFrame(
        Pipeline& pipeline,
        const float* inputData, std::size_t inputFloats,
        float* outputData, std::size_t outputFloats,
        int width, int height, float param3,
        std::string& error) {

    if (inputData == nullptr || outputData == nullptr) {
        error = "Vulkan compute received null input/output buffers.";
        return false;
    }
    if (inputFloats == 0u || outputFloats == 0u) {
        error = "Vulkan compute received an empty frame buffer.";
        return false;
    }
    if (inputFloats > (std::numeric_limits<VkDeviceSize>::max() / sizeof(float)) ||
        outputFloats > (std::numeric_limits<VkDeviceSize>::max() / sizeof(float))) {
        error = "Vulkan compute buffer size overflow.";
        return false;
    }

    const VkDeviceSize inBytes  = inputFloats * sizeof(float);
    const VkDeviceSize outBytes = outputFloats * sizeof(float);

    // ── Create staging buffers ──────────────────────────────────
    auto createBuffer = [&](VkDeviceSize size, VkBufferUsageFlags usage,
                            VkBuffer& buf, VkDeviceMemory& mem) -> bool {
        VkBufferCreateInfo bCI{};
        bCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bCI.size = size;
        bCI.usage = usage;
        bCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &bCI, nullptr, &buf) != VK_SUCCESS) {
            error = "Failed to create buffer";
            return false;
        }

        VkMemoryRequirements memReqs;
        vkGetBufferMemoryRequirements(device, buf, &memReqs);

        VkPhysicalDeviceMemoryProperties memProps;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);

        uint32_t memType = UINT32_MAX;
        for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
            if ((memReqs.memoryTypeBits & (1u << i)) &&
                (memProps.memoryTypes[i].propertyFlags &
                 (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))) {
                memType = i;
                break;
            }
        }
        if (memType == UINT32_MAX) {
            vkDestroyBuffer(device, buf, nullptr);
            error = "No suitable host-visible memory type";
            return false;
        }

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memReqs.size;
        allocInfo.memoryTypeIndex = memType;

        if (vkAllocateMemory(device, &allocInfo, nullptr, &mem) != VK_SUCCESS) {
            vkDestroyBuffer(device, buf, nullptr);
            error = "Failed to allocate buffer memory";
            return false;
        }

        if (vkBindBufferMemory(device, buf, mem, 0) != VK_SUCCESS) {
            vkFreeMemory(device, mem, nullptr);
            vkDestroyBuffer(device, buf, nullptr);
            mem = VK_NULL_HANDLE;
            buf = VK_NULL_HANDLE;
            error = "Failed to bind buffer memory";
            return false;
        }
        return true;
    };

    VkBuffer inBuf = VK_NULL_HANDLE, outBuf = VK_NULL_HANDLE;
    VkDeviceMemory inMem = VK_NULL_HANDLE, outMem = VK_NULL_HANDLE;

    auto cleanup = [&]() {
        if (inBuf  != VK_NULL_HANDLE) vkDestroyBuffer(device, inBuf, nullptr);
        if (outBuf != VK_NULL_HANDLE) vkDestroyBuffer(device, outBuf, nullptr);
        if (inMem  != VK_NULL_HANDLE) vkFreeMemory(device, inMem, nullptr);
        if (outMem != VK_NULL_HANDLE) vkFreeMemory(device, outMem, nullptr);
    };

    if (!createBuffer(inBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, inBuf, inMem)) {
        cleanup();
        return false;
    }
    if (!createBuffer(outBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, outBuf, outMem)) {
        cleanup();
        return false;
    }

    // ── Allocate descriptor set ─────────────────────────────────
    VkDescriptorSet descSet = VK_NULL_HANDLE;
    {
        VkDescriptorSetAllocateInfo dsAI{};
        dsAI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsAI.descriptorPool = descriptorPool;
        dsAI.descriptorSetCount = 1;
        dsAI.pSetLayouts = &pipeline.setLayout;
        if (vkAllocateDescriptorSets(device, &dsAI, &descSet) != VK_SUCCESS) {
            cleanup();
            error = "Failed to allocate descriptor set";
            return false;
        }
    }

    // Upload input data
    void* mapped = nullptr;
    if (vkMapMemory(device, inMem, 0, inBytes, 0, &mapped) != VK_SUCCESS || mapped == nullptr) {
        vkFreeDescriptorSets(device, descriptorPool, 1, &descSet);
        cleanup();
        error = "Failed to map Vulkan input buffer memory";
        return false;
    }
    std::memcpy(mapped, inputData, static_cast<std::size_t>(inBytes));
    vkUnmapMemory(device, inMem);

    // Update descriptor set
    std::array<VkDescriptorBufferInfo, 2> bufInfos{};
    bufInfos[0].buffer = inBuf;   bufInfos[0].offset = 0; bufInfos[0].range = inBytes;
    bufInfos[1].buffer = outBuf;  bufInfos[1].offset = 0; bufInfos[1].range = outBytes;

    std::array<VkWriteDescriptorSet, 2> writes{};
    for (uint32_t i = 0; i < 2; i++) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = descSet;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufInfos[i];
    }
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()),
                           writes.data(), 0, nullptr);

    // ── Record and submit command buffer ────────────────────────
    VkCommandBuffer cmdBuf = VK_NULL_HANDLE;
    {
        VkCommandBufferAllocateInfo cbAI{};
        cbAI.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbAI.commandPool = commandPool;
        cbAI.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbAI.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(device, &cbAI, &cmdBuf) != VK_SUCCESS) {
            vkFreeDescriptorSets(device, descriptorPool, 1, &descSet);
            cleanup();
            error = "Failed to allocate command buffer";
            return false;
        }
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(cmdBuf, &beginInfo) != VK_SUCCESS) {
        vkFreeCommandBuffers(device, commandPool, 1, &cmdBuf);
        vkFreeDescriptorSets(device, descriptorPool, 1, &descSet);
        cleanup();
        error = "Failed to begin Vulkan command buffer recording";
        return false;
    }

    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline.pipeLayout, 0, 1, &descSet, 0, nullptr);

    // Push constants: {int width, int height, float param}
    struct PushConstants {
        int32_t width;
        int32_t height;
        float   param;
    } pc{width, height, param3};
    vkCmdPushConstants(cmdBuf, pipeline.pipeLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    // Determine dispatch size
    const uint32_t totalPixelsOut = static_cast<uint32_t>(
        outputFloats / 3u); // output pixel count
    const uint32_t workGroups = (totalPixelsOut + 255u) / 256u;
    vkCmdDispatch(cmdBuf, workGroups, 1, 1);

    if (vkEndCommandBuffer(cmdBuf) != VK_SUCCESS) {
        vkFreeCommandBuffers(device, commandPool, 1, &cmdBuf);
        vkFreeDescriptorSets(device, descriptorPool, 1, &descSet);
        cleanup();
        error = "Failed to end Vulkan command buffer recording";
        return false;
    }

    // Submit
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmdBuf;

    VkFence fence = VK_NULL_HANDLE;
    VkFenceCreateInfo fenceCI{};
    fenceCI.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (vkCreateFence(device, &fenceCI, nullptr, &fence) != VK_SUCCESS) {
        vkFreeCommandBuffers(device, commandPool, 1, &cmdBuf);
        vkFreeDescriptorSets(device, descriptorPool, 1, &descSet);
        cleanup();
        error = "Failed to create Vulkan fence";
        return false;
    }

    if (vkQueueSubmit(computeQueue, 1, &submitInfo, fence) != VK_SUCCESS) {
        vkDestroyFence(device, fence, nullptr);
        vkFreeCommandBuffers(device, commandPool, 1, &cmdBuf);
        vkFreeDescriptorSets(device, descriptorPool, 1, &descSet);
        cleanup();
        error = "Failed to submit compute queue";
        return false;
    }

    // Wait for completion (10-second timeout per frame)
    const VkResult waitResult = vkWaitForFences(device, 1, &fence, VK_TRUE, 10'000'000'000ULL);
    vkDestroyFence(device, fence, nullptr);

    if (waitResult != VK_SUCCESS) {
        vkFreeCommandBuffers(device, commandPool, 1, &cmdBuf);
        vkFreeDescriptorSets(device, descriptorPool, 1, &descSet);
        cleanup();
        error = "Vulkan compute timed out or failed (VkResult " + std::to_string(waitResult) + ")";
        return false;
    }

    // Read back output
    if (vkMapMemory(device, outMem, 0, outBytes, 0, &mapped) != VK_SUCCESS || mapped == nullptr) {
        vkFreeCommandBuffers(device, commandPool, 1, &cmdBuf);
        vkFreeDescriptorSets(device, descriptorPool, 1, &descSet);
        cleanup();
        error = "Failed to map Vulkan output buffer memory";
        return false;
    }
    std::memcpy(outputData, mapped, static_cast<std::size_t>(outBytes));
    vkUnmapMemory(device, outMem);

    // Cleanup per-frame resources
    vkFreeCommandBuffers(device, commandPool, 1, &cmdBuf);
    vkFreeDescriptorSets(device, descriptorPool, 1, &descSet);
    cleanup();

    return true;
}

#endif // AVE_HAVE_VULKAN

}  // namespace ave
