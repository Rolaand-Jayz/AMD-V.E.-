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
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "ave/frame_io.hpp"
#include "ave/stage.hpp"
#include "ave/types.hpp"

#ifdef AVE_HAVE_VULKAN
#include <vulkan/vulkan.h>
#endif

namespace ave {

// ─────────────────────────────────────────────────────────────────
// Embedded GLSL compute shader sources
// ─────────────────────────────────────────────────────────────────
namespace {

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

bool fileExistsVk(const std::string& p) {
    std::error_code ec;
    return std::filesystem::exists(p, ec);
}

bool commandInPathVk(const std::string& cmd) {
    const char* pathEnv = std::getenv("PATH");
    if (!pathEnv) return false;
    std::string path(pathEnv);
    std::size_t start = 0;
    while (start <= path.size()) {
        auto end = path.find(':', start);
        if (end == std::string::npos) end = path.size();
        const std::string dir = path.substr(start, end - start);
        if (!dir.empty() && fileExistsVk((std::filesystem::path(dir) / cmd).string()))
            return true;
        if (end == path.size()) break;
        start = end + 1;
    }
    return false;
}

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
    if (fileExistsVk(cachePath)) {
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
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        error = "Failed to start glslc";
        return false;
    }
    std::string output;
    std::array<char, 512> buf{};
    while (std::fgets(buf.data(), static_cast<int>(buf.size()), pipe))
        output += buf.data();
    const int rc = pclose(pipe);
    if (rc != 0) {
        error = "glslc compilation failed: " + output;
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

struct ProbeResultVk {
    int width       = 0;
    int height      = 0;
    double fps      = 0.0;
    int64_t totalFrames = 0;
};

bool probeVideoVk(const std::string& path, ProbeResultVk& result, std::string& error) {
    {
        const std::string cmd =
            "ffprobe -v error -select_streams v:0 "
            "-show_entries stream=width,height,r_frame_rate "
            "-of csv=p=0 \"" + path + "\" 2>/dev/null";
        FILE* p = popen(cmd.c_str(), "r");
        if (!p) { error = "ffprobe failed (popen)"; return false; }
        std::array<char, 256> buf{};
        if (std::fgets(buf.data(), static_cast<int>(buf.size()), p)) {
            int num = 0, den = 1;
            if (std::sscanf(buf.data(), "%d,%d,%d/%d",
                            &result.width, &result.height, &num, &den) >= 3) {
                if (den > 0) result.fps = static_cast<double>(num) / static_cast<double>(den);
            }
        }
        pclose(p);
    }
    {
        const std::string cmd =
            "ffprobe -v error -count_packets -select_streams v:0 "
            "-show_entries stream=nb_read_packets -of csv=p=0 \"" + path + "\" 2>/dev/null";
        FILE* p = popen(cmd.c_str(), "r");
        if (p) {
            std::array<char, 64> buf{};
            if (std::fgets(buf.data(), static_cast<int>(buf.size()), p))
                result.totalFrames = std::atoll(buf.data());
            pclose(p);
        }
    }
    if (result.width <= 0 || result.height <= 0) {
        error = "ffprobe returned invalid dimensions for " + path;
        return false;
    }
    return true;
}

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
    if (!commandInPathVk("glslc")) {
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
        auto it = stage.params.find("strength");
        if (it != stage.params.end()) {
            if (const auto* d = std::get_if<double>(&it->second))
                param = static_cast<float>(*d);
        }
        if (stage.kind == StageKind::Upscale) {
            auto sit = stage.params.find("scale");
            if (sit != stage.params.end()) {
                if (const auto* d = std::get_if<double>(&sit->second))
                    param = static_cast<float>(*d);
            }
        }
    }

    // Probe input video
    ProbeResultVk probe;
    if (!probeVideoVk(inputVideo, probe, error)) {
        return StageResult::Error;
    }

    const int inW = probe.width;
    const int inH = probe.height;
    const int64_t totalFrames = probe.totalFrames;

    int outW = inW;
    int outH = inH;
    if (stage.kind == StageKind::Upscale) {
        // Check for explicit width/height params
        auto wit = stage.params.find("width");
        auto hit = stage.params.find("height");
        if (wit != stage.params.end() && hit != stage.params.end()) {
            if (const auto* dw = std::get_if<double>(&wit->second))
                outW = static_cast<int>(*dw);
            if (const auto* dh = std::get_if<double>(&hit->second))
                outH = static_cast<int>(*dh);
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

    frame_io::VulkanVideoReader reader;
    if (!reader.open(inputVideo, error)) {
        return StageResult::Error;
    }

    frame_io::VulkanVideoWriter writer;
    AVRational fps = reader.frameRate();
    if (fps.num <= 0 || fps.den <= 0) {
        fps = AVRational{static_cast<int>(std::round(probe.fps > 0.0 ? probe.fps : 30.0)), 1};
    }
    if (!writer.open(outputVideo, outW, outH, fps, error)) {
        reader.close();
        return StageResult::Error;
    }

    const std::size_t inFrameBytes  = static_cast<std::size_t>(inW) * static_cast<std::size_t>(inH) * 3u;
    const std::size_t outFrameBytes = static_cast<std::size_t>(outW) * static_cast<std::size_t>(outH) * 3u;
    const std::size_t inFloats  = static_cast<std::size_t>(inW) * static_cast<std::size_t>(inH) * 3u;
    const std::size_t outFloats = static_cast<std::size_t>(outW) * static_cast<std::size_t>(outH) * 3u;

    const int64_t maxFrames = opts.previewDurationSec > 0.0
        ? static_cast<int64_t>(opts.previewDurationSec * (probe.fps > 0.0 ? probe.fps : 30.0) + 0.5)
        : 0;

    std::vector<uint8_t> rgbIn(inFrameBytes);
    std::vector<uint8_t> rgbOut(outFrameBytes);
    std::vector<float> tensorIn(inFloats);
    std::vector<float> tensorOut(outFloats);
    int frameIdx = 0;
    bool cancelled = false;

    while (true) {
        if (maxFrames > 0 && frameIdx >= maxFrames) {
            break;
        }

        // ── Cancel / Pause check ────────────────────────────────
        if (opts.cancelFlag && opts.cancelFlag->load(std::memory_order_relaxed)) {
            std::cout << "[vulkan-compute] Cancelled at frame " << frameIdx << std::endl;
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

        AVFrame* inputFrame = nullptr;
        if (!reader.readFrame(inputFrame, error)) {
            writer.close();
            reader.close();
            return StageResult::Error;
        }
        if (inputFrame == nullptr) {
            break;
        }

        if (!frame_io::avFrameToRgb24(inputFrame, inW, inH, rgbIn, error)) {
            writer.close();
            reader.close();
            return StageResult::Error;
        }

        for (std::size_t i = 0; i < inFrameBytes; ++i) {
            tensorIn[i] = static_cast<float>(rgbIn[i]) / 255.0f;
        }

        {
            std::lock_guard<std::mutex> lk(impl_->mtx);
            if (!impl_->processFrame(pipeline,
                                      tensorIn.data(), inFloats,
                                      tensorOut.data(), outFloats,
                                      inW, inH, param, error)) {
                std::cerr << "[vulkan-compute] Frame " << frameIdx
                          << " compute FAILED: " << error << std::endl;
                writer.close();
                reader.close();
                return StageResult::Error;
            }
        }

        for (std::size_t i = 0; i < outFrameBytes; ++i) {
            float v = std::clamp(tensorOut[i], 0.0f, 1.0f);
            rgbOut[i] = static_cast<uint8_t>(std::round(v * 255.0f));
        }

        AVFrame* outputFrame = frame_io::rgb24ToAvFrame(rgbOut.data(), outW, outH, error);
        if (outputFrame == nullptr) {
            writer.close();
            reader.close();
            return StageResult::Error;
        }
        const bool writeOk = writer.writeFrame(outputFrame, error);
        av_frame_free(&outputFrame);
        if (!writeOk) {
            writer.close();
            reader.close();
            return StageResult::Error;
        }

        ++frameIdx;

        // Emit live frame preview
        const int pvInterval = opts.previewFrameInterval > 0 ? opts.previewFrameInterval : 15;
        if (opts.framePreviewCb && (frameIdx % pvInterval == 1 || pvInterval == 1)) {
            opts.framePreviewCb(rgbOut.data(), outW, outH);
        }

        if (progressCb) {
            float frac = 0.0f;
            const int64_t effectiveTotal = maxFrames > 0 ? maxFrames : totalFrames;
            if (effectiveTotal > 0) {
                frac = static_cast<float>(frameIdx) / static_cast<float>(effectiveTotal);
                frac = std::min(frac, 1.0f);
            } else {
                frac = 1.0f - 1.0f / (1.0f + static_cast<float>(frameIdx) * 0.01f);
            }
            progressCb(frac, "Vulkan compute: frame " + std::to_string(frameIdx)
                        + (totalFrames > 0 ? "/" + std::to_string(totalFrames) : ""));
        }

        if (frameIdx % 30 == 0) {
            std::cout << "[vulkan-compute] Processed " << frameIdx << " frames"
                      << (totalFrames > 0 ? " / " + std::to_string(totalFrames) : "")
                      << std::endl;
        }
    }

    writer.close();
    reader.close();

    if (cancelled) {
        error = "Processing cancelled by user at frame " + std::to_string(frameIdx);
        return StageResult::Cancelled;
    }
    if (frameIdx == 0) {
        error = "No frames decoded from " + inputVideo;
        return StageResult::Error;
    }

    std::cout << "[vulkan-compute] Complete: " << frameIdx
              << " frames via Vulkan compute for " << toString(stage.kind)
              << std::endl;

    if (progressCb) {
        progressCb(1.0f, "Vulkan compute complete — "
                   + std::to_string(frameIdx) + " frames.");
    }

    return StageResult::Processed;
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

    // Prefer AMD GPU
    physicalDevice = devices[0];
    for (auto& dev : devices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(dev, &props);
        if (props.vendorID == 0x1002) { // AMD vendor ID
            physicalDevice = dev;
            std::cout << "[vulkan-compute] Selected AMD GPU: "
                      << props.deviceName << std::endl;
            break;
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

        vkBindBufferMemory(device, buf, mem, 0);
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

    // Upload input data
    void* mapped = nullptr;
    vkMapMemory(device, inMem, 0, inBytes, 0, &mapped);
    std::memcpy(mapped, inputData, static_cast<std::size_t>(inBytes));
    vkUnmapMemory(device, inMem);

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
    vkBeginCommandBuffer(cmdBuf, &beginInfo);

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

    vkEndCommandBuffer(cmdBuf);

    // Submit
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmdBuf;

    VkFence fence = VK_NULL_HANDLE;
    VkFenceCreateInfo fenceCI{};
    fenceCI.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vkCreateFence(device, &fenceCI, nullptr, &fence);

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
    vkMapMemory(device, outMem, 0, outBytes, 0, &mapped);
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
