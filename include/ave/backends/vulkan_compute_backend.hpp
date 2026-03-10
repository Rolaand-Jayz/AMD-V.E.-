#pragma once

// ─────────────────────────────────────────────────────────────────
// vulkan_compute_backend.hpp — Vulkan compute shader backend
//
// GPU-accelerated image processing using raw Vulkan compute
// pipelines.  Does not require MiGraphX, NCNN, or any external AI
// runtime — only the Vulkan SDK.
//
// Supported stages:
//   Sharpen        — Unsharp-mask via compute shader
//   Denoise        — Bilateral filter approximation
//   Upscale        — Lanczos-approximation upscale
//   Dehalo         — Halo removal (inverse unsharp)
//
// Stages without a dedicated shader return Deferred so the FFmpeg
// filter chain handles them instead.
//
// Requires:
//   - AVE_HAVE_VULKAN=ON (compile time)
//   - Vulkan-capable GPU at runtime
//   - `glslc` in PATH (Vulkan SDK shader compiler) for first-run
//     SPIR-V compilation; compiled shaders are cached locally.
// ─────────────────────────────────────────────────────────────────

#include <memory>
#include <string>

#include "ave/backend.hpp"

namespace ave {

class VulkanComputeBackend final : public IAcceleratorBackend {
  public:
    VulkanComputeBackend();
    ~VulkanComputeBackend() override;

    // IAcceleratorBackend
    BackendType type()  const override;
    std::string name()  const override;
    bool isAvailable(std::string& reason) const override;
    bool initialize(std::string& error)   override;
    StageResult runStage(const EnhancementStage& stage, std::string& error) override;

    StageResult processVideoFile(
        const EnhancementStage& stage,
        const std::string& inputVideo,
        const std::string& outputVideo,
        const FrameProgressCb& progressCb,
        std::string& error,
        const ProcessVideoOptions& opts = {}) override;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ave
