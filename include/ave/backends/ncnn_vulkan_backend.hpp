#pragma once

#include <memory>
#include <string>

#include "ave/backend.hpp"

namespace ave {

// ─────────────────────────────────────────────────────────────────
// NcnnVulkanBackend
// ─────────────────────────────────────────────────────────────────
// Runs inference via NCNN with a Vulkan GPU backend.  Used as the
// fallback when MiGraphX / ROCm are unavailable, or when the user
// explicitly selects an NCNN model variant from the catalog.
//
// NCNN models are stored as paired .param/.bin files in the
// ModelManager's downloaded/ directory.
//
// Frame tiles are submitted as ncnn::Mat tensors.
// ─────────────────────────────────────────────────────────────────

class NcnnVulkanBackend final : public IAcceleratorBackend {
  public:
    NcnnVulkanBackend();
    ~NcnnVulkanBackend() override;

    // IAcceleratorBackend
    BackendType type()  const override;
    std::string name()  const override;
    bool isAvailable(std::string& reason) const override;
    bool initialize(std::string& error)   override;
    StageResult runStage(const EnhancementStage& stage, std::string& error) override;

    // Process a directory of PNG frames through the loaded NCNN
    // network using Vulkan GPU inference.
    StageResult processVideoFile(
        const EnhancementStage& stage,
        const std::string& inputVideo,
        const std::string& outputVideo,
        const FrameProgressCb& progressCb,
        std::string& error) override;

    // ── Extended API ────────────────────────────────────────────

    // Pre-load a model by catalog id.
    bool preloadModel(const std::string& modelId, std::string& error);
    void evictModel(const std::string& modelId);
    void evictAll();

    int gpuIndex() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ave
