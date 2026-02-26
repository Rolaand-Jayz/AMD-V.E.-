// DISABLED: VapourSynth/GLSL/FilterCatalog feature — commented out, not removed.
#if 0  // ── entire file disabled ──────────────────────────────

#pragma once

// ─────────────────────────────────────────────────────────────────
// glsl_shader_backend.hpp — GLSL shader processing backend
//
// Applies GLSL compute/fragment shaders to video frames using
// Vulkan via the libplacebo library (or mpv's GPU pipeline).
//
// This backend supports the popular .glsl shader format used by
// projects like Anime4K, FSRCNNX, SSimSuperRes, KrigBilateral, etc.
// Shaders are applied per-frame through a Vulkan compute pipeline.
//
// Requires one of:
//   - libplacebo with Vulkan support (preferred)
//   - mpv with --glsl-shaders support (fallback)
//
// The backend discovers available Vulkan devices and selects the
// first AMD GPU for processing.
// ─────────────────────────────────────────────────────────────────

#include <memory>
#include <string>
#include <vector>

#include "ave/backend.hpp"

namespace ave {

class GlslShaderBackend final : public IAcceleratorBackend {
  public:
    GlslShaderBackend();
    ~GlslShaderBackend() override;

    // IAcceleratorBackend
    BackendType type()  const override;
    std::string name()  const override;
    bool isAvailable(std::string& reason) const override;
    bool initialize(std::string& error)   override;
    StageResult runStage(const EnhancementStage& stage, std::string& error) override;

    // Process a directory of PNG frames through GLSL shaders
    // for the given enhancement stage.
    StageResult processVideoFile(
        const EnhancementStage& stage,
        const std::string& inputVideo,
        const std::string& outputVideo,
        const FrameProgressCb& progressCb,
        std::string& error) override;

    // ── Extended API ────────────────────────────────────────────

    // Get list of bundled shader paths for a stage kind.
    static std::vector<std::string> defaultShadersForStage(StageKind kind);

    // Accept enabled filters from the catalog UI.
    void setCatalogFilters(
        const std::vector<ActiveFilter>& filters) override;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ave

#endif // ── entire file disabled ──────────────────────────────
