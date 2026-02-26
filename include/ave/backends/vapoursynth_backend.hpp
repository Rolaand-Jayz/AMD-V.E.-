// DISABLED: VapourSynth/GLSL/FilterCatalog feature — commented out, not removed.
#if 0  // ── entire file disabled ──────────────────────────────

#pragma once

// ─────────────────────────────────────────────────────────────────
// vapoursynth_backend.hpp — VapourSynth processing backend
//
// Runs video enhancement through VapourSynth filter scripts via the
// vspipe command-line tool.  VapourSynth scripts can apply arbitrary
// frame-level processing including AI super-resolution filters
// (e.g., vs-mlrt with NCNN/TRT), denoising (BM3D, KNLMeansCL),
// dehaloing, etc.
//
// Requires:
//   - VapourSynth R55+ with vspipe in $PATH
//   - Desired VS plugins installed (e.g., vs-mlrt, mvtools, etc.)
//
// The backend generates a temporary .vpy script per stage, feeds
// input frames, and collects output frames.
// ─────────────────────────────────────────────────────────────────

#include <memory>
#include <string>

#include "ave/backend.hpp"

namespace ave {

class VapourSynthBackend final : public IAcceleratorBackend {
  public:
    VapourSynthBackend();
    ~VapourSynthBackend() override;

    // IAcceleratorBackend
    BackendType type()  const override;
    std::string name()  const override;
    bool isAvailable(std::string& reason) const override;
    bool initialize(std::string& error)   override;
    StageResult runStage(const EnhancementStage& stage, std::string& error) override;

    // Process a directory of PNG frames through a VapourSynth
    // filter script for the given enhancement stage.
    StageResult processVideoFile(
        const EnhancementStage& stage,
        const std::string& inputVideo,
        const std::string& outputVideo,
        const FrameProgressCb& progressCb,
        std::string& error) override;

    // Accept enabled filters from the catalog UI.
    void setCatalogFilters(
        const std::vector<ActiveFilter>& filters) override;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ave

#endif // ── entire file disabled ──────────────────────────────
