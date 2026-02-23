#pragma once

#include <functional>
#include <string>

#include "ave/stage.hpp"

namespace ave {

enum class BackendType {
    Auto,
    MiGraphX,
    NcnnVulkan
};

std::string toString(BackendType type);

struct BackendInfo {
    BackendType type;
    std::string name;
    bool available;
    std::string detail;
};

// Result of IAcceleratorBackend::runStage().
//   Processed — backend performed actual frame-by-frame inference;
//               FFmpeg should NOT re-apply this stage as a filter.
//   Deferred  — backend could not do inference (model loaded but
//               no GPU frame loop); FFmpeg SHOULD apply its filter
//               chain for this stage instead.
//   Error     — fatal failure; abort the pipeline.
enum class StageResult {
    Processed,
    Deferred,
    Error
};

// Progress callback for frame-by-frame AI processing.
//   frac — fraction complete [0.0, 1.0]
//   msg  — human-readable status string
using FrameProgressCb = std::function<void(float frac, const std::string& msg)>;

class IAcceleratorBackend {
  public:
    virtual ~IAcceleratorBackend() = default;

    virtual BackendType type() const = 0;
    virtual std::string name() const = 0;
    virtual bool isAvailable(std::string& reason) const = 0;
    virtual bool initialize(std::string& error) = 0;

    // Run a single enhancement stage.  Must return:
    //   Processed — frames were actually enhanced via AI inference.
    //   Deferred  — model was validated/loaded, but no frame
    //               processing occurred; the caller should fall back
    //               to FFmpeg filters for this stage.
    //   Error     — a fatal error occurred (details in `error`).
    virtual StageResult runStage(const EnhancementStage& stage, std::string& error) = 0;

    // Process a directory of PNG frames through the model for a
    // given enhancement stage.
    //
    //   inputDir  — directory of %08d.png frames (RGB24, sequential).
    //   outputDir — destination for processed frames (same naming).
    //   progressCb — optional per-frame progress callback.
    //
    // Returns:
    //   Processed — all frames were enhanced via AI inference.
    //   Deferred  — AI inference not available (library not compiled,
    //               model format mismatch, etc.); caller should fall
    //               back to FFmpeg filters for this stage.
    //   Error     — a fatal error occurred (details in `error`).
    virtual StageResult processFrameDir(
        const EnhancementStage& stage,
        const std::string& inputDir,
        const std::string& outputDir,
        const FrameProgressCb& progressCb,
        std::string& error) = 0;
};

}  // namespace ave
