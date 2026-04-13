#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>

#include "ave/stage.hpp"

namespace ave {

enum class BackendType {
    Auto,
    MiGraphX,
    RocmHip,
    NcnnVulkan,
    VulkanCompute,
    VapourSynth,
    GlslShader
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
    Error,
    Cancelled
};

// Progress callback for frame-by-frame AI processing.
//   frac — fraction complete [0.0, 1.0]
//   msg  — human-readable status string
using FrameProgressCb = std::function<void(float frac, const std::string& msg)>;

// Callback for delivering live preview frames during processing.
//   rgb    — pointer to RGB24 pixel data (3 bytes per pixel, row-major)
//   width  — frame width in pixels
//   height — frame height in pixels
// The data is only valid for the duration of the call.
using FramePreviewCb = std::function<void(const std::uint8_t* rgb, int width, int height)>;

// Options passed to processVideoFile for preview and control functionality.
struct ProcessVideoOptions {
    double previewDurationSec = 0.0;     ///< Active preview trim budget for this processing session (0 = full video)
    FramePreviewCb framePreviewCb;       ///< Optional live frame preview callback
    int previewFrameInterval = 15;       ///< Emit preview every N frames (reduces overhead)
    std::atomic<bool>* cancelFlag = nullptr;  ///< Non-null → check each frame; true = stop
    std::atomic<bool>* pauseFlag  = nullptr;  ///< Non-null → check each frame; true = sleep-wait
    bool directOutputEncode = false;     ///< Backend may write the final delivery encode directly.
    std::string outputCodec = "libx264";
    std::string outputProfile;
    int outputCrf = 18;
    std::string outputPreset = "medium";
    int outputThreads = 0;
};

class IAcceleratorBackend {
  public:
    virtual ~IAcceleratorBackend() = default;

    virtual BackendType type() const = 0;
    virtual std::string name() const = 0;
    virtual bool isAvailable(std::string& reason) const = 0;
    virtual bool initialize(std::string& error) = 0;
    virtual bool supportsDirectOutputEncode() const { return false; }

    // Run a single enhancement stage.  Must return:
    //   Processed — frames were actually enhanced via AI inference.
    //   Deferred  — model was validated/loaded, but no frame
    //               processing occurred; the caller should fall back
    //               to FFmpeg filters for this stage.
    //   Error     — a fatal error occurred (details in `error`).
    virtual StageResult runStage(const EnhancementStage& stage, std::string& error) = 0;

    // Process a video file through the model for a given enhancement stage.
    //
    //   inputVideo  — path to the input video file.
    //   outputVideo — destination path for the processed video file.
    //   progressCb  — optional per-frame progress callback.
    //   opts        — optional preview settings (duration limit, frame preview).
    //
    // Returns:
    //   Processed — all frames were enhanced via AI inference.
    //   Deferred  — AI inference not available (library not compiled,
    //               model format mismatch, etc.); caller should fall
    //               back to FFmpeg filters for this stage.
    //   Error     — a fatal error occurred (details in `error`).
    virtual StageResult processVideoFile(
        const EnhancementStage& stage,
        const std::string& inputVideo,
        const std::string& outputVideo,
        const FrameProgressCb& progressCb,
        std::string& error,
        const ProcessVideoOptions& opts = {}) = 0;
};

}  // namespace ave
