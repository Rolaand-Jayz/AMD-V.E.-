#pragma once

#include <atomic>
#include <functional>
#include <string>
#include <vector>

#include "ave/backend.hpp"
#include "ave/filter_catalog.hpp"
#include "ave/stage.hpp"

namespace ave {

struct EncodeSettings {
    std::string codec   = "libx264";
    std::string profile;            ///< e.g. "high", "main"; empty = codec default.
    int crf = 18;
    std::string preset  = "medium";
    int threads = 0;                ///< 0 = let FFmpeg auto-select.
};

/// Progress callback fired from the processing thread.
/// @param overallPct  Overall job progress 0–100.
/// @param taskPct     Current task progress 0–100.
/// @param msg         Human-readable status message.
using JobProgressCb = std::function<void(int overallPct, int taskPct, const std::string& msg)>;

struct VideoJob {
    std::string inputPath;
    std::string outputPath;
    BackendType requestedBackend = BackendType::Auto;
    std::vector<EnhancementStage> requestedStages;
    EncodeSettings encode;
    bool dryRun = false;
    JobProgressCb progressCb; ///< Optional; called periodically from the processing thread.

    // ── Preview settings ────────────────────────────────────────
    bool previewMode = false;            ///< Process only a short clip for quick preview.
    double previewDurationSec = 10.0;    ///< Duration of the preview clip in seconds.
    FramePreviewCb framePreviewCb;       ///< Optional; called with processed frames for live display.
    int previewFrameInterval = 15;       ///< Emit preview callback every N frames.

    // ── Pause / Cancel ──────────────────────────────────────────
    std::atomic<bool>* cancelFlag = nullptr;  ///< Shared with UI; true = abort processing.
    std::atomic<bool>* pauseFlag  = nullptr;  ///< Shared with UI; true = sleep-wait between frames.

    // ── Catalog filters ──────────────────────────────────────────
    std::vector<ActiveFilter> catalogFilters;  ///< Filters from the filter browser.
};

}  // namespace ave
