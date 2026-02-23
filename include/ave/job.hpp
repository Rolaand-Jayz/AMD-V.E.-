#pragma once

#include <functional>
#include <string>
#include <vector>

#include "ave/backend.hpp"
#include "ave/stage.hpp"

namespace ave {

struct EncodeSettings {
    std::string codec = "libx264";
    int crf = 18;
    std::string preset = "medium";
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
};

}  // namespace ave
