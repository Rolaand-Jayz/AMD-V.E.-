#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ave {

// ─────────────────────────────────────────────────────────────────
// SceneDetector
// ─────────────────────────────────────────────────────────────────
// Detects scene cuts in a video file using FFmpeg's scene-detection
// filter.  The resulting frame indices can be used by the frame
// interpolation path to avoid generating blurred ghosting frames
// across hard cut boundaries.
// ─────────────────────────────────────────────────────────────────

struct SceneDetectorOptions {
    // Detection threshold: 0.0 (detect every frame) – 1.0 (nothing)
    // Typical useful range: 0.20 – 0.45
    double threshold = 0.30;

    // Limit to the first N seconds of the video (0 = no limit)
    double maxDurationSeconds = 0.0;
};

struct SceneCut {
    std::int64_t frameIndex = 0; // 0-based frame number of the detected cut
    double       pts        = 0.0; // presentation timestamp in seconds
    double       score      = 0.0; // detection score (0.0–1.0)
};

class SceneDetector {
  public:
    // Returns false and sets error if ffprobe is unavailable or the
    // input file cannot be read.
    bool detect(const std::string&        inputPath,
                const SceneDetectorOptions& options,
                std::vector<SceneCut>&      cuts,
                std::string&                error) const;

    // Returns true if the frame at 'frameIndex' is within 'margin'
    // frames of any detected scene cut.
    static bool isCutAdjacent(const std::vector<SceneCut>& cuts,
                               std::int64_t frameIndex,
                               std::int64_t margin = 1);
};

}  // namespace ave
