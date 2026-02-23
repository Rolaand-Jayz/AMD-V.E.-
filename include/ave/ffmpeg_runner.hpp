#pragma once

#include <string>
#include <vector>

#include "ave/backend.hpp"
#include "ave/job.hpp"
#include "ave/stage.hpp"

namespace ave {

class FfmpegRunner {
  public:
    bool isAvailable(std::string& error) const;

    // Encode a video applying the given ordered enhancement stages.
    // When a non-null backend is provided, stages that carry a
    // "model_path" parameter are processed via AI frame-by-frame
    // inference through the backend rather than basic FFmpeg filters.
    bool encode(const VideoJob& job,
                const std::vector<EnhancementStage>& orderedStages,
                IAcceleratorBackend* backend,
                std::string& error) const;
};

}  // namespace ave
