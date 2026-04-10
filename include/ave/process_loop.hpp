#pragma once

#include <cstdint>
#include <string>

#include "ave/backend.hpp"

namespace ave {

std::int64_t computePreviewFrameLimit(double previewDurationSec,
                                      double fps,
                                      double defaultFps = 30.0);

bool waitForProcessingResumeOrCancel(const ProcessVideoOptions& opts,
                                     const std::string& backendTag,
                                     int frameIdx);

float computeProcessingProgress(std::int64_t completedFrames,
                                std::int64_t totalFrames);

}  // namespace ave
