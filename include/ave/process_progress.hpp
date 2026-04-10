#pragma once

#include <cstdint>
#include <string>

#include "ave/backend.hpp"

namespace ave::process_progress {

bool isFfmpegProgressField(const std::string& line);

void reportProgressFraction(const FrameProgressCb& progressCb,
                            float base,
                            float span,
                            float frac,
                            const std::string& msg);

void parseFfmpegProgress(const std::string& line,
                         std::int64_t totalFrames,
                         std::int64_t& lastFrame,
                         const FrameProgressCb& progressCb,
                         float base,
                         float span,
                         const std::string& label);

}  // namespace ave::process_progress
