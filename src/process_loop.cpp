#include "ave/process_loop.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <thread>

namespace ave {

std::int64_t computePreviewFrameLimit(const double previewDurationSec,
                                      const double fps,
                                      const double defaultFps) {
    if (previewDurationSec <= 0.0) {
        return 0;
    }
    const double effectiveFps = fps > 0.0 ? fps : defaultFps;
    if (effectiveFps <= 0.0) {
        return 0;
    }
    return std::max<std::int64_t>(
        1, static_cast<std::int64_t>(previewDurationSec * effectiveFps + 0.5));
}

bool waitForProcessingResumeOrCancel(const ProcessVideoOptions& opts,
                                     const std::string& backendTag,
                                     const int frameIdx) {
    const auto logCancelled = [&]() {
        std::cout << '[' << backendTag << "] Cancelled at frame "
                  << frameIdx << std::endl;
    };

    if (opts.cancelFlag && opts.cancelFlag->load(std::memory_order_relaxed)) {
        logCancelled();
        return true;
    }
    while (opts.pauseFlag && opts.pauseFlag->load(std::memory_order_relaxed)) {
        if (opts.cancelFlag && opts.cancelFlag->load(std::memory_order_relaxed)) {
            logCancelled();
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return false;
}

float computeProcessingProgress(const std::int64_t completedFrames,
                                const std::int64_t totalFrames) {
    if (totalFrames > 0) {
        const auto completed = static_cast<float>(completedFrames);
        const auto total = static_cast<float>(totalFrames);
        return std::clamp(completed / total, 0.0f, 1.0f);
    }
    const auto completed = static_cast<float>(completedFrames);
    return 1.0f - 1.0f / (1.0f + completed * 0.01f);
}

}  // namespace ave
