#include "ave/process_progress.hpp"

#include <algorithm>

#include "ave/process_observer.hpp"

namespace ave::process_progress {

bool isFfmpegProgressField(const std::string& line) {
    return line.rfind("frame=", 0) == 0 ||
           line.rfind("fps=", 0) == 0 ||
           line.rfind("stream_", 0) == 0 ||
           line.rfind("bitrate=", 0) == 0 ||
           line.rfind("total_size=", 0) == 0 ||
           line.rfind("out_time_us=", 0) == 0 ||
           line.rfind("out_time_ms=", 0) == 0 ||
           line.rfind("out_time=", 0) == 0 ||
           line.rfind("dup_frames=", 0) == 0 ||
           line.rfind("drop_frames=", 0) == 0 ||
           line.rfind("speed=", 0) == 0 ||
           line.rfind("progress=", 0) == 0;
}

void reportProgressFraction(const FrameProgressCb& progressCb,
                            const float base,
                            const float span,
                            const float frac,
                            const std::string& msg) {
    if (!progressCb) {
        return;
    }
    const float clamped = std::clamp(frac, 0.0f, 1.0f);
    progressCb(base + clamped * span, msg);
}

void parseFfmpegProgress(const std::string& line,
                         const std::int64_t totalFrames,
                         std::int64_t& lastFrame,
                         const FrameProgressCb& progressCb,
                         const float base,
                         const float span,
                         const std::string& label) {
    if (!progressCb) {
        return;
    }

    if (line.rfind("frame=", 0) == 0) {
        try {
            const auto frame =
                std::stoll(process_observer::trimOutput(line.substr(6)));
            if (frame <= lastFrame) {
                return;
            }
            lastFrame = frame;
            const float frac = totalFrames > 0
                ? std::min(1.0f, static_cast<float>(frame) / static_cast<float>(totalFrames))
                : 0.0f;
            std::string msg = label + " - frame " + std::to_string(frame);
            if (totalFrames > 0) {
                msg += "/" + std::to_string(totalFrames);
            }
            reportProgressFraction(progressCb, base, span, frac, msg);
        } catch (...) {
        }
        return;
    }

    if (line == "progress=end") {
        reportProgressFraction(progressCb, base, span, 1.0f, label + " complete.");
    }
}

}  // namespace ave::process_progress
