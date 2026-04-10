#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "ave/process_progress.hpp"

namespace {

void check(const bool condition, const char* message) {
    if (condition) {
        return;
    }
    std::cerr << "process_progress_tests failed: " << message << '\n';
    std::abort();
}

void testIsFfmpegProgressFieldRecognizesKnownKeys() {
    check(ave::process_progress::isFfmpegProgressField("frame=42"),
          "frame= should be recognized as FFmpeg progress");
    check(ave::process_progress::isFfmpegProgressField("progress=end"),
          "progress=end should be recognized as FFmpeg progress");
    check(!ave::process_progress::isFfmpegProgressField("random diagnostic"),
          "non-progress lines should not be classified as FFmpeg progress");
}

void testParseFfmpegProgressReportsFramesAndCompletion() {
    std::vector<std::pair<float, std::string>> updates;
    ave::FrameProgressCb progressCb = [&](const float frac, const std::string& msg) {
        updates.emplace_back(frac, msg);
    };

    std::int64_t lastFrame = -1;
    ave::process_progress::parseFfmpegProgress(
        "frame=3", 6, lastFrame, progressCb, 0.2f, 0.5f, "Pipeline");
    ave::process_progress::parseFfmpegProgress(
        "progress=end", 6, lastFrame, progressCb, 0.2f, 0.5f, "Pipeline");

    check(updates.size() == 2,
          "parseFfmpegProgress should emit a frame update and completion update");
    check(std::fabs(updates[0].first - 0.45f) < 0.0001f,
          "frame update should respect base/span progress scaling");
    check(updates[0].second == "Pipeline - frame 3/6",
          "frame update should include the label and frame count");
    check(std::fabs(updates[1].first - 0.7f) < 0.0001f,
          "completion update should map to base + span");
    check(updates[1].second == "Pipeline complete.",
          "completion update should use the standard completion label");
}

}  // namespace

int main() {
    testIsFfmpegProgressFieldRecognizesKnownKeys();
    testParseFfmpegProgressReportsFramesAndCompletion();
    return 0;
}
