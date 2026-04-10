#include <atomic>
#include <cstdlib>
#include <iostream>

#include "ave/process_loop.hpp"

namespace {

void check(const bool condition, const char* message) {
    if (condition) {
        return;
    }
    std::cerr << "process_loop_tests failed: " << message << '\n';
    std::abort();
}

void testPreviewFrameLimit() {
    check(ave::computePreviewFrameLimit(0.0, 24.0) == 0,
          "zero preview duration should disable the frame limit");
    check(ave::computePreviewFrameLimit(2.0, 24.0) == 48,
          "preview frame limit should round from duration and fps");
    check(ave::computePreviewFrameLimit(1.0, 0.0, 25.0) == 25,
          "preview frame limit should use the fallback fps when needed");
}

void testProgressComputation() {
    check(ave::computeProcessingProgress(5, 10) == 0.5f,
          "known totals should use linear progress");
    const float fallback = ave::computeProcessingProgress(25, 0);
    check(fallback > 0.0f && fallback < 1.0f,
          "unknown totals should use the logarithmic fallback progress");
}

void testCancelCheck() {
    std::atomic<bool> cancelled{true};
    ave::ProcessVideoOptions opts;
    opts.cancelFlag = &cancelled;
    check(ave::waitForProcessingResumeOrCancel(opts, "test", 3),
          "an active cancel flag should stop processing immediately");
}

}  // namespace

int main() {
    testPreviewFrameLimit();
    testProgressComputation();
    testCancelCheck();
    return 0;
}
