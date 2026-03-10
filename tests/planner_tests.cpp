#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "ave/planner.hpp"
#include "ave/types.hpp"

namespace {

using ave::EnhancementStage;
using ave::PipelinePlanner;
using ave::StageKind;

void check(bool condition, const char* message) {
    if (condition) {
        return;
    }
    std::cerr << "planner_tests failed: " << message << '\n';
    std::abort();
}

EnhancementStage stage(StageKind kind) {
    EnhancementStage s;
    s.kind = kind;
    return s;
}

int indexOf(const std::vector<EnhancementStage>& v, StageKind kind) {
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (v[i].kind == kind) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

void testInterpolationLast() {
    PipelinePlanner planner;
    const auto ordered = planner.plan({stage(StageKind::Interpolate), stage(StageKind::Sharpen), stage(StageKind::Upscale)});

    check(!ordered.empty(), "planned pipeline should not be empty");
    check(ordered.back().kind == StageKind::Interpolate, "interpolate must execute last");
}

void testRestoreBeforeUpscaleAndSharpen() {
    PipelinePlanner planner;
    const auto ordered = planner.plan(
        {stage(StageKind::Sharpen), stage(StageKind::RestoreCompression), stage(StageKind::RemoveArtifacts), stage(StageKind::Upscale)});

    const int restoreIdx = indexOf(ordered, StageKind::RestoreCompression);
    const int removeIdx = indexOf(ordered, StageKind::RemoveArtifacts);
    const int upscaleIdx = indexOf(ordered, StageKind::Upscale);
    const int sharpenIdx = indexOf(ordered, StageKind::Sharpen);

    check(restoreIdx != -1, "restore stage missing");
    check(removeIdx != -1, "remove-artifacts stage missing");
    check(upscaleIdx != -1, "upscale stage missing");
    check(sharpenIdx != -1, "sharpen stage missing");

    check(restoreIdx < upscaleIdx, "restore must come before upscale");
    check(removeIdx < upscaleIdx, "remove-artifacts must come before upscale");
    check(restoreIdx < sharpenIdx, "restore must come before sharpen");
    check(removeIdx < sharpenIdx, "remove-artifacts must come before sharpen");
}

void testStableWithinCategory() {
    PipelinePlanner planner;
    const auto ordered = planner.plan(
        {stage(StageKind::RemoveArtifacts), stage(StageKind::Denoise), stage(StageKind::Deblur), stage(StageKind::Sharpen)});

    check(ordered.size() == 4, "planner should preserve all stages");
    check(ordered[0].kind == StageKind::RemoveArtifacts, "remove-artifacts order changed");
    check(ordered[1].kind == StageKind::Denoise, "denoise order changed");
    check(ordered[2].kind == StageKind::Deblur, "deblur order changed");
}

}  // namespace

int main() {
    testInterpolationLast();
    testRestoreBeforeUpscaleAndSharpen();
    testStableWithinCategory();
    return 0;
}
