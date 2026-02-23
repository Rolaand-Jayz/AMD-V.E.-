#include <cassert>
#include <vector>

#include "ave/planner.hpp"
#include "ave/types.hpp"

namespace {

using ave::EnhancementStage;
using ave::PipelinePlanner;
using ave::StageKind;

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

    assert(!ordered.empty());
    assert(ordered.back().kind == StageKind::Interpolate);
}

void testRestoreBeforeUpscaleAndSharpen() {
    PipelinePlanner planner;
    const auto ordered = planner.plan(
        {stage(StageKind::Sharpen), stage(StageKind::RestoreCompression), stage(StageKind::RemoveArtifacts), stage(StageKind::Upscale)});

    const int restoreIdx = indexOf(ordered, StageKind::RestoreCompression);
    const int removeIdx = indexOf(ordered, StageKind::RemoveArtifacts);
    const int upscaleIdx = indexOf(ordered, StageKind::Upscale);
    const int sharpenIdx = indexOf(ordered, StageKind::Sharpen);

    assert(restoreIdx != -1);
    assert(removeIdx != -1);
    assert(upscaleIdx != -1);
    assert(sharpenIdx != -1);

    assert(restoreIdx < upscaleIdx);
    assert(removeIdx < upscaleIdx);
    assert(restoreIdx < sharpenIdx);
    assert(removeIdx < sharpenIdx);
}

void testStableWithinCategory() {
    PipelinePlanner planner;
    const auto ordered = planner.plan(
        {stage(StageKind::RemoveArtifacts), stage(StageKind::Denoise), stage(StageKind::Deblur), stage(StageKind::Sharpen)});

    assert(ordered.size() == 4);
    assert(ordered[0].kind == StageKind::RemoveArtifacts);
    assert(ordered[1].kind == StageKind::Denoise);
    assert(ordered[2].kind == StageKind::Deblur);
}

}  // namespace

int main() {
    testInterpolationLast();
    testRestoreBeforeUpscaleAndSharpen();
    testStableWithinCategory();
    return 0;
}
