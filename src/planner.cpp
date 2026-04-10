#include "ave/planner.hpp"

#include <algorithm>

namespace ave {
namespace {

int orderGroup(StageKind kind) {
    switch (kind) {
        case StageKind::RestoreCompression:
        case StageKind::RemoveArtifacts:
        case StageKind::Denoise:
        case StageKind::Deblur:
        case StageKind::Dehalo:
            return 0;
        case StageKind::ColorFix:
            return 1;
        case StageKind::Upscale:
            return 2;
        case StageKind::Sharpen:
            return 3;
        case StageKind::Stereo3D:
            return 4;
        case StageKind::Interpolate:
            return 5;
    }
    return 100;
}

}  // namespace

std::vector<EnhancementStage> PipelinePlanner::plan(const std::vector<EnhancementStage>& requested) const {
    std::vector<EnhancementStage> ordered = requested;

    std::stable_sort(ordered.begin(), ordered.end(), [](const EnhancementStage& lhs, const EnhancementStage& rhs) {
        return orderGroup(lhs.kind) < orderGroup(rhs.kind);
    });

    return ordered;
}

}  // namespace ave
