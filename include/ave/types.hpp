#pragma once

#include <optional>
#include <string>

namespace ave {

enum class StageKind {
    RestoreCompression,
    RemoveArtifacts,
    Denoise,
    Deblur,
    Dehalo,
    ColorFix,
    Upscale,
    Sharpen,
    Stereo3D,
    Interpolate
};

std::string toString(StageKind kind);
std::optional<StageKind> stageKindFromString(const std::string& value);

}  // namespace ave
