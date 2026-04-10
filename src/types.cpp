#include "ave/types.hpp"

#include <algorithm>
#include <cctype>
#include <unordered_map>

namespace ave {
namespace {

std::string normalize(const std::string& input) {
    std::string out;
    for (char ch : input) {
        if (ch == '-' || ch == ' ' || ch == '.') {
            out.push_back('_');
        } else {
            out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
        }
    }
    return out;
}

}  // namespace

std::string toString(StageKind kind) {
    switch (kind) {
        case StageKind::RestoreCompression:
            return "restore_compression";
        case StageKind::RemoveArtifacts:
            return "remove_artifacts";
        case StageKind::Denoise:
            return "denoise";
        case StageKind::Deblur:
            return "deblur";
        case StageKind::Dehalo:
            return "dehalo";
        case StageKind::ColorFix:
            return "color_fix";
        case StageKind::Upscale:
            return "upscale";
        case StageKind::Sharpen:
            return "sharpen";
        case StageKind::Stereo3D:
            return "stereo_3d";
        case StageKind::Interpolate:
            return "interpolate";
    }
    return "unknown";
}

std::optional<StageKind> stageKindFromString(const std::string& value) {
    static const std::unordered_map<std::string, StageKind> map = {
        {"restore_compression", StageKind::RestoreCompression},
        {"compression_restore", StageKind::RestoreCompression},
        {"decompress", StageKind::RestoreCompression},
        {"deh264", StageKind::RestoreCompression},
        {"restore", StageKind::RestoreCompression},
        {"remove_artifacts", StageKind::RemoveArtifacts},
        {"deartifact", StageKind::RemoveArtifacts},
        {"artifact_removal", StageKind::RemoveArtifacts},
        {"deblock", StageKind::RemoveArtifacts},
        {"cleanup", StageKind::RemoveArtifacts},
        {"denoise", StageKind::Denoise},
        {"deblur", StageKind::Deblur},
        {"dehalo", StageKind::Dehalo},
        {"color_fix", StageKind::ColorFix},
        {"color_correction", StageKind::ColorFix},
        {"upscale", StageKind::Upscale},
        {"sharpen", StageKind::Sharpen},
        {"stereo_3d", StageKind::Stereo3D},
        {"stereo3d", StageKind::Stereo3D},
        {"stereo", StageKind::Stereo3D},
        {"sbs", StageKind::Stereo3D},
        {"3d_sbs", StageKind::Stereo3D},
        {"2d_to_3d", StageKind::Stereo3D},
        {"interpolate", StageKind::Interpolate},
        {"interpolation", StageKind::Interpolate},
        {"frame_interpolation", StageKind::Interpolate},
    };

    const std::string key = normalize(value);
    const auto it = map.find(key);
    if (it == map.end()) {
        return std::nullopt;
    }
    return it->second;
}

}  // namespace ave
