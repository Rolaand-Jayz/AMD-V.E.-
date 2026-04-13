#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "ave/model_catalog.hpp"
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
    const auto ordered = planner.plan({
        stage(StageKind::Interpolate),
        stage(StageKind::Stereo3D),
        stage(StageKind::Sharpen),
        stage(StageKind::Upscale)});

    check(!ordered.empty(), "planned pipeline should not be empty");
    check(ordered.back().kind == StageKind::Interpolate, "interpolate must execute last");
    check(ordered[ordered.size() - 2].kind == StageKind::Stereo3D,
          "stereo_3d must execute immediately before interpolate");
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

void testCatalogIncludesRequestedUpscaleModels() {
    bool sawOpenProteus = false;
    bool sawNcnnGeneral = false;
    int ncnnUpscalerCount = 0;

    for (const auto& entry : ave::builtinModelCatalog()) {
        if (entry.id == "openproteus-compact-x2" &&
            entry.stage == StageKind::Upscale &&
            entry.sourceFormat == ave::ModelFormat::Onnx) {
            sawOpenProteus = true;
        }
        if (entry.id == "realesrgan-x4plus-ncnn" &&
            entry.stage == StageKind::Upscale &&
            entry.sourceFormat == ave::ModelFormat::NcnnBin) {
            sawNcnnGeneral = true;
        }
        if (entry.stage == StageKind::Upscale &&
            entry.sourceFormat == ave::ModelFormat::NcnnBin) {
            ++ncnnUpscalerCount;
            check(!entry.filenameAux.empty(), "NCNN catalog entries must define a .bin filename");
            check(!entry.archiveSubPath.empty(), "NCNN catalog entries must define a zip archive param path");
            check(!entry.archiveSubPathAux.empty(), "NCNN catalog entries must define a zip archive bin path");
        }
    }

    check(sawOpenProteus, "OpenProteus upscale model missing from catalog");
    check(sawNcnnGeneral, "Real-ESRGAN NCNN upscale model missing from catalog");
    check(ncnnUpscalerCount >= 5, "expected multiple NCNN upscale models in catalog");
}

void testBackendPreferredModelSelection() {
    const auto* ncnnPreferred =
        ave::preferredBackendModelForStage(StageKind::Upscale, ave::BackendType::NcnnVulkan);
    check(ncnnPreferred != nullptr, "NCNN should expose a preferred upscale model");
    check(ncnnPreferred->id == "realesrgan-x4plus-ncnn",
          "NCNN should prefer the general non-animation upscale model");
    check(ncnnPreferred->sourceFormat == ave::ModelFormat::NcnnBin,
          "NCNN preferred model must be an NCNN model");
    check(!ave::modelLooksAnimationFocused(*ncnnPreferred),
          "NCNN preferred general model should not be animation-focused");

    const auto* migraphxPreferred =
        ave::preferredBackendModelForStage(StageKind::Upscale, ave::BackendType::MiGraphX);
    check(migraphxPreferred != nullptr, "MiGraphX should expose a preferred upscale model");
    check(migraphxPreferred->id == "realesrgan-x4-general",
          "MiGraphX should retain the general ONNX upscale default");

    const auto* rocmHipPreferred =
        ave::preferredBackendModelForStage(StageKind::Upscale, ave::BackendType::RocmHip);
    check(rocmHipPreferred != nullptr, "ROCm/HIP should expose a preferred upscale model");
    check(rocmHipPreferred->id == "realesrgan-x4-general",
          "ROCm/HIP should prefer the general ONNX upscale default");
    check(rocmHipPreferred->sourceFormat == ave::ModelFormat::Onnx,
          "ROCm/HIP preferred model must be an ONNX model");

    const auto* ncnnDenoise =
        ave::preferredBackendModelForStage(StageKind::Denoise, ave::BackendType::NcnnVulkan);
    check(ncnnDenoise == nullptr,
          "NCNN should not advertise a default denoise model when the catalog has no NCNN entry");
}

void testCatalogIncludesStereoDepthModels() {
    bool sawDaV2Small = false;
    bool sawDaV2Base = false;
    bool sawDaV2Large = false;
    bool sawDistillSmall = false;
    bool sawDepthAnythingFp32Variant = false;

    int stereoModelCount = 0;
    for (const auto& entry : ave::builtinModelCatalog()) {
        if (entry.stage != StageKind::Stereo3D) {
            continue;
        }
        ++stereoModelCount;
        check(entry.sourceFormat == ave::ModelFormat::Onnx,
              "stereo_3d catalog entries should be ONNX");
        check(!entry.downloadUrl.empty(),
              "stereo_3d catalog entries must provide a download URL");
        if (entry.id == "depth-anything-v2-small-fp16") {
            sawDaV2Small = true;
        } else if (entry.id == "depth-anything-v2-base-fp16") {
            sawDaV2Base = true;
        } else if (entry.id == "depth-anything-v2-large-fp16") {
            sawDaV2Large = true;
        } else if (entry.id == "distill-any-depth-small") {
            sawDistillSmall = true;
        } else if (entry.id.find("depth-anything-v2-") == 0 &&
                   entry.id.find("-fp32") != std::string::npos) {
            sawDepthAnythingFp32Variant = true;
        }
    }

    check(sawDaV2Small, "Depth Anything V2 Small fp16 stereo model missing");
    check(sawDaV2Base, "Depth Anything V2 Base fp16 stereo model missing");
    check(sawDaV2Large, "Depth Anything V2 Large fp16 stereo model missing");
    check(sawDistillSmall, "Distill Any Depth Small stereo model missing");
    check(!sawDepthAnythingFp32Variant,
          "Depth Anything fp32 stereo variants should not be exposed in the catalog");
    check(stereoModelCount >= 6, "expected multiple stereo depth models in catalog");
}

void testModelFamilyCapabilities() {
    const auto* clearReality = ave::catalogEntryById("clearreality-x4-denoise");
    check(clearReality != nullptr, "clearreality catalog entry missing");
    const auto clearRealityCaps = ave::modelCapabilities(*clearReality);
    check(clearRealityCaps.size() >= 3, "clearreality family should expose multiple capabilities");
    check(ave::modelCanFuseRequestedCapabilities(
              *clearReality,
              {StageKind::Denoise, StageKind::Deblur, StageKind::Upscale}),
          "clearreality family should be fusible for its full declared capability set");
    check(!ave::modelCanFuseRequestedCapabilities(
               *clearReality,
               {StageKind::Denoise, StageKind::Upscale}),
          "non-selective families should not fuse partial capability subsets");

    ave::ModelEntry customSelective = *clearReality;
    customSelective.supportsSelectiveCapabilities = true;
    check(ave::modelCanFuseRequestedCapabilities(
              customSelective,
              {StageKind::Denoise, StageKind::Upscale}),
          "selective families should fuse partial capability subsets");
}

void testScopedStageParamHelpers() {
    EnhancementStage fused;
    fused.kind = StageKind::Upscale;
    fused.params["width"] = std::int64_t{3840};
    fused.params[ave::scopedStageParamKey(StageKind::Denoise, "strength")] = 0.85;
    fused.params[ave::scopedStageParamKey(StageKind::ColorFix, "gamma")] = 1.1;
    fused.params[ave::scopedStageParamKey(StageKind::Stereo3D, "depth_aa")] = std::string("true");
    fused.params[ave::scopedStageParamKey(StageKind::Stereo3D, "format")] = std::string("full_sbs");

    double strength = 0.0;
    check(ave::tryGetDouble(fused, StageKind::Denoise, "strength", strength),
          "scoped fused strength should be readable");
    check(strength > 0.84 && strength < 0.86, "scoped fused strength value mismatch");

    double width = 0.0;
    check(ave::tryGetDouble(fused, StageKind::Upscale, "width", width),
          "plain primary width should still be readable through scoped accessor");
    check(width == 3840.0, "plain primary width mismatch");

    bool depthAa = false;
    check(ave::tryGetBool(fused, StageKind::Stereo3D, "depth_aa", depthAa),
          "scoped fused bool should be readable");
    check(depthAa, "scoped fused bool mismatch");

    std::string format;
    check(ave::tryGetString(fused, StageKind::Stereo3D, "format", format),
          "scoped fused string should be readable");
    check(format == "full_sbs", "scoped fused string mismatch");

    std::string widthString;
    check(ave::tryGetString(fused, StageKind::Upscale, "width", widthString),
          "numeric values should stringify through scoped accessor");
    check(widthString == "3840", "scoped fused numeric string mismatch");
}

}  // namespace

int main() {
    testInterpolationLast();
    testRestoreBeforeUpscaleAndSharpen();
    testStableWithinCategory();
    testCatalogIncludesRequestedUpscaleModels();
    testBackendPreferredModelSelection();
    testCatalogIncludesStereoDepthModels();
    testModelFamilyCapabilities();
    testScopedStageParamHelpers();
    return 0;
}
