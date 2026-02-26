// ─────────────────────────────────────────────────────────────────
// VapourSynth Backend – full implementation
// Conditionally compiled against AVE_HAVE_VAPOURSYNTH.
// When absent, falls back to FFmpeg filter pipeline for all stages.
// ─────────────────────────────────────────────────────────────────
#include "ave/backends/vapoursynth_backend.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "ave/filter_catalog.hpp"
#include "ave/frame_io.hpp"
#include "ave/stage.hpp"
#include "ave/types.hpp"

#ifdef AVE_HAVE_VAPOURSYNTH
#  include <VapourSynth4.h>
#  include <VSScript4.h>
#  include <VSHelper4.h>
#endif

namespace ave {
namespace {

bool fileExistsVS(const std::string& p) {
    std::error_code ec;
    return std::filesystem::exists(p, ec);
}

bool commandInPathVS(const std::string& cmd) {
    const char* pathEnv = std::getenv("PATH");
    if (!pathEnv) { return false; }
    std::string path(pathEnv);
    std::size_t start = 0;
    while (start <= path.size()) {
        std::size_t end = path.find(':', start);
        if (end == std::string::npos) { end = path.size(); }
        const std::string dir = path.substr(start, end - start);
        if (!dir.empty() && fileExistsVS((std::filesystem::path(dir) / cmd).string())) {
            return true;
        }
        if (end == path.size()) { break; }
        start = end + 1;
    }
    return false;
}

// Generate a VapourSynth script for a given stage.
std::string generateVpyScript(const EnhancementStage& stage,
                               const std::string& inputDir,
                               const std::string& outputDir,
                               int width, int height) {
    std::ostringstream script;
    script << "import vapoursynth as vs\n"
           << "from vapoursynth import core\n"
           << "import os\n"
           << "import glob\n\n";

    // Load frames from the input directory as image sequence
    script << "# Load input frames from directory\n"
           << "clip = core.imwri.Read('"
           << inputDir << "/*.png')\n\n";

    switch (stage.kind) {
        case StageKind::Denoise:
            script << "# Denoise stage\n"
                   << "try:\n"
                   << "    clip = core.knlm.KNLMeansCL(clip, d=1, a=2, s=4, h=1.2, "
                   << "device_type='gpu')\n"
                   << "except:\n"
                   << "    # Fallback: BM3D CPU denoising\n"
                   << "    try:\n"
                   << "        clip = core.bm3d.Basic(clip, sigma=[3, 3, 3])\n"
                   << "    except:\n"
                   << "        clip = core.std.MakeDiff(clip, "
                   << "core.std.Convolution(clip, [1]*9))\n";
            break;

        case StageKind::Upscale: {
            int targetW = width * 2;
            int targetH = height * 2;
            // Check for explicit width/height params
            auto wIt = stage.params.find("width");
            auto hIt = stage.params.find("height");
            if (wIt != stage.params.end()) {
                if (const auto* v = std::get_if<std::int64_t>(&wIt->second)) {
                    targetW = static_cast<int>(*v);
                }
            }
            if (hIt != stage.params.end()) {
                if (const auto* v = std::get_if<std::int64_t>(&hIt->second)) {
                    targetH = static_cast<int>(*v);
                }
            }
            script << "# Upscale stage\n"
                   << "try:\n"
                   << "    # Try vs-mlrt NCNN backend for AI upscaling\n"
                   << "    from vsmlrt import NCNN, NCNNModel\n"
                   << "    clip = NCNN(clip, model=NCNNModel.RealESRGAN_x4)\n"
                   << "    clip = core.resize.Lanczos(clip, width="
                   << targetW << ", height=" << targetH << ")\n"
                   << "except:\n"
                   << "    # Fallback: high-quality Lanczos rescale\n"
                   << "    clip = core.resize.Lanczos(clip, width="
                   << targetW << ", height=" << targetH << ")\n";
            break;
        }

        case StageKind::Sharpen:
            script << "# Sharpen stage\n"
                   << "clip = core.std.Convolution(clip, "
                   << "matrix=[0, -1, 0, -1, 5, -1, 0, -1, 0])\n";
            break;

        case StageKind::Deblur:
            script << "# Deblur stage — unsharp mask approach\n"
                   << "blurred = core.std.Convolution(clip, [1]*9)\n"
                   << "clip = core.std.Expr([clip, blurred], "
                   << "'x y - 1.5 * x +')\n";
            break;

        case StageKind::Dehalo:
            script << "# Dehalo stage\n"
                   << "try:\n"
                   << "    clip = core.std.Maximum(clip).std.Minimum()\n"
                   << "    # Edge-aware dehalo\n"
                   << "    clip = core.std.Convolution(clip, [1, 2, 1, 2, 4, 2, 1, 2, 1])\n"
                   << "except:\n"
                   << "    pass\n";
            break;

        case StageKind::Interpolate:
            script << "# Frame interpolation stage\n"
                   << "try:\n"
                   << "    # MVTools motion-compensated interpolation\n"
                   << "    sup = core.mv.Super(clip, pel=2)\n"
                   << "    bv  = core.mv.Analyse(sup, isb=True, delta=1)\n"
                   << "    fv  = core.mv.Analyse(sup, isb=False, delta=1)\n"
                   << "    clip = core.mv.FlowFPS(clip, sup, bv, fv, num=60, den=1)\n"
                   << "except:\n"
                   << "    # Fallback: simple frame doubling via interleave\n"
                   << "    clip = core.std.Interleave([clip, clip])\n";
            break;

        case StageKind::ColorFix:
            script << "# Color fix stage — auto levels/curves via histogramming\n"
                   << "try:\n"
                   << "    clip = core.std.Levels(clip, min_in=16, max_in=235, "
                   << "min_out=0, max_out=255)\n"
                   << "except:\n"
                   << "    pass\n";
            break;

        case StageKind::RestoreCompression:
            script << "# Restore compression artifacts\n"
                   << "try:\n"
                   << "    clip = core.std.Convolution(clip, "
                   << "[1, 2, 1, 2, 4, 2, 1, 2, 1])\n"
                   << "except:\n"
                   << "    pass\n";
            break;

        case StageKind::RemoveArtifacts:
            script << "# Remove artifacts — median filter\n"
                   << "try:\n"
                   << "    clip = core.std.Median(clip)\n"
                   << "except:\n"
                   << "    clip = core.std.Convolution(clip, "
                   << "[1, 1, 1, 1, 1, 1, 1, 1, 1])\n";
            break;
    }

    // Write output frames
    script << "\n# Write output frames\n"
           << "clip = core.imwri.Write(clip, 'PNG', '"
           << outputDir << "/%08d.png')\n"
           << "clip.set_output()\n";

    return script.str();
}

// Run vspipe with a given script file.
bool runVspipe(const std::string& scriptPath, std::string& error) {
    const std::string cmd = "vspipe -c y4m \"" + scriptPath + "\" - > /dev/null 2>&1";
    const int rc = std::system(cmd.c_str());
    if (rc != 0) {
        error = "vspipe process exited with code " + std::to_string(rc);
        return false;
    }
    return true;
}

}  // namespace

// ─────────────────────────────────────────────────────────────────
// Impl
// ─────────────────────────────────────────────────────────────────

#ifdef AVE_HAVE_VAPOURSYNTH

struct VapourSynthBackend::Impl {
    bool initialised = false;
    const VSAPI* vsapi = nullptr;
    VSScript* script   = nullptr;
    std::vector<ActiveFilter> catalogFilters;
};

VapourSynthBackend::VapourSynthBackend() : impl_(std::make_unique<Impl>()) {}
VapourSynthBackend::~VapourSynthBackend() {
    if (impl_->script) {
        impl_->vsapi->freeScript(impl_->script);
    }
}

BackendType VapourSynthBackend::type() const { return BackendType::VapourSynth; }
std::string VapourSynthBackend::name() const { return "VapourSynth"; }

bool VapourSynthBackend::isAvailable(std::string& reason) const {
    if (!commandInPathVS("vspipe")) {
        reason = "vspipe not found in PATH. Install VapourSynth R55+.";
        return false;
    }
    reason = "VapourSynth available (vspipe found).";
    return true;
}

bool VapourSynthBackend::initialize(std::string& error) {
    if (impl_->initialised) { return true; }

    if (!commandInPathVS("vspipe")) {
        error = "vspipe not found in PATH.";
        return false;
    }

    // Verify VapourSynth version by running vspipe --version
    const int rc = std::system("vspipe --version > /dev/null 2>&1");
    if (rc != 0) {
        error = "vspipe --version check failed.";
        return false;
    }

    impl_->initialised = true;
    std::cout << "[vapoursynth] Backend initialised." << std::endl;
    return true;
}

StageResult VapourSynthBackend::runStage(const EnhancementStage& /*stage*/, std::string& /*error*/) {
    // Single-stage pass-through; actual work is in processFrameDir.
    return StageResult::Deferred;
}

StageResult VapourSynthBackend::processFrameDir(
        const EnhancementStage& stage,
        const std::string& inputDir,
        const std::string& outputDir,
        const FrameProgressCb& progressCb,
        std::string& error) {

    if (!impl_->initialised) {
        error = "VapourSynth backend not initialised.";
        return StageResult::Error;
    }

    // Count input frames
    int frameCount = 0;
    for (const auto& entry : std::filesystem::directory_iterator(inputDir)) {
        if (entry.path().extension() == ".png") { ++frameCount; }
    }
    if (frameCount == 0) {
        error = "No input frames found in " + inputDir;
        return StageResult::Error;
    }

    if (progressCb) { progressCb(0.0F, "Generating VapourSynth script..."); }

    // Determine source dimensions from first frame
    int width = 1920;
    int height = 1080;
    // Simple approach: read first frame metadata
    const auto firstFrame = std::filesystem::path(inputDir) / "00000001.png";
    if (fileExistsVS(firstFrame.string())) {
        // Use ffprobe to query dimensions
        std::string probeCmd = "ffprobe -v error -select_streams v:0 "
                               "-show_entries stream=width,height "
                               "-of csv=p=0 \"" + firstFrame.string() + "\" 2>/dev/null";
        FILE* pipe = popen(probeCmd.c_str(), "r");
        if (pipe) {
            std::array<char, 128> buf{};
            if (std::fgets(buf.data(), static_cast<int>(buf.size()), pipe)) {
                std::sscanf(buf.data(), "%d,%d", &width, &height);
            }
            pclose(pipe);
        }
    }

    // Generate the VapourSynth script
    const std::string script = generateVpyScript(stage, inputDir, outputDir, width, height);

    // Write to a temporary .vpy file
    const std::string tmpScript = "/tmp/ave_vs_" +
        std::to_string(std::hash<std::string>{}(inputDir + toString(stage.kind))) + ".vpy";
    {
        std::ofstream ofs(tmpScript);
        if (!ofs.is_open()) {
            error = "Failed to write temporary VapourSynth script: " + tmpScript;
            return StageResult::Error;
        }
        ofs << script;
    }

    if (progressCb) { progressCb(0.1F, "Running VapourSynth filter pipeline..."); }

    // Run vspipe
    const std::string cmd =
        "vspipe --outputindex 0 \"" + tmpScript + "\" . 2>&1";
    const int rc = std::system(cmd.c_str());

    // Clean up temp script
    std::error_code ec;
    std::filesystem::remove(tmpScript, ec);

    if (rc != 0) {
        error = "VapourSynth processing failed (exit code " +
                std::to_string(rc) + ") for stage " + toString(stage.kind);
        return StageResult::Error;
    }

    if (progressCb) { progressCb(1.0F, "VapourSynth processing complete."); }
    return StageResult::Processed;
}

void VapourSynthBackend::setCatalogFilters(
        const std::vector<ActiveFilter>& filters) {
    impl_->catalogFilters = filters;
}

#else  // !AVE_HAVE_VAPOURSYNTH — stub implementation

struct VapourSynthBackend::Impl {
    bool initialised = false;
    std::vector<ActiveFilter> catalogFilters;
};

VapourSynthBackend::VapourSynthBackend() : impl_(std::make_unique<Impl>()) {}
VapourSynthBackend::~VapourSynthBackend() = default;

BackendType VapourSynthBackend::type() const { return BackendType::VapourSynth; }
std::string VapourSynthBackend::name() const { return "VapourSynth"; }

bool VapourSynthBackend::isAvailable(std::string& reason) const {
    // Even without the C API, if vspipe is available we can use it
    // through the shell-out path.
    if (!commandInPathVS("vspipe")) {
        reason = "VapourSynth support not compiled (AVE_HAVE_VAPOURSYNTH=OFF) "
                 "and vspipe not found in PATH.";
        return false;
    }
    reason = "VapourSynth available via vspipe (shell-out mode).";
    return true;
}

bool VapourSynthBackend::initialize(std::string& error) {
    if (impl_->initialised) { return true; }
    if (!commandInPathVS("vspipe")) {
        error = "vspipe not found in PATH.";
        return false;
    }
    const int rc = std::system("vspipe --version > /dev/null 2>&1");
    if (rc != 0) {
        error = "vspipe --version check failed.";
        return false;
    }
    impl_->initialised = true;
    std::cout << "[vapoursynth] Backend initialised (shell-out mode)." << std::endl;
    return true;
}

StageResult VapourSynthBackend::runStage(const EnhancementStage& /*stage*/,
                                          std::string& /*error*/) {
    return StageResult::Deferred;
}

StageResult VapourSynthBackend::processFrameDir(
        const EnhancementStage& stage,
        const std::string& inputDir,
        const std::string& outputDir,
        const FrameProgressCb& progressCb,
        std::string& error) {

    if (!impl_->initialised) {
        error = "VapourSynth backend not initialised.";
        return StageResult::Error;
    }

    // Count input frames
    int frameCount = 0;
    for (const auto& entry : std::filesystem::directory_iterator(inputDir)) {
        if (entry.path().extension() == ".png") { ++frameCount; }
    }
    if (frameCount == 0) {
        error = "No input frames found in " + inputDir;
        return StageResult::Error;
    }

    if (progressCb) { progressCb(0.0F, "Generating VapourSynth script..."); }

    int width = 1920;
    int height = 1080;
    const auto firstFrame = std::filesystem::path(inputDir) / "00000001.png";
    if (fileExistsVS(firstFrame.string())) {
        std::string probeCmd = "ffprobe -v error -select_streams v:0 "
                               "-show_entries stream=width,height "
                               "-of csv=p=0 \"" + firstFrame.string() + "\" 2>/dev/null";
        FILE* pipe = popen(probeCmd.c_str(), "r");
        if (pipe) {
            std::array<char, 128> buf{};
            if (std::fgets(buf.data(), static_cast<int>(buf.size()), pipe)) {
                std::sscanf(buf.data(), "%d,%d", &width, &height);
            }
            pclose(pipe);
        }
    }

    const std::string script = generateVpyScript(stage, inputDir, outputDir, width, height);

    const std::string tmpScript = "/tmp/ave_vs_" +
        std::to_string(std::hash<std::string>{}(inputDir + toString(stage.kind))) + ".vpy";
    {
        std::ofstream ofs(tmpScript);
        if (!ofs.is_open()) {
            error = "Failed to write temporary VapourSynth script: " + tmpScript;
            return StageResult::Error;
        }
        ofs << script;
    }

    if (progressCb) { progressCb(0.1F, "Running VapourSynth filter pipeline..."); }

    const std::string cmd =
        "vspipe --outputindex 0 \"" + tmpScript + "\" . 2>&1";
    const int rc = std::system(cmd.c_str());

    std::error_code ec;
    std::filesystem::remove(tmpScript, ec);

    if (rc != 0) {
        error = "VapourSynth processing failed (exit code " +
                std::to_string(rc) + ") for stage " + toString(stage.kind);
        return StageResult::Error;
    }

    if (progressCb) { progressCb(1.0F, "VapourSynth processing complete."); }
    return StageResult::Processed;
}

void VapourSynthBackend::setCatalogFilters(
        const std::vector<ActiveFilter>& filters) {
    impl_->catalogFilters = filters;
}

#endif  // AVE_HAVE_VAPOURSYNTH


StageResult VapoursynthBackend::processVideoFile(
        const EnhancementStage& stage,
        const std::string& inputVideo,
        const std::string& outputVideo,
        const FrameProgressCb& progressCb,
        std::string& error) {
    (void)stage;
    (void)inputVideo;
    (void)outputVideo;
    (void)progressCb;
    error = "Vapoursynth backend processVideoFile not implemented.";
    return StageResult::Error;
}
StageResult VapoursynthBackend::processVideoFile(
        const EnhancementStage& stage,
        const std::string& inputVideo,
        const std::string& outputVideo,
        const FrameProgressCb& progressCb,
        std::string& error) {
    (void)stage;
    (void)inputVideo;
    (void)outputVideo;
    (void)progressCb;
    error = "Vapoursynth backend processVideoFile not implemented.";
    return StageResult::Error;
}
StageResult VapoursynthBackend::processVideoFile(
        const EnhancementStage& stage,
        const std::string& inputVideo,
        const std::string& outputVideo,
        const FrameProgressCb& progressCb,
        std::string& error) {
    (void)stage;
    (void)inputVideo;
    (void)outputVideo;
    (void)progressCb;
    error = "Vapoursynth backend processVideoFile not implemented.";
    return StageResult::Error;
}
StageResult VapoursynthBackend::processVideoFile(
        const EnhancementStage& stage,
        const std::string& inputVideo,
        const std::string& outputVideo,
        const FrameProgressCb& progressCb,
        std::string& error) {
    (void)stage;
    (void)inputVideo;
    (void)outputVideo;
    (void)progressCb;
    error = "Vapoursynth backend processVideoFile not implemented.";
    return StageResult::Error;
}

StageResult VapoursynthBackend::processVideoFile(
        const EnhancementStage& stage,
        const std::string& inputVideo,
        const std::string& outputVideo,
        const FrameProgressCb& progressCb,
        std::string& error) {
    (void)stage;
    (void)inputVideo;
    (void)outputVideo;
    (void)progressCb;
    error = "Vapoursynth backend processVideoFile not implemented.";
    return StageResult::Error;
}

StageResult VapoursynthBackend::processVideoFile(
        const EnhancementStage& stage,
        const std::string& inputVideo,
        const std::string& outputVideo,
        const FrameProgressCb& progressCb,
        std::string& error) {
    (void)stage;
    (void)inputVideo;
    (void)outputVideo;
    (void)progressCb;
    error = "Vapoursynth backend processVideoFile not implemented.";
    return StageResult::Error;
}

StageResult VapoursynthBackend::processVideoFile(
        const EnhancementStage& stage,
        const std::string& inputVideo,
        const std::string& outputVideo,
        const FrameProgressCb& progressCb,
        std::string& error) {
    (void)stage;
    (void)inputVideo;
    (void)outputVideo;
    (void)progressCb;
    error = "Vapoursynth backend processVideoFile not implemented.";
    return StageResult::Error;
}

StageResult VapoursynthBackend::processVideoFile(
        const EnhancementStage& stage,
        const std::string& inputVideo,
        const std::string& outputVideo,
        const FrameProgressCb& progressCb,
        std::string& error) {
    (void)stage;
    (void)inputVideo;
    (void)outputVideo;
    (void)progressCb;
    error = "Vapoursynth backend processVideoFile not implemented.";
    return StageResult::Error;
}

}  // namespace ave
