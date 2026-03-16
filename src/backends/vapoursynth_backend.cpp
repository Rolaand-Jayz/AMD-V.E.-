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

std::string quoteArgVS(const std::string& value) {
    std::string out = "\"";
    for (const char ch : value) {
        if (ch == '\\' || ch == '"') {
            out.push_back('\\');
        }
        out.push_back(ch);
    }
    out.push_back('"');
    return out;
}

std::string quotePyString(const std::string& value) {
    std::string out = "'";
    for (const char ch : value) {
        if (ch == '\\' || ch == '\'') {
            out.push_back('\\');
        }
        out.push_back(ch);
    }
    out.push_back('\'');
    return out;
}

struct VideoInfoVS {
    int width = 0;
    int height = 0;
    double fps = 30.0;
};

bool runProbeScriptVS(const std::string& scriptName,
                      const std::string& scriptBody,
                      const std::string& successReason,
                      std::string& reason) {
    const std::string probePath = "/tmp/ave_vs_probe_" +
        std::to_string(static_cast<unsigned long long>(std::hash<std::string>{}(scriptName))) +
        ".vpy";
    {
        std::ofstream probe(probePath);
        if (!probe.is_open()) {
            reason = "Failed to write temporary VapourSynth probe script.";
            return false;
        }
        probe << scriptBody;
    }

    const std::string probeCmd =
        "vspipe --outputindex 0 " + quoteArgVS(probePath) + " - > /dev/null 2>&1";
    const int rc = std::system(probeCmd.c_str());
    std::error_code ec;
    std::filesystem::remove(probePath, ec);
    if (rc != 0) {
        return false;
    }

    reason = successReason;
    return true;
}

bool vapoursynthDirectSourceAvailable(std::string& reason) {
    const std::string script =
        "import vapoursynth as vs\n"
        "core = vs.core\n"
        "if not (hasattr(core, 'ffms2') or hasattr(core, 'lsmas') or hasattr(core, 'bs')):\n"
        "    raise RuntimeError('missing direct video source plugin')\n"
        "clip = core.std.BlankClip(width=8, height=8, format=vs.YUV420P8, length=1)\n"
        "clip.set_output()\n";
    return runProbeScriptVS(
        "vspipe-direct-source-probe",
        script,
        "VapourSynth available via vspipe with direct video source plugins.",
        reason);
}

bool vapoursynthImageSequenceAvailable(std::string& reason) {
    const std::string script =
        "import vapoursynth as vs\n"
        "core = vs.core\n"
        "if not hasattr(core, 'imwri'):\n"
        "    raise RuntimeError('missing imwri plugin')\n"
        "clip = core.std.BlankClip(width=8, height=8, format=vs.RGB24, length=1)\n"
        "clip.set_output()\n";
    return runProbeScriptVS(
        "vspipe-imwri-probe",
        script,
        "VapourSynth available via vspipe with image-sequence fallback support.",
        reason);
}

bool vapoursynthScriptPrereqsAvailable(std::string& reason) {
    if (!commandInPathVS("vspipe")) {
        reason = "vspipe not found in PATH. Install VapourSynth R55+.";
        return false;
    }

    std::string directReason;
    if (vapoursynthDirectSourceAvailable(directReason)) {
        reason = directReason;
        return true;
    }

    std::string imageReason;
    if (vapoursynthImageSequenceAvailable(imageReason)) {
        reason = imageReason;
        return true;
    }

    reason = "vspipe is present but required script plugins are missing "
             "(need ffms2/lsmas/bestsource for direct video input or imwri for fallback).";
    return false;
}

bool probeInputVideo(const std::string& inputVideo,
                     VideoInfoVS& info,
                     std::string& error) {
    const std::string probeCmd =
        "ffprobe -v error -select_streams v:0 "
        "-show_entries stream=width,height,r_frame_rate "
        "-of csv=p=0 " + quoteArgVS(inputVideo) + " 2>/dev/null";
    FILE* pipe = popen(probeCmd.c_str(), "r");
    if (pipe == nullptr) {
        error = "Failed to launch ffprobe for " + inputVideo;
        return false;
    }

    std::array<char, 256> buf{};
    if (std::fgets(buf.data(), static_cast<int>(buf.size()), pipe) != nullptr) {
        int fpsNum = 0;
        int fpsDen = 1;
        std::sscanf(buf.data(), "%d,%d,%d/%d", &info.width, &info.height, &fpsNum, &fpsDen);
        if (fpsDen > 0) {
            info.fps = static_cast<double>(fpsNum) / static_cast<double>(fpsDen);
        }
    }
    pclose(pipe);

    if (info.width <= 0 || info.height <= 0) {
        error = "Failed to probe input video dimensions: " + inputVideo;
        return false;
    }
    return true;
}

std::string trimVS(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::string stripVsOutputCalls(const std::string& source) {
    std::istringstream input(source);
    std::ostringstream output;
    std::string line;
    while (std::getline(input, line)) {
        const std::string trimmed = trimVS(line);
        if (trimmed.find(".set_output(") != std::string::npos) {
            continue;
        }
        output << line << '\n';
    }
    return output.str();
}

void appendCatalogVsFilters(std::ostringstream& script,
                            const StageKind stageKind,
                            const std::vector<ActiveFilter>& catalogFilters) {
    for (const auto& active : catalogFilters) {
        if (!active.enabled) {
            continue;
        }
        const EmbeddedFilter* filter = findFilter(active.id);
        if (filter == nullptr ||
            filter->runtime != FilterRuntime::VapourSynth ||
            filter->stageKind != stageKind) {
            continue;
        }

        script << "\n# Catalog filter: " << filter->name << '\n'
               << "video = clip\n"
               << stripVsOutputCalls(resolveSource(*filter, active.paramValues));
    }
}

void appendStageScript(std::ostringstream& script,
                       const EnhancementStage& stage,
                       int width,
                       int height) {
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
}

void appendEncodeArgs(std::ostringstream& cmd,
                      const std::string& outputVideo,
                      const std::string& audioInputVideo,
                      const ProcessVideoOptions& opts) {
    cmd << "-i " << quoteArgVS(audioInputVideo) << ' '
        << "-map 0:v:0 -map 1:a? ";
    if (opts.directOutputEncode) {
        cmd << "-c:v " << opts.outputCodec << ' '
            << "-crf " << opts.outputCrf << ' '
            << "-preset " << opts.outputPreset << ' ';
        if (!opts.outputProfile.empty()) {
            cmd << "-profile:v " << opts.outputProfile << ' ';
        }
        if (opts.outputThreads > 0) {
            cmd << "-threads " << opts.outputThreads << ' ';
        }
    } else {
        cmd << "-c:v libx264 -crf 0 -preset ultrafast ";
    }
    cmd << "-c:a copy -shortest " << quoteArgVS(outputVideo);
}

// Generate a VapourSynth script for a given stage.
std::string generateFrameSequenceVpyScript(const EnhancementStage& stage,
                                           const std::string& inputDir,
                                           const std::string& outputDir,
                                           int width,
                                           int height,
                                           const std::vector<ActiveFilter>& catalogFilters) {
    std::ostringstream script;
    script << "import vapoursynth as vs\n"
           << "from vapoursynth import core\n"
           << "import os\n"
           << "import glob\n\n";

    script << "# Load input frames from directory\n"
           << "clip = core.imwri.Read("
           << quotePyString(inputDir + "/*.png") << ")\n\n";

    appendStageScript(script, stage, width, height);

    appendCatalogVsFilters(script, stage.kind, catalogFilters);

    script << "\n# Write output frames\n"
           << "clip = core.imwri.Write(clip, 'PNG', "
           << quotePyString(outputDir + "/%08d.png") << ")\n"
           << "clip.set_output()\n";

    return script.str();
}

std::string generateDirectVideoVpyScript(const EnhancementStage& stage,
                                         const std::string& inputVideo,
                                         int width,
                                         int height,
                                         int previewFrames,
                                         const std::vector<ActiveFilter>& catalogFilters) {
    std::ostringstream script;
    script << "import vapoursynth as vs\n"
           << "from vapoursynth import core\n\n"
           << "def load_source(path):\n"
           << "    if hasattr(core, 'ffms2'):\n"
           << "        try:\n"
           << "            return core.ffms2.Source(source=path)\n"
           << "        except Exception:\n"
           << "            pass\n"
           << "    if hasattr(core, 'lsmas'):\n"
           << "        try:\n"
           << "            return core.lsmas.LWLibavSource(source=path)\n"
           << "        except Exception:\n"
           << "            pass\n"
           << "    if hasattr(core, 'bs'):\n"
           << "        try:\n"
           << "            return core.bs.VideoSource(source=path)\n"
           << "        except Exception:\n"
           << "            pass\n"
           << "    raise RuntimeError('No direct video source plugin available')\n\n"
           << "clip = load_source(" << quotePyString(inputVideo) << ")\n"
           << "try:\n"
           << "    clip = core.resize.Bicubic(clip, format=vs.RGB24, matrix_in_s='709')\n"
           << "except Exception:\n"
           << "    clip = core.resize.Bicubic(clip, format=vs.RGB24)\n";

    if (previewFrames > 0) {
        script << "clip = core.std.Trim(clip, first=0, last=" << (previewFrames - 1) << ")\n";
    }

    script << '\n';
    appendStageScript(script, stage, width, height);
    appendCatalogVsFilters(script, stage.kind, catalogFilters);

    script << "\n# Convert to a y4m-safe delivery format for the ffmpeg pipe\n"
           << "try:\n"
           << "    clip = core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s='709')\n"
           << "except Exception:\n"
           << "    clip = core.resize.Bicubic(clip, format=vs.YUV420P8)\n"
           << "clip.set_output()\n";
    return script.str();
}

StageResult processVideoFileViaFrameSequence(
        const EnhancementStage& stage,
        const std::string& inputVideo,
        const std::string& outputVideo,
        const std::vector<ActiveFilter>& catalogFilters,
        const FrameProgressCb& progressCb,
        std::string& error,
        const ProcessVideoOptions& opts) {
    VideoInfoVS info;
    if (!probeInputVideo(inputVideo, info, error)) {
        return StageResult::Error;
    }

    // Create temp directories for frame I/O.
    const std::string tmpBase = "/tmp/ave_vs_" +
        std::to_string(std::hash<std::string>{}(inputVideo + toString(stage.kind) + outputVideo));
    const std::string tmpIn  = tmpBase + "_in";
    const std::string tmpOut = tmpBase + "_out";
    std::error_code ec;
    std::filesystem::create_directories(tmpIn, ec);
    std::filesystem::create_directories(tmpOut, ec);

    if (progressCb) { progressCb(0.0F, "Extracting frames for VapourSynth..."); }

    // Extract frames from input video.
    std::ostringstream extractOss;
    extractOss << "ffmpeg -hide_banner -loglevel error -y ";
    if (opts.previewDurationSec > 0.0) {
        extractOss << "-t " << opts.previewDurationSec << " ";
    }
    extractOss << "-i \"" << inputVideo << "\" "
               << "\"" << tmpIn << "/%08d.png\" 2>&1";
    int rc = std::system(extractOss.str().c_str());
    if (rc != 0) {
        error = "Frame extraction failed (exit code " + std::to_string(rc) + ")";
        std::filesystem::remove_all(tmpIn, ec);
        std::filesystem::remove_all(tmpOut, ec);
        return StageResult::Error;
    }

    if (progressCb) { progressCb(0.15F, "Running VapourSynth pipeline..."); }

    // Generate VPY script and write to temp file.
    const std::string script = generateFrameSequenceVpyScript(
        stage, tmpIn, tmpOut, info.width, info.height, catalogFilters);
    const std::string tmpScript = tmpBase + ".vpy";
    {
        std::ofstream ofs(tmpScript);
        if (!ofs.is_open()) {
            error = "Failed to write temp VPY script: " + tmpScript;
            std::filesystem::remove_all(tmpIn, ec);
            std::filesystem::remove_all(tmpOut, ec);
            return StageResult::Error;
        }
        ofs << script;
    }

    // Run vspipe.
    const std::string vsCmd = "vspipe --outputindex 0 \"" + tmpScript + "\" . 2>&1";
    rc = std::system(vsCmd.c_str());
    std::filesystem::remove(tmpScript, ec);

    if (rc != 0) {
        error = "VapourSynth processing failed (exit code " +
                std::to_string(rc) + ") for " + toString(stage.kind);
        std::filesystem::remove_all(tmpIn, ec);
        std::filesystem::remove_all(tmpOut, ec);
        return StageResult::Error;
    }

    if (progressCb) { progressCb(0.8F, "Re-encoding output video..."); }

    std::ostringstream encOss;
    encOss << "ffmpeg -hide_banner -loglevel error -y "
           << "-framerate " << info.fps << ' '
           << "-i " << quoteArgVS(tmpOut + "/%08d.png") << ' ';
    appendEncodeArgs(encOss, outputVideo, inputVideo, opts);
    encOss << " 2>&1";
    rc = std::system(encOss.str().c_str());

    // Cleanup temp directories.
    std::filesystem::remove_all(tmpIn, ec);
    std::filesystem::remove_all(tmpOut, ec);

    if (rc != 0) {
        error = "Re-encoding output frames failed (exit code " + std::to_string(rc) + ")";
        return StageResult::Error;
    }

    if (progressCb) { progressCb(1.0F, "VapourSynth processing complete."); }
    std::cout << "[vapoursynth] Done: " << toString(stage.kind)
              << " for " << inputVideo << std::endl;
    return StageResult::Processed;
}

StageResult processVideoFileViaStreaming(
        const EnhancementStage& stage,
        const std::string& inputVideo,
        const std::string& outputVideo,
        const std::vector<ActiveFilter>& catalogFilters,
        const FrameProgressCb& progressCb,
        std::string& error,
        const ProcessVideoOptions& opts) {
    VideoInfoVS info;
    if (!probeInputVideo(inputVideo, info, error)) {
        return StageResult::Error;
    }

    int previewFrames = 0;
    if (opts.previewDurationSec > 0.0) {
        previewFrames = std::max(1, static_cast<int>(opts.previewDurationSec * info.fps + 0.5));
    }

    const std::string tmpBase = "/tmp/ave_vs_stream_" +
        std::to_string(std::hash<std::string>{}(inputVideo + toString(stage.kind) + outputVideo));
    const std::string tmpScript = tmpBase + ".vpy";
    {
        std::ofstream ofs(tmpScript);
        if (!ofs.is_open()) {
            error = "Failed to write temp VPY script: " + tmpScript;
            return StageResult::Error;
        }
        ofs << generateDirectVideoVpyScript(
            stage, inputVideo, info.width, info.height, previewFrames, catalogFilters);
    }

    if (progressCb) {
        progressCb(0.05F, "Running VapourSynth streaming pipeline...");
    }

    std::ostringstream vspipeCmd;
    vspipeCmd << "vspipe --outputindex 0 -c y4m " << quoteArgVS(tmpScript) << " -";

    std::ostringstream ffmpegCmd;
    ffmpegCmd << "ffmpeg -hide_banner -loglevel error -y "
              << "-f yuv4mpegpipe -i pipe:0 ";
    appendEncodeArgs(ffmpegCmd, outputVideo, inputVideo, opts);

    const std::string pipeline = "set -o pipefail; " + vspipeCmd.str() + " | " + ffmpegCmd.str();
    const std::string shellCmd = "bash -lc " + quoteArgVS(pipeline);
    const int rc = std::system(shellCmd.c_str());

    std::error_code ec;
    std::filesystem::remove(tmpScript, ec);

    if (rc != 0) {
        error = "VapourSynth streaming pipeline failed (exit code " +
                std::to_string(rc) + ") for " + toString(stage.kind);
        return StageResult::Error;
    }

    if (progressCb) {
        progressCb(1.0F, "VapourSynth processing complete.");
    }
    std::cout << "[vapoursynth] Done: " << toString(stage.kind)
              << " for " << inputVideo << " via direct stream" << std::endl;
    return StageResult::Processed;
}

// Shared implementation: prefer direct video streaming, fall back to
// frame-directory processing only when source plugins are unavailable.
StageResult processVideoFileImpl(
        const EnhancementStage& stage,
        const std::string& inputVideo,
        const std::string& outputVideo,
        const std::vector<ActiveFilter>& catalogFilters,
        const FrameProgressCb& progressCb,
        std::string& error,
        const ProcessVideoOptions& opts) {
    std::string directReason;
    if (vapoursynthDirectSourceAvailable(directReason)) {
        return processVideoFileViaStreaming(
            stage, inputVideo, outputVideo, catalogFilters, progressCb, error, opts);
    }

    std::cout << "[vapoursynth] Direct source plugins unavailable; falling back to image sequence path."
              << std::endl;
    return processVideoFileViaFrameSequence(
        stage, inputVideo, outputVideo, catalogFilters, progressCb, error, opts);
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
bool VapourSynthBackend::supportsDirectOutputEncode() const { return true; }

bool VapourSynthBackend::isAvailable(std::string& reason) const {
    return vapoursynthScriptPrereqsAvailable(reason);
}

bool VapourSynthBackend::initialize(std::string& error) {
    if (impl_->initialised) { return true; }

    if (!vapoursynthScriptPrereqsAvailable(error)) {
        return false;
    }

    impl_->initialised = true;
    std::cout << "[vapoursynth] Backend initialised." << std::endl;
    return true;
}

StageResult VapourSynthBackend::runStage(const EnhancementStage& /*stage*/, std::string& /*error*/) {
    // Pre-validation only; actual work is in processVideoFile.
    return StageResult::Deferred;
}

StageResult VapourSynthBackend::processVideoFile(
        const EnhancementStage& stage,
        const std::string& inputVideo,
        const std::string& outputVideo,
        const FrameProgressCb& progressCb,
        std::string& error,
        const ProcessVideoOptions& opts) {
    if (!impl_->initialised) {
        error = "VapourSynth backend not initialised.";
        return StageResult::Error;
    }
    return processVideoFileImpl(stage, inputVideo, outputVideo,
                                 impl_->catalogFilters, progressCb, error, opts);
}

void VapourSynthBackend::setCatalogFilters(
        const std::vector<ActiveFilter>& filters) {
    impl_->catalogFilters = filters;
}

#else  // !AVE_HAVE_VAPOURSYNTH — shell-out implementation

struct VapourSynthBackend::Impl {
    bool initialised = false;
    std::vector<ActiveFilter> catalogFilters;
};

VapourSynthBackend::VapourSynthBackend() : impl_(std::make_unique<Impl>()) {}
VapourSynthBackend::~VapourSynthBackend() = default;

BackendType VapourSynthBackend::type() const { return BackendType::VapourSynth; }
std::string VapourSynthBackend::name() const { return "VapourSynth"; }
bool VapourSynthBackend::supportsDirectOutputEncode() const { return true; }

bool VapourSynthBackend::isAvailable(std::string& reason) const {
    return vapoursynthScriptPrereqsAvailable(reason);
}

bool VapourSynthBackend::initialize(std::string& error) {
    if (impl_->initialised) { return true; }
    if (!vapoursynthScriptPrereqsAvailable(error)) {
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

StageResult VapourSynthBackend::processVideoFile(
        const EnhancementStage& stage,
        const std::string& inputVideo,
        const std::string& outputVideo,
        const FrameProgressCb& progressCb,
        std::string& error,
        const ProcessVideoOptions& opts) {
    if (!impl_->initialised) {
        error = "VapourSynth backend not initialised.";
        return StageResult::Error;
    }
    return processVideoFileImpl(stage, inputVideo, outputVideo,
                                 impl_->catalogFilters, progressCb, error, opts);
}

void VapourSynthBackend::setCatalogFilters(
        const std::vector<ActiveFilter>& filters) {
    impl_->catalogFilters = filters;
}

#endif  // AVE_HAVE_VAPOURSYNTH

}  // namespace ave
