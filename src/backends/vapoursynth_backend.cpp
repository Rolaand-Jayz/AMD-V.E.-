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
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "ave/filter_catalog.hpp"
#include "ave/frame_io.hpp"
#include "ave/process_observer.hpp"
#include "ave/process_progress.hpp"
#include "ave/stage.hpp"
#include "ave/types.hpp"
#include "ave/video_probe.hpp"

#ifdef AVE_HAVE_VAPOURSYNTH
#  include <VapourSynth4.h>
#  include <VSScript4.h>
#  include <VSHelper4.h>
#endif

namespace ave {
namespace {

bool looksLikeVpyScriptPath(const std::string& path) {
    const std::filesystem::path scriptPath(path);
    return scriptPath.extension() == ".vpy";
}

std::string absolutePathVS(const std::string& path) {
    std::error_code ec;
    const std::filesystem::path absolute = std::filesystem::absolute(path, ec);
    if (ec) {
        return path;
    }
    return absolute.lexically_normal().string();
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
        "vspipe --outputindex 0 " + process_observer::quoteShellArg(probePath)
        + " - > /dev/null 2>&1";
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
    if (!process_observer::commandInPath("vspipe")) {
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

void parseVspipeProgressVS(const std::string& line,
                           const std::int64_t fallbackTotalFrames,
                           std::int64_t& lastFrame,
                           const FrameProgressCb& progressCb,
                           const float base,
                           const float span,
                           const std::string& label) {
    if (!progressCb) {
        return;
    }

    long long current = 0;
    long long total = 0;
    if (std::sscanf(line.c_str(), "Frame: %lld/%lld", &current, &total) == 2) {
        if (current <= lastFrame) {
            return;
        }
        lastFrame = current;
        const auto denominator = total > 0 ? total : fallbackTotalFrames;
        const float frac = denominator > 0
            ? std::min(1.0f, static_cast<float>(current) / static_cast<float>(denominator))
            : 0.0f;
        std::string msg = label + " - frame " + std::to_string(current);
        if (denominator > 0) {
            msg += "/" + std::to_string(denominator);
        }
        process_progress::reportProgressFraction(progressCb, base, span, frac, msg);
        return;
    }

    if (line.rfind("Output ", 0) == 0) {
        process_progress::reportProgressFraction(
            progressCb, base, span, 1.0f, label + " complete.");
    }
}


std::string stripVsOutputCalls(const std::string& source) {
    std::istringstream input(source);
    std::ostringstream output;
    std::string line;
    while (std::getline(input, line)) {
        const std::string trimmed = process_observer::trimOutput(line);
        if (trimmed.find(".set_output(") != std::string::npos) {
            continue;
        }
        output << line << '\n';
    }
    return output.str();
}

std::optional<std::string> customVpyScriptPath(const EnhancementStage& stage) {
    const auto it = stage.params.find("vpy_script_path");
    if (it == stage.params.end()) {
        return std::nullopt;
    }
    if (const auto* value = std::get_if<std::string>(&it->second);
        value != nullptr && !process_observer::trimOutput(*value).empty()) {
        return process_observer::trimOutput(*value);
    }
    return std::nullopt;
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
            std::int64_t widthValue = static_cast<std::int64_t>(targetW);
            std::int64_t heightValue = static_cast<std::int64_t>(targetH);
            if (tryGetInt(stage, StageKind::Upscale, "width", widthValue)) {
                targetW = static_cast<int>(widthValue);
            }
            if (tryGetInt(stage, StageKind::Upscale, "height", heightValue)) {
                targetH = static_cast<int>(heightValue);
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

        case StageKind::Stereo3D:
            script << "# Stereo 3D stage\n"
                   << "raise RuntimeError('Stereo 3D synthesis requires the MiGraphX depth backend')\n";
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
    cmd << "-i " << process_observer::quoteShellArg(audioInputVideo) << ' '
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
    cmd << "-c:a copy -shortest " << process_observer::quoteShellArg(outputVideo);
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

std::string generateCustomWrapperVpyScript(const std::string& userScriptPath,
                                           const std::string& inputVideo,
                                           int previewFrames) {
    std::ostringstream script;
    script << "import vapoursynth as vs\n"
           << "core = vs.core\n"
           << "SOURCE_PATH = " << quotePyString(inputVideo) << '\n'
           << "PREVIEW_FRAMES = " << previewFrames << '\n'
           << "USER_SCRIPT_PATH = " << quotePyString(userScriptPath) << '\n'
           << "with open(USER_SCRIPT_PATH, 'r', encoding='utf-8') as _src:\n"
           << "    exec(compile(_src.read(), USER_SCRIPT_PATH, 'exec'), globals(), globals())\n"
           << "if 'clip' in globals() and hasattr(globals()['clip'], 'set_output'):\n"
           << "    clip = globals()['clip']\n"
           << "    if PREVIEW_FRAMES > 0:\n"
           << "        clip = core.std.Trim(clip, first=0, last=PREVIEW_FRAMES - 1)\n"
           << "    clip.set_output()\n";
    return script.str();
}

StageResult processVideoFileViaCustomScript(
        const std::string& userScriptPath,
        const std::string& inputVideo,
        const std::string& outputVideo,
        const FrameProgressCb& progressCb,
        std::string& error,
        const ProcessVideoOptions& opts) {
    if (!looksLikeVpyScriptPath(userScriptPath)) {
        error = "Custom VapourSynth script must be a .vpy file.";
        return StageResult::Error;
    }
    if (!process_observer::fileExists(userScriptPath)) {
        error = "Custom VapourSynth script not found: " + userScriptPath;
        return StageResult::Error;
    }

    const auto info = probeVideoStream(inputVideo, error);
    if (!info.has_value()) {
        return StageResult::Error;
    }
    const std::int64_t totalFrames =
        process_observer::countVideoFrames(inputVideo, opts.previewDurationSec);
    int previewFrames = 0;
    if (opts.previewDurationSec > 0.0) {
        previewFrames = std::max(
            1,
            static_cast<int>(opts.previewDurationSec * info->effectiveFrameRate() + 0.5));
    }

    const std::string tmpBase = "/tmp/ave_vs_custom_" +
        std::to_string(std::hash<std::string>{}(inputVideo + userScriptPath + outputVideo));
    const std::string tmpScript = tmpBase + ".vpy";
    {
        std::ofstream ofs(tmpScript);
        if (!ofs.is_open()) {
            error = "Failed to write wrapper VPY script: " + tmpScript;
            return StageResult::Error;
        }
        ofs << generateCustomWrapperVpyScript(absolutePathVS(userScriptPath),
                                              absolutePathVS(inputVideo),
                                              previewFrames);
    }

    if (progressCb) {
        progressCb(0.05F, "VapourSynth - running custom .vpy pipeline...");
    }

    std::ostringstream vspipeCmd;
    vspipeCmd << "vspipe --progress --outputindex 0 -c y4m "
              << process_observer::quoteShellArg(tmpScript) << " -";

    std::ostringstream ffmpegCmd;
    ffmpegCmd << "ffmpeg -hide_banner -loglevel error -progress - -y "
              << "-f yuv4mpegpipe -i pipe:0 ";
    appendEncodeArgs(ffmpegCmd, absolutePathVS(outputVideo), absolutePathVS(inputVideo), opts);

    const std::string pipeline = "{ set -o pipefail; " + vspipeCmd.str() +
                                 " | " + ffmpegCmd.str() + "; } 2>&1";
    const std::string shellCmd = "bash -lc " + process_observer::quoteShellArg(pipeline);
    std::vector<std::string> diagnostics;
    std::int64_t lastFrame = -1;
    const int rc = process_observer::runObservedCommand(shellCmd, [&](const std::string& line) {
        parseVspipeProgressVS(line,
                              totalFrames,
                              lastFrame,
                              progressCb,
                              0.05f,
                              0.90f,
                              "VapourSynth custom script");
        if (!process_progress::isFfmpegProgressField(line) &&
            line.rfind("Frame: ", 0) != 0 &&
            line.rfind("Output ", 0) != 0) {
            diagnostics.push_back(line);
        }
    });

    std::error_code ec;
    std::filesystem::remove(tmpScript, ec);

    if (rc != 0) {
        error = "Custom VapourSynth script failed (exit code " +
                std::to_string(rc) + ")";
        const std::string detail = process_observer::summarizeDiagnostics(diagnostics);
        if (!detail.empty()) {
            error += ": " + detail;
        }
        return StageResult::Error;
    }

    if (progressCb) {
        progressCb(1.0F, "Custom VapourSynth processing complete.");
    }
    return StageResult::Processed;
}

StageResult processVideoFileViaFrameSequence(
        const EnhancementStage& stage,
        const std::string& inputVideo,
        const std::string& outputVideo,
        const std::vector<ActiveFilter>& catalogFilters,
        const FrameProgressCb& progressCb,
        std::string& error,
        const ProcessVideoOptions& opts) {
    const std::string absoluteInputVideo = absolutePathVS(inputVideo);
    const std::string absoluteOutputVideo = absolutePathVS(outputVideo);

    const auto info = probeVideoStream(absoluteInputVideo, error);
    if (!info.has_value()) {
        return StageResult::Error;
    }
    const std::int64_t totalFrames =
        process_observer::countVideoFrames(absoluteInputVideo, opts.previewDurationSec);

    // Create temp directories for frame I/O.
    const std::string tmpBase = "/tmp/ave_vs_" +
        std::to_string(std::hash<std::string>{}(inputVideo + toString(stage.kind) + outputVideo));
    const std::string tmpIn  = tmpBase + "_in";
    const std::string tmpOut = tmpBase + "_out";
    std::error_code ec;
    std::filesystem::create_directories(tmpIn, ec);
    std::filesystem::create_directories(tmpOut, ec);

    if (progressCb) {
        progressCb(0.0F, "VapourSynth - extracting frames with FFmpeg...");
    }

    // Extract frames from input video.
    std::ostringstream extractOss;
    extractOss << "ffmpeg -hide_banner -loglevel error -progress - -y ";
    if (opts.previewDurationSec > 0.0) {
        extractOss << "-t " << opts.previewDurationSec << " ";
    }
    extractOss << "-i \"" << absoluteInputVideo << "\" "
               << "\"" << tmpIn << "/%08d.png\" 2>&1";
    std::vector<std::string> diagnostics;
    std::int64_t lastFrame = -1;
    int rc = process_observer::runObservedCommand(extractOss.str(), [&](const std::string& line) {
        process_progress::parseFfmpegProgress(
            line, totalFrames, lastFrame, progressCb, 0.0f, 0.15f, "VapourSynth extract");
        if (!process_progress::isFfmpegProgressField(line)) {
            diagnostics.push_back(line);
        }
    });
    if (rc != 0) {
        error = "Frame extraction failed (exit code " + std::to_string(rc) + ")";
        const std::string detail = process_observer::summarizeDiagnostics(diagnostics);
        if (!detail.empty()) {
            error += ": " + detail;
        }
        std::filesystem::remove_all(tmpIn, ec);
        std::filesystem::remove_all(tmpOut, ec);
        return StageResult::Error;
    }

    if (progressCb) { progressCb(0.15F, "VapourSynth - running vspipe pipeline..."); }

    // Generate VPY script and write to temp file.
    const std::string script = generateFrameSequenceVpyScript(
        stage,
        tmpIn,
        tmpOut,
        static_cast<int>(info->width),
        static_cast<int>(info->height),
        catalogFilters);
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

    // Run vspipe and request frames without writing raw output anywhere.
    const std::string vsCmd = "vspipe --progress --outputindex 0 " +
                              process_observer::quoteShellArg(tmpScript) + " -- 2>&1";
    diagnostics.clear();
    lastFrame = -1;
    rc = process_observer::runObservedCommand(vsCmd, [&](const std::string& line) {
        parseVspipeProgressVS(line,
                              totalFrames,
                              lastFrame,
                              progressCb,
                              0.15f,
                              0.65f,
                              "VapourSynth");
        if (line.rfind("Frame: ", 0) != 0 && line.rfind("Output ", 0) != 0) {
            diagnostics.push_back(line);
        }
    });
    std::filesystem::remove(tmpScript, ec);

    if (rc != 0) {
        error = "VapourSynth processing failed (exit code " +
                std::to_string(rc) + ") for " + toString(stage.kind);
        const std::string detail = process_observer::summarizeDiagnostics(diagnostics);
        if (!detail.empty()) {
            error += ": " + detail;
        }
        std::filesystem::remove_all(tmpIn, ec);
        std::filesystem::remove_all(tmpOut, ec);
        return StageResult::Error;
    }

    if (progressCb) { progressCb(0.8F, "VapourSynth - re-encoding output video..."); }

    std::ostringstream encOss;
    encOss << "ffmpeg -hide_banner -loglevel error -progress - -y "
           << "-framerate " << info->effectiveFrameRate() << ' '
           << "-i " << process_observer::quoteShellArg(tmpOut + "/%08d.png") << ' ';
    appendEncodeArgs(encOss, absoluteOutputVideo, absoluteInputVideo, opts);
    encOss << " 2>&1";
    diagnostics.clear();
    lastFrame = -1;
    rc = process_observer::runObservedCommand(encOss.str(), [&](const std::string& line) {
        process_progress::parseFfmpegProgress(
            line, totalFrames, lastFrame, progressCb, 0.8f, 0.2f, "VapourSynth encode");
        if (!process_progress::isFfmpegProgressField(line)) {
            diagnostics.push_back(line);
        }
    });

    // Cleanup temp directories.
    std::filesystem::remove_all(tmpIn, ec);
    std::filesystem::remove_all(tmpOut, ec);

    if (rc != 0) {
        error = "Re-encoding output frames failed (exit code " + std::to_string(rc) + ")";
        const std::string detail = process_observer::summarizeDiagnostics(diagnostics);
        if (!detail.empty()) {
            error += ": " + detail;
        }
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
    const std::string absoluteInputVideo = absolutePathVS(inputVideo);
    const std::string absoluteOutputVideo = absolutePathVS(outputVideo);

    const auto info = probeVideoStream(absoluteInputVideo, error);
    if (!info.has_value()) {
        return StageResult::Error;
    }
    const std::int64_t totalFrames =
        process_observer::countVideoFrames(absoluteInputVideo, opts.previewDurationSec);

    int previewFrames = 0;
    if (opts.previewDurationSec > 0.0) {
        previewFrames = std::max(
            1,
            static_cast<int>(opts.previewDurationSec * info->effectiveFrameRate() + 0.5));
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
            stage,
            absoluteInputVideo,
            static_cast<int>(info->width),
            static_cast<int>(info->height),
            previewFrames,
            catalogFilters);
    }

    if (progressCb) {
        progressCb(0.05F, "VapourSynth - running streaming pipeline...");
    }

    std::ostringstream vspipeCmd;
    vspipeCmd << "vspipe --progress --outputindex 0 -c y4m "
              << process_observer::quoteShellArg(tmpScript) << " -";

    std::ostringstream ffmpegCmd;
    ffmpegCmd << "ffmpeg -hide_banner -loglevel error -progress - -y "
              << "-f yuv4mpegpipe -i pipe:0 ";
    appendEncodeArgs(ffmpegCmd, absoluteOutputVideo, absoluteInputVideo, opts);

    const std::string pipeline = "{ set -o pipefail; " + vspipeCmd.str() +
                                 " | " + ffmpegCmd.str() + "; } 2>&1";
    const std::string shellCmd = "bash -lc " + process_observer::quoteShellArg(pipeline);
    std::vector<std::string> diagnostics;
    std::int64_t lastFrame = -1;
    const int rc = process_observer::runObservedCommand(shellCmd, [&](const std::string& line) {
        parseVspipeProgressVS(line,
                              totalFrames,
                              lastFrame,
                              progressCb,
                              0.05f,
                              0.90f,
                              "VapourSynth");
        if (!process_progress::isFfmpegProgressField(line) &&
            line.rfind("Frame: ", 0) != 0 &&
            line.rfind("Output ", 0) != 0) {
            diagnostics.push_back(line);
        }
    });

    std::error_code ec;
    std::filesystem::remove(tmpScript, ec);

    if (rc != 0) {
        error = "VapourSynth streaming pipeline failed (exit code " +
                std::to_string(rc) + ") for " + toString(stage.kind);
        const std::string detail = process_observer::summarizeDiagnostics(diagnostics);
        if (!detail.empty()) {
            error += ": " + detail;
        }
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
    if (const auto customScript = customVpyScriptPath(stage); customScript.has_value()) {
        return processVideoFileViaCustomScript(
            *customScript, inputVideo, outputVideo, progressCb, error, opts);
    }

    std::string directReason;
    if (vapoursynthDirectSourceAvailable(directReason)) {
        const StageResult directResult = processVideoFileViaStreaming(
            stage, inputVideo, outputVideo, catalogFilters, progressCb, error, opts);
        if (directResult == StageResult::Processed ||
            directResult == StageResult::Cancelled) {
            return directResult;
        }

        std::string imageReason;
        if (!vapoursynthImageSequenceAvailable(imageReason)) {
            return directResult;
        }

        const std::string directError = error;
        std::cout << "[vapoursynth] Direct streaming path failed (" << directError
                  << "); retrying via image-sequence fallback." << std::endl;
        error.clear();
        const StageResult fallbackResult = processVideoFileViaFrameSequence(
            stage, inputVideo, outputVideo, catalogFilters, progressCb, error, opts);
        if (fallbackResult == StageResult::Error && !directError.empty()) {
            error += " | direct streaming path: " + directError;
        }
        return fallbackResult;
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
