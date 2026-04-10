#include "ave/video_processor.hpp"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <variant>

#include "ave/backends/glsl_shader_backend.hpp"
#include "ave/backends/vapoursynth_backend.hpp"
#include "ave/model_catalog.hpp"
#include "ave/stage.hpp"
#include "ave/telemetry.hpp"
#include "ave/types.hpp"

namespace ave {

namespace {

using Clock = std::chrono::steady_clock;

struct StageTimingEntry {
    std::string stageName;
    std::string outcome;
    double milliseconds = 0.0;
};

bool pathHasMxrExtension(const std::string& path) {
    if (path.size() < 4) {
        return false;
    }
    const std::string ext = path.substr(path.size() - 4);
    return ext == ".mxr" || ext == ".MXR";
}

bool backendUsesScriptPipeline(const BackendType type) {
    return type == BackendType::VapourSynth ||
           type == BackendType::GlslShader;
}

std::string joinCapabilitiesCsv(const std::vector<StageKind>& capabilities) {
    std::string out;
    for (std::size_t i = 0; i < capabilities.size(); ++i) {
        if (i > 0) {
            out += ",";
        }
        out += toString(capabilities[i]);
    }
    return out;
}

std::string serializeControlBindings(
    const std::unordered_map<std::string, std::string>& bindings) {
    std::string out;
    bool first = true;
    for (const auto& [inputName, binding] : bindings) {
        if (inputName.empty() || binding.empty()) {
            continue;
        }
        if (!first) {
            out += ";";
        }
        first = false;
        out += inputName;
        out += "=";
        out += binding;
    }
    return out;
}

double elapsedMilliseconds(const Clock::time_point start,
                           const Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

}  // namespace

// ─── resolveModelPath ────────────────────────────────────────────
// If the stage carries a "model" parameter (a model catalog id),
// look it up in ModelManager and inject a backend-appropriate on-disk
// path as "model_path" so backends can load it directly.
//
// When no explicit "model" parameter is present, fall back to the
// default model for the stage kind from the built-in catalog.  This
// ensures AI inference is attempted for all backend-eligible stages,
// not only those where the user manually specified a model.
EnhancementStage VideoProcessor::resolveModelPath(
        const EnhancementStage& stage,
        std::optional<BackendType> activeBackend) const {
    EnhancementStage out = stage;

    // Preserve a user-specified model path exactly as provided.
    auto modelPathIt = stage.params.find("model_path");
    if (modelPathIt != stage.params.end() && std::holds_alternative<std::string>(modelPathIt->second)) {
        const auto& explicitPath = std::get<std::string>(modelPathIt->second);
        if (!explicitPath.empty()) {
            std::string explicitModelId;
            auto modelIt = stage.params.find("model");
            if (modelIt != stage.params.end() && std::holds_alternative<std::string>(modelIt->second)) {
                explicitModelId = std::get<std::string>(modelIt->second);
            }
            if (explicitModelId.empty()) {
                explicitModelId = inferModelIdFromPath(explicitPath);
            }
            if (explicitModelId.empty()) {
                const auto entries = catalogEntriesForStage(stage.kind);
                for (const auto* e : entries) {
                    if (e->isDefault) { explicitModelId = e->id; break; }
                }
                if (explicitModelId.empty() && !entries.empty()) {
                    explicitModelId = entries.front()->id;
                }
            }
            if (!explicitModelId.empty()) {
                out.params["model"] = explicitModelId;
            }
            out.params["model_path"] = explicitPath;
            out.params["model_path_explicit"] = true;
            std::cout << "[model] " << toString(stage.kind)
                      << " using explicit model path: " << explicitPath << std::endl;
            return out;
        }
    }

    auto it = stage.params.find("model");
    const bool explicitModelRequested =
        it != stage.params.end() && std::holds_alternative<std::string>(it->second)
        && !std::get<std::string>(it->second).empty();

    // Determine the model ID: explicit param → backend-preferred default.
    std::string modelId;
    if (it != stage.params.end() && std::holds_alternative<std::string>(it->second)) {
        modelId = std::get<std::string>(it->second);
    }
    if (modelId.empty()) {
        if (activeBackend.has_value()) {
            if (const auto* preferred = preferredBackendModelForStage(stage.kind, *activeBackend);
                preferred != nullptr) {
                modelId = preferred->id;
            } else if (*activeBackend == BackendType::NcnnVulkan) {
                std::cout << "[model] " << toString(stage.kind)
                          << " — no NCNN-compatible catalog model; stage will use FFmpeg fallback."
                          << std::endl;
                return out;
            }
        }
        if (modelId.empty()) {
            // Fall back to the generic stage default when the backend has no
            // stage-specific model preference.
            const auto entries = catalogEntriesForStage(stage.kind);
            for (const auto* e : entries) {
                if (e->isDefault) { modelId = e->id; break; }
            }
            if (modelId.empty() && !entries.empty()) {
                modelId = entries.front()->id;
            }
        }
    }
    if (modelId.empty()) return out;

    // Inject the model ID so downstream code can find it.
    out.params["model"] = modelId;

    const auto managed = modelManager_.findModel(modelId);
    if (!managed.has_value()) {
        return out;
    }

    out.params["model_family_id"] = modelFamilyId(managed->entry);
    out.params["model_family_name"] = modelFamilyName(managed->entry);
    out.params["model_capabilities"] = joinCapabilitiesCsv(modelCapabilities(managed->entry));
    out.params["model_supports_fused_execution"] = managed->entry.supportsFusedExecution;
    out.params["model_supports_selective_capabilities"] =
        managed->entry.supportsSelectiveCapabilities;
    if (!managed->entry.controlInputBindings.empty()) {
        out.params["model_control_bindings"] =
            serializeControlBindings(managed->entry.controlInputBindings);
    }

    const bool migraphxActive =
        activeBackend.has_value() && *activeBackend == BackendType::MiGraphX;

    std::optional<std::string> selectedPath;
    if (migraphxActive) {
        if (!managed->convertedPath.empty()) {
            selectedPath = managed->convertedPath;
        } else if (!managed->downloadedPath.empty() && pathHasMxrExtension(managed->downloadedPath)) {
            selectedPath = managed->downloadedPath;
        }
    } else if (!managed->downloadedPath.empty() &&
               managed->downloadedPath != "(builtin)" &&
               !pathHasMxrExtension(managed->downloadedPath)) {
        // Non-MiGraphX backends need the original source model, not a cached
        // MiGraphX artifact that happens to sort as the "best" path overall.
        selectedPath = managed->downloadedPath;
    }

    if (selectedPath.has_value() && !selectedPath->empty() && *selectedPath != "(builtin)") {
        out.params["model_path"] = *selectedPath;
        out.params["model_path_explicit"] = false;
        std::cout << "[model] " << toString(stage.kind);
        if (migraphxActive && pathHasMxrExtension(*selectedPath)) {
            std::cout << " found cached compiled artifact candidate: " << *selectedPath
                      << " (MiGraphX will validate/select the exact runtime artifact).";
        } else {
            std::cout << " using model: " << *selectedPath;
        }
        std::cout << std::endl;
    } else if (migraphxActive &&
               !managed->downloadedPath.empty() &&
               managed->downloadedPath != "(builtin)") {
        std::cout << "[model] " << toString(stage.kind)
                  << " — model '" << modelId
                  << "' is available on disk;"
                  << " MiGraphX will validate/select the required cached artifact or compile one if needed."
                  << std::endl;
    } else if (!explicitModelRequested) {
        std::cout << "[model] " << toString(stage.kind)
                  << " — model '" << modelId
                  << "' not on disk, will fall back to FFmpeg filter" << std::endl;
    } else {
        std::cout << "[model] " << toString(stage.kind)
                  << " — explicit model '" << modelId
                  << "' not available on disk" << std::endl;
    }
    return out;
}

// ─── process ─────────────────────────────────────────────────────
bool VideoProcessor::process(const VideoJob& job, std::string& error) const {
    const auto totalStart = Clock::now();

    // Ensure model state is up to date
    modelManager_.refresh();

    const auto planningStart = Clock::now();
    const std::vector<EnhancementStage> ordered = planner_.plan(job.requestedStages);
    const auto planningEnd = Clock::now();

    // ── Log the planned pipeline ──────────────────────────────────
    std::ostringstream planSS;
    planSS << "Pipeline planned (" << ordered.size() << " stage"
           << (ordered.size() == 1 ? "" : "s") << ")";
    for (std::size_t i = 0; i < ordered.size(); ++i)
        planSS << (i == 0 ? ": " : " \u2192 ") << toString(ordered[i].kind);
    std::cout << "[plan] " << planSS.str() << '\n';
    if (ordered.empty())
        std::cout << "  (no enhancement stages \u2014 pass-through encode only)\n";

    if (job.progressCb) job.progressCb(0, 0, planSS.str());

    const auto telemetryProbe = probeAmdTelemetrySupport();
    std::cout << "[telemetry] " << telemetryProbe.summary() << std::endl;
    std::string telemetryError;
    if (const auto telemetry = collectAmdTelemetry(telemetryError); telemetry.has_value()) {
        std::cout << "[telemetry] initial: " << telemetry->summary() << std::endl;
    } else if (!telemetryError.empty()) {
        std::cout << "[telemetry] sample unavailable: " << telemetryError << std::endl;
    }

    if (job.dryRun) return true;

    std::error_code ec;
    if (!std::filesystem::exists(job.inputPath, ec)) {
        error = "Input video does not exist: " + job.inputPath;
        return false;
    }

    std::filesystem::path outputDir = std::filesystem::path(job.outputPath).parent_path();
    std::filesystem::create_directories(outputDir, ec);

    // ── Identify which stages are backend-eligible ────────────────
    int backendEligibleCount = 0;
    for (const auto& s : ordered) {
        if (backendUsesScriptPipeline(job.requestedBackend) ||
            (s.kind != StageKind::Interpolate && s.kind != StageKind::Sharpen)) {
            ++backendEligibleCount;
        }
    }

    // Only create and initialise the backend if there are stages that
    // could actually benefit from it.
    std::unique_ptr<IAcceleratorBackend> backend;
    const auto backendSetupStart = Clock::now();
    if (backendEligibleCount > 0) {
        std::string backendSummary;
        backend = backendManager_.createBackend(job.requestedBackend, backendSummary);
        if (!backend) {
            if (job.requestedBackend != BackendType::Auto) {
                error = backendSummary;
                std::cout << "[backend] " << backendSummary << std::endl;
                return false;
            }
            std::cout << "[backend] " << backendSummary
                      << "\n  All stages will use FFmpeg filter chain." << std::endl;
        } else {
            std::cout << "[backend] " << backendSummary << '\n';

            std::string backendError;
            if (!backend->initialize(backendError)) {
                if (job.requestedBackend != BackendType::Auto) {
                    error = backend->name() + " initialization failed: " + backendError;
                    std::cout << "[backend] init failed: " << backendError << std::endl;
                    return false;
                }
                std::cout << "[backend] init failed: " << backendError
                          << "\n  All stages will use FFmpeg filter chain." << std::endl;
                backend.reset();  // discard; we'll fall back to FFmpeg
            } else {
                std::cout << "[backend] ready: " << backend->name() << '\n';
            }

            // Pass catalog filters to backends that support them.
            if (backend && !job.catalogFilters.empty()) {
                if (auto* glsl = dynamic_cast<GlslShaderBackend*>(backend.get())) {
                    glsl->setCatalogFilters(job.catalogFilters);
                    std::cout << "[backend] Passed " << job.catalogFilters.size()
                              << " catalog filter(s) to GLSL backend." << std::endl;
                } else if (auto* vs = dynamic_cast<VapourSynthBackend*>(backend.get())) {
                    vs->setCatalogFilters(job.catalogFilters);
                    std::cout << "[backend] Passed " << job.catalogFilters.size()
                              << " catalog filter(s) to VapourSynth backend." << std::endl;
                }
            }
        }
    }
    const auto backendSetupEnd = Clock::now();

    // The backend pre-load pass is fast (model validation only);
    // real AI work happens inside ffmpeg_.encode() as one continuous
    // processing session for AI-capable stages.
    // So we allocate only 5% to the pre-load phase, 95% to the encode phase.
    constexpr int kPreloadPct = 5;

    auto reportProgress = [&](int overallPct, int taskPct, const std::string& msg) {
        std::cout << "[progress] " << msg << '\n';
        if (job.progressCb) job.progressCb(overallPct, taskPct, msg);
    };

    reportProgress(0, 0, "Starting enhancement pipeline\u2026");

    // ── Scene detection (for interpolation stages) ────────────────
    std::vector<SceneCut> sceneCuts;
    for (const auto& s : ordered) {
        if (s.kind != StageKind::Interpolate) continue;
        auto sdIt = s.params.find("scene_detect");
        bool doDetect = false;
        if (sdIt != s.params.end() && std::holds_alternative<bool>(sdIt->second))
            doDetect = std::get<bool>(sdIt->second);
        if (!doDetect) break;

        reportProgress(0, 8, "Detecting scene cuts in input video\u2026");
        std::string sceneError;
        SceneDetectorOptions opts;
        sceneDetector_.detect(job.inputPath, opts, sceneCuts, sceneError);
        if (!sceneError.empty()) {
            reportProgress(0, 10, "Scene detection warning: " + sceneError);
            std::cout << "[scene] warning: " << sceneError << '\n';
        } else {
            const std::string sceneMsg = "Scene detection complete: "
                + std::to_string(sceneCuts.size()) + " cut(s) found";
            reportProgress(0, 10, sceneMsg);
            std::cout << "[scene] " << sceneMsg << '\n';
        }
        break;
    }

    // ── Build a model-resolved copy of the ordered stages ──────────
    std::vector<EnhancementStage> resolvedOrdered;
    resolvedOrdered.reserve(ordered.size());
    const std::optional<BackendType> activeBackend =
        backend ? std::optional<BackendType>(backend->type()) : std::nullopt;
    const auto modelResolveStart = Clock::now();
    for (const auto& s : ordered) {
        EnhancementStage rs = resolveModelPath(s, activeBackend);
        if (rs.kind == StageKind::Interpolate && !sceneCuts.empty()) {
            rs.params["scene_cut_count"] = static_cast<std::int64_t>(sceneCuts.size());
        }
        resolvedOrdered.push_back(std::move(rs));
    }
    const auto modelResolveEnd = Clock::now();

    // ── Per-stage backend pass ─────────────────────────────────────
    int completedBackend = 0;
    int aiProcessedCount = 0;
    int deferredCount    = 0;
    std::vector<StageTimingEntry> preloadTimings;
    const bool scriptBackendActive = backend &&
        backendUsesScriptPipeline(backend->type());

    if (scriptBackendActive) {
        std::cout << "[pipeline] " << backend->name()
                  << " stages will execute during the encode phase."
                  << std::endl;
    } else {
        for (EnhancementStage& stage : resolvedOrdered) {
            if (stage.kind == StageKind::Interpolate ||
                stage.kind == StageKind::Sharpen) {
                continue;
            }

            const std::string stageName = toString(stage.kind);

            if (!backend) {
                // No backend available — stage deferred to FFmpeg.
                ++deferredCount;
                ++completedBackend;
                std::cout << "[pipeline] " << stageName
                          << " → FFmpeg filter (no backend)" << std::endl;
                continue;
            }

            const int overallBefore = backendEligibleCount > 0
                ? (completedBackend * kPreloadPct) / backendEligibleCount
                : 0;
            reportProgress(overallBefore, 0,
                "Backend pass " + std::to_string(completedBackend + 1) + "/" +
                std::to_string(backendEligibleCount) + ": " + stageName +
                " via " + backend->name());

            const auto stageStart = Clock::now();
            std::string backendError;
            const StageResult result = backend->runStage(stage, backendError);
            const auto stageEnd = Clock::now();
            StageTimingEntry timing{stageName, "deferred", elapsedMilliseconds(stageStart, stageEnd)};

            switch (result) {
                case StageResult::Processed:
                    stage.backendProcessed = true;
                    ++aiProcessedCount;
                    timing.outcome = "processed";
                    std::cout << "[pipeline] " << stageName
                              << " → AI inference complete (" << backend->name() << ")"
                              << std::endl;
                    break;

                case StageResult::Deferred:
                    ++deferredCount;
                    timing.outcome = "deferred";
                    std::cout << "[pipeline] " << stageName
                              << " → deferred to FFmpeg filter chain"
                              << std::endl;
                    break;

                case StageResult::Error: {
                    timing.outcome = "error";
                    preloadTimings.push_back(timing);
                    std::ostringstream os;
                    os << "Backend stage " << stageName << " failed: " << backendError;
                    reportProgress(overallBefore, 0, os.str());
                    error = os.str();
                    return false;
                }

                case StageResult::Cancelled:
                    timing.outcome = "cancelled";
                    preloadTimings.push_back(timing);
                    error = "Processing cancelled by user.";
                    reportProgress(overallBefore, 0, error);
                    return false;
            }

            preloadTimings.push_back(timing);
            ++completedBackend;
            const int overallAfter = backendEligibleCount > 0
                ? (completedBackend * kPreloadPct) / backendEligibleCount
                : kPreloadPct;
            reportProgress(overallAfter, 100, stageName + " \u2014 complete");
        }
    }

    // ── Summary of processing modes ────────────────────────────────
    if (aiProcessedCount > 0 || deferredCount > 0) {
        std::cout << "[pipeline] stage summary: " << aiProcessedCount << " AI-processed, "
                  << deferredCount << " deferred to FFmpeg filters" << std::endl;
    }

    // ── FFmpeg encode pass ─────────────────────────────────────────
    const int encodeBase = kPreloadPct;
    reportProgress(encodeBase, 0,
        backend ? "Starting continuous AI/FFmpeg encode session…"
                : "Starting FFmpeg encode pipeline…");

    // Wrap progressCb: encode task% maps from encodeBase (5%) to 100%.
    // The inner FFmpeg/AI pipeline reports 0-100 as taskPct; we map that
    // into the remaining 95% of overall progress so the bar moves
    // proportionally to actual frame-processing work.
    VideoJob encodeJob = job;
    if (job.progressCb) {
        encodeJob.progressCb = [origCb = job.progressCb, encodeBase](
                int /*unused*/, int taskPct, const std::string& msg) {
            const int overall = encodeBase + (100 - encodeBase) * taskPct / 100;
            origCb(overall, taskPct, msg);
        };
    }

    std::string ffmpegError;
    const auto encodeStart = Clock::now();
    if (!ffmpeg_.encode(encodeJob, resolvedOrdered, backend.get(), ffmpegError)) {
        reportProgress(encodeBase, 0, "Encode failed: " + ffmpegError);
        error = ffmpegError;
        return false;
    }
    const auto encodeEnd = Clock::now();

    std::cout << "[pipeline-timing] planning_ms="
              << elapsedMilliseconds(planningStart, planningEnd)
              << " backend_setup_ms=" << elapsedMilliseconds(backendSetupStart, backendSetupEnd)
              << " model_resolution_ms=" << elapsedMilliseconds(modelResolveStart, modelResolveEnd)
              << " encode_ms=" << elapsedMilliseconds(encodeStart, encodeEnd)
              << " total_ms=" << elapsedMilliseconds(totalStart, Clock::now())
              << std::endl;
    for (const auto& entry : preloadTimings) {
        std::cout << "[pipeline-timing] preload_stage=" << entry.stageName
                  << " outcome=" << entry.outcome
                  << " ms=" << entry.milliseconds
                  << std::endl;
    }

    telemetryError.clear();
    if (const auto telemetry = collectAmdTelemetry(telemetryError); telemetry.has_value()) {
        std::cout << "[telemetry] final: " << telemetry->summary() << std::endl;
        if (const std::string hint = telemetry->pressureHint(); !hint.empty()) {
            std::cout << "[telemetry] hint: " << hint << std::endl;
        }
    } else if (!telemetryError.empty()) {
        std::cout << "[telemetry] final sample unavailable: " << telemetryError << std::endl;
    }

    if (job.progressCb) job.progressCb(100, 100,
        "Job complete \u2014 output saved to: " + job.outputPath);
    return true;
}

}  // namespace ave
