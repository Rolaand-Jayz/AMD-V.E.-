#include "ave/video_processor.hpp"

#include <filesystem>
#include <iostream>
#include <sstream>
#include <variant>

#include "ave/model_catalog.hpp"
#include "ave/stage.hpp"
#include "ave/types.hpp"

namespace ave {

// ─── resolveModelPath ────────────────────────────────────────────
// If the stage carries a "model" parameter (a model catalog id),
// look it up in ModelManager and inject the best on-disk path as
// "model_path" so backends can load it directly.
//
// When no explicit "model" parameter is present, fall back to the
// default model for the stage kind from the built-in catalog.  This
// ensures AI inference is attempted for all backend-eligible stages,
// not only those where the user manually specified a model.
EnhancementStage VideoProcessor::resolveModelPath(const EnhancementStage& stage) const {
    EnhancementStage out = stage;

    // Determine the model ID: explicit param → catalog default.
    std::string modelId;
    auto it = stage.params.find("model");
    if (it != stage.params.end() && std::holds_alternative<std::string>(it->second)) {
        modelId = std::get<std::string>(it->second);
    }
    if (modelId.empty()) {
        // Fall back to the default model for this stage kind.
        const auto entries = catalogEntriesForStage(stage.kind);
        for (const auto* e : entries) {
            if (e->isDefault) { modelId = e->id; break; }
        }
        if (modelId.empty() && !entries.empty()) {
            modelId = entries.front()->id;
        }
    }
    if (modelId.empty()) return out;

    // Inject the model ID so downstream code can find it.
    out.params["model"] = modelId;

    auto bestPath = modelManager_.bestPathForModel(modelId);
    if (bestPath) {
        out.params["model_path"] = *bestPath;
        std::cout << "[model] " << toString(stage.kind)
                  << " using model: " << *bestPath << std::endl;
    } else {
        std::cout << "[model] " << toString(stage.kind)
                  << " — model '" << modelId
                  << "' not on disk, will fall back to FFmpeg filter" << std::endl;
    }
    return out;
}

// ─── process ─────────────────────────────────────────────────────
bool VideoProcessor::process(const VideoJob& job, std::string& error) const {
    // Ensure model state is up to date
    modelManager_.refresh();

    const std::vector<EnhancementStage> ordered = planner_.plan(job.requestedStages);

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

    if (job.dryRun) return true;

    std::error_code ec;
    if (!std::filesystem::exists(job.inputPath, ec)) {
        error = "Input video does not exist: " + job.inputPath;
        return false;
    }

    std::filesystem::path outputDir = std::filesystem::path(job.outputPath).parent_path();
    std::filesystem::create_directories(outputDir, ec);

    // ── Identify which stages are backend-eligible ────────────────
    // Interpolate and Sharpen are FFmpeg-only.
    // All other stages (including Upscale) are candidates for AI backend inference.
    int backendEligibleCount = 0;
    for (const auto& s : ordered) {
        if (s.kind != StageKind::Interpolate &&
            s.kind != StageKind::Sharpen)
            ++backendEligibleCount;
    }

    // Only create and initialise the backend if there are stages that
    // could actually benefit from it.
    std::unique_ptr<IAcceleratorBackend> backend;
    if (backendEligibleCount > 0) {
        std::string backendSummary;
        backend = backendManager_.createBackend(job.requestedBackend, backendSummary);
        if (!backend) {
            // No backend available — all eligible stages will use FFmpeg filters.
            std::cout << "[backend] " << backendSummary
                      << "\n  All stages will use FFmpeg filter chain." << std::endl;
        } else {
            std::cout << "[backend] " << backendSummary << '\n';

            std::string backendError;
            if (!backend->initialize(backendError)) {
                std::cout << "[backend] init failed: " << backendError
                          << "\n  All stages will use FFmpeg filter chain." << std::endl;
                backend.reset();  // discard; we'll fall back to FFmpeg
            } else {
                std::cout << "[backend] ready: " << backend->name() << '\n';
            }
        }
    }

    const int totalItems = backendEligibleCount + 1; // +1 for the FFmpeg encode pass

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
    for (const auto& s : ordered) {
        EnhancementStage rs = resolveModelPath(s);
        if (rs.kind == StageKind::Interpolate && !sceneCuts.empty()) {
            rs.params["scene_cut_count"] = static_cast<std::int64_t>(sceneCuts.size());
        }
        resolvedOrdered.push_back(std::move(rs));
    }

    // ── Per-stage backend pass ─────────────────────────────────────
    int completedBackend = 0;
    int aiProcessedCount = 0;
    int deferredCount    = 0;

    for (EnhancementStage& stage : resolvedOrdered) {
        if (stage.kind == StageKind::Interpolate ||
            stage.kind == StageKind::Sharpen)       continue;

        const std::string stageName = toString(stage.kind);

        if (!backend) {
            // No backend available — stage deferred to FFmpeg.
            ++deferredCount;
            ++completedBackend;
            std::cout << "[pipeline] " << stageName
                      << " → FFmpeg filter (no backend)" << std::endl;
            continue;
        }

        const int overallBefore = (completedBackend * 100) / totalItems;
        reportProgress(overallBefore, 0,
            "Backend pass " + std::to_string(completedBackend + 1) + "/" +
            std::to_string(backendEligibleCount) + ": " + stageName +
            " via " + backend->name());

        std::string backendError;
        const StageResult result = backend->runStage(stage, backendError);

        switch (result) {
            case StageResult::Processed:
                stage.backendProcessed = true;
                ++aiProcessedCount;
                std::cout << "[pipeline] " << stageName
                          << " → AI inference complete (" << backend->name() << ")"
                          << std::endl;
                break;

            case StageResult::Deferred:
                ++deferredCount;
                std::cout << "[pipeline] " << stageName
                          << " → deferred to FFmpeg filter chain"
                          << std::endl;
                break;

            case StageResult::Error: {
                std::ostringstream os;
                os << "Backend stage " << stageName << " failed: " << backendError;
                reportProgress(overallBefore, 0, os.str());
                error = os.str();
                return false;
            }
        }

        ++completedBackend;
        const int overallAfter = (completedBackend * 100) / totalItems;
        reportProgress(overallAfter, 100, stageName + " \u2014 complete");
    }

    // ── Summary of processing modes ────────────────────────────────
    if (aiProcessedCount > 0 || deferredCount > 0) {
        std::cout << "[pipeline] stage summary: " << aiProcessedCount << " AI-processed, "
                  << deferredCount << " deferred to FFmpeg filters" << std::endl;
    }

    // ── FFmpeg encode pass ─────────────────────────────────────────
    const int encodeBase = (completedBackend * 100) / totalItems;
    reportProgress(encodeBase, 0, "Starting FFmpeg encode pipeline\u2026");

    // Wrap progressCb: encode task% maps overall from encodeBase to 100
    VideoJob encodeJob = job;
    if (job.progressCb) {
        encodeJob.progressCb = [origCb = job.progressCb, encodeBase](
                int /*unused*/, int taskPct, const std::string& msg) {
            const int overall = encodeBase + (100 - encodeBase) * taskPct / 100;
            origCb(overall, taskPct, msg);
        };
    }

    std::string ffmpegError;
    if (!ffmpeg_.encode(encodeJob, resolvedOrdered, backend.get(), ffmpegError)) {
        reportProgress(encodeBase, 0, "Encode failed: " + ffmpegError);
        error = ffmpegError;
        return false;
    }

    if (job.progressCb) job.progressCb(100, 100,
        "Job complete \u2014 output saved to: " + job.outputPath);
    return true;
}

}  // namespace ave
