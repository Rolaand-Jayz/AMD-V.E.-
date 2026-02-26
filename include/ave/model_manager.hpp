#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "ave/app_settings.hpp"
#include "ave/model_catalog.hpp"

namespace ave {

// ─────────────────────────────────────────────────────────────────
// ModelState – lifecycle of a model file on disk
// ─────────────────────────────────────────────────────────────────
enum class ModelState {
    NotDownloaded,
    Downloading,
    Downloaded,         // ONNX or NCNN files present
    Converting,         // MiGraphX compilation in progress
    Converted,          // .mxr file present, not hardware-tuned
    Optimizing,         // GPU-tuned compile in progress
    Optimized,          // GPU-tuned .mxr present
    Error
};

std::string toString(ModelState state);

// ─────────────────────────────────────────────────────────────────
// ManagedModel – runtime record combining catalog + disk state
// ─────────────────────────────────────────────────────────────────
struct ManagedModel {
    ModelEntry      entry;
    ModelState      state           = ModelState::NotDownloaded;
    std::string     downloadedPath;    // path to ONNX / NCNN param file
    std::string     downloadedPathAux; // path to NCNN .bin file (if any)
    std::string     convertedPath;     // path to .mxr (MiGraphX compiled)
    std::string     optimizedPath;     // path to .mxr (GPU-tuned)
    float           downloadProgress  = 0.0f;
    std::string     errorMessage;
};

// ─────────────────────────────────────────────────────────────────
// Callback types
// ─────────────────────────────────────────────────────────────────
// Called from worker threads – receiver must post to UI thread if
// updating Qt widgets.
using ModelProgressCb = std::function<void(const std::string& modelId,
                                           float progress,
                                           const std::string& statusMsg)>;

using ModelStateCb    = std::function<void(const std::string& modelId,
                                           ModelState newState)>;

// ─────────────────────────────────────────────────────────────────
// ModelManager
// ─────────────────────────────────────────────────────────────────
class ModelManager {
  public:
    ModelManager();
    ~ModelManager();

    // ── Directory configuration ──────────────────────────────────

    // Default: ~/.local/share/ave/models  (created on first call)
    void setModelsDirectory(const std::string& dir);
    std::string modelsDirectory() const;

    // ── Catalog + scan ───────────────────────────────────────────

    // Scan modelsDirectory() and reconcile state with catalog.
    // Must be called once after construction (and again whenever
    // external files are added/removed).
    void refresh();

    // Full list of all catalog entries with their current on-disk state.
    std::vector<ManagedModel> allModels() const;

    // Filtered by stage kind.
    std::vector<ManagedModel> modelsForStage(StageKind kind) const;

    // Lookup by catalog id.
    std::optional<ManagedModel> findModel(const std::string& modelId) const;

    // ── Path selection ───────────────────────────────────────────

    // Returns the best inference-ready path for the given model:
    //   optimized > converted > downloaded
    // Returns nullopt when nothing is available.
    std::optional<std::string> bestPathForModel(const std::string& modelId) const;

    // ── Download ─────────────────────────────────────────────────

    // Starts an async download for the given model.
    // Returns false immediately if the model has no downloadUrl or is
    // already downloaded / downloading.
    bool startDownload(const std::string& modelId,
                       const ModelProgressCb& progressCb,
                       const ModelStateCb&    stateCb,
                       std::string&           error);

    // Cancel an in-flight download.  No-op if not downloading.
    void cancelDownload(const std::string& modelId);

    // ── MiGraphX compilation ─────────────────────────────────────

    // Synchronously compile the downloaded ONNX to a MiGraphX .mxr
    // program file.  Requires migraphx-driver on PATH or the MiGraphX
    // C++ runtime headers/libs linked in.
    // Progress callback is called periodically (0.0–1.0).
    // precisionOverride: if set, overrides the catalog entry precision.
    bool convertToMiGraphX(const std::string& modelId,
                           const ModelProgressCb& progressCb,
                           const ModelStateCb&    stateCb,
                           std::string&           error,
                           std::optional<ModelPrecision> precisionOverride = std::nullopt);

    // Compile with --gpu tuning enabled (slow first run but faster inference).
    bool optimizeForHardware(const std::string& modelId,
                             const ModelProgressCb& progressCb,
                             const ModelStateCb&    stateCb,
                             std::string&           error,
                             std::optional<ModelPrecision> precisionOverride = std::nullopt);

    // ── UI helpers ───────────────────────────────────────────────

    // Returns a UI-ready label for a model: "[Optimized] Real-ESRGAN x4"
    std::string modelDropdownLabel(const std::string& modelId) const;

    // Returns sorted list of dropdown labels for a given stage, plus the
    // "ffmpeg fallback" entry as the last item when applicable.
    struct DropdownEntry {
        std::string modelId;
        std::string label;
        bool        inferenceReady; // downloaded, converted, or optimized
    };
    std::vector<DropdownEntry> dropdownEntriesForStage(StageKind kind) const;

  private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

}  // namespace ave
