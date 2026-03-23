#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

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
    Converted,          // .mxr file present
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

    // Default writable cache: AVE_MODELS_DIR, then XDG/HOME fallback.
    // Bundled read-only install assets are discovered separately.
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
    //   converted > downloaded
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
    bool convertToMiGraphX(const std::string& modelId,
                           const ModelProgressCb& progressCb,
                           const ModelStateCb&    stateCb,
                           std::string&           error,
                           ModelPrecision         compilePrecision = ModelPrecision::Fp16);

    // ── Auto-compile ──────────────────────────────────────────────

    // Automatically compile a model for inference if not already compiled.
    // Returns the path to the compiled .mxr, or nullopt on failure.
    // First-time compilation requires inputWidth and inputHeight so the
    // artifact matches the real video frame size. Without dimensions this
    // only returns an already-compiled .mxr. Int8 compilation additionally
    // requires a calibration video path so representative frames can be
    // sampled during quantization. compileBatch lets runtime callers request
    // a batched artifact for tiled inference without changing the batch-1
    // default used elsewhere.
    std::optional<std::string> autoCompileForInference(
        const std::string& modelId,
        std::string& error,
        std::optional<std::int64_t> inputWidth = std::nullopt,
        std::optional<std::int64_t> inputHeight = std::nullopt,
        ModelPrecision compilePrecision = ModelPrecision::Fp16,
        int compileBatch = 1,
        std::optional<std::string> calibrationVideoPath = std::nullopt);

    // Returns an already-compiled .mxr artifact only when its manifest matches
    // the current effective MiGraphX/MiOpen runtime identity.
    std::optional<std::string> validatedCompiledArtifactPath(
        const std::string& modelId,
        ModelPrecision compilePrecision,
        std::optional<std::int64_t> inputWidth = std::nullopt,
        std::optional<std::int64_t> inputHeight = std::nullopt,
        int compileBatch = 1,
        std::string* validationDetail = nullptr) const;

    // ── UI helpers ───────────────────────────────────────────────

    // Returns a UI-ready label for a model, e.g. "[Compiled] Real-ESRGAN x4"
    std::string modelDropdownLabel(const std::string& modelId) const;

    // Returns sorted list of dropdown labels for a given stage, plus the
    // "ffmpeg fallback" entry as the last item when applicable.
    struct DropdownEntry {
        std::string modelId;
        std::string label;
        bool        inferenceReady;
    };
    std::vector<DropdownEntry> dropdownEntriesForStage(StageKind kind) const;

  private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

}  // namespace ave
