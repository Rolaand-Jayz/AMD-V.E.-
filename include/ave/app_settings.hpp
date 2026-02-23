#pragma once

#include <optional>
#include <string>
#include <unordered_map>

#include "ave/backend.hpp"
#include "ave/model_catalog.hpp"

namespace ave {

// ─────────────────────────────────────────────────────────────────
// AppSettings — global + per-model user preferences
// ─────────────────────────────────────────────────────────────────
// Persisted as a plain INI file at ~/.config/ave/settings.ini.
// All fields have sensible defaults so the app works out-of-the-box
// without any saved file.
//
// Priority hierarchy for compilation precision:
//   per-model override  →  globalQuantization  →  catalog default
//
// Environment compatibility:
//   Fp16 / Int8  require MiGraphX (ROCm).
//   If MiGraphX is unavailable, both fall back to Fp32.
// ─────────────────────────────────────────────────────────────────
struct AppSettings {

    // ── Inference ────────────────────────────────────────────────
    // Global default quantization used when compiling any model that
    // does not have an explicit per-model override.
    ModelPrecision  globalQuantization = ModelPrecision::Fp32;

    // Default backend selection shown in the job UI on startup.
    BackendType     defaultBackend     = BackendType::Auto;

    // ── Compilation ──────────────────────────────────────────────
    // Pass --gpu to migraphx-driver by default.  Produces faster
    // inference kernels but the first compile is significantly longer.
    bool gpuTuningByDefault = false;

    // Enable exhaustive kernel search during GPU tuning (extremely slow;
    // only available in recent MiGraphX versions).
    bool exhaustiveTuning   = false;

    // ── Encode defaults ──────────────────────────────────────────
    std::string defaultCodec   = "libx264";
    int         defaultCrf     = 18;
    std::string defaultPreset  = "medium";
    int         ffmpegThreads  = 0;   // 0 = let FFmpeg auto-detect

    // ── Interface ────────────────────────────────────────────────
    bool verboseLog = false;

    // ── Per-model precision overrides ────────────────────────────
    // Keyed by model id (ModelEntry::id).  Present = user override;
    // absent = fall through to globalQuantization.
    std::unordered_map<std::string, ModelPrecision> modelPrecisionOverrides;

    // ── Helpers ──────────────────────────────────────────────────

    // Returns the effective compile precision for a given model by
    // applying the full priority chain:
    //   per-model override → globalQuantization → catalogPrecision
    ModelPrecision effectivePrecisionFor(
        const std::string& modelId,
        ModelPrecision     catalogPrecision) const;

    // Clamp a requested precision to the capabilities of the current
    // execution environment.  Fp16 and Int8 both require MiGraphX; if
    // unavailable the function walks downward toward Fp32.
    static ModelPrecision clampToEnvironment(
        ModelPrecision requested,
        bool           migraphxAvailable);

    // ── Persistence ──────────────────────────────────────────────
    // Default path: ~/.config/ave/settings.ini
    // The parent directory is created on first save.
    void load(const std::string& path = "");
    void save(const std::string& path = "") const;

    static std::string defaultPath();
};

// ── String serialisation helpers ─────────────────────────────────
// Used internally by AppSettings::load/save and the settings UI.
std::string toString(ModelPrecision p);
std::optional<ModelPrecision> precisionFromString(const std::string& s);

// backendTypeFromString is the inverse of the existing toString(BackendType).
std::optional<BackendType> backendTypeFromString(const std::string& s);

}  // namespace ave
