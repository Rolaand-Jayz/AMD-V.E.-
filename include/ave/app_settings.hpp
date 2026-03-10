#pragma once

#include <optional>
#include <string>

#include "ave/backend.hpp"
#include "ave/model_catalog.hpp"

namespace ave {

// ─────────────────────────────────────────────────────────────────
// AppSettings — global user preferences
// ─────────────────────────────────────────────────────────────────
// Persisted as a plain INI file at ~/.config/ave/settings.ini.
// All fields have sensible defaults so the app works out-of-the-box
// without any saved file.
// ─────────────────────────────────────────────────────────────────
struct AppSettings {

    // ── Inference ────────────────────────────────────────────────
    // Default backend selection shown in the job UI on startup.
    BackendType     defaultBackend     = BackendType::Auto;

    // ── Encode defaults ──────────────────────────────────────────
    std::string defaultCodec   = "libx264";
    std::string defaultProfile;
    int         defaultCrf     = 18;
    std::string defaultPreset  = "medium";
    int         ffmpegThreads  = 0;   // 0 = let FFmpeg auto-detect

    // ── Workflow defaults ────────────────────────────────────────
    bool        defaultDryRun            = false;
    int         defaultPreviewDurationSec = 10;
    int         previewFrameInterval     = 15;
    bool        autoFillOutputPath       = true;
    std::string outputSuffix             = "_enhanced";
    bool        rememberLastPaths        = true;
    std::string lastInputPath;
    std::string lastOutputPath;
    bool        confirmBeforeRun         = false;
    bool        confirmBeforeClearPipeline = true;

    // ── Interface ────────────────────────────────────────────────
    bool autoScrollLog = true;
    bool verboseLog = false;

    // ── Persistence ──────────────────────────────────────────────
    // Default path: ~/.config/ave/settings.ini
    // The parent directory is created on first save.
    void load(const std::string& path = "");
    void save(const std::string& path = "") const;

    static std::string defaultPath();
};

// backendTypeFromString is the inverse of the existing toString(BackendType).
std::optional<BackendType> backendTypeFromString(const std::string& s);

}  // namespace ave
