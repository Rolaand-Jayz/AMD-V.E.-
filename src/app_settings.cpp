// ─────────────────────────────────────────────────────────────────
// app_settings.cpp — AppSettings persistence and helpers
// ─────────────────────────────────────────────────────────────────
#include "ave/app_settings.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

namespace ave {

// ─────────────────────────────────────────────────────────────────
// String helpers
// ─────────────────────────────────────────────────────────────────

std::optional<BackendType> backendTypeFromString(const std::string& s) {
    if (s == "auto")         { return BackendType::Auto;       }
    if (s == "migraphx")     { return BackendType::MiGraphX;    }
    if (s == "rocm-hip"
     || s == "rocm_hip"
     || s == "rocmhip"
     || s == "rocm"
     || s == "hip")         { return BackendType::RocmHip;     }
    if (s == "ncnn-vulkan"
     || s == "ncnn_vulkan"
     || s == "ncnn")         { return BackendType::NcnnVulkan;  }
    if (s == "vulkan"
     || s == "vulkan-compute"
     || s == "vulkan_compute") { return BackendType::VulkanCompute; }
    if (s == "vapoursynth"
     || s == "vapourynth"
     || s == "vs")          { return BackendType::VapourSynth; }
    if (s == "glsl"
     || s == "glsl-shader"
     || s == "glsl_shader") { return BackendType::GlslShader; }
    return std::nullopt;
}

// ─────────────────────────────────────────────────────────────────
// INI helpers
// ─────────────────────────────────────────────────────────────────
namespace {

// Trim leading/trailing whitespace
std::string trim(const std::string& s) {
    const auto b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) { return {}; }
    const auto e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

bool parseBool(const std::string& v, bool def) {
    const std::string t = trim(v);
    if (t == "1" || t == "true"  || t == "yes") { return true; }
    if (t == "0" || t == "false" || t == "no")  { return false; }
    return def;
}

int parseInt(const std::string& v, int def) {
    try { return std::stoi(trim(v)); }
    catch (...) { return def; }
}

int clampInt(const int value, const int lo, const int hi) {
    return std::max(lo, std::min(value, hi));
}

std::string boolStr(bool b) { return b ? "true" : "false"; }

}  // namespace

// ─────────────────────────────────────────────────────────────────
// AppSettings::defaultPath
// ─────────────────────────────────────────────────────────────────
std::string AppSettings::defaultPath() {
    const char* home = std::getenv("HOME");
    if (home != nullptr) {
        return std::string(home) + "/.config/ave/settings.ini";
    }
    return "/tmp/ave_settings.ini";
}

// ─────────────────────────────────────────────────────────────────
// AppSettings::load
// ─────────────────────────────────────────────────────────────────
void AppSettings::load(const std::string& pathArg) {
    const std::string path = pathArg.empty() ? defaultPath() : pathArg;

    std::ifstream f(path);
    if (!f.is_open()) {
        // No saved file: keep all defaults – silently OK.
        return;
    }

    // Simple INI parser: [Section] / key=value / # comments
    std::string currentSection;
    std::string line;
    while (std::getline(f, line)) {
        const std::string l = trim(line);
        if (l.empty() || l[0] == '#' || l[0] == ';') { continue; }

        if (l[0] == '[' && l.back() == ']') {
            currentSection = l.substr(1, l.size() - 2);
            continue;
        }

        const auto eq = l.find('=');
        if (eq == std::string::npos) { continue; }
        const std::string key = trim(l.substr(0, eq));
        const std::string val = trim(l.substr(eq + 1));

        if (currentSection == "Inference") {
            if (key == "default_backend") {
                if (auto opt = backendTypeFromString(val)) { defaultBackend = *opt; }
            }
        } else if (currentSection == "Encode") {
            if (key == "codec")           { defaultCodec  = val; }
            if (key == "profile")         { defaultProfile = val; }
            if (key == "crf")             { defaultCrf    = parseInt(val, 18); }
            if (key == "preset")          { defaultPreset = val; }
            if (key == "ffmpeg_threads")  { ffmpegThreads = parseInt(val, 0); }
        } else if (currentSection == "Workflow") {
            if (key == "default_dry_run")            { defaultDryRun = parseBool(val, false); }
            if (key == "preview_duration_sec")       { defaultPreviewDurationSec = parseInt(val, 10); }
            if (key == "preview_frame_interval")     { previewFrameInterval = parseInt(val, 15); }
            if (key == "auto_fill_output_path")      { autoFillOutputPath = parseBool(val, true); }
            if (key == "output_suffix")              { outputSuffix = val; }
            if (key == "remember_last_paths")        { rememberLastPaths = parseBool(val, true); }
            if (key == "last_input_path")            { lastInputPath = val; }
            if (key == "last_output_path")           { lastOutputPath = val; }
            if (key == "confirm_before_run")         { confirmBeforeRun = parseBool(val, false); }
            if (key == "confirm_before_clear_pipeline") {
                confirmBeforeClearPipeline = parseBool(val, true);
            }
        } else if (currentSection == "Interface") {
            if (key == "auto_scroll_log") { autoScrollLog = parseBool(val, true); }
            if (key == "verbose_log")     { verboseLog = parseBool(val, false); }
        }
    }

    defaultCrf = clampInt(defaultCrf, 0, 63);
    ffmpegThreads = std::max(0, ffmpegThreads);
    defaultPreviewDurationSec = clampInt(defaultPreviewDurationSec, 1, 300);
    previewFrameInterval = clampInt(previewFrameInterval, 1, 240);
    if (outputSuffix.empty()) {
        outputSuffix = "_enhanced";
    }
}

// ─────────────────────────────────────────────────────────────────
// AppSettings::save
// ─────────────────────────────────────────────────────────────────
void AppSettings::save(const std::string& pathArg) const {
    const std::string path = pathArg.empty() ? defaultPath() : pathArg;

    // Ensure parent directory exists
    std::error_code ec;
    std::filesystem::create_directories(
        std::filesystem::path(path).parent_path(), ec);

    std::ofstream f(path, std::ios::trunc);
    if (!f.is_open()) {
        std::cerr << "[AppSettings] Cannot save to " << path << std::endl;
        return;
    }

    f << "# AMD Video Enhancer — user settings\n"
      << "# Generated automatically. Edit with care.\n\n";

    f << "[Inference]\n"
      << "default_backend=" << toString(defaultBackend) << "\n\n";

    f << "[Encode]\n"
      << "codec="          << defaultCodec   << "\n"
      << "profile="        << defaultProfile << "\n"
      << "crf="            << defaultCrf     << "\n"
      << "preset="         << defaultPreset  << "\n"
      << "ffmpeg_threads=" << ffmpegThreads  << "\n\n";

    f << "[Workflow]\n"
      << "default_dry_run=" << boolStr(defaultDryRun) << "\n"
      << "preview_duration_sec=" << defaultPreviewDurationSec << "\n"
      << "preview_frame_interval=" << previewFrameInterval << "\n"
      << "auto_fill_output_path=" << boolStr(autoFillOutputPath) << "\n"
      << "output_suffix=" << outputSuffix << "\n"
      << "remember_last_paths=" << boolStr(rememberLastPaths) << "\n"
      << "last_input_path=" << lastInputPath << "\n"
      << "last_output_path=" << lastOutputPath << "\n"
      << "confirm_before_run=" << boolStr(confirmBeforeRun) << "\n"
      << "confirm_before_clear_pipeline=" << boolStr(confirmBeforeClearPipeline) << "\n\n";

    f << "[Interface]\n"
      << "auto_scroll_log=" << boolStr(autoScrollLog) << "\n"
      << "verbose_log=" << boolStr(verboseLog) << "\n\n";
}

}  // namespace ave
