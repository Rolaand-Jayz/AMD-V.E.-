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
#include <unordered_map>

namespace ave {

// ─────────────────────────────────────────────────────────────────
// String helpers
// ─────────────────────────────────────────────────────────────────

std::string toString(ModelPrecision p) {
    switch (p) {
        case ModelPrecision::Fp32: return "fp32";
        case ModelPrecision::Fp16: return "fp16";
        case ModelPrecision::Int8: return "int8";
    }
    return "fp32";
}

std::optional<ModelPrecision> precisionFromString(const std::string& s) {
    if (s == "fp32") { return ModelPrecision::Fp32; }
    if (s == "fp16") { return ModelPrecision::Fp16; }
    if (s == "int8") { return ModelPrecision::Int8; }
    return std::nullopt;
}

std::optional<BackendType> backendTypeFromString(const std::string& s) {
    if (s == "auto")         { return BackendType::Auto;       }
    if (s == "migraphx")     { return BackendType::MiGraphX;    }
    if (s == "ncnn-vulkan"
     || s == "ncnn_vulkan"
     || s == "ncnn")         { return BackendType::NcnnVulkan;  }
    return std::nullopt;
}

// ─────────────────────────────────────────────────────────────────
// AppSettings::effectivePrecisionFor
// ─────────────────────────────────────────────────────────────────
ModelPrecision AppSettings::effectivePrecisionFor(
        const std::string& modelId,
        ModelPrecision     catalogPrecision) const {
    // 1. Per-model override takes priority
    const auto it = modelPrecisionOverrides.find(modelId);
    if (it != modelPrecisionOverrides.end()) {
        return it->second;
    }
    // 2. Global setting
    // If the global is still Fp32 (the default) and the catalog says
    // Fp16, honour the catalogue's more precise recommendation.
    // Otherwise the user's explicit global selection wins.
    if (globalQuantization == ModelPrecision::Fp32
            && catalogPrecision == ModelPrecision::Fp16) {
        return ModelPrecision::Fp16;
    }
    return globalQuantization;
}

// ─────────────────────────────────────────────────────────────────
// AppSettings::clampToEnvironment
// ─────────────────────────────────────────────────────────────────
ModelPrecision AppSettings::clampToEnvironment(
        ModelPrecision requested,
        bool           migraphxAvailable) {
    if (!migraphxAvailable) {
        // Without MiGraphX, compilation is a no-op; precision is irrelevant,
        // but we still return Fp32 as the nearest safe value.
        return ModelPrecision::Fp32;
    }
    // MiGraphX supports all three presently. Pass through.
    return requested;
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
            if (key == "global_quantization") {
                if (auto opt = precisionFromString(val)) { globalQuantization = *opt; }
            } else if (key == "default_backend") {
                if (auto opt = backendTypeFromString(val)) { defaultBackend = *opt; }
            }
        } else if (currentSection == "Compilation") {
            if (key == "gpu_tuning_by_default") { gpuTuningByDefault = parseBool(val, false); }
            if (key == "exhaustive_tuning")      { exhaustiveTuning   = parseBool(val, false); }
        } else if (currentSection == "Encode") {
            if (key == "codec")           { defaultCodec  = val; }
            if (key == "crf")             { defaultCrf    = parseInt(val, 18); }
            if (key == "preset")          { defaultPreset = val; }
            if (key == "ffmpeg_threads")  { ffmpegThreads = parseInt(val, 0); }
        } else if (currentSection == "Interface") {
            if (key == "verbose_log")     { verboseLog = parseBool(val, false); }
        } else if (currentSection == "ModelPrecision") {
            if (auto opt = precisionFromString(val)) {
                modelPrecisionOverrides[key] = *opt;
            }
        }
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
      << "global_quantization=" << toString(globalQuantization) << "\n"
      << "default_backend="     << toString(defaultBackend)     << "\n\n";

    f << "[Compilation]\n"
      << "gpu_tuning_by_default=" << boolStr(gpuTuningByDefault) << "\n"
      << "exhaustive_tuning="     << boolStr(exhaustiveTuning)   << "\n\n";

    f << "[Encode]\n"
      << "codec="          << defaultCodec   << "\n"
      << "crf="            << defaultCrf     << "\n"
      << "preset="         << defaultPreset  << "\n"
      << "ffmpeg_threads=" << ffmpegThreads  << "\n\n";

    f << "[Interface]\n"
      << "verbose_log=" << boolStr(verboseLog) << "\n\n";

    if (!modelPrecisionOverrides.empty()) {
        f << "[ModelPrecision]\n";
        for (const auto& [id, prec] : modelPrecisionOverrides) {
            f << id << "=" << toString(prec) << "\n";
        }
        f << "\n";
    }
}

}  // namespace ave
