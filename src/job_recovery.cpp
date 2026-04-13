#include "ave/job_recovery.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <optional>
#include <sstream>
#include <system_error>
#include <unordered_map>
#include <vector>

#include "ave/backend.hpp"
#include "ave/filter_catalog.hpp"
#include "ave/stage.hpp"

namespace ave {
namespace {

std::string trimRecovery(const std::string& input) {
    std::size_t begin = 0;
    while (begin < input.size() &&
           std::isspace(static_cast<unsigned char>(input[begin])) != 0) {
        ++begin;
    }

    std::size_t end = input.size();
    while (end > begin &&
           std::isspace(static_cast<unsigned char>(input[end - 1])) != 0) {
        --end;
    }
    return input.substr(begin, end - begin);
}

std::optional<bool> parseBoolRecovery(const std::string& value) {
    const std::string trimmed = trimRecovery(value);
    if (trimmed == "true" || trimmed == "1" || trimmed == "yes") {
        return true;
    }
    if (trimmed == "false" || trimmed == "0" || trimmed == "no") {
        return false;
    }
    return std::nullopt;
}

std::optional<int> parseIntRecovery(const std::string& value) {
    try {
        const std::string trimmed = trimRecovery(value);
        std::size_t consumed = 0;
        const int parsed = std::stoi(trimmed, &consumed);
        if (consumed != trimmed.size()) {
            return std::nullopt;
        }
        return parsed;
    } catch (...) {
        return std::nullopt;
    }
}

std::string quoteRecovery(const std::string& value) {
    std::ostringstream out;
    out << std::quoted(value);
    return out.str();
}

std::optional<std::string> unquoteRecovery(const std::string& value) {
    std::istringstream input(value);
    std::string out;
    if ((input >> std::quoted(out)).fail()) {
        return std::nullopt;
    }
    return out;
}

std::string backendToStringRecovery(const BackendType backend) {
    return toString(backend);
}

std::optional<BackendType> backendFromStringRecovery(const std::string& value) {
    if (value == "auto") {
        return BackendType::Auto;
    }
    if (value == "migraphx") {
        return BackendType::MiGraphX;
    }
    if (value == "rocm-hip" || value == "rocm_hip" ||
        value == "rocmhip" || value == "rocm" || value == "hip") {
        return BackendType::RocmHip;
    }
    if (value == "vulkan-compute" || value == "vulkan_compute" ||
        value == "vulkan") {
        return BackendType::VulkanCompute;
    }
    if (value == "ncnn-vulkan" || value == "ncnn_vulkan" || value == "ncnn") {
        return BackendType::NcnnVulkan;
    }
    if (value == "vapoursynth" || value == "vapourynth" || value == "vs") {
        return BackendType::VapourSynth;
    }
    if (value == "glsl-shader" || value == "glsl_shader" || value == "glsl") {
        return BackendType::GlslShader;
    }
    return std::nullopt;
}

std::string stageToSpecRecovery(const EnhancementStage& stage) {
    std::vector<std::string> keys;
    keys.reserve(stage.params.size());
    for (const auto& [key, _] : stage.params) {
        keys.push_back(key);
    }
    std::sort(keys.begin(), keys.end());

    std::ostringstream spec;
    spec << toString(stage.kind);
    if (!keys.empty()) {
        spec << ":";
    }
    for (std::size_t index = 0; index < keys.size(); ++index) {
        if (index > 0) {
            spec << ",";
        }
        const auto it = stage.params.find(keys[index]);
        spec << keys[index] << "=" << parameterValueToString(it->second);
    }
    return spec.str();
}

std::string filterToSpecRecovery(const ActiveFilter& filter) {
    std::vector<std::string> keys;
    keys.reserve(filter.paramValues.size());
    for (const auto& [key, _] : filter.paramValues) {
        keys.push_back(key);
    }
    std::sort(keys.begin(), keys.end());

    std::ostringstream spec;
    spec << filter.id;
    if (!keys.empty()) {
        spec << ":";
    }
    spec << std::setprecision(12);
    for (std::size_t index = 0; index < keys.size(); ++index) {
        if (index > 0) {
            spec << ",";
        }
        const auto it = filter.paramValues.find(keys[index]);
        spec << keys[index] << "=" << it->second;
    }
    return spec.str();
}

std::optional<ActiveFilter> filterFromSpecRecovery(const std::string& spec,
                                                   std::string& error) {
    const std::size_t colon = spec.find(':');
    ActiveFilter filter;
    filter.id = colon == std::string::npos ? spec : spec.substr(0, colon);
    filter.enabled = true;
    if (filter.id.empty()) {
        error = "Recovery filter entry is missing an identifier.";
        return std::nullopt;
    }
    if (colon == std::string::npos) {
        error.clear();
        return filter;
    }

    const std::string params = spec.substr(colon + 1);
    std::size_t start = 0;
    while (start <= params.size()) {
        std::size_t end = params.find(',', start);
        if (end == std::string::npos) {
            end = params.size();
        }
        const std::string assignment = params.substr(start, end - start);
        if (!assignment.empty()) {
            const std::size_t eq = assignment.find('=');
            if (eq == std::string::npos || eq == 0 || eq == assignment.size() - 1) {
                error = "Malformed recovery filter parameter: " + assignment;
                return std::nullopt;
            }
            const std::string key = assignment.substr(0, eq);
            const std::optional<int> intValue =
                parseIntRecovery(assignment.substr(eq + 1));
            if (intValue.has_value()) {
                filter.paramValues[key] = static_cast<double>(*intValue);
            } else {
                try {
                    filter.paramValues[key] =
                        std::stod(trimRecovery(assignment.substr(eq + 1)));
                } catch (...) {
                    error = "Recovery filter parameter is not numeric: " + assignment;
                    return std::nullopt;
                }
            }
        }
        if (end == params.size()) {
            break;
        }
        start = end + 1;
    }

    error.clear();
    return filter;
}

std::string recoveryTimestampUtcNow() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm utc{};
#if defined(_WIN32)
    gmtime_s(&utc, &tt);
#else
    gmtime_r(&tt, &utc);
#endif
    char buffer[32];
    if (std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &utc) == 0) {
        return {};
    }
    return buffer;
}

}  // namespace

JobRecoveryStore::JobRecoveryStore(std::string path) : path_(std::move(path)) {
    if (path_.empty()) {
        path_ = defaultPath();
    }
}

std::string JobRecoveryStore::path() const {
    return path_;
}

bool JobRecoveryStore::save(const RecoveredJobState& state,
                            std::string& error) const {
    std::error_code filesystemError;
    std::filesystem::create_directories(
        std::filesystem::path(path_).parent_path(), filesystemError);
    if (filesystemError) {
        error = "Unable to create recovery directory: " +
                filesystemError.message();
        return false;
    }

    std::ofstream output(path_, std::ios::trunc);
    if (!output.is_open()) {
        error = "Unable to open recovery store for writing: " + path_;
        return false;
    }

    const std::string startedAt =
        state.startedAtUtc.empty() ? recoveryTimestampUtcNow() : state.startedAtUtc;

    output << "schema_version 1\n";
    output << "started_at_utc " << quoteRecovery(startedAt) << "\n";
    output << "input_path " << quoteRecovery(state.job.inputPath) << "\n";
    output << "output_path " << quoteRecovery(state.job.outputPath) << "\n";
    output << "backend "
           << quoteRecovery(backendToStringRecovery(state.job.requestedBackend))
           << "\n";
    output << "codec " << quoteRecovery(state.job.encode.codec) << "\n";
    output << "profile " << quoteRecovery(state.job.encode.profile) << "\n";
    output << "crf " << state.job.encode.crf << "\n";
    output << "preset " << quoteRecovery(state.job.encode.preset) << "\n";
    output << "threads " << state.job.encode.threads << "\n";
    output << "dry_run " << (state.job.dryRun ? "true" : "false") << "\n";
    output << "preview_mode "
           << (state.job.previewMode ? "true" : "false") << "\n";
    output << "preview_duration_sec "
           << std::setprecision(12) << state.job.previewDurationSec << "\n";
    output << "preview_frame_interval " << state.job.previewFrameInterval << "\n";

    for (const auto& stage : state.job.requestedStages) {
        output << "stage " << quoteRecovery(stageToSpecRecovery(stage)) << "\n";
    }
    for (const auto& filter : state.job.catalogFilters) {
        output << "filter " << quoteRecovery(filterToSpecRecovery(filter)) << "\n";
    }

    if (!output.good()) {
        error = "Unable to finish writing recovery store: " + path_;
        return false;
    }

    error.clear();
    return true;
}

std::optional<RecoveredJobState> JobRecoveryStore::load(std::string& error) const {
    std::ifstream input(path_);
    if (!input.is_open()) {
        error.clear();
        return std::nullopt;
    }

    RecoveredJobState state;
    for (std::string line; std::getline(input, line);) {
        const std::string trimmed = trimRecovery(line);
        if (trimmed.empty() || trimmed[0] == '#') {
            continue;
        }

        const std::size_t space = trimmed.find(' ');
        if (space == std::string::npos) {
            error = "Malformed recovery line: " + trimmed;
            return std::nullopt;
        }

        const std::string key = trimmed.substr(0, space);
        const std::string value = trimRecovery(trimmed.substr(space + 1));
        if (key == "schema_version") {
            const auto schema = parseIntRecovery(value);
            if (!schema.has_value() || *schema != 1) {
                error = "Unsupported recovery schema version.";
                return std::nullopt;
            }
            continue;
        }
        if (key == "started_at_utc") {
            const auto parsed = unquoteRecovery(value);
            if (!parsed.has_value()) {
                error = "Recovery timestamp is malformed.";
                return std::nullopt;
            }
            state.startedAtUtc = *parsed;
            continue;
        }
        if (key == "input_path") {
            const auto parsed = unquoteRecovery(value);
            if (!parsed.has_value()) {
                error = "Recovery input path is malformed.";
                return std::nullopt;
            }
            state.job.inputPath = *parsed;
            continue;
        }
        if (key == "output_path") {
            const auto parsed = unquoteRecovery(value);
            if (!parsed.has_value()) {
                error = "Recovery output path is malformed.";
                return std::nullopt;
            }
            state.job.outputPath = *parsed;
            continue;
        }
        if (key == "backend") {
            const auto parsed = unquoteRecovery(value);
            if (!parsed.has_value()) {
                error = "Recovery backend value is malformed.";
                return std::nullopt;
            }
            const auto backend = backendFromStringRecovery(*parsed);
            if (!backend.has_value()) {
                error = "Recovery backend is unknown: " + *parsed;
                return std::nullopt;
            }
            state.job.requestedBackend = *backend;
            continue;
        }
        if (key == "codec") {
            const auto parsed = unquoteRecovery(value);
            if (!parsed.has_value()) {
                error = "Recovery codec is malformed.";
                return std::nullopt;
            }
            state.job.encode.codec = *parsed;
            continue;
        }
        if (key == "profile") {
            const auto parsed = unquoteRecovery(value);
            if (!parsed.has_value()) {
                error = "Recovery profile is malformed.";
                return std::nullopt;
            }
            state.job.encode.profile = *parsed;
            continue;
        }
        if (key == "crf") {
            const auto parsed = parseIntRecovery(value);
            if (!parsed.has_value()) {
                error = "Recovery CRF is malformed.";
                return std::nullopt;
            }
            state.job.encode.crf = *parsed;
            continue;
        }
        if (key == "preset") {
            const auto parsed = unquoteRecovery(value);
            if (!parsed.has_value()) {
                error = "Recovery preset is malformed.";
                return std::nullopt;
            }
            state.job.encode.preset = *parsed;
            continue;
        }
        if (key == "threads") {
            const auto parsed = parseIntRecovery(value);
            if (!parsed.has_value()) {
                error = "Recovery thread count is malformed.";
                return std::nullopt;
            }
            state.job.encode.threads = *parsed;
            continue;
        }
        if (key == "dry_run") {
            const auto parsed = parseBoolRecovery(value);
            if (!parsed.has_value()) {
                error = "Recovery dry-run flag is malformed.";
                return std::nullopt;
            }
            state.job.dryRun = *parsed;
            continue;
        }
        if (key == "preview_mode") {
            const auto parsed = parseBoolRecovery(value);
            if (!parsed.has_value()) {
                error = "Recovery preview flag is malformed.";
                return std::nullopt;
            }
            state.job.previewMode = *parsed;
            continue;
        }
        if (key == "preview_duration_sec") {
            try {
                state.job.previewDurationSec = std::stod(trimRecovery(value));
            } catch (...) {
                error = "Recovery preview duration is malformed.";
                return std::nullopt;
            }
            continue;
        }
        if (key == "preview_frame_interval") {
            const auto parsed = parseIntRecovery(value);
            if (!parsed.has_value()) {
                error = "Recovery preview frame interval is malformed.";
                return std::nullopt;
            }
            state.job.previewFrameInterval = *parsed;
            continue;
        }
        if (key == "stage") {
            const auto parsed = unquoteRecovery(value);
            if (!parsed.has_value()) {
                error = "Recovery stage entry is malformed.";
                return std::nullopt;
            }
            std::string parseError;
            const auto stage = ave::parseStageSpec(*parsed, parseError);
            if (!stage.has_value()) {
                error = "Unable to parse recovery stage: " + parseError;
                return std::nullopt;
            }
            state.job.requestedStages.push_back(*stage);
            continue;
        }
        if (key == "filter") {
            const auto parsed = unquoteRecovery(value);
            if (!parsed.has_value()) {
                error = "Recovery filter entry is malformed.";
                return std::nullopt;
            }
            std::string parseError;
            const auto filter = filterFromSpecRecovery(*parsed, parseError);
            if (!filter.has_value()) {
                error = parseError;
                return std::nullopt;
            }
            state.job.catalogFilters.push_back(*filter);
            continue;
        }

        error = "Unknown recovery key: " + key;
        return std::nullopt;
    }

    if (state.job.inputPath.empty() || state.job.outputPath.empty()) {
        error = "Recovery snapshot is incomplete.";
        return std::nullopt;
    }

    if (state.job.previewFrameInterval <= 0) {
        state.job.previewFrameInterval = 15;
    }

    error.clear();
    return state;
}

bool JobRecoveryStore::clear(std::string& error) const {
    std::error_code filesystemError;
    const bool removed = std::filesystem::remove(path_, filesystemError);
    if (filesystemError) {
        error = "Unable to clear recovery store: " + filesystemError.message();
        return false;
    }
    static_cast<void>(removed);
    error.clear();
    return true;
}

std::string JobRecoveryStore::defaultPath() {
    const char* home = std::getenv("HOME");
    if (home != nullptr) {
        return std::string(home) + "/.config/ave/job_recovery.state";
    }
    return "/tmp/ave_job_recovery.state";
}

}  // namespace ave
