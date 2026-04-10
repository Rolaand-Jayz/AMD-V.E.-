#include "ave/job_queue.hpp"

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
#include <string>
#include <system_error>
#include <vector>

#include "ave/app_settings.hpp"
#include "ave/backend.hpp"
#include "ave/filter_catalog.hpp"
#include "ave/stage.hpp"

namespace ave {
namespace {

std::string trimQueue(const std::string& input) {
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

std::optional<bool> parseBoolQueue(const std::string& value) {
    const std::string trimmed = trimQueue(value);
    if (trimmed == "true" || trimmed == "1" || trimmed == "yes") {
        return true;
    }
    if (trimmed == "false" || trimmed == "0" || trimmed == "no") {
        return false;
    }
    return std::nullopt;
}

std::optional<int> parseIntQueue(const std::string& value) {
    try {
        const std::string trimmed = trimQueue(value);
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

std::string quoteQueue(const std::string& value) {
    std::ostringstream out;
    out << std::quoted(value);
    return out.str();
}

std::optional<std::string> unquoteQueue(const std::string& value) {
    std::istringstream input(value);
    std::string out;
    if ((input >> std::quoted(out)).fail()) {
        return std::nullopt;
    }
    return out;
}

std::string stageToSpecQueue(const EnhancementStage& stage) {
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

std::string filterToSpecQueue(const ActiveFilter& filter) {
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

std::optional<ActiveFilter> filterFromSpecQueue(const std::string& spec,
                                                std::string& error) {
    const std::size_t colon = spec.find(':');
    ActiveFilter filter;
    filter.id = colon == std::string::npos ? spec : spec.substr(0, colon);
    filter.enabled = true;
    if (filter.id.empty()) {
        error = "Queue filter entry is missing an identifier.";
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
                error = "Malformed queue filter parameter: " + assignment;
                return std::nullopt;
            }
            const std::string key = assignment.substr(0, eq);
            const std::optional<int> intValue =
                parseIntQueue(assignment.substr(eq + 1));
            if (intValue.has_value()) {
                filter.paramValues[key] = static_cast<double>(*intValue);
            } else {
                try {
                    filter.paramValues[key] =
                        std::stod(trimQueue(assignment.substr(eq + 1)));
                } catch (...) {
                    error = "Queue filter parameter is not numeric: " + assignment;
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

std::string timestampUtcNow() {
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

std::string lowercase(const std::string& input) {
    std::string lowered = input;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                   [](const unsigned char ch) {
                       return static_cast<char>(std::tolower(ch));
                   });
    return lowered;
}

bool containsAny(const std::string& haystack,
                 const std::vector<std::string>& needles) {
    for (const auto& needle : needles) {
        if (haystack.find(needle) != std::string::npos) {
            return true;
        }
    }
    return false;
}

std::optional<bool> classifyStructuredQueueError(
        const std::string& error) {
    const std::string trimmed = trimQueue(error);
    if (trimmed.size() < 3 || trimmed.front() != '[') {
        return std::nullopt;
    }

    const std::size_t closing = trimmed.find(']');
    if (closing == std::string::npos || closing <= 1) {
        return std::nullopt;
    }

    const std::string kindToken = lowercase(trimmed.substr(1, closing - 1));
    if (kindToken == "runtimefailure" ||
        kindToken == "ffmpeginteropfailure") {
        return true;
    }
    if (kindToken == "modelincompatible" ||
        kindToken == "compilefailure" ||
        kindToken == "artifactinvalid" ||
        kindToken == "vulkandevicefailure" ||
        kindToken == "interopfailure" ||
        kindToken == "synchazard") {
        return false;
    }
    return std::nullopt;
}

void writeVideoJob(std::ofstream& output, const VideoJob& job) {
    output << "input_path " << quoteQueue(job.inputPath) << "\n";
    output << "output_path " << quoteQueue(job.outputPath) << "\n";
    output << "backend " << quoteQueue(toString(job.requestedBackend)) << "\n";
    output << "codec " << quoteQueue(job.encode.codec) << "\n";
    output << "profile " << quoteQueue(job.encode.profile) << "\n";
    output << "crf " << job.encode.crf << "\n";
    output << "preset " << quoteQueue(job.encode.preset) << "\n";
    output << "threads " << job.encode.threads << "\n";
    output << "dry_run " << (job.dryRun ? "true" : "false") << "\n";
    output << "preview_mode " << (job.previewMode ? "true" : "false") << "\n";
    output << "preview_duration_sec " << std::setprecision(12)
           << job.previewDurationSec << "\n";
    output << "preview_frame_interval " << job.previewFrameInterval << "\n";
    for (const auto& stage : job.requestedStages) {
        output << "stage " << quoteQueue(stageToSpecQueue(stage)) << "\n";
    }
    for (const auto& filter : job.catalogFilters) {
        output << "filter " << quoteQueue(filterToSpecQueue(filter)) << "\n";
    }
}

bool readVideoJobField(const std::string& key,
                       const std::string& value,
                       VideoJob& job,
                       std::string& error) {
    if (key == "input_path") {
        const auto parsed = unquoteQueue(value);
        if (!parsed.has_value()) {
            error = "Queue input path is malformed.";
            return false;
        }
        job.inputPath = *parsed;
        return true;
    }
    if (key == "output_path") {
        const auto parsed = unquoteQueue(value);
        if (!parsed.has_value()) {
            error = "Queue output path is malformed.";
            return false;
        }
        job.outputPath = *parsed;
        return true;
    }
    if (key == "backend") {
        const auto parsed = unquoteQueue(value);
        if (!parsed.has_value()) {
            error = "Queue backend value is malformed.";
            return false;
        }
        const auto backend = backendTypeFromString(*parsed);
        if (!backend.has_value()) {
            error = "Queue backend is unknown: " + *parsed;
            return false;
        }
        job.requestedBackend = *backend;
        return true;
    }
    if (key == "codec") {
        const auto parsed = unquoteQueue(value);
        if (!parsed.has_value()) {
            error = "Queue codec is malformed.";
            return false;
        }
        job.encode.codec = *parsed;
        return true;
    }
    if (key == "profile") {
        const auto parsed = unquoteQueue(value);
        if (!parsed.has_value()) {
            error = "Queue profile is malformed.";
            return false;
        }
        job.encode.profile = *parsed;
        return true;
    }
    if (key == "crf") {
        const auto parsed = parseIntQueue(value);
        if (!parsed.has_value()) {
            error = "Queue CRF is malformed.";
            return false;
        }
        job.encode.crf = *parsed;
        return true;
    }
    if (key == "preset") {
        const auto parsed = unquoteQueue(value);
        if (!parsed.has_value()) {
            error = "Queue preset is malformed.";
            return false;
        }
        job.encode.preset = *parsed;
        return true;
    }
    if (key == "threads") {
        const auto parsed = parseIntQueue(value);
        if (!parsed.has_value()) {
            error = "Queue thread count is malformed.";
            return false;
        }
        job.encode.threads = *parsed;
        return true;
    }
    if (key == "dry_run") {
        const auto parsed = parseBoolQueue(value);
        if (!parsed.has_value()) {
            error = "Queue dry-run flag is malformed.";
            return false;
        }
        job.dryRun = *parsed;
        return true;
    }
    if (key == "preview_mode") {
        const auto parsed = parseBoolQueue(value);
        if (!parsed.has_value()) {
            error = "Queue preview flag is malformed.";
            return false;
        }
        job.previewMode = *parsed;
        return true;
    }
    if (key == "preview_duration_sec") {
        try {
            job.previewDurationSec = std::stod(trimQueue(value));
        } catch (...) {
            error = "Queue preview duration is malformed.";
            return false;
        }
        return true;
    }
    if (key == "preview_frame_interval") {
        const auto parsed = parseIntQueue(value);
        if (!parsed.has_value()) {
            error = "Queue preview frame interval is malformed.";
            return false;
        }
        job.previewFrameInterval = *parsed;
        return true;
    }
    if (key == "stage") {
        const auto parsed = unquoteQueue(value);
        if (!parsed.has_value()) {
            error = "Queue stage entry is malformed.";
            return false;
        }
        std::string parseError;
        const auto stage = parseStageSpec(*parsed, parseError);
        if (!stage.has_value()) {
            error = "Unable to parse queued stage: " + parseError;
            return false;
        }
        job.requestedStages.push_back(*stage);
        return true;
    }
    if (key == "filter") {
        const auto parsed = unquoteQueue(value);
        if (!parsed.has_value()) {
            error = "Queue filter entry is malformed.";
            return false;
        }
        std::string parseError;
        const auto filter = filterFromSpecQueue(*parsed, parseError);
        if (!filter.has_value()) {
            error = parseError;
            return false;
        }
        job.catalogFilters.push_back(*filter);
        return true;
    }
    return false;
}

}  // namespace

std::string toString(const QueuedJobStatus status) {
    switch (status) {
        case QueuedJobStatus::Pending: return "pending";
        case QueuedJobStatus::Running: return "running";
        case QueuedJobStatus::RetryableFailure: return "retryable_failure";
        case QueuedJobStatus::Failed: return "failed";
        case QueuedJobStatus::Succeeded: return "succeeded";
        case QueuedJobStatus::Cancelled: return "cancelled";
    }
    return "unknown";
}

std::optional<QueuedJobStatus> queuedJobStatusFromString(const std::string& value) {
    if (value == "pending") {
        return QueuedJobStatus::Pending;
    }
    if (value == "running") {
        return QueuedJobStatus::Running;
    }
    if (value == "retryable_failure") {
        return QueuedJobStatus::RetryableFailure;
    }
    if (value == "failed") {
        return QueuedJobStatus::Failed;
    }
    if (value == "succeeded") {
        return QueuedJobStatus::Succeeded;
    }
    if (value == "cancelled") {
        return QueuedJobStatus::Cancelled;
    }
    return std::nullopt;
}

bool isTerminalQueueStatus(const QueuedJobStatus status) {
    return status == QueuedJobStatus::Failed ||
           status == QueuedJobStatus::Succeeded ||
           status == QueuedJobStatus::Cancelled;
}

bool isRetryableQueueFailure(const std::string& error) {
    const std::string lowered = lowercase(error);
    if (lowered.empty()) {
        return false;
    }

    if (const auto structured = classifyStructuredQueueError(error); structured.has_value()) {
        return *structured;
    }

    const std::vector<std::string> fatalMarkers{
        "modelincompatible", "compilefailure", "artifactinvalid",
        "input path is required", "output path is required", "codec is required",
        "preset is required", "not downloaded", "cannot compile", "unsupported",
        "malformed", "custom vapoursynth script not found", "does not exist"
    };
    if (containsAny(lowered, fatalMarkers)) {
        return false;
    }

    const std::vector<std::string> retryableMarkers{
        "runtimefailure", "runtime failure", "ffmpeginteropfailure",
        "ffmpeg encode pipe exited", "device", "driver", "kfd", "hip",
        "rocm", "vulkan", "timed out", "timeout", "temporarily unavailable",
        "resource busy", "broken pipe", "interrupted", "stopped while processing",
        "cancelled by signal"
    };
    return containsAny(lowered, retryableMarkers);
}

std::string generateQueueJobId() {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    const auto micros =
        std::chrono::duration_cast<std::chrono::microseconds>(now).count();
    std::ostringstream out;
    out << "job-" << micros;
    return out.str();
}

JobQueueStore::JobQueueStore(std::string path) : path_(std::move(path)) {
    if (path_.empty()) {
        path_ = defaultPath();
    }
}

std::string JobQueueStore::path() const {
    return path_;
}

bool JobQueueStore::save(const std::vector<QueuedJobRecord>& jobs,
                         std::string& error) const {
    std::error_code filesystemError;
    std::filesystem::create_directories(
        std::filesystem::path(path_).parent_path(), filesystemError);
    if (filesystemError) {
        error = "Unable to create queue directory: " + filesystemError.message();
        return false;
    }

    std::ofstream output(path_, std::ios::trunc);
    if (!output.is_open()) {
        error = "Unable to open queue store for writing: " + path_;
        return false;
    }

    output << "schema_version 1\n";
    for (const auto& job : jobs) {
        output << "job_begin\n";
        output << "id " << quoteQueue(job.id.empty() ? generateQueueJobId() : job.id) << "\n";
        output << "status " << quoteQueue(toString(job.status)) << "\n";
        output << "retryable " << (job.retryable ? "true" : "false") << "\n";
        output << "attempt_count " << job.attemptCount << "\n";
        output << "created_at_utc " << quoteQueue(job.createdAtUtc.empty() ? timestampUtcNow() : job.createdAtUtc) << "\n";
        output << "started_at_utc " << quoteQueue(job.startedAtUtc) << "\n";
        output << "completed_at_utc " << quoteQueue(job.completedAtUtc) << "\n";
        output << "last_error " << quoteQueue(job.lastError) << "\n";
        writeVideoJob(output, job.job);
        output << "job_end\n";
    }

    if (!output.good()) {
        error = "Unable to finish writing queue store: " + path_;
        return false;
    }

    error.clear();
    return true;
}

std::vector<QueuedJobRecord> JobQueueStore::load(std::string& error) const {
    std::ifstream input(path_);
    if (!input.is_open()) {
        error.clear();
        return {};
    }

    std::vector<QueuedJobRecord> jobs;
    std::optional<QueuedJobRecord> current;
    for (std::string line; std::getline(input, line);) {
        const std::string trimmed = trimQueue(line);
        if (trimmed.empty() || trimmed[0] == '#') {
            continue;
        }
        if (trimmed == "job_begin") {
            if (current.has_value()) {
                error = "Queue store contains nested job blocks.";
                return {};
            }
            current = QueuedJobRecord{};
            continue;
        }
        if (trimmed == "job_end") {
            if (!current.has_value()) {
                error = "Queue store closed a job block before it started.";
                return {};
            }
            if (current->id.empty()) {
                current->id = generateQueueJobId();
            }
            if (current->createdAtUtc.empty()) {
                current->createdAtUtc = timestampUtcNow();
            }
            if (current->job.previewFrameInterval <= 0) {
                current->job.previewFrameInterval = 15;
            }
            if (current->status == QueuedJobStatus::Running) {
                current->status = QueuedJobStatus::RetryableFailure;
                current->retryable = true;
                if (current->lastError.empty()) {
                    current->lastError =
                        "Application stopped while processing this queued job.";
                }
                if (current->completedAtUtc.empty()) {
                    current->completedAtUtc = timestampUtcNow();
                }
            }
            jobs.push_back(*current);
            current.reset();
            continue;
        }

        const std::size_t space = trimmed.find(' ');
        if (space == std::string::npos) {
            error = "Malformed queue line: " + trimmed;
            return {};
        }

        const std::string key = trimmed.substr(0, space);
        const std::string value = trimQueue(trimmed.substr(space + 1));
        if (key == "schema_version") {
            const auto schema = parseIntQueue(value);
            if (!schema.has_value() || *schema != 1) {
                error = "Unsupported queue schema version.";
                return {};
            }
            continue;
        }

        if (!current.has_value()) {
            error = "Queue field appeared outside a job block: " + key;
            return {};
        }

        if (key == "id") {
            const auto parsed = unquoteQueue(value);
            if (!parsed.has_value()) {
                error = "Queue job id is malformed.";
                return {};
            }
            current->id = *parsed;
            continue;
        }
        if (key == "status") {
            const auto parsed = unquoteQueue(value);
            if (!parsed.has_value()) {
                error = "Queue job status is malformed.";
                return {};
            }
            const auto status = queuedJobStatusFromString(*parsed);
            if (!status.has_value()) {
                error = "Queue job status is unknown: " + *parsed;
                return {};
            }
            current->status = *status;
            continue;
        }
        if (key == "retryable") {
            const auto parsed = parseBoolQueue(value);
            if (!parsed.has_value()) {
                error = "Queue retryable flag is malformed.";
                return {};
            }
            current->retryable = *parsed;
            continue;
        }
        if (key == "attempt_count") {
            const auto parsed = parseIntQueue(value);
            if (!parsed.has_value()) {
                error = "Queue attempt count is malformed.";
                return {};
            }
            current->attemptCount = *parsed;
            continue;
        }
        if (key == "created_at_utc") {
            const auto parsed = unquoteQueue(value);
            if (!parsed.has_value()) {
                error = "Queue creation timestamp is malformed.";
                return {};
            }
            current->createdAtUtc = *parsed;
            continue;
        }
        if (key == "started_at_utc") {
            const auto parsed = unquoteQueue(value);
            if (!parsed.has_value()) {
                error = "Queue started timestamp is malformed.";
                return {};
            }
            current->startedAtUtc = *parsed;
            continue;
        }
        if (key == "completed_at_utc") {
            const auto parsed = unquoteQueue(value);
            if (!parsed.has_value()) {
                error = "Queue completed timestamp is malformed.";
                return {};
            }
            current->completedAtUtc = *parsed;
            continue;
        }
        if (key == "last_error") {
            const auto parsed = unquoteQueue(value);
            if (!parsed.has_value()) {
                error = "Queue error text is malformed.";
                return {};
            }
            current->lastError = *parsed;
            continue;
        }
        if (!readVideoJobField(key, value, current->job, error)) {
            error = "Unknown queue key: " + key;
            return {};
        }
    }

    if (current.has_value()) {
        error = "Queue store ended before job_end.";
        return {};
    }

    error.clear();
    return jobs;
}

bool JobQueueStore::clear(std::string& error) const {
    std::error_code filesystemError;
    const bool removed = std::filesystem::remove(path_, filesystemError);
    if (filesystemError) {
        error = "Unable to clear queue store: " + filesystemError.message();
        return false;
    }
    static_cast<void>(removed);
    error.clear();
    return true;
}

std::string JobQueueStore::defaultPath() {
    const char* home = std::getenv("HOME");
    if (home != nullptr) {
        return std::string(home) + "/.config/ave/job_queue.state";
    }
    return "/tmp/ave_job_queue.state";
}

}  // namespace ave
