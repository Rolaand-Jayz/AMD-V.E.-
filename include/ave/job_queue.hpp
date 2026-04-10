#pragma once

#include <optional>
#include <string>
#include <vector>

#include "ave/job.hpp"

namespace ave {

enum class QueuedJobStatus {
    Pending,
    Running,
    RetryableFailure,
    Failed,
    Succeeded,
    Cancelled
};

std::string toString(QueuedJobStatus status);
std::optional<QueuedJobStatus> queuedJobStatusFromString(const std::string& value);
bool isTerminalQueueStatus(QueuedJobStatus status);
bool isRetryableQueueFailure(const std::string& error);
std::string generateQueueJobId();

struct QueuedJobRecord {
    std::string id;
    VideoJob job;
    QueuedJobStatus status = QueuedJobStatus::Pending;
    bool retryable = false;
    int attemptCount = 0;
    std::string createdAtUtc;
    std::string startedAtUtc;
    std::string completedAtUtc;
    std::string lastError;
};

class JobQueueStore {
  public:
    explicit JobQueueStore(std::string path = {});

    std::string path() const;
    bool save(const std::vector<QueuedJobRecord>& jobs, std::string& error) const;
    std::vector<QueuedJobRecord> load(std::string& error) const;
    bool clear(std::string& error) const;

    static std::string defaultPath();

  private:
    std::string path_;
};

}  // namespace ave
