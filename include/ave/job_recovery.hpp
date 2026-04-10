#pragma once

#include <optional>
#include <string>

#include "ave/job.hpp"

namespace ave {

struct RecoveredJobState {
    VideoJob job;
    std::string startedAtUtc;
};

class JobRecoveryStore {
  public:
    explicit JobRecoveryStore(std::string path = {});

    std::string path() const;
    bool save(const RecoveredJobState& state, std::string& error) const;
    std::optional<RecoveredJobState> load(std::string& error) const;
    bool clear(std::string& error) const;

    static std::string defaultPath();

  private:
    std::string path_;
};

}  // namespace ave
