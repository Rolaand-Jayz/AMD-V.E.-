#pragma once

#include <optional>
#include <string>

#include "ave/job.hpp"

namespace ave {

struct CliResult {
    bool shouldExit = false;
    int exitCode = 0;
    bool listBackends = false;
    std::optional<VideoJob> job;
};

CliResult parseCli(int argc, char** argv);
void printUsage(const std::string& executableName);

}  // namespace ave
