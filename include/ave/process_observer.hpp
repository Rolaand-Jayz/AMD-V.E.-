#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ave {

struct VideoStreamProbe;

namespace process_observer {

struct CommandResult {
    int exitCode = -1;
    std::string mergedOutput;
};

bool fileExists(const std::string& path);
std::optional<std::filesystem::path> resolveCommandPath(const std::string& cmd);
bool commandInPath(const std::string& cmd);
int normalizeShellExitCode(int rawStatus);
std::string quoteShellArg(const std::string& value);
std::string trimOutput(std::string value);
std::string wrapCommandWithEnv(
    const std::string& cmd,
    const std::vector<std::pair<std::string, std::string>>& envOverrides);

std::optional<std::string> captureCommandStdout(const std::string& cmd, int& exitCode);
std::optional<std::string> captureCommandStdout(
    const std::string& cmd,
    const std::vector<std::pair<std::string, std::string>>& envOverrides,
    int& exitCode);

int runObservedCommand(const std::string& cmd,
                       const std::function<void(const std::string&)>& onLine);
int runObservedCommand(
    const std::string& cmd,
    const std::vector<std::pair<std::string, std::string>>& envOverrides,
    const std::function<void(const std::string&)>& onLine);
bool runCommandArgs(const std::vector<std::string>& args,
                    CommandResult& result,
                    std::string& error);
bool runCommandArgs(
    const std::vector<std::string>& args,
    const std::vector<std::pair<std::string, std::string>>& envOverrides,
    CommandResult& result,
    std::string& error);

std::int64_t countVideoFrames(const std::string& inputVideo,
                              double previewDurationSec);
std::int64_t countVideoFrames(const VideoStreamProbe& probe,
                              double previewDurationSec);

std::string summarizeDiagnostics(const std::vector<std::string>& diagnostics);

}  // namespace ave::process_observer

}  // namespace ave
