#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ave::process_observer {

bool fileExists(const std::string& path);
std::optional<std::filesystem::path> resolveCommandPath(const std::string& cmd);
bool commandInPath(const std::string& cmd);
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

std::int64_t countVideoFrames(const std::string& inputVideo,
                              double previewDurationSec);

std::string summarizeDiagnostics(const std::vector<std::string>& diagnostics);

}  // namespace ave::process_observer
