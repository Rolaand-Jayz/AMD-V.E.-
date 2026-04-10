#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "ave/process_observer.hpp"

namespace {

void check(const bool condition, const char* message) {
    if (condition) {
        return;
    }
    std::cerr << "process_observer_tests failed: " << message << '\n';
    std::abort();
}

bool commandSucceeds(const std::string& command) {
    return std::system(command.c_str()) == 0;
}

std::filesystem::path prepareTestClip() {
    if (!commandSucceeds("command -v ffmpeg >/dev/null 2>&1")) {
        return {};
    }

    const auto tempDir =
        std::filesystem::temp_directory_path() / "ave_process_observer_tests";
    std::error_code ec;
    std::filesystem::remove_all(tempDir, ec);
    std::filesystem::create_directories(tempDir, ec);
    const auto videoPath = tempDir / "sample.mp4";

    const std::string createVideoCommand =
        "ffmpeg -hide_banner -loglevel error -y "
        "-f lavfi -i testsrc=size=4x2:rate=2 "
        "-frames:v 2 -pix_fmt yuv420p \"" + videoPath.string() + "\"";
    check(commandSucceeds(createVideoCommand),
          "ffmpeg should create the process observer test clip");
    return videoPath;
}

void cleanupArtifacts() {
    const auto tempDir =
        std::filesystem::temp_directory_path() / "ave_process_observer_tests";
    std::error_code ec;
    std::filesystem::remove_all(tempDir, ec);
}

void testRunObservedCommandSplitsLines() {
    std::vector<std::string> lines;
    const int rc = ave::process_observer::runObservedCommand(
        "printf 'alpha\\nbeta\\n'",
        [&](const std::string& line) {
            lines.push_back(line);
        });

    check(rc == 0, "runObservedCommand should return the process exit status");
    check(lines.size() == 2, "runObservedCommand should surface each output line");
    check(lines[0] == "alpha" && lines[1] == "beta",
          "runObservedCommand should preserve line order");
}

void testRunObservedCommandSupportsEnvOverrides() {
    std::vector<std::string> lines;
    const int rc = ave::process_observer::runObservedCommand(
        "printf '%s\\n' \"$AVE_PROCESS_OBSERVER_TEST_VAR\"",
        {{"AVE_PROCESS_OBSERVER_TEST_VAR", "process-env-ok"}},
        [&](const std::string& line) {
            lines.push_back(line);
        });

    check(rc == 0, "env-aware runObservedCommand should return the process exit status");
    check(lines.size() == 1, "env-aware runObservedCommand should capture the overridden output");
    check(lines[0] == "process-env-ok",
          "env-aware runObservedCommand should apply environment overrides");
}

void testCaptureCommandStdoutSupportsEnvOverrides() {
    int exitCode = -1;
    const auto output = ave::process_observer::captureCommandStdout(
        "printf '%s' \"$AVE_PROCESS_OBSERVER_CAPTURE_VAR\"",
        {{"AVE_PROCESS_OBSERVER_CAPTURE_VAR", "capture-env-ok"}},
        exitCode);

    check(output.has_value(), "env-aware captureCommandStdout should capture command output");
    check(exitCode == 0, "env-aware captureCommandStdout should return the exit status");
    check(*output == "capture-env-ok",
          "env-aware captureCommandStdout should apply environment overrides");
}

void testCountVideoFramesHonorsPreviewLimit() {
    const auto inputPath = prepareTestClip();
    if (inputPath.empty()) {
        return;
    }

    const auto fullCount =
        ave::process_observer::countVideoFrames(inputPath.string(), 0.0);
    const auto previewCount =
        ave::process_observer::countVideoFrames(inputPath.string(), 0.4);

    check(fullCount == 2, "countVideoFrames should report the full clip length");
    check(previewCount == 1, "countVideoFrames should apply preview trimming");

    cleanupArtifacts();
}

void testResolveCommandPathAndSummarizeDiagnostics() {
    const auto shPath = ave::process_observer::resolveCommandPath("sh");
    check(shPath.has_value(), "resolveCommandPath should find commands from PATH");
    check(!shPath->empty(), "resolved command path should not be empty");
    check(ave::process_observer::summarizeDiagnostics({}) == "",
          "empty diagnostics should summarize to an empty string");
    check(ave::process_observer::summarizeDiagnostics({"first", "last"}) == "last",
          "diagnostic summary should prefer the most recent entry");
}

}  // namespace

int main() {
    testRunObservedCommandSplitsLines();
    testRunObservedCommandSupportsEnvOverrides();
    testCaptureCommandStdoutSupportsEnvOverrides();
    testCountVideoFramesHonorsPreviewLimit();
    testResolveCommandPathAndSummarizeDiagnostics();
    return 0;
}
