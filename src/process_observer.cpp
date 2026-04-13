#include "ave/process_observer.hpp"

#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <sstream>
#include <unordered_map>

#if defined(__unix__) || defined(__APPLE__)
#  include <sys/wait.h>
#  include <unistd.h>
#endif

#include "ave/video_probe.hpp"

namespace ave::process_observer {
namespace {

struct CommandPathCache {
    std::mutex mutex;
    std::unordered_map<std::string, std::optional<std::filesystem::path>> entries;
};

CommandPathCache& commandPathCache() {
    static CommandPathCache cache;
    return cache;
}

std::string makeCommandPathCacheKey(const std::string& pathEnv, const std::string& cmd) {
    std::string key = pathEnv;
    key.push_back('\0');
    key += cmd;
    return key;
}

std::string quoteShellLiteral(const std::string& value) {
    std::string out = "'";
    for (const char ch : value) {
        if (ch == '\'') {
            out += "'\\''";
        } else {
            out.push_back(ch);
        }
    }
    out.push_back('\'');
    return out;
}

}  // namespace

int normalizeShellExitCode(const int rawStatus) {
    if (rawStatus < 0) {
        return rawStatus;
    }
#if defined(__unix__) || defined(__APPLE__)
    if (WIFEXITED(rawStatus)) {
        return WEXITSTATUS(rawStatus);
    }
    if (WIFSIGNALED(rawStatus)) {
        return 128 + WTERMSIG(rawStatus);
    }
#endif
    return rawStatus;
}

bool fileExists(const std::string& path) {
    std::error_code ec;
    return std::filesystem::exists(path, ec);
}

std::optional<std::filesystem::path> resolveCommandPath(const std::string& cmd) {
    if (cmd.empty()) {
        return std::nullopt;
    }

    const char* pathEnv = std::getenv("PATH");
    if (pathEnv == nullptr) {
        return std::nullopt;
    }

    const std::string cacheKey = makeCommandPathCacheKey(pathEnv, cmd);
    {
        std::lock_guard<std::mutex> lock(commandPathCache().mutex);
        const auto it = commandPathCache().entries.find(cacheKey);
        if (it != commandPathCache().entries.end()) {
            return it->second;
        }
    }

    const std::string path(pathEnv);
    std::size_t start = 0;
    std::optional<std::filesystem::path> resolved;
    while (start <= path.size()) {
        std::size_t end = path.find(':', start);
        if (end == std::string::npos) {
            end = path.size();
        }
        const std::string dir = path.substr(start, end - start);
        if (!dir.empty()) {
            const auto candidate = std::filesystem::path(dir) / cmd;
            std::error_code ec;
            if (std::filesystem::exists(candidate, ec) && !ec) {
#if defined(__unix__) || defined(__APPLE__)
                if (::access(candidate.c_str(), X_OK) == 0) {
                    resolved = candidate;
                    break;
                }
#else
                resolved = candidate;
                break;
#endif
            }
        }
        if (end == path.size()) {
            break;
        }
        start = end + 1;
    }

    {
        std::lock_guard<std::mutex> lock(commandPathCache().mutex);
        commandPathCache().entries.emplace(cacheKey, resolved);
    }
    return resolved;
}

bool commandInPath(const std::string& cmd) {
    return resolveCommandPath(cmd).has_value();
}

std::string quoteShellArg(const std::string& value) {
    return quoteShellLiteral(value);
}

std::string wrapCommandWithEnv(
    const std::string& cmd,
    const std::vector<std::pair<std::string, std::string>>& envOverrides) {
    if (envOverrides.empty()) {
        return cmd;
    }

    std::ostringstream wrapped;
    wrapped << "env";
    for (const auto& [name, value] : envOverrides) {
        wrapped << ' ' << quoteShellLiteral(name + "=" + value);
    }
    wrapped << " sh -c " << quoteShellLiteral(cmd);
    return wrapped.str();
}

std::string trimOutput(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::optional<std::string> captureCommandStdout(const std::string& cmd, int& exitCode) {
    return captureCommandStdout(cmd, {}, exitCode);
}

std::optional<std::string> captureCommandStdout(
        const std::string& cmd,
        const std::vector<std::pair<std::string, std::string>>& envOverrides,
        int& exitCode) {
    const std::string wrappedCommand = wrapCommandWithEnv(cmd, envOverrides);
    FILE* pipe = popen(wrappedCommand.c_str(), "r");
    if (pipe == nullptr) {
        exitCode = -1;
        return std::nullopt;
    }

    std::string output;
    std::array<char, 512> buffer{};
    while (true) {
        const std::size_t bytes = std::fread(buffer.data(), sizeof(char), buffer.size(), pipe);
        if (bytes == 0) {
            break;
        }
        output.append(buffer.data(), bytes);
    }

    exitCode = normalizeShellExitCode(pclose(pipe));
    return output;
}

int runObservedCommand(const std::string& cmd,
                       const std::function<void(const std::string&)>& onLine) {
    return runObservedCommand(cmd, {}, onLine);
}

int runObservedCommand(
        const std::string& cmd,
        const std::vector<std::pair<std::string, std::string>>& envOverrides,
        const std::function<void(const std::string&)>& onLine) {
    const std::string wrappedCommand = wrapCommandWithEnv(cmd, envOverrides);
    FILE* pipe = popen(wrappedCommand.c_str(), "r");
    if (pipe == nullptr) {
        return -1;
    }

    std::array<char, 512> buffer{};
    std::string line;
    while (true) {
        const std::size_t bytes = std::fread(buffer.data(), sizeof(char), buffer.size(), pipe);
        if (bytes == 0) {
            break;
        }
        for (std::size_t i = 0; i < bytes; ++i) {
            const char ch = buffer[i];
            if (ch == '\n' || ch == '\r') {
                const std::string trimmed = trimOutput(line);
                if (!trimmed.empty()) {
                    onLine(trimmed);
                }
                line.clear();
            } else {
                line.push_back(ch);
            }
        }
    }

    const std::string trimmed = trimOutput(line);
    if (!trimmed.empty()) {
        onLine(trimmed);
    }

    return normalizeShellExitCode(pclose(pipe));
}

bool runCommandArgs(const std::vector<std::string>& args,
                    CommandResult& result,
                    std::string& error) {
    return runCommandArgs(args, {}, result, error);
}

bool runCommandArgs(
        const std::vector<std::string>& args,
        const std::vector<std::pair<std::string, std::string>>& envOverrides,
        CommandResult& result,
        std::string& error) {
    result = {};
    if (args.empty() || args.front().empty()) {
        error = "Cannot run an empty command.";
        return false;
    }

#if defined(__unix__) || defined(__APPLE__)
    int pipeFds[2] = {-1, -1};
    if (::pipe(pipeFds) != 0) {
        error = std::string("Failed to create subprocess pipe: ") + std::strerror(errno);
        return false;
    }

    const pid_t pid = ::fork();
    if (pid < 0) {
        const int savedErrno = errno;
        ::close(pipeFds[0]);
        ::close(pipeFds[1]);
        error = std::string("Failed to fork subprocess: ") + std::strerror(savedErrno);
        return false;
    }

    if (pid == 0) {
        ::close(pipeFds[0]);
        if (::dup2(pipeFds[1], STDOUT_FILENO) < 0 ||
            ::dup2(pipeFds[1], STDERR_FILENO) < 0) {
            const std::string message =
                std::string("Failed to redirect subprocess output: ") + std::strerror(errno) + "\n";
            const auto size = static_cast<unsigned long>(message.size());
            ::write(pipeFds[1], message.c_str(), size);
            ::close(pipeFds[1]);
            _exit(126);
        }
        ::close(pipeFds[1]);

        for (const auto& [name, value] : envOverrides) {
            if (::setenv(name.c_str(), value.c_str(), 1) != 0) {
                const std::string message =
                    std::string("Failed to set subprocess environment variable '") + name +
                    "': " + std::strerror(errno) + "\n";
                const auto size = static_cast<unsigned long>(message.size());
                ::write(STDERR_FILENO, message.c_str(), size);
                _exit(126);
            }
        }

        std::vector<char*> argv;
        argv.reserve(args.size() + 1);
        for (const auto& arg : args) {
            argv.push_back(const_cast<char*>(arg.c_str()));
        }
        argv.push_back(nullptr);

        if (args.front().find('/') != std::string::npos) {
            ::execv(args.front().c_str(), argv.data());
        } else {
            ::execvp(args.front().c_str(), argv.data());
        }

        const std::string message =
            std::string("Failed to exec '") + args.front() + "': " + std::strerror(errno) + "\n";
        const auto size = static_cast<unsigned long>(message.size());
        ::write(STDERR_FILENO, message.c_str(), size);
        _exit(errno == ENOENT ? 127 : 126);
    }

    ::close(pipeFds[1]);
    std::array<char, 4096> buffer{};
    while (true) {
        const auto bytesRead = ::read(pipeFds[0], buffer.data(), buffer.size());
        if (bytesRead == 0) {
            break;
        }
        if (bytesRead < 0) {
            if (errno == EINTR) {
                continue;
            }
            ::close(pipeFds[0]);
            int status = 0;
            (void)::waitpid(pid, &status, 0);
            error = std::string("Failed to read subprocess output: ") + std::strerror(errno);
            return false;
        }
        result.mergedOutput.append(buffer.data(), static_cast<std::size_t>(bytesRead));
    }
    ::close(pipeFds[0]);

    int status = 0;
    pid_t waited = 0;
    do {
        waited = ::waitpid(pid, &status, 0);
    } while (waited < 0 && errno == EINTR);
    if (waited < 0) {
        error = std::string("Failed to wait for subprocess: ") + std::strerror(errno);
        return false;
    }

    result.exitCode = normalizeShellExitCode(status);
    return true;
#else
    std::ostringstream command;
    for (std::size_t i = 0; i < args.size(); ++i) {
        if (i > 0) {
            command << ' ';
        }
        command << quoteShellArg(args[i]);
    }
    int exitCode = -1;
    const auto output = captureCommandStdout(command.str(), envOverrides, exitCode);
    if (!output.has_value()) {
        error = "Failed to launch subprocess command.";
        return false;
    }
    result.exitCode = exitCode;
    result.mergedOutput = *output;
    return true;
#endif
}

std::int64_t countVideoFrames(const std::string& inputVideo,
                              const double previewDurationSec) {
    std::string error;
    const auto probe = probeVideoStream(inputVideo, error);
    if (!probe.has_value()) {
        return 0;
    }

    return countVideoFrames(*probe, previewDurationSec);
}

std::int64_t countVideoFrames(const VideoStreamProbe& probe,
                              const double previewDurationSec) {
    const std::int64_t countedFrames = probe.estimatedFrameCount();
    if (previewDurationSec > 0.0) {
        const double fps = probe.effectiveFrameRate(0.0);
        if (fps > 0.0) {
            const auto previewFrames =
                static_cast<std::int64_t>(previewDurationSec * fps + 0.5);
            if (countedFrames > 0) {
                return std::min(countedFrames, previewFrames);
            }
            return previewFrames;
        }
    }

    return countedFrames;
}

std::string summarizeDiagnostics(const std::vector<std::string>& diagnostics) {
    if (diagnostics.empty()) {
        return {};
    }
    return diagnostics.back();
}

}  // namespace ave::process_observer
