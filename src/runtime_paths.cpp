#include "ave/runtime_paths.hpp"

#include <algorithm>
#include <cstdlib>
#include <string>
#include <unordered_set>
#include <vector>

#if defined(__linux__)
#  include <limits.h>
#  include <unistd.h>
#elif defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif

namespace ave {
namespace {

namespace fs = std::filesystem;

fs::path tempScopedFallbackRoot() {
    std::error_code ec;
    fs::path tempRoot = fs::temp_directory_path(ec);
    if (ec || tempRoot.empty()) {
        tempRoot = fs::path("/tmp");
    }

#if defined(__linux__)
    return tempRoot / ("ave-" + std::to_string(static_cast<long long>(::geteuid())));
#else
    return tempRoot / "ave";
#endif
}

bool pathExists(const fs::path& path) {
    std::error_code ec;
    return fs::exists(path, ec);
}

void appendUniqueExistingPath(std::vector<fs::path>& paths, const fs::path& candidate) {
    if (!pathExists(candidate)) {
        return;
    }
    if (std::find(paths.begin(), paths.end(), candidate) == paths.end()) {
        paths.push_back(candidate);
    }
}

std::optional<fs::path> envPath(const char* name) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') {
        return std::nullopt;
    }
    return fs::path(raw);
}

std::optional<fs::path> currentExecutablePath() {
#if defined(__linux__)
    char buffer[PATH_MAX] = {};
    const ssize_t size = ::readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
    if (size <= 0) {
        return std::nullopt;
    }
    buffer[size] = '\0';
    return fs::path(buffer);
#elif defined(_WIN32)
    char buffer[MAX_PATH] = {};
    const DWORD size = ::GetModuleFileNameA(nullptr, buffer, MAX_PATH);
    if (size == 0 || size == MAX_PATH) {
        return std::nullopt;
    }
    return fs::path(buffer);
#else
    return std::nullopt;
#endif
}

std::optional<fs::path> detectInstallPrefix() {
    if (const auto overridePrefix = envPath("AVE_APP_INSTALL_PREFIX");
        overridePrefix.has_value() && pathExists(*overridePrefix)) {
        return overridePrefix;
    }

    const auto executablePath = currentExecutablePath();
    if (!executablePath.has_value()) {
        return std::nullopt;
    }

    const fs::path exeDir = executablePath->parent_path();
    if (exeDir.filename() == "ave" && exeDir.parent_path().filename() == "libexec") {
        const fs::path prefix = exeDir.parent_path().parent_path();
        if (pathExists(prefix / "share" / "ave") || pathExists(prefix / "lib" / "ave")) {
            return prefix;
        }
    }

    if (exeDir.filename() == "bin") {
        const fs::path prefix = exeDir.parent_path();
        if (pathExists(prefix / "share" / "ave") || pathExists(prefix / "lib" / "ave")) {
            return prefix;
        }
    }

    return std::nullopt;
}

std::optional<fs::path> derivedMiGraphXPrefixFromDriverOverride() {
    const auto overrideDriver = envPath("AVE_MIGRAPHX_DRIVER");
    if (!overrideDriver.has_value() || !pathExists(*overrideDriver)) {
        return std::nullopt;
    }
    const fs::path parent = overrideDriver->parent_path();
    if (parent.filename() != "bin") {
        return std::nullopt;
    }
    const fs::path prefix = parent.parent_path();
    if (pathExists(prefix)) {
        return prefix;
    }
    return std::nullopt;
}

char pathListSeparator() {
#if defined(_WIN32)
    return ';';
#else
    return ':';
#endif
}

std::string prependPathList(const std::vector<fs::path>& entries, const char* envVarName) {
    std::vector<std::string> merged;
    merged.reserve(entries.size() + 8);
    std::unordered_set<std::string> seen;

    for (const auto& entry : entries) {
        if (!pathExists(entry)) {
            continue;
        }
        const std::string text = entry.string();
        if (seen.insert(text).second) {
            merged.push_back(text);
        }
    }

    const char separator = pathListSeparator();
    const char* current = std::getenv(envVarName);
    if (current != nullptr && *current != '\0') {
        std::string pathList(current);
        std::size_t start = 0;
        while (start <= pathList.size()) {
            std::size_t end = pathList.find(separator, start);
            if (end == std::string::npos) {
                end = pathList.size();
            }
            const std::string token = pathList.substr(start, end - start);
            if (!token.empty() && seen.insert(token).second) {
                merged.push_back(token);
            }
            if (end == pathList.size()) {
                break;
            }
            start = end + 1;
        }
    }

    std::string result;
    for (std::size_t index = 0; index < merged.size(); ++index) {
        if (index > 0) {
            result.push_back(separator);
        }
        result += merged[index];
    }
    return result;
}

std::optional<fs::path> bundledMiGraphXPrefixImpl() {
    if (const auto overridePrefix = envPath("AVE_BUNDLED_MIGRAPHX_PREFIX");
        overridePrefix.has_value() && pathExists(*overridePrefix)) {
        return overridePrefix;
    }

    if (const auto derivedPrefix = derivedMiGraphXPrefixFromDriverOverride();
        derivedPrefix.has_value()) {
        return derivedPrefix;
    }

    const auto prefix = detectInstallPrefix();
    if (!prefix.has_value()) {
        return std::nullopt;
    }

    const fs::path dir = *prefix / "lib" / "ave" / "migraphx";
    if (pathExists(dir)) {
        return dir;
    }
    return std::nullopt;
}

}  // namespace

std::filesystem::path defaultWritableCacheDir() {
    if (const auto overrideDir = envPath("AVE_CACHE_DIR"); overrideDir.has_value()) {
        return *overrideDir;
    }

    const char* xdgCacheHome = std::getenv("XDG_CACHE_HOME");
    if (xdgCacheHome != nullptr && *xdgCacheHome != '\0') {
        return fs::path(xdgCacheHome) / "ave";
    }

    const char* home = std::getenv("HOME");
    if (home != nullptr && *home != '\0') {
        return fs::path(home) / ".cache" / "ave";
    }

    return tempScopedFallbackRoot() / "cache";
}

std::filesystem::path defaultWritableModelsDir() {
    if (const auto overrideDir = envPath("AVE_MODELS_DIR"); overrideDir.has_value()) {
        return *overrideDir;
    }

    const char* xdgDataHome = std::getenv("XDG_DATA_HOME");
    if (xdgDataHome != nullptr && *xdgDataHome != '\0') {
        return fs::path(xdgDataHome) / "ave" / "models";
    }

    const char* home = std::getenv("HOME");
    if (home != nullptr && *home != '\0') {
        return fs::path(home) / ".local" / "share" / "ave" / "models";
    }

    return tempScopedFallbackRoot() / "models";
}

std::optional<std::filesystem::path> bundledModelsDirectory() {
    if (const auto overrideDir = envPath("AVE_BUNDLED_MODELS_DIR");
        overrideDir.has_value() && pathExists(*overrideDir)) {
        return overrideDir;
    }

    const auto prefix = detectInstallPrefix();
    if (!prefix.has_value()) {
        return std::nullopt;
    }

    const fs::path dir = *prefix / "share" / "ave" / "models";
    if (pathExists(dir)) {
        return dir;
    }
    return std::nullopt;
}

std::optional<std::filesystem::path> bundledRuntimeDirectory() {
    const auto prefix = detectInstallPrefix();
    if (!prefix.has_value()) {
        return std::nullopt;
    }

    const fs::path dir = *prefix / "lib" / "ave" / "runtime";
    if (pathExists(dir)) {
        return dir;
    }
    return std::nullopt;
}

std::optional<std::filesystem::path> bundledMiGraphXPrefix() {
    return bundledMiGraphXPrefixImpl();
}

std::optional<std::filesystem::path> bundledMiGraphXDriverPath() {
    if (const auto overrideDriver = envPath("AVE_MIGRAPHX_DRIVER");
        overrideDriver.has_value() && pathExists(*overrideDriver)) {
        return overrideDriver;
    }

    const auto prefix = bundledMiGraphXPrefixImpl();
    if (!prefix.has_value()) {
        return std::nullopt;
    }

    const fs::path wrappedDriver = *prefix / "bin" / "ave-migraphx-driver";
    if (pathExists(wrappedDriver)) {
        return wrappedDriver;
    }

    const fs::path driver = *prefix / "bin" / "migraphx-driver";
    if (pathExists(driver)) {
        return driver;
    }
    return std::nullopt;
}

std::vector<std::filesystem::path> bundledMiGraphXCompilerLibraryDirectories() {
    std::vector<fs::path> paths;
    if (const auto runtimeDir = bundledRuntimeDirectory(); runtimeDir.has_value()) {
        appendUniqueExistingPath(paths, *runtimeDir);
    }

    const auto prefix = bundledMiGraphXPrefixImpl();
    if (!prefix.has_value()) {
        return paths;
    }

    appendUniqueExistingPath(paths, *prefix / "lib");
    appendUniqueExistingPath(paths, *prefix / "lib" / "migraphx" / "lib");
    return paths;
}

std::vector<std::filesystem::path> bundledMiGraphXCompilerPathDirectories() {
    std::vector<fs::path> paths;
    const auto prefix = bundledMiGraphXPrefixImpl();
    if (!prefix.has_value()) {
        return paths;
    }

    appendUniqueExistingPath(paths, *prefix / "bin");
    return paths;
}

std::vector<std::pair<std::string, std::string>> bundledMiGraphXCompilerEnvOverrides() {
    std::vector<std::pair<std::string, std::string>> overrides;

    const auto pathEntries = bundledMiGraphXCompilerPathDirectories();
    const std::string mergedPath = prependPathList(pathEntries, "PATH");
    if (!mergedPath.empty()) {
        overrides.emplace_back("PATH", mergedPath);
    }

    const auto libraryEntries = bundledMiGraphXCompilerLibraryDirectories();
    const std::string mergedLibraryPath = prependPathList(libraryEntries, "LD_LIBRARY_PATH");
    if (!mergedLibraryPath.empty()) {
        overrides.emplace_back("LD_LIBRARY_PATH", mergedLibraryPath);
    }

    if (const auto prefix = bundledMiGraphXPrefixImpl(); prefix.has_value()) {
        overrides.emplace_back("AVE_BUNDLED_MIGRAPHX_PREFIX", prefix->string());
    }

    return overrides;
}

}  // namespace ave
