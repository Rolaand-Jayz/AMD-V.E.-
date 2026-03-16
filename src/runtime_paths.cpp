#include "ave/runtime_paths.hpp"

#include <cstdlib>
#include <string>

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

bool pathExists(const fs::path& path) {
    std::error_code ec;
    return fs::exists(path, ec);
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

}  // namespace

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

    return fs::path("/tmp/ave_models");
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

std::optional<std::filesystem::path> bundledMiGraphXPrefix() {
    if (const auto overridePrefix = envPath("AVE_BUNDLED_MIGRAPHX_PREFIX");
        overridePrefix.has_value() && pathExists(*overridePrefix)) {
        return overridePrefix;
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

std::optional<std::filesystem::path> bundledMiGraphXDriverPath() {
    if (const auto overrideDriver = envPath("AVE_MIGRAPHX_DRIVER");
        overrideDriver.has_value() && pathExists(*overrideDriver)) {
        return overrideDriver;
    }

    const auto prefix = bundledMiGraphXPrefix();
    if (!prefix.has_value()) {
        return std::nullopt;
    }

    const fs::path driver = *prefix / "bin" / "migraphx-driver";
    if (pathExists(driver)) {
        return driver;
    }
    return std::nullopt;
}

}  // namespace ave
