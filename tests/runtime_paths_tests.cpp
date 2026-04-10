#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "ave/runtime_paths.hpp"

namespace {

void check(const bool condition, const char* message) {
    if (condition) {
        return;
    }
    std::cerr << "runtime_paths_tests failed: " << message << '\n';
    std::abort();
}

void setEnvVar(const char* name, const std::string& value) {
#if defined(_WIN32)
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

void unsetEnvVar(const char* name) {
#if defined(_WIN32)
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

void testBundledMiGraphXPrefixDoesNotRequireCompilerTool() {
    const auto tempDir =
        std::filesystem::temp_directory_path() / "ave_runtime_paths_tests";
    std::error_code ec;
    std::filesystem::remove_all(tempDir, ec);
    std::filesystem::create_directories(tempDir / "lib" / "migraphx" / "lib", ec);

    setEnvVar("AVE_BUNDLED_MIGRAPHX_PREFIX", tempDir.string());
    const auto prefix = ave::bundledMiGraphXPrefix();
    check(prefix.has_value(), "bundledMiGraphXPrefix should honor the override path");
    check(*prefix == tempDir, "bundledMiGraphXPrefix should return the overridden prefix");

    const auto driver = ave::bundledMiGraphXDriverPath();
    check(!driver.has_value(),
          "bundledMiGraphXDriverPath should stay empty when no compiler tool is bundled");

    unsetEnvVar("AVE_BUNDLED_MIGRAPHX_PREFIX");
    std::filesystem::remove_all(tempDir, ec);
}

void testBundledMiGraphXDriverOverrideWins() {
    const auto tempDir =
        std::filesystem::temp_directory_path() / "ave_runtime_paths_driver_tests";
    std::error_code ec;
    std::filesystem::remove_all(tempDir, ec);
    std::filesystem::create_directories(tempDir / "bin", ec);

    const auto driverPath = tempDir / "bin" / "migraphx-driver";
    std::ofstream out(driverPath);
    out << "#!/bin/sh\nexit 0\n";
    out.close();

    setEnvVar("AVE_MIGRAPHX_DRIVER", driverPath.string());
    const auto driver = ave::bundledMiGraphXDriverPath();
    check(driver.has_value(), "bundledMiGraphXDriverPath should honor AVE_MIGRAPHX_DRIVER");
    check(*driver == driverPath, "bundledMiGraphXDriverPath should return the explicit override");

    unsetEnvVar("AVE_MIGRAPHX_DRIVER");
    std::filesystem::remove_all(tempDir, ec);
}

void testBundledMiGraphXDriverPrefersLauncherWrapper() {
    const auto tempDir =
        std::filesystem::temp_directory_path() / "ave_runtime_paths_wrapper_tests";
    std::error_code ec;
    std::filesystem::remove_all(tempDir, ec);
    std::filesystem::create_directories(tempDir / "bin", ec);

    const auto wrappedDriverPath = tempDir / "bin" / "ave-migraphx-driver";
    std::ofstream wrapped(wrappedDriverPath);
    wrapped << "#!/bin/sh\nexit 0\n";
    wrapped.close();

    const auto rawDriverPath = tempDir / "bin" / "migraphx-driver";
    std::ofstream raw(rawDriverPath);
    raw << "#!/bin/sh\nexit 0\n";
    raw.close();

    setEnvVar("AVE_BUNDLED_MIGRAPHX_PREFIX", tempDir.string());
    const auto driver = ave::bundledMiGraphXDriverPath();
    check(driver.has_value(), "bundledMiGraphXDriverPath should find the bundled driver");
    check(*driver == wrappedDriverPath,
          "bundledMiGraphXDriverPath should prefer the bundled launcher wrapper");

    unsetEnvVar("AVE_BUNDLED_MIGRAPHX_PREFIX");
    std::filesystem::remove_all(tempDir, ec);
}

void testBundledMiGraphXCompilerEnvOverridesPrependToolchainPaths() {
    const auto tempDir =
        std::filesystem::temp_directory_path() / "ave_runtime_paths_env_tests";
    std::error_code ec;
    std::filesystem::remove_all(tempDir, ec);
    std::filesystem::create_directories(tempDir / "bin", ec);
    std::filesystem::create_directories(tempDir / "lib", ec);
    std::filesystem::create_directories(tempDir / "lib" / "migraphx" / "lib", ec);

    setEnvVar("AVE_BUNDLED_MIGRAPHX_PREFIX", tempDir.string());
    setEnvVar("PATH", "/usr/bin");
    setEnvVar("LD_LIBRARY_PATH", "/usr/lib");

    const auto overrides = ave::bundledMiGraphXCompilerEnvOverrides();
    check(overrides.size() >= 2,
          "bundledMiGraphXCompilerEnvOverrides should emit PATH and LD_LIBRARY_PATH");

    std::string pathValue;
    std::string libraryValue;
    for (const auto& [name, value] : overrides) {
        if (name == "PATH") {
            pathValue = value;
        } else if (name == "LD_LIBRARY_PATH") {
            libraryValue = value;
        }
    }

    check(pathValue.find((tempDir / "bin").string()) == 0,
          "bundled PATH override should prepend the bundled compiler bin directory");
    check(libraryValue.find((tempDir / "lib").string()) == 0,
          "bundled LD_LIBRARY_PATH override should prepend the bundled compiler lib directory");
    check(libraryValue.find((tempDir / "lib" / "migraphx" / "lib").string()) != std::string::npos,
          "bundled LD_LIBRARY_PATH override should include nested MiGraphX libs");
    check(pathValue.find("/usr/bin") != std::string::npos,
          "bundled PATH override should preserve the existing PATH");
    check(libraryValue.find("/usr/lib") != std::string::npos,
          "bundled LD_LIBRARY_PATH override should preserve the existing library path");

    unsetEnvVar("AVE_BUNDLED_MIGRAPHX_PREFIX");
    unsetEnvVar("PATH");
    unsetEnvVar("LD_LIBRARY_PATH");
    std::filesystem::remove_all(tempDir, ec);
}

}  // namespace

int main() {
    testBundledMiGraphXPrefixDoesNotRequireCompilerTool();
    testBundledMiGraphXDriverOverrideWins();
    testBundledMiGraphXDriverPrefersLauncherWrapper();
    testBundledMiGraphXCompilerEnvOverridesPrependToolchainPaths();
    return 0;
}
