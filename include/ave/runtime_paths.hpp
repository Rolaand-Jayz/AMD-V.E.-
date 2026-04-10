#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ave {

std::filesystem::path defaultWritableModelsDir();

std::optional<std::filesystem::path> bundledModelsDirectory();

std::optional<std::filesystem::path> bundledRuntimeDirectory();

std::optional<std::filesystem::path> bundledMiGraphXPrefix();

std::optional<std::filesystem::path> bundledMiGraphXDriverPath();

std::vector<std::filesystem::path> bundledMiGraphXCompilerLibraryDirectories();

std::vector<std::filesystem::path> bundledMiGraphXCompilerPathDirectories();

std::vector<std::pair<std::string, std::string>> bundledMiGraphXCompilerEnvOverrides();

}  // namespace ave
