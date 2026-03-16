#pragma once

#include <filesystem>
#include <optional>

namespace ave {

std::filesystem::path defaultWritableModelsDir();

std::optional<std::filesystem::path> bundledModelsDirectory();

std::optional<std::filesystem::path> bundledMiGraphXPrefix();

std::optional<std::filesystem::path> bundledMiGraphXDriverPath();

}  // namespace ave
