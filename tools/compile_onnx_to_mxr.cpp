#include <algorithm>
#include <cctype>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "ave/model_manager.hpp"

namespace {

std::string toLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

void printUsage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " [options]\n"
        << "\n"
        << "Compiles all downloaded ONNX catalog models to MiGraphX .mxr artifacts.\n"
        << "\n"
        << "Options:\n"
        << "  --precision <fp16|fp32>    compile precision (default: fp16)\n"
        << "  --model <id>               compile only a specific model id (repeatable)\n"
        << "  --exclude-animation        skip anime / animation-specific models\n"
        << "  --exclude-stereo3d         skip 2D->3D stereo depth models\n"
        << "  --force                    recompile even when a validated artifact exists\n"
        << "  --help                     show this message\n";
}

std::optional<ave::ModelPrecision> parsePrecision(const std::string& value) {
    const std::string lowered = toLower(value);
    if (lowered == "fp16") {
        return ave::ModelPrecision::Fp16;
    }
    if (lowered == "fp32") {
        return ave::ModelPrecision::Fp32;
    }
    return std::nullopt;
}

bool containsAnimationKeyword(const std::string& value) {
    const std::string lowered = toLower(value);
    return lowered.find("anime") != std::string::npos
        || lowered.find("animation") != std::string::npos
        || lowered.find("spanimation") != std::string::npos;
}

bool isAnimationModel(const ave::ManagedModel& model) {
    return containsAnimationKeyword(model.entry.id)
        || containsAnimationKeyword(model.entry.displayName)
        || containsAnimationKeyword(model.entry.description);
}

bool isKnownUnsupportedModelError(const std::string& error) {
    return error.find("quantized operators not supported by MiGraphX") != std::string::npos;
}

bool shouldIncludeModel(const ave::ManagedModel& model,
                        const std::vector<std::string>& requestedIds,
                        bool excludeAnimation,
                        bool excludeStereo3D) {
    if (model.entry.sourceFormat != ave::ModelFormat::Onnx) {
        return false;
    }
    if (model.downloadedPath.empty() || model.downloadedPath == "(builtin)") {
        return false;
    }
    const bool explicitlyRequested =
        std::find(requestedIds.begin(), requestedIds.end(), model.entry.id) != requestedIds.end();
    if (!requestedIds.empty()) {
        return explicitlyRequested;
    }
    if (excludeStereo3D && model.entry.stage == ave::StageKind::Stereo3D) {
        return false;
    }
    if (excludeAnimation && isAnimationModel(model)) {
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    ave::ModelPrecision precision = ave::ModelPrecision::Fp16;
    bool force = false;
    bool excludeAnimation = false;
    bool excludeStereo3D = false;
    std::vector<std::string> requestedIds;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        }
        if (arg == "--force") {
            force = true;
            continue;
        }
        if (arg == "--exclude-animation") {
            excludeAnimation = true;
            continue;
        }
        if (arg == "--exclude-stereo3d") {
            excludeStereo3D = true;
            continue;
        }
        if (arg == "--precision") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --precision\n";
                return 1;
            }
            ++i;
            const auto parsed = parsePrecision(argv[i]);
            if (!parsed.has_value()) {
                std::cerr << "Unsupported precision: " << argv[i] << '\n';
                return 1;
            }
            precision = *parsed;
            continue;
        }
        if (arg == "--model") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --model\n";
                return 1;
            }
            requestedIds.emplace_back(argv[++i]);
            continue;
        }

        std::cerr << "Unknown option: " << arg << '\n';
        printUsage(argv[0]);
        return 1;
    }

    ave::ModelManager manager;
    manager.refresh();
    auto models = manager.allModels();
    std::sort(models.begin(), models.end(),
              [](const ave::ManagedModel& lhs, const ave::ManagedModel& rhs) {
                  return lhs.entry.id < rhs.entry.id;
              });

    int considered = 0;
    int cached = 0;
    int compiled = 0;
    int failed = 0;
    int unsupported = 0;
    int skipped = 0;

    for (const auto& model : models) {
        if (!shouldIncludeModel(model, requestedIds, excludeAnimation, excludeStereo3D)) {
            ++skipped;
            continue;
        }

        ++considered;
        std::string validationDetail;
        const auto cachedPath = manager.validatedCompiledArtifactPath(
            model.entry.id, precision, std::nullopt, std::nullopt, 1, &validationDetail);
        if (!force && cachedPath.has_value()) {
            ++cached;
            std::cout << "[cached] " << model.entry.id << " -> " << *cachedPath << '\n';
            continue;
        }

        std::cout << "[compile] " << model.entry.id
                  << " (" << model.downloadedPath << ")" << std::endl;

        int lastPercent = -1;
        std::string lastStatus;
        const auto progressCb = [&](const std::string& modelId,
                                    float progress,
                                    const std::string& status) {
            const int percent = progress >= 0.0f ? static_cast<int>(progress * 100.0f) : -1;
            if (percent == lastPercent && status == lastStatus) {
                return;
            }
            lastPercent = percent;
            lastStatus = status;
            std::cout << "[" << modelId << "] ";
            if (percent >= 0) {
                std::cout << percent << "% ";
            }
            std::cout << status << '\n';
        };
        const auto stateCb = [&](const std::string& modelId, ave::ModelState state) {
            std::cout << "[" << modelId << "] state=" << ave::toString(state) << '\n';
        };

        std::string error;
        if (manager.convertToMiGraphX(model.entry.id, progressCb, stateCb, error, precision)) {
            ++compiled;
            const auto outputPath = manager.validatedCompiledArtifactPath(
                model.entry.id, precision);
            if (outputPath.has_value()) {
                std::cout << "[ok] " << model.entry.id << " -> " << *outputPath << '\n';
            } else {
                std::cout << "[ok] " << model.entry.id << " -> artifact created\n";
            }
        } else {
            if (isKnownUnsupportedModelError(error)) {
                ++unsupported;
                std::cout << "[unsupported] " << model.entry.id << '\n'
                          << error << '\n';
            } else {
                ++failed;
                std::cout << "[fail] " << model.entry.id << '\n'
                          << error << '\n';
            }
        }
    }

    std::cout << "\nSummary:\n"
              << "  considered: " << considered << '\n'
              << "  cached:     " << cached << '\n'
              << "  compiled:   " << compiled << '\n'
              << "  unsupported:" << ' ' << unsupported << '\n'
              << "  failed:     " << failed << '\n'
              << "  skipped:    " << skipped << '\n';

    return failed == 0 ? 0 : 1;
}
