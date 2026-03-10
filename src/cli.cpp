#include "ave/cli.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "ave/types.hpp"

namespace ave {
namespace {

std::string trim(const std::string& input) {
    std::size_t begin = 0;
    while (begin < input.size() && std::isspace(static_cast<unsigned char>(input[begin])) != 0) {
        ++begin;
    }
    std::size_t end = input.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(input[end - 1])) != 0) {
        --end;
    }
    return input.substr(begin, end - begin);
}

std::string toLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

std::vector<std::string> split(const std::string& value, char delimiter) {
    std::vector<std::string> parts;
    std::size_t start = 0;
    while (start <= value.size()) {
        std::size_t end = value.find(delimiter, start);
        if (end == std::string::npos) {
            end = value.size();
        }
        parts.push_back(value.substr(start, end - start));
        if (end == value.size()) {
            break;
        }
        start = end + 1;
    }
    return parts;
}

ParameterValue parseValue(const std::string& value) {
    const std::string lowered = toLower(trim(value));
    if (lowered == "true") {
        return true;
    }
    if (lowered == "false") {
        return false;
    }

    char* intEnd = nullptr;
    const long long intParsed = std::strtoll(lowered.c_str(), &intEnd, 10);
    if (intEnd != lowered.c_str() && *intEnd == '\0') {
        return static_cast<std::int64_t>(intParsed);
    }

    char* floatEnd = nullptr;
    const double floatParsed = std::strtod(lowered.c_str(), &floatEnd);
    if (floatEnd != lowered.c_str() && *floatEnd == '\0') {
        return floatParsed;
    }

    return trim(value);
}

std::optional<BackendType> parseBackend(const std::string& value) {
    const std::string normalized = toLower(value);
    if (normalized == "auto") {
        return BackendType::Auto;
    }
    if (normalized == "migraphx") {
        return BackendType::MiGraphX;
    }
    if (normalized == "ncnn-vulkan" || normalized == "ncnn_vulkan" || normalized == "ncnn") {
        return BackendType::NcnnVulkan;
    }
    if (normalized == "vulkan" || normalized == "vulkan-compute" || normalized == "vulkan_compute") {
        return BackendType::VulkanCompute;
    }
    if (normalized == "vapoursynth" || normalized == "vapourynth" || normalized == "vs") {
        return BackendType::VapourSynth;
    }
    if (normalized == "glsl" || normalized == "glsl-shader" || normalized == "glsl_shader") {
        return BackendType::GlslShader;
    }
    return std::nullopt;
}

std::optional<EnhancementStage> parseStageSpec(const std::string& spec, std::string& error) {
    const std::size_t colon = spec.find(':');
    const std::string kindToken = colon == std::string::npos ? spec : spec.substr(0, colon);
    const std::optional<StageKind> kind = stageKindFromString(trim(kindToken));
    if (!kind.has_value()) {
        error = "Unknown stage kind: " + kindToken;
        return std::nullopt;
    }

    EnhancementStage stage;
    stage.kind = *kind;

    if (colon == std::string::npos) {
        return stage;
    }

    const std::string paramString = spec.substr(colon + 1);
    if (paramString.empty()) {
        return stage;
    }

    const std::vector<std::string> assignments = split(paramString, ',');
    for (const std::string& assignment : assignments) {
        const std::size_t eq = assignment.find('=');
        if (eq == std::string::npos) {
            error = "Malformed stage parameter in: " + assignment;
            return std::nullopt;
        }

        const std::string key = trim(assignment.substr(0, eq));
        const std::string rawValue = assignment.substr(eq + 1);
        if (key.empty()) {
            error = "Empty stage parameter key in: " + assignment;
            return std::nullopt;
        }

        stage.params[key] = parseValue(rawValue);
    }

    return stage;
}

bool requireValue(int argc, char** argv, int& i, std::string& outValue, const std::string& flag, CliResult& result) {
    if (i + 1 >= argc) {
        std::cerr << "Missing value for " << flag << std::endl;
        result.shouldExit = true;
        result.exitCode = 1;
        return false;
    }
    ++i;
    outValue = argv[i];
    return true;
}

}  // namespace

void printUsage(const std::string& executableName) {
    std::cout << "Usage: " << executableName << " --input <in.mp4> --output <out.mp4> [options]\n"
              << "\n"
              << "Options:\n"
              << "  --backend <auto|migraphx|ncnn-vulkan|vulkan>\n"
              << "  --stage <name[:key=value,key=value]>      (repeatable, stackable)\n"
              << "  --codec <ffmpeg video codec>               default: libx264\n"
              << "  --profile <codec profile>                  default: (auto)\n"
              << "  --crf <int>                                default: 18\n"
              << "  --preset <ffmpeg preset>                   default: medium\n"
              << "  --preview                                  process only a short clip for preview\n"
              << "  --preview-duration <seconds>               preview clip length (default: 10)\n"
              << "  --dry-run                                  print ordered plan only\n"
              << "  --list-backends                            probe AMD backends\n"
              << "  --help\n"
              << "\n"
              << "Stage examples:\n"
              << "  --stage restore_compression:strength=0.9\n"
              << "  --stage remove_artifacts:strength=0.7\n"
              << "  --stage upscale:width=3840,height=2160\n"
              << "  --stage sharpen:amount=0.5\n"
              << "  --stage interpolate:fps=60\n"
              << "\n"
              << "Ordering guarantees:\n"
              << "  restore/deartifact/cleanup before upscale+sharpen\n"
              << "  interpolation is always last before encode\n";
}

CliResult parseCli(int argc, char** argv) {
    CliResult result;

    if (argc <= 1) {
        printUsage(argv[0]);
        result.shouldExit = true;
        result.exitCode = 1;
        return result;
    }

    VideoJob job;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            result.shouldExit = true;
            result.exitCode = 0;
            return result;
        }

        if (arg == "--list-backends") {
            result.listBackends = true;
            continue;
        }

        if (arg == "--dry-run") {
            job.dryRun = true;
            continue;
        }

        if (arg == "--input") {
            std::string value;
            if (!requireValue(argc, argv, i, value, arg, result)) {
                return result;
            }
            job.inputPath = value;
            continue;
        }

        if (arg == "--output") {
            std::string value;
            if (!requireValue(argc, argv, i, value, arg, result)) {
                return result;
            }
            job.outputPath = value;
            continue;
        }

        if (arg == "--backend") {
            std::string value;
            if (!requireValue(argc, argv, i, value, arg, result)) {
                return result;
            }
            const std::optional<BackendType> backend = parseBackend(value);
            if (!backend.has_value()) {
                std::cerr << "Unsupported backend: " << value << std::endl;
                result.shouldExit = true;
                result.exitCode = 1;
                return result;
            }
            job.requestedBackend = *backend;
            continue;
        }

        if (arg == "--stage") {
            std::string value;
            if (!requireValue(argc, argv, i, value, arg, result)) {
                return result;
            }

            std::string parseError;
            std::optional<EnhancementStage> stage = parseStageSpec(value, parseError);
            if (!stage.has_value()) {
                std::cerr << parseError << std::endl;
                result.shouldExit = true;
                result.exitCode = 1;
                return result;
            }

            job.requestedStages.push_back(*stage);
            continue;
        }

        if (arg == "--codec") {
            std::string value;
            if (!requireValue(argc, argv, i, value, arg, result)) {
                return result;
            }
            job.encode.codec = value;
            continue;
        }

        if (arg == "--profile") {
            std::string value;
            if (!requireValue(argc, argv, i, value, arg, result)) {
                return result;
            }
            job.encode.profile = value;
            continue;
        }

        if (arg == "--crf") {
            std::string value;
            if (!requireValue(argc, argv, i, value, arg, result)) {
                return result;
            }
            try {
                const int crf = std::stoi(value);
                if (crf < 0 || crf > 51) {
                    std::cerr << "--crf must be in range 0-51, got: " << crf << std::endl;
                    result.shouldExit = true;
                    result.exitCode = 1;
                    return result;
                }
                job.encode.crf = crf;
            } catch (...) {
                std::cerr << "Invalid --crf value: " << value << std::endl;
                result.shouldExit = true;
                result.exitCode = 1;
                return result;
            }
            continue;
        }

        if (arg == "--preset") {
            std::string value;
            if (!requireValue(argc, argv, i, value, arg, result)) {
                return result;
            }
            job.encode.preset = value;
            continue;
        }

        if (arg == "--preview") {
            job.previewMode = true;
            continue;
        }

        if (arg == "--preview-duration") {
            std::string value;
            if (!requireValue(argc, argv, i, value, arg, result)) {
                return result;
            }
            try {
                const double dur = std::stod(value);
                if (dur <= 0.0 || dur > 300.0) {
                    std::cerr << "--preview-duration must be in range (0, 300], got: "
                              << dur << std::endl;
                    result.shouldExit = true;
                    result.exitCode = 1;
                    return result;
                }
                job.previewDurationSec = dur;
                job.previewMode = true;  // Implies preview mode
            } catch (...) {
                std::cerr << "Invalid --preview-duration value: " << value << std::endl;
                result.shouldExit = true;
                result.exitCode = 1;
                return result;
            }
            continue;
        }

        std::cerr << "Unknown argument: " << arg << std::endl;
        result.shouldExit = true;
        result.exitCode = 1;
        return result;
    }

    if (result.listBackends) {
        result.shouldExit = false;
        return result;
    }

    if (job.inputPath.empty()) {
        std::cerr << "--input is required." << std::endl;
        result.shouldExit = true;
        result.exitCode = 1;
        return result;
    }

    if (job.outputPath.empty()) {
        std::filesystem::path inputP(job.inputPath);
        std::filesystem::path outputDir = inputP.parent_path() / "output";
        std::filesystem::create_directories(outputDir);
        std::string stem = inputP.stem().string();
        std::string suffix = job.previewMode ? "_preview" : "_enhanced";
        job.outputPath = (outputDir / (stem + suffix + ".mp4")).string();
    }

    result.job = job;
    return result;
}

}  // namespace ave
