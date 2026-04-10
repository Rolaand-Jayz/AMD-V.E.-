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

std::string toLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
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
              << "  --stage denoise:vpy_script_path=/path/custom_pipeline.vpy --backend vapoursynth\n"
              << "  --stage stereo_3d:model=depth-anything-v2-small-fp16,format=full_sbs,divergence=2.0\n"
              << "  --stage interpolate:fps=60\n"
              << "\n"
              << "Ordering guarantees:\n"
              << "  restore/deartifact/cleanup before upscale+sharpen\n"
              << "  stereo_3d runs after sharpening and before interpolation\n"
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
