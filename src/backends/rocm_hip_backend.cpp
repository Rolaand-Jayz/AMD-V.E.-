#include "ave/backends/rocm_hip_backend.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "ave/model_catalog.hpp"
#include "ave/model_manager.hpp"
#include "ave/process_observer.hpp"
#include "ave/rgb_video_loop.hpp"
#include "ave/runtime_diagnostics.hpp"
#include "ave/tensor_contract.hpp"
#include "ave/video_probe.hpp"

#ifdef AVE_HAVE_ONNXRUNTIME_ROCM
#  include <onnxruntime/onnxruntime_cxx_api.h>
#endif

namespace ave {
namespace {

bool pathHasMxrExtension(const std::string& path) {
    if (path.size() < 4) {
        return false;
    }
    const std::string ext = path.substr(path.size() - 4);
    return ext == ".mxr" || ext == ".MXR";
}

std::optional<int> readNonNegativeEnvInt(const char* name) {
    if (name == nullptr) {
        return std::nullopt;
    }
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') {
        return std::nullopt;
    }
    char* end = nullptr;
    const long value = std::strtol(raw, &end, 10);
    if (end == raw || *end != '\0' || value < 0L) {
        return std::nullopt;
    }
    return static_cast<int>(value);
}

std::optional<int> preferredRocmGpuIndexFromEnv() {
    if (const auto explicitIndex = readNonNegativeEnvInt("AVE_ORT_ROCM_DEVICE_INDEX");
        explicitIndex.has_value()) {
        return explicitIndex;
    }
    return preferredAmdDeviceIndexFromEnv();
}

bool stageSupportedByRocmHip(const StageKind kind) {
    return kind != StageKind::Interpolate && kind != StageKind::Stereo3D;
}

std::string defaultModelIdFor(const StageKind kind) {
    if (const auto* preferred = preferredBackendModelForStage(kind, BackendType::RocmHip);
        preferred != nullptr) {
        return preferred->id;
    }
    return {};
}

std::string resolveModelId(const EnhancementStage& stage) {
    const auto it = stage.params.find("model");
    if (it != stage.params.end()) {
        if (const auto* modelId = std::get_if<std::string>(&it->second);
            modelId != nullptr && !modelId->empty()) {
            return *modelId;
        }
    }
    return defaultModelIdFor(stage.kind);
}

std::string resolveModelPath(const EnhancementStage& stage) {
    const auto pathIt = stage.params.find("model_path");
    if (pathIt != stage.params.end()) {
        if (const auto* explicitPath = std::get_if<std::string>(&pathIt->second);
            explicitPath != nullptr && !explicitPath->empty()) {
            return *explicitPath;
        }
    }

    const std::string modelId = resolveModelId(stage);
    if (modelId.empty()) {
        return {};
    }

    ModelManager manager;
    const auto managed = manager.findModel(modelId);
    if (!managed.has_value()) {
        return {};
    }

    if (!managed->downloadedPath.empty() &&
        managed->downloadedPath != "(builtin)" &&
        !pathHasMxrExtension(managed->downloadedPath)) {
        return managed->downloadedPath;
    }
    return {};
}

std::string cacheKeyForStage(const EnhancementStage& stage) {
    const std::string modelPath = resolveModelPath(stage);
    if (!modelPath.empty()) {
        return modelPath;
    }
    return resolveModelId(stage);
}

std::string toLowerCopy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](const unsigned char ch) {
                       return static_cast<char>(std::tolower(ch));
                   });
    return value;
}

bool onnxRuntimeProviderLooksAvailable() {
#ifdef AVE_HAVE_ONNXRUNTIME_ROCM
    try {
        const auto providers = Ort::GetAvailableProviders();
        return std::any_of(providers.begin(), providers.end(),
                           [](const std::string& provider) {
                               return toLowerCopy(provider).find("rocm") != std::string::npos;
                           });
    } catch (...) {
        return false;
    }
#else
    return false;
#endif
}

TensorDtype tensorDtypeFromOnnxType(const ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            return TensorDtype::Fp32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            return TensorDtype::Fp16;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
            return TensorDtype::Bf16;
        default:
            return TensorDtype::Unknown;
    }
}

std::string onnxTensorTypeName(const ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            return "float";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            return "float16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
            return "bfloat16";
        default:
            return "unsupported";
    }
}

struct ImageTensorShape {
    TensorLayout layout = TensorLayout::Unknown;
    int batch = 1;
    int channels = 0;
    int height = 0;
    int width = 0;
};

bool resolveImageTensorShape(const std::vector<std::int64_t>& dims,
                             const TensorLayout layout,
                             const int fallbackHeight,
                             const int fallbackWidth,
                             ImageTensorShape& shape,
                             std::string& error) {
    shape = {};
    shape.layout = layout;

    const auto expectDynamicOrMatch = [&](const std::int64_t dim,
                                          const int actual,
                                          const char* axisName,
                                          int& outValue) -> bool {
        if (dim <= 0) {
            outValue = actual;
            return true;
        }
        if (actual > 0 && dim != actual) {
            error = std::string("Model requires a fixed ") + axisName + " of "
                + std::to_string(dim) + ", but the current frame is " + std::to_string(actual) + ".";
            return false;
        }
        outValue = static_cast<int>(dim);
        return true;
    };

    if (dims.size() == 4) {
        if (layout == TensorLayout::NCHW) {
            shape.batch = dims[0] > 0 ? static_cast<int>(dims[0]) : 1;
            shape.channels = dims[1] > 0 ? static_cast<int>(dims[1]) : 3;
            if (!expectDynamicOrMatch(dims[2], fallbackHeight, "height", shape.height) ||
                !expectDynamicOrMatch(dims[3], fallbackWidth, "width", shape.width)) {
                return false;
            }
        } else if (layout == TensorLayout::NHWC) {
            shape.batch = dims[0] > 0 ? static_cast<int>(dims[0]) : 1;
            if (!expectDynamicOrMatch(dims[1], fallbackHeight, "height", shape.height) ||
                !expectDynamicOrMatch(dims[2], fallbackWidth, "width", shape.width)) {
                return false;
            }
            shape.channels = dims[3] > 0 ? static_cast<int>(dims[3]) : 3;
        } else {
            error = "Unsupported 4D tensor layout for image inference.";
            return false;
        }
    } else if (dims.size() == 3) {
        if (layout == TensorLayout::CHW) {
            shape.channels = dims[0] > 0 ? static_cast<int>(dims[0]) : 3;
            if (!expectDynamicOrMatch(dims[1], fallbackHeight, "height", shape.height) ||
                !expectDynamicOrMatch(dims[2], fallbackWidth, "width", shape.width)) {
                return false;
            }
        } else if (layout == TensorLayout::HWC) {
            if (!expectDynamicOrMatch(dims[0], fallbackHeight, "height", shape.height) ||
                !expectDynamicOrMatch(dims[1], fallbackWidth, "width", shape.width)) {
                return false;
            }
            shape.channels = dims[2] > 0 ? static_cast<int>(dims[2]) : 3;
        } else {
            error = "Unsupported 3D tensor layout for image inference.";
            return false;
        }
    } else {
        error = "Only 3D/4D image tensors are supported by the ROCm/HIP backend.";
        return false;
    }

    if (shape.batch != 1) {
        error = "Only batch size 1 image models are supported by the ROCm/HIP backend.";
        return false;
    }
    if (shape.channels != 3) {
        error = "Only 3-channel RGB image models are supported by the ROCm/HIP backend.";
        return false;
    }
    if (shape.height <= 0 || shape.width <= 0) {
        error = "Resolved tensor dimensions are invalid for image inference.";
        return false;
    }
    return true;
}

template <typename T>
T normalisedTensorValue(const float value) {
    return static_cast<T>(value);
}

#ifdef AVE_HAVE_ONNXRUNTIME_ROCM
template <>
Ort::Float16_t normalisedTensorValue<Ort::Float16_t>(const float value) {
    return Ort::Float16_t(value);
}
#endif

template <typename T>
void fillTensorFromRgb(const std::vector<std::uint8_t>& rgb,
                       const ImageTensorShape& shape,
                       std::vector<T>& tensor) {
    const std::size_t elementCount = static_cast<std::size_t>(shape.height) *
                                     static_cast<std::size_t>(shape.width) * 3u;
    tensor.resize(elementCount);

    const auto pixelValue = [&](const int y, const int x, const int channel) {
        const std::size_t index = (static_cast<std::size_t>(y) * static_cast<std::size_t>(shape.width)
                                   + static_cast<std::size_t>(x)) * 3u
                                  + static_cast<std::size_t>(channel);
        return static_cast<float>(rgb[index]) / 255.0f;
    };

    if (shape.layout == TensorLayout::NCHW || shape.layout == TensorLayout::CHW) {
        for (int channel = 0; channel < 3; ++channel) {
            for (int y = 0; y < shape.height; ++y) {
                for (int x = 0; x < shape.width; ++x) {
                    const std::size_t index = (static_cast<std::size_t>(channel) *
                                               static_cast<std::size_t>(shape.height)
                                               + static_cast<std::size_t>(y)) *
                                                  static_cast<std::size_t>(shape.width)
                                              + static_cast<std::size_t>(x);
                    tensor[index] = normalisedTensorValue<T>(pixelValue(y, x, channel));
                }
            }
        }
        return;
    }

    for (int y = 0; y < shape.height; ++y) {
        for (int x = 0; x < shape.width; ++x) {
            for (int channel = 0; channel < 3; ++channel) {
                const std::size_t index = (static_cast<std::size_t>(y) *
                                           static_cast<std::size_t>(shape.width)
                                           + static_cast<std::size_t>(x)) * 3u
                                          + static_cast<std::size_t>(channel);
                tensor[index] = normalisedTensorValue<T>(pixelValue(y, x, channel));
            }
        }
    }
}

template <typename T, typename ToFloatFn>
void tensorToRgb(const T* tensorData,
                 const ImageTensorShape& shape,
                 std::vector<std::uint8_t>& rgb,
                 ToFloatFn&& toFloat) {
    rgb.resize(static_cast<std::size_t>(shape.width) *
               static_cast<std::size_t>(shape.height) * 3u);

    const auto tensorIndex = [&](const int y, const int x, const int channel) -> std::size_t {
        if (shape.layout == TensorLayout::NCHW || shape.layout == TensorLayout::CHW) {
            return (static_cast<std::size_t>(channel) * static_cast<std::size_t>(shape.height)
                    + static_cast<std::size_t>(y)) * static_cast<std::size_t>(shape.width)
                   + static_cast<std::size_t>(x);
        }
        return (static_cast<std::size_t>(y) * static_cast<std::size_t>(shape.width)
                + static_cast<std::size_t>(x)) * 3u
               + static_cast<std::size_t>(channel);
    };

    for (int y = 0; y < shape.height; ++y) {
        for (int x = 0; x < shape.width; ++x) {
            for (int channel = 0; channel < 3; ++channel) {
                float value = toFloat(tensorData[tensorIndex(y, x, channel)]) * 255.0f;
                value = std::clamp(value, 0.0f, 255.0f);
                const std::size_t rgbIndex = (static_cast<std::size_t>(y) *
                                              static_cast<std::size_t>(shape.width)
                                              + static_cast<std::size_t>(x)) * 3u
                                             + static_cast<std::size_t>(channel);
                rgb[rgbIndex] = static_cast<std::uint8_t>(std::lround(value));
            }
        }
    }
}

bool resizeRgbBufferBilinear(const std::vector<std::uint8_t>& input,
                             const int inputWidth,
                             const int inputHeight,
                             const int outputWidth,
                             const int outputHeight,
                             std::vector<std::uint8_t>& output,
                             std::string& error) {
    if (inputWidth <= 0 || inputHeight <= 0 || outputWidth <= 0 || outputHeight <= 0) {
        error = "Cannot resize an RGB buffer with non-positive dimensions.";
        return false;
    }
    if (static_cast<std::size_t>(inputWidth) * static_cast<std::size_t>(inputHeight) * 3u !=
        input.size()) {
        error = "RGB resize input buffer does not match the declared dimensions.";
        return false;
    }
    if (inputWidth == outputWidth && inputHeight == outputHeight) {
        output = input;
        return true;
    }

    output.resize(static_cast<std::size_t>(outputWidth) *
                  static_cast<std::size_t>(outputHeight) * 3u);

    const float xScale = outputWidth > 1
        ? static_cast<float>(inputWidth - 1) / static_cast<float>(outputWidth - 1)
        : 0.0f;
    const float yScale = outputHeight > 1
        ? static_cast<float>(inputHeight - 1) / static_cast<float>(outputHeight - 1)
        : 0.0f;

    const auto sample = [&](const int x, const int y, const int channel) {
        const std::size_t index = (static_cast<std::size_t>(y) * static_cast<std::size_t>(inputWidth)
                                   + static_cast<std::size_t>(x)) * 3u
                                  + static_cast<std::size_t>(channel);
        return static_cast<float>(input[index]);
    };

    for (int y = 0; y < outputHeight; ++y) {
        const float sourceY = yScale * static_cast<float>(y);
        const int y0 = static_cast<int>(std::floor(sourceY));
        const int y1 = std::min(y0 + 1, inputHeight - 1);
        const float yLerp = sourceY - static_cast<float>(y0);

        for (int x = 0; x < outputWidth; ++x) {
            const float sourceX = xScale * static_cast<float>(x);
            const int x0 = static_cast<int>(std::floor(sourceX));
            const int x1 = std::min(x0 + 1, inputWidth - 1);
            const float xLerp = sourceX - static_cast<float>(x0);

            for (int channel = 0; channel < 3; ++channel) {
                const float top = sample(x0, y0, channel) * (1.0f - xLerp) +
                                  sample(x1, y0, channel) * xLerp;
                const float bottom = sample(x0, y1, channel) * (1.0f - xLerp) +
                                     sample(x1, y1, channel) * xLerp;
                const float value = top * (1.0f - yLerp) + bottom * yLerp;
                const std::size_t outIndex = (static_cast<std::size_t>(y) *
                                              static_cast<std::size_t>(outputWidth)
                                              + static_cast<std::size_t>(x)) * 3u
                                             + static_cast<std::size_t>(channel);
                output[outIndex] = static_cast<std::uint8_t>(
                    std::clamp(std::lround(value), 0L, 255L));
            }
        }
    }

    return true;
}

void resolveDesiredOutputSize(const EnhancementStage& stage,
                              const int inputWidth,
                              const int inputHeight,
                              int& outputWidth,
                              int& outputHeight) {
    outputWidth = inputWidth;
    outputHeight = inputHeight;
    if (stage.kind != StageKind::Upscale) {
        return;
    }

    std::int64_t widthValue = 0;
    std::int64_t heightValue = 0;
    if (tryGetInt(stage, StageKind::Upscale, "width", widthValue) && widthValue > 0) {
        outputWidth = static_cast<int>(widthValue);
    }
    if (tryGetInt(stage, StageKind::Upscale, "height", heightValue) && heightValue > 0) {
        outputHeight = static_cast<int>(heightValue);
    }

    if (outputWidth == inputWidth || outputHeight == inputHeight) {
        const std::string modelId = resolveModelId(stage);
        if (const auto* entry = catalogEntryById(modelId);
            entry != nullptr && entry->scale > 1) {
            if (outputWidth == inputWidth) {
                outputWidth = inputWidth * entry->scale;
            }
            if (outputHeight == inputHeight) {
                outputHeight = inputHeight * entry->scale;
            }
        }
    }
}

}  // namespace

#ifdef AVE_HAVE_ONNXRUNTIME_ROCM

struct OrtSessionBundle {
    Ort::Session session{nullptr};
    std::string modelId;
    std::string modelPath;
    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    TensorLayout inputLayout = TensorLayout::Unknown;
    TensorDtype inputDtype = TensorDtype::Unknown;
    std::vector<std::int64_t> inputDims;
    TensorLayout outputLayoutHint = TensorLayout::Unknown;
    TensorDtype outputDtype = TensorDtype::Unknown;
    std::vector<std::int64_t> outputDims;
};

struct RocmHipBackend::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "ave-rocm-hip"};
    bool initialised = false;
    int gpuIdx = 0;
    std::mutex mtx;
    std::unordered_map<std::string, std::unique_ptr<OrtSessionBundle>> sessions;

    bool loadSession(const std::string& cacheKey,
                     const std::string& modelId,
                     const std::string& modelPath,
                     std::string& error) {
        if (sessions.count(cacheKey) > 0) {
            return true;
        }
        if (modelPath.empty()) {
            error = "No ONNX model path was resolved for ROCm/HIP inference.";
            return false;
        }
        if (!std::filesystem::exists(modelPath)) {
            error = "Resolved ONNX model path does not exist: " + modelPath;
            return false;
        }

        auto bundle = std::make_unique<OrtSessionBundle>();
        bundle->modelId = modelId;
        bundle->modelPath = modelPath;

        try {
            Ort::SessionOptions options;
            options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

            OrtROCMProviderOptions providerOptions;
            providerOptions.device_id = gpuIdx;
            providerOptions.do_copy_in_default_stream = 1;
            options.AppendExecutionProvider_ROCM(providerOptions);

            bundle->session = Ort::Session(env, modelPath.c_str(), options);
        } catch (const std::exception& ex) {
            error = "Unable to create ONNX Runtime ROCm session for '" + modelPath
                + "': " + ex.what();
            return false;
        }

        const std::size_t inputCount = bundle->session.GetInputCount();
        const std::size_t outputCount = bundle->session.GetOutputCount();
        if (inputCount != 1 || outputCount != 1) {
            error = "ROCm/HIP backend currently supports single-input, single-output image models only.";
            return false;
        }

        Ort::AllocatorWithDefaultOptions allocator;
        auto inputName = bundle->session.GetInputNameAllocated(0, allocator);
        auto outputName = bundle->session.GetOutputNameAllocated(0, allocator);
        bundle->inputNames.push_back(inputName != nullptr ? inputName.get() : "input");
        bundle->outputNames.push_back(outputName != nullptr ? outputName.get() : "output");

        const auto inputTypeInfo = bundle->session.GetInputTypeInfo(0);
        if (inputTypeInfo.GetONNXType() != ONNX_TYPE_TENSOR) {
            error = "ROCm/HIP backend only supports tensor model inputs.";
            return false;
        }
        const auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
        bundle->inputDims = inputTensorInfo.GetShape();
        bundle->inputLayout = inferTensorLayout(bundle->inputDims);
        bundle->inputDtype = tensorDtypeFromOnnxType(inputTensorInfo.GetElementType());
        if (bundle->inputLayout == TensorLayout::Unknown) {
            error = "Unable to determine the ONNX input tensor layout for '" + modelPath + "'.";
            return false;
        }
        if (bundle->inputDtype != TensorDtype::Fp32 && bundle->inputDtype != TensorDtype::Fp16) {
            error = "Unsupported ONNX input tensor type for ROCm/HIP backend: "
                + onnxTensorTypeName(inputTensorInfo.GetElementType());
            return false;
        }

        const auto outputTypeInfo = bundle->session.GetOutputTypeInfo(0);
        if (outputTypeInfo.GetONNXType() != ONNX_TYPE_TENSOR) {
            error = "ROCm/HIP backend only supports tensor model outputs.";
            return false;
        }
        const auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
        bundle->outputDims = outputTensorInfo.GetShape();
        bundle->outputLayoutHint = inferTensorLayout(bundle->outputDims);
        bundle->outputDtype = tensorDtypeFromOnnxType(outputTensorInfo.GetElementType());
        if (bundle->outputLayoutHint == TensorLayout::Unknown && !bundle->outputDims.empty()) {
            bundle->outputLayoutHint = bundle->inputLayout;
        }
        if (bundle->outputDtype != TensorDtype::Fp32 && bundle->outputDtype != TensorDtype::Fp16) {
            error = "Unsupported ONNX output tensor type for ROCm/HIP backend: "
                + onnxTensorTypeName(outputTensorInfo.GetElementType());
            return false;
        }

        sessions[cacheKey] = std::move(bundle);
        return true;
    }
};

#else

struct RocmHipBackend::Impl {
    bool initialised = false;
    int gpuIdx = 0;
    std::mutex mtx;

    bool loadSession(const std::string&, const std::string&, const std::string&, std::string& error) {
        error = "ROCm/HIP ONNX Runtime backend is not compiled in this build (AVE_HAVE_ONNXRUNTIME_ROCM=OFF).";
        return false;
    }
};

#endif

RocmHipBackend::RocmHipBackend() : impl_(std::make_unique<Impl>()) {}
RocmHipBackend::~RocmHipBackend() = default;

BackendType RocmHipBackend::type() const {
    return BackendType::RocmHip;
}

std::string RocmHipBackend::name() const {
    return "ROCm/HIP (ONNX Runtime)";
}

bool RocmHipBackend::isAvailable(std::string& reason) const {
#ifndef AVE_HAVE_ONNXRUNTIME_ROCM
    reason = "ROCm/HIP ONNX Runtime backend is not compiled in this build (AVE_HAVE_ONNXRUNTIME_ROCM=OFF).";
    return false;
#else
    const auto snapshot = probeAmdRuntimeSnapshot();
    if (!snapshot.rocmRootPresent) {
        reason = "ROCm root was not detected.";
        return false;
    }
    if (!snapshot.kfdDevicePresent || !snapshot.kfdAccessible) {
        reason = "ROCm kernel device access is unavailable (/dev/kfd missing or inaccessible).";
        return false;
    }
    if (!snapshot.hipRuntimePresent) {
        reason = "HIP runtime library (libamdhip64.so) was not detected.";
        return false;
    }
    if (!onnxRuntimeProviderLooksAvailable()) {
        reason = "ONNX Runtime was found, but ROCMExecutionProvider is not available in this runtime.";
        return false;
    }

    try {
        Ort::SessionOptions options;
        OrtROCMProviderOptions providerOptions;
        providerOptions.device_id = preferredRocmGpuIndexFromEnv().value_or(0);
        options.AppendExecutionProvider_ROCM(providerOptions);
    } catch (const std::exception& ex) {
        reason = "ONNX Runtime ROCm execution provider could not be initialised: " + std::string(ex.what());
        return false;
    }

    reason = "ONNX Runtime ROCm execution provider detected and ready.";
    return true;
#endif
}

bool RocmHipBackend::initialize(std::string& error) {
    std::string reason;
    if (!isAvailable(reason)) {
        error = "ROCm/HIP init: " + reason;
        return false;
    }
    impl_->gpuIdx = preferredRocmGpuIndexFromEnv().value_or(0);
    impl_->initialised = true;
    std::cout << "[backend] ROCm/HIP ONNX Runtime initialised on GPU " << impl_->gpuIdx << std::endl;
    return true;
}

StageResult RocmHipBackend::runStage(const EnhancementStage& stage, std::string& error) {
    if (!stageSupportedByRocmHip(stage.kind)) {
        std::cout << "[rocm-hip] " << toString(stage.kind)
                  << " is not currently supported by the ROCm/HIP backend — deferring."
                  << std::endl;
        return StageResult::Deferred;
    }

    if (!impl_->initialised) {
        std::string initError;
        if (!initialize(initError)) {
            error = std::move(initError);
            return StageResult::Deferred;
        }
    }

    const std::string modelId = resolveModelId(stage);
    const std::string modelPath = resolveModelPath(stage);
    if (modelId.empty() || modelPath.empty()) {
        std::cout << "[rocm-hip] no usable ONNX model resolved for " << toString(stage.kind)
                  << " — deferring to FFmpeg filter chain." << std::endl;
        return StageResult::Deferred;
    }

    const std::string cacheKey = cacheKeyForStage(stage);
    {
        std::lock_guard<std::mutex> lock(impl_->mtx);
        if (!impl_->loadSession(cacheKey, modelId, modelPath, error)) {
            std::cerr << "[rocm-hip] model load failed for stage '" << toString(stage.kind)
                      << "': " << error << "\n  → Deferring to FFmpeg filter chain." << std::endl;
            error.clear();
            return StageResult::Deferred;
        }
    }

    std::cout << "[rocm-hip] stage=" << toString(stage.kind) << " model=" << modelId
              << " — AI inference will run via processVideoFile() during encode." << std::endl;
    return StageResult::Deferred;
}

StageResult RocmHipBackend::processVideoFile(
        const EnhancementStage& stage,
        const std::string& inputVideo,
        const std::string& outputVideo,
        const FrameProgressCb& progressCb,
        std::string& error,
        const ProcessVideoOptions& opts) {
#ifndef AVE_HAVE_ONNXRUNTIME_ROCM
    (void)stage;
    (void)inputVideo;
    (void)outputVideo;
    (void)progressCb;
    (void)opts;
    error = "ROCm/HIP ONNX Runtime backend is not compiled in this build (AVE_HAVE_ONNXRUNTIME_ROCM=OFF).";
    return StageResult::Deferred;
#else
    if (!stageSupportedByRocmHip(stage.kind)) {
        error.clear();
        return StageResult::Deferred;
    }
    if (!impl_->initialised) {
        if (!initialize(error)) {
            return StageResult::Deferred;
        }
    }

    const std::string modelId = resolveModelId(stage);
    const std::string modelPath = resolveModelPath(stage);
    if (modelId.empty() || modelPath.empty()) {
        error.clear();
        return StageResult::Deferred;
    }

    OrtSessionBundle* bundle = nullptr;
    {
        const std::string cacheKey = cacheKeyForStage(stage);
        std::lock_guard<std::mutex> lock(impl_->mtx);
        if (!impl_->loadSession(cacheKey, modelId, modelPath, error)) {
            error.clear();
            return StageResult::Deferred;
        }
        bundle = impl_->sessions.at(cacheKey).get();
    }

    const auto probe = probeVideoStream(inputVideo, error);
    if (!probe.has_value()) {
        return StageResult::Error;
    }
    const int inputWidth = static_cast<int>(probe->width);
    const int inputHeight = static_cast<int>(probe->height);
    const double inputFps = probe->effectiveFrameRate(25.0);
    const int totalFrames = static_cast<int>(probe->estimatedFrameCount());

    int desiredOutputWidth = inputWidth;
    int desiredOutputHeight = inputHeight;
    resolveDesiredOutputSize(stage, inputWidth, inputHeight,
                             desiredOutputWidth, desiredOutputHeight);

    ImageTensorShape inputTensorShape;
    if (!resolveImageTensorShape(bundle->inputDims,
                                 bundle->inputLayout,
                                 inputHeight,
                                 inputWidth,
                                 inputTensorShape,
                                 error)) {
        return StageResult::Error;
    }

    std::cout << "[rocm-hip] processVideoFile: " << inputWidth << "x" << inputHeight
              << " → " << desiredOutputWidth << "x" << desiredOutputHeight
              << " fps=" << inputFps
              << " frames=" << totalFrames << std::endl;

    RgbVideoLoopOptions loopOptions;
    loopOptions.inputVideo = inputVideo;
    loopOptions.outputVideo = outputVideo;
    loopOptions.inputWidth = inputWidth;
    loopOptions.inputHeight = inputHeight;
    loopOptions.outputWidth = desiredOutputWidth;
    loopOptions.outputHeight = desiredOutputHeight;
    loopOptions.fps = inputFps;
    loopOptions.fallbackFps = 25.0;
    loopOptions.totalFrames = totalFrames;
    loopOptions.backendTag = "rocm-hip";
    loopOptions.progressLabel = "ROCm/HIP";
    loopOptions.noFramesError = "No frames decoded from input video.";
    loopOptions.preferredSourceMode = frame_io::RgbVideoSourceMode::VulkanTransfer;
    loopOptions.allowSourceFallback = true;
    loopOptions.processOptions = opts;

    const auto loopResult = runRgbVideoLoop(
        loopOptions,
        [&](const std::vector<std::uint8_t>& inputRgb,
            std::vector<std::uint8_t>& outputRgb,
            const int /*frameIdx*/,
            std::string& loopError) {
            std::vector<float> floatInput;
            std::vector<Ort::Float16_t> float16Input;
            std::vector<std::int64_t> inputDims = bundle->inputDims;
            if (inputDims.size() == 4) {
                if (bundle->inputLayout == TensorLayout::NCHW) {
                    inputDims[0] = 1;
                    inputDims[1] = 3;
                    inputDims[2] = inputTensorShape.height;
                    inputDims[3] = inputTensorShape.width;
                } else {
                    inputDims[0] = 1;
                    inputDims[1] = inputTensorShape.height;
                    inputDims[2] = inputTensorShape.width;
                    inputDims[3] = 3;
                }
            } else if (inputDims.size() == 3) {
                if (bundle->inputLayout == TensorLayout::CHW) {
                    inputDims[0] = 3;
                    inputDims[1] = inputTensorShape.height;
                    inputDims[2] = inputTensorShape.width;
                } else {
                    inputDims[0] = inputTensorShape.height;
                    inputDims[1] = inputTensorShape.width;
                    inputDims[2] = 3;
                }
            }

            Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value inputTensor{nullptr};
            if (bundle->inputDtype == TensorDtype::Fp16) {
                fillTensorFromRgb(inputRgb, inputTensorShape, float16Input);
                inputTensor = Ort::Value::CreateTensor<Ort::Float16_t>(
                    memoryInfo,
                    float16Input.data(),
                    float16Input.size(),
                    inputDims.data(),
                    inputDims.size());
            } else {
                fillTensorFromRgb(inputRgb, inputTensorShape, floatInput);
                inputTensor = Ort::Value::CreateTensor<float>(
                    memoryInfo,
                    floatInput.data(),
                    floatInput.size(),
                    inputDims.data(),
                    inputDims.size());
            }

            std::array<const char*, 1> inputNames{bundle->inputNames.front().c_str()};
            std::array<const char*, 1> outputNames{bundle->outputNames.front().c_str()};
            std::vector<Ort::Value> outputs;
            try {
                Ort::RunOptions runOptions;
                outputs = bundle->session.Run(runOptions,
                                              inputNames.data(),
                                              &inputTensor,
                                              inputNames.size(),
                                              outputNames.data(),
                                              outputNames.size());
            } catch (const std::exception& ex) {
                loopError = "ONNX Runtime ROCm inference failed: " + std::string(ex.what());
                return false;
            }

            if (outputs.empty()) {
                loopError = "ONNX Runtime returned no output tensors.";
                return false;
            }

            Ort::Value& outputTensor = outputs.front();
            const auto outputInfo = outputTensor.GetTensorTypeAndShapeInfo();
            const auto actualDims = outputInfo.GetShape();
            TensorLayout actualLayout = inferTensorLayout(actualDims);
            if (actualLayout == TensorLayout::Unknown) {
                actualLayout = bundle->outputLayoutHint;
            }

            ImageTensorShape actualOutputShape;
            if (!resolveImageTensorShape(actualDims,
                                         actualLayout,
                                         0,
                                         0,
                                         actualOutputShape,
                                         loopError)) {
                return false;
            }

            std::vector<std::uint8_t> modelOutputRgb;
            if (bundle->outputDtype == TensorDtype::Fp16) {
                auto* rawTensorData = outputTensor.GetTensorMutableData<Ort::Float16_t>();
                tensorToRgb(rawTensorData,
                            actualOutputShape,
                            modelOutputRgb,
                            [](const Ort::Float16_t value) { return value.ToFloat(); });
            } else {
                auto* rawTensorData = outputTensor.GetTensorMutableData<float>();
                tensorToRgb(rawTensorData,
                            actualOutputShape,
                            modelOutputRgb,
                            [](const float value) { return value; });
            }

            if (actualOutputShape.width != desiredOutputWidth ||
                actualOutputShape.height != desiredOutputHeight) {
                return resizeRgbBufferBilinear(modelOutputRgb,
                                               actualOutputShape.width,
                                               actualOutputShape.height,
                                               desiredOutputWidth,
                                               desiredOutputHeight,
                                               outputRgb,
                                               loopError);
            }

            outputRgb = std::move(modelOutputRgb);
            return true;
        },
        progressCb,
        error);

    if (loopResult.stageResult != StageResult::Processed) {
        return loopResult.stageResult;
    }

    {
        std::lock_guard<std::mutex> lock(impl_->mtx);
        impl_->sessions.clear();
    }

    std::cout << "[rocm-hip] processVideoFile: processed " << loopResult.frameCount
              << " frames → " << outputVideo << std::endl;
    return loopResult.stageResult;
#endif
}

}  // namespace ave