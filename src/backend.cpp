#include "ave/backend.hpp"

namespace ave {

std::string toString(BackendType type) {
    switch (type) {
        case BackendType::Auto:
            return "auto";
        case BackendType::MiGraphX:
            return "migraphx";
        case BackendType::RocmHip:
            return "rocm-hip";
        case BackendType::NcnnVulkan:
            return "ncnn-vulkan";
        case BackendType::VulkanCompute:
            return "vulkan";
        case BackendType::VapourSynth:
            return "vapoursynth";
        case BackendType::GlslShader:
            return "glsl";
    }
    return "unknown";
}

}  // namespace ave
