#include "ave/backend.hpp"

namespace ave {

std::string toString(BackendType type) {
    switch (type) {
        case BackendType::Auto:
            return "auto";
        case BackendType::MiGraphX:
            return "migraphx";
        case BackendType::NcnnVulkan:
            return "ncnn-vulkan";
    }
    return "unknown";
}

}  // namespace ave
