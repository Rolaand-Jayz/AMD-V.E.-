#include "ave/backend_manager.hpp"

#include <iostream>
#include <memory>
#include <sstream>

#include "ave/backends/glsl_shader_backend.hpp"
#include "ave/backends/migraphx_backend.hpp"
#include "ave/backends/ncnn_vulkan_backend.hpp"
#include "ave/backends/vapoursynth_backend.hpp"
#include "ave/backends/vulkan_compute_backend.hpp"
#include "ave/interop_bridge.hpp"

namespace ave {

std::vector<BackendInfo> BackendManager::probeBackends() const {
    std::vector<BackendInfo> infos;

    {
        MiGraphXBackend backend;
        std::string detail;
        const bool ok = backend.isAvailable(detail);
        infos.push_back(BackendInfo{BackendType::MiGraphX, backend.name(), ok, detail});
    }

    {
        VulkanComputeBackend backend;
        std::string detail;
        const bool ok = backend.isAvailable(detail);
        infos.push_back(BackendInfo{BackendType::VulkanCompute, backend.name(), ok, detail});
    }

    {
        NcnnVulkanBackend backend;
        std::string detail;
        const bool ok = backend.isAvailable(detail);
        infos.push_back(BackendInfo{BackendType::NcnnVulkan, backend.name(), ok, detail});
    }

    {
        VapourSynthBackend backend;
        std::string detail;
        const bool ok = backend.isAvailable(detail);
        infos.push_back(BackendInfo{BackendType::VapourSynth, backend.name(), ok, detail});
    }

    {
        GlslShaderBackend backend;
        std::string detail;
        const bool ok = backend.isAvailable(detail);
        infos.push_back(BackendInfo{BackendType::GlslShader, backend.name(), ok, detail});
    }

    return infos;
}

std::unique_ptr<IAcceleratorBackend> BackendManager::createBackend(BackendType requested,
                                                                    std::string& selectionSummary) const {
    MiGraphXBackend migraphx;
    VulkanComputeBackend vkCompute;
    NcnnVulkanBackend ncnn;

    std::string migraphxReason;
    const bool migraphxAvailable = migraphx.isAvailable(migraphxReason);

    std::string vkReason;
    const bool vkAvailable = vkCompute.isAvailable(vkReason);

    std::string ncnnReason;
    const bool ncnnAvailable = ncnn.isAvailable(ncnnReason);

    auto selectMigraphx = [&selectionSummary]() -> std::unique_ptr<IAcceleratorBackend> {
        selectionSummary = "Selected MiGraphX (ROCm).";
        // Log InteropBridge status alongside backend selection so the
        // pipeline mode (GPU interop vs CPU staging) appears in startup logs.
        InteropBridge bridge;
        std::string bridgeReason;
        const bool bridgeUp = bridge.isAvailable(bridgeReason);
        std::cout << "[backend-manager] interop-bridge: "
                  << (bridgeUp ? "GPU interop available" : bridgeReason)
                  << std::endl;
        return std::make_unique<MiGraphXBackend>();
    };

    // Explicit requests must fail honestly. Only Auto is allowed to fall back.
    if (requested == BackendType::MiGraphX) {
        if (migraphxAvailable) {
            return selectMigraphx();
        }
        selectionSummary = "MiGraphX explicitly requested but unavailable: " + migraphxReason;
        return nullptr;
    }

    if (requested == BackendType::VulkanCompute) {
        if (vkAvailable) {
            selectionSummary = "Selected Vulkan Compute.";
            return std::make_unique<VulkanComputeBackend>();
        }
        selectionSummary = "Vulkan Compute explicitly requested but unavailable: " + vkReason;
        return nullptr;
    }

    if (requested == BackendType::NcnnVulkan) {
        if (ncnnAvailable) {
            selectionSummary = "Selected NCNN Vulkan.";
            return std::make_unique<NcnnVulkanBackend>();
        }
        selectionSummary = "NCNN Vulkan explicitly requested but unavailable: " + ncnnReason;
        return nullptr;
    }

    if (requested == BackendType::VapourSynth) {
        VapourSynthBackend vs;
        std::string vsReason;
        if (vs.isAvailable(vsReason)) {
            selectionSummary = "Selected VapourSynth.";
            return std::make_unique<VapourSynthBackend>();
        }
        selectionSummary = "VapourSynth explicitly requested but unavailable: " + vsReason;
        return nullptr;
    }

    if (requested == BackendType::GlslShader) {
        GlslShaderBackend glsl;
        std::string glslReason;
        if (glsl.isAvailable(glslReason)) {
            selectionSummary = "Selected GLSL Shader.";
            return std::make_unique<GlslShaderBackend>();
        }
        selectionSummary = "GLSL Shader explicitly requested but unavailable: " + glslReason;
        return nullptr;
    }

    if (requested == BackendType::Auto) {
        if (migraphxAvailable) {
            return selectMigraphx();
        }
        if (vkAvailable) {
            std::ostringstream os;
            os << "MiGraphX unavailable: " << migraphxReason
               << " Falling back to Vulkan Compute.";
            selectionSummary = os.str();
            return std::make_unique<VulkanComputeBackend>();
        }
        if (ncnnAvailable) {
            std::ostringstream os;
            os << "MiGraphX unavailable: " << migraphxReason
               << " Vulkan Compute unavailable: " << vkReason
               << " Falling back to NCNN Vulkan.";
            selectionSummary = os.str();
            return std::make_unique<NcnnVulkanBackend>();
        }
    }

    std::ostringstream os;
    os << "No supported AMD backend is available. "
       << "MiGraphX: " << migraphxReason << " | "
       << "Vulkan Compute: " << vkReason << " | "
       << "NCNN Vulkan: " << ncnnReason;
    selectionSummary = os.str();
    return nullptr;
}

}  // namespace ave
