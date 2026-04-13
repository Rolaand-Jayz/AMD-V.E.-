#include "ave/backend_manager.hpp"

#include <iostream>
#include <memory>
#include <sstream>

#include "ave/backends/glsl_shader_backend.hpp"
#include "ave/backends/migraphx_backend.hpp"
#include "ave/backends/ncnn_vulkan_backend.hpp"
#include "ave/backends/rocm_hip_backend.hpp"
#include "ave/backends/vapoursynth_backend.hpp"
#include "ave/backends/vulkan_compute_backend.hpp"
#include "ave/interop_bridge.hpp"
#include "ave/runtime_diagnostics.hpp"

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
        RocmHipBackend backend;
        std::string detail;
        const bool ok = backend.isAvailable(detail);
        infos.push_back(BackendInfo{BackendType::RocmHip, backend.name(), ok, detail});
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

RuntimeDiagnosticsReport BackendManager::runtimeDiagnostics() const {
    return collectRuntimeDiagnostics();
}

std::unique_ptr<IAcceleratorBackend> BackendManager::createBackend(BackendType requested,
                                                                    std::string& selectionSummary) const {
    const RuntimeDiagnosticsReport diagnostics = runtimeDiagnostics();
    auto selectMigraphx = [&selectionSummary]() -> std::unique_ptr<IAcceleratorBackend> {
        selectionSummary = "Selected MiGraphX (ROCm).";
        InteropBridge bridge;
        std::string bridgeReason;
        const bool bridgeUp = bridge.isAvailable(bridgeReason);
        std::cout << "[backend-manager] interop-bridge: "
                  << (bridgeUp ? "GPU interop available" : bridgeReason)
                  << std::endl;
        return std::make_unique<MiGraphXBackend>();
    };

    auto selectRocmHip = [&selectionSummary]() -> std::unique_ptr<IAcceleratorBackend> {
        selectionSummary = "Selected ROCm/HIP (ONNX Runtime).";
        return std::make_unique<RocmHipBackend>();
    };

    if (requested == BackendType::MiGraphX) {
        MiGraphXBackend migraphx;
        std::string reason;
        if (migraphx.isAvailable(reason)) {
            return selectMigraphx();
        }
        selectionSummary = "MiGraphX explicitly requested but unavailable: " + reason;
        return nullptr;
    }

    if (requested == BackendType::VulkanCompute) {
        VulkanComputeBackend vkCompute;
        std::string reason;
        if (vkCompute.isAvailable(reason)) {
            selectionSummary = "Selected Vulkan Compute.";
            return std::make_unique<VulkanComputeBackend>();
        }
        selectionSummary = "Vulkan Compute explicitly requested but unavailable: " + reason;
        return nullptr;
    }

    if (requested == BackendType::RocmHip) {
        RocmHipBackend rocmHip;
        std::string reason;
        if (rocmHip.isAvailable(reason)) {
            return selectRocmHip();
        }
        selectionSummary = "ROCm/HIP explicitly requested but unavailable: " + reason;
        return nullptr;
    }

    if (requested == BackendType::NcnnVulkan) {
        NcnnVulkanBackend ncnn;
        std::string reason;
        if (ncnn.isAvailable(reason)) {
            selectionSummary = "Selected NCNN Vulkan.";
            return std::make_unique<NcnnVulkanBackend>();
        }
        selectionSummary = "NCNN Vulkan explicitly requested but unavailable: " + reason;
        return nullptr;
    }

    if (requested == BackendType::VapourSynth) {
        VapourSynthBackend vs;
        std::string reason;
        if (vs.isAvailable(reason)) {
            selectionSummary = "Selected VapourSynth.";
            return std::make_unique<VapourSynthBackend>();
        }
        selectionSummary = "VapourSynth explicitly requested but unavailable: " + reason;
        return nullptr;
    }

    if (requested == BackendType::GlslShader) {
        GlslShaderBackend glsl;
        std::string reason;
        if (glsl.isAvailable(reason)) {
            selectionSummary = "Selected GLSL Shader.";
            return std::make_unique<GlslShaderBackend>();
        }
        selectionSummary = "GLSL Shader explicitly requested but unavailable: " + reason;
        return nullptr;
    }

    if (requested == BackendType::Auto) {
        MiGraphXBackend migraphx;
        std::string migraphxReason;
        if (migraphx.isAvailable(migraphxReason)) {
            return selectMigraphx();
        }

        RocmHipBackend rocmHip;
        std::string rocmHipReason;
        if (rocmHip.isAvailable(rocmHipReason)) {
            std::ostringstream os;
            os << "MiGraphX unavailable: " << migraphxReason
               << " Falling back to ROCm/HIP (ONNX Runtime).";
            if (!diagnostics.migraphxReady) {
                os << " Runtime diagnostics: " << diagnostics.summary();
            }
            selectionSummary = os.str();
            return std::make_unique<RocmHipBackend>();
        }

        VulkanComputeBackend vkCompute;
        std::string vkReason;
        if (vkCompute.isAvailable(vkReason)) {
            std::ostringstream os;
            os << "MiGraphX unavailable: " << migraphxReason
               << " ROCm/HIP unavailable: " << rocmHipReason
               << " Falling back to Vulkan Compute.";
            if (!diagnostics.migraphxReady) {
                os << " Runtime diagnostics: " << diagnostics.summary();
            }
            selectionSummary = os.str();
            return std::make_unique<VulkanComputeBackend>();
        }

        NcnnVulkanBackend ncnn;
        std::string ncnnReason;
        if (ncnn.isAvailable(ncnnReason)) {
            std::ostringstream os;
            os << "MiGraphX unavailable: " << migraphxReason
               << " ROCm/HIP unavailable: " << rocmHipReason
               << " Vulkan Compute unavailable: " << vkReason
               << " Falling back to NCNN Vulkan.";
            if (!diagnostics.migraphxReady) {
                os << " Runtime diagnostics: " << diagnostics.summary();
            }
            selectionSummary = os.str();
            return std::make_unique<NcnnVulkanBackend>();
        }

        std::ostringstream os;
        os << "No supported AMD backend is available. "
           << "MiGraphX: " << migraphxReason << " | "
              << "ROCm/HIP: " << rocmHipReason << " | "
           << "Vulkan Compute: " << vkReason << " | "
           << "NCNN Vulkan: " << ncnnReason;
        if (!diagnostics.checks.empty()) {
            os << " | Runtime diagnostics: " << diagnostics.summary();
        }
        selectionSummary = os.str();
        return nullptr;
    }

    selectionSummary = "Unsupported backend request.";
    return nullptr;
}

}  // namespace ave
