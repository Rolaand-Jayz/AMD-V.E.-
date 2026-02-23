#include "ave/backend_manager.hpp"

#include <iostream>
#include <memory>
#include <sstream>

#include "ave/backends/migraphx_backend.hpp"
#include "ave/backends/ncnn_vulkan_backend.hpp"
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
        NcnnVulkanBackend backend;
        std::string detail;
        const bool ok = backend.isAvailable(detail);
        infos.push_back(BackendInfo{BackendType::NcnnVulkan, backend.name(), ok, detail});
    }

    return infos;
}

std::unique_ptr<IAcceleratorBackend> BackendManager::createBackend(BackendType requested,
                                                                    std::string& selectionSummary) const {
    MiGraphXBackend migraphx;
    NcnnVulkanBackend ncnn;

    std::string migraphxReason;
    const bool migraphxAvailable = migraphx.isAvailable(migraphxReason);

    std::string ncnnReason;
    const bool ncnnAvailable = ncnn.isAvailable(ncnnReason);

    if (requested == BackendType::Auto || requested == BackendType::MiGraphX) {
        if (migraphxAvailable) {
            selectionSummary = "Selected MiGraphX (ROCm).";
            // Log InteropBridge status alongside backend selection so the
            // pipeline mode (GPU interop vs CPU staging) appears in startup logs.
            {
                InteropBridge bridge;
                std::string bridgeReason;
                const bool bridgeUp = bridge.isAvailable(bridgeReason);
                std::cout << "[backend-manager] interop-bridge: "
                          << (bridgeUp ? "GPU interop available" : bridgeReason)
                          << std::endl;
            }
            return std::make_unique<MiGraphXBackend>();
        }
        if (ncnnAvailable) {
            std::ostringstream os;
            os << "MiGraphX unavailable: " << migraphxReason << " Falling back to NCNN Vulkan.";
            selectionSummary = os.str();
            return std::make_unique<NcnnVulkanBackend>();
        }
    }

    if (requested == BackendType::NcnnVulkan) {
        if (ncnnAvailable) {
            selectionSummary = "Selected NCNN Vulkan.";
            return std::make_unique<NcnnVulkanBackend>();
        }
        selectionSummary = "NCNN Vulkan explicitly requested but unavailable: " + ncnnReason;
        return nullptr;
    }

    std::ostringstream os;
    os << "No supported AMD backend is available. "
       << "MiGraphX: " << migraphxReason << " | "
       << "NCNN Vulkan: " << ncnnReason;
    selectionSummary = os.str();
    return nullptr;
}

}  // namespace ave
