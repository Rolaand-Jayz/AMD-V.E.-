#pragma once

#include <memory>
#include <string>
#include <vector>

#include "ave/backend.hpp"
#include "ave/runtime_diagnostics.hpp"

namespace ave {

class BackendManager {
  public:
    std::vector<BackendInfo> probeBackends() const;
    RuntimeDiagnosticsReport runtimeDiagnostics() const;
    std::unique_ptr<IAcceleratorBackend> createBackend(BackendType requested, std::string& selectionSummary) const;
};

}  // namespace ave
