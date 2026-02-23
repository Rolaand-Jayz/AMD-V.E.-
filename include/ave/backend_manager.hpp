#pragma once

#include <memory>
#include <string>
#include <vector>

#include "ave/backend.hpp"

namespace ave {

class BackendManager {
  public:
    std::vector<BackendInfo> probeBackends() const;
    std::unique_ptr<IAcceleratorBackend> createBackend(BackendType requested, std::string& selectionSummary) const;
};

}  // namespace ave
