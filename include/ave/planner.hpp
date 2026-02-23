#pragma once

#include <vector>

#include "ave/stage.hpp"

namespace ave {

class PipelinePlanner {
  public:
    std::vector<EnhancementStage> plan(const std::vector<EnhancementStage>& requested) const;
};

}  // namespace ave
