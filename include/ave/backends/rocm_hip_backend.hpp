#pragma once

#include <memory>
#include <string>

#include "ave/backend.hpp"

namespace ave {

class RocmHipBackend final : public IAcceleratorBackend {
  public:
    RocmHipBackend();
    ~RocmHipBackend() override;

    BackendType type() const override;
    std::string name() const override;
    bool isAvailable(std::string& reason) const override;
    bool initialize(std::string& error) override;
    StageResult runStage(const EnhancementStage& stage, std::string& error) override;
    StageResult processVideoFile(
        const EnhancementStage& stage,
        const std::string& inputVideo,
        const std::string& outputVideo,
        const FrameProgressCb& progressCb,
        std::string& error,
        const ProcessVideoOptions& opts = {}) override;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ave