#pragma once

#include <optional>

#include "ave/backend_manager.hpp"
#include "ave/ffmpeg_runner.hpp"
#include "ave/job.hpp"
#include "ave/model_manager.hpp"
#include "ave/planner.hpp"
#include "ave/scene_detector.hpp"

namespace ave {

class VideoProcessor {
  public:
    bool process(const VideoJob& job, std::string& error) const;

  private:
    // Resolve the best backend-appropriate on-disk inference path for a
    // stage that has a "model" parameter, then inject "model_path" into
    // a copy of the stage so backends can use it directly.
    //
    // MiGraphX consumes compiled .mxr artifacts, while Vulkan/NCNN need
    // the original downloaded model files. Keeping that split explicit
    // prevents a generic "best path" lookup from handing a MiGraphX
    // artifact to a backend that cannot load it.
    EnhancementStage resolveModelPath(const EnhancementStage& stage,
                                      std::optional<BackendType> activeBackend) const;

    mutable ModelManager  modelManager_;
    PipelinePlanner       planner_;
    BackendManager        backendManager_;
    FfmpegRunner          ffmpeg_;
    SceneDetector         sceneDetector_;
};

}  // namespace ave
