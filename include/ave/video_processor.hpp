#pragma once

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
    // Resolve the best on-disk inference path for a stage that has a
    // "model" parameter, then inject "model_path" into a copy of the
    // stage so backends can use it directly.
    EnhancementStage resolveModelPath(const EnhancementStage& stage) const;

    mutable ModelManager  modelManager_;
    PipelinePlanner       planner_;
    BackendManager        backendManager_;
    FfmpegRunner          ffmpeg_;
    SceneDetector         sceneDetector_;
};

}  // namespace ave
