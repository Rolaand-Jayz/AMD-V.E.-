#include <cassert>
#include <filesystem>
#include <string>

#include "ave/job_recovery.hpp"
#include "ave/stage.hpp"

namespace {

void testParseStageSpecParsesTypedValues() {
    std::string error;
    const auto stage = ave::parseStageSpec(
        "upscale:width=3840,height=2160,enabled=true,strength=0.75,model=compact",
        error);
    assert(stage.has_value());
    assert(error.empty());
    assert(std::holds_alternative<std::int64_t>(stage->params.at("width")));
    assert(std::get<std::int64_t>(stage->params.at("width")) == 3840);
    assert(std::holds_alternative<bool>(stage->params.at("enabled")));
    assert(std::get<bool>(stage->params.at("enabled")));
    assert(std::holds_alternative<double>(stage->params.at("strength")));
    assert(std::get<double>(stage->params.at("strength")) == 0.75);
    assert(std::holds_alternative<std::string>(stage->params.at("model")));
    assert(std::get<std::string>(stage->params.at("model")) == "compact");
}

void testJobRecoveryRoundTrip() {
    const std::filesystem::path tempPath =
        std::filesystem::temp_directory_path() / "ave_job_recovery_tests.state";
    std::error_code cleanupError;
    std::filesystem::remove(tempPath, cleanupError);

    ave::JobRecoveryStore store(tempPath.string());
    ave::RecoveredJobState state;
    state.startedAtUtc = "2026-04-01T12:00:00Z";
    state.job.inputPath = "/tmp/input clip.mp4";
    state.job.outputPath = "/tmp/output clip.mp4";
    state.job.requestedBackend = ave::BackendType::RocmHip;
    state.job.encode.codec = "libx265";
    state.job.encode.profile = "main10";
    state.job.encode.crf = 14;
    state.job.encode.preset = "slow";
    state.job.encode.threads = 6;
    state.job.previewMode = false;
    state.job.previewDurationSec = 10.0;
    state.job.previewFrameInterval = 12;

    ave::EnhancementStage denoise;
    denoise.kind = ave::StageKind::Denoise;
    denoise.params["strength"] = 0.7;
    denoise.params["model"] = std::string("clearreality-denoise");
    state.job.requestedStages.push_back(denoise);

    ave::EnhancementStage upscale;
    upscale.kind = ave::StageKind::Upscale;
    upscale.params["width"] = std::int64_t{3840};
    upscale.params["height"] = std::int64_t{2160};
    upscale.params["model"] = std::string("openproteus-x2");
    state.job.requestedStages.push_back(upscale);

    ave::ActiveFilter cas;
    cas.id = "glsl.cas";
    cas.paramValues["intensity"] = 0.2;
    state.job.catalogFilters.push_back(cas);

    std::string error;
    assert(store.save(state, error));
    assert(error.empty());

    const auto loaded = store.load(error);
    assert(loaded.has_value());
    assert(error.empty());
    assert(loaded->startedAtUtc == state.startedAtUtc);
    assert(loaded->job.inputPath == state.job.inputPath);
    assert(loaded->job.outputPath == state.job.outputPath);
    assert(loaded->job.requestedBackend == ave::BackendType::RocmHip);
    assert(loaded->job.encode.codec == "libx265");
    assert(loaded->job.encode.profile == "main10");
    assert(loaded->job.encode.crf == 14);
    assert(loaded->job.encode.preset == "slow");
    assert(loaded->job.encode.threads == 6);
    assert(loaded->job.requestedStages.size() == 2);
    assert(loaded->job.catalogFilters.size() == 1);
    assert(loaded->job.catalogFilters.front().id == "glsl.cas");
    assert(loaded->job.catalogFilters.front().paramValues.at("intensity") == 0.2);

    assert(store.clear(error));
    assert(error.empty());
    const auto afterClear = store.load(error);
    assert(!afterClear.has_value());
    assert(error.empty());
}

}  // namespace

int main() {
    testParseStageSpecParsesTypedValues();
    testJobRecoveryRoundTrip();
    return 0;
}
