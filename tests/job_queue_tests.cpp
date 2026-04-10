#include <cassert>
#include <cstdint>
#include <filesystem>
#include <string>

#include "ave/job_queue.hpp"

namespace {

void testQueueStoreRoundTrip() {
    const std::filesystem::path tempPath =
        std::filesystem::temp_directory_path() / "ave_job_queue_tests.state";
    std::error_code cleanupError;
    std::filesystem::remove(tempPath, cleanupError);

    ave::QueuedJobRecord pending;
    pending.id = "job-1";
    pending.status = ave::QueuedJobStatus::Pending;
    pending.retryable = false;
    pending.attemptCount = 0;
    pending.createdAtUtc = "2026-04-01T12:00:00Z";
    pending.job.inputPath = "/tmp/a.mp4";
    pending.job.outputPath = "/tmp/a-out.mp4";
    pending.job.requestedBackend = ave::BackendType::MiGraphX;
    pending.job.encode.codec = "libx265";
    pending.job.encode.preset = "slow";
    pending.job.encode.crf = 16;

    ave::EnhancementStage upscale;
    upscale.kind = ave::StageKind::Upscale;
    upscale.params["width"] = std::int64_t{3840};
    upscale.params["height"] = std::int64_t{2160};
    pending.job.requestedStages.push_back(upscale);

    ave::ActiveFilter cas;
    cas.id = "glsl.cas";
    cas.paramValues["intensity"] = 0.2;
    pending.job.catalogFilters.push_back(cas);

    ave::QueuedJobRecord staleRunning;
    staleRunning.id = "job-2";
    staleRunning.status = ave::QueuedJobStatus::Running;
    staleRunning.retryable = false;
    staleRunning.attemptCount = 1;
    staleRunning.createdAtUtc = "2026-04-01T12:05:00Z";
    staleRunning.startedAtUtc = "2026-04-01T12:06:00Z";
    staleRunning.job.inputPath = "/tmp/b.mp4";
    staleRunning.job.outputPath = "/tmp/b-out.mp4";
    staleRunning.job.encode.codec = "libx264";
    staleRunning.job.encode.preset = "medium";
    staleRunning.job.encode.crf = 18;

    ave::JobQueueStore store(tempPath.string());
    std::string error;
    assert(store.save({pending, staleRunning}, error));
    assert(error.empty());

    const auto loaded = store.load(error);
    assert(error.empty());
    assert(loaded.size() == 2);
    assert(loaded[0].id == "job-1");
    assert(loaded[0].status == ave::QueuedJobStatus::Pending);
    assert(!loaded[0].retryable);
    assert(loaded[0].job.catalogFilters.size() == 1);
    assert(loaded[0].job.catalogFilters.front().id == "glsl.cas");
    assert(loaded[0].job.catalogFilters.front().paramValues.at("intensity") == 0.2);
    assert(loaded[1].id == "job-2");
    assert(loaded[1].status == ave::QueuedJobStatus::RetryableFailure);
    assert(loaded[1].retryable);
    assert(!loaded[1].lastError.empty());

    assert(store.clear(error));
    assert(error.empty());
    const auto afterClear = store.load(error);
    assert(afterClear.empty());
    assert(error.empty());
}

void testRetryableFailureClassifier() {
    assert(ave::isRetryableQueueFailure("[RuntimeFailure] ROCm device lost during processing"));
    assert(ave::isRetryableQueueFailure("[FFmpegInteropFailure] hwframe export failed"));
    assert(ave::isRetryableQueueFailure("ffmpeg encode pipe exited after runtime interruption"));
    assert(!ave::isRetryableQueueFailure("[CompileFailure] model compilation failed"));
    assert(!ave::isRetryableQueueFailure("[ArtifactInvalid] manifest mismatch"));
    assert(!ave::isRetryableQueueFailure("[InteropFailure] external semaphore import failed"));
    assert(!ave::isRetryableQueueFailure("[SyncHazard] missing semaphore handshake"));
    assert(!ave::isRetryableQueueFailure("Model not yet downloaded – cannot compile."));
}

}  // namespace

int main() {
    testQueueStoreRoundTrip();
    testRetryableFailureClassifier();
    return 0;
}
