#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "ave/model_catalog.hpp"
#include "ave/model_manager.hpp"
#include "ave/observability.hpp"

namespace {

void check(const bool condition, const char* message) {
    if (condition) {
        return;
    }
    std::cerr << "model_manager_profile_tests failed: " << message << '\n';
    std::abort();
}

void writeFile(const std::filesystem::path& path, const std::string& content) {
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << content;
    check(out.good(), "test file write failed");
}

void testStandardProfilesForDynamicUpscaleModel() {
    ave::ModelManager manager;
    const auto profiles = manager.standardCompileProfilesForModel("realesrgan-x4-general");
    check(profiles.size() == 2, "dynamic upscale models should expose 1080p and 4K profiles");
    check(profiles[0].width == 1920 && profiles[0].height == 1080,
          "first standard profile should be 1080p");
    check(profiles[1].width == 3840 && profiles[1].height == 2160,
          "second standard profile should be 4K");
    check(!profiles[0].fixedShape && !profiles[1].fixedShape,
          "dynamic upscale profiles should not be marked fixed");
}

void testStandardProfilesForFixedShapeModel() {
    ave::ModelManager manager;
    const auto profiles = manager.standardCompileProfilesForModel("depth-anything-v2-small-fp16");
    check(profiles.size() == 1, "fixed-shape stereo model should expose a single default profile");
    check(profiles[0].fixedShape, "fixed-shape profile should be marked fixed");
    check(profiles[0].width == profiles[0].height && profiles[0].width > 0,
          "stereo model default compile profile should remain a positive fixed square resolution");
    check(!(profiles[0].width == 1920 && profiles[0].height == 1080),
          "fixed-shape stereo model should not advertise the generic 1080p prewarm profile");
}

void testPrewarmFailsCleanlyWithoutDownloadedModel() {
    ave::ModelManager manager;
    const auto tempDir = std::filesystem::temp_directory_path() / "ave_model_manager_profile_tests";
    std::error_code ec;
    std::filesystem::remove_all(tempDir, ec);
    std::filesystem::create_directories(tempDir, ec);
    manager.setModelsDirectory(tempDir.string());
    manager.refresh();

    std::string error;
    const bool ok = manager.prewarmStandardArtifacts("realesrgan-x4-general",
                                                     ave::ModelProgressCb{},
                                                     ave::ModelStateCb{},
                                                     error);
    check(!ok, "prewarm should fail when the source model is not downloaded");
    check(error.find("Model not yet downloaded") != std::string::npos,
          "prewarm failure should explain missing source model");

    std::filesystem::remove_all(tempDir, ec);
}

void testManifestValidationTracksSourceFingerprint() {
    const auto tempDir =
        std::filesystem::temp_directory_path() / "ave_model_manager_manifest_tests";
    std::error_code ec;
    std::filesystem::remove_all(tempDir, ec);
    std::filesystem::create_directories(tempDir, ec);

    const auto sourcePath = tempDir / "source.onnx";
    writeFile(sourcePath, "abcd");

    ave::obs::ArtifactManifestFields fields;
    fields.manifestSchemaVersion = "2";
    fields.migraphxVersion = "migraphx";
    fields.rocmVersion = "rocm";
    fields.gpuGfxTarget = "gfx";
    fields.onnxFileSizeStr = "4";
    fields.onnxMtimeStr = "1234";
    fields.sourceFingerprint = ave::obs::buildArtifactSourceFingerprint(sourcePath.string());
    fields.offloadCopy = "1";
    fields.precision = "fp16";
    fields.compileProfile = "balanced";
    fields.disableMlir = "0";
    fields.enableNhwc = "1";
    fields.enableCk = "1";
    fields.problemCachePath = "/tmp/problem_cache.json";
    fields.miopenUserDbPath = "/tmp/miopen_user_db";
    fields.miopenCustomCacheDir = "/tmp/miopen_cache";
    fields.miopenFindMode = "DYNAMIC_HYBRID";
    fields.miopenCompileParallelLevel = "4";
    fields.visibleDevices = "0";
    fields.runtimeFingerprint = "runtime";

    const auto manifestPath = tempDir / "artifact.mxr.manifest";
    std::string error;
    check(ave::obs::writeArtifactManifest(manifestPath.string(), fields, error),
          "manifest write should succeed");

    std::string mismatchReason;
    check(ave::obs::validateArtifactManifest(manifestPath.string(), fields, mismatchReason),
          "matching manifest should validate");

        auto overwritten = fields;
        overwritten.precision = "fp32";
        overwritten.runtimeFingerprint = "runtime-overwrite";
        check(ave::obs::writeArtifactManifest(manifestPath.string(), overwritten, error),
            "manifest overwrite should succeed");
        check(ave::obs::validateArtifactManifest(manifestPath.string(), overwritten, mismatchReason),
            "overwritten manifest should validate against the latest contents");

    auto mismatched = fields;
    mismatched.sourceFingerprint = "changed";
    check(!ave::obs::validateArtifactManifest(manifestPath.string(), mismatched, mismatchReason),
          "source fingerprint mismatch should invalidate manifest");
    check(mismatchReason.find("source_fingerprint") != std::string::npos,
          "mismatch reason should mention source fingerprint");

    std::filesystem::remove_all(tempDir, ec);
}

void testBestPathFallsBackWhenCompiledArtifactManifestIsMissing() {
    ave::ModelManager manager;
    const auto tempDir =
        std::filesystem::temp_directory_path() / "ave_model_manager_best_path_tests";
    std::error_code ec;
    std::filesystem::remove_all(tempDir, ec);
    std::filesystem::create_directories(tempDir, ec);

    manager.setModelsDirectory(tempDir.string());
    manager.refresh();

    const auto model = manager.findModel("realesrgan-x4-general");
    check(model.has_value(), "expected built-in model catalog entry");

    const auto downloadedPath =
        tempDir / "downloaded" / model->entry.filename;
    const auto compiledPath =
        tempDir / "migraphx" / "realesrgan-x4-general_fp16.mxr";

    writeFile(downloadedPath, "fake-onnx");
    writeFile(compiledPath, "fake-mxr");

    manager.refresh();
    const auto refreshedModel = manager.findModel("realesrgan-x4-general");
    check(refreshedModel.has_value(), "refreshed model lookup should succeed");
    check(refreshedModel->state == ave::ModelState::Downloaded,
          "stale compiled artifacts should not mark the model as converted");
    check(refreshedModel->convertedPath.empty(),
          "stale compiled artifacts should not populate convertedPath");
    const auto bestPath = manager.bestPathForModel("realesrgan-x4-general");
    check(bestPath.has_value(), "bestPath should fall back to the downloaded source");
    check(*bestPath == downloadedPath.string(),
          "bestPath should ignore compiled artifacts without a valid manifest");

    std::filesystem::remove_all(tempDir, ec);
}

void testAllModelsPreserveCatalogOrder() {
    ave::ModelManager manager;
    const auto allModels = manager.allModels();
    const auto& catalog = ave::builtinModelCatalog();

    check(allModels.size() == catalog.size(),
          "allModels should return every built-in catalog entry");
    for (std::size_t i = 0; i < catalog.size(); ++i) {
        check(allModels[i].entry.id == catalog[i].id,
              "allModels should preserve built-in catalog order");
    }
}

}  // namespace

int main() {
    testStandardProfilesForDynamicUpscaleModel();
    testStandardProfilesForFixedShapeModel();
    testPrewarmFailsCleanlyWithoutDownloadedModel();
    testManifestValidationTracksSourceFingerprint();
    testBestPathFallsBackWhenCompiledArtifactManifestIsMissing();
    testAllModelsPreserveCatalogOrder();
    return 0;
}
