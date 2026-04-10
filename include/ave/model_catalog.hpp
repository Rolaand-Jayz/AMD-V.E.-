#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ave/backend.hpp"
#include "ave/types.hpp"

namespace ave {

// ─────────────────────────────────────────────────────────────────
// Enumerations
// ─────────────────────────────────────────────────────────────────

enum class ModelFormat {
    Onnx,   // ONNX – primary interchange format, directly ingestible by MiGraphX
    NcnnBin, // NCNN .bin/.param pair
    Pytorch  // raw .pt/.pth – compiled directly via torch-MiGraphX (torch.onnx.export + MiGraphX); no separate pre-export step required
};

enum class ModelPrecision {
    Fp32,
    Fp16,
    Int8   // MiGraphX driver-level PTQ: --int8 flag quantizes from fp32 at compile time
};

enum class MiGraphXOnnxTransform {
    None,
    ResizeCubicToLinear,
};

// ─────────────────────────────────────────────────────────────────
// ModelEntry – static metadata for every supported model
// ─────────────────────────────────────────────────────────────────

struct ModelEntry {
    ModelEntry() = default;

    ModelEntry(std::string id_,
               std::string displayName_,
               StageKind stage_,
               ModelFormat sourceFormat_,
               ModelPrecision precision_,
               int scale_,
               double fpsMul_,
               std::string downloadUrl_,
               std::string filename_,
               std::string downloadUrlAux_,
               std::string filenameAux_,
               std::string description_,
               bool isDefault_,
               int minVramMib_,
               std::string archiveSubPath_ = {},
               std::string archiveSubPathAux_ = {},
               int migraphxCompileWidth_ = 0,
               int migraphxCompileHeight_ = 0,
               MiGraphXOnnxTransform migraphxOnnxTransform_ = MiGraphXOnnxTransform::None)
        : id(std::move(id_)),
          displayName(std::move(displayName_)),
          stage(stage_),
          sourceFormat(sourceFormat_),
          precision(precision_),
          scale(scale_),
          fpsMul(fpsMul_),
          downloadUrl(std::move(downloadUrl_)),
          filename(std::move(filename_)),
          downloadUrlAux(std::move(downloadUrlAux_)),
          filenameAux(std::move(filenameAux_)),
          description(std::move(description_)),
          isDefault(isDefault_),
          minVramMib(minVramMib_),
          archiveSubPath(std::move(archiveSubPath_)),
          archiveSubPathAux(std::move(archiveSubPathAux_)),
          migraphxCompileWidth(migraphxCompileWidth_),
          migraphxCompileHeight(migraphxCompileHeight_),
          migraphxOnnxTransform(migraphxOnnxTransform_) {}

    // Unique token used as the "model" parameter in EnhancementStage
    std::string id;

    // Human-readable name shown in the UI dropdown
    std::string displayName;

    // Which enhancement stage this model serves
    StageKind stage;

    // Optional family metadata. When empty, helpers derive a stable
    // single-model family from the model id/display name.
    std::string familyId;
    std::string familyName;

    // Capabilities this model can satisfy in one model invocation. When
    // empty, the model is treated as serving only `stage`.
    std::vector<StageKind> capabilities;

    // True when the model can satisfy multiple capabilities in a single
    // model pass. If false, the pipeline must treat each requested stage
    // independently even if they share a family name.
    bool supportsFusedExecution = false;

    // True when the model can selectively enable only a requested subset
    // of its capabilities. False means a fused run is only valid when the
    // requested set matches the model's full capability set.
    bool supportsSelectiveCapabilities = false;

    // Optional explicit mapping for scalar auxiliary model inputs used by
    // fused/selective custom models. Keys are input tensor names and values
    // are binding expressions such as `denoise.enabled`,
    // `stereo_3d.divergence`, `upscale.width`, or `literal:1`.
    std::unordered_map<std::string, std::string> controlInputBindings;

    // Source file format
    ModelFormat sourceFormat = ModelFormat::Onnx;

    // Typical inference precision
    ModelPrecision precision = ModelPrecision::Fp32;

    // Upscaling factor (meaningful for StageKind::Upscale)
    int scale = 1;

    // Target FPS hint (meaningful for StageKind::Interpolate)
    double fpsMul = 2.0;

    // Primary download URL (ONNX file)
    std::string downloadUrl;

    // Expected on-disk filename after download (inside the downloaded/ subdir)
    std::string filename;

    // Optional second URL used for NCNN param file (NcnnBin models have two files)
    std::string downloadUrlAux;
    std::string filenameAux;

    // Short description shown as tooltip in the UI
    std::string description;

    // Whether this is the recommended default for its stage kind
    bool isDefault = false;

    // Minimum VRAM required in MiB for this model at its native precision
    int minVramMib = 512;

    // If downloadUrl points to a zip archive, these are the paths of the
    // primary and auxiliary files *inside* the archive.  When set, the
    // ModelManager downloads the archive to a temp file, extracts just
    // these two files, and removes the archive afterwards.
    std::string archiveSubPath;    // e.g. "models/realesr-animevideov3-x4.param"
    std::string archiveSubPathAux; // e.g. "models/realesr-animevideov3-x4.bin"

    // Optional fixed compile shape for MiGraphX artifacts. When set, explicit
    // conversion and runtime auto-compile ignore requested dimensions and use
    // this model-native size instead.
    int migraphxCompileWidth = 0;
    int migraphxCompileHeight = 0;

    // Optional ONNX rewrite step required before MiGraphX can parse or compile
    // the downloaded model correctly.
    MiGraphXOnnxTransform migraphxOnnxTransform = MiGraphXOnnxTransform::None;
};

// ─────────────────────────────────────────────────────────────────
// Catalog accessor
// ─────────────────────────────────────────────────────────────────

// Returns the complete built-in model catalog.  Entries are ordered
// by stage kind and then by relevance (default model first).
const std::vector<ModelEntry>& builtinModelCatalog();

// Returns only entries matching a given stage kind.
std::vector<const ModelEntry*> catalogEntriesForStage(StageKind stage);

// Lookup a single entry by id, returns nullptr if not found.
const ModelEntry* catalogEntryById(const std::string& id);

// Best-effort lookup of a catalog id from a downloaded/compiled model path.
// This understands compiled MiGraphX artifact suffixes such as
// `_192x192_b8_fp16.mxr` and `_fp16.mxr`.
std::string inferModelIdFromPath(const std::string& path);

// Family/capability helpers used by the UI and runtime.
std::string modelFamilyId(const ModelEntry& entry);
std::string modelFamilyName(const ModelEntry& entry);
std::vector<StageKind> modelCapabilities(const ModelEntry& entry);
bool modelSupportsCapability(const ModelEntry& entry, StageKind capability);
bool modelCanFuseRequestedCapabilities(const ModelEntry& entry,
                                       const std::vector<StageKind>& requested);
bool modelLooksAnimationFocused(const ModelEntry& entry);
bool modelSupportsBackend(const ModelEntry& entry, BackendType backend);
const ModelEntry* preferredBackendModelForStage(StageKind stage, BackendType backend);

}  // namespace ave
