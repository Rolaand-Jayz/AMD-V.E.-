#pragma once

#include <string>
#include <vector>

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
               std::string archiveSubPathAux_ = {})
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
          archiveSubPathAux(std::move(archiveSubPathAux_)) {}

    // Unique token used as the "model" parameter in EnhancementStage
    std::string id;

    // Human-readable name shown in the UI dropdown
    std::string displayName;

    // Which enhancement stage this model serves
    StageKind stage;

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

}  // namespace ave
