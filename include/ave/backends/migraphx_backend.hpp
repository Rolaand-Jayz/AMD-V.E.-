#pragma once

// ─────────────────────────────────────────────────────────────────
// migraphx_backend.hpp — Gold-standard MiGraphX / ROCm backend
//
// Implements the gold-standard "InferenceRuntime" module contract:
//   parse → compile → cache/load → eval → finish
//
// Key gold-standard compliance points:
//   •  ONNX opset ≤19 gate enforced at parse time.
//   •  Compile options are explicit and minimal.
//   •  Artifact cache key = model identity + ROCm/MiGraphX version +
//      GPU gfx target + compile options + MIGRAPHX_* env vars.
//   •  Manifest sidecar (.mxr.manifest) guards cached artifacts;
//      mismatch → fail-fast recompile (never silently load stale).
//   •  program::get_output_shapes() asserted at load and per frame.
//   •  program::finish() called after every eval.
//   •  Version tuple and MIGRAPHX_* env vars logged at initialize().
//   •  Tensor contracts tracked per model (TensorContract).
//   •  InteropBridge hook points documented for Vulkan↔HIP path.
//   •  Structured error taxonomy (InferenceError) throughout.
// ─────────────────────────────────────────────────────────────────

#include <memory>
#include <string>
#include <vector>

#include "ave/backend.hpp"
#include "ave/error_taxonomy.hpp"
#include "ave/tensor_contract.hpp"

namespace ave {

enum class MiGraphXPrecision {
    Fp32,
    Fp16,
    Int8,
};

// ─────────────────────────────────────────────────────────────────
// CompileOptions — MiGraphX compilation policy
//
// These options are encoded into the artifact cache key.  Changing
// any field automatically invalidates cached .mxr artifacts; the
// backend detects this via manifest sidecar validation and recompiles.
//
// Gold standard ref §"Compiled model caching and invalidation":
//   "Known reliability issues occur if a compiled artifact expects
//    offload-copy semantics but runtime does not enable them."
// ─────────────────────────────────────────────────────────────────
struct CompileOptions {
    // If true, MiGraphX allocates tensor buffers on CPU even for GPU
    // target.  offload_copy=true → CPU pointers; =false → GPU pointers.
    // Must be consistent between compile time and runtime.
    bool offloadCopy    = true;

    // Precision used for compiled MiGraphX artifacts. The current host
    // staging path accepts fp32/fp16 model I/O tensors; int8 compilation is
    // supported for models whose host-facing contracts stay float while the
    // internal graph is quantized.
    MiGraphXPrecision precision = MiGraphXPrecision::Fp16;

    // Returns true if all settings are within known-safe ranges.
    bool validate(std::string& error) const;

    // Human-readable description for logging and manifest.
    std::string format() const;
};

// ─────────────────────────────────────────────────────────────────
// MiGraphXBackend
// ─────────────────────────────────────────────────────────────────
class MiGraphXBackend final : public IAcceleratorBackend {
  public:
    MiGraphXBackend();
    ~MiGraphXBackend() override;

    // ── IAcceleratorBackend ─────────────────────────────────────
    BackendType type()  const override;
    std::string name()  const override;

    // Probes ROCm/MiGraphX availability on the system.
    // Safe to call before initialize().
    bool isAvailable(std::string& reason) const override;
    bool supportsDirectOutputEncode() const override;

    // Initialises the HIP device, logs the version tuple
    // (ROCm/MiGraphX/Vulkan/FFmpeg/GPU/kernel) and all MIGRAPHX_*
    // env vars that affect compilation.
    bool initialize(std::string& error) override;

    // Run a single enhancement stage.  Resolves the model from
    // stage.params["model"] or the catalog default.  Enforces the
    // tensor contract (shape/layout/dtype) at the input gate.
    //   Processed — GPU inference actually enhanced frames.
    //   Deferred  — model loaded but frame loop not yet wired;
    //               FFmpeg filter chain handles this stage.
    //   Error     — fatal failure.
    StageResult runStage(const EnhancementStage& stage, std::string& error) override;

    // Process a video through the loaded MiGraphX program. The shared frame
    // source can now preserve Vulkan hardware frames until they reach the
    // backend boundary, but tensor staging still falls back to host RGB
    // conversion unless the future Vulkan/HIP interop path is enabled.
    StageResult processVideoFile(
        const EnhancementStage& stage,
        const std::string& inputVideo,
        const std::string& outputVideo,
        const FrameProgressCb& progressCb,
        std::string& error,
        const ProcessVideoOptions& opts = {}) override;

    // ── Extended API ────────────────────────────────────────────

    // Set compile options used for artifact compilation and manifest
    // keying.  Call before preloadModel / convertToMiGraphX.
    void setCompileOptions(const CompileOptions& opts);
    CompileOptions compileOptions() const;

    // Pre-load and validate a model by catalog id.
    //   – Checks ONNX opset ≤19 (ModelIncompatible on violation).
    //   – Validates manifest cache key; rejects mismatches (ArtifactInvalid).
    //   – Loads .mxr via migraphx::load.
    //   – Asserts output shapes via program::get_output_shapes().
    //   – Builds TensorContracts for input and output parameters.
    // Subsequent runStage calls with the same id skip the load step.
    bool preloadModel(const std::string& modelId, std::string& error);

    // Release GPU memory for a cached model program.
    void evictModel(const std::string& modelId);

    // Release all loaded model programs.
    void evictAll();

    // Returns the HIP device index in use (0-based).
    int deviceIndex() const;

    // Returns the resolved input tensor contracts for a loaded model.
    // Empty vector if the model has not been loaded yet.
    std::vector<TensorContract> inputContracts (const std::string& modelId) const;

    // Returns the resolved output tensor contracts for a loaded model.
    std::vector<TensorContract> outputContracts(const std::string& modelId) const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ave
