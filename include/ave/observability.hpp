#pragma once

// ─────────────────────────────────────────────────────────────────
// observability.hpp — Diagnostics, tracing, and version logging
//
// Gold-standard-compliant observability primitives:
//
// 1. ROCTx integration (zero-overhead when disabled)
//    Provides stage-boundary markers for rocprof --roctx-trace.
//    Enable with -DAVE_HAVE_ROCTX=ON and linking roctracer.
//    rocprof command: rocprof --roctx-trace --hip-trace ./ave ...
//
// 2. Version tuple logging (call once at backend initialisation)
//    Logs: ROCm version, MiGraphX version, Vulkan driver/API version,
//    FFmpeg version, GPU gfx target, kernel version.
//    Together these form the reproducibility anchor from the gold std.
//
// 3. MiGraphX environment variable audit
//    Logs all MIGRAPHX_* env vars that affect compilation so CI logs
//    capture the full compile context, not just source code.
//
// 4. Tensor contract violation reporter
//    Structured fail-fast log for boundary assertion failures.
// ─────────────────────────────────────────────────────────────────

#include <string>

namespace ave {
namespace obs {

// ─────────────────────────────────────────────────────────────────
// ROCTx integration
// ─────────────────────────────────────────────────────────────────
// These macros wrap roctxRangePush/roctxRangePop for rocprof integration.
// When AVE_HAVE_ROCTX is not defined (default) they expand to nothing.
//
// Usage:
//   AVE_ROCTX_RANGE("migraphx:eval");
//   // ... work ...
//   AVE_ROCTX_RANGE_END();
//
// MiGraphX driver has a matching roctx integration path for per-operator
// timing; the two combine to give stage→operator attribution in rocprof.

#ifdef AVE_HAVE_ROCTX
#  include <roctracer/roctx.h>
#  define AVE_ROCTX_RANGE(label)   roctxRangePush(label)
#  define AVE_ROCTX_RANGE_END()    roctxRangePop()
#  define AVE_ROCTX_MARK(label)    roctxMark(label)
#else
#  define AVE_ROCTX_RANGE(label)   ((void)0)
#  define AVE_ROCTX_RANGE_END()    ((void)0)
#  define AVE_ROCTX_MARK(label)    ((void)0)
#endif

// ─────────────────────────────────────────────────────────────────
// Version tuple logging
// ─────────────────────────────────────────────────────────────────
// Logs the reproducibility anchor at backend initialisation.
// Output format (each on its own line, grep-friendly):
//   [version] rocm=<ver>
//   [version] migraphx=<ver>
//   [version] vulkan_api=<1.x.y>  vulkan_driver=<string>
//   [version] ffmpeg=<ver>
//   [version] gpu_gfx_target=<gfxNNNN>
//   [version] kernel=<uname -r>
//
// Call this ONCE in MiGraphXBackend::initialize() before compilation.
void logVersionTuple();

// ─────────────────────────────────────────────────────────────────
// MiGraphX environment variable audit
// ─────────────────────────────────────────────────────────────────
// Logs all MIGRAPHX_* environment variables that affect compilation
// or execution.  These must appear in CI logs next to compile artifacts
// so that "works on machine A / fails on machine B" is diagnosable.
//
// Variables covered (gold standard §"Diagnostics and instrumentation"):
//   MIGRAPHX_TRACE_COMPILE    – log pass-by-pass compile diffs
//   MIGRAPHX_TRACE_PASSES     – detailed pass list
//   MIGRAPHX_TIME_PASSES      – pass timing
//   MIGRAPHX_TRACE_HIPRTC     – GPU JIT source
//   MIGRAPHX_GPU_DUMP_SRC     – dump generated HIPRTC source
//   MIGRAPHX_GPU_DUMP_ASM     – dump compiled assembly
//   MIGRAPHX_DEBUG_SAVE_TEMP_DIR – save temp compilation artefacts
//   MIGRAPHX_GPU_DEBUG        – GPU debug mode
//   MIGRAPHX_DISABLE_MLIR     – disable rocMLIR path (isolation)
//   MIGRAPHX_TRACE_MLIR       – rocMLIR pass tracing
//   MIGRAPHX_ENABLE_NHWC      – force NHWC layout pass (changes tensor interpretation!)
//   MIGRAPHX_ENABLE_CK        – enable composable-kernel path
void logMiGraphXEnvironment();

// Logs an explicit effective MiGraphX/MiOpen environment snapshot for the
// supplied phase (for example compile-time, runtime-load, runtime-warmup).
void logMiGraphXEnvironment(const struct ArtifactManifestFields& effective,
                            const std::string& phase,
                            const std::string& artifactPath = {},
                            const std::string& warmupStatus = {});

// ─────────────────────────────────────────────────────────────────
// Tensor contract violation logging
// ─────────────────────────────────────────────────────────────────
// Call when a boundary assertion fails.  Logs a structured, searchable
// message to stderr before the caller returns the error.
//   location – e.g. "MiGraphXBackend::runInference (input gate)"
//   expected – human-readable expected value, e.g. "1×3×720×1280 fp32"
//   actual   – human-readable actual value encountered
void logTensorContractViolation(const std::string& location,
                                const std::string& expected,
                                const std::string& actual);

// ─────────────────────────────────────────────────────────────────
// Artifact manifest helpers (compile-option caching)
// ─────────────────────────────────────────────────────────────────
// Read/write the plain-text manifest sidecar that guards cached .mxr
// artifacts.  The manifest stores the full cache key (see gold standard
// §"Compiled model caching and invalidation") alongside the artifact.

// Returns true if the manifest file at manifestPath matches all
// key fields derived from the supplied options.  Used before loading
// a cached artifact.  Returns false (and sets reason) on any mismatch
// so the caller recompiles and writes a fresh manifest.
struct ArtifactManifestFields {
    std::string migraphxVersion;  // from migraphx::version() or rocm version_info
    std::string rocmVersion;      // content of /opt/rocm/.info/version
    std::string gpuGfxTarget;     // gfxNNNN from rocminfo or HIP device props
    std::string onnxFileSizeStr;  // std::to_string(file size in bytes)
    std::string onnxMtimeStr;     // std::to_string(mtime as epoch seconds)
    std::string offloadCopy;      // "0" or "1"
    std::string precision;        // "fp32" or "fp16"
    std::string compileProfile;   // compile/runtime tuning profile label
    std::string disableMlir;      // MIGRAPHX_DISABLE_MLIR env var value or "0"
    std::string enableNhwc;       // MIGRAPHX_ENABLE_NHWC env var value or "0"
    std::string enableCk;         // MIGRAPHX_ENABLE_CK env var value or "0"
    std::string problemCachePath;         // MIGRAPHX_PROBLEM_CACHE
    std::string miopenUserDbPath;         // MIOPEN_USER_DB_PATH
    std::string miopenCustomCacheDir;     // MIOPEN_CUSTOM_CACHE_DIR
    std::string miopenFindMode;           // MIOPEN_FIND_MODE
    std::string miopenCompileParallelLevel; // MIOPEN_COMPILE_PARALLEL_LEVEL
    std::string visibleDevices;           // effective HIP/ROCR visible device binding
    std::string runtimeFingerprint;       // deterministic tuning-context fingerprint
};

// Write a manifest file.  Overwrites any existing manifest.
bool writeArtifactManifest(const std::string&            manifestPath,
                           const ArtifactManifestFields& fields,
                           std::string&                  error);

// Read and validate a manifest file against expected fields.
// Returns true only when every field matches exactly.
bool validateArtifactManifest(const std::string&            manifestPath,
                              const ArtifactManifestFields& expected,
                              std::string&                  mismatchReason);

}  // namespace obs
}  // namespace ave
