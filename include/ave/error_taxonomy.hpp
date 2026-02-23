#pragma once

// ─────────────────────────────────────────────────────────────────
// error_taxonomy.hpp — Structured inference error classification
//
// Gold-standard-compliant error taxonomy.  Every failure that crosses
// a subsystem boundary (model, interop, Vulkan, FFmpeg) is classified
// into one of the kinds below so root-cause is unambiguous in logs
// and so recovery policy (fail-fast vs soft-fallback) can be applied
// per-kind rather than per-exception-message.
//
// Ref: Gold Standard §"Error Taxonomy and Recovery Policy":
//   ModelIncompatible  – fail-fast (contract violation)
//   SyncHazard         – fail-fast (data hazard detected)
//   CompileFailure     – fail-fast (compiler contract violation)
//   ArtifactInvalid    – fail-fast (manifest mismatch)
//   InteropFailure     – fail-fast (export/import failure)
//   VulkanDeviceFailure– fail-fast (missing extensions)
//   FFmpegInteropFailure – log + fallback
//   RuntimeFailure     – log + optional retry gate
// ─────────────────────────────────────────────────────────────────

#include <string>

namespace ave {

// ─────────────────────────────────────────────────────────────────
// InferenceErrorKind
// ─────────────────────────────────────────────────────────────────
enum class InferenceErrorKind {
    // Model contains unsupported ONNX operators, opsets, or dtypes.
    // Recovery: reject model; re-export at opset ≤19; replace ops.
    ModelIncompatible,

    // MiGraphX failed to compile the model (pass failure, JIT error, etc.).
    // Recovery: enable MIGRAPHX_TRACE_COMPILE to locate failing pass.
    CompileFailure,

    // Cached .mxr artifact is invalid: cache-key mismatch, load error,
    // msgpack/ABI conflict, or GPU/version incompatibility.
    // Recovery: delete artifact; recompile with correct key.
    ArtifactInvalid,

    // Vulkan device does not support required extensions
    // (VK_KHR_external_memory_fd, VK_KHR_external_semaphore_fd,
    // VK_KHR_synchronization2) or validation layer reported an error.
    // Recovery: check driver/Mesa version; query extension support.
    VulkanDeviceFailure,

    // Vulkan↔HIP external memory or semaphore export/import failed.
    // Recovery: verify handle types (OPAQUE_FD vs dma_buf); run interop
    // smoke test; check extension availability at runtime.
    InteropFailure,

    // Detected a cross-API data hazard (Vulkan↔HIP buffer accessed
    // concurrently without semaphore synchronisation).
    // Recovery: enforce semaphore handshake at every buffer handoff.
    // This is a fail-fast error – never silently continue.
    SyncHazard,

    // FFmpeg hwframe mapping or hwcontext operation failed.
    // Recovery: check av_vkfmt_from_pixfmt result; insert explicit
    // convert stage or fallback to CPU transfer path (log degraded mode).
    FFmpegInteropFailure,

    // MiGraphX program::eval failed; output shape mismatch;
    // general GPU runtime error.
    // Recovery: validate tensor contract at init; check offload_copy flag.
    RuntimeFailure,
};

std::string toString(InferenceErrorKind kind);

// ─────────────────────────────────────────────────────────────────
// InferenceError — structured error value with kind + messages
// ─────────────────────────────────────────────────────────────────
struct InferenceError {
    InferenceErrorKind kind    = InferenceErrorKind::RuntimeFailure;
    std::string        message;   // Primary human-readable description
    std::string        context;   // Extra diagnostic detail (file, version, etc.)

    // Format for log output: "[ArtifactInvalid] <message>\n  context: <context>"
    std::string format() const;

    // Factory helpers for common failure sites
    static InferenceError modelIncompatible(const std::string& msg,
                                            const std::string& ctx = {});
    static InferenceError compileFailure   (const std::string& msg,
                                            const std::string& ctx = {});
    static InferenceError artifactInvalid  (const std::string& msg,
                                            const std::string& ctx = {});
    static InferenceError vulkanDevice     (const std::string& msg,
                                            const std::string& ctx = {});
    static InferenceError interopFailure   (const std::string& msg,
                                            const std::string& ctx = {});
    static InferenceError syncHazard       (const std::string& msg,
                                            const std::string& ctx = {});
    static InferenceError ffmpegInterop    (const std::string& msg,
                                            const std::string& ctx = {});
    static InferenceError runtimeFailure   (const std::string& msg,
                                            const std::string& ctx = {});
};

}  // namespace ave
