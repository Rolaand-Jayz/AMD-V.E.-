#pragma once

// ─────────────────────────────────────────────────────────────────
// tensor_contract.hpp — Explicit tensor layout / dtype / shape types
//
// The gold standard mandates a single, explicit Tensor Contract at
// every subsystem boundary (decode→preprocess, preprocess→inference,
// inference→postprocess, postprocess→encode) with assertions at each
// boundary to fail-fast on contract violations rather than producing
// silently wrong output.
//
// Key requirement from gold standard:
//   "Enforce a single, explicit Tensor Contract (layout, dtype, shape,
//    strides, color pipeline) and assert it at every boundary."
//
// NCHW vs NHWC note:
//   MiGraphX has MIGRAPHX_ENABLE_NHWC env var which can silently
//   change tensor interpretation.  Layout MUST be explicit in the
//   contract and MUST be logged at compile time so layout-forcing env
//   vars don't produce silent mismatches.
// ─────────────────────────────────────────────────────────────────

#include <cstdint>
#include <string>
#include <vector>

namespace ave {

// ─────────────────────────────────────────────────────────────────
// Enumerations
// ─────────────────────────────────────────────────────────────────

// Tensor axis ordering.
// MiGraphX defaults to NCHW; MIGRAPHX_ENABLE_NHWC triggers a layout
// pass that may change this.  Always log which layout is active.
enum class TensorLayout {
    NCHW,     // Batch × Channels × Height × Width  (MiGraphX default)
    NHWC,     // Batch × Height × Width × Channels  (requires NHWC pass)
    CHW,      // Channels × Height × Width (single-image, no batch dim)
    HWC,      // Height × Width × Channels
    Unknown,  // Layout not yet determined
};

// Inference data type.
// Supported by MiGraphX (opset ≤19):
//   Fp32, Fp16, Bf16, Int8 (with calibration), Fp8E4M3FNUZ (limited arch).
// Ref: Gold Standard §"MiGraphX ONNX support envelope".
enum class TensorDtype {
    Fp32,
    Fp16,
    Bf16,
    Int8,
    Fp8E4M3FNUZ,  // FP8 constrained to E4M3FNUZ; non-trivial conv semantics
    Unknown,
};

std::string toString(TensorLayout layout);
std::string toString(TensorDtype  dtype);

// ─────────────────────────────────────────────────────────────────
// TensorShape
// ─────────────────────────────────────────────────────────────────
struct TensorShape {
    std::vector<std::int64_t> dims;  // e.g. {1, 3, 720, 1280} for NCHW

    // Total element count (product of all dims).  Returns 0 for empty.
    std::int64_t elements() const;

    // Human-readable: "1×3×720×1280"
    std::string format() const;

    bool operator==(const TensorShape& other) const;
    bool operator!=(const TensorShape& other) const;
};

// ─────────────────────────────────────────────────────────────────
// TensorContract
// ─────────────────────────────────────────────────────────────────
// Captures all dimensions of the tensor data contract for one model
// input or output parameter.  Instances are created at model-load
// time from program::get_parameter_shapes() / get_output_shapes()
// and are asserted per-frame during inference.
struct TensorContract {
    std::string  name;    // MiGraphX parameter name (from get_parameter_shapes)
    TensorShape  shape;   // Expected shape (static or per-batch-size)
    TensorLayout layout = TensorLayout::NCHW;
    TensorDtype  dtype  = TensorDtype::Fp32;

    // Optional descriptive context for error messages (e.g. "upscale input")
    std::string  description;

    // Validate internal consistency.
    // - shape.dims must not be empty
    // - name must not be empty
    // - layout/dtype must not be Unknown for production contracts
    bool validate(std::string& error) const;

    // Human-readable summary: "input [1×3×720×1280 NCHW fp32]"
    std::string format() const;
};

// ─────────────────────────────────────────────────────────────────
// Contract assertion helpers
// ─────────────────────────────────────────────────────────────────

// Assert that actual element count matches expected contract.
// Returns false and sets error on mismatch.  Call at every boundary.
bool assertElementCount(const TensorContract& contract,
                        std::size_t           actualElements,
                        std::string&          error);

// Assert that two contracts are compatible (same shape and dtype).
// Useful for checking preprocess output matches inference input.
bool assertContractsMatch(const TensorContract& expected,
                          const TensorContract& actual,
                          std::string&          error);

}  // namespace ave
