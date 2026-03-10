# Feature Parity Matrix

This matrix tracks the implementation status of the AMD C++ video enhancement pipeline.

## Legend

- `Implemented`: Complete and tested.
- `Removed`: Deliberately unsupported by design.

## Inference stages

| Stage | Status | Notes |
| --- | --- | --- |
| Compression restore / DeH264 | Implemented | `restore_compression`; model-aware via ModelManager; FFmpeg filter fallback. |
| Artifact cleanup / deblock | Implemented | `remove_artifacts`; model-aware; FFmpeg filter fallback. |
| Denoise | Implemented | `denoise`; model-aware; FFmpeg `hqdn3d` fallback. |
| Deblur | Implemented | `deblur`; model-aware; FFmpeg `unsharp` fallback. |
| Dehalo | Implemented | `dehalo`; model-aware; FFmpeg filter fallback. |
| Color fix | Implemented | `color_fix`; contrast, brightness, saturation, gamma, vibrance parameters. |
| Upscale | Implemented | `upscale`; model-aware; FFmpeg `scale` fallback. |
| Sharpen | Implemented | `sharpen`; model-aware; FFmpeg `unsharp` fallback. |
| Frame interpolation | Implemented | `interpolate`; target FPS; scene-cut integration; FFmpeg `minterpolate` fallback. |
| Scene-change detection | Implemented | `ffprobe`-based detection; scene cuts injected into interpolation pipeline. |
| Multi-model enhancement stack | Implemented | Arbitrary stage stacking with deterministic planner ordering. |

## Platform and backend

| Category | Status | Notes |
| --- | --- | --- |
| MiGraphX / ROCm acceleration | Implemented | Full ONNX load, compile, and inference when `-DAVE_HAVE_MIGRAPHX=ON`; model path validation without library present. |
| Vulkan Compute backend | Implemented | GPU stage execution via Vulkan compute shaders when `-DAVE_HAVE_VULKAN=ON`. |
| NCNN Vulkan fallback | Implemented | Full model load path when `-DAVE_HAVE_NCNN=ON`; Vulkan GPU detection; FFmpeg fallback when library absent. |
| CUDA / NVIDIA support | Removed | Deliberately unsupported by design. |
| Python runtime backend | Removed | Deliberately unsupported by design. |

## Pipeline requirements

| Requirement | Status | Notes |
| --- | --- | --- |
| Cleanup before upscale/sharpen | Implemented | Enforced by deterministic planner; covered by unit tests. |
| Interpolation last before encode | Implemented | Enforced by deterministic planner; covered by unit tests. |
| Stackable adjustable enhancements | Implemented | Per-stage typed parameters; sliders in GUI; `--stage key=value` in CLI. |
| Model selection per stage | Implemented | 26 bundled model entries; per-stage dropdown in GUI; `model=<id>` param in CLI. |
| Model download / MiGraphX compile / cache | Implemented | ModelManager with curl downloads, ONNX export for supported PyTorch models, and cached `.mxr` compilation through `migraphx-driver`. |
| GUI with toggle controls | Implemented | Qt6 GUI, animated ToggleSwitch widget, per-stage sliders, drag-reorder pipeline. |
| Profile save/load | Implemented | JSON profiles with schema versioning. |
