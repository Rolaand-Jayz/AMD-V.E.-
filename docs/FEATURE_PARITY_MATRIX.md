# Feature Parity Matrix

Current implementation status for the AMD C++ video enhancement pipeline.

## Legend

- `Implemented`: available in the current tree
- `Planned`: intended work, not finished
- `Removed`: intentionally out of scope

## Stages

| Capability | Status | Notes |
| --- | --- | --- |
| Restore compression | Implemented | `restore_compression` stage with model-aware and FFmpeg fallback paths |
| Remove artifacts | Implemented | `remove_artifacts` stage with model-aware and FFmpeg fallback paths |
| Denoise | Implemented | `denoise` stage plus FFmpeg fallback |
| Deblur | Implemented | `deblur` stage plus FFmpeg fallback |
| Dehalo | Implemented | `dehalo` stage plus FFmpeg fallback |
| Color fix | Implemented | `color_fix` stage with typed parameters |
| Upscale | Implemented | model-aware stage with FFmpeg scale fallback |
| Sharpen | Implemented | model-aware stage plus FFmpeg fallback |
| Interpolate | Implemented | interpolation stage with scene-cut aware fallback behavior |
| Deterministic stage ordering | Implemented | planner enforces cleanup before upscale and interpolation last |

## Runtime and model management

| Capability | Status | Notes |
| --- | --- | --- |
| MiGraphX inference | Implemented | primary ROCm backend |
| MiGraphX compile and cache | Implemented | `.mxr` compilation and reuse through ModelManager |
| MiGraphX tiled execution | Implemented | fixed-tile runtime path for large-frame inference |
| Vulkan Compute backend | Implemented | native GPU compute fallback path |
| NCNN Vulkan backend | Implemented | alternate GPU inference path when supported |
| FFmpeg fallback path | Implemented | always-available media path when AI backends are unavailable |
| Model download management | Implemented | libcurl-backed downloads with local model storage |
| PyTorch-to-ONNX export path | Implemented | conversion helper for supported checkpoints |
| INT8 calibration workflow | Implemented | calibration-video-driven MiGraphX int8 path |

## Product surface

| Capability | Status | Notes |
| --- | --- | --- |
| CLI workflow | Implemented | `ave` target |
| Qt GUI | Implemented | `ave_gui` target when Qt6 is available |
| Settings persistence | Implemented | INI-backed app settings |
| Profile save/load | Implemented | JSON profile support in the GUI |
| Planner tests | Implemented | `planner_tests` target |
| End-to-end backend regression tests | Planned | coverage still light outside planner tests |

## Out of scope

| Capability | Status | Notes |
| --- | --- | --- |
| CUDA / NVIDIA backend | Removed | not part of this project |
| Python runtime backend | Removed | Python is only used for specific conversion helpers |
| Windows support | Removed | Linux-only target |
