# Backend Implementation Guide

This folder contains the backend-specific execution paths for AMD Video Enhancer. These files are where the abstract idea of “run this stage” turns into concrete work on MiGraphX, ROCm/HIP, Vulkan, NCNN, GLSL, or VapourSynth.

## The important distinction

Not every backend here participates in the same way.

- some are part of the normal `--backend auto` chain
- some are explicit/specialized backends you request on purpose
- some exist to widen compatibility rather than to be the fastest default path

## Actual backend order for `--backend auto`

Today the normal auto-selected order is:

1. `migraphx_backend.cpp`
2. `rocm_hip_backend.cpp`
3. `vulkan_compute_backend.cpp`
4. `ncnn_vulkan_backend.cpp`
5. FFmpeg fallback outside this folder

That order is managed by `BackendManager` in [`../backend_manager.cpp`](../backend_manager.cpp).

## File-by-file map

| File | Role |
| --- | --- |
| `migraphx_backend.cpp` | primary AMD inference backend; handles artifact selection, `.mxr` runtime validation, tensor contracts, tile planning, and MiGraphX execution |
| `rocm_hip_backend.cpp` | ONNX Runtime ROCm execution-provider fallback for AMD GPUs when MiGraphX is unavailable or unsuitable |
| `vulkan_compute_backend.cpp` | Vulkan compute fallback path for GPU processing without relying on MiGraphX |
| `ncnn_vulkan_backend.cpp` | NCNN-based Vulkan inference path for compatible model/runtime combinations |
| `glsl_shader_backend.cpp` | shader-based backend that applies GLSL processing through FFmpeg `libplacebo` or `mpv` GPU shader support |
| `vapoursynth_backend.cpp` | VapourSynth-based processing backend with direct-script and image-sequence fallback logic when that ecosystem is available |

## How to read these files

If you only have time to understand one backend, start with `migraphx_backend.cpp`. It contains the most important AMD-first implementation work in the project:

- model artifact discipline
- runtime fingerprint checks
- input/output tensor setup
- tile planning
- performance-minded execution decisions

Then read `rocm_hip_backend.cpp` to understand the backup AMD path. After that, `vulkan_compute_backend.cpp` and `ncnn_vulkan_backend.cpp` show how the project stays useful even when the primary inference stack is not available.

## Why the explicit backends matter too

`glsl_shader_backend.cpp` and `vapoursynth_backend.cpp` are not just clutter. They show that the project is willing to widen compatibility and give users alternate execution styles instead of pretending there is only one acceptable path.

## Pair this with

- [`../README.md`](../README.md) for the whole source-tree map
- [`../../docs/GOLD_STANDARD_FOR_IMPLEMENTATION.md`](../../docs/GOLD_STANDARD_FOR_IMPLEMENTATION.md) for the current architecture story
- [`../../include/ave/README.md`](../../include/ave/README.md) for the public header side of these backends
