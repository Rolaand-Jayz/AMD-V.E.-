# AMD Video Enhancer

AMD-focused C++ video enhancement for Linux. The point of this project is to run restoration, upscale, and interpolation workloads on AMD hardware without CUDA and without a Python-heavy runtime in the main app.

MiGraphX is the primary inference path. Vulkan Compute is the second-choice GPU path. NCNN Vulkan is a fallback for models that fit that runtime. FFmpeg filters remain the last-resort path so the app can still process video when AI backends are unavailable.

## Who This Is For

This app is for:

- Linux users with a modern AMD discrete GPU
- users willing to install and debug ROCm, Vulkan, FFmpeg, and model files
- users who care more about AMD-native performance and control than one-click convenience

This app is not for:

- NVIDIA or CUDA systems
- Windows or macOS users
- users expecting instant first-run behavior with zero model compilation
- unsupported Linux distributions where ROCm packaging is unpredictable

## Target Hardware and OS

The intended target is an AMD Radeon or Radeon Pro system on a ROCm-supported Linux distribution.

As of March 6, 2026, AMD's ROCm system requirements list Radeon RX 7900 GRE support only on `Ubuntu 24.04.3`, `Ubuntu 22.04.5`, `RHEL 10.1`, and `RHEL 9.7`. ROCm also publishes the full supported OS matrix separately. If you are on Arch, CachyOS, Fedora, or another unsupported distro, treat the setup as best-effort and expect backend issues to be part of the environment, not just the app.

Official references:

- ROCm system requirements: <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html>
- ROCm compatibility matrix: <https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html>
- MiGraphX driver reference: <https://rocm.docs.amd.com/projects/AMDMIGraphX/en/latest/migraphx-driver.html>

## What The App Does

The pipeline combines enhancement stages into a deterministic order:

1. restoration and cleanup
2. color correction
3. upscale
4. sharpen
5. interpolation

User-specified order is not trusted. The planner reorders stages so cleanup happens before upscale and interpolation always runs last.

Available stages:

| Stage | Aliases |
| --- | --- |
| `restore_compression` | `decompress`, `deh264` |
| `remove_artifacts` | `deartifact`, `deblock` |
| `denoise` | — |
| `deblur` | — |
| `dehalo` | — |
| `color_fix` | — |
| `upscale` | — |
| `sharpen` | — |
| `interpolate` | — |

## Backend Roles

- `MiGraphX (ROCm)`: loads or compiles `.mxr` artifacts and runs inference on AMD GPUs. This is the main backend the project is built around.
- `Vulkan Compute`: runs compute-shader-based GPU stages when MiGraphX is unavailable or not appropriate.
- `NCNN Vulkan`: fallback GPU inference path for models that exist in NCNN form.
- `FFmpeg`: software and filter fallback. Always available if `ffmpeg` and `ffprobe` are installed.

When `--backend auto` is used, the current priority order is:

1. `MiGraphX`
2. `Vulkan Compute`
3. `NCNN Vulkan`
4. `FFmpeg` fallback inside the pipeline

Explicit backend requests fail honestly if that backend is unavailable. They do not silently switch to another AI backend.

## Model Formats and Storage

The app supports three user-visible model forms:

- downloaded source models: `ONNX`
- downloaded source models: `PyTorch` (`.pth` / `.pt`) when export to ONNX is possible
- compiled MiGraphX artifacts: `.mxr`

PyTorch support is a conversion path, not a native runtime path. Exporting a `.pth` or `.pt` model to ONNX still requires an ambient `python3` + `torch` environment at conversion time.

The project does not expose a general-purpose "hardware optimization" step anymore. The supported backend-specific preparation step is MiGraphX compilation to `.mxr`.

Models live under `~/.local/share/ave/models/`:

| Directory | Contents |
| --- | --- |
| `downloaded/` | Original downloaded ONNX, PyTorch, NCNN, or prebuilt `.mxr` files |
| `migraphx/` | Cached MiGraphX compiled artifacts |

Models are sourced from the curated AMD-focused collection:
<https://github.com/Rolaand-Jayz/awesome-AI-video-enhancing-models-AMD>

## MiGraphX Compilation Behavior

MiGraphX compilation is intentionally real, not fake progress wrapped around a stub.

- First use can take a long time. Small models may compile in seconds; larger frame-size-specific artifacts can take minutes.
- MiGraphX progress is phase-based. On heavier models you may see the same status text repeat for a long time while one compiler phase runs.
- The app now compiles MiGraphX artifacts for the real input frame size instead of blindly compiling a generic 4K artifact.
- MiGraphX artifacts are cached as `.mxr` files and reused on later runs.
- The default MiGraphX compile precision in this app is `fp16`.
- The downloaded source model is kept intact. The app generates a compiled `.mxr` next to it in the cache.

MiGraphX driver features the app relies on:

- `--fp16`: quantizes for fp16
- `--enable-offload-copy`: enables implicit offload copying

Those options are documented by AMD here:
<https://rocm.docs.amd.com/projects/AMDMIGraphX/en/latest/migraphx-driver.html>

## Build

Always build with every supported backend enabled. Do not do a bare CMake configure without backend flags.

Requirements:

- CMake 3.21+
- C++20 compiler such as GCC 12+ or Clang 16+
- FFmpeg CLI tools in `PATH`
- Qt 6.2+ with Widgets for the GUI
- ROCm + MiGraphX for the primary backend
- Vulkan loader and development packages
- NCNN if you want the NCNN Vulkan fallback
- libcurl for model downloads

Standard build for this repo:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DAVE_HAVE_CURL=ON \
  -DAVE_HAVE_MIGRAPHX=ON \
  -DAVE_HAVE_HIP=ON \
  -DAVE_HAVE_VULKAN=ON \
  -DAVE_HAVE_NCNN=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## CLI Usage

List detected backends:

```bash
./build/ave --list-backends
```

Plan a job without running it:

```bash
./build/ave \
  --input input.mp4 \
  --output output.mp4 \
  --stage restore_compression \
  --stage upscale:model=clearreality-x4-fast \
  --stage interpolate:fps=60 \
  --dry-run
```

Run a MiGraphX upscale:

```bash
./build/ave \
  --input input.mp4 \
  --output output.mp4 \
  --backend migraphx \
  --stage upscale:model=clearreality-x4-fast
```

Run a Vulkan Compute stage:

```bash
./build/ave \
  --input input.mp4 \
  --output output.mp4 \
  --backend vulkan \
  --stage denoise:strength=0.6
```

Launch the GUI:

```bash
./build/ave_gui
```

## GUI Overview

The GUI provides:

- stage builder with per-stage parameter controls
- backend-aware model selection
- compile-on-demand action when a MiGraphX-compatible model still needs a `.mxr`
- model manager for downloads and MiGraphX compilation
- preview runs
- profile save/load
- command preview for the equivalent CLI invocation

## Operational Notes

- `ffmpeg` and `ffprobe` are required at runtime for every path.
- Mixed iGPU + dGPU systems can confuse ROCm. Set `HIP_VISIBLE_DEVICES` if MiGraphX picks the wrong GPU.
- The first MiGraphX run is dominated by compile time. Long jobs benefit far more than short previews.
- Current MiGraphX throughput is still constrained by host-staged frame I/O; compile-time improvements and fp16 help, but zero-copy GPU-native frame flow is the next major performance step.

## Troubleshooting

- If MiGraphX is available but model load fails, check whether the selected model actually has a compiled `.mxr` for the real input frame size.
- If an explicit backend is requested and unavailable, the app now reports that honestly instead of silently switching backends.
- If ROCm is installed on an unsupported distro, reproduce on a supported Ubuntu or RHEL build before assuming the app is the only problem.
- If long runs fail during encoding, inspect the reported FFmpeg stderr. Early pipe-close errors usually mean the encoder exited first.
