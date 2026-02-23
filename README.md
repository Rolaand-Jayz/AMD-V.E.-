# AMD Video Enhancer

AMD-exclusive C++ video enhancement pipeline. No Python, no CUDA, no NVIDIA.

- **Preferred backend**: MiGraphX (ROCm) — full ONNX load, compile, and GPU inference
- **Fallback backend**: NCNN Vulkan — Vulkan-accelerated model inference
- **Always available**: FFmpeg filter chain — no model required

## Enhancement stages

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

Stages are ordered deterministically: restoration and cleanup before upscale/sharpen; interpolation always last before encode.

## Build

### Dependencies

- CMake 3.21+, C++20 compiler (GCC 12+ / Clang 16+)
- Qt 6.2+ (Widgets)
- FFmpeg CLI tools (`ffmpeg` and `ffprobe` in PATH)
- *(optional)* libcurl — model downloads (`-DAVE_HAVE_CURL=ON`, default ON)
- *(optional)* ROCm / MiGraphX — GPU acceleration (`-DAVE_HAVE_MIGRAPHX=ON -DAVE_HAVE_HIP=ON`)
- *(optional)* NCNN — Vulkan fallback (`-DAVE_HAVE_NCNN=ON`)

### Standard build (FFmpeg-only)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

### Full ROCm build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DAVE_HAVE_CURL=ON \
  -DAVE_HAVE_MIGRAPHX=ON \
  -DAVE_HAVE_HIP=ON
cmake --build build -j
```

## Usage

### CLI

```bash
./build/ave \
  --input input.mp4 \
  --output output.mp4 \
  --backend auto \
  --stage restore_compression:strength=0.9 \
  --stage remove_artifacts:strength=0.8 \
  --stage upscale:width=3840,height=2160 \
  --stage sharpen:amount=0.5 \
  --stage interpolate:fps=60
```

Dry-run (plan only, no processing):

```bash
./build/ave --input input.mp4 --output output.mp4 \
  --stage sharpen --stage restore_compression --stage interpolate --dry-run
```

Backend discovery:

```bash
./build/ave --list-backends
```

### GUI

```bash
./build/ave_gui
```

The GUI provides:

- Stage stack builder with drag-and-drop reordering
- Per-stage parameter sliders (strength, color correction, resolution, sharpness, FPS)
- Per-stage model selection from downloaded, converted, or hardware-optimised models
- Model Manager — download, MiGraphX conversion, and hardware optimisation for all 26 bundled models
- Profile save/load as JSON
- Quick templates (Web Cleanup 1080p60, Anime Upscale 4K60, Archive Restore 1440p)
- Planned execution order preview
- One-click CLI command copy

## Model storage

Models are sourced from [awesome-AI-video-enhancing-models-AMD](https://github.com/Rolaand-Jayz/awesome-AI-video-enhancing-models-AMD) and stored in `~/.local/share/ave/models/`:

| Directory | Contents |
| --- | --- |
| `downloaded/` | Downloaded PyTorch (.pth), ONNX, and NCNN model files |
| `migraphx/` | MiGraphX `.mxr` compiled programs |
| `optimised/` | Hardware-optimised compiled programs |

### Model Sources

All models are available from the curated collection at:
**https://github.com/Rolaand-Jayz/awesome-AI-video-enhancing-models-AMD**

This repository includes:
- **FBCNN** - JPEG artifact removal
- **SCUNet** - Blind denoising
- **DnCNN** - Gaussian denoising
- **NAFNet** - Deblurring/denoising (3 variants)
- **Restormer** - Multi-task restoration
- **SwinIR** - Super-resolution (2 variants)
- **Real-ESRGAN** - Upscaling (3 variants)
- **DAT** - Dual Aggregation Transformer SR
- **RIFE** - Frame interpolation (v4.6, v4.14, v4.25)

## Backend selection

When `--backend auto` is used (the default), backends are probed in priority order:

1. **MiGraphX (ROCm)** — requires ROCm stack (`/opt/rocm`, `rocminfo`) and `libmigraphx.so` or `migraphx-driver`
2. **NCNN (Vulkan)** — requires Vulkan runtime (`libvulkan.so` or `vulkaninfo`)
3. **FFmpeg filters** — always available; uses classic filter chains without model inference
