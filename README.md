# AMD Video Enhancer

Linux-first C++ video enhancement for AMD GPUs. The project keeps the main app native, uses MiGraphX as the primary inference path, and falls back to Vulkan Compute, NCNN Vulkan, or FFmpeg when a given model or runtime is not available.

## Scope

- AMD / ROCm focused
- C++ core with CLI and Qt GUI frontends
- Deterministic stage planning for restoration, cleanup, upscale, sharpen, and interpolation
- Model download, MiGraphX compilation, and local artifact caching

Deliberately out of scope:

- CUDA / NVIDIA support
- Windows and macOS support
- Python as the primary runtime path

## Backends

When `--backend auto` is used, the app prefers backends in this order:

1. MiGraphX
2. Vulkan Compute
3. NCNN Vulkan
4. FFmpeg-only fallback

MiGraphX is the main path for ONNX-based model execution on AMD hardware. The runtime supports cached `.mxr` artifacts and compile-on-demand behavior for first use.

## Build

Always configure with every supported backend enabled:

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

Do not use a bare `cmake -S . -B build` here. The repo expects the backend flags above.

## Run

List detected backends:

```bash
./build/ave --list-backends
```

Dry-run a planned job:

```bash
./build/ave \
  --input input.mp4 \
  --output output.mp4 \
  --stage restore_compression \
  --stage upscale:model=clearreality-x4-fast \
  --dry-run
```

Run the GUI:

```bash
./build/ave_gui
```

## Models

Models are stored under `~/.local/share/ave/models/`:

- `downloaded/`: source ONNX, PyTorch, NCNN, or prebuilt `.mxr` files
- `migraphx/`: cached MiGraphX artifacts

For MiGraphX, first use may trigger a compile step. Those compiled artifacts are reused on later runs.

## Notes

- `ffmpeg` and `ffprobe` must be in `PATH`
- ROCm and MiGraphX must be installed for the primary backend
- Mixed iGPU+dGPU systems may need `HIP_VISIBLE_DEVICES` or `ROCR_VISIBLE_DEVICES`
- On unsupported ROCm distributions, backend behavior is best-effort

MiGraphX compile tuning:

- `AVE_MIGRAPHX_COMPILE_PROFILE=fast|balanced|exhaustive`
- `AVE_MIGRAPHX_PROBLEM_CACHE=/path/to/problem_cache.json`
- `AVE_MIGRAPHX_MIOPEN_FIND_MODE=FAST|DYNAMIC_HYBRID|NORMAL`
- `AVE_MIGRAPHX_MIOPEN_COMPILE_PARALLEL_LEVEL=<n>`
- `AVE_MIGRAPHX_VISIBLE_DEVICES=<gpu-list>`

## Docs

- `docs/FEATURE_PARITY_MATRIX.md`
- `docs/PARITY_PLAN.md`
- `docs/GOLD_STANDARD_FOR_IMPLEMENTATION.md`
- `docs/migraphx_debugging_playbook.md`
