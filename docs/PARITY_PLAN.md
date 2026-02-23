# Architecture Overview

This document describes the key subsystems of the AMD Video Enhancer pipeline.

## 1. Enhancement stage model

Each requested enhancement is represented as an `EnhancementStage` with a `StageKind` and a typed parameter map (`std::variant<bool, std::int64_t, double, std::string>`). The `Planner` applies deterministic ordering rules: restoration and cleanup stages run before upscale/sharpen, and interpolation is always last before encode.

## 2. Backend selection

`BackendManager::createBackend()` probes available AMD GPU backends in priority order:

1. **MiGraphX (ROCm)** — probed via presence of `/opt/rocm`, `rocminfo`, and `libmigraphx.so` or `migraphx-driver`. Full ONNX load and GPU compile path implemented when compiled with `-DAVE_HAVE_MIGRAPHX=ON -DAVE_HAVE_HIP=ON`. Model path validation runs without the library present.
2. **NCNN Vulkan** — probed via `libvulkan.so` / `vulkaninfo`. Full model load path implemented when compiled with `-DAVE_HAVE_NCNN=ON`. GPU device selection via `ncnn::get_gpu_count()`.
3. **FFmpeg filters** — always available; no GPU or model required.

## 3. Model management

`ModelManager` (backed by the 26-entry `ModelCatalog`) handles:

- **Download** — HTTP(S) via libcurl with progress callbacks; cancellable.
- **Conversion** — `migraphx-driver compile --onnx <in> --output <out.mxr>` CLI invocation.
- **Optimisation** — Hardware-specific compile with `migraphx-driver` tuning flags.
- **State tracking** — Per-model `ModelState` enum persisted via JSON sidecar files in `~/.local/share/ave/models/`.
- **Best-path resolution** — Returns the most refined available file (optimised > converted > downloaded) for a given model ID.

## 4. Video processing pipeline

`VideoProcessor::process()` orchestrates:

1. `ModelManager::refresh()` — scan model directories for state changes.
2. `Planner::plan()` — deterministic stage ordering.
3. Backend initialisation and per-stage `runStage()` calls (non-FFmpeg stages).
4. Scene cut detection via `ffprobe` for interpolation stages with `scene_detect=true`.
5. Model path injection into each stage's parameter map.
6. `FfmpegRunner::encode()` — frame extraction, AI filter application, re-encode.

## 5. FFmpeg integration

`FfmpegRunner` constructs filter graphs from the resolved stage stack. Each `StageKind` maps to one or more FFmpeg filters with parameter-driven arguments. Interpolation stages consume scene cut counts to adapt `minterpolate` threshold.

## 6. GUI architecture

The Qt6 GUI (`ave_gui`) shares the same `ave_core` library as the CLI. Key components:

- `MainWindow` — stage builder, planned-order preview, parameter sliders, drag-reorder `QListWidget`, `ToggleSwitch` widgets.
- `ModelManagerDialog` — per-model download/convert/optimise with thread-safe Qt-queued callbacks.
- `ToggleSwitch` — custom `QAbstractButton` with `QPropertyAnimation` replacing all `QCheckBox` usage.
- Profile serialisation — JSON with `schema_version` field for forward compatibility.
