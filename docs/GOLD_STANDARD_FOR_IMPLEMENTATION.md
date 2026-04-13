# Real Implementation Reference for AMD Video Enhancer

AMD Video Enhancer is a Linux-first C++ video enhancer that uses ML and AI models to restore, clean, and upscale video on AMD hardware. It intentionally does **not** target NVIDIA or CUDA, because the whole point of this project is to treat AMD GPUs, ROCm, MiGraphX, HIP, and Vulkan as first-class citizens instead of second-tier fallbacks.

These tools matter because real video is messy: old footage is soft, noisy, blocky, over-compressed, repeatedly re-encoded, badly scaled, or damaged by poor capture pipelines. A good enhancer can recover detail, suppress compression artifacts, clean up ringing and blur, and make low-quality footage more usable for archiving, restoration, editing, research, and everyday viewing.

---

## What this document is

This file is the **current-state technical reference** for the implementation that actually exists in this repository.

It replaces the older version of this document, which described a more aspirational end-to-end Vulkan↔HIP↔MiGraphX architecture than the repo presently runs in production. In other words: this is the “no fantasy football” edition.

---

## Why an AMD-first implementation matters

The software landscape for AI video enhancement has largely grown around NVIDIA-first tooling. Commercial products and open-source projects alike often build their fastest paths around CUDA, Tensor cores, TensorRT-style assumptions, or ecosystems that matured there first, which means AMD users are often left with slower generic paths, thinner feature coverage, or no serious support at all.

That gap keeps widening because CUDA has had a long head start in adoption, documentation, mindshare, production tooling, and example code. ROCm has grown a lot, but pieces of the AMD stack—especially **MiGraphX**—remain far less visible in public end-user applications, which makes it harder for developers to see what is possible, harder to copy proven patterns, and harder for AMD users to get a polished experience comparable to what NVIDIA users routinely receive.

This project exists to close part of that gap with a public, inspectable, native C++ implementation.

---

## What is actually shipped here

At a high level, the repository implements:

- a **native C++20 core**
- a **CLI frontend** and an optional **Qt GUI frontend**
- a **deterministic stage planner**
- a **model catalog + model manager**
- a **MiGraphX-first inference path** for AMD GPUs
- a **ROCm/HIP ONNX Runtime fallback**
- a **Vulkan Compute fallback**
- an **NCNN Vulkan fallback**
- an **FFmpeg-based media pipeline and fallback filter path**
- a **native Linux packaging pipeline** for multiple distros

The real implementation is broader than “MiGraphX only,” and more honest than “full zero-copy interop everywhere.”

---

## Source of truth in the codebase

If you want to verify anything in this document, start with these files:

| Area | File(s) |
| --- | --- |
| Backend selection | `src/backend_manager.cpp`, `include/ave/backend_manager.hpp` |
| End-to-end processing flow | `src/video_processor.cpp`, `include/ave/video_processor.hpp` |
| MiGraphX backend | `src/backends/migraphx_backend.cpp`, `include/ave/backends/migraphx_backend.hpp` |
| ROCm/HIP fallback | `src/backends/rocm_hip_backend.cpp`, `include/ave/backends/rocm_hip_backend.hpp` |
| Vulkan compute fallback | `src/backends/vulkan_compute_backend.cpp`, `include/ave/backends/vulkan_compute_backend.hpp` |
| NCNN Vulkan fallback | `src/backends/ncnn_vulkan_backend.cpp`, `include/ave/backends/ncnn_vulkan_backend.hpp` |
| Vulkan↔HIP bridge surface | `src/interop_bridge.cpp`, `include/ave/interop_bridge.hpp` |
| Model lifecycle | `src/model_manager.cpp`, `include/ave/model_manager.hpp` |
| Stage ordering | `src/planner.cpp`, `include/ave/planner.hpp` |
| Build + install topology | `CMakeLists.txt` |
| Native packaging | `tools/build_native_packages.py` |

---

## The real architecture in one pass

Here is the actual runtime story today:

1. The user configures a job in the **CLI** or **Qt GUI**.
2. The app turns requested stages into a deterministic pipeline using `PipelinePlanner`.
3. `VideoProcessor` resolves models, probes available runtimes, and selects a backend.
4. The selected backend performs a lightweight stage-preload phase through `runStage(...)`.
5. Real frame-by-frame AI work happens during the encode session through `processVideoFile(...)`.
6. FFmpeg remains the media spine around the AI work: probing, frame flow, filters when needed, and final encode.

That matters, because the repo is **not** structured as “load one magical engine and it does everything.” It is a layered video pipeline with multiple execution paths, where AI inference plugs into a larger media-processing framework.

---

## Actual backend priority order

When the user selects `--backend auto`, the backend order is currently:

1. **MiGraphX**
2. **ROCm/HIP (ONNX Runtime)**
3. **Vulkan Compute**
4. **NCNN Vulkan**
5. **FFmpeg-only fallback** if no AMD AI backend is usable

This order comes from `BackendManager::createBackend(...)` in `src/backend_manager.cpp`.

### Why that order exists

- **MiGraphX** is the preferred AMD-native inference path.
- **ROCm/HIP ONNX Runtime** is the next best general AMD fallback when MiGraphX is unavailable or unsuitable.
- **Vulkan Compute** provides a native GPU compute fallback that does not depend on MiGraphX.
- **NCNN Vulkan** provides another GPU path for compatible NCNN models.
- **FFmpeg** is the safety net when AI backends are unavailable or a specific stage/model cannot run there.

---

## Deterministic stage planning

The app does not trust raw user stage order as final execution order.

`PipelinePlanner` sorts stages into a fixed sequence so cleanup happens before upscale and interpolation stays last:

| Planner group | Stages |
| --- | --- |
| 0 | `restore_compression`, `remove_artifacts`, `denoise`, `deblur`, `dehalo` |
| 1 | `color_fix` |
| 2 | `upscale` |
| 3 | `sharpen` |
| 4 | `stereo_3d` |
| 5 | `interpolate` |

This is intentionally simple and deterministic. The planner is not a speculative optimizer; it is a correctness guardrail.

---

## Frontends: how users drive the pipeline

### CLI

The CLI binary is `ave`.

It supports:

- explicit backend selection
- stage declarations and stage parameters
- dry-run planning
- runtime diagnostics and backend listing

Relevant code:

- `src/main.cpp`
- `src/cli.cpp`

### Qt GUI

The optional GUI binary is `ave_gui`.

It provides:

- stage selection
- filter browsing
- profile save/load
- settings persistence
- queueing and preview/run flows
- model-management dialogs

Relevant code:

- `src/gui/main_window.cpp`
- `src/gui/model_manager_dialog.cpp`
- `src/gui/settings_dialog.cpp`

---

## Media pipeline: what wraps the AI work

The repository is not an image-only ML demo. It is a video application, so **FFmpeg and frame I/O are central**.

### The real media spine

- `VideoProcessor` coordinates the whole job.
- `FfmpegRunner` drives encode-time processing.
- `frame_io` and `rgb_video_loop` move RGB frame data through the selected backend.
- `video_probe` and runtime diagnostics gather media and platform metadata.

The app typically runs a **continuous encode session**, where the selected backend processes frames while FFmpeg handles the surrounding video workflow.

Relevant code:

- `src/video_processor.cpp`
- `src/ffmpeg_runner.cpp`
- `src/frame_io.cpp`
- `src/frame_io_vulkan.cpp`
- `src/rgb_video_loop.cpp`

---

## Model management and artifact lifecycle

One of the repo’s strongest implemented pieces is the **ModelManager**.

### What it does

`ModelManager` is responsible for:

- scanning the model cache
- tracking model state
- downloading source models
- preparing ONNX variants for MiGraphX compatibility when needed
- compiling MiGraphX `.mxr` artifacts
- validating compiled artifacts against the active runtime fingerprint
- selecting the safest path for each backend

### Model states

The model lifecycle is explicit:

- `NotDownloaded`
- `Downloading`
- `Downloaded`
- `Converting`
- `Converted`
- `Error`

### Important directories

The repo and docs describe three especially important model areas:

- `downloaded/` — source ONNX, NCNN, or PyTorch assets
- `prepared/` — ONNX variants rewritten for MiGraphX compatibility when required
- compiled MiGraphX artifact storage (`.mxr`) under the model cache

### Custom manifests

Custom models are supported through `.avemodel` manifest files, which define things like:

- model id and display name
- stage and capabilities
- source format
- precision
- fused capability behavior
- explicit control-input bindings

Relevant code and docs:

- `src/model_manager.cpp`
- `include/ave/model_manager.hpp`
- `docs/CUSTOM_MODEL_MANIFEST.md`

---

## MiGraphX: the primary backend

MiGraphX is the preferred inference backend in this repo.

### What the backend actually does today

The MiGraphX backend is implemented in `src/backends/migraphx_backend.cpp`. Its real responsibilities include:

- checking availability of ROCm/MiGraphX artifacts
- preparing or selecting compiled `.mxr` artifacts
- validating artifact manifests against the active runtime fingerprint
- loading programs and building tensor contracts
- driving tiled frame inference during `processVideoFile(...)`
- logging runtime configuration, timing, and ROCTx ranges

### Compilation model

This repo does **not** rely on an in-process “magic compile everything however you want” approach.

Instead, it uses an explicit artifact model:

1. find or prepare an ONNX source
2. compile it with `migraphx-driver`
3. emit an `.mxr` program artifact
4. write a manifest sidecar
5. reuse the artifact only when the manifest still matches the active runtime

That is one of the most important real implementation details in the project.

### Runtime identity and manifest validation

Compiled artifacts are keyed against things such as:

- MiGraphX version
- ROCm version
- GPU architecture / visible device binding
- compile profile
- precision
- selected runtime env knobs
- source fingerprint of the ONNX input

This is why the MiGraphX path is more robust than “find any `.mxr` and hope.”

### Tiled inference

For large inputs, MiGraphX processing uses tile planning rather than assuming every model can or should run as a single full-frame tensor all the time.

The backend supports:

- tile width / height / overlap configuration
- adaptive batch sizing
- exact-shape artifact reuse when available
- fallback artifact selection when needed

### Input path today: honest version

This is the part the old document overstated.

The repo **does** contain a Vulkan↔HIP interop bridge surface and the MiGraphX backend is clearly written with that future in mind. But the current MiGraphX production path still centers on **host staging**, with an optional **HIP-mapped host staging** optimization for input buffers when conditions allow.

That means the project is already serious about performance engineering, but it is **not yet a universal always-on zero-copy Vulkan→HIP→MiGraphX→Vulkan pipeline**.

### Frame source modes

MiGraphX can prefer different frame source modes, including:

- `RawPipe`
- `VulkanTransfer`
- `VulkanHardware`

Those are runtime choices around how frames are sourced into the inference loop, not proof that the full end-to-end path is always a pure GPU-memory interop path.

### What to learn from this backend

If you want to build on this work, study:

- artifact validation
- runtime fingerprinting
- tile planning
- tensor contract checking
- explicit fallback behavior when a model is incompatible

This backend is the best example in the repo of “real system engineering around ML inference,” not just “run a model once.”

---

## ROCm/HIP fallback via ONNX Runtime

The secondary AMD path is `ROCm/HIP (ONNX Runtime)` in `src/backends/rocm_hip_backend.cpp`.

### What it is

This backend uses **ONNX Runtime with the ROCM execution provider** as a fallback when MiGraphX is unavailable or unsuitable.

### What it currently supports

The implementation is intentionally narrower than MiGraphX:

- single-input, single-output image models
- RGB image inference
- batch size 1
- fp32 or fp16 tensor I/O
- stages other than `interpolate` and `stereo_3d`

### Why it matters

This backend keeps the app from becoming “MiGraphX or bust.” That is a real product decision, not just a convenience: it gives AMD users another ROCm-native inference option when MiGraphX is not the right fit.

---

## Vulkan Compute fallback

The Vulkan Compute backend lives in `src/backends/vulkan_compute_backend.cpp`.

### What the Vulkan path does

It provides a native GPU compute fallback path using embedded GLSL compute shaders and Vulkan pipeline management.

### How it is used

- selected explicitly by the user, or
- chosen automatically when MiGraphX and ROCm/HIP are unavailable

### Practical role

This backend matters because it keeps the app AMD/GPU-first even when the preferred ML path is unavailable.

---

## NCNN Vulkan fallback

The NCNN backend lives in `src/backends/ncnn_vulkan_backend.cpp`.

### What the NCNN path does

It runs compatible NCNN models through a Vulkan-backed NCNN runtime.

### Why it exists

It broadens model compatibility and gives the app one more GPU-side fallback before dropping all the way to non-AI filtering.

---

## FFmpeg fallback and filter path

FFmpeg is not just a decode helper here. It is the **guaranteed fallback path** and the encode backbone.

### What FFmpeg is responsible for

- media probing
- frame flow
- encode session orchestration
- fallback filter execution when AI backends defer a stage

### Why this is good engineering

The app does not pretend AI can or should do every job. When a model is missing, incompatible, or unsupported for a stage, the pipeline can still finish the job using FFmpeg filters.

That keeps the app useful instead of brittle.

---

## The Vulkan↔HIP interop bridge: what is real today

The interop bridge is implemented in:

- `include/ave/interop_bridge.hpp`
- `src/interop_bridge.cpp`

### What it actually provides

The bridge exposes:

- availability probing
- Vulkan external memory import into HIP
- Vulkan external semaphore import into HIP
- signal/wait operations for explicit synchronization
- configuration logging

### What this means in practice

This is a **real implemented bridge component**, not a fake placeholder.

### What it does **not** mean

It does **not** mean the entire repo already routes all MiGraphX inference through exported Vulkan device memory by default. The bridge exists, is implemented, and is logged by the backend manager and MiGraphX backend, but the default production MiGraphX path still relies on host staging plus optional HIP-mapped host optimizations.

That distinction is crucial.

---

## Build system and feature toggles

The build is defined in `CMakeLists.txt` and is much richer than a one-binary prototype.

### Important options

- `AVE_HAVE_MIGRAPHX`
- `AVE_HAVE_HIP`
- `AVE_HAVE_ONNXRUNTIME_ROCM`
- `AVE_HAVE_VULKAN`
- `AVE_HAVE_NCNN`
- `AVE_HAVE_CURL`
- `AVE_BUILD_GUI`
- `AVE_HAVE_ROCTX`

### Recommended full build

The repo guidance is clear: configure with all major backends enabled.

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

This app is deliberately native and Linux-focused. It is not trying to hide the platform reality behind a fake “works everywhere equally” story.

---

## Packaging and release engineering

The packaging story is also real, and it matters.

### What is implemented

The repo supports:

- staged install roots
- bundled app-private runtime payloads under `/opt/amd-video-enhancer`
- distro-native package generation
- portable bundle staging
- wrapper scripts for installed binaries

### Native package generation

`tools/build_native_packages.py` builds packages for targets including:

- Arch Linux
- Ubuntu
- Debian
- Fedora
- openSUSE
- Rocky Linux
- AlmaLinux

The script computes a version, stages the payload root, and emits distro-specific package artifacts.

This is not “throw a binary in a zip and call it release engineering.”

---

## Testing, diagnostics, and observability

The repo contains real tests across core subsystems, including:

- planner behavior
- tensor contracts
- runtime diagnostics
- runtime paths
- model-manager profiles
- video probing
- process loops and observers
- frame I/O
- FFmpeg runner behavior
- job recovery and queueing
- telemetry
- MiGraphX backend behavior
- interop bridge behavior

### Observability features worth learning from

- runtime diagnostics reports
- telemetry sampling for AMD hardware
- ROCTx markers for rocprof integration
- pipeline timing logs
- artifact manifest logging and validation

This is another place where the repo is stronger than a typical AI toy project: it is trying to be debuggable.

---

## Honest boundaries and non-goals

This project currently **is**:

- Linux-first
- C++-native
- AMD-focused
- MiGraphX-first with fallback layers
- usable through CLI and Qt GUI
- designed to be read, modified, and extended

This project currently **is not**:

- a CUDA/NVIDIA application
- a Windows or macOS app
- a Python runtime product
- a universal zero-copy Vulkan↔HIP end-to-end pipeline on every inference path
- proof of a verified “first ever consumer MiGraphX app” claim

On that last point: public MiGraphX consumer-facing examples appear rare, and this repo clearly pushes the stack farther into end-user application territory than most public examples. But unless a claim is verified, this document intentionally does **not** declare “first ever.”

---

## How to learn from this repo in the right order

If you want to replicate or extend this work, read the code in this sequence:

1. `src/video_processor.cpp` — the end-to-end control flow
2. `src/backend_manager.cpp` — backend probing and fallback order
3. `src/planner.cpp` — stage ordering rules
4. `src/model_manager.cpp` — model lifecycle and artifact policy
5. `src/backends/migraphx_backend.cpp` — primary inference path
6. `src/backends/rocm_hip_backend.cpp` — ROCm/HIP ONNX Runtime fallback
7. `src/backends/vulkan_compute_backend.cpp` — Vulkan compute fallback
8. `src/backends/ncnn_vulkan_backend.cpp` — NCNN Vulkan fallback
9. `src/interop_bridge.cpp` — explicit GPU interop bridge surface
10. `tools/build_native_packages.py` — packaging pipeline

That reading order mirrors the real architecture.

---

## Bottom line

The real “gold standard” for this repository is not a hypothetical future where every frame stays in GPU memory from decode to inference to encode with no compromise. The real gold standard is an **honest, layered, production-minded AMD video enhancement application** that already combines deterministic planning, model management, MiGraphX artifact discipline, ROCm/HIP fallback, Vulkan/NCNN fallback layers, FFmpeg integration, diagnostics, tests, and distro packaging.

That is the implementation this repository actually contains today, and that is the foundation others can study, reproduce, modify, and improve.
