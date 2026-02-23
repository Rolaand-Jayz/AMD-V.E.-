# AI Inference Pipeline — Handoff Document

**Date**: 2026-02-22  
**Status**: Partially implemented, build fails — MiGraphX API incompatibilities remain  
**Goal**: Make the video enhancer actually run neural network inference on video frames instead of only applying basic FFmpeg signal-processing filters.

---

## 1. The Core Problem

The AMD Video Enhancer was producing output that looked identical before and after "AI enhancement" because **no model inference was ever executed**. Two root causes:

### Root Cause A: All AI libraries compiled OFF

Despite MiGraphX, HIP, NCNN, and Vulkan all being installed on the system, the CMake build had:

```
AVE_HAVE_MIGRAPHX=OFF
AVE_HAVE_NCNN=OFF
AVE_HAVE_HIP=OFF
```

This caused every backend to compile as a stub that returns `StageResult::Deferred` for every stage.

### Root Cause B: No frame-processing loop existed

Even if the libraries were compiled in:

- `IAcceleratorBackend::runStage()` only validated/preloaded models — it never touched frame data.
- There was no method to feed frames through a model and write results back.
- The `FfmpegRunner::encode()` had no access to the backend at all (no parameter for it).
- The only working AI path was `realesrgan-ncnn-vulkan` for upscale (an external CLI tool, not the backend).

For every other stage (denoise, deblur, dehalo, restore-compression, remove-artifacts, color-fix, sharpen), only FFmpeg signal-processing filters (`hqdn3d`, `deblock`, `unsharp`, etc.) were ever applied.

---

## 2. System Environment

All AI libraries **are installed** on this system:

| Component | Location | Version |
|-----------|----------|---------|
| ROCm | `/opt/rocm/` | 6.x / 7.2.x |
| MiGraphX | `/opt/rocm/lib/libmigraphx_c.so.3`, headers at `/opt/rocm/include/migraphx/` | installed |
| HIP | `/opt/rocm/include/hip/hip_runtime.h` | 7.2.x |
| NCNN | `/usr/lib/libncnn.so`, headers at `/usr/include/ncnn/` | 20260203 |
| Vulkan | `vulkaninfo` in PATH, `libvulkan.so` | installed |
| realesrgan-ncnn-vulkan | in PATH | installed |
| FFmpeg / ffprobe | in PATH | installed |

CMake config files:
- ncnn: `/usr/lib/cmake/ncnn/ncnnConfig.cmake`
- MiGraphX: `/opt/rocm/lib/cmake/migraphx/migraphx-config.cmake`
- HIP: `/opt/rocm/lib/cmake/hip/hip-config.cmake`
- hiprtc: `/opt/rocm/lib/cmake/hiprtc/hiprtc-config.cmake` ← **required by MiGraphX**

---

## 3. What Has Been Done

### 3.1 New files created

| File | Purpose |
|------|---------|
| `include/ave/frame_io.hpp` | PNG I/O utilities header — dimension probe, RGB24 load/save via FFmpeg pipes, NCHW↔RGB24 tensor conversion, directory listing |
| `src/frame_io.cpp` | Implementation of above — uses `popen()` + FFmpeg for PNG decode/encode, direct IHDR parsing for dimensions, manual tensor layout conversion |

### 3.2 Modified files

| File | Changes |
|------|---------|
| `include/ave/backend.hpp` | Added `StageResult` enum (`Processed`/`Deferred`/`Error`), `FrameProgressCb` typedef, `processFrameDir()` pure virtual method to `IAcceleratorBackend` |
| `include/ave/stage.hpp` | Added `bool backendProcessed = false` field to `EnhancementStage` |
| `include/ave/backends/migraphx_backend.hpp` | Added `processFrameDir` override declaration |
| `include/ave/backends/ncnn_vulkan_backend.hpp` | Added `processFrameDir` override declaration |
| `src/backends/migraphx_backend.cpp` | Added `#include "ave/frame_io.hpp"`, full `processFrameDir()` impl (~140 lines) — loads PNG → tensor → `runInference()` → save PNG |
| `src/backends/ncnn_vulkan_backend.cpp` | Added `#include "ave/frame_io.hpp"`, fixed NCNN include paths (`<net.h>` not `<ncnn/ncnn/net.h>`), full `processFrameDir()` impl (~120 lines) — load PNG → `ncnn::Mat` → Extractor → save PNG |
| `include/ave/ffmpeg_runner.hpp` | Changed `encode()` signature to accept `IAcceleratorBackend* backend` parameter |
| `src/ffmpeg_runner.cpp` | Added `hasModelPath()`, `isAiProcessable()` helpers. Added `encodeWithAiProcessing()` (~250 lines) — generalized pipeline that extracts frames, processes AI stages via `processFrameDir()` or realesrgan, handles intermediate FFmpeg filters, and re-encodes. Updated `FfmpegRunner::encode()` to route through the new pipeline when AI stages are detected. |
| `src/video_processor.cpp` | Changed `ffmpeg_.encode()` call to pass `backend.get()` as the new parameter |
| `CMakeLists.txt` | Added `src/frame_io.cpp` to `ave_core` sources. Added `find_package(hiprtc QUIET)` before `find_package(migraphx)` (MiGraphX depends on hiprtc). |

### 3.3 CMake reconfiguration

Successfully configured with:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DAVE_HAVE_NCNN=ON \
  -DAVE_HAVE_MIGRAPHX=ON \
  -DAVE_HAVE_HIP=ON \
  -DCMAKE_PREFIX_PATH=/opt/rocm
```

All libraries found. Cache now shows `AVE_HAVE_MIGRAPHX=ON`, `AVE_HAVE_NCNN=ON`, `AVE_HAVE_HIP=ON`.

---

## 4. Current Build Errors (What Remains)

The build command `cmake --build . -j$(nproc)` produces the following errors:

### 4.1 `src/backends/migraphx_backend.cpp` — MiGraphX C++ API Mismatches

The code was written against an assumed older MiGraphX API. The installed version uses different types and methods.

#### Error 1: `migraphx::shape::type_t` does not exist

**File**: `src/backends/migraphx_backend.cpp` **around line 382**

```cpp
// CURRENT (broken):
TensorDtype mapMiGraphXType(migraphx::shape::type_t t) {
    using T = migraphx::shape::type_t;
    switch (t) {
        case T::float_type: ...
```

**Fix**: The correct type is `migraphx_shape_datatype_t` (a C enum, not a nested C++ type). The enum values are like `migraphx_shape_float_type`, `migraphx_shape_half_type`, etc.

```cpp
// CORRECT:
TensorDtype mapMiGraphXType(migraphx_shape_datatype_t t) {
    switch (t) {
        case migraphx_shape_float_type:       return TensorDtype::Fp32;
        case migraphx_shape_half_type:        return TensorDtype::Fp16;
        case migraphx_shape_bf16_type:        return TensorDtype::Bf16;
        case migraphx_shape_int8_type:        return TensorDtype::Int8;
        case migraphx_shape_fp8e4m3fnuz_type: return TensorDtype::Fp8E4M3FNUZ;
        default:                              return TensorDtype::Unknown;
    }
}
```

#### Error 2: `shape.lens()` → `shape.lengths()`

**File**: `src/backends/migraphx_backend.cpp` **lines 452 and 573**

```cpp
// CURRENT (broken):
for (const auto len : shape.lens()) {
// CORRECT:
for (const auto len : shape.lengths()) {
```

Appears in `buildContracts()` and in the output-contract building loop in `loadProgram()`.

#### Error 3: `buildContracts()` parameter type mismatch

**File**: `src/backends/migraphx_backend.cpp` **around line 440 and called at line 566**

```cpp
// CURRENT (broken):
static std::vector<TensorContract> buildContracts(
    const std::unordered_map<std::string, migraphx::shape>& shapes, ...)
```

`program.get_parameter_shapes()` returns `migraphx::program_parameter_shapes`, NOT `std::unordered_map`. The `program_parameter_shapes` type has:
- `size_t size()`
- `shape operator[](const char* name)`
- `std::vector<const char*> names()`

**Fix**: Change `buildContracts` signature and body:

```cpp
static std::vector<TensorContract> buildContracts(
    const migraphx::program_parameter_shapes& shapes,
    const std::string& role) {
    std::vector<TensorContract> result;
    for (const char* name : shapes.names()) {
        if (std::string(name).rfind("#output", 0) == 0) continue;
        const auto shape = shapes[name];
        TensorContract c;
        c.name        = name;
        c.description = role + " parameter";
        c.dtype       = mapMiGraphXType(shape.type());
        c.shape.dims.clear();
        for (const auto len : shape.lengths()) {
            c.shape.dims.push_back(static_cast<std::int64_t>(len));
        }
        const bool nhwcEnv = std::getenv("MIGRAPHX_ENABLE_NHWC") != nullptr;
        c.layout = nhwcEnv ? TensorLayout::NHWC : TensorLayout::NCHW;
        result.push_back(std::move(c));
    }
    return result;
}
```

#### Error 4: `program_parameter_shapes::at()` does not exist

**File**: `src/backends/migraphx_backend.cpp` **line 658** (in `runInference()`)

```cpp
// CURRENT (broken):
const auto& inShape = mp.prog.get_parameter_shapes().at(inName);
// CORRECT:
const auto inShape = mp.prog.get_parameter_shapes()[inName.c_str()];
```

Note: returns by value (not reference), and takes `const char*`.

#### Error 5: `migraphx::argument` constructor mismatch

**File**: `src/backends/migraphx_backend.cpp` **around line 662**

```cpp
// CURRENT (broken):
migraphx::argument inArg{inShape, const_cast<void*>(
    static_cast<const void*>(inputData))};
```

The `argument(shape, void*)` constructor requires a non-const `void*`. The `inputData` parameter is `const float*`. You need to `const_cast` it to `void*`.

```cpp
// CORRECT:
migraphx::argument inArg(inShape,
    const_cast<void*>(static_cast<const void*>(inputData)));
```

This may already be correct syntactically but the braced-init-list (`{}`) form fails template deduction. Use parentheses `()` instead.

#### Error 6: `program::finish()` does not exist

**File**: `src/backends/migraphx_backend.cpp` **line 675**

```cpp
// CURRENT (broken):
mp.prog.finish();
```

The installed MiGraphX API has no `finish()` method on `program`. The gold standard document mentions it, but the actual C++ API doesn't expose it. **Remove this line entirely.** The `eval()` call is synchronous in the current CPU-staging path.

#### Error 7: `hipSetDevice()` return value warning

**File**: `src/backends/migraphx_backend.cpp` **line 796**

```cpp
// CURRENT (warning, not error):
hipSetDevice(impl_->deviceIdx);
// FIX:
(void)hipSetDevice(impl_->deviceIdx);
// Or check the return value properly.
```

### 4.2 `src/observability.cpp` — `MIGRAPHX_VERSION` macro

**File**: `src/observability.cpp` **line 76**

```cpp
// CURRENT (broken):
const std::string migraphxVer = MIGRAPHX_VERSION;
```

`MIGRAPHX_VERSION` is not defined. Available macros are `MIGRAPHX_VERSION_MAJOR`, `MIGRAPHX_VERSION_MINOR`, `MIGRAPHX_VERSION_PATCH`.

**Fix**:
```cpp
const std::string migraphxVer =
    std::to_string(MIGRAPHX_VERSION_MAJOR) + "." +
    std::to_string(MIGRAPHX_VERSION_MINOR) + "." +
    std::to_string(MIGRAPHX_VERSION_PATCH);
```

### 4.3 `src/ffmpeg_runner.cpp` — Unused variables/functions (warnings)

These are `-Wunused-variable` and `-Wunused-function` warnings treated as errors with strict warnings:

| Location | Issue | Fix |
|----------|-------|-----|
| Line 1069 | `bool pastLastAi = false;` — unused | Remove the variable |
| Line 748 | `encodeWithAiUpscale()` — unused function | Remove it or wrap in `#if 0` (it's superseded by `encodeWithAiProcessing`) |
| Line 641 | `firstAiUpscaleIndex()` — unused function | Remove it (superseded) |

### 4.4 `src/backends/ncnn_vulkan_backend.cpp` — Double namespace closing

**File**: end of `src/backends/ncnn_vulkan_backend.cpp` (line ~382)

```cpp
// CURRENT (broken):
}  // namespace ave}  // namespace ave
// CORRECT:
}  // namespace ave
```

There's a doubled `}  // namespace ave` that will cause a parse error.

---

## 5. Architecture of the New Pipeline

### Before (broken):

```
VideoProcessor::process()
  → backend->runStage()  →  returns Deferred (always)
  → FfmpegRunner::encode()  →  builds FFmpeg filter chain  →  output
```

Every stage used an FFmpeg filter (hqdn3d, deblock, unsharp, etc.). No neural network ever ran.

### After (new design):

```
VideoProcessor::process()
  → backend->runStage()  →  preloads model, returns Deferred
  → FfmpegRunner::encode(job, stages, backend, error)
      → isAiProcessable(stage)?
        YES → encodeWithAiProcessing():
          1. Extract frames from video (with pre-AI FFmpeg filters applied)
          2. For each AI stage:
             - Upscale with engine=ai → realesrgan-ncnn-vulkan (external tool)
             - Other AI stage → backend->processFrameDir(stage, inDir, outDir)
               → Load PNG frame → tensor → model inference → save PNG
             - Intermediate FFmpeg filters applied between AI stages
          3. Re-encode final frames to output (with post-AI filters)
        NO → Simple FFmpeg filter chain (existing path)
```

### Key new interfaces:

```cpp
// In IAcceleratorBackend:
virtual StageResult processFrameDir(
    const EnhancementStage& stage,
    const std::string& inputDir,
    const std::string& outputDir,
    const FrameProgressCb& progressCb,
    std::string& error) = 0;

// In FfmpegRunner:
bool encode(const VideoJob& job,
            const std::vector<EnhancementStage>& orderedStages,
            IAcceleratorBackend* backend,  // NEW — was not here before
            std::string& error) const;
```

### What determines "AI-processable":

A stage is AI-processable if:
1. `shouldUseAiUpscale(stage)` is true (upscale with `engine=ai`), OR
2. `backend != nullptr` AND `stage.params["model_path"]` exists (set by `VideoProcessor::resolveModelPath()` when a model file is found on disk) AND `stage.backendProcessed == false`

---

## 6. Model Availability

### Model catalog

`src/model_catalog.cpp` defines 26+ ONNX models for all stages. All are `ModelFormat::Onnx`, `ModelPrecision::Fp32`.

### For MiGraphX to work

Models must be:
1. **Downloaded** as ONNX files (via Model Manager or manually to `~/.local/share/ave/models/downloaded/`)
2. **Compiled** to `.mxr` format (via Model Manager "Convert to MiGraphX" or `migraphx-driver` CLI)

The `loadProgram()` in the MiGraphX backend **rejects non-.mxr files**. If only the `.onnx` exists, inference is deferred to FFmpeg. Consider adding an auto-compile fallback (parse ONNX → compile → cache) as a future enhancement.

### For NCNN to work

Models must be converted to NCNN `.param`/`.bin` format (using `onnx2ncnn` or `pnnx` tools). The NCNN backend's `loadNet()` looks for `.param`/`.bin` pairs. All catalog entries are ONNX format, so a conversion step is needed. Currently NCNN will defer to FFmpeg if no NCNN-format model files exist.

### For realesrgan-ncnn-vulkan (upscale)

This already works. It's an external CLI tool with its own bundled models at `/usr/share/realesrgan-ncnn-vulkan/models/`. No compilation/conversion needed.

---

## 7. Exact Steps to Complete

### Step 1: Fix all build errors

Apply the fixes documented in Section 4 above. All are localized code changes:

1. Fix `mapMiGraphXType()` — change type from `migraphx::shape::type_t` to `migraphx_shape_datatype_t`, fix enum value names
2. Fix `shape.lens()` → `shape.lengths()` (2 occurrences)
3. Fix `buildContracts()` — change parameter type from `unordered_map` to `migraphx::program_parameter_shapes`, iterate with `.names()` and `operator[]`
4. Fix `get_parameter_shapes().at()` → `get_parameter_shapes()[name.c_str()]`
5. Fix `argument{}` brace init → `argument()` paren init
6. Remove `mp.prog.finish()` — API doesn't have it
7. Fix `hipSetDevice` return value (cast to void or check)
8. Fix `MIGRAPHX_VERSION` → construct from `_MAJOR/_MINOR/_PATCH`
9. Remove unused `pastLastAi` variable
10. Remove unused `encodeWithAiUpscale()` and `firstAiUpscaleIndex()` functions
11. Fix doubled `}  // namespace ave` in ncnn_vulkan_backend.cpp

### Step 2: Rebuild

```bash
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DAVE_HAVE_NCNN=ON \
  -DAVE_HAVE_MIGRAPHX=ON \
  -DAVE_HAVE_HIP=ON \
  -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build . -j$(nproc)
```

### Step 3: Run tests

```bash
ctest --test-dir build --output-on-failure
```

### Step 4: Verify with a real video

```bash
# List backends — should show MiGraphX and/or NCNN as available
./build/ave --list-backends

# Test with a video
./build/ave --input test.mp4 --output out.mp4 --backend auto --stage denoise
```

Look for log lines like:
- `[backend] MiGraphX initialised on device 0`
- `[migraphx] processFrameDir: model='...' scale=1 frames=N stage=Denoise`
- `[ai-pipeline] Denoise → AI complete via MiGraphX`

If those appear, AI inference is actually running.

### Step 5: Ensure models are available

For MiGraphX inference to work on non-upscale stages, the user needs compiled `.mxr` models. Use the GUI Model Manager or CLI to download and compile a model:

```bash
# Example: download and compile a denoise model
./build/ave_gui  # Use Model Manager tab
```

If no compiled model exists for a stage, the pipeline gracefully falls back to FFmpeg filters (produces a log like `deferring to FFmpeg`).

---

## 8. Potential Pitfalls

1. **MiGraphX compiled models have fixed input shapes.** A `.mxr` compiled for 720×1280 won't work on 1080×1920 frames. Consider adding tiling support or auto-recompilation in the future.

2. **The NCNN include path**: Fixed to `#include <net.h>` (not `<ncnn/ncnn/net.h>`) because the ncnn CMake config sets the include directory to `/usr/include/ncnn`.

3. **`-Werror` is not enabled**, but `-Wconversion -Wsign-conversion` are on. Some warnings may slip through that are effectively harmless but noisy.

4. **`encodeWithAiUpscale()` is now dead code** — it was superseded by `encodeWithAiProcessing()` which handles both upscale (via realesrgan) and all other AI stages (via `processFrameDir()`). Remove it cleanly.

5. **Thread safety**: `processFrameDir()` locks `impl_->mtx` for model loading and per-frame inference. This is correct but makes inference single-threaded. Future work could batch frames or pipeline GPU work.

6. **The `program_parameter_shapes` iteration pattern** in the installed MiGraphX uses `names()` returning `vector<const char*>` and `operator[]` returning `shape` by value. The code must NOT hold references to these transient objects.

---

## 9. File Reference

### Files that need error fixes (see Section 4):
- `src/backends/migraphx_backend.cpp` — 7 errors
- `src/observability.cpp` — 1 error
- `src/ffmpeg_runner.cpp` — 3 unused warnings (may be errors with strict flags)
- `src/backends/ncnn_vulkan_backend.cpp` — 1 doubled namespace brace

### Files that are complete and correct:
- `include/ave/frame_io.hpp` ✓
- `src/frame_io.cpp` ✓
- `include/ave/backend.hpp` ✓
- `include/ave/stage.hpp` ✓
- `include/ave/backends/migraphx_backend.hpp` ✓
- `include/ave/backends/ncnn_vulkan_backend.hpp` ✓
- `include/ave/ffmpeg_runner.hpp` ✓
- `src/video_processor.cpp` ✓
- `CMakeLists.txt` ✓

### Key reference documents:
- `docs/GOLD_STANDARD_FOR_IMPLEMENTATION.md` — 808-line spec document
- `docs/FEATURE_PARITY_MATRIX.md` — feature coverage matrix
- `docs/PARITY_PLAN.md` — implementation plan
- `AGENTS.md` — project conventions and build instructions
