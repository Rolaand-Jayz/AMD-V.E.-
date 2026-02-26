# MiGraphX AI Video Enhancer — Debugging & Validation Playbook

## Purpose

This document isolates the **debugging, diagnostics, and validation** portions of the Linux C++ AMD (ROCm + MiGraphX + Vulkan + FFmpeg) research/spec work into a single Markdown file.

It is designed for a project that is already near alpha and needs a **systematic way to verify wiring correctness**, isolate faults, and fix issues without guessing.

---

## 1) Core debugging philosophy

The biggest trap in this stack is that multiple failure classes can look the same:

- **Interop sync bug** (Vulkan ↔ HIP) can look like random corruption
- **Stride/plane mapping bug** can look like model artifacts
- **Color pipeline bug** can look like “bad AI output”
- **Timestamp bug** can look like interpolation issues
- **Precision/quantization bug** can look like banding or shimmer
- **ROCm/MiGraphX version mismatch** can look like app instability

### The solution: use an execution ladder

Always debug using these three modes in order:

1. **Mode A — CPU Reference Path**
   - CPU decode → CPU preprocess → CPU inference baseline (ONNX Runtime CPU) → CPU postprocess → CPU encode
   - Goal: prove model + preprocessing/postprocessing correctness independent of GPU plumbing

2. **Mode B — GPU Compute without Interop**
   - CPU decode → GPU preprocess/postprocess (Vulkan) + GPU inference (MiGraphX/HIP) with **explicit copies**
   - Goal: prove Vulkan + MiGraphX correctness without Vulkan↔HIP shared-memory complexity

3. **Mode C — Interop Fast Path**
   - Enable Vulkan↔HIP external memory + external semaphore path
   - Goal: validate performance path after correctness is already proven

If Mode A works and Mode C fails, the problem is almost never the model.

---

## 2) Observability requirements (must-have)

Debugging this stack without structured observability is pain with extra steps.

### 2.1 Structured log fields (every stage)

Every diagnostic log entry should include:

- `clip_id`
- `frame_id`
- `tile_id` (if tiled)
- `stage` (decode / preprocess / infer / postprocess / encode)
- `pts`
- `dts` (if available)
- `time_base`
- `pix_fmt`
- `color_range`
- `color_primaries`
- `color_trc`
- `colorspace`
- `gpu_id`
- `rocm_version`
- `migraphx_version`
- `driver_version`
- `mesa_version` (Linux)
- `model_hash`
- `opset`
- `precision_profile` (FP32/FP16/INT8)
- `tile_profile` (tile size / overlap / pad)
- `interop_enabled` (true/false)

### 2.2 Diagnostic modes (runtime toggles)

Implement these as real flags, not “we’ll add them later”:

- `slow_safe`
  - Forces hard sync between GPU stages (`vkQueueWaitIdle`, HIP sync)
  - If corruption disappears in this mode, it’s likely a sync bug

- `cpu_reference`
  - Runs ONNX Runtime CPU baseline for the same tensors/frames
  - Used to confirm whether the model path is correct

- `tensor_dump`
  - Dumps model input/output tensors for selected frames/tiles

- `frame_dump`
  - Dumps intermediate visual outputs (decoded, preprocessed, postprocessed, encoded roundtrip)

- `sync_validate`
  - Enables Vulkan validation layers and stricter sync assertions

- `migraphx_trace`
  - Enables MiGraphX parser/pass/quantization tracing

- `no_ai`
  - Disables inference and performs media-only path
  - Critical for separating media/color bugs from model bugs

- `no_interop`
  - Forces explicit copies instead of Vulkan↔HIP sharing

---

## 3) Systematic triage decision tree

Use this flow every time. Don’t freestyle.

## 3.1 Start: what kind of failure is it?

- **Crash**
- **Hang / deadlock**
- **Visual defect**
- **Performance regression**

---

### 3.2 Visual defect triage

#### Step V1 — Re-run with `slow_safe` + `frame_dump`
- If defect disappears:
  - Likely **sync / interop** problem
  - Jump to [Interop Debugging](#6-interop-debugging-deep-probes)

- If defect remains:
  - Continue

#### Step V2 — Run `cpu_reference` on the same frame(s)
- If CPU reference looks correct but GPU output is wrong:
  - Likely **MiGraphX lowering / precision / GPU binding** issue
  - Jump to [Model & Precision Debugging](#7-model--precision-debugging)

- If CPU reference is also wrong:
  - Likely **preprocess / color / media contract** issue
  - Jump to [Color & Media Debugging](#5-color--media-debugging)

#### Step V3 — Run `no_ai`
- If output is still wrong:
  - Not the model
  - It’s **media pipeline** (decode/encode/timestamps) or **color conversion**

- If output becomes correct:
  - Model path issue (preprocess, inference, postprocess, precision, tiling)

---

### 3.3 Hang / deadlock triage

1. Enable `sync_validate`
2. Log semaphore timeline values for every stage transition
3. Add hard debug timeouts around:
   - frame dequeue
   - GPU wait
   - encode enqueue
4. Re-run with `slow_safe`
   - If hang disappears → ordering/sync bug
5. Disable interop (`no_interop`)
   - If hang disappears → Vulkan↔HIP interop path is suspect

---

### 3.4 Crash triage

1. Capture:
   - stack trace
   - loaded shared libraries
   - runtime manifest (ROCm/MiGraphX/GPU/driver)
   - last successful `frame_id`, `tile_id`, `stage`

2. Re-run:
   - `slow_safe`
   - `no_interop`
   - `no_ai`
   - `cpu_reference`

3. Categorize crash:
   - **Startup crash** → packaging / ABI / missing library / version skew
   - **Compile-time model crash** → ONNX import / unsupported op / MiGraphX compile issue
   - **Runtime frame crash** → buffer ownership / shape mismatch / sync / OOM

---

### 3.5 Performance regression triage

1. Confirm no hidden fallback occurred:
   - CPU fallback
   - no fusion / wrong precision
   - interop disabled unexpectedly

2. Compare against previous baseline:
   - resolution
   - model
   - tile profile
   - precision profile
   - ROCm/MiGraphX version

3. Profile:
   - CPU stage timing
   - Vulkan timestamps
   - HIP/MiGraphX kernels (rocprof / ROCProfiler)
   - MiGraphX pass timing and trace

---

## 4) Step-by-step audit procedure for an existing implementation

This is the “walk every wire” checklist.

### 4.1 Audit Step A — Inventory & version capture

Capture and store in a run manifest:

- GPU model + architecture
- Kernel version
- ROCm version
- MiGraphX version
- HIP runtime version
- Mesa/driver versions
- FFmpeg build/version
- Vulkan instance/device extensions
- Model hash + ONNX opset
- Precision profile
- Tile profile

If this data is not captured, reproducibility is already broken.

---

### 4.2 Audit Step B — Verify stage toggles exist

Confirm you can independently enable/disable:

- preprocess
- inference
- postprocess
- tiling
- interop
- hw decode
- hw encode
- precision mode (FP32/FP16/INT8)

If not, add the toggles before debugging further.

---

### 4.3 Audit Step C — Run the execution ladder

Run a fixed test clip (short, deterministic) through:

1. **Mode A** (CPU reference)
2. **Mode B** (GPU, no interop)
3. **Mode C** (interop fast path)

For each run:
- record first failing frame
- record first failing stage
- dump artifacts (see Section 8)
- store metrics (SSIM/PSNR/VMAF if available)

---

### 4.4 Audit Step D — Subsystem-by-subsystem verification

#### D1) MediaIO (FFmpeg)
- Decode-only test:
  - dump frame metadata and representative frames
- Encode-only test:
  - feed known-good frames and verify output timeline/container
- No-AI passthrough test:
  - decode → encode only, preserve timestamps and metadata as much as possible

#### D2) Color pipeline
- Conversion-only test:
  - compare against FFmpeg reference conversion (zscale/tonemap path for HDR cases)
- Run grayscale ramp / color bars / skin patch tests

#### D3) MiGraphX inference
- Use a standalone MiGraphX harness:
  - parse ONNX
  - compile for GPU
  - run fixed-shape inputs
  - dump outputs
- Enable MiGraphX traces:
  - parser trace
  - pass timing
  - quantization trace (if relevant)

#### D4) Vulkan↔HIP interop
- Run standalone loopback (no FFmpeg, no MiGraphX)
- Verify:
  - exported memory handle import
  - exported semaphore import
  - deterministic pattern roundtrip
- Run with Vulkan validation layers on

---

## 5) Color & Media Debugging

This is where “the AI looks bad” usually turns into “the pipeline is lying.”

## 5.1 Common symptom → likely cause

- **Washed out / gray blacks**
  - full vs limited range mismatch
  - wrong transfer function
  - incorrect YUV↔RGB conversion coefficients

- **Too saturated / weird skin tones**
  - wrong colorspace / primaries mapping
  - channel order swap (RGB/BGR)

- **Banding**
  - low-precision conversion
  - missing dithering
  - quantization issue (not always model-related)

- **Chroma weirdness / color fringes**
  - 4:2:0 plane offset/stride bug
  - chroma siting mismatch

---

## 5.2 Color debugging procedure

1. Run `no_ai`
   - If still wrong, the AI is innocent

2. Dump and log:
   - `pix_fmt`
   - `color_range`
   - `colorspace`
   - `color_primaries`
   - `color_trc`

3. Run synthetic sentinel patterns through the pipeline:
   - grayscale ramp
   - RGB primaries
   - color bars
   - chroma alignment pattern
   - skin-tone patch

4. Compare against reference conversion path
   - FFmpeg (zscale/tonemap path where applicable)

5. Add a debug overlay (dev-only) onto frames:
   - show chosen color policy on each frame
   - makes silent wrong assumptions visible immediately

---

## 5.3 Timestamp / media integrity checks

For interpolation and general correctness, you need frame timeline truth.

Validate for every frame:
- PTS monotonicity (or expected pattern if B-frames/reorder are involved)
- output frame count expectations
- VFR preservation policy (document it)
- interpolation-generated timestamps are deterministic and explicit

When debugging interpolation:
- test with scene cuts
- ensure scene-cut reset behavior is logged
- verify no blended-cut ghost frames unless explicitly desired

---

## 6) Interop Debugging (Deep Probes)

This is the critical plumbing zone: Vulkan ↔ HIP external memory/semaphores.

## 6.1 Interop bug signatures

### Likely sync bug
- nondeterministic corruption
- changes with system load
- disappears with `slow_safe`

### Likely mapping/stride bug
- deterministic corruption
- diagonal tearing / consistent offsets / channel shifts
- persists even with hard sync

---

## 6.2 Mandatory interop probes (implement all six)

### Probe P1 — Sync Hard Fence
Force synchronization after every stage:
- `vkQueueWaitIdle`
- HIP stream/device sync

**Use:** If issue disappears, it’s a sync problem.

---

### Probe P2 — Single-API Ownership
Temporarily split the path:
- Vulkan preprocess/postprocess only
- MiGraphX inference with explicit copies
- no interop

**Use:** Proves whether the interop bridge is the issue.

---

### Probe P3 — Pattern Fill
Fill buffers with deterministic patterns + sentinel borders:
- gradient
- checkerboard
- unique edge borders

Verify after each stage.

**Use:** Detects overwrites, wrong offsets, partial writes.

---

### Probe P4 — FD Lifecycle Audit
Log:
- exported memory FD
- imported FD
- exported semaphore FD
- ownership / close responsibility

**Rule:** one handle, one owner, one close.

**Use:** Catches stale/duplicate/invalid FD lifecycle bugs.

---

### Probe P5 — Stride/Offset Snapshot
Capture and log:
- plane offsets
- row pitch / stride
- image extent
- format
- any layout metadata used in pointer math

**Use:** Catches silent plane math errors.

---

### Probe P6 — Per-stage CRC + Image Dump
For selected frames/tiles:
- CRC/hash raw bytes after every stage
- dump visual intermediate images

**Use:** Pinpoints the first stage where corruption appears.

---

## 6.3 Interop loopback test (standalone)

Build a small standalone executable (no FFmpeg, no MiGraphX) that:

1. Creates exportable Vulkan memory
2. Exports memory handle
3. Imports into HIP
4. Writes a pattern in Vulkan and verifies in HIP
5. Writes a pattern in HIP and verifies in Vulkan
6. Synchronizes using external semaphores
7. Repeats in a loop

This becomes your “shared-memory truth test.”  
If this fails, stop debugging the main app and fix interop first.

---

## 7) Model & Precision Debugging

## 7.1 MiGraphX model bring-up harness (required)

Before touching the full video pipeline, every model should pass a standalone harness:

1. Load ONNX
2. Validate fixed input shapes
3. Import into MiGraphX
4. Compile for target GPU
5. Run deterministic test input(s)
6. Dump output tensors
7. Save compile report + traces

This avoids guessing whether the problem is the model or the media path.

---

## 7.2 Cross-validation ladder for model correctness

For a fixed tensor input, compare outputs across:

1. **PyTorch eager** (research lane only, if applicable)
2. **Torch-MiGraphX** (research lane only)
3. **ONNX Runtime CPU** (reference baseline)
4. **MiGraphX C++ harness**
5. **Integrated video pipeline path**

If #3 and #4 differ:
- likely MiGraphX import/lowering/precision issue  
If #4 and #5 differ:
- likely preprocess/postprocess/binding issue

---

## 7.3 Precision debug (FP32 / FP16 / INT8)

### Precision bisect mode
Run the same frames/tiles through:
- FP32
- FP16
- INT8 (if enabled)

Dump:
- input tensors
- output tensors
- per-layer or per-stage metrics (where available)
- image-space diffs

### Common precision symptoms

- **FP16 instability**
  - subtle texture shifts
  - edge ringing changes
  - mild flicker increase

- **INT8 issues**
  - banding
  - crushed shadow detail
  - aggressive texture loss
  - temporal shimmer

### Rule
Do not ship INT8 for a model until it passes:
- objective thresholds (SSIM/PSNR/VMAF bands)
- visual checks on dark scenes, skin, foliage, text
- temporal stability checks

---

## 7.4 MiGraphX-specific debugging aids

Turn on MiGraphX tracing when needed:
- ONNX parser tracing
- pass tracing
- pass timing
- quantization tracing (INT8 path)
- verification/diff dump options

Capture these into your run artifacts so failures are explainable later.

---

## 8) Exact artifacts to dump (per debug run)

For selected `frame_id` values (recommend: first 3 frames + one later frame), dump the following.

## 8.1 Media artifacts

### `frame_meta.json`
Include:
- frame_id
- pts/dts
- duration
- time_base
- pix_fmt
- width/height
- keyframe flag
- interlaced flags
- color metadata

---

## 8.2 Visual artifacts

Dump images for each stage (PNG or EXR where needed):

- `decoded.png`
- `preprocess_rgb.png`
- `post_infer_raw.png` (if meaningful)
- `postprocess.png`
- `encoded_roundtrip.png`

For tiled runs, also dump:
- `tile_overlay.png` (tile boundaries + overlap visualization)

---

## 8.3 Tensor artifacts

Per selected frame/tile:

- `input_tensor.bin`
- `input_tensor_header.json`
  - shape, dtype, strides, normalization, channel order
- `output_tensor.bin`
- `output_tensor_header.json`

For tiled mode:
- `tile_manifest.json`
  - tile coords
  - overlap
  - padding policy
  - stitch blend mode

---

## 8.4 Checksums & stats

Per stage:
- raw buffer checksum (CRC32 or xxHash)
- per-channel histogram
- min / max / mean / std

These catch bad ranges, clipping, and silent corruption fast.

---

## 8.5 Timing traces

Collect:
- CPU stage timings
- Vulkan timestamp queries
- HIP/MiGraphX traces (rocprof / ROCProfiler)
- MiGraphX compile/pass timing (when compiling)

---

## 9) Minimal repro protocol (for stubborn bugs)

When something is hard to reproduce, make it smaller.

## 9.1 Media repro
Create a tiny clip (1–3 seconds) that preserves the bug:
- remux if possible (avoid unnecessary re-encode)
- preserve timestamps and original metadata
- document exact frame where failure begins

## 9.2 Model repro
Extract just the failing tensor(s):
- save input tensor
- save expected vs actual output tensor
- include model hash + opset + precision profile

## 9.3 Interop repro
Use the standalone Vulkan↔HIP loopback test:
- no FFmpeg
- no MiGraphX
- same GPU/device selection path

The goal is to produce one-file, one-command repros for each bug class.

---

## 10) Validation strategy (what “fixed” means)

A bug is not “fixed” when it looks okay once. It’s fixed when it passes the right checks.

## 10.1 Correctness checkpoints

### V0 — Media no-op
- decode → encode (no AI) is timeline-correct and visually sane

### V1 — Conversion roundtrip
- preprocess/postprocess path matches reference conversion within tolerance

### V2 — Model correctness
- MiGraphX output matches ONNX Runtime CPU baseline within defined error envelope

### V3 — Full pipeline quality
- clip-level quality metrics within expected band
- no new visible artifacts
- no temporal instability regressions

---

## 10.2 Quality metrics (non-Python friendly)

Use C/C++-friendly tools/libs where possible:

- **PSNR**
- **SSIM**
- **VMAF** (via libvmaf / FFmpeg integration)

For video enhancement, also add temporal checks:
- frame-diff energy trends
- flicker heatmap / threshold alerts (custom is fine)

---

## 11) Operational hardening & fallback behavior

Your app should fail usefully, not dramatically.

### 11.1 Fallbacks to implement

- **Interop failure**
  - fallback to explicit-copy path (Mode B)

- **MiGraphX compile/import failure**
  - fail model load with actionable error
  - report unsupported op / shape issue if possible

- **OOM**
  - retry with smaller tiles
  - optionally reduce queue depth
  - optionally switch precision profile (configurable)

- **HW decode/encode unavailable**
  - fallback to software path (with warning)

### 11.2 Logging rule
Every fallback must emit a structured event.  
Silent fallback = silent performance regression = future pain.

---

## 12) Recommended debugging checklist (practical)

Use this when things go sideways.

- [ ] Capture runtime version manifest
- [ ] Reproduce on a short deterministic clip
- [ ] Run `no_ai`
- [ ] Run `cpu_reference`
- [ ] Run `no_interop`
- [ ] Run `slow_safe`
- [ ] Dump frame/tensor artifacts
- [ ] Compare MiGraphX vs ONNX Runtime CPU outputs
- [ ] Check color metadata + conversion path
- [ ] Check tile overlap/padding/stitch policy
- [ ] Enable MiGraphX traces
- [ ] Enable Vulkan validation layers
- [ ] Run interop loopback test (if interop path suspected)
- [ ] Re-run and confirm fix in Mode A → B → C

---

## 13) Official docs (quick links)

### AMD / ROCm / MiGraphX / HIP
- ROCm docs: https://rocm.docs.amd.com/
- ROCm compatibility matrix: https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html
- MiGraphX docs: https://rocm.docs.amd.com/projects/AMDMIGraphX/
- MiGraphX C++ API (parse/save/load): https://rocm.docs.amd.com/projects/AMDMIGraphX/en/docs-6.2.0/reference/cpp.html
- MiGraphX ONNX operator support: https://rocm.docs.amd.com/projects/AMDMIGraphX/en/develop/dev/onnx_operators.html
- MiGraphX env vars / tracing: https://rocm.docs.amd.com/projects/AMDMIGraphX/en/latest/reference/MIGraphX-dev-env-vars.html
- HIP external interop: https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/external_interop.html
- ROCProfiler / rocprof: https://rocm.docs.amd.com/projects/rocprofiler/en/latest/

### Vulkan
- Vulkan spec: https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html
- VK_EXT_debug_utils: https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_debug_utils.html
- VK_EXT_external_memory_dma_buf: https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_external_memory_dma_buf.html
- VK_KHR_external_semaphore_fd: https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_external_semaphore_fd.html
- Vulkan validation overview: https://docs.vulkan.org/guide/latest/validation_overview.html

### FFmpeg / Media / Quality
- FFmpeg docs: https://ffmpeg.org/documentation.html
- FFmpeg hwcontext.h (doxygen): https://ffmpeg.org/doxygen/6.0/hwcontext_8h.html
- FFmpeg Vulkan hwcontext source: https://www.ffmpeg.org/doxygen/trunk/hwcontext__vulkan_8c_source.html
- FFmpeg filters (libvmaf, etc.): https://ffmpeg.org/ffmpeg-filters.html
- libvmaf: https://github.com/Netflix/vmaf

### ONNX / ORT
- ONNX IR: https://onnx.ai/onnx/repo-docs/IR.html
- ONNX versioning: https://onnx.ai/onnx/repo-docs/Versioning.html
- ONNX operators: https://onnx.ai/onnx/operators/
- ONNX Runtime MIGraphX EP: https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html

---

## 14) Final note

If you keep the execution ladder, the stage toggles, and the artifact dumps, you’ll stop chasing ghosts and start finding the exact wire that’s loose.

That’s the whole game here.
