# Proper Implementation Technique and Gold Standard Specification for a Linux C++ AI Video Enhancer Using ROCm, MiGraphX, and Vulkan

## Detailed Technical Report

**Executive Summary**

1) **What “proper technique” looks like for this stack**

A correct, reliable implementation treats the system as two tightly-coupled GPU subsystems with an explicit contract boundary:

- A **Vulkan subsystem** that owns video-frame residency, format conversion, scaling/normalization, postprocessing, and (optionally) FFmpeg-driven Vulkan hardware frames.
- A **MiGraphX/ROCm subsystem** that owns model compilation and inference execution lifecycle.
- A **cross-API interop bridge** that provides *explicit*, *auditable* **GPU memory sharing and synchronization** between Vulkan and HIP (MiGraphX uses HIP under ROCm) using external memory + external semaphore primitives, instead of implicit copies or undefined cache behavior. HIP explicitly supports importing memory/semaphores created by Vulkan and warns that buffer access must be synchronized between APIs. citeturn11view0

In practice, the gold-standard approach for correctness and debuggability is:

- **Keep video frames as Vulkan images**, but **share inference tensors as Vulkan buffers**.
- Convert image → tensor buffer in Vulkan compute, then **export the buffer’s VkDeviceMemory as an FD** and **import/map it into HIP** for MiGraphX input.
- Use **external semaphores** to guarantee cross-API ordering and memory visibility.
- Treat MiGraphX model compilation as a *separately testable artifact pipeline* (parse → compile → cache → load → execute), with deterministic inputs and explicit cache keys.

2) **Top causes of MiGraphX integration failures (most common in real projects)**

The dominant failure cluster is **not** “MiGraphX is broken”; it’s *contract mismatch*:

- **ONNX compatibility mismatch**: MiGraphX supports ONNX operators up to **Opset 19** (current ONNX exporters commonly default to higher opsets; ONNX 1.21.0 reports opset 26). citeturn8view0turn14search9  
- **Unsupported ONNX operator semantics, data types, or dynamic-shape requirements**: MiGraphX exposes limitations *per operator* (e.g., “dynamic shape is not supported” for many ops). citeturn8view0
- **Layout/stride misunderstanding** (NCHW vs NHWC, and “channels-last” style transforms): MiGraphX has an environment variable to enable/force NHWC layout passes, which can silently change tensor interpretation if your pipeline assumes NCHW. citeturn4view1turn5search11
- **GPU/CPU residency mismatch** driven by compile flags like **offload_copy** (or equivalent driver flag): MiGraphX documents that GPU compilation normally allocates buffers on GPU, but `offload_copy=true` makes buffers reside on CPU even when compiling for GPU. citeturn18search21  
  A known failure mode occurs when a compiled model expects offload-copy semantics but runtime does not enable them (driver throws “GPU arguments or memory objects” style errors). citeturn18search13
- **Cross-API synchronization bugs** (Vulkan↔HIP) that manifest as intermittent corruption/stalls: Vulkan requires correct stage/access scopes for barriers, and synchronization2 formalizes this via Vk*Barrier2 structures and vkCmdPipelineBarrier2. citeturn10search18turn10search2turn10search5turn10search9  
  HIP’s Vulkan interop example explicitly notes that cross-API access must be synchronized (queue syncs or semaphores). citeturn11view0
- **Serialized artifact instability**: MiGraphX compiled artifacts (“MXR” commonly used in practice) depend on GPU and compile/runtime parameters; moving them across machines/GPUs without correct cache keying breaks. citeturn17search21
- **Binary/ABI conflicts** in serialization dependencies (e.g., msgpack symbol conflicts) can break `migraphx::load()` even when the compiled file is valid. citeturn17search5

3) **Highest-priority corrections (stabilize before alpha)**

- **Enforce a single, explicit Tensor Contract** (layout, dtype, shape, strides, color pipeline) and assert it at every boundary (decode→preprocess, preprocess→inference, inference→postprocess, postprocess→encode).
- **Pin and log version sets** (ROCm + MiGraphX + kernel/driver + Vulkan loader/driver + FFmpeg) on every run, and key compiled model cache by that tuple plus GPU architecture.
- **Make cross-API interop first-class**: use Vulkan external memory FD + Vulkan external semaphore FD + HIP external memory/semaphore import; treat any fallback copy path as an explicitly logged “degraded mode.” citeturn10search4turn21search2turn11view0
- **Operationalize MiGraphX diagnostics** via MIGRAPHX_* env vars for compile/MLIR/HIPRTC tracing and GPU debug dumps, plus MiGraphX driver “verify” workflows in CI. citeturn5search11turn18search12
- **Separate “model correctness” debugging from “pipeline correctness” debugging** with small, deterministic harnesses: one for MiGraphX inference only, one for Vulkan preprocess/postprocess only, then end-to-end.

---

**Evidence-Based Findings (facts vs inference, with version sensitivity)**

2) **Verified facts (directly supported by primary sources)**

- **MiGraphX ONNX support envelope**
  - MiGraphX supports ONNX operators up to **Opset 19**. citeturn8view0turn14search8
  - MiGraphX enumerates supported ONNX data types including **FLOAT8**, **FLOAT16**, **FLOAT32**, and **INT8** (and others), and notes FP8 support is limited to **E4M3FNUZ**. citeturn8view0turn15search16
  - The operator support matrix includes explicit limitations such as “dynamic shape is not supported” for various operators, and other per-op constraints. citeturn8view0
- **MiGraphX execution lifecycle surfaces needed for C++ integration**
  - `program::compile(...)`, `program::eval(...)`, `program::finish()`, `program::get_parameter_shapes()`, and `program::get_output_shapes()` are part of MiGraphX’s program API surface. citeturn9search0
  - Compilation targets provide utilities for resource handling such as `target::allocate(shape)` plus `copy_to` / `copy_from`. citeturn9search1
- **MiGraphX quantization entry points (C++ internal APIs)**
  - MiGraphX provides `quantize_fp16`, `quantize_bf16`, and `quantize_int8` with calibration data and a default set of ops for int8 quantization (“dot”, “convolution”). citeturn20view0
- **MiGraphX memory residency behavior for GPU compilation**
  - MiGraphX documents that compiling for GPU normally allocates buffers on GPU; setting `offload_copy=true` while compiling for GPU places buffers on CPU. citeturn18search21
- **MiGraphX environment-variable knobs exist specifically for tracing and debugging compilation**
  - Environment variables include compile tracing (`MIGRAPHX_TRACE_COMPILE`, `MIGRAPHX_TRACE_PASSES`, `MIGRAPHX_TIME_PASSES`), GPU JIT debug (`MIGRAPHX_TRACE_HIPRTC`, `MIGRAPHX_GPU_DUMP_SRC`, `MIGRAPHX_GPU_DUMP_ASM`, `MIGRAPHX_GPU_DEBUG`), MLIR toggles (`MIGRAPHX_DISABLE_MLIR`, `MIGRAPHX_TRACE_MLIR`), and layout control (`MIGRAPHX_ENABLE_NHWC`). citeturn5search11turn5search1turn4view1
- **Vulkan synchronization correctness requirements are explicit and non-optional**
  - Vulkan synchronization depends on correctly setting source/destination stage masks and access masks; those scopes define what is ordered and what memory becomes visible. citeturn10search5
  - `VK_KHR_synchronization2` (promoted to Vulkan 1.3) modernizes barriers and submission and is explicitly intended to improve and simplify synchronization. citeturn10search18
  - `vkCmdPipelineBarrier2` and `VkImageMemoryBarrier2` define the synchronization2 pipeline barrier mechanisms and required fields (stage masks, access masks, layouts, queue-family transfers). citeturn10search2turn10search9
- **Cross-API Vulkan↔HIP interop is officially supported in HIP**
  - HIP’s External Resource Interoperability documentation states resources created by Vulkan can be imported and used in HIP, including an official HIP–Vulkan interoperation example demonstrating importing **VkDeviceMemory** via FD and importing Vulkan semaphores via FD, then synchronizing with external semaphores. citeturn11view0
- **Vulkan external-memory and external-semaphore mechanisms for FD handles exist**
  - `VK_KHR_external_memory_fd` enables export/import of Vulkan memory objects as POSIX file descriptors. citeturn10search4
  - `VK_EXT_external_memory_dma_buf` enables exporting VkDeviceMemory as dma_buf and importing dma_buf into VkDeviceMemory; it is sufficient (with `VK_KHR_external_memory_fd`) to bind VkBuffer to dma_buf. citeturn22search20turn10search0
  - `VK_KHR_external_semaphore_fd` enables exporting/importing semaphores as POSIX file descriptors. citeturn21search2turn21search26
  - In synchronization2 submission, `VkSemaphoreSubmitInfo::stageMask` limits the synchronization scope for semaphore wait/signal operations. citeturn21search7turn21search11
- **FFmpeg Vulkan hwcontext facts relevant to correct interop**
  - For Vulkan hardware frames, FFmpeg requires that custom pools return `AVBufferRef` objects whose data pointer is an `AVVkFrame`. citeturn12view0
  - `av_vkfmt_from_pixfmt()` returns optimal per-plane Vulkan formats for a given FFmpeg pixel format and returns NULL for unsupported formats. citeturn12view1
  - FFmpeg provides hardware-frame transfer/mapping primitives (`av_hwframe_transfer_data`, `av_hwframe_map`) and derived frames contexts (`av_hwframe_ctx_create_derived`), with explicit notes about indirect mappings when direct interop is missing (e.g., VAAPI→(DRM)→OpenCL/Vulkan). citeturn13view0
- **ROCm platform compatibility is version- and OS-specific**
  - The ROCm compatibility matrix is centrally maintained and explicitly enumerates GPU architecture and supported OS/kernel combinations per ROCm release. citeturn7search0turn7search1
- **Model caching portability caution**
  - Practical tooling notes that MiGraphX compiled “MXR” artifacts vary depending on GPU and parameters (e.g., batch size), and moving MXR between servers requires ensuring it was compiled for the target hardware. citeturn17search21

3) **Strong inferences (high confidence, but still must be validated against your codebase)**

- If MiGraphX integration is “not working reliably,” and the project also uses Vulkan for frame processing, the *highest-likelihood* root cause is **cross-API memory/synchronization ambiguity** (implicit copies, missing external semaphores, or mismatched device selection) rather than MiGraphX compiler instability. This inference is strongly supported by the fact that HIP’s official interop guidance explicitly warns that shared memory access must be synchronized between APIs. citeturn11view0
- A large fraction of “works on machine A but fails on machine B” issues will trace to **cache key incompleteness** (GPU gfx target, ROCm/MiGraphX version, compile options like offload_copy/fast_math/exhaustive tuning, and environment-variable toggles). MiGraphX users explicitly observe that offload_copy differences are not encoded into compiled artifacts (driver issue) and that MXR differs by GPU/parameters. citeturn18search13turn17search21

4) **Unknowns requiring empirical validation (do not guess; measure)**

- Which Vulkan driver path you target in production (Mesa RADV vs AMDVLK) and whether the specific external-memory + external-semaphore handle types you need are supported on your deployment GPU/OS stack must be determined by runtime extension queries (Vulkan) and a real interop smoke test (HIP import). Vulkan defines the extensions; support is driver-dependent. citeturn10search4turn21search2turn11view0
- Whether your model(s) rely on operators beyond MiGraphX’s supported set (or rely on dynamic shapes in operators where MiGraphX lists “dynamic shape is not supported”) must be determined by **operator inventory** per model. citeturn8view0turn14search8
- Whether your inference is intended to use FP8/INT8 in production must be decided with accuracy/regression thresholds; low precision support varies by architecture (e.g., FP8 matrix-core support appears starting at later architectures). citeturn19search18turn15search16

---

**Architecture Blueprint (conceptual, end-to-end)**

3) **Pipeline stages and module boundaries**

A robust Linux C++ enhancer typically splits into four “hard boundaries” with explicit contracts:

- **Decode / ingest**
  - Responsibilities: demux/decode, timestamps, color metadata ingestion, produce frames in either CPU memory or GPU memory.
  - If FFmpeg is used, hardware-frame management depends on `AVHWDeviceContext` and `AVHWFramesContext` lifetimes. citeturn13view0
- **Preprocess (Vulkan)**
  - Responsibilities: colorspace conversion (NV12/P010/etc → model input), resizing, normalization, optional temporal buffering, pack into tensor layout.
  - Output: **VkBuffer** containing tensor payload for inference (preferred), plus sideband metadata (original colorimetry).
- **Inference (MiGraphX/ROCm)**
  - Responsibilities: parse/compile/load cached program, prepare parameter_map, execute `program::eval`, enforce shape/type invariants, and return output tensors. citeturn9search0
  - Input/Output memory: ideally GPU buffers imported via HIP external memory interop from Vulkan. citeturn11view0
- **Postprocess (Vulkan)**
  - Responsibilities: convert output tensor → display/encode format (RGB→YUV, tone mapping, denoise/sharpen), optional compositing, pack into Vulkan image(s).
- **Encode / egress**
  - Responsibilities: encode, mux, output file/stream.
  - If FFmpeg is used with Vulkan frames, pixel format mapping to VkFormat is constrained; fixed mapping helper exists and can return NULL for unsupported formats, so format negotiation must be explicit. citeturn12view1

4) **Where MiGraphX should sit and how it interacts with Vulkan/FFmpeg**

The simplest “correct wiring” is:

- FFmpeg provides decoded frames into a Vulkan-accessible representation (best case: Vulkan hwframes; otherwise CPU frames uploaded).
- Vulkan preprocess writes into a **Vulkan-allocated buffer** (VkBuffer + VkDeviceMemory configured for external export).
- HIP imports that **VkDeviceMemory** FD, maps it to a device pointer, wraps it in MiGraphX arguments, and runs inference.
- HIP signals a **HIP external semaphore** imported from a Vulkan-exported semaphore FD.
- Vulkan waits on that semaphore and runs postprocessing/encode.

HIP explicitly documents this flow and provides an import/mapping example using `VkMemoryGetFdInfoKHR` → `hipImportExternalMemory` → `hipExternalMemoryGetMappedBuffer`, and similarly for semaphores (`vkGetSemaphoreFdKHR` → `hipImportExternalSemaphore`). citeturn11view0turn21search2turn10search4

5) **Data ownership and synchronization boundaries**

- **Vulkan owns**: VkDevice/VkQueue, Vulkan allocations, VkCommandBuffers, VkImages/VkBuffers.  
- **HIP/MiGraphX owns**: hipStream(s), MiGraphX `program`, and parameter_map execution calls.
- **Handshake objects**: exported FDs for memory and semaphores (opaque FD or dma_buf FD), with explicit lifetime rules and closure.

Synchronization must be explicit in both APIs:

- Vulkan internal hazards: use `vkCmdPipelineBarrier2` and precise stage/access scopes; Vulkan defines how stage masks limit synchronization scope. citeturn10search2turn10search5turn10search9
- Vulkan↔HIP hazards: HIP explicitly warns shared buffers must be synchronized between APIs (queue syncs or semaphores). citeturn11view0
- In synchronization2 submission, `VkSemaphoreSubmitInfo::stageMask` limits ordering for semaphore wait/signal operations, so “wait at ALL_COMMANDS” is safe but heavy; “wait at COMPUTE_SHADER” is more correct if you only need compute writes visible. citeturn21search7turn21search11

---

**MiGraphX Proper Technique (load → compile → cache → execute)**

4) **Correct lifecycle and invariant checks**

A stable MiGraphX integration should treat program compilation as an artifact pipeline:

1. **Model ingestion**
   - Parse ONNX into a `program` (MiGraphX provides `parse_onnx` surfaces in its C++ API documentation set). citeturn16search23turn9search18  
   - Enforce ONNX opset policy: MiGraphX supports up to Opset 19, while ONNX toolchains may produce higher opsets; reject or down-convert models outside your allowed set. citeturn8view0turn14search9
   - If the ONNX model uses external data, remember ONNX external tensors are stored in external files in the same raw byte format as TensorProto raw_data; your loader pipeline must ensure those files are present and resolved consistently. citeturn16search2
2. **Static/dynamic shapes policy**
   - Inventory model inputs and determine if shapes are static or dynamic.
   - Treat MiGraphX operator-matrix limitations regarding dynamic shapes as a compile-time gate (not a runtime surprise). citeturn8view0
3. **Target selection and compilation**
   - Compile program for GPU target; program exposes `compile(...)` and target surfaces expose allocation/copy helpers. citeturn9search0turn9search1
   - Decide **offload_copy** strategy:
     - If using Vulkan↔HIP external memory buffers (GPU pointers), you generally want an execution path that consumes GPU-resident buffers.  
     - MiGraphX documents that `offload_copy=true` places buffers on CPU even when compiling for GPU. citeturn18search21  
     - Known reliability issues occur if a compiled artifact expects offload-copy semantics but runtime does not provide them. citeturn18search13
4. **Optional quantization**
   - FP16/BF16: `quantize_fp16` / `quantize_bf16` exist. citeturn20view0
   - INT8: `quantize_int8` requires calibration inputs as parameter_map entries and defaults to targeting {dot, convolution}. citeturn20view0
   - FP8: MiGraphX FP8 support is constrained (E4M3FNUZ); ensure model dtype and calibration semantics align (FP8 types have non-trivial conversion semantics vs other float8 variants). citeturn8view0turn15search16
5. **Execution**
   - To run: supply parameters as a `parameter_map` and call `program::eval(...)`. citeturn9search0turn9search10  
   - Ensure lifetime: command submission and device usage remain valid until operations complete; use `program::finish()` semantics as part of a “frame slot retirement” step. citeturn9search0

5) **Compiled model caching and invalidation**

Caching is essential near-alpha because compile latency can be large and nondeterminism hides bugs, but it must be *correctly keyed*:

- MiGraphX compiled artifacts used in practice (MXR) vary with GPU and load-time parameters; moving them across machines requires recompile or careful matching. citeturn17search21
- Offload-copy semantics create a known mismatch class where runtime must match compile-time expectations (driver issue suggests storing the flag). citeturn18search13

**Recommended cache key (minimum viable correctness)**

Cache key should include:

- Model identity: SHA-256 of ONNX graph bytes **plus external-data file hashes** (if any). citeturn16search2
- MiGraphX version + ROCm version (MiGraphX docs are published under ROCm releases; e.g., MiGraphX 2.15 docs are under ROCm Software 7.2.0). citeturn20view0turn23search27
- GPU architecture identifier (gfx target) and driver stack identity (at minimum, ROCm runtime + Vulkan driver version).
- Compile options: offload_copy, fast-math enable/disable, exhaustive tune enable/disable, precision/quantization mode.
- MiGraphX debug/behavior env vars that affect compilation (e.g., `MIGRAPHX_DISABLE_MLIR`, `MIGRAPHX_ENABLE_CK`, `MIGRAPHX_ENABLE_NHWC`). citeturn4view1turn5search11

**Serialization technique note**

MiGraphX exposes `load`/`save` in its documented API surface (the C++ reference shows file_options with `set_file_format`). citeturn17search0  
In the field, artifacts are commonly msgpack-based (MXR). citeturn17search21turn17search3  
Be aware of dependency interactions: msgpack symbol conflicts have been reported to break MiGraphX load paths in certain build environments. citeturn17search5

6) **Diagnostics and instrumentation strategy**

- Compile tracing: `MIGRAPHX_TRACE_COMPILE`, `MIGRAPHX_TRACE_PASSES`, `MIGRAPHX_TIME_PASSES` to capture pass-by-pass diffs and timing. citeturn5search11
- Kernel/JIT tracing: `MIGRAPHX_TRACE_HIPRTC`, `MIGRAPHX_GPU_DUMP_SRC`, `MIGRAPHX_GPU_DUMP_ASM`, `MIGRAPHX_DEBUG_SAVE_TEMP_DIR`, `MIGRAPHX_GPU_DEBUG`. citeturn5search11
- MLIR toggles: `MIGRAPHX_DISABLE_MLIR` and `MIGRAPHX_TRACE_MLIR` to isolate rocMLIR-related failures. citeturn5search1turn4view1
- Layout forcing: `MIGRAPHX_ENABLE_NHWC` to validate whether NHWC transforms are implicated in output correctness issues. citeturn5search11turn4view1
- Performance correlation: use ROC-TX markers/ranges (ROCTx) and rocprof’s `--roctx-trace` for end-to-end correlation. citeturn23search0turn23search8  
  MiGraphX driver has a `roctx` path specifically meant to integrate with rocprof for per-operator marker timing; an example combined command is documented. citeturn23search17

---

**Vulkan Proper Technique for AI Video Pipelines**

5) **Synchronization patterns (synchronization2-first)**

Treat Vulkan synchronization as a formal proof obligation:

- Vulkan defines that stage masks limit synchronization scope and access scope; insufficient masks create hazards even when code “usually works.” citeturn10search5
- Use `VK_KHR_synchronization2` / Vulkan 1.3 synchronization2 structures so the barrier data is self-contained and easier to audit. citeturn10search18

**Core patterns you should standardize**

- **Compute writes → compute reads (same queue)**
  - After preprocess compute writes into tensor buffer, record `vkCmdPipelineBarrier2` with a `VkBufferMemoryBarrier2` equivalent that transitions:
    - srcStage=COMPUTE_SHADER, srcAccess=SHADER_WRITE
    - dstStage=COMPUTE_SHADER, dstAccess=SHADER_READ
  - This is the minimum safe form for “preprocess produced tensor buffer; inference consumer (still Vulkan) can read.”
  - If the next consumer is **HIP/MiGraphX**, you use a semaphore boundary (see interop contract below) and still must ensure Vulkan finished writes before signaling. The scope of semaphore signal/wait in submit2 is controlled by `VkSemaphoreSubmitInfo::stageMask`. citeturn21search7turn21search11

- **Image layout transitions**
  - Use `VkImageMemoryBarrier2` and set oldLayout/newLayout explicitly; Vulkan describes this structure as a memory dependency limited to an image subresource range and includes fields for queue-family ownership transfer and layout transitions. citeturn10search9turn10search28

6) **Queue usage model**

For verifiability, prefer:

- One queue family supporting compute + transfer for the main pipeline.
- Avoid cross-queue ownership transfers unless needed; when needed, record explicit queue-family ownership transfers via image barriers. citeturn10search9turn10search28

7) **Memory allocation strategy (including external-memory constraints)**

You have two categories of allocations:

- **“Internal Vulkan” allocations** (images, staging buffers) optimized for Vulkan-only usage.
- **“Interop allocations”** for inference tensors, which must:
  - Be backed by VkDeviceMemory allocated with exportable handle types (e.g., OPAQUE_FD or dma_buf).
  - Use Vulkan external-memory extensions (FD export/import). citeturn10search4turn22search20

If you use VMA:

- VMA supports specifying external memory handle-types per memory type (`pTypeExternalMemoryHandleTypes`) so it can insert `VkExportMemoryAllocateInfoKHR` automatically for those allocations (when configured). citeturn22search6turn21search17  
- This is very relevant for your “tensor buffer allocations” because you want those to be consistently exportable for HIP import.

8) **Common Vulkan anti-patterns in AI pipelines**

- **“ALL_COMMANDS everywhere”** barriers and semaphore stage masks: correct but can serialize the GPU and hide real dependency structure; use only for initial bring-up, then narrow scopes based on actual usage. (The Vulkan spec explains stage masks limit synchronization scopes; overspecification degrades concurrency.) citeturn10search5turn21search7
- **Missing per-plane format handling** when dealing with YUV/multi-plane formats: FFmpeg’s `av_vkfmt_from_pixfmt` explicitly returns per-plane VkFormats and can return NULL if unsupported; you must not assume a single VkFormat. citeturn12view1
- **Implicit ownership assumptions**: if decode/filters allocate on one Vulkan device/context and your pipeline uses another, you can end up with hidden copies or mapping failures.

---

**FFmpeg/Vulkan Interop Proper Technique (if FFmpeg is used)**

6) **Device/frames context ownership and lifetime rules**

FFmpeg’s hardware context model matters for stability:

- `av_hwdevice_ctx_create` creates a device context (including Vulkan). citeturn13view0
- Frames contexts are allocated with `av_hwframe_ctx_alloc` and initialized with `av_hwframe_ctx_init`; frames are allocated via `av_hwframe_get_buffer`. citeturn13view0
- Mapping and transfer paths (`av_hwframe_transfer_data`, `av_hwframe_map`) have explicit semantics; failure with `AVERROR(ENOSYS)` indicates mapping is not possible with the current setup and must be treated as a contract break, not a “retry later.” citeturn13view0
- If you provide your own Vulkan frame pool, the pool must return buffers whose data pointer is an `AVVkFrame`. citeturn12view0

7) **Zero-copy decision tree**

Because you must ultimately feed MiGraphX/HIP, the “best” decision tree is:

- Decode → Vulkan image (zero-copy) **if and only if** FFmpeg emits Vulkan hwframes that are on the same VkDevice you will use for preprocess.
- Otherwise:
  - Decode to CPU frames → Vulkan upload (explicitly logged “CPU→GPU upload path”).
  - Or decode using another hwaccel (VAAPI/DRM) then derive/mapping to Vulkan. FFmpeg explicitly notes indirect mapping scenarios like VAAPI→(DRM)→OpenCL/Vulkan when direct interop is missing, which often implies extra steps/copies. citeturn13view0

8) **Pixel format negotiation and mapping**

- Use FFmpeg’s Vulkan mapping helper for pixel format → VkFormat per plane; if NULL, you must choose a different internal format strategy (e.g., convert in CPU or earlier stage). citeturn12view1

---

**Version Compatibility Matrix (must match / should match / can vary)**

7) **Core constraints derived from sources**

| Component | Must match (hard requirement) | Should match (strongly recommended) | Can vary (with testing) | Evidence basis |
|---|---|---|---|---|
| ROCm stack | OS/kernel/GPU must be in ROCm compatibility matrix for your ROCm release | Use a single pinned ROCm minor release across CI and shipping | Patch-level within same minor once validated | ROCm compatibility matrix is the authoritative source. citeturn7search0turn7search1 |
| MiGraphX | MiGraphX version must be consistent with your ROCm release; MiGraphX docs are published per ROCm version set | Pin MiGraphX to the ROCm release you ship | Rebuild MiGraphX only if you also rebuild entire ROCm user stack | MiGraphX 2.15 docs appear under ROCm Software 7.2.0. citeturn20view0turn23search27 |
| ONNX models | Opset **≤19** (or explicitly verified subset) | Avoid “skip unknown operators” except for experimental triage | Exact opset within ≤19 can vary if supported operators stable | MiGraphX supports opset 19; ONNX tooling may emit opset 26 by default. citeturn8view0turn14search9 |
| Vulkan API | Vulkan 1.3 or `VK_KHR_synchronization2`; external memory/semaphore FD extensions for interop | Prefer sync2 path throughout | Vulkan minor/loader versions as long as extensions and behavior validated | `VK_KHR_synchronization2` is promoted to core in Vulkan 1.3; external FD extensions define the mechanisms. citeturn10search18turn10search4turn21search2 |
| External memory handles | Must have `VK_KHR_external_memory_fd`; optional `VK_EXT_external_memory_dma_buf` depending on handle type | Prefer OPAQUE_FD for same-process HIP import (matches HIP docs) | dma_buf path if needed for other integrations | External memory FD + dma_buf extension definitions; HIP uses OPAQUE_FD example. citeturn10search4turn22search20turn11view0 |
| External semaphore handles | Must have `VK_KHR_external_semaphore_fd` and HIP external semaphore import available | Make semaphore stageMask explicit in submit2 | Implementation details might vary by driver | Vulkan ext defines export/import; submit2 stage mask semantics are defined. citeturn21search2turn21search7turn21search11turn11view0 |
| FFmpeg | Must have Vulkan hwcontext support if you require Vulkan frames | Pin FFmpeg major/minor in your release | Patch level can vary | FFmpeg Vulkan hwcontext and mapping functions exist and define constraints. citeturn12view0turn13view0turn12view1 |
| Mesa RADV (if used) | Needs to support the Vulkan extensions you rely on | Use known-good Mesa versions for production distro | Minor variants by distro | RADV is a Vulkan driver; Mesa docs exist for driver context. citeturn22search10turn22search3 |

---

**Failure Pattern Catalog (symptom → root cause → verify → fix → prevent)**

8) **MiGraphX-centric failure modes**

- **Model parses but fails to compile**
  - Likely root causes:
    - Unsupported operator in MiGraphX’s opset≤19 support surface.
    - Operator supported in principle but not for your dtype/shape; dynamic shape unsupported for an op.
  - How to verify:
    - Inventory ONNX operators in the graph and cross-check against MiGraphX’s support matrix and per-op limitations. citeturn8view0turn14search8
    - Enable compile tracing (`MIGRAPHX_TRACE_COMPILE`, `MIGRAPHX_TRACE_PASSES`) to locate the failing pass. citeturn5search11
  - How to fix:
    - Re-export model to opset≤19 and/or replace unsupported ops in the model graph.
    - Remove dynamic-shape requirements or constrain shapes.
  - Prevent recurrence:
    - Add a CI gate that fails builds if models contain unsupported ops/opsets; treat this as “G1 must pass.”

- **Compiles but fails at runtime with “argument / memory object” style errors**
  - Likely root causes:
    - Offload-copy mismatch between compile-time and runtime expectations; known driver issue indicates you must enable offload-copy when running a model compiled with it. citeturn18search13
  - How to verify:
    - Inspect compile options or driver flags used to produce the cached artifact.
    - Persist compile options in artifact metadata; compare against runtime config.
  - How to fix:
    - Align compile and runtime flags; if using caching, rebuild artifacts or encode flags into cache key.
  - Prevent recurrence:
    - Store compile options in the artifact manifest; fail-fast if mismatch.

- **Runs but produces wrong output**
  - Likely root causes:
    - Layout mismatch (NCHW vs NHWC) or incorrect packing/normalization.
    - FP16/FP8 quantization altering numerics beyond acceptable tolerance.
  - How to verify:
    - Run deterministic golden tests with known input frames and compare outputs; use MiGraphX driver verify strategy where possible (CPU/ref vs GPU). citeturn18search12
    - Toggle NHWC pass forcing (`MIGRAPHX_ENABLE_NHWC`) to see if output changes systematically. citeturn5search11turn4view1
  - How to fix:
    - Enforce tensor contract; lock layout explicitly; add a one-time “tensor visualizer dump” for debugging.
  - Prevent recurrence:
    - Golden-output regression tests.

- **Loading cached compiled model fails intermittently or on some setups**
  - Likely root causes:
    - Artifact not portable across GPU/driver/version; MXR differs by GPU and parameters. citeturn17search21
    - ABI/symbol conflict with msgpack causing load failures. citeturn17search5
  - How to verify:
    - Confirm cache key includes GPU+version tuple; reproduce on “clean process” with only MiGraphX loaded first (to isolate symbol conflicts).
  - How to fix:
    - Fix build/linking to avoid incompatible msgpack symbol resolution; rebuild MiGraphX and dependent libs consistently.
  - Prevent recurrence:
    - CI test that loads cached artifacts in a clean minimal runner process.

9) **Vulkan / interop failure modes**

- **Corruption/stalls that vary with timing (classic “heisenbugs”)**
  - Likely root causes:
    - Missing or incorrect stage/access scopes in barriers; Vulkan defines these scopes precisely. citeturn10search5turn10search9
    - Cross-API synchronization missing: HIP explicitly warns shared buffer must be synchronized between APIs. citeturn11view0
  - How to verify:
    - Disable pipelining (one frame in flight) to see if bug disappears.
    - Add external semaphore handshake and remove implicit waits.
  - How to fix:
    - Implement explicit external semaphore wait/signal around the shared buffer handoff.
  - Prevent recurrence:
    - Never allow “shared buffer used by both APIs without semaphore protocol” in code review (hard rule).

- **External-memory import fails**
  - Likely root causes:
    - Vulkan memory not allocated with exportable handle types (`VkExportMemoryAllocateInfo` missing). citeturn21search17turn10search4
    - Wrong handle type: using dma_buf handle type when HIP code expects opaque FD (or vice versa).
  - How to verify:
    - Compare Vulkan allocation handleTypes with HIP import type; HIP example shows OPAQUE_FD route. citeturn11view0
  - How to fix:
    - Allocate tensor buffers using a consistent export handle type and explicitly test export→import in an isolated smoke test.

10) **FFmpeg/Vulkan interop failure modes**

- **FFmpeg reports unsupported format / mapping failures**
  - Likely root causes:
    - `av_vkfmt_from_pixfmt` returns NULL for certain formats; you attempted unsupported mapping. citeturn12view1
  - How to verify:
    - Log pix_fmt and check mapping result; if NULL, branch.
  - How to fix:
    - Insert explicit convert stage at decode output (either CPU or Vulkan) into a supported internal format.

- **Derived context mapping fails (ENOSYS)**
  - Likely root causes:
    - Unsupported hwframe mapping path; FFmpeg warns mapping feasibility depends on compatible contexts and devices; indirect mapping is sometimes required. citeturn13view0
  - Fix:
    - Use explicit transfer to Vulkan or CPU as a fallback, but log and count as “non-zero-copy mode.”

---

**Practical Verification Strategy (what to test first, what to log, invariants, isolation)**

9) **Test order (fastest isolation of root cause)**

- **First**: MiGraphX-only harness (no Vulkan, no FFmpeg)
  - Load/parse/compile program; run `program::eval` against fixed synthetic tensors; validate output shapes via `get_output_shapes`. citeturn9search0
  - Enable MiGraphX tracing env vars; capture compilation logs. citeturn5search11
- **Second**: Vulkan-only harness (no MiGraphX)
  - Run preprocess kernel on known patterns and read back to CPU for verification.
  - Verify barrier correctness by stress (many frames, randomized scheduling) and by narrowing stage/access scopes. citeturn10search5turn10search2
- **Third**: Interop smoke test (Vulkan↔HIP only, no MiGraphX)
  - Allocate exportable VkDeviceMemory for a buffer, export FD, import to HIP, write in HIP kernel, signal external semaphore, wait in Vulkan, validate contents.
  - HIP provides exactly this style of workflow in its Vulkan interop example and warns about synchronization. citeturn11view0
- **Fourth**: End-to-end but minimal (single frame, single-thread)
  - Decode one frame, preprocess, inference, postprocess, encode.
- **Fifth**: Full pipeline (multi-frame in flight, real clips, perf profiling)

10) **What to log (minimum fields that make failures fixable)**

- Version tuple: ROCm version, MiGraphX version, Vulkan API version + driver string, FFmpeg version, kernel version, GPU model/gfx target, and whether you are using ROCm-supported OS stack per matrix. citeturn7search0turn20view0turn7search1
- MiGraphX compile options: offload_copy, fast_math, exhaustive_tune; plus relevant MIGRAPHX_* env vars that affect compilation. citeturn5search11turn18search12turn18search21
- ONNX model metadata: opset version(s), external data presence, operator histogram. citeturn15search15turn16search2turn8view0
- Interop path: external memory handle type used (OPAQUE_FD vs dma_buf), and whether external semaphores are used. citeturn10search4turn22search20turn21search2turn11view0

11) **Invariants to assert (fail fast)**

- No “skip unknown operators” allowed in production ONNX parse path (it can hide incompatibility).
- MiGraphX program output shapes must match postprocess expectations exactly; use `get_output_shapes` at init and assert per frame. citeturn9search0
- Vulkan resource states: every VkImage used in compute/transfer must have a recorded layout transition when required; every shared VkBuffer must have a barrier or semaphore-defined ordering and visibility.
- Cross-API: A shared buffer must never be accessed by HIP unless Vulkan has signaled “buffer ready” semaphore, and Vulkan must never read results unless HIP has signaled “inference done.” HIP explicitly demands synchronized access across APIs. citeturn11view0

---

## Systematic File-by-File Verification Guide

This section is written so a separate agent can audit an existing near-alpha codebase for wiring correctness.

**A. File/Module Inventory Template**

Organize codebase into these categories (actual filenames differ; the point is coverage):

- **Build and configuration**
  - CMake toolchain, target definitions, compiler flags, feature toggles
- **Core pipeline orchestration**
  - Frame graph / scheduler, module boundaries, pipeline stage contracts
- **Model and inference**
  - ONNX ingest, MiGraphX program builder, compile/cache manager, execution runtime
- **Tensor & preprocessing**
  - Tensor descriptor types, layout converters, normalization, batching, padding/tiling
- **Vulkan subsystem**
  - Instance/device selection, queue selection, command submission, sync2 barriers, resource manager
- **Interop layer**
  - External memory export/import, external semaphore export/import, HIP stream sync policy
- **FFmpeg integration (if present)**
  - Decoder/encoder wrappers, hwcontext setup, frames context ownership, format negotiation
- **Observability**
  - Logging, structured error types, metrics, tracing (ROCTx/rocprof hooks)
- **Tests**
  - Unit tests for tensor/layout, integration tests for interop, end-to-end clips and golden outputs
- **Runtime configuration**
  - CLI, config file parsing, cache directory management & invalidation

**B. Verification checklist per file type (what “correct” looks like)**

To keep this reusable, treat each category as an audit target:

1) **MiGraphX integration module**
- Inspect for:
  - Explicit opset gating (reject >19 unless proven compatible) and operator inventory logging. citeturn8view0turn14search9
  - Clear separation between parse/compile/load/run.
  - Compile options are explicit and logged (offload_copy/fast_math/exhaustive tune). citeturn18search12turn18search21
- Correct looks like:
  - `program::get_parameter_shapes` and `get_output_shapes` are called at init and shape contracts are asserted per frame. citeturn9search0
- Red flags:
  - “skip unknown operators” enabled outside of a strict debug mode. citeturn17search16
  - Cached artifact loaded without verifying compile options (known offload_copy mismatch class). citeturn18search13
- Typical fixes:
  - Add a `CompiledArtifactManifest` written alongside the binary containing compile flags + version tuple; refuse load if mismatch.

2) **Interop layer (Vulkan↔HIP)**
- Inspect for:
  - VkDeviceMemory allocations for tensor buffers contain exportable handleTypes (`VkExportMemoryAllocateInfo`), and code exports FD via KHR external memory FD. citeturn21search17turn10search4
  - HIP imports external memory using correct HIP handle type and maps it to pointer, per HIP interop example. citeturn11view0
  - External semaphore export/import is implemented and used; HIP waits/signals via external semaphore APIs. citeturn21search2turn11view0
- Correct looks like:
  - Every shared buffer handoff is guarded by a semaphore handshake; there is no cross-API “trust me” ordering. citeturn11view0
- Red flags:
  - vkQueueWaitIdle/hipDeviceSynchronize used “to make it work” without external semaphores (hides data hazards and kills perf).
- Typical fixes:
  - Add external semaphore fencepoints around *only* shared-buffer boundaries, keep internal Vulkan barriers for Vulkan-only dependencies.

3) **Vulkan resource/sync module**
- Inspect for:
  - Exclusive use of synchronization2 APIs (`vkCmdPipelineBarrier2`, submit2), or a documented reason to remain on legacy barriers.
  - Barriers use specific stage/access masks; not always ALL_COMMANDS. citeturn10search5turn10search2turn10search18
- Correct looks like:
  - Every VkImage layout transition uses `VkImageMemoryBarrier2` with explicit old/new layout and correct subresource ranges. citeturn10search9turn10search28
- Red flags:
  - Missing layout transitions around storage-image or sampled-image usage.
  - Queue-family transfers implied but not encoded.

4) **FFmpeg integration module**
- Inspect for:
  - Correct lifecycle for av_hwdevice_ctx_create / frames contexts, and correct handling of mapping errors. citeturn13view0
  - If using Vulkan hwframes pool, ensure pool returns AVVkFrame as required. citeturn12view0
  - Pixel format negotiation uses av_vkfmt_from_pixfmt and checks for NULL. citeturn12view1
- Red flags:
  - Assuming a single VkFormat for multi-plane formats.
  - Silent fallbacks to CPU frames without logging.

5) **Observability**
- Inspect for:
  - MiGraphX env-var tooling support in debug builds (trace compile/passes, dump src/asm). citeturn5search11turn5search1
  - ROCTx markers around each stage; rocprof integration and/or MiGraphX driver roctx workflow used for correlation. citeturn23search0turn23search17turn23search8

**C. Wiring Verification Sequence (order of operations)**

Audit in this order to avoid chasing ghosts:

1. **Version/runtime assumptions**  
   Confirm OS/GPU/ROCm compatibility and log the tuple at startup. citeturn7search0turn7search1

2. **Model compatibility assumptions**  
   Confirm model opset≤19 and operator support; refuse unknown ops. citeturn8view0turn14search9

3. **MiGraphX compile/load path**  
   Confirm compile options, caching, and artifact identity (GPU+flags). citeturn18search13turn17search21

4. **Tensor contracts**  
   Assert shape/layout/dtype at boundaries; verify NHWC/NCHW expectations and whether NHWC pass is enabled. citeturn5search11turn4view1

5. **Vulkan resource/sync correctness**  
   Validate barrier scopes and image layout transitions with sync2. citeturn10search18turn10search9turn10search5

6. **FFmpeg interop contracts**  
   Validate device/frames context lifetimes and format mapping. citeturn13view0turn12view1

7. **End-to-end frame correctness**  
   Run short clips with deterministic settings; compare against golden outputs.

8. **Performance instrumentation**  
   Add ROCTx ranges and verify rocprof can attribute time to stages. citeturn23search0turn23search8

9. **Stability/repeatability**  
   Multi-frame in flight; stress; confirm no drift.

10. **Regression tests**  
   Encode known failure signatures into tests (see acceptance gates).

**D. “If Incorrect, Fix It” Rules**

- Prefer **contract tightening** over “workarounds”:
  - If you find unknown ops, do not enable skip-unknown; instead change model export or add explicit rewrite passes upstream.
- Do not change multiple subsystems at once:
  - Fix interop first (semaphores + external memory), then re-evaluate MiGraphX stability.
- After each fix, re-run:
  - MiGraphX-only harness → interop smoke test → end-to-end single frame.

**E. Acceptance Gates (pass/fail)**

- **G1: Model parse + compile deterministic**  
  Same input model + same version tuple + same flags produces identical compile logs and stable output shapes. citeturn5search11turn9search0
- **G2: Inference outputs valid tensor shape/type**  
  Runtime asserts `get_output_shapes()` matches postprocess contract. citeturn9search0
- **G3: Vulkan resources transition correctly under validation**  
  No missing layout transitions; barrier scopes consistent with usage. citeturn10search5turn10search9
- **G4: Frame path correct for test clips**  
  Pixel formats validated; av_vkfmt_from_pixfmt non-NULL or explicit conversion path. citeturn12view1
- **G5: No hidden CPU fallback**  
  Every copy path is logged and counted; “skip unknown ops” disabled in production. citeturn17search16
- **G6: Repeatable outputs across runs**  
  Golden-frame tests match within tolerance across multiple runs.
- **G7: Performance counters/logging present**  
  ROCTx ranges present for each stage; rocprof produces meaningful traces. citeturn23search0turn23search8

---

## Gold Standard Implementation Specification

This is written as a normative engineering spec (MUST/SHOULD/MAY). It assumes Linux-only, C++-only, AMD GPU-only, no CUDA/Python/PyTorch runtime.

**Scope and Non-Goals**

1) **Scope**
- The system MUST implement an AI video enhancement pipeline in C++ on Linux using:
  - Vulkan for GPU preprocessing/postprocessing and resource management.
  - ROCm + MiGraphX for ONNX model inference execution.
  - Optional FFmpeg integration for decode/encode and Vulkan hwframes interop. citeturn13view0turn12view0

2) **Non-goals**
- The system MUST NOT support CUDA.
- The system MUST NOT require Python at runtime.
- The system MUST NOT provide any PyTorch runtime path.

**System Modules and Responsibilities**

The implementation MUST provide these modules (names are conceptual):

- `VideoIO` (optional FFmpeg-based)
  - MUST manage decode/encode lifetimes and explicitly expose frame format/color metadata.
  - MUST support a Vulkan hwcontext path if enabled; if using a custom Vulkan frame pool, it MUST return AVVkFrame-backed buffers. citeturn12view0turn13view0
- `VulkanRuntime`
  - MUST initialize VkInstance/VkDevice/VkQueue(s), and MUST use synchronization2 APIs when available (`VK_KHR_synchronization2` or Vulkan 1.3). citeturn10search18
  - MUST manage VkImages/VkBuffers and record command buffers with explicit barriers.
- `PreprocessGraph`
  - MUST convert decoded frames into inference tensor buffers using Vulkan compute.
- `InferenceRuntime`
  - MUST implement MiGraphX program lifecycle: parse → compile → cache/load → eval → finish. citeturn9search0turn16search23
  - MUST surface `get_parameter_shapes()` and `get_output_shapes()` at init time and enforce tensor contract invariants. citeturn9search0
- `InteropBridge`
  - MUST implement Vulkan↔HIP external memory + semaphore interop as the primary path, using FD handle export/import for memory and semaphores. citeturn10search4turn21search2turn11view0
  - MUST define explicit semaphore handshake states: `BufferReady` and `InferenceDone`.
- `PostprocessGraph`
  - MUST map inference output tensor buffer into output frames using Vulkan compute.
- `Scheduler`
  - MUST provide a frame-slot model with explicit stage transitions and deterministic resource lifetimes.
- `Observability`
  - MUST support structured logs and MUST expose MiGraphX/Vulkan/interop debug toggles.

**Data Contracts**

1) **Frame formats**
- The system MUST define supported input pixel formats and internal canonical formats.
- If FFmpeg Vulkan path is used, the system MUST validate that `av_vkfmt_from_pixfmt` returns a non-NULL mapping for the selected format(s); otherwise it MUST select a different format strategy. citeturn12view1

2) **Tensor formats**
- The system MUST define, per model:
  - Input tensor shape(s) and axis meanings.
  - Layout: NCHW or NHWC (or other).
  - Dtype: fp32/fp16/int8/fp8, with explicit quantization policy.
- The system MUST validate model opset≤19 unless explicitly waived by a tested exception list. citeturn8view0turn14search8

3) **Color pipeline**
- The system MUST preserve and propagate color metadata through preprocessing and postprocessing.
- Any colorspace conversion MUST be explicitly configured (no “assume BT.709 always” behavior) and MUST be testable on known clips.

**MiGraphX Integration Contract**

- Model ingestion MUST:
  - Reject models requiring unsupported ops/opsets or unsupported dynamic-shape semantics based on MiGraphX operator support matrix. citeturn8view0turn14search8
- Compilation MUST:
  - Explicitly set/record compile options (offload_copy, fast-math, exhaustive tuning).
  - MUST log all options and relevant MIGRAPHX_* env var values that affect compilation. citeturn5search11turn18search21turn18search12
- Caching MUST:
  - Key artifacts on (model hash + external data hashes) + (ROCm/MiGraphX version) + (GPU architecture) + (compile options) + (layout/precision settings). citeturn16search2turn20view0turn17search21
  - MUST refuse loading an artifact if cache key mismatch is detected.
- Execution MUST:
  - Use `program::eval(...)` and MUST call `program::finish()` before releasing per-frame shared resources. citeturn9search0
  - MUST validate output shapes against `get_output_shapes()` and postprocess contract. citeturn9search0
- Quantization MUST:
  - If INT8 quantization is enabled, MUST provide calibration data and MUST document which ops are quantized (dot/convolution default). citeturn20view0
  - If FP8 is enabled, MUST restrict to E4M3FNUZ and MUST document numeric implications. citeturn8view0turn15search16

**Vulkan Contract**

- Device/queue requirements:
  - MUST support `VK_KHR_synchronization2` or Vulkan 1.3. citeturn10search18
  - MUST support required external handle extensions if interop is enabled: `VK_KHR_external_memory_fd` and `VK_KHR_external_semaphore_fd`. citeturn10search4turn21search2
- Synchronization contract:
  - MUST use synchronization2 barriers (`vkCmdPipelineBarrier2`) with correct, minimal stage/access scopes per resource usage. citeturn10search2turn10search5turn10search9
- Resource transition contract:
  - MUST record explicit layout transitions for images via `VkImageMemoryBarrier2`. citeturn10search9turn10search28
- Allocation strategy:
  - Interop tensor buffers MUST be allocated from exportable VkDeviceMemory (handleTypes include OPAQUE_FD or dma_buf), and the chosen handle type MUST match HIP import type. citeturn21search17turn10search4turn22search20turn11view0
  - If VMA is used, external handle configuration SHOULD be provided via allocator create info when appropriate. citeturn22search6
- Validation/debug:
  - MUST support enabling Vulkan debug layers and MUST log validation errors verbosely in debug builds.

**FFmpeg Interop Contract (if used)**

- Context creation/ownership MUST:
  - Use FFmpeg’s hwdevice/hwframes context APIs correctly and preserve lifetimes until frames are retired. citeturn13view0
- Frame transfer/interop MUST:
  - Treat `av_hwframe_map` failures (ENOSYS) as a contract failure requiring fallback path selection (explicit transfer). citeturn13view0
- Failure fallback behavior:
  - MAY fall back to CPU frames/upload, but MUST log and mark degraded mode.

**Threading and Scheduling Contract**

- Modules MAY run concurrently, but MUST obey:
  - A per-frame-slot state machine: DecodeReady → PreprocessDone → InferenceDone → PostprocessDone → EncodeDone.
  - Cross-API boundaries MUST be synchronized with external semaphores, not ad-hoc sleeps or global device waits. citeturn11view0turn21search2

**Error Taxonomy and Recovery Policy**

- The system MUST categorize errors at minimum into:
  - ModelIncompatible (unsupported ops/opset/dtype)
  - CompileFailure (MiGraphX compile)
  - ArtifactInvalid (cache mismatch, load failures incl. msgpack conflicts) citeturn17search5turn18search13
  - VulkanDeviceFailure (missing extensions, validation errors)
  - InteropFailure (export/import failure)
  - SyncHazard (detected hazard, corruption)
  - FFmpegInteropFailure (mapping/transfer)
- Recovery MUST:
  - Fail-fast for contract violations (ModelIncompatible, SyncHazard).
  - MAY retry transient failures (some FFmpeg decode errors), but must not retry compile endlessly.

**Observability and Diagnostics**

- MUST provide:
  - MiGraphX tracing toggles via env vars and/or config passthrough. citeturn5search11turn5search1
  - ROCTx ranges around each pipeline stage, and documentation for using rocprof `--roctx-trace`. citeturn23search0turn23search8
  - Optional MiGraphX driver-based verification path (driver “verify”, “roctx” usage) for CI and diagnosis. citeturn18search12turn23search17

**Test Specification**

- Unit tests MUST cover:
  - Tensor layout packing/unpacking, normalization, dtype conversions.
- Integration tests MUST cover:
  - Vulkan preprocess correctness on known patterns.
  - Vulkan↔HIP external memory/semaphore handshake smoke test. citeturn11view0turn21search2turn10search4
  - MiGraphX compile + eval determinism test using fixed seeds and known inputs. citeturn9search0turn5search11
- End-to-end tests MUST cover:
  - Short “golden clip” outputs and regression thresholds.
- Stress tests MUST cover:
  - Multi-frame in flight + long durations to surface sync hazards.

**Verification Matrix (Requirement → method → expected result)**

| Requirement | Verification method | Expected result |
|---|---|---|
| Opset gate ≤19 | Model inspector + CI gate | CI fails on opset>19 unless exception list; artifact not built citeturn8view0turn14search9 |
| External memory interop | Vulkan export FD → HIP import/map → Vulkan wait | Deterministic buffer contents, no corruption citeturn11view0turn10search4 |
| External semaphore correctness | HIP wait/signal with Vulkan-exported semaphore FD | Correct ordering; no read-before-write citeturn21search2turn11view0 |
| Sync2 barrier correctness | Run with narrow stage/access scopes in stress | No hazards/corruption; predictable perf citeturn10search5turn10search2 |
| MiGraphX artifact validity | Cache-key comparison + load test | Load succeeds only for matching tuple; mismatch fails-fast citeturn17search21turn18search13 |

**Implementation Readiness Checklist**

- All module contracts written and referenced in code.
- Cache key and artifact manifest implemented.
- Interop smoke test passes on target GPUs.
- Golden clip regression suite exists.
- MiGraphX compile tracing and Vulkan validation toggles exist.

---

## Risk Register + Priority Fix Plan

**Top 10 risks (highest impact first)**

1) **Cross-API memory visibility bug** (Vulkan↔HIP) → corruption/stalls  
Mitigation: external semaphore protocol required; HIP warns sync is needed. citeturn11view0turn21search2

2) **Model opset/operator mismatch** (exported opset > 19, unsupported ops) → compile/runtime failure  
Mitigation: opset gate ≤19; operator inventory vs MiGraphX matrix. citeturn8view0turn14search9

3) **Tensor layout mismatch (NHWC/NCHW)** → wrong output that “looks plausible”  
Mitigation: enforce explicit tensor contract; NHWC pass toggles logged. citeturn5search11turn4view1

4) **Compile/runtime flag mismatch (offload_copy)** → runtime failures only when loading cached artifact  
Mitigation: encode compile options into cache key + manifest; known mismatch exists. citeturn18search13turn18search21

5) **Vulkan format mismatch in FFmpeg pipeline** → mapping failures, silent fallback, wrong colors  
Mitigation: handle per-plane VkFormats and NULL mapping result from FFmpeg helper. citeturn12view1

6) **Artifact non-portability across GPUs/versions** → “works on dev box, fails on user box”  
Mitigation: strict cache keying; MXR differs by GPU/params. citeturn17search21turn7search0

7) **Msgpack/ABI conflict breaks MiGraphX load path** → intermittent startup failures  
Mitigation: control symbol resolution and dependency versions; known issue exists. citeturn17search5

8) **Over-broad Vulkan synchronization** → stable but too slow; hides real hazards  
Mitigation: start safe (ALL_COMMANDS), then tighten stage/access scopes; spec defines scope semantics. citeturn10search5turn21search7

9) **ROCm/OS matrix mismatch in deployment** → unpredictable failures  
Mitigation: ship only tested OS/kernel/GPU combos as per ROCm compatibility matrix. citeturn7search0turn7search1

10) **Quantization drift (FP16/INT8/FP8)** → visual artifacts or failures  
Mitigation: controlled rollout; MiGraphX quantization APIs and FP8 type constraints documented. citeturn20view0turn15search16turn19search18

**Priority fix plan (actionable)**

- P0: Implement interop smoke test + external semaphore handshake as mandatory infrastructure. citeturn11view0turn21search2
- P0: Lock ONNX opset/operator gate and generate operator histogram per model. citeturn8view0turn14search9
- P1: Implement cache manifest (compile flags + version tuple + GPU id) and refuse mismatched loads. citeturn18search13turn17search21
- P1: Add shape/layout assertions and golden-frame tests (one frame, deterministic).
- P2: Add MiGraphX tracing toggles and capture compile pass logs in CI for regressions. citeturn5search11
- P2: Harden FFmpeg pixel format negotiation; fail fast on unsupported mapping. citeturn12view1

---

## Concise First 72 Hours Remediation Sequence

Day 1 (stabilize contracts and isolate)
- Build a **MiGraphX-only** harness that loads/compiles/evals the target model; log opset, output shapes, compile flags, and enable `MIGRAPHX_TRACE_COMPILE` + `MIGRAPHX_TRACE_PASSES` in a debug run. citeturn9search0turn5search11turn8view0
- Extract an operator histogram and compare to MiGraphX supported operator matrix; fail if unsupported ops are present. citeturn8view0turn14search8

Day 2 (make interop correct and testable)
- Implement Vulkan↔HIP external memory + external semaphore smoke test using the exact HIP–Vulkan interop flow from HIP docs (FD export/import and semaphore handshake). citeturn11view0turn10search4turn21search2
- Integrate this interop handshake into the real pipeline around the tensor buffers (not images).

Day 3 (reintegrate full pipeline with observability)
- Convert preprocess output to shared VkBuffer tensor; run inference; postprocess; validate a single frame end-to-end.
- Add ROCTx ranges around each stage and confirm rocprof `--roctx-trace` captures stage boundaries. citeturn23search0turn23search8
- Implement cache manifest and invalidate/rebuild compiled artifacts to eliminate flag mismatches (especially offload_copy). citeturn18search13

---

## Known Unknowns

- Which specific GPUs (Radeon vs Instinct families, exact gfx targets) are in your deployment set and which Vulkan driver (RADV vs AMDVLK) you standardize on; extension availability must be measured per target stack. citeturn7search0turn22search10turn10search4turn21search2
- Whether you require decode/encode to stay entirely on GPU, and whether FFmpeg’s Vulkan hwframes path is used in your codebase or only Vulkan for filters. citeturn13view0turn12view0
- Whether your shipped models depend on dynamic shapes in operators where MiGraphX explicitly notes “dynamic shape is not supported.” citeturn8view0
- Whether you plan to ship INT8/FP8 inference; the appropriate calibration and acceptance thresholds are product-specific even though APIs exist. citeturn20view0turn15search16turn19search18

## Assumptions made

- The inference model is ONNX and is intended to be executed directly using MiGraphX (not via ONNX Runtime), consistent with your stated non-goals.
- Vulkan is used for preprocess/postprocess and you want a mostly GPU-resident pipeline to avoid PCIe copies.
- FFmpeg is either used now or is a likely integration point; the guidance is written to be compatible with FFmpeg’s Vulkan hwcontext design. citeturn13view0turn12view0turn12view1

## What to verify in the actual codebase next

- Whether inference tensors currently cross Vulkan↔MiGraphX via:
  - CPU staging copies (likely) or ad-hoc waits (risk), instead of external semaphores. citeturn11view0
- Whether MiGraphX cached artifacts are keyed on GPU+version+compile flags, and whether offload_copy mismatches are possible. citeturn18search13turn17search21turn18search21
- Whether the model opset is pinned ≤19 and whether any unsupported ops are masked by “skip unknown operators.” citeturn8view0turn14search9turn17search16
- Whether tensor layout is explicitly defined and asserted, and whether NHWC forcing is accidentally enabled in production via environment variables. citeturn5search11turn4view1
- Whether FFmpeg Vulkan pixfmt→VkFormat mapping is validated (NULL mapping handled) and whether multi-plane formats are correctly supported. citeturn12view1