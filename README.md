# AMD Video Enhancer

AMD Video Enhancer is a Linux-first video enhancer that uses ML and AI models to restore, clean, and upscale footage on AMD GPUs. It is intentionally built around AMD's ROCm, MiGraphX, HIP, and Vulkan stack rather than NVIDIA/CUDA, because the whole point of this project is to give AMD hardware a first-class experience instead of treating it like an afterthought.

## Release snapshot

**Current public state:** `pre-release-staging` is preparing the first public GitHub beta prerelease. No public beta prerelease is published on the GitHub Releases page yet.

**Verified primary system:** Arch Linux + Ryzen 7 7800X3D + Radeon RX 7900 GRE.

**Primary verified backend path:** MiGraphX inference with the FFmpeg media pipeline on that reference system.

**Bundled MiGraphX reality:** the planned beta packages bundle a custom MiGraphX runtime/toolchain because the required upstream behavior is not yet available in the stock system path.

**Support boundary:** package targets are broader than verified compatibility. Packaging reach is not the same thing as support proof.

Start with these canonical release-facing docs before assuming more than the repo can currently prove:

- [`docs/RELEASE_STATUS.md`](docs/RELEASE_STATUS.md)
- [`docs/SUPPORT_TIERS.md`](docs/SUPPORT_TIERS.md)
- [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md)
- [`docs/VALIDATION_AND_EVIDENCE.md`](docs/VALIDATION_AND_EVIDENCE.md)
- [`docs/BETA_TESTING_PROGRAM.md`](docs/BETA_TESTING_PROGRAM.md)

## Why this kind of app matters

Old video is usually not “bad” in just one way. It is often blurry, noisy, blocky, over-compressed, poorly scaled, repeatedly re-encoded, damaged by weak capture hardware, or scarred by years of format conversion.

That is why video enhancement tools matter. When they are done well, they can:

- recover detail that low-quality scaling threw away
- reduce visible compression damage
- clean up ringing, blocking, and mosquito noise
- improve damaged or neglected archive footage
- make low-quality source material more usable for editing, research, restoration, and personal preservation
- turn “barely watchable” clips into footage that is genuinely useful again

This repository exists because that value should not belong only to closed tools, only to expensive workflows, or only to one GPU vendor.

## Why the AMD focus is deliberate

A lot of the AI video software landscape has been shaped around NVIDIA-first assumptions. Commercial products like Topaz Video AI can be powerful, but they are expensive for many hobbyists, archivists, tinkerers, and small creators, and the broader tooling ecosystem often aims its fastest path at CUDA, Tensor cores, and NVIDIA-specific stacks first.

The result is familiar: capable AMD hardware ends up pushed onto generic paths, partial support, slower fallbacks, or no serious support at all. AMD users do not consistently get the same polished experience NVIDIA users get, even when the underlying GPUs are more than capable of doing real work.

That gap keeps growing because industry habits keep reinforcing themselves. CUDA launched in 2007, which means it has had nearly twenty years of real-world adoption, documentation, production hardening, tutorials, examples, and mindshare; that long head start taught much of the industry to see GPU compute through an NVIDIA-shaped lens, while AMD's ROCm ecosystem developed with far less public visibility and far fewer end-user application examples.

## Where MiGraphX fits into that story

MiGraphX is AMD's graph compiler and inference runtime layer inside the ROCm stack. In plain English: it sits in the part of the stack that takes a model graph, lowers it into something AMD GPUs can execute efficiently, and gives applications a path to better inference performance than “just run something generic and hope for the best.”

The problem is that MiGraphX is still almost invisible in public end-user applications. Public, consumer-facing examples are rare enough that many developers have never seen what it looks like in a real app, what kind of performance discipline it enables, or how to build supporting systems around it for model preparation, artifact reuse, runtime validation, and shipping.

That lack of public history makes the stack harder to learn, harder to trust, and harder to copy. One of the big reasons this repository matters is that it is a public implementation of a thing very few people ever get to inspect: a real AMD-first video enhancement app that treats MiGraphX as a serious production backend instead of a footnote.

That scarcity also matters for the way people talk about AI. A common dismissal is that AI can only remix patterns from mature, heavily documented ecosystems that already have endless tutorials, examples, and stack-overflowed history behind them. This project pushes against that claim: the MiGraphX side of the stack is sparse enough in public examples that there was no comfortable library of consumer-app patterns to simply copy.

To be careful and honest: this README does **not** claim that nobody anywhere has ever built anything remotely similar in private. What it does say is that, to the best of this project's public view, there was no rich public trail of MiGraphX-powered consumer video-enhancer implementations to follow. The lack of documentation and lack of examples is part of the story, not a footnote.

If you want the blunt version of why that matters beyond this app itself, read [`docs/WHY_THIS_PROJECT_MATTERS.md`](docs/WHY_THIS_PROJECT_MATTERS.md).

## Yes, this project is 100% vibe coded

And here is the part that makes the whole thing even wilder: this app is 100% vibe coded.

Not “vibe coded by a veteran C++ graphics engineer who already knew the stack by heart.” Vibe coded by a zero-knowledge vibe coder. That is part of the point of the project too: not just to build useful software, but to prove that a difficult, AMD-first, systems-heavy application can be explored, assembled, tested, documented, and improved in public rather than being locked behind specialist gatekeeping.

That point gets even sharper when you connect it to the MiGraphX situation above. The usual comfort argument is that AI can only regurgitate what it has already seen, so deeply novel or under-documented work is supposedly protected. This repository is evidence that the story is not that simple. Here, AI-assisted work had to operate inside a stack with thin docs, sparse examples, immature edges, packaging problems, runtime-validation problems, and performance questions that did not come with a neat cookbook.

In other words: the AI was not just copying a familiar pattern from a saturated ecosystem. It helped move work forward in an area where the public examples are scarce, the documentation is incomplete, and parts of the surrounding software stack still needed to be pushed, organized, and made more usable. That is an important distinction.

This does **not** make human judgment irrelevant. It does mean that the reassuring line “AI only copies what already exists, so real software engineering is safe” is a much weaker defense than many developers want it to be. If a zero-knowledge vibe coder and AI can produce a real, inspectable application in an under-documented MiGraphX/ROCm problem space, then the ceiling on AI-assisted development is already higher than a lot of people are comfortable admitting.

## What is in the repo today

This repository currently contains:

- a native **C++20** application core
- a **CLI** frontend (`ave`)
- an optional **Qt GUI** frontend (`ave_gui`)
- a deterministic **pipeline planner** for stage ordering
- a **model manager** that downloads, prepares, compiles, validates, and reuses model artifacts
- a **MiGraphX-first** inference path
- a **ROCm/HIP ONNX Runtime** fallback path
- a **Vulkan Compute** fallback path
- an **NCNN Vulkan** fallback path
- an **FFmpeg-based** media pipeline and fallback filter path
- a packaging toolchain for **portable bundles** and **native Linux packages**

If you want the implementation-level version of that story, read [`docs/GOLD_STANDARD_FOR_IMPLEMENTATION.md`](docs/GOLD_STANDARD_FOR_IMPLEMENTATION.md). That document is the technical reality check. This README is the guided front door.

## How the app works at a high level

The short version of the runtime flow is:

1. the user picks stages in the CLI or GUI
2. `PipelinePlanner` puts those stages into a safe deterministic order
3. `VideoProcessor` resolves models, probes runtimes, and selects a backend
4. the backend pre-validates work through `runStage(...)`
5. real frame-by-frame processing happens during `processVideoFile(...)`
6. FFmpeg stays underneath the whole thing as the media spine for probing, frame flow, filters, and final encode

When `--backend auto` is used, the current backend order is:

1. **MiGraphX**
2. **ROCm/HIP (ONNX Runtime)**
3. **Vulkan Compute**
4. **NCNN Vulkan**
5. **FFmpeg fallback**

That order matters because this is not a single-engine toy app. It is a layered video pipeline with multiple real execution paths.

## Learn the repo without getting lost

If you want the educational tour instead of diving blind into C++ files, use these guides:

| Read this | What it helps you understand |
| --- | --- |
| [`docs/README.md`](docs/README.md) | the documentation set and which doc to read first |
| [`docs/WHY_THIS_PROJECT_MATTERS.md`](docs/WHY_THIS_PROJECT_MATTERS.md) | the sharp strategic argument for why this repo matters beyond the app itself |
| [`docs/GOLD_STANDARD_FOR_IMPLEMENTATION.md`](docs/GOLD_STANDARD_FOR_IMPLEMENTATION.md) | the current technical truth of the app |
| [`src/README.md`](src/README.md) | how the implementation is laid out |
| [`src/backends/README.md`](src/backends/README.md) | what each backend does and why it exists |
| [`src/gui/README.md`](src/gui/README.md) | how the Qt frontend is organized |
| [`include/ave/README.md`](include/ave/README.md) | what the public headers expose |
| [`tests/README.md`](tests/README.md) | how the test suite is organized |
| [`tools/README.md`](tools/README.md) | build, packaging, benchmark, and ROCm-stack helper scripts |
| [`packaging/README.md`](packaging/README.md) | distro package templates and packaging metadata |
| [`cmake/README.md`](cmake/README.md) | launcher, bundling, and install-time CMake helpers |
| [`benchmarks/README.md`](benchmarks/README.md) | benchmark assets, generated results, and history |

Recommended reading order if you are new:

1. this README
2. [`docs/WHY_THIS_PROJECT_MATTERS.md`](docs/WHY_THIS_PROJECT_MATTERS.md)
3. [`docs/GOLD_STANDARD_FOR_IMPLEMENTATION.md`](docs/GOLD_STANDARD_FOR_IMPLEMENTATION.md)
4. [`src/README.md`](src/README.md)
5. [`src/backends/README.md`](src/backends/README.md)
6. [`tools/README.md`](tools/README.md)

## Prerequisites

### To run packaged builds

You still need the host-side AMD GPU stack that cannot be bundled inside the app archive:

- Linux
- a supported AMD GPU
- working `amdgpu` / KFD device access
- Vulkan driver support on the host
- a functioning ROCm-capable environment for MiGraphX / HIP execution paths

Portable bundles and native packages can ship userspace dependencies, but they cannot ship your kernel driver stack.

### To build from source

The source build expects these core pieces:

- **CMake 3.21+**
- a **C++20 compiler**
- **pkg-config**
- FFmpeg development packages for:
  - `libavcodec`
  - `libavformat`
  - `libavutil`
  - `libavfilter`
  - `libswscale`
- **ROCm / HIP** headers if you want the AMD GPU paths
- **MiGraphX** if you want the primary backend
- **ONNX Runtime** with ROCm support if you want the ROCm/HIP fallback backend
- **Vulkan** loader/SDK support if you want Vulkan-based paths
- **NCNN** if you want the NCNN Vulkan backend
- **Qt 6 Widgets** if you want the GUI target
- **Python 3** for model-bundling and portable/package workflows

### Runtime notes

- release packages bundle app-private `ffmpeg` and `ffprobe`
- source builds still expect `ffmpeg` and `ffprobe` in `PATH`
- archive-backed model extraction expects `unzip` in `PATH`
- mixed iGPU+dGPU systems may need `HIP_VISIBLE_DEVICES` or `ROCR_VISIBLE_DEVICES`
- unsupported ROCm distributions are still best-effort even if the app itself builds there

## Build from source

Always configure with the major backends enabled:

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

Do **not** use a bare `cmake -S . -B build` in this repository. The project is designed around explicit backend selection.

If you need to bundle a custom MiGraphX runtime/toolchain for packaging work, pass it explicitly with `-DAVE_BUNDLED_MIGRAPHX_PREFIX=/path/to/custom/migraphx`. Release-facing packaging should not depend on a hidden maintainer-local home-directory path.

## How to use the app

### Check what the machine can actually run

```bash
./build/ave --list-backends
```

### Preview a job without rendering it

```bash
./build/ave \
  --input input.mp4 \
  --output output.mp4 \
  --stage restore_compression \
  --stage upscale:model=clearreality-x4-fast \
  --dry-run
```

### Run a CLI job

```bash
./build/ave \
  --input input.mp4 \
  --output output.mp4 \
  --stage denoise \
  --stage upscale:model=openproteus-compact-x2
```

### Run the GUI

```bash
./build/ave_gui
```

### What to expect on first use

The first run of a MiGraphX-backed model may trigger model preparation or compilation. Compiled `.mxr` artifacts are cached and reused later when the runtime fingerprint still matches the current environment.

That first-run path can take minutes on the reference stack. The app now emits compile progress while `migraphx-driver` is working, but some preparation phases are still coarser than a perfect progress bar. Treat first-run preparation as deliberate setup work, not as a sign that later runs will always be that slow.

## ROCm and MiGraphX customization

This repository does more than just “use ROCm if it happens to be installed.” It also supports a customized, app-bundled MiGraphX toolchain flow for packaging and portable builds.

Important pieces of that story:

- release tooling expects any custom MiGraphX prefix to be passed explicitly through `AVE_BUNDLED_MIGRAPHX_PREFIX` or `-DAVE_BUNDLED_MIGRAPHX_PREFIX=...`
- packaging can bundle a custom MiGraphX runtime under the app tree
- compile-enabled portable bundles can ship `migraphx-driver` and its side libraries
- helper scripts can stage ABI-matched MiGraphX runtime overlays and compiler-side dependencies from the local package cache
- portable and packaged builds isolate the app runtime from unrelated host libraries as much as possible

The best detailed references for that part of the project are:

- [`docs/PACKAGING.md`](docs/PACKAGING.md)
- [`docs/migraphx_debugging_playbook.md`](docs/migraphx_debugging_playbook.md)
- [`tools/README.md`](tools/README.md)
- [`packaging/README.md`](packaging/README.md)
- [`cmake/README.md`](cmake/README.md)

Useful runtime tuning variables include:

- `AVE_MIGRAPHX_COMPILE_PROFILE=fast|balanced|exhaustive`
- `AVE_MIGRAPHX_PROBLEM_CACHE=/path/to/problem_cache.json`
- `AVE_MIGRAPHX_MIOPEN_FIND_MODE=FAST|DYNAMIC_HYBRID|NORMAL`
- `AVE_MIGRAPHX_MIOPEN_COMPILE_PARALLEL_LEVEL=<n>`
- `AVE_MIGRAPHX_VISIBLE_DEVICES=<gpu-list>`

## Current project status

- `pre-release-staging` is preparing the first public GitHub beta prerelease; no public release is published yet
- the primary verified stack is **Arch Linux + Ryzen 7 7800X3D + Radeon RX 7900 GRE**
- support tiers live in [`docs/SUPPORT_TIERS.md`](docs/SUPPORT_TIERS.md)
- the current release truth surface lives in [`docs/RELEASE_STATUS.md`](docs/RELEASE_STATUS.md)
- known limitations live in [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md)
- validation scope and evidence rules live in [`docs/VALIDATION_AND_EVIDENCE.md`](docs/VALIDATION_AND_EVIDENCE.md)
- native package, AUR handoff, and portable bundle details live in [`docs/PACKAGING.md`](docs/PACKAGING.md)
- benchmark snapshots and reproduction notes live in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)

## More project docs

- [`docs/WHY_THIS_PROJECT_MATTERS.md`](docs/WHY_THIS_PROJECT_MATTERS.md)
- [`docs/FEATURE_PARITY_MATRIX.md`](docs/FEATURE_PARITY_MATRIX.md)
- [`docs/CUSTOM_MODEL_MANIFEST.md`](docs/CUSTOM_MODEL_MANIFEST.md)
- [`docs/RELEASING.md`](docs/RELEASING.md)
- [`CHANGELOG.md`](CHANGELOG.md)
- [`SECURITY.md`](SECURITY.md)
- [`CONTRIBUTING.md`](CONTRIBUTING.md)

## License and security

- license: [`MIT`](LICENSE)
- security guidance: [`SECURITY.md`](SECURITY.md)

## Bottom line

This project is trying to do two things at once: build a useful AMD-first AI video enhancer, and make the path visible enough that other people can learn from it instead of treating the whole ROCm/MiGraphX side of the world like a black box. If you care about restoration, upscaling, AMD GPU compute, open implementation details, or the sheer chaos-energy of a zero-knowledge vibe-coded systems project actually becoming real software, you are in the right repo.
