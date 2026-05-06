# Implementation File Map

> Generated on 2026-05-05 from remote GitHub connector inspection plus the R2 handoff archive.
>
> This map is a Phase 0 guardrail. It is not a substitute for local `rg`, build, and test validation on the implementation workstation.

## Discovery commands / queries used

Remote connector searches and file fetches were used because the container could not resolve `github.com` for a normal clone.

Equivalent local commands to rerun:

```bash
rg -n "int main|PipelinePlanner|StageKind|stageKindFromString|processVideoFile|runStage|Backend|MiGraphX|Vulkan|NCNN|FFmpeg|model.*manifest|family|profile|preview|cache|QMainWindow" src include tests docs packaging tools cmake
find src include tests tools packaging cmake docs -maxdepth 3 -type f | sort
```

## Summary map

| Area | Real files/classes/functions observed | Status | Notes |
| --- | --- | --- | --- |
| CLI entry point | `src/main.cpp`, `src/cli.cpp` | Found | `src/README.md` says these translate user intent into app actions. |
| GUI entry point | `src/gui/main_gui.cpp` | Found | Optional Qt frontend entrypoint. |
| Main window/layout | `src/gui/main_window.cpp`, `src/gui/main_window.hpp` | Found | Current main workflow wiring; workstation redesign must start here after Phase 0. |
| GUI supporting controls | `src/gui/filter_browser.*`, `src/gui/model_manager_dialog.*`, `src/gui/settings_dialog.*`, `src/gui/toggle_switch.*` | Found | Listed in `src/gui/README.md`. |
| Preview widget/viewer | No dedicated preview widget verified in remote search | Not verified | `src/gui/README.md` says GUI can preview work, but a dedicated split-wipe/timeline viewer seam was not verified. |
| Timeline/cache UI | No dedicated timeline/cache UI verified | Not found / not verified | R2 preview-cache and workstation work must introduce this carefully after file discovery. |
| Stage planner | `include/ave/planner.hpp`, `src/planner.cpp`, `tests/planner_tests.cpp` | Found | Roadmap fusion work should begin with planner tests before implementation. |
| Stage taxonomy | `include/ave/types.hpp`, `src/stage.cpp`, `src/types.cpp` likely | Found / verify locally | Search found `StageKind` in `include/ave/types.hpp`; exact enum contract must be inspected locally before changes. |
| Video orchestration | `src/video_processor.cpp`, `include/ave/video_processor.hpp` | Found | `src/README.md` names this as orchestration layer. |
| Backend interface | `include/ave/backend.hpp`, `src/backend.cpp`, `src/backend_manager.cpp` | Found | Backend order/support tiers must stay aligned with docs. |
| Backend selection | `src/backend_manager.cpp` | Found / verify locally | README documents auto order: MiGraphX, ROCm/HIP, Vulkan Compute, NCNN Vulkan, FFmpeg fallback. |
| MiGraphX integration | `src/backends/migraphx_backend.cpp`, `include/ave/backends/migraphx_backend.hpp`, `src/model_manager.cpp` | Found | Verified primary path on reference system. |
| ROCm/HIP fallback | `src/backends/rocm_hip_backend.cpp`, `include/ave/backends/rocm_hip_backend.hpp` | Found | Preview-target support tier. |
| Vulkan fallback | `src/backends/vulkan_compute_backend.cpp`, likely header under `include/ave/backends/` | Found | Preview-target support tier. |
| NCNN fallback | `include/ave/backends/ncnn_vulkan_backend.hpp`; implementation to verify | Partially verified | Search found header; local file listing should confirm `.cpp`. |
| GLSL / VapourSynth paths | `src/backends/glsl_shader_backend.cpp`, `src/backends/vapoursynth_backend.cpp`, matching headers | Found | Experimental/manual, not part of main release promise. |
| FFmpeg media spine | `src/ffmpeg_runner.cpp`, `src/frame_io.cpp`, `src/frame_io_vulkan.cpp`, `src/rgb_video_loop.cpp`, `src/video_probe.cpp` | Found | Core media path and fallback spine. |
| Model catalog | `src/model_catalog.cpp`, likely `include/ave/model_catalog.hpp` | Found via source guide | Local inspection required before family metadata changes. |
| Model manager / lifecycle | `src/model_manager.cpp`, `include/ave/model_manager.hpp`, `tests/model_manager_profile_tests.cpp` | Found | Contains manifest parsing, download validation, artifact naming, MiGraphX preparation behavior. |
| Tensor contracts | `src/tensor_contract.cpp`, tests per `tests/README.md` | Found via source guide | Important for model family/mode contracts. |
| Runtime diagnostics | `src/runtime_paths.cpp`, `src/runtime_diagnostics.cpp`, `src/telemetry.cpp`, `src/observability.cpp`, `src/process_*.cpp` | Found via source guide | Required for beta issue triage and failure transparency. |
| Job state/recovery | `src/job_queue.cpp`, `src/job_recovery.cpp`, `src/app_settings.cpp` | Found via source guide | Preview-cache work likely needs these seams. |
| Render/preview job system | Job queue/recovery exists; dedicated segment-preview system not verified | Partial | Do not assume R2 preview cache concepts already exist. |
| Tests/fixtures | `tests/README.md`, `tests/planner_tests.cpp`, many subsystem tests | Found | Tests are meaningful subsystem/smoke coverage, not broad golden-clip regression proof. |
| Packaging | `docs/PACKAGING.md`, `tools/package_release.sh`, `tools/build_native_packages.py`, `tools/stage_release_root.sh`, `packaging/README.md`, `cmake/README.md` | Found | Release tooling exists; published asset proof still pending. |
| Benchmarks | `benchmarks/README.md`, `docs/BENCHMARKS.md`, benchmark history logs | Found | Current public evidence is reference-system throughput oriented. |
| Website specification pack | Present on a later commit/main-line search result, not on fetched `pre-release-staging` path | Branch-sensitive | It contains published-beta wording that may contradict beta-prep docs if merged back without reconciliation. |

## Roadmap seams that must not be assumed

| Roadmap concept | Current repo status | Required action before implementation |
| --- | --- | --- |
| Mode/family UX data model | Existing model metadata has family-like fields in `model_manager.cpp`, but full R2 mode contract not verified | Inspect model structs/headers and current manifest format; add compatibility tests first. |
| Fused selective execution planner | Planner exists; fused-family behavior not verified | Add failing planner tests before planner implementation. |
| Preview cache manifest | R2 schema exists in archive; repo implementation not verified | Add core cache contract and tests before GUI. |
| Segment preview timeline | Not verified in repo | Introduce cache/job/viewer layers in small patches. |
| Split-wipe viewer | Not verified in repo | Implement after cache core and frame access are stable. |
| Workstation UI shell | Current GUI exists; workstation shell not present | Preserve legacy advanced access and wire to shared core behavior. |
| CLI/GUI profile parity | CLI/GUI both exist; parity contract not verified | Create shared profile serialization tests before GUI-only behavior. |
| Temporal Forge ML/refiner path | R2 docs/specs only | Must remain experimental and disabled by default until Phase 1 contracts land. |
| Clean-room temporal architecture | R2 policy exists | Any SDK/source dependency must pass license/IP review before merge. |

## Missing seams / adapter candidates

| Missing seam | Needed by task | Proposed smallest adapter |
| --- | --- | --- |
| Canonical mode definition registry | Mode-first UI and CLI parity | Add a small data-only registry that maps mode IDs to default capabilities/families without changing execution first. |
| Family metadata validation boundary | Model consolidation/fusion | Extend manifest validation with backward-compatible defaults and focused tests. |
| Planner diagnostic representation | Fused planning and preview parity | Add serializable planned-pipeline output used by both CLI dry-run and GUI debug display. |
| Preview cache manifest writer/reader | Segment preview timeline | Add cache metadata before adding UI. |
| Segment preview job abstraction | Timeline rendering | Use existing job queue/recovery patterns if appropriate, but keep segment preview isolated from full render jobs. |
| Quality evidence capture path | Public proof | Add a documented workflow for before/after/failure-case artifacts before making strong quality claims. |

## Do not assume

- Do not assume preview distro packages are validated just because package tooling exists.
- Do not assume every backend visible in the tree is a public support promise.
- Do not assume a public beta exists until the GitHub Releases page and release assets are visible.
- Do not assume the R2 schemas are already implemented in the codebase.
- Do not assume Temporal Forge belongs on the release branch before Phase 1 contracts and gates are in place.
- Do not assume model-preparation subprocess hardening is closed until every subprocess call is classified.

## Local verification required

Before the first feature PR, rerun local discovery and update this file with exact line-level references:

```bash
rg -n "enum class StageKind|StageKind|stageKindFromString" include src tests
rg -n "class PipelinePlanner|PipelinePlanner|plan" include src tests
rg -n "ModelEntry|ModelCatalog|family|capabilities|manifest" include src tests models
rg -n "Backend|BackendManager|processVideoFile|runStage" include src tests
rg -n "preview|cache|timeline|QWidget|QGraphics|QImage" src/gui include src tests
rg -n "std::system|popen|system\(|python3|unzip|migraphx-driver|process_observer" src include tools tests
```
