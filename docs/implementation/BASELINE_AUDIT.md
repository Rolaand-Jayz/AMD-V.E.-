# Baseline Audit

> Generated on 2026-05-05 from the R2 handoff archive, `amd-ve-pre-release-staging-audit-and-gold-standard(1).md`, and remote GitHub connector inspection.
>
> This is a Phase 0 baseline. It is intentionally conservative. It records what is visible and what still needs local executable proof.

## Repo state

| Field | Current value |
| --- | --- |
| Repository | `Rolaand-Jayz/AMD-VE` |
| Branch targeted | `pre-release-staging` |
| Public repo state observed | Public repository, default branch `main`; staging branch exists |
| Last remotely observed staging state | README and canonical docs available on `pre-release-staging` |
| Dirty state | Not locally cloned; no worktree state available |
| Agent/tool | ChatGPT + GitHub connector + uploaded R2 archive |
| Code changes in this pass | None. Documentation/audit scaffolding only. |

## Source inputs used

- Uploaded audit: `amd-ve-pre-release-staging-audit-and-gold-standard(1).md`
- Uploaded handoff archive: `amd-ve-temporal-forge-roadmap-conditional-handoff-r2(2).tar`
- Uploaded checksum file: `amd-ve-temporal-forge-roadmap-conditional-handoff-r2(2).sha256`
- Remote repo docs inspected through GitHub connector:
  - `README.md`
  - `CONTRIBUTING.md`
  - `docs/RELEASE_STATUS.md`
  - `docs/SUPPORT_TIERS.md`
  - `docs/LIMITATIONS.md`
  - `docs/VALIDATION_AND_EVIDENCE.md`
  - `docs/BETA_TESTING_PROGRAM.md`
  - `docs/PACKAGING.md`
  - `tests/README.md`
  - `src/README.md`
  - `src/gui/README.md`
  - `src/model_manager.cpp`

## R2 archive integrity

The uploaded R2 tarball hash was manually checked against the checksum line embedded in the uploaded `.sha256` file.

| Artifact | Observed status |
| --- | --- |
| R2 tarball | Hash matched expected `ca8e14cf93cf4dae7d6b37dcd4ffc6acb4ee8d3844fd82c22d3dd62c9e0ce122` |
| Embedded checksum paths | The checksum file referenced original non-`(2)` filenames, so `sha256sum -c` could not be used directly after upload renaming |
| Archive extraction | Successful in the analysis container |
| Manifest scope | Archive exposed roadmap docs, agent tasks, schemas, fixtures, tests, and templates |

## Release truth baseline

The staging README now states the important release facts directly at the front door:

- `pre-release-staging` is preparing the first public GitHub beta prerelease.
- No public beta prerelease is published on the GitHub Releases page yet.
- The verified primary system is Arch Linux + Ryzen 7 7800X3D + Radeon RX 7900 GRE.
- The primary verified backend path is MiGraphX inference with the FFmpeg media pipeline on that reference system.
- Planned beta packages bundle a custom MiGraphX runtime/toolchain because required upstream behavior is not yet available in the stock system path.
- Package targets are broader than verified compatibility; packaging reach is not support proof.

This directly addresses the highest-risk audit theme: public trust through an aligned truth surface.

## Canonical release-facing docs baseline

| Doc | Baseline finding |
| --- | --- |
| `docs/RELEASE_STATUS.md` | Correctly frames the branch as beta-prep, says no public beta prerelease is published yet, and warns that package targets are not support proof. |
| `docs/SUPPORT_TIERS.md` | Separates verified primary, preview target, experimental/manual, and out-of-scope environments and backends. |
| `docs/LIMITATIONS.md` | Centralizes release, validation, runtime, support-scope, and beta-program limitations. |
| `docs/VALIDATION_AND_EVIDENCE.md` | Separates maintainer-verified evidence, community-submitted evidence, and non-public proof. |
| `docs/BETA_TESTING_PROGRAM.md` | Defines outside validation intake requirements for environment, benchmark, and quality reports. |
| `docs/PACKAGING.md` | Explains package profiles, bundled MiGraphX behavior, runtime dependency closure, host driver limits, and preview package scope. |
| `CONTRIBUTING.md` | Front-loads release status and verified platform constraints for contributors. |

## Build baseline

No local build was run in this pass because the execution environment could not clone GitHub directly and does not have the AMD/ROCm reference hardware stack.

The repo-documented release build command is:

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

### Required local follow-up

The next implementing agent must run the build on the real target workstation and update this file with:

- exact commit hash
- OS/kernel/driver/ROCm/MiGraphX/FFmpeg/CMake/compiler versions
- full configure command
- build result
- test result
- package-smoke result, if packaging is touched

## Test baseline

Remote docs say the current visible test suite covers:

- planner ordering
- tensor contracts
- runtime diagnostics and runtime paths
- model-manager profile behavior
- video probing
- process loops, observers, and progress reporting
- frame I/O paths
- FFmpeg runner behavior
- job queue and recovery logic
- telemetry
- MiGraphX backend behavior
- Vulkan/HIP interop behavior

The same docs also state what this does **not** prove yet:

- no broad golden-clip media regression suite across every backend/model class yet
- package target breadth is not compatibility proof
- output quality needs separate public evidence beyond green subsystem tests

## CLI baseline

| Area | Current baseline |
| --- | --- |
| CLI entrypoint | `src/main.cpp`, `src/cli.cpp` per `src/README.md` and GitHub search |
| CLI binary | `ave` per README examples |
| Backend listing | `./build/ave --list-backends` documented |
| Dry-run preview | README documents `--dry-run` pipeline preview examples |
| Required local validation | Capture `./build/ave --help`, `./build/ave --list-backends`, and at least one dry-run output after local build |

## GUI baseline

| Area | Current baseline |
| --- | --- |
| GUI entrypoint | `src/gui/main_gui.cpp` |
| Main window | `src/gui/main_window.cpp` / `.hpp` |
| Supporting widgets/dialogs | `filter_browser`, `model_manager_dialog`, `settings_dialog`, `toggle_switch` |
| GUI binary | `ave_gui` per README examples |
| Required local validation | Launch GUI on target system, capture screenshot or notes, verify it drives shared core pipeline behavior |

## Backend baseline

| Backend/path | Support tier in docs | Implementation surface observed |
| --- | --- | --- |
| MiGraphX | Verified primary on reference system | `src/backends/migraphx_backend.cpp`, `include/ave/backends/migraphx_backend.hpp`, `src/model_manager.cpp` |
| FFmpeg fallback | Supported fallback on verified system | `src/ffmpeg_runner.cpp`, media pipeline files per `src/README.md` |
| ROCm/HIP ONNX Runtime | Preview target | `src/backends/rocm_hip_backend.cpp`, `include/ave/backends/rocm_hip_backend.hpp` |
| Vulkan Compute | Preview target | `src/backends/vulkan_compute_backend.cpp` |
| NCNN Vulkan | Preview target | `include/ave/backends/ncnn_vulkan_backend.hpp`; implementation should be confirmed locally |
| GLSL shader / VapourSynth | Experimental/manual | `src/backends/glsl_shader_backend.cpp`, `src/backends/vapoursynth_backend.cpp` observed through search |

## Model-preparation baseline

`src/model_manager.cpp` contains substantial model lifecycle behavior, including:

- custom manifest parsing
- HTTPS download validation
- safe filename validation
- secure temporary file/directory helpers
- MiGraphX driver resolution
- compiled artifact naming and discovery
- compile profile behavior
- environment override handling

The old audit flagged shell/PATH risk in model-preparation paths. Remote inspection shows significant hardening exists, but this pass did not fully prove every subprocess path is non-shell and argument-safe.

Required follow-up:

```bash
rg -n "std::system|popen|system\(|/bin/sh|python3|unzip|migraphx-driver|process_observer" src include tools tests
```

Then classify each subprocess path as:

- bounded/argument-vector safe
- PATH-dependent but documented
- shell-dependent and needs hardening
- test-only acceptable

## Packaging baseline

Packaging docs describe:

- standard installed layout
- portable extracted layout
- native distro packages
- runtime-portable profile
- compiler-portable profile
- install-tree profile
- explicit `AVE_BUNDLED_MIGRAPHX_PREFIX`
- strict bundled MiGraphX dependency closure checks
- app-private runtime libraries
- bundled `ffmpeg` / `ffprobe`
- remaining host AMD kernel/driver requirements

The packaging story is much clearer than the original audit baseline, but it still needs release-asset proof once the public beta is actually published.

## Release blockers before public beta

| Blocker | Why it blocks |
| --- | --- |
| Local build/test results missing from this file | Phase 0 is not executable proof until the target workstation validates it. |
| Public release assets absent or unpublished | Docs correctly say no public beta exists yet; release claims must stay beta-prep until assets exist. |
| Output-quality proof still thin | Beta can ship, but claims must stay conservative until before/after and failure-case evidence exists. |
| Subprocess/PATH hardening not fully closed | The original audit flagged this; local code review must classify every path before calling it closed. |

## Baseline status

Phase 0 scaffolding is complete.

Executable validation is still pending.

No feature implementation should begin until this file is updated with local build/test evidence from the actual implementation machine.
