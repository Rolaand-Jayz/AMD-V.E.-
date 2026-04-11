# Tools Guide

This folder contains the helper scripts and small utilities that make the repository practical to build, package, benchmark, and inspect. If `src/` is the engine room, `tools/` is the workbench.

## High-value scripts

| Tool | What it is for |
| --- | --- |
| `package_release.sh` | repeatable packaging entrypoint for runtime-portable, compiler-portable, and install-tree builds |
| `build_native_packages.py` | turn a staged release root into distro-native packages |
| `create_portable_bundle.py` | assemble a self-contained portable folder/archive |
| `stage_release_root.sh` | stage the canonical install root used by packaging |
| `stage_migraphx_runtime_overlay.sh` | overlay ABI-matched MiGraphX runtime sonames into a bundled MiGraphX prefix |
| `stage_migraphx_compiler_deps.sh` | stage ABI-matched compiler-side MiGraphX dependencies from the local package cache |
| `bundle_models.py` | collect and stage model assets |
| `compile_onnx_to_mxr.cpp` | helper utility for MiGraphX compilation sweeps |
| `bench_ave.sh` | run a benchmark against the built app |
| `bench_upscale_pair.sh` | run the paired benchmark snapshot used by the docs |
| `create_benchmark_clip.sh` | generate benchmark media input |
| `benchmark_model_matrix.py` / `benchmark_compiled_migraphx.py` | scripted benchmark helpers |

## The custom ROCm / MiGraphX story lives here

This repository does not just rely on whatever `/opt/rocm` happens to look like on a user's machine. Some of the most important “make AMD usable” work is in this folder:

- `package_release.sh` can build **runtime-portable** and **compiler-portable** bundles
- `stage_migraphx_runtime_overlay.sh` can stage the exact cached MiGraphX runtime SONAME set into a bundled prefix so the portable app matches the ABI it expects
- `stage_migraphx_compiler_deps.sh` can vendor ABI-matched compiler-side libraries like Abseil and Protobuf from the local package cache

In other words: this folder contains the practical glue that makes a customized MiGraphX toolchain shippable instead of merely interesting.

## If you are new, read in this order

1. `package_release.sh`
2. `build_native_packages.py`
3. `stage_migraphx_runtime_overlay.sh`
4. `stage_migraphx_compiler_deps.sh`
5. the benchmark helpers

That order tells the story from “how do I ship this?” to “how do I benchmark it?”

## Companion docs

- packaging deep dive: [`../docs/PACKAGING.md`](../docs/PACKAGING.md)
- debugging/validation playbook: [`../docs/migraphx_debugging_playbook.md`](../docs/migraphx_debugging_playbook.md)
- packaging template map: [`../packaging/README.md`](../packaging/README.md)
- install/bundle CMake helpers: [`../cmake/README.md`](../cmake/README.md)
