# CMake Helper Guide

This folder contains the install-time and packaging-time CMake templates that help turn the built binaries into a usable app layout. These files are not the main build graph; they are the support machinery that creates launchers, bundles runtime dependencies, and stages extra assets.

## File map

| File | Purpose |
| --- | --- |
| `ave-launch.sh.in` | launcher wrapper template for the installed CLI and GUI entrypoints |
| `ave-migraphx-driver.sh.in` | wrapper template for the bundled MiGraphX compiler driver |
| `install_bundled_migraphx_deps.cmake.in` | install-time script template for bundling MiGraphX runtime/compiler dependencies |
| `install_model_bundle.cmake.in` | install-time script template for staging bundled models |
| `install_qt_plugins.cmake.in` | install-time script template for bundling Qt plugins for the GUI |
| `install_runtime_deps.cmake.in` | install-time script template for copying non-system runtime libraries into the app tree |

## Why this folder matters

This project is trying to ship more than a naked binary. It wants:

- wrapper launchers that resolve the right private runtime paths
- bundled runtime libraries for portable/package builds
- optional bundled MiGraphX tooling
- optional bundled models
- optional bundled Qt plugin trees

That is exactly the kind of work these templates support.

## The ROCm / MiGraphX angle

If you care about the custom AMD stack story, this folder is part of it. These templates participate in bundling the MiGraphX runtime and, when enabled, the compiler-side tooling used for `.mxr` generation.

## Pair this with

- root build/install options: [`../CMakeLists.txt`](../CMakeLists.txt)
- packaging workflow: [`../docs/PACKAGING.md`](../docs/PACKAGING.md)
- helper scripts: [`../tools/README.md`](../tools/README.md)
- package templates: [`../packaging/README.md`](../packaging/README.md)
