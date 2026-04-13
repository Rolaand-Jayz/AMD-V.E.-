# Packaging

## Release status

This branch is preparing the first public **GitHub beta prerelease**.

As of 2026-04-12, no public prerelease is published yet. This document describes the asset set and packaging behavior intended for that first beta publication.

- Primary verified stack: Arch Linux on Ryzen 7 7800X3D + Radeon RX 7900 GRE
- Other distro package formats are preview targets that still need target-system validation
- Package payloads now include the project license, changelog, security policy, and release-operation docs

Before reading this page as a support promise, also read:

- [`RELEASE_STATUS.md`](./RELEASE_STATUS.md)
- [`SUPPORT_TIERS.md`](./SUPPORT_TIERS.md)
- [`LIMITATIONS.md`](./LIMITATIONS.md)

The app now supports three packaging layers:

### Standard installed layout

- `bin/ave` and `bin/ave_gui` are launcher scripts.
- Real binaries live under `libexec/ave/`.
- Bundled runtime libraries live under `lib/ave/runtime/`.
- Optional MiGraphX compiler tooling lives under `lib/ave/migraphx/`.
- Bundled models live under `share/ave/models/` when model bundling is enabled.

### Portable extracted layout

- The install tree is staged into a single folder.
- That folder can be archived as `tar.gz`, extracted anywhere, and run in place.
- The launchers resolve all bundled paths relative to their own location.
- The bundle includes a root-level `PORTABLE_RUNTIME_NOTES.txt` file that documents bundled paths, MiGraphX tooling, and the remaining host-side GPU driver requirements.

### Native distro packages

- Arch Linux `.pkg.tar.zst`
- Arch Linux AUR handoff assets for `amd-video-enhancer-bin`
- Debian/Ubuntu `.deb`
- Fedora/openSUSE/Rocky `.rpm`
- All native packages are built from one canonical Arch-origin staged release root.
- The app payload remains private under `/opt/amd-video-enhancer`.
- Public entrypoints remain thin under `/usr/bin`.
- Each build emits a native package plus a matching SHA-256 checksum sidecar.

## Build

Always configure with all backends enabled:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DAVE_HAVE_CURL=ON \
  -DAVE_HAVE_MIGRAPHX=ON \
  -DAVE_HAVE_HIP=ON \
  -DAVE_HAVE_VULKAN=ON \
  -DAVE_HAVE_NCNN=ON
cmake --build build -j
```

## Install

```bash
cmake --install build --prefix /path/to/install-root
```

Important packaging options:

- `AVE_INSTALL_BUNDLED_MIGRAPHX=ON`
  - Install the custom MiGraphX compiler toolchain under the app tree.
  - The app runtime itself is bundled through `AVE_INSTALL_RUNTIME_DEPS`, so the launcher does not inject this toolchain into the app process.
- `AVE_INSTALL_BUNDLED_MIGRAPHX_COMPILER=ON`
  - Install `migraphx-driver` plus the MiGraphX ONNX/TF sidecars.
  - Leave this `OFF` for a runtime-only portable bundle when you want a smaller archive that runs inference but does not compile new `.mxr` artifacts on the target machine.
- `AVE_BUNDLED_MIGRAPHX_EXTRA_LIBRARY_DIRS=/path/one;/path/two`
  - Extra directories searched when the installer closes transitive dependencies for the bundled custom MiGraphX compiler runtime.
  - Use this when your custom `migraphx-driver` depends on side libraries that live outside the custom prefix.
  - `tools/package_release.sh compiler-portable` now auto-stages matching ABI-pinned Abseil and Protobuf side libraries from the local package cache into the build tree before packaging.
  - Set this manually when you are packaging on a different machine or have a bespoke dependency cache layout.
- `AVE_STRICT_BUNDLED_MIGRAPHX=ON`
  - Fail packaging if the bundled MiGraphX runtime is missing transitive shared libraries.
  - This is the safe default for portable archives because compile-enabled MiGraphX bundles must be self-contained.
- `AVE_PREFER_BUNDLED_MIGRAPHX_FOR_BUILD=ON`
  - Build the app itself against `AVE_BUNDLED_MIGRAPHX_PREFIX`.
  - Leave this `OFF` unless that prefix is a complete MiGraphX runtime suitable for linking the app, not just an external compiler toolchain.
- `AVE_BUNDLED_MIGRAPHX_PREFIX=/path/to/custom/migraphx`
  - Release packaging should set this explicitly when a bundled custom MiGraphX runtime/toolchain is part of the artifact contract.
  - Do not rely on a hidden maintainer-local home-directory path when preparing public release assets.
- `AVE_INSTALL_BUNDLED_MODELS=ON`
  - Download and stage the model catalog into the app tree.
- `AVE_INSTALL_RUNTIME_DEPS=ON`
  - Copy non-system runtime shared libraries into `lib/ave/runtime`.

Release packages also bundle `ffmpeg` and `ffprobe` into the private app tools directory so packaged installs do not depend on host media-tool binaries for the primary execution path.

## Portable Bundle

Create an extracted self-contained folder:

```bash
cmake --build build --target portable_stage
```

Create an extracted folder plus a `tar.gz` archive:

```bash
cmake --build build --target portable_bundle
```

Outputs default to:

```text
build/dist/amd-video-enhancer-portable-<system>-<arch>/
build/dist/amd-video-enhancer-portable-<system>-<arch>.tar.gz
```

The bundle name can be overridden with `AVE_PORTABLE_BUNDLE_NAME`.

## Packaging Script

Use the helper script when you want a repeatable packaging workflow without
hand-editing CMake flags:

```bash
tools/package_release.sh runtime-portable
ARCHIVE_MODE=archive tools/package_release.sh runtime-portable
AVE_BUNDLED_MIGRAPHX_PREFIX=/path/to/custom/migraphx \
EXTRA_MIGRAPHX_LIBRARY_DIRS="/vendor/lib;/custom/lib" \
tools/package_release.sh compiler-portable
INSTALL_PREFIX="$PWD/dist/install-root" tools/package_release.sh install-tree
```

Profiles:

1. `runtime-portable`
   - portable app bundle
   - bundles the app runtime and linked libraries
   - does not bundle the custom MiGraphX compiler toolchain

2. `compiler-portable`
   - portable app bundle plus custom MiGraphX compiler tooling
   - uses `AVE_BUNDLED_MIGRAPHX_EXTRA_LIBRARY_DIRS` when set
   - auto-stages the known ABI-matched compiler-side dependencies from `/var/cache/pacman/pkg` when available
   - overlays the bundled MiGraphX runtime tree with the exact cached SONAME set required by the app when that cached package is available
   - excludes the bundled MiGraphX/ROCm stack from the generic app dependency scan so the portable bundle does not accidentally fall back to host `/opt/rocm` libraries
   - fails fast if the compiler-side dependency closure is incomplete
   - only use this profile when the custom MiGraphX compiler prefix also ships
     the exact ABI-matched side libraries it was linked against

3. `install-tree`
   - standard installed layout under `INSTALL_PREFIX`
   - useful for system packaging or integration into a larger installer

## Native Packages

Stage the canonical release root:

```bash
./tools/stage_release_root.sh
```

Build native packages from that staged root:

```bash
python3 ./tools/build_native_packages.py \
  --staged-root ./dist/stage-root \
  --output-dir ./dist/native-packages
```

The packaging helper writes one `.sha256` file per produced artifact. GitHub prereleases additionally publish a consolidated `SHA256SUMS` manifest.

For Arch, the packaging helper can also emit:

- a staged-root release asset tarball for the bundled `/opt/amd-video-enhancer` payload
- an AUR handoff archive containing `PKGBUILD`, `.SRCINFO`, and submission notes for `amd-video-enhancer-bin`

Default targets:

- `archlinux`
- `ubuntu-24.04`
- `ubuntu-22.04`
- `debian-12`
- `fedora-41`
- `opensuse-leap-15.6`
- `opensuse-tumbleweed`
- `rocky-9`
- `almalinux-9`

The native packages all carry the same private payload layout and differ only in package manager metadata and containerized release validation path.

## Installed documentation

Release packages install:

- `LICENSE` under the app license directory
- `README.md`
- `CHANGELOG.md`
- `SECURITY.md`
- `docs/BENCHMARKS.md`
- `docs/BETA_TESTING_PROGRAM.md`
- `docs/LIMITATIONS.md`
- `docs/PACKAGING.md`
- `docs/RELEASE_STATUS.md`
- `docs/RELEASING.md`
- `docs/SUPPORT_TIERS.md`
- `docs/VALIDATION_AND_EVIDENCE.md`

## Verification

Portable bundle creation now verifies runtime dependency closure after staging:

- installed executables are checked with `ldd`
- bundled `ffmpeg` and `ffprobe` binaries are checked with `ldd` when present
- bundled MiGraphX drivers are checked with `ldd` when compiler tools are present
- bundled MiGraphX shared libraries are checked with `ldd`, not just the app binaries
- portable runtime bundling now seeds MiGraphX runtime sonames explicitly instead of depending only on generic dependency discovery

The launchers now keep the bundled MiGraphX compiler toolchain isolated from the app runtime. The app resolves its own shared libraries from `lib/ave/runtime`, while `AVE_BUNDLED_MIGRAPHX_PREFIX` is exported only so model compilation flows can find the bundled driver/toolchain when requested.

Portable bundles intentionally ship the user-space dependency closure, but they cannot bundle the host kernel-side AMD stack. A target machine still needs a working `amdgpu`/KFD environment and GPU device access for ROCm/MiGraphX execution.

If the custom MiGraphX prefix is incomplete, packaging fails with the unresolved sonames instead of producing a broken archive.

The same host-side limitation applies to native packages. These packages isolate bundled userspace dependencies from the host, but they still rely on a working host AMD kernel/driver stack.

Packaging reach is still not the same thing as verified compatibility. A package format existing in the tree or in a future release does not widen the verified support statement by itself.

## Portable Profiles

### Full compile-enabled portable bundle

- Configure with `AVE_INSTALL_BUNDLED_MIGRAPHX=ON` and `AVE_INSTALL_BUNDLED_MIGRAPHX_COMPILER=ON`.
- Requires a custom MiGraphX prefix whose compiler binaries and ONNX/TF/Python sidecars have a closed shared-library dependency set.
- If some compiler-side libraries live outside that prefix, point `AVE_BUNDLED_MIGRAPHX_EXTRA_LIBRARY_DIRS` at the directories containing the exact required sonames.
- In practice this usually includes versioned Abseil, Protobuf, and `utf8_validity` libraries that match the custom MiGraphX compiler build exactly. If those sonames are not present, packaging stops instead of emitting a broken archive.

### Runtime-only portable bundle

- Configure with `AVE_INSTALL_BUNDLED_MIGRAPHX=OFF` and `AVE_INSTALL_BUNDLED_MIGRAPHX_COMPILER=OFF`, or use `tools/package_release.sh runtime-portable`.
- Bundles the app-linked MiGraphX runtime needed to load/evaluate existing `.mxr` artifacts.
- This is the safest portable profile when you want the app to extract anywhere and run without depending on host ROCm package updates.
- This is the lighter self-contained archive profile when you want users to extract the app and run existing compiled artifacts immediately.

### Beta-release portable bundle

- Use `tools/package_release.sh compiler-portable` with `ARCHIVE_MODE=archive`.
- This is the beta-release profile when the bundled custom MiGraphX runtime/compiler must ship with the app instead of relying on a system MiGraphX install.
- The produced archive can be unpacked into a folder and run in place, with checksum sidecars emitted next to the archive.
