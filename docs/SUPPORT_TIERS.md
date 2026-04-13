# Support Tiers

Code presence is **not** the same thing as release support. This file is the support promise.

## Platform tiers

| Tier | Scope | Current statement |
| --- | --- | --- |
| Verified primary | Arch Linux + Ryzen 7 7800X3D + Radeon RX 7900 GRE | This is the only environment currently treated as verified for the beta-prep branch |
| Preview target | Ubuntu 24.04, Ubuntu 22.04, Debian 12, Fedora 41, openSUSE Leap 15.6, openSUSE Tumbleweed, Rocky 9, AlmaLinux 9 | Package targets exist or are planned in the release tooling, but these are still validation targets rather than broad compatibility proof |
| Experimental / manual | Other AMD Linux distros, custom Mesa/ROCm/kernel mixes, mixed iGPU+dGPU setups, bespoke source-build environments | Reports are welcome, but these paths require full environment detail and should not be marketed as stable support yet |
| Out of scope | Windows, macOS, NVIDIA/CUDA workflows, AppImage/Flatpak/Snap | Not part of the current product promise |

## Backend tiers

| Tier | Backend/path | Current statement |
| --- | --- | --- |
| Verified primary | MiGraphX inference on the reference system | This is the main AMD-first release path |
| Supported fallback on the verified system | FFmpeg filter-chain fallback | Always available media fallback path when AI acceleration cannot be used for a given operation |
| Preview target | ROCm/HIP ONNX Runtime fallback, Vulkan Compute, NCNN Vulkan | Real code paths exist, but the public release promise is narrower than simple code presence |
| Experimental / manual | GLSL shader backend, VapourSynth-oriented/specialized paths, custom unsupported stack combinations | Present for technical depth or compatibility exploration, not as a broad release promise |

## Packaging tiers

| Tier | Packaging surface | Current statement |
| --- | --- | --- |
| Planned primary beta asset | Arch package + portable bundle + checksum sidecars | Intended for the first public beta release once published |
| Preview target | AUR handoff bundle, Debian/Ubuntu packages, RPM-family packages | Packaging reach exists in tooling, but these assets still need public validation on target systems |
| Not proof by itself | Any package emitted by the build pipeline | A package format existing does not widen verified compatibility on its own |

## Bundled vs host-required boundary

### Bundled with the planned beta payload

- app binaries and launchers
- app-private runtime libraries
- app-private `ffmpeg` / `ffprobe`
- bundled custom MiGraphX runtime/toolchain when the beta asset requires it

### Still required from the host

- Linux kernel and AMD GPU driver stack (`amdgpu` / KFD access)
- working Vulkan driver support
- functioning ROCm-capable environment for MiGraphX / HIP execution paths
- source-build-only helper tools like `python3` and `unzip` when those flows are used outside packaged builds

If the host GPU stack is broken, the app cannot bundle its way around that.
