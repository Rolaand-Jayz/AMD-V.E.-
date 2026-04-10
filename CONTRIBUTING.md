# Contributing

## Release Status

## **OPEN ALPHA**

## **ONLY TESTED ON ARCH LINUX**

This project is currently an open alpha.

The app and release payloads have only been tested on:

- Arch Linux
- AMD Ryzen 7 7800X3D
- AMD Radeon RX 7900 GRE

All other distros, kernels, ROCm stacks, Mesa stacks, and GPU configurations are experimental.
Feedback is wanted, but reports outside the tested Arch hardware/software stack should be filed with full environment details.

## What To Contribute

- Reproducible bug reports
- Experimental distro packaging fixes
- ROCm / MiGraphX compatibility fixes
- GUI usability improvements
- Test coverage
- Documentation fixes

## Before You Start

1. Read [README.md](/home/rolaandjayz/Desktop/C++%20Video%20Enhancer/README.md) and [docs/PACKAGING.md](/home/rolaandjayz/Desktop/C++%20Video%20Enhancer/docs/PACKAGING.md).
2. Build with every backend flag enabled.
3. Keep changes free of stubs, placeholders, and fake-success paths.
4. Prefer narrow, reviewable pull requests.

## Build And Test

Use the full backend-enabled build only:

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

## Filing Issues

When filing an issue, include:

- Distro and version
- Kernel version
- GPU model
- ROCm version
- MiGraphX version
- FFmpeg version
- Whether the package was Arch, Debian/Ubuntu, RPM, or a source build
- Exact command or workflow that failed
- Full stderr or log tail
- Whether the problem reproduces on the tested Arch stack

Use the distro-report issue template when the problem is on a non-Arch system.

## Pull Requests

Pull requests should:

- Explain the user-facing problem
- Explain the technical fix
- Note packaging or distro impact when relevant
- Include tests when behavior changed
- Avoid unrelated cleanup

If your PR changes packaging, document:

- Which package format(s) changed
- Whether the app-private dependency closure changed
- Whether host-side requirements changed

## Code Style

- Keep all app code in `namespace ave`
- Use the existing naming and layout conventions from [AGENTS.md](/home/rolaandjayz/Desktop/C++%20Video%20Enhancer/AGENTS.md)
- Add comments only where the code would otherwise be hard to follow
- Do not add dead fallback paths or "temporary" fake completion

## Packaging Contributions

Native packages are built from one canonical Arch-origin staged install tree with bundled app-private userspace dependencies.

If you are working on packaging:

- Do not move bundled libraries into global linker paths
- Keep the app payload private under `/opt/amd-video-enhancer`
- Keep public entrypoints as thin wrappers or symlinks only
- Do not replace native packages with AppImage, Flatpak, or Snap

## Release Expectations

Until this project exits alpha:

- Arch Linux is the only claimed tested platform
- All other distro packages are experimental
- Regressions in packaging or startup isolation should be treated as high priority
