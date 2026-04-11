# Contributing

## Release Status

## **BETA PRERELEASE**

## **ONLY TESTED ON ARCH LINUX**

This project is currently a beta prerelease.

The app and release payloads have only been tested on:

- Arch Linux
- AMD Ryzen 7 7800X3D
- AMD Radeon RX 7900 GRE

All other distros, kernels, ROCm stacks, Mesa stacks, and GPU configurations are experimental.
Feedback is wanted, but reports outside the tested Arch hardware/software stack should be filed with full environment details.

## Why contributing here matters

This is not just another app repo asking for bug fixes.

This project lives in a part of the software world where public examples are still thin: AMD-first consumer-facing video enhancement, ROCm-heavy deployment, and especially MiGraphX used as a serious application backend. That scarcity matters technically, because there are fewer examples to learn from. It also matters strategically, because this repo pushes directly against the comforting claim that AI can only copy from mature, over-documented ecosystems.

To be precise: this document is **not** claiming nobody has ever built anything similar in private. It is saying that, from the public side, there has not been a rich trail of MiGraphX-powered consumer video-enhancer projects to imitate. A zero-knowledge vibe coder and AI still managed to push a real implementation into existence in that under-documented territory.

That should matter to contributors for two reasons:

1. every contribution here improves a real tool for AMD users who are usually underserved
2. every contribution here also makes an under-documented technical space more visible, more reproducible, and harder to dismiss as “AI can't go there yet”

If someone's professional comfort depends on the idea that thin docs and sparse examples still form a protective moat around real engineering work, this repository is not a comforting artifact. It is evidence that the moat is already shallower than many people want to admit.

## What To Contribute

- Reproducible bug reports
- Experimental distro packaging fixes
- ROCm / MiGraphX compatibility fixes
- GUI usability improvements
- Test coverage
- Documentation fixes

## Before You Start

1. Read [README.md](/home/rolaandjayz/Desktop/C++%20Video%20Enhancer/README.md) and [docs/PACKAGING.md](/home/rolaandjayz/Desktop/C++%20Video%20Enhancer/docs/PACKAGING.md).
2. Read [docs/WHY_THIS_PROJECT_MATTERS.md](/home/rolaandjayz/Desktop/C++%20Video%20Enhancer/docs/WHY_THIS_PROJECT_MATTERS.md) if you want the strategic context for why this repo exists.
3. Build with every backend flag enabled.
4. Keep changes free of stubs, placeholders, and fake-success paths.
5. Prefer narrow, reviewable pull requests.

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

Until this project exits beta:

- Arch Linux is the only claimed tested platform
- All other distro packages are experimental
- Regressions in packaging or startup isolation should be treated as high priority
