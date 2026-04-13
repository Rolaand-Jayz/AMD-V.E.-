# Release Status

## Current public state

AMD Video Enhancer has published its first public **GitHub beta prerelease** as `v0.1.0-beta.1` from `main`.

As of **2026-04-13**, the public GitHub Releases page should show that beta prerelease and its downloadable assets. The repository should describe the beta as published because outsiders can now inspect the release page, release notes, and checksum assets directly.

## What is true today

- the codebase contains real packaging, staging, release, benchmark, and validation machinery
- the primary verified stack is **Arch Linux + Ryzen 7 7800X3D + Radeon RX 7900 GRE**
- the primary verified backend path is **MiGraphX + FFmpeg media pipeline** on that reference system
- the current beta release relies on a **bundled custom MiGraphX runtime/toolchain** because the required upstream behavior is not yet available in the stock system path
- package recipes and public beta assets exist for Arch/AUR, Debian/Ubuntu, Fedora/openSUSE, and RHEL-family RPM targets

## What is not true yet

- package targets do **not** equal broad compatibility proof
- the public evidence stack is still narrower than the long-term gold standard
- the public release surface should not imply that preview distro targets are already fully validated

## Current truth vs next-proof state

| Surface | Current truth | Next proof threshold |
| --- | --- | --- |
| Release page | Public GitHub beta prerelease `v0.1.0-beta.1` with inspectable assets | Keep release notes, assets, and checksums aligned with docs |
| Verified platform | Arch Linux on the reference system | Same reference system, plus any added proof surfaced explicitly |
| Preview package targets | Published with clear preview/validation labels | Broaden only when target-system validation expands |
| Bundled MiGraphX story | Real, but easy to miss if you skim | Impossible to miss in README, release notes, and packaging docs |
| Public evidence | Maintainer benchmark snapshot and test surface | Same plus cleaner external validation and quality evidence |

## What the public beta must say clearly

- it is a **GitHub beta prerelease**, not a general-availability release
- Arch Linux on the reference system is the only **verified primary** environment today
- preview distro assets are **validation targets**, not broad compatibility guarantees
- the bundled custom MiGraphX path is **deliberate** and part of the support boundary
- host AMD kernel/driver stack requirements are still outside the bundle

## Read these next

- [`SUPPORT_TIERS.md`](./SUPPORT_TIERS.md)
- [`LIMITATIONS.md`](./LIMITATIONS.md)
- [`VALIDATION_AND_EVIDENCE.md`](./VALIDATION_AND_EVIDENCE.md)
- [`BETA_TESTING_PROGRAM.md`](./BETA_TESTING_PROGRAM.md)
- [`PACKAGING.md`](./PACKAGING.md)
- [`RELEASING.md`](./RELEASING.md)
