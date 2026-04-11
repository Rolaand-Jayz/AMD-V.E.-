# Releasing

## Release channel

The current public channel is a **GitHub beta prerelease**.

That is deliberate and should remain the default until validation broadens beyond
the primary reference system.

Current support statement:

- Primary verified system: Arch Linux
- Primary verified hardware: Ryzen 7 7800X3D + Radeon RX 7900 GRE
- Other distro packages are published as preview builds
- ROCm, Mesa, Vulkan, and MiGraphX compatibility still vary across distros and kernels

Do not market current tags as a general-availability release.

## Verified versus preview targets

Tested:

- Arch Linux
- Ryzen 7 7800X3D
- Radeon RX 7900 GRE

Preview targets:

- Ubuntu 24.04
- Ubuntu 22.04
- Debian 12
- Fedora 41
- openSUSE Leap 15.6
- openSUSE Tumbleweed
- Rocky Linux 9
- AlmaLinux 9

## Release strategy

1. Build the canonical staged release root on Arch Linux
2. Keep the app payload private under `/opt/amd-video-enhancer`
3. Keep public entrypoints thin under `/usr/bin`
4. Bundle the app-private userspace dependency closure with the payload
5. Repackage that exact staged root into native distro packages
6. Publish the compiler-portable archive that bundles the custom MiGraphX runtime/toolchain with the app
7. Publish local-install Arch packages, AUR handoff assets, distro-native packages, and checksum sidecars on GitHub as beta prerelease assets
8. Publish a consolidated `SHA256SUMS` manifest

Every staged release root should also ship:

- `LICENSE`
- `CHANGELOG.md`
- `SECURITY.md`
- packaging and release-operation docs under `share/doc/amd-video-enhancer/`

This project does not publish AppImage, Flatpak, or Snap builds.

## Commands

Stage the canonical release root:

```bash
./tools/stage_release_root.sh
```

Build native packages from the staged root:

```bash
python3 ./tools/build_native_packages.py \
  --staged-root ./dist/stage-root \
  --output-dir ./dist/native-packages
```

Each package build emits a matching `.sha256` sidecar next to the native package artifact.

Generate the Arch AUR handoff assets from the same staged root:

```bash
python3 ./tools/build_native_packages.py \
  --staged-root ./dist/stage-root \
  --output-dir ./dist/native-packages \
  --targets archlinux-aur \
  --version 0.1.0-beta.1
```

Build the beta portable archive with the bundled custom MiGraphX runtime/toolchain:

```bash
ARCHIVE_MODE=archive ./tools/package_release.sh compiler-portable
```

## GitHub Actions

The repository contains a native release workflow:

- `.github/workflows/release-native-packages.yml`

Workflow model:

1. A self-hosted Arch runner builds and stages the canonical release root
2. Downstream containerized jobs emit native packages for each distro target
3. The Arch release job emits the compiler-portable archive and the Arch AUR handoff assets
4. Package jobs emit per-artifact checksum sidecars
5. Tag pushes create a GitHub beta prerelease, generate a consolidated `SHA256SUMS`, and upload all release assets

## Release checklist highlights

Before publishing a beta prerelease, confirm:

- the canonical Arch stage root builds and passes tests
- package payloads include `LICENSE`, `CHANGELOG.md`, and `SECURITY.md`
- the portable archive includes the bundled custom MiGraphX runtime/toolchain instead of requiring a system MiGraphX install
- the Arch release includes both a local `pacman -U` package and an AUR handoff bundle
- release assets include checksum sidecars and `SHA256SUMS`
- README benchmark numbers match the latest recorded benchmark run
- support messaging matches the current verified system statement

## Support statement

Every release should state:

- published as a GitHub beta prerelease
- primarily verified on Arch Linux
- primarily verified on Ryzen 7 7800X3D + Radeon RX 7900 GRE
- other distro packages are preview builds
