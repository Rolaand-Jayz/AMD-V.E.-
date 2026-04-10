# Releasing

## Release Channel

The current public release channel is **open alpha**.

That is intentional.

Reasons:

- The only tested system is Arch Linux
- The only claimed tested hardware is Ryzen 7 7800X3D + Radeon RX 7900 GRE
- Native packages for other distros are produced, but they are experimental
- ROCm, Mesa, Vulkan, and MiGraphX compatibility still vary heavily across distros

Do not market current releases as beta.

## Tested Versus Experimental

Tested:

- Arch Linux
- Ryzen 7 7800X3D
- Radeon RX 7900 GRE

Experimental:

- Ubuntu 24.04
- Ubuntu 22.04
- Debian 12
- Fedora 41
- openSUSE Leap 15.6
- openSUSE Tumbleweed
- Rocky Linux 9
- AlmaLinux 9

## Release Strategy

1. Build the canonical staged release root on Arch Linux
2. Keep the app payload private under `/opt/amd-video-enhancer`
3. Keep public entrypoints thin under `/usr/bin`
4. Bundle the app-private userspace dependency closure with the payload
5. Repackage that exact staged root into native distro packages
6. Publish all packages on GitHub as prerelease assets

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

## GitHub Actions

The repository contains a native release workflow:

- `.github/workflows/release-native-packages.yml`

Workflow model:

1. A self-hosted Arch runner builds and stages the canonical release root
2. Downstream containerized jobs emit native packages for each distro target
3. Tag pushes create a GitHub prerelease and upload all package assets

## Support Statement

Every release should state:

- open alpha
- only tested on Arch Linux
- only tested on Ryzen 7 7800X3D + Radeon RX 7900 GRE
- all other distro packages are experimental
