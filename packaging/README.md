# Packaging Template Guide

This folder contains the packaging metadata templates for the native Linux package targets. The actual release logic lives in `tools/` and CMake; this folder is where the distro-specific packaging formats get their metadata.

## Folder map

| Path | What it contains |
| --- | --- |
| `arch/PKGBUILD.in` | Arch Linux package template |
| `debian/control.in` | Debian/Ubuntu package metadata template |
| `rpm/amd-video-enhancer.spec.in` | RPM spec template for Fedora/openSUSE/Rocky/AlmaLinux style packaging |
| `common/amd-video-enhancer.desktop` | desktop entry shared by package builds |

## How this folder fits the release pipeline

The packaging flow works roughly like this:

1. CMake and the helper scripts stage a canonical install tree
2. the staged payload is built around the app-private runtime under `/opt/amd-video-enhancer`
3. `tools/build_native_packages.py` uses the templates in this folder to emit distro-native package artifacts
4. public launchers stay thin while the real payload remains private and isolated inside the packaged tree

So this folder does not build packages by itself. It provides the distro-specific metadata that turns one staged payload into multiple native package formats.

## Why this matters

Packaging is part of the app's architecture here, not a decorative afterthought. The repo is trying to ship an AMD-heavy userspace stack in a controlled, repeatable way, and these templates are one layer of that work.

## Pair this with

- packaging workflow details: [`../docs/PACKAGING.md`](../docs/PACKAGING.md)
- helper scripts: [`../tools/README.md`](../tools/README.md)
- install/bundling templates: [`../cmake/README.md`](../cmake/README.md)
