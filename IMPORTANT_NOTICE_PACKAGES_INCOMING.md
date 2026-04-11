# Important Notice: packages incoming

This staging branch intentionally does **not** include generated package artifacts.

That means you will **not** see committed release outputs such as:

- Arch Linux package files (`.pkg.tar.zst`)
- AUR handoff bundles
- Debian/Ubuntu packages (`.deb`)
- RPM packages (`.rpm`)
- portable release archives
- generated checksum manifests and sidecar outputs from local packaging runs

## Why they are not in this branch

Those files are build outputs, not source-of-truth repository content.

This branch is being pushed as a **pre-release staging snapshot** of the code, docs, packaging workflow, and release preparation changes only.

## ETA

**Package assets are expected in less than 24 hours.**

Planned beta release assets include:

- local-install Arch package
- Arch AUR handoff bundle
- Debian/Ubuntu `.deb` packages
- RPM preview packages for the current target distros
- portable archive with the bundled custom MiGraphX runtime/toolchain

## Current hold point

Release asset publication is being held for the upcoming **icon/logo integration** pass.

Once the branding work lands, the release artifacts will be rebuilt from the updated staging state and published as the beta package set.
