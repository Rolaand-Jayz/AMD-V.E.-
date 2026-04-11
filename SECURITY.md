# Security Policy

## Supported versions

Security fixes are best-effort for:

- the current `main` branch
- the latest published prerelease tag

Older prereleases and locally patched builds may receive guidance, but they are not guaranteed to receive coordinated fixes.

## Reporting a vulnerability

Please do **not** post sensitive vulnerability details in a public GitHub issue.

Preferred reporting path:

1. Use GitHub's private vulnerability reporting for this repository if it is enabled.
2. If private reporting is not enabled, contact the maintainer through the repository owner's GitHub profile or another private maintainer contact channel before public disclosure.

When reporting, include:

- the affected commit, branch, or release tag
- the Linux distribution and kernel version
- ROCm, Vulkan, and FFmpeg versions when relevant
- clear reproduction steps or a proof-of-concept
- impact assessment and any known mitigations

## Disclosure expectations

- Please allow reasonable time for triage, confirmation, and remediation before public disclosure.
- We will try to acknowledge credible reports promptly and keep the reporter updated on fix status.
- Once a fix is available, remediation notes should be published in release notes, the changelog, or a dedicated advisory.

## Security scope notes

This project processes local media files and can download model artifacts. Security-sensitive areas include:

- model download and archive extraction flows
- shell-spawned media tooling and compiler tooling
- temporary-file handling
- bundled runtime dependencies and release artifacts
- custom model manifests and user-supplied stage parameters

If you are unsure whether something is security-relevant, report it privately first.
