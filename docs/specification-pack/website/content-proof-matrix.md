# Content Proof Matrix

Use this matrix when writing homepage or downloads-page copy.

| Topic | Allowed public wording | Proof source | Overclaim to avoid |
| --- | --- | --- | --- |
| Current release state | `v0.1.0-beta.1` is the current public beta prerelease | GitHub release page + release docs | claiming a GA release or implying unreleased assets |
| Verified platform | verified on Arch Linux + Ryzen 7 7800X3D + Radeon RX 7900 GRE | `README.md`, `docs/RELEASE_STATUS.md`, `docs/SUPPORT_TIERS.md` | implying broad Linux-wide validation |
| Preview packages | preview package assets are available for the listed target distros | release assets + support tiers | presenting preview packages as guaranteed support |
| Downloads integrity | checksums are published for release assets | GitHub release assets + `SHA256SUMS` | claiming signed installers or stronger integrity guarantees than exist |
| AMD-first angle | AMD-first Linux video enhancement pipeline using ROCm, MiGraphX, HIP, and Vulkan | repo README + implementation docs | claiming universal AMD compatibility or mature support across every AMD stack |
| Performance evidence | benchmark and validation docs document the current evidence scope | `docs/BENCHMARKS.md`, `docs/VALIDATION_AND_EVIDENCE.md` | implying broad benchmark leadership or generalized speed claims |
| Why the project matters | public MiGraphX / AMD-first consumer examples are rare, and this repo makes that path visible | README + `docs/WHY_THIS_PROJECT_MATTERS.md` | claiming absolute uniqueness that cannot be proven |

## Copy rule

If a line cannot be mapped to one of the proof sources above, either:

- add proof first
- narrow the wording
- or cut the line
