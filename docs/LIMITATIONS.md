# Known Limitations

This file is the blunt version. Beta users should not have to discover these by accident.

## Release-state limitations

- The repository is still preparing its **first public GitHub beta prerelease**.
- No public release assets are published yet, so outsiders cannot inspect the final artifact set from the release page today.
- Public repo metadata and external website polish are still lighter than the long-term gold standard.

## Validation limitations

- The only verified primary environment today is **Arch Linux + Ryzen 7 7800X3D + Radeon RX 7900 GRE**.
- Preview distro targets are packaging targets, not broad support proof.
- Current benchmark evidence is still narrow and centered on one maintainer-verified reference system.
- Public visual-quality evidence is not yet as strong as the throughput/packaging story, so quality claims should stay narrow and specific.

## Runtime limitations

- First-run MiGraphX preparation/compilation can take minutes.
- The app emits compile progress during `migraphx-driver` work, but some preparation phases are still coarser than a perfect progress bar.
- Source-build model preparation paths still depend on external tools like `python3` and `unzip` being present in `PATH`.
- Packaged builds bundle app-private userspace dependencies, but they still rely on a working host AMD kernel/driver stack.

## Support-scope limitations

- Backend presence in the tree is broader than the current public support promise.
- Specialized or exploratory backend paths should not be treated as the default user promise just because they compile.
- Windows, macOS, and NVIDIA/CUDA workflows are out of scope.

## Beta-program limitations

- Community validation is still growing, so most public proof remains maintainer-verified.
- Outside reports are valuable, but they must include enough detail to be comparable and actionable.

## What to read next

- [`RELEASE_STATUS.md`](./RELEASE_STATUS.md)
- [`SUPPORT_TIERS.md`](./SUPPORT_TIERS.md)
- [`VALIDATION_AND_EVIDENCE.md`](./VALIDATION_AND_EVIDENCE.md)
- [`BETA_TESTING_PROGRAM.md`](./BETA_TESTING_PROGRAM.md)
