# Audit Closure Matrix

> Generated on 2026-05-05 from `amd-ve-pre-release-staging-audit-and-gold-standard(1).md`.
>
> Status values:
>
> - **Closed in docs**: the public truth surface now appears aligned for that finding.
> - **Partially closed**: the repo now documents the issue correctly, but executable/public proof is still missing.
> - **Open**: still requires implementation, public asset proof, metadata work, or local verification.

## Summary

| Status | Count |
| --- | ---: |
| Closed in docs | 7 |
| Partially closed | 16 |
| Open | 5 |

The biggest improvement is that the repo now presents a much stronger front-door truth surface. The biggest remaining risks are executable proof, quality evidence, public release assets, subprocess/PATH classification, and external public presence.

## Matrix

| # | Audit finding | Current closure status | Evidence / next action |
| ---: | --- | --- | --- |
| 1 | Contradictory release-state claims | Closed in docs | README and `RELEASE_STATUS.md` now say no public beta prerelease is published yet. Keep all future release docs synchronized. |
| 2 | Public repo surface and branch narrative out of sync | Partially closed | Staging docs now explain beta-prep state. Full closure requires public assets or all docs staying beta-prep. |
| 3 | Support language broader than evidence | Closed in docs | `SUPPORT_TIERS.md`, README, and limitations separate verified primary from preview targets. |
| 4 | Thesis stronger than operational status | Closed in docs | README now places release snapshot and canonical docs before deeper thesis sections. |
| 5 | Bundled custom MiGraphX not explicit enough | Closed in docs | README, `PACKAGING.md`, and support docs now explain bundled custom MiGraphX and host boundaries. |
| 6 | Repo metadata underdeveloped | Open | Requires GitHub description/topics/homepage or website/public presence update. |
| 7 | Benchmark coverage too narrow | Partially closed | Docs acknowledge reference-system scope. Full closure requires broader benchmark corpus/results. |
| 8 | Quality proof weaker than throughput proof | Partially closed | Docs acknowledge gap. Full closure requires before/after examples, failure cases, and model/backend notes. |
| 9 | External tester intake informal | Closed in docs | `BETA_TESTING_PROGRAM.md` defines required environment, benchmark, and quality-report details. |
| 10 | Compatibility claims outrun proof | Closed in docs | Support tiers and release docs separate package targets from compatibility proof. |
| 11 | Docs do not behave as one source of truth | Partially closed | Canonical docs now exist. Full closure requires ongoing governance and release-note/website alignment. |
| 12 | Test maturity described inconsistently | Partially closed | `tests/README.md` and `VALIDATION_AND_EVIDENCE.md` now describe subsystem/smoke limits. Verify no stale contradictory docs remain. |
| 13 | Public docs expose maintainer-local paths | Partially closed | Fetched `CONTRIBUTING.md` looked clean. Full closure requires repo-wide `rg '/home/|/mnt/|rolaand|jayz'` review. |
| 14 | Known limitations not centralized | Closed in docs | `docs/LIMITATIONS.md` centralizes release, validation, runtime, support, and beta limitations. |
| 15 | Release path shaped around maintainer machine | Partially closed | Docs now require explicit `AVE_BUNDLED_MIGRAPHX_PREFIX`; local packaging scripts need validation. |
| 16 | Public reproducibility proof thin | Partially closed | Packaging docs are clearer. Full closure requires external reproduction/build/package evidence. |
| 17 | Omission of package artifacts creates proof gap | Partially closed | Docs correctly describe intended artifacts and absence of public release. Closure requires published assets. |
| 18 | Shell-based subprocess execution in model prep | Open | Requires line-level classification of `std::system`, `popen`, `system(`, `python3`, `unzip`, and process runner paths. |
| 19 | Critical helper paths depend on PATH tools | Partially closed | Runtime notes document `python3`, `unzip`, `ffmpeg`, `ffprobe`, and `migraphx-driver` expectations. Full closure requires behavior/failure tests. |
| 20 | First-run model prep black box | Partially closed | Limitations mention long first-run compile and coarse progress. Full closure requires better progress states or proof existing UX is adequate. |
| 21 | Bundled vs host-required boundary must be legible | Closed in docs | Support and packaging docs clearly separate bundled app-private userspace from host AMD driver stack. |
| 22 | Release-confidence story incomplete | Partially closed | Test/evidence docs explain current coverage. Full closure requires local test run plus stronger media-level regression proof. |
| 23 | Failure behavior not release-grade contract | Partially closed | Evidence docs discuss failure contract as evolving. Needs explicit recovery/fallback support contract and tests. |
| 24 | Backend presence vs support not separated | Closed in docs | `SUPPORT_TIERS.md` separates primary, fallback, preview, experimental/manual, and out-of-scope backend tiers. |
| 25 | Experimental paths create product ambiguity | Partially closed | Support tiers label specialized paths. Keep UI/docs/release notes aligned so users do not treat them as default promises. |
| 26 | No public website yet | Open | A website/spec pack may exist on another branch/commit, but staging/public presence is not closed. |
| 27 | Community intake not shaped as sustained beta program | Partially closed | `BETA_TESTING_PROGRAM.md` exists. Full closure requires issue templates, linked release notes, and actual tester reports. |
| 28 | Thesis invites hostile scrutiny; sloppiness costs more | Partially closed | Truth surface is much stronger. Full closure requires executable proof, asset proof, quality proof, and ongoing doc governance. |

## Immediate next actions

1. Run local build/test/package smoke and update `BASELINE_AUDIT.md`.
2. Run repo-wide stale-truth searches for prerelease/beta/release wording.
3. Run repo-wide local-path search for maintainer machine leakage.
4. Classify all subprocess/external-tool paths.
5. Decide whether the active release state is still beta-prep or whether `v0.1.0-beta.1` exists and must become the canonical truth.
6. Add quality-evidence workflow before making stronger output claims.

## Do not mark gold-standard complete yet

The repo is much cleaner, but the gold standard from the audit requires public evidence and reproducibility, not just better docs. Treat this matrix as a living release gate until the first beta is published and validated.
