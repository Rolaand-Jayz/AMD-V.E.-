# Known Limitations — Implementation View

> Generated on 2026-05-05 as part of Phase 0. This file is narrower than `docs/LIMITATIONS.md`: it is meant for implementation agents deciding what to fix next.
>
> Do not hide limitations to make a milestone look done. Move fixed items to the resolved section with commit/PR references.

## Active limitations

| ID | Limitation | Area | Severity | Blocking? | Workaround / current handling | Target fix |
| --- | --- | --- | ---: | --- | --- | --- |
| KL-001 | Local build/test evidence has not been recorded in `docs/implementation/BASELINE_AUDIT.md`. | Phase 0 validation | Critical | Yes, for feature work | Treat this directory as scaffolded only. | Run build/test on target workstation and update baseline. |
| KL-002 | Public beta release assets are not yet externally proven on this branch's truth surface. | Release | Critical | Yes, for release publication | Docs correctly say no public beta is published yet. | Publish beta assets or keep all docs in beta-prep language. |
| KL-003 | Package target breadth is wider than verified compatibility. | Packaging/support | Major | No, if labeled | `SUPPORT_TIERS.md` labels preview targets. | Collect structured validation reports per distro/package. |
| KL-004 | Visual output-quality evidence is thinner than throughput/build evidence. | Validation/evidence | Major | No, if claims stay narrow | `VALIDATION_AND_EVIDENCE.md` warns quality proof is still narrower. | Add before/after examples, failure cases, and versioned quality notes. |
| KL-005 | Broad golden-clip media regression coverage does not exist yet. | Testing | Major | No, if not overclaimed | Tests remain subsystem/smoke oriented. | Add golden-clip regression suite after clips/licensing are settled. |
| KL-006 | Subprocess/PATH hardening is not fully closed in this Phase 0 pass. | Runtime/security | Major | Yes, for marking audit item closed | Remote inspection shows hardening exists, but line-level classification is pending. | Classify all subprocess/external-tool paths and harden shell-dependent flows. |
| KL-007 | First-run MiGraphX preparation still has coarse progress in some phases. | Runtime UX | Major | No, if documented | `LIMITATIONS.md` warns first-run compilation can take minutes. | Improve progress states and make compile/quantization/cache states user-visible. |
| KL-008 | Backend code presence is broader than public support. | Product scope | Major | No, if tiered | `SUPPORT_TIERS.md` defines backend tiers. | Keep support-tier labels in README, release notes, UI, and issue templates. |
| KL-009 | Preview-cache/timeline/split-wipe concepts from R2 are not verified as current repo features. | Future implementation | Major | Yes, for UI work | Treat R2 as design target only. | Add contracts/tests before UI implementation. |
| KL-010 | Mode/family metadata may exist partially but R2's full mode/family contract is not implemented/proven. | Future implementation | Major | Yes, for mode work | Inspect current `ModelEntry`/catalog/manifest behavior first. | Add backward-compatible schemas/tests before changing UX. |
| KL-011 | Temporal Forge is documentation/specification only and must not be default behavior. | Phase 2 | Critical | Yes, for release promise | Decision log marks Temporal Forge experimental/disabled by default. | Implement only after Phase 1 contracts and clean-room review. |
| KL-012 | Repo metadata / website public presence may still lag the project quality. | Adoption | Moderate | No | Repo docs now carry the truth burden. | Add metadata/site only after release truth language is reconciled. |
| KL-013 | Branch-sensitive docs may drift: a later website spec observed on another commit used published-beta wording while staging docs used beta-prep wording. | Documentation cohesion | Major | Yes, before merging website pack into staging | Keep `pre-release-staging` release truth as source of truth. | Reconcile website/spec language with actual release state before merge/publication. |

## Release blocker list

These must be resolved or explicitly accepted before the first public beta:

1. KL-001 — local executable baseline missing.
2. KL-002 — release assets not externally inspectable while docs say beta-prep.
3. KL-006 — subprocess/PATH classification pending.
4. KL-013 — any website/spec language must match actual release state before publication.

## Non-blocking but credibility-sensitive

These should not block a clearly labeled beta, but they will affect public trust:

- KL-003 — preview package validation.
- KL-004 — visual-quality evidence.
- KL-005 — golden-clip regression coverage.
- KL-007 — first-run progress clarity.
- KL-012 — repo metadata / website polish.

## Rules for closing limitations

A limitation can be moved to the resolved section only when there is concrete evidence:

- commit/PR reference
- test/build output, when applicable
- updated public doc, when applicable
- release asset or validation report, when applicable

Do not close a limitation only because a roadmap says it should be fixed.

## Resolved limitations

| ID | Fixed in | Evidence | Notes |
| --- | --- | --- | --- |
| RL-001 | Existing staging docs before this Phase 0 pass | `README.md`, `RELEASE_STATUS.md`, `SUPPORT_TIERS.md`, `LIMITATIONS.md`, `VALIDATION_AND_EVIDENCE.md`, `BETA_TESTING_PROGRAM.md` | The highest-risk release-truth language is now much more aligned than the original audit baseline. |
| RL-002 | Existing staging docs before this Phase 0 pass | `CONTRIBUTING.md` | Public contributor doc no longer appears to expose maintainer-local filesystem paths in the fetched staging copy. |
