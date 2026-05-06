# Implementation Decisions

> Generated on 2026-05-05 as part of Phase 0 release-hardening and implementation-readiness work.
>
> Add new decisions here when a change affects public behavior, release claims, security posture, support tiers, packaging, or CLI/GUI parity.

## Decision format

```markdown
## D-YYYYMMDD-NN — Title

Status: proposed / accepted / superseded

Context:

Decision:

Consequences:

Related tasks/findings:
```

---

## D-20260505-01 — Phase 0 before feature work

Status: accepted

Context:

The R2 roadmap contains major Phase 1 and Phase 2 work: mode/family consolidation, planner fusion, preview cache, workstation UI, CLI/profile parity, and Temporal Forge experiments. The pre-release audit also identified release-truth, evidence, packaging, and support-boundary risks that are more urgent than feature expansion.

Decision:

No feature work begins until Phase 0 files exist and are updated with local build/test evidence:

- `docs/implementation/BASELINE_AUDIT.md`
- `docs/implementation/FILE_MAP.md`
- `docs/implementation/IMPLEMENTATION_DECISIONS.md`
- `docs/implementation/KNOWN_LIMITATIONS.md`
- `docs/implementation/AUDIT_CLOSURE_MATRIX.md`

Consequences:

Agents must map the repo before modifying it. If an implementation task cannot name the real files/classes/functions it will touch, it is not ready.

Related tasks/findings:

- R2 `P0-E0-T1`, `P0-E0-T2`, `P0-E0-T3`
- Audit findings 1, 11, 16, 22, 28

---

## D-20260505-02 — Release truth beats roadmap excitement

Status: accepted

Context:

The project thesis is strong and provocative, but hostile reviewers will punish even small contradictions about release state, support scope, evidence, benchmarks, and package readiness.

Decision:

Whenever roadmap ambition conflicts with current proof, public docs must prefer the narrower proven statement.

Consequences:

- README, release docs, website specs, packaging docs, issue templates, and release notes must use the same release-state language.
- A public beta cannot be described as published until the GitHub Releases page and assets are externally inspectable.
- Package targets cannot be described as validated support unless target-system validation exists.
- Quality claims must stay conservative until before/after and failure-case evidence exists.

Related tasks/findings:

- Audit findings 1, 2, 3, 4, 7, 8, 10, 11, 14, 16, 17, 26, 28

---

## D-20260505-03 — Support tiers are the public contract

Status: accepted

Context:

The code tree exposes more backend and platform surface than the current release can prove. Users and critics may confuse code presence with support.

Decision:

`docs/SUPPORT_TIERS.md` is the support promise unless superseded by a release note that says the same thing in compatible language.

Consequences:

Every backend/platform/package must fit one of these categories:

- verified primary
- supported fallback
- preview target
- experimental/manual
- out of scope

A feature or package existing in the repository does not widen the support promise by itself.

Related tasks/findings:

- Audit findings 3, 10, 21, 24, 25

---

## D-20260505-04 — Bundled custom MiGraphX must stay explicit

Status: accepted

Context:

The planned beta relies on a custom bundled MiGraphX runtime/toolchain because required upstream behavior is not yet available in the stock system path. This is legitimate, but only if it is impossible to miss.

Decision:

Any release, packaging, website, or installation surface that talks about beta artifacts must explain whether custom MiGraphX is bundled, why it is bundled, and which host GPU requirements remain outside the bundle.

Consequences:

- `AVE_BUNDLED_MIGRAPHX_PREFIX` must be explicit for release packaging.
- Hidden maintainer-local paths are not acceptable release inputs.
- Packaging docs must keep the app-private runtime boundary legible.
- Host kernel/driver/KFD/Vulkan/ROCm requirements remain visible.

Related tasks/findings:

- Audit findings 5, 15, 19, 21

---

## D-20260505-05 — CLI/GUI parity is required for product behavior

Status: accepted

Context:

The project has both a CLI and a Qt GUI. R2 calls for mode/family UX, preview behavior, profile parity, and workstation-style UI improvements.

Decision:

Any new user-facing behavior added to the GUI must have either CLI-equivalent behavior, shared profile/config serialization, or an explicit documented reason why it is GUI-only.

Consequences:

- GUI-only hidden behavior is not considered complete.
- Planned pipeline representation should be shared between CLI dry-run/profile output and GUI diagnostics where practical.
- Mode/family selections must serialize deterministically.

Related tasks/findings:

- R2 Phase 1 mode/family, planner, preview, UI, CLI/profile tasks

---

## D-20260505-06 — Temporal Forge is experimental until Phase 1 contracts land

Status: accepted

Context:

Temporal Forge is high-potential but high-risk. It touches temporal buffers, motion/depth providers, masks, possible ML refiners, quality evaluation, performance, and IP/licensing boundaries.

Decision:

Temporal Forge must remain experimental and disabled by default until Phase 1 contracts are implemented and validated:

- mode/family metadata
- planner diagnostics/fusion behavior
- preview/cache foundations
- CLI/GUI profile parity
- release gates

Consequences:

Temporal Forge can be documented and prototyped behind explicit experimental flags, but it must not become part of the default beta promise prematurely.

Related tasks/findings:

- R2 Phase 2
- Audit findings 8, 22, 24, 25, 28

---

## D-20260505-07 — Clean-room and license-first temporal work

Status: accepted

Context:

The roadmap discusses FSR-class temporal concepts. The project must avoid license/IP contamination, leaked code, redacted SDK material, or vendor source copied without permission.

Decision:

Temporal reconstruction, masks, motion/depth providers, and ML refiners must be clean-room unless a dependency has explicit compatible licensing and passes review.

Consequences:

- No leaked, accidentally published, redacted, or proprietary source enters the repo.
- Any AMD SDK/API use requires a license review note before merge.
- Dataset/model provenance must be recorded before training/evaluation claims.

Related tasks/findings:

- R2 license/IP docs
- Audit strategic trust concerns

---

## D-20260505-08 — Subprocess and PATH behavior must be classified before closure

Status: accepted

Context:

The audit identified shell/PATH risk around model preparation, archive extraction, Python export helpers, and external tools. Remote inspection shows some hardening exists, but full closure requires line-level review.

Decision:

Before marking subprocess hardening closed, every subprocess or external-tool path must be classified as one of:

- argument-vector safe and bounded
- PATH-dependent but documented and acceptable
- shell-dependent and requiring hardening
- test-only acceptable

Consequences:

The next implementation pass must run a focused search for `std::system`, `popen`, `system(`, `python3`, `unzip`, `migraphx-driver`, and `process_observer` and update `BASELINE_AUDIT.md` / `KNOWN_LIMITATIONS.md`.

Related tasks/findings:

- Audit findings 18, 19, 20, 23

---

## D-20260505-09 — Evidence must be versioned and tiered

Status: accepted

Context:

The repo has meaningful subsystem tests and benchmark evidence, but visual output quality and community validation are still immature.

Decision:

Public evidence must remain tiered:

- maintainer-verified
- community-submitted
- not-public-proof-yet

Evidence tied to a release should identify the release/commit it supports.

Consequences:

- Do not mix local anecdotes into public claims.
- Community submissions stay labeled until reproduced or validated.
- Throughput proof and quality proof must remain separate.

Related tasks/findings:

- Audit findings 7, 8, 9, 16, 22, 27
