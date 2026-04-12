---
description: "Use when planning, auditing, documenting, packaging, benchmarking, release-prepping, or aligning public messaging in this AMD Video Enhancer repository. Enforces the repo's gold-standard truth surface for release state, support tiers, evidence, limitations, and bundled MiGraphX disclosure."
applyTo: "**"
---

# Release gold standard for AMD Video Enhancer

This repository is allowed to be provocative.
It is **not** allowed to be sloppy.

Because this repo makes strong public claims about AI-assisted development, AMD-first engineering, and zero-experience vibe coding, small mismatches between what the repo says and what outsiders can verify count as release blockers, not cosmetic nits.

## Core rule

Visible claims must match externally verifiable proof.

If a reader cannot confirm a release state, support statement, package promise, benchmark claim, or operational readiness statement from the public repo surface that exists today, do not present that statement as already true.

## Always enforce these disciplines

### 1. Keep current truth separate from target state

Do not blur:

- what is already published
- what exists only on a branch
- what is planned for the next beta build
- what is long-term roadmap material

### 2. Use support tiers, not fuzzy wording

When discussing platforms, backends, or distribution surfaces, separate them into explicit tiers such as:

- verified primary
- supported fallback
- preview target
- experimental/manual path
- out of scope

Packaging reach is not the same thing as validated support.

### 3. Keep the public truth surface aligned

README, release docs, packaging docs, issue templates, workflow messaging, future website text, and repo metadata should all tell the same story about:

- current release state
- verified environments
- preview/experimental targets
- known limitations
- what is bundled versus what still depends on the host stack

### 4. Be blunt about the bundled custom MiGraphX path

When relevant, make it obvious that:

- this beta relies on a bundled custom MiGraphX path
- that choice is deliberate, not accidental
- it exists because required upstream behavior is not yet available in the stock path
- users must not confuse the bundled beta path with a generally supported system-MiGraphX install path

### 5. Treat evidence as part of the product surface

Do not let performance, quality, compatibility, or launch claims outrun the evidence.

Separate:

- maintainer-verified evidence
- community-submitted evidence
- aspirational expectations

If the proof corpus is narrow, say so clearly.

### 6. Centralize limitations and rough edges

Beta users can tolerate limitations.
They should not have to discover them by accident.

Important limitations should be easy to find and should stay consistent across the repo.

## What good alignment looks like here

A strong change in this repo should make it easier for an outsider to answer, quickly and correctly:

- Is there a public beta release yet?
- What platform is actually verified?
- Which distro/package outputs are preview only?
- What is the primary backend path?
- What are the support tiers?
- What is bundled with the app?
- What still depends on the host GPU stack?
- What is still rough, limited, or experimental?

## Skill-routing note

For work that touches these surfaces, prefer the canonical shared skills that reinforce this discipline:

- `release-readiness`
- `specification-pack`
- `devils-advocate`

But repo facts, repo constraints, and this instruction file win if there is any conflict.
