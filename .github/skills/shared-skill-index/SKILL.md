---
name: shared-skill-index
description: "Discover the personal shared skill pack available in this environment and choose the right reusable workflow for this repository. Use when you need routing guidance across specification, anti-stub, security, release-readiness, hardening, and optimization skills without duplicating the whole pack into the repo."
argument-hint: "Task type or problem area you want routed"
user-invocable: true
disable-model-invocation: false
---

# Shared Skill Index

Use this skill to choose the right reusable specialist workflow from the personal shared pack while staying aligned with this repository's local rules.

## Canonical shared source

The reusable personal skill pack for this environment lives in:

- `/home/rolaandjayz/.agents/skills/`

## Use this index when you need

- requirements or specification packs -> `specification-pack`
- skeptical challenge or scope pressure testing -> `devils-advocate`
- release go/no-go review -> `release-readiness`
- full no-stubs workflow -> `clean-stubs`
- stub audit only -> `stub-sweep`
- security audit -> `security-audit`
- security remediation planning -> `security-remediation-plan`
- security implementation -> `security-hardening-implement`
- correctness / stability / privacy / efficiency reviews -> the corresponding hardening skills
- broad hardening routing -> `harden-system`
- code-path optimization -> `code-logic-optimization`
- broad optimization routing -> `optimize-system`

## Repo-first rule

Before invoking a shared skill, use this repository's `AGENTS.md` and repo-local instructions for build/test/runtime facts.

## No-duplication rule

This repo intentionally does **not** vendor the entire personal shared pack.
Use the canonical personal skills when available; only add repo-local copies when true repo portability is required.