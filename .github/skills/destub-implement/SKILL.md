---
name: destub-implement
description: "Implement a de-stub remediation plan end-to-end. Use when you already have a stub findings document and a de-stub plan and need to replace stubbed features with real, ready-to-use behavior, tests, and validation without introducing new stubs."
argument-hint: "Optional: path to remediation plan, findings document, or narrowed work item IDs"
---

# De-stub Implementation

Use this skill only after a stub findings document and remediation plan exist.

## What this skill does

- Reads the remediation plan and linked findings.
- Implements real feature behavior incrementally.
- Updates tests and validation as needed.
- Reports completed work and any real blockers.

## Hard rules

- Do **not** introduce new stubs, placeholder behavior, fake success paths, or TODO-backed user flows.
- Do **not** claim a feature is complete unless the real behavior exists and is validated.
- If a planned item cannot be completed end-to-end, stop at the last correct state and report the blocker explicitly.
- Prefer small, testable changes, but each merged step must remain truthful and usable.

## Default inputs

- Findings document: `docs/stub-audit/stub-findings.md`
- Remediation plan: `docs/stub-audit/destub-remediation-plan.md`
- Execution checklist: [implementation checklist](./assets/execution-checklist.md)

## Procedure

1. Read the remediation plan and the linked findings.
2. Pick the highest-priority unimplemented work item unless the user narrowed scope.
3. Read all relevant files before editing.
4. Implement the real behavior.
5. Add or update tests that prove the feature now works.
6. Build and run the relevant validation.
7. Update the plan status or produce an implementation summary.
8. Move to the next work item only if the repo remains in a healthy state.

## Required implementation standard

Every completed work item must have:

- real behavior wired through the actual runtime path
- tests or equivalent validation
- no hidden placeholder branches
- accurate user-facing messaging
- clear note of any remaining blocked items

## Completion standard

This skill is complete only when the requested work items are implemented and validated, or when a concrete blocker is reported without leaving fake behavior behind.
