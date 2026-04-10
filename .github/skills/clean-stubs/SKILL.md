---
name: clean-stubs
description: "Run the full no-stubs cleanup workflow by invoking the stub-sweep, destub-plan, and destub-implement skills in order. Use when you want one slash command to audit the repository for stubs, create a remediation plan, and then implement real feature behavior without introducing new stubs."
argument-hint: "Optional: narrowed scope, findings/plan paths, or priority filters"
---

# Clean Stubs

Use this skill when you want a single slash command to drive the full de-stub workflow from audit through implementation.

## What this skill does

This workflow runs the existing de-stub skills in order:

1. `stub-sweep`
2. `destub-plan`
3. `destub-implement`

It exists so the repository can be cleaned of stubbed behavior without skipping the audit or planning phases.

## Hard rules

- Do **not** skip the `stub-sweep` phase.
- Do **not** skip the `destub-plan` phase.
- Do **not** implement from memory alone; use the findings document and remediation plan as required inputs.
- Do **not** introduce new stubs, placeholder behavior, fake-success paths, mock-complete flows, or TODO-backed user paths.
- If the sweep finds no confirmed stubs, stop after documenting that result and report that no remediation was needed.
- If some items are blocked, implement only the unblocked items and document the blockers clearly.

## Default artifacts

- Findings document: `docs/stub-audit/stub-findings.md`
- Remediation plan: `docs/stub-audit/destub-remediation-plan.md`

## Procedure

1. Run the `stub-sweep` workflow first.
   - Audit every tracked, user-authored text file in scope.
   - Update the findings document.
   - Do not modify product code during this phase.
2. Run the `destub-plan` workflow second.
   - Read the findings document.
   - Produce or update the remediation plan.
   - Do not implement code during this phase.
3. Run the `destub-implement` workflow third.
   - Read the findings and the remediation plan.
   - Implement the highest-priority unblocked work items with real behavior.
   - Add or update validation and tests as needed.
4. Summarize the full pass.
   - Findings count
   - Planned work items
   - Completed implementations
   - Remaining blockers
   - Validation performed

## Completion standard

This skill is complete only when the audit, planning, and implementation phases have all been executed in order, or when the workflow truthfully stops because there were no stubs to remediate or blockers prevented further real implementation.
