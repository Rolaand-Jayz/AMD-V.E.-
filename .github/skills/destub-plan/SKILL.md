---
name: destub-plan
description: "Turn a stub audit document into a concrete remediation plan. Use when you have a stub findings report and need a systematic, no-stubs implementation plan that converts each stubbed feature into a complete, ready-to-use feature without writing placeholder behavior."
argument-hint: "Optional: path to findings document and desired plan output path"
---

# De-stub Planning

Use this skill after a stub sweep has produced a findings document.

## What this skill does

- Reads the stub findings document.
- Reads the affected code to understand the real feature boundaries.
- Produces a remediation plan that replaces stubbed behavior with complete implementations.
- Does **not** implement the code yet.

## Hard rules

- Do **not** edit implementation files while using this skill.
- Do **not** recommend stubs, temporary placeholders, or fake-success fallbacks as part of the plan.
- If a finding cannot be completed safely with current dependencies or architecture, mark it as blocked and explain the blocker plainly.
- Group findings by feature or subsystem when multiple findings point to the same incomplete capability.

## Default inputs and outputs

- Default findings document: `docs/stub-audit/stub-findings.md`
- Default plan document: `docs/stub-audit/destub-remediation-plan.md`
- Template: [plan template](./assets/plan-template.md)

## Procedure

1. Read the findings document.
2. For each finding, inspect the referenced code and adjacent files until the real scope of the missing feature is understood.
3. Merge duplicate findings into feature-level work items.
4. Write a plan that is implementation-ready, testable, and ordered.
5. Report back with the prioritized workstreams and the path to the generated plan.

## Required plan contents

For each work item include:

- work item ID and title
- linked finding IDs
- current behavior and why it is incomplete
- target end-state behavior
- files likely to change
- dependencies / prerequisites
- migration or compatibility concerns
- test strategy
- acceptance criteria
- sequencing / rollout order
- explicit note that no placeholder behavior is allowed

## Completion standard

This skill is complete only when the plan document exists or is updated and no implementation files were changed.
