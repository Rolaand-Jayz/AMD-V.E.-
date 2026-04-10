---
name: stub-sweep
description: "Systematically hunt stubs, placeholders, fake completion paths, TODO-backed features, NotImplemented-style stand-ins, and similar unfinished behavior across an entire repository. Use when you need a full stub audit of every tracked user-authored file and a documented findings report without fixing anything yet."
argument-hint: "Optional: findings document path, scope notes, or extra stub patterns to search for"
---

# Stub Sweep

Use this skill to perform a systematic repository-wide audit for stubbed or stub-like behavior.

## What this skill does

- Audits every tracked, user-authored text file in the repository.
- Looks for explicit and implicit stub patterns in code, tests, scripts, docs, prompts, agents, instructions, and skill files.
- Produces a findings document for later planning.
- Does **not** remediate or edit product code as part of the sweep.

## Hard rules

- Do **not** fix, refactor, or delete stubbed code while using this skill.
- Do **not** change runtime behavior, tests, or configuration during the audit.
- The only allowed write is the findings document, unless the user explicitly asks for something else.
- If a file or directory is excluded because it is generated, vendored, binary, or outside the requested scope, list the exclusion explicitly in the report.

## Default outputs

- Findings document: `docs/stub-audit/stub-findings.md`
- Template: [findings template](./assets/findings-template.md)

## Stub patterns to look for

Search for both exact markers and behavioral equivalents, including:

- `stub`, `placeholder`, `temporary`, `todo`, `fixme`, `not implemented`
- `NotImplementedError`, `pass`, empty methods, sentinel returns
- fake success values, silent no-ops, `return true` / `return {}` / `return nullptr` paths that bypass real work
- UI controls or commands wired to message-only behavior instead of real execution
- config flags or code paths that advertise a feature but intentionally skip implementation
- docs, prompts, skills, or instructions that tell agents to use placeholder implementations

## Procedure

1. Inventory the repository files first.
   - Cover every tracked, user-authored text file.
   - Note generated/build output directories separately instead of silently skipping them.
2. Run broad searches for explicit stub markers.
3. Read surrounding code for each hit to decide whether it is a real stub, a false positive, or an intentional hard failure with a truthful message.
4. Perform a second pass for implicit stubs that may not use obvious keywords.
5. Write or update the findings document using the template.
6. Report back with:
   - total files reviewed
   - exclusions
   - confirmed findings
   - likely findings needing manual confirmation
   - notable false positives

## Required report contents

For each confirmed or likely finding, capture:

- unique finding ID
- file path
- line or symbol
- category (`explicit stub`, `implicit stub`, `doc/instruction stub guidance`, `fake success path`, `placeholder UX`, etc.)
- evidence snippet or summary
- why it is stub-like
- user-visible impact
- recommended next action (`plan`, not `fix now`)

## Completion standard

This skill is complete only when the findings document exists or is updated and the chat summary clearly states that no remediation was performed.
