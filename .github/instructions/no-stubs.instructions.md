---
description: "Use when writing, changing, planning, auditing, or implementing anything in this repository: stubs, placeholder behavior, fake completion paths, mock implementations presented as real, TODO-backed user flows, and NotImplemented-style stand-ins are forbidden. This applies to code, tests, plans, instructions, prompts, agents, and skills."
applyTo: "**"
---

# No stubs in this repository

These rules apply to source code, tests, docs, plans, skills, prompts, instructions, and agent guidance in this workspace.

- Do not add stubs, placeholder behavior, fake success paths, mock implementations presented as complete, TODO-backed user flows, `pass`, `NotImplementedError`, or equivalent stand-ins in app code.
- Do not write plans, skills, or instructions that recommend using stubs as an acceptable intermediate state.
- If a feature cannot be implemented end-to-end, say so explicitly and document the blocker instead of pretending the feature exists.
- When auditing for stubs, document them accurately and do not silently normalize them.
- When replacing stubbed behavior, replace it with real, usable behavior plus validation and tests.
- Treat stub-like shortcuts as correctness issues, not polish items.
