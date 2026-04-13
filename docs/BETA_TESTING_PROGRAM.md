# Beta Testing Program

This project needs outside validation, but it needs **clean** outside validation.

## What this beta program is for

- compatibility checks on preview distro targets
- benchmark submissions that can be compared fairly
- output-quality reports with enough context to matter
- failure reports that help separate app issues from host-stack issues

## Before you submit anything

Read these first:

- [`RELEASE_STATUS.md`](./RELEASE_STATUS.md)
- [`SUPPORT_TIERS.md`](./SUPPORT_TIERS.md)
- [`LIMITATIONS.md`](./LIMITATIONS.md)
- [`VALIDATION_AND_EVIDENCE.md`](./VALIDATION_AND_EVIDENCE.md)

## Required environment details

Every beta submission should include:

- distro and version
- kernel version
- CPU model
- GPU model
- ROCm version
- MiGraphX version if installed separately
- Mesa/Vulkan driver details when relevant
- package asset name or source-build commit/branch
- whether the report came from the verified reference stack or a preview/experimental stack

## Benchmark submission rules

When submitting a benchmark report, include:

- the exact command used
- the input clip path/name
- output resolution and model name
- wall-clock time
- backend timing if available
- full stdout/stderr or log tail

If you can use the repo benchmark clip and helper scripts, prefer that for comparability.

## Quality evidence rules

When submitting quality feedback, include:

- the source clip or a trimmed reproducible segment
- the model/backend used
- the exact command or GUI settings
- before/after stills or short clips when possible
- notes about temporal artifacts, ringing, halos, oversharpening, or failure modes

## Issue templates to use

- use **Bug report** for reproducible app/package/runtime failures
- use **Experimental distro report** for preview/experimental Linux stacks
- use **Benchmark / validation report** when the goal is evidence rather than a bug fix request

## Evidence handling rule

Community-submitted evidence is valuable, but it should stay clearly labeled as community evidence until it is reproduced or otherwise validated by the maintainer.
