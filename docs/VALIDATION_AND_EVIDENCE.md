# Validation and Evidence

This file explains what the current public evidence actually proves.

## Evidence tiers

| Tier | Meaning | Current state |
| --- | --- | --- |
| Maintainer-verified | Reproduced on the reference system with checked commands and outputs | Current benchmark snapshot and the visible subsystem test surface live here |
| Community-submitted | Submitted by outside testers with complete environment detail and required artifacts | Wanted, but still early and should stay clearly separated from maintainer-verified proof |
| Not public proof yet | Maintainer-local runs, unpublished packaging attempts, or undocumented anecdotes | Do not treat these as release claims |

## Current test coverage categories

### Subsystem regression coverage

The repo currently has visible tests for:

- planner ordering
- tensor contracts
- runtime diagnostics and runtime paths
- model-manager profile behavior
- video probing
- process loops, observers, and progress reporting
- frame I/O paths
- FFmpeg runner behavior
- job queue and recovery logic
- telemetry
- MiGraphX backend behavior
- Vulkan/HIP interop behavior

### Runtime smoke / environment coverage

The runtime diagnostics and process-observer surfaces help prove that the app can inspect and explain parts of its environment rather than failing silently.

### What is still incomplete

- there is not yet a broad golden-clip regression suite covering every backend and every model class
- release confidence is still stronger at the subsystem/smoke layer than at the full media-output layer
- package-target breadth is still wider than the current public validation breadth

## Benchmark policy

The current benchmark snapshot in [`BENCHMARKS.md`](./BENCHMARKS.md) is a **reference-system throughput snapshot**.

It is useful because it proves:

- the app runs real work on the verified system
- the benchmark clip, commands, and output locations are reproducible
- MiGraphX timing and end-to-end wall time can both be recorded

It does **not** by itself prove:

- broad compatibility across preview distro targets
- broad quality superiority across every model/path
- stability across all hardware variations

## Quality evidence status

The public quality story is still narrower than the runtime/packaging story.

That means:

- quality claims should stay specific and conservative
- before/after examples and failure-case examples are still a needed growth area
- hostile reviewers should be able to tell the difference between throughput proof and output-quality proof

## Failure-behavior contract

The project aims to fail in an explainable way, but the clean public contract is still evolving.

Today the repo can already point users toward:

- explicit environment reporting
- diagnostics-oriented issue templates
- runtime fallback boundaries

But the public release story should still avoid overstating how complete the failure/recovery contract already is.

## For outside validation

Use [`BETA_TESTING_PROGRAM.md`](./BETA_TESTING_PROGRAM.md) for:

- benchmark submissions
- compatibility reports
- quality evidence submissions
- structured issue intake
