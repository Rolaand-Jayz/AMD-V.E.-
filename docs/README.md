# Documentation Guide

This folder is where the repository explains itself in long form. If the root [`README.md`](../README.md) is the front door, this folder is the bookshelf behind it.

## Start here depending on what you need

| Read this file | Use it when you want... |
| --- | --- |
| [`RELEASE_STATUS.md`](./RELEASE_STATUS.md) | the current public truth about release state, what is published, and what is still staging-only |
| [`SUPPORT_TIERS.md`](./SUPPORT_TIERS.md) | the platform, backend, and packaging support tiers without guessing from the code tree |
| [`LIMITATIONS.md`](./LIMITATIONS.md) | the blunt user-facing list of current caveats, rough edges, and release blockers |
| [`VALIDATION_AND_EVIDENCE.md`](./VALIDATION_AND_EVIDENCE.md) | what the current tests and benchmarks really prove, and what they do not |
| [`BETA_TESTING_PROGRAM.md`](./BETA_TESTING_PROGRAM.md) | how to submit clean compatibility, benchmark, and quality reports |
| [`WHY_THIS_PROJECT_MATTERS.md`](./WHY_THIS_PROJECT_MATTERS.md) | the sharp argument for why this repo matters as evidence about AI-assisted development, MiGraphX visibility, and under-documented stacks |
| [`GOLD_STANDARD_FOR_IMPLEMENTATION.md`](./GOLD_STANDARD_FOR_IMPLEMENTATION.md) | the current technical truth of the app and how the real stack works |
| [`FEATURE_PARITY_MATRIX.md`](./FEATURE_PARITY_MATRIX.md) | a feature-by-feature view of what is implemented and what is not |
| [`CUSTOM_MODEL_MANIFEST.md`](./CUSTOM_MODEL_MANIFEST.md) | the format and meaning of custom `.avemodel` manifests |
| [`BENCHMARKS.md`](./BENCHMARKS.md) | current benchmark snapshots and how to reproduce them |
| [`PACKAGING.md`](./PACKAGING.md) | portable bundles, native packages, bundled runtimes, and release payload layout |
| [`RELEASING.md`](./RELEASING.md) | release process and shipping workflow |
| [`migraphx_debugging_playbook.md`](./migraphx_debugging_playbook.md) | systematic debugging for ROCm, MiGraphX, Vulkan, FFmpeg, and interop problems |
| [`AGENT_FINDINGS.md`](./AGENT_FINDINGS.md) | repository findings and internal analysis notes |

## Best reading order for humans

If you are new to the project, this order usually makes the most sense:

1. [`../README.md`](../README.md)
2. [`RELEASE_STATUS.md`](./RELEASE_STATUS.md)
3. [`SUPPORT_TIERS.md`](./SUPPORT_TIERS.md)
4. [`LIMITATIONS.md`](./LIMITATIONS.md)
5. [`VALIDATION_AND_EVIDENCE.md`](./VALIDATION_AND_EVIDENCE.md)
6. [`WHY_THIS_PROJECT_MATTERS.md`](./WHY_THIS_PROJECT_MATTERS.md)
7. [`GOLD_STANDARD_FOR_IMPLEMENTATION.md`](./GOLD_STANDARD_FOR_IMPLEMENTATION.md)
8. [`PACKAGING.md`](./PACKAGING.md)

That order goes from “what is this?” to “how is it built?” to “how do I debug and ship it?”

## Why this doc set exists at all

One reason this documentation set matters is that the project lives in a part of the AMD software stack that still has thin public examples, especially on the MiGraphX side. That means these docs are not just convenience material for this repository; they are part of making a barely visible implementation space legible.

If you want the strongest version of that argument, read [`WHY_THIS_PROJECT_MATTERS.md`](./WHY_THIS_PROJECT_MATTERS.md). It is the explicit statement of why this repo matters beyond “here is an app.”

## How this folder relates to the rest of the repo

This directory holds the narrative and reference docs, but it is not the only place the repo explains itself. The implementation folders also have educational maps now:

- [`../src/README.md`](../src/README.md)
- [`../src/backends/README.md`](../src/backends/README.md)
- [`../src/gui/README.md`](../src/gui/README.md)
- [`../include/ave/README.md`](../include/ave/README.md)
- [`../tests/README.md`](../tests/README.md)
- [`../tools/README.md`](../tools/README.md)
- [`../packaging/README.md`](../packaging/README.md)
- [`../cmake/README.md`](../cmake/README.md)
- [`../benchmarks/README.md`](../benchmarks/README.md)

## One simple rule

If you need the repo's current architecture, trust [`GOLD_STANDARD_FOR_IMPLEMENTATION.md`](./GOLD_STANDARD_FOR_IMPLEMENTATION.md) first. It exists specifically to stop older aspirational descriptions from drifting away from the code.

If you need the repo's current release state, support boundary, limitations, or public evidence claims, trust [`RELEASE_STATUS.md`](./RELEASE_STATUS.md), [`SUPPORT_TIERS.md`](./SUPPORT_TIERS.md), [`LIMITATIONS.md`](./LIMITATIONS.md), and [`VALIDATION_AND_EVIDENCE.md`](./VALIDATION_AND_EVIDENCE.md) before inferring anything from packaging scripts or the code tree.
