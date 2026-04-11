# Benchmarks Folder Guide

This folder holds the benchmark inputs, generated outputs, and historical benchmark snapshots used to measure the project over time.

## Layout

| Path | Meaning |
| --- | --- |
| `ave_benchmark_960x540_20s.mp4` | the current canonical short benchmark clip |
| `generated/` | freshly generated benchmark outputs |
| `history/` | stored benchmark snapshots from earlier runs |

## How this folder is used

The benchmark scripts in [`../tools/README.md`](../tools/README.md) treat this folder as their workspace:

- input media comes from here
- generated outputs are written here
- historical results are preserved here

## Where the human-readable benchmark story lives

The best narrative benchmark doc is not in this folder. It is in:

- [`../docs/BENCHMARKS.md`](../docs/BENCHMARKS.md)

That file explains the current numbers, the test environment, and how to reproduce the documented benchmark snapshot.

## Why keep this folder separate?

Because benchmark assets are evidence, not just prose. This directory keeps the raw materials and the history, while the docs folder explains what the numbers mean.
