# Parity Plan

This document tracks the next cleanup and implementation priorities for the repo.

## Current baseline

- CLI and GUI both build from the same `ave_core` library
- MiGraphX is the primary AMD inference backend
- Vulkan Compute and NCNN Vulkan are available as fallback GPU paths
- FFmpeg remains the guaranteed fallback when AI backends are unavailable
- Model download, compile, and cache flows exist in the app today

## Near-term priorities

1. Keep documentation aligned with the actual code and supported workflows
2. Expand regression coverage beyond planner-only tests
3. Improve MiGraphX compile-time behavior, cache reuse, and first-run UX
4. Tighten backend-specific error reporting so failures stay actionable
5. Reduce gaps between CLI, GUI, and backend capability reporting

## MiGraphX-specific focus

- Preserve and reuse compiled `.mxr` artifacts aggressively
- Prefer fixed-shape tiled compilation where it reduces compile churn
- Improve compile fallback policy for hard-to-compile models
- Expose more compile diagnostics without making the default UX noisy

## Not planned

- CUDA / NVIDIA support
- Windows or macOS ports
- Reintroducing a Python-first runtime architecture
