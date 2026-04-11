# Public Header Guide

This folder is the public face of the core library. If `src/` is where the work is implemented, `include/ave/` is where the repo tells you what the important types, interfaces, and concepts are supposed to be.

## How to read this folder

Do not treat these headers as random declarations. They form the vocabulary of the app.

## Header groups

| Group | Key headers | What they describe |
| --- | --- | --- |
| backend abstraction | `backend.hpp`, `backend_manager.hpp` | what a backend is and how the app selects one |
| pipeline semantics | `planner.hpp`, `stage.hpp`, `types.hpp`, `video_processor.hpp` | the meaning of stages, ordering, and end-to-end execution |
| model + tensor management | `model_catalog.hpp`, `model_manager.hpp`, `tensor_contract.hpp` | how models are described, managed, compiled, and validated |
| media + runtime plumbing | `ffmpeg_runner.hpp`, `frame_io.hpp`, `rgb_video_loop.hpp`, `video_probe.hpp`, `vulkan_runtime.hpp`, `interop_bridge.hpp` | how frames move and how GPU/runtime plumbing is represented |
| diagnostics + observability | `runtime_diagnostics.hpp`, `runtime_paths.hpp`, `telemetry.hpp`, `observability.hpp`, `error_taxonomy.hpp` | how the app explains itself when something goes wrong |
| app + workflow support | `cli.hpp`, `app_settings.hpp`, `job.hpp`, `job_queue.hpp`, `job_recovery.hpp`, `process_loop.hpp`, `process_observer.hpp`, `process_progress.hpp` | user-facing flow control, settings, job handling, and progress reporting |
| backend-specific public surfaces | `backends/*.hpp` | backend-specific interfaces and types |

## Best headers to start with

If you only read a few files first, make them these:

1. `video_processor.hpp`
2. `backend.hpp`
3. `backend_manager.hpp`
4. `planner.hpp`
5. `model_manager.hpp`
6. `stage.hpp`
7. `types.hpp`

Those files give you the core nouns and verbs of the system.

## Why this folder matters

When a codebase gets large, it becomes easy to understand the implementation details but forget the shape of the system. These headers are the map that keeps the implementation honest.

## Pair this with

- implementation files: [`../../src/README.md`](../../src/README.md)
- backend details: [`../../src/backends/README.md`](../../src/backends/README.md)
- architecture reference: [`../../docs/GOLD_STANDARD_FOR_IMPLEMENTATION.md`](../../docs/GOLD_STANDARD_FOR_IMPLEMENTATION.md)
