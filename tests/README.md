# Test Suite Guide

This folder contains the project's C++ test executables. The repo keeps tests close to the subsystem they validate instead of hiding everything behind one giant test binary.

## What the tests are trying to do

The tests here are meant to prove that the important plumbing still works:

- planner ordering
- tensor contracts
- runtime diagnostics and paths
- model-manager profile behavior
- media probing
- process loops and observers
- frame I/O paths
- FFmpeg runner behavior
- job queue and recovery logic
- telemetry
- MiGraphX backend behavior
- interop bridge behavior

## How the folder is organized

Most files follow a simple pattern: one executable per subsystem.

| Test file | What it checks |
| --- | --- |
| `planner_tests.cpp` | deterministic pipeline ordering |
| `tensor_contract_tests.cpp` | tensor shape/type contract checks |
| `runtime_*_tests.cpp` | environment and path diagnostics |
| `model_manager_profile_tests.cpp` | model manager profile behavior |
| `video_probe_tests.cpp` | media probing logic |
| `process_*_tests.cpp` | observer, progress, and process-loop behavior |
| `frame_io_*_tests.cpp` | frame movement and frame I/O variants |
| `ffmpeg_runner_tests.cpp` | FFmpeg orchestration behavior |
| `job_*_tests.cpp` | queueing and recovery flows |
| `telemetry_tests.cpp` | runtime telemetry behavior |
| `migraphx_backend_tests.cpp` | MiGraphX backend behavior |
| `interop_bridge_tests.cpp` | Vulkan↔HIP interop bridge behavior |

## How to run the tests

Run everything:

```bash
ctest --test-dir build --output-on-failure
```

Run one subsystem:

```bash
ctest --test-dir build -R planner_tests --output-on-failure
```

You can also run the produced test binaries directly from `build/` if you are debugging a specific failure.

## How to read these tests

Start with the test that matches the subsystem you are modifying, then read the production file it exercises. This repo is much easier to understand when tests and implementation are read together.

## Pair this with

- implementation map: [`../src/README.md`](../src/README.md)
- architecture reference: [`../docs/GOLD_STANDARD_FOR_IMPLEMENTATION.md`](../docs/GOLD_STANDARD_FOR_IMPLEMENTATION.md)
