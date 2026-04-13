# Source Tree Guide

This folder is the implementation body of AMD Video Enhancer. If you want to know where the app actually makes decisions, talks to backends, moves frames, manages models, or updates the GUI, this is where you live.

## Big picture

The `src/` tree is organized around a few core jobs:

| Area | Main files | What they do |
| --- | --- | --- |
| entrypoints | `main.cpp`, `cli.cpp`, `gui/main_gui.cpp` | start the CLI or GUI and translate user intent into app actions |
| pipeline orchestration | `video_processor.cpp`, `planner.cpp`, `stage.cpp`, `types.cpp` | decide stage order, resolve work, and run jobs end to end |
| backend selection | `backend.cpp`, `backend_manager.cpp` | describe backends and choose the one that will run |
| model lifecycle | `model_catalog.cpp`, `model_manager.cpp`, `tensor_contract.cpp` | track models, prepare them, compile MiGraphX artifacts, and validate tensor shapes/contracts |
| media pipeline | `ffmpeg_runner.cpp`, `frame_io.cpp`, `frame_io_vulkan.cpp`, `rgb_video_loop.cpp`, `video_probe.cpp` | probe media, move frames, and keep the video path working around the AI path |
| runtime + diagnostics | `runtime_paths.cpp`, `runtime_diagnostics.cpp`, `telemetry.cpp`, `observability.cpp`, `process_*.cpp` | explain the environment, record state, and make the app debuggable |
| job state | `job_queue.cpp`, `job_recovery.cpp`, `app_settings.cpp` | persist settings and support queue/recovery flows |
| backends | `backends/` | backend-specific execution code |
| GUI | `gui/` | Qt widgets, dialogs, and UI wiring |

## Read this folder in a useful order

If you are trying to learn the code instead of just searching for a symbol, read in this order:

1. `video_processor.cpp`
2. `backend_manager.cpp`
3. `planner.cpp`
4. `model_manager.cpp`
5. `ffmpeg_runner.cpp`
6. the files in [`backends/`](./backends/README.md)
7. the files in [`gui/`](./gui/README.md) if you care about the desktop UI

That order mirrors the actual shape of the app: orchestration first, then backends, then UI.

## Where to go next

- backend deep dive: [`backends/README.md`](./backends/README.md)
- GUI map: [`gui/README.md`](./gui/README.md)
- public headers that pair with this code: [`../include/ave/README.md`](../include/ave/README.md)
- implementation truth doc: [`../docs/GOLD_STANDARD_FOR_IMPLEMENTATION.md`](../docs/GOLD_STANDARD_FOR_IMPLEMENTATION.md)

## A good mental model

Think of `src/` as a layered system:

1. user asks for work
2. planner turns that into a safe ordered pipeline
3. model/runtime code decides what can run
4. backend code executes the heavy lifting
5. FFmpeg/media code keeps the video workflow intact
6. diagnostics and telemetry make the whole thing explainable when it breaks

That model will save you a lot of random file hopping.
