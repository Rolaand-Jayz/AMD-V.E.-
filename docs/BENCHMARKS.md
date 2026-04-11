# Benchmarks

## Current benchmark snapshot

Latest recorded release-readiness benchmark run: **2026-04-10**.

Test environment:

- OS: CachyOS / Linux 6.19.11
- CPU: AMD Ryzen 7 7800X3D
- GPU: AMD Radeon RX 7900 GRE (`gfx1100`)
- Binary: `build/ave`
- Input clip: `benchmarks/ave_benchmark_960x540_20s.mp4` (600 frames, 960x540, 30 FPS)
- Backend: MiGraphX with precompiled `.mxr` artifacts from `~/.local/share/ave/models/migraphx/`
- Benchmark driver: `tools/bench_ave.sh`

| Model | Output | Wall time | MiGraphX total | MiGraphX FPS | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| `openproteus-compact-x2` | 1920x1080 | 23.799 s | 11.1084 s | 54.0133 | full-frame exact-shape artifact, direct final encode, 1 tile/frame |
| `clearreality-x4-fast` | 3840x2160 | 49.723 s | 36.7233 s | 16.3384 | exact-shape tiled artifact, direct final encode, 2 tiles/frame |

## How to read the numbers

- **Wall time** is the end-to-end elapsed time reported by the benchmark wrapper, so it includes the app run plus final encode/write overhead.
- **MiGraphX total** and **MiGraphX FPS** come from the backend timing emitted by the app itself.
- The x2 model demonstrates the current fast 1080p delivery path from a 960x540 source.
- The x4 model demonstrates the current lightweight 4K delivery path from the same source clip.

## Reproducing the benchmark

Run the paired benchmark snapshot:

```bash
./tools/bench_upscale_pair.sh
```

Run a single benchmark with explicit paths:

```bash
./tools/bench_ave.sh ./build/ave ./benchmarks/ave_benchmark_960x540_20s.mp4 ./benchmarks/generated/upscale_pair/openproteus-compact-x2_1920x1080.mp4 1920 1080 openproteus-compact-x2
```

The recorded TSV snapshot for this run lives at:

- `benchmarks/history/20260410-release-readiness/results.tsv`
