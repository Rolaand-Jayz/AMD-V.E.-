#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


@dataclass
class ModelEntry:
    model_id: str
    stage: str
    fmt: str
    scale: int
    download_url: str
    filename: str
    filename_aux: str


@dataclass
class RunResult:
    model_id: str
    stage: str
    backend: str
    fmt: str
    scale: int
    status: str
    reason: str
    seconds: float | None
    input_width: int | None
    input_height: int | None
    input_fps: float | None
    input_frames: int | None
    clip_seconds: float | None
    output_path: str | None
    width: int | None
    height: int | None
    output_fps: float | None
    throughput_fps: float | None
    realtime_factor: float | None
    input_mp_per_sec: float | None
    output_mp_per_sec: float | None
    log_path: str | None


STAGE_TO_CLI = {
    "RestoreCompression": "restore_compression",
    "RemoveArtifacts": "remove_artifacts",
    "Denoise": "denoise",
    "Deblur": "deblur",
    "Dehalo": "dehalo",
    "ColorFix": "color_fix",
    "Upscale": "upscale",
    "Sharpen": "sharpen",
    "Stereo3D": "stereo_3d",
    "Interpolate": "interpolate",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark AVE catalog models on a 10-second clip.")
    parser.add_argument("--binary", default="build/ave")
    parser.add_argument("--catalog", default="src/model_catalog.cpp")
    parser.add_argument("--input", required=True, help="Source video used to create the 10-second benchmark clip.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--clip-duration", type=float, default=10.0)
    parser.add_argument("--gpu-index", type=int, default=None)
    parser.add_argument("--max-runs", type=int, default=0, help="Limit actual benchmark runs for debugging (0 = all).")
    return parser.parse_args()


def parse_catalog(catalog_path: Path) -> list[ModelEntry]:
    text = catalog_path.read_text(encoding="utf-8")
    blocks = re.findall(r"\{\n(.*?)\n\s*\},", text, re.S)
    entries: list[ModelEntry] = []

    def field(block: str, label: str, prefix: str = "") -> str:
        if prefix:
            pattern = rf"/\*\s*{re.escape(label)}\s*\*/\s*{re.escape(prefix)}(?P<value>[A-Za-z0-9_]+)"
        else:
            pattern = rf"/\*\s*{re.escape(label)}\s*\*/\s*\"(?P<value>[^\"]*)\""
        match = re.search(pattern, block)
        return match.group("value") if match else ""

    for block in blocks:
        model_id = field(block, "id")
        if not model_id:
            continue
        scale_match = re.search(r"/\*\s*scale\s*\*/\s*(?P<value>\d+)", block)
        entries.append(
            ModelEntry(
                model_id=model_id,
                stage=field(block, "stage", "StageKind::"),
                fmt=field(block, "format", "ModelFormat::"),
                scale=int(scale_match.group("value")) if scale_match else 1,
                download_url=field(block, "downloadUrl"),
                filename=field(block, "filename"),
                filename_aux=field(block, "filenameAux"),
            )
        )
    return entries


def ffprobe_stream(path: Path) -> dict[str, str]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames:format=duration",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    payload = json.loads(proc.stdout)
    stream = payload["streams"][0]
    if "format" in payload:
        stream["duration"] = payload["format"].get("duration", "0")
    return stream


def parse_rate(rate: str) -> float:
    if not rate:
        return 0.0
    if "/" in rate:
        num_s, den_s = rate.split("/", 1)
        num = float(num_s)
        den = float(den_s)
        return num / den if den else 0.0
    return float(rate)


def parse_int_maybe(value: str | None) -> int | None:
    if not value or value == "N/A":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def parse_float_maybe(value: str | None) -> float | None:
    if not value or value == "N/A":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def ensure_benchmark_clip(source: Path, clip_path: Path, duration: float) -> dict[str, str]:
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        "0",
        "-i",
        str(source),
        "-t",
        str(duration),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(clip_path),
    ]
    subprocess.run(cmd, check=True)
    return ffprobe_stream(clip_path)


def compatible_backend(entry: ModelEntry) -> tuple[str | None, str | None]:
    if not entry.filename or not entry.download_url:
        return None, "non-model catalog entry"
    if entry.fmt == "NcnnBin":
        return "ncnn", None
    if entry.fmt in {"Onnx", "Pytorch"}:
        return "migraphx", None
    return None, f"unsupported catalog format {entry.fmt}"


def stage_spec(entry: ModelEntry, width: int, height: int, fps_num: int, fps_den: int) -> str:
    cli_stage = STAGE_TO_CLI[entry.stage]
    parts = [cli_stage, f"model={entry.model_id}"]
    if entry.stage == "Upscale" and entry.scale > 1:
        parts.append(f"width={width * entry.scale}")
        parts.append(f"height={height * entry.scale}")
    elif entry.stage == "Interpolate":
        target_fps = max(1, round((fps_num / fps_den) * max(1, entry.scale)))
        parts.append(f"fps={target_fps}")
    elif entry.stage == "Stereo3D":
        parts.append("format=full_sbs")
    return ":".join([parts[0], ",".join(parts[1:])])


def make_output_name(entry: ModelEntry) -> str:
    suffix = ".mp4"
    if entry.stage == "Stereo3D":
        suffix = ".mkv"
    return f"{entry.model_id}{suffix}"


def run_benchmark(
    binary: Path,
    clip_path: Path,
    output_root: Path,
    entry: ModelEntry,
    backend: str,
    width: int,
    height: int,
    fps_num: int,
    fps_den: int,
    input_fps: float,
    input_frames: int | None,
    clip_seconds: float,
    env: dict[str, str],
) -> RunResult:
    backend_dir = output_root / "outputs" / backend / entry.stage
    log_dir = output_root / "logs" / backend / entry.stage
    backend_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    output_path = backend_dir / make_output_name(entry)
    log_path = log_dir / f"{entry.model_id}.log"
    spec = stage_spec(entry, width, height, fps_num, fps_den)
    cmd = [
        str(binary),
        "--input",
        str(clip_path),
        "--output",
        str(output_path),
        "--backend",
        backend,
        "--stage",
        spec,
        "--preview",
        "--preview-duration",
        "10",
    ]

    started = time.perf_counter()
    proc = subprocess.run(cmd, text=True, capture_output=True, env=env)
    elapsed = time.perf_counter() - started
    log_path.write_text(
        "COMMAND: " + " ".join(shlex.quote(part) for part in cmd) + "\n\n"
        + proc.stdout
        + ("\n" if proc.stdout and not proc.stdout.endswith("\n") else "")
        + proc.stderr,
        encoding="utf-8",
    )

    if proc.returncode != 0:
        reason = f"exit code {proc.returncode}"
        stderr = (proc.stderr or proc.stdout).strip().splitlines()
        if stderr:
            reason += f": {stderr[-1]}"
        return RunResult(
            model_id=entry.model_id,
            stage=entry.stage,
            backend=backend,
            fmt=entry.fmt,
            scale=entry.scale,
            status="failed",
            reason=reason,
            seconds=elapsed,
            input_width=width,
            input_height=height,
            input_fps=input_fps,
            input_frames=input_frames,
            clip_seconds=clip_seconds,
            output_path=None,
            width=None,
            height=None,
            output_fps=None,
            throughput_fps=(input_frames / elapsed) if input_frames and elapsed > 0 else None,
            realtime_factor=(clip_seconds / elapsed) if clip_seconds > 0 and elapsed > 0 else None,
            input_mp_per_sec=((width * height * (input_frames / elapsed)) / 1_000_000.0) if input_frames and elapsed > 0 else None,
            output_mp_per_sec=None,
            log_path=str(log_path),
        )

    stream = ffprobe_stream(output_path)
    output_fps = parse_rate(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0")
    throughput_fps = (input_frames / elapsed) if input_frames and elapsed > 0 else None
    output_mp_per_sec = None
    if throughput_fps is not None:
        output_mp_per_sec = ((int(stream["width"]) * int(stream["height"]) * throughput_fps) / 1_000_000.0)
    return RunResult(
        model_id=entry.model_id,
        stage=entry.stage,
        backend=backend,
        fmt=entry.fmt,
        scale=entry.scale,
        status="ok",
        reason="",
        seconds=elapsed,
        input_width=width,
        input_height=height,
        input_fps=input_fps,
        input_frames=input_frames,
        clip_seconds=clip_seconds,
        output_path=str(output_path),
        width=int(stream["width"]),
        height=int(stream["height"]),
        output_fps=output_fps,
        throughput_fps=throughput_fps,
        realtime_factor=(clip_seconds / elapsed) if clip_seconds > 0 and elapsed > 0 else None,
        input_mp_per_sec=((width * height * throughput_fps) / 1_000_000.0) if throughput_fps is not None else None,
        output_mp_per_sec=output_mp_per_sec,
        log_path=str(log_path),
    )


def format_metric(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return str(value)
    if math.isfinite(value):
        return f"{value:.{digits}f}"
    return "—"


def write_markdown_report(repo_root: Path, output_root: Path, clip_path: Path, results: list[RunResult]) -> None:
    lines: list[str] = []
    lines.append("# Benchmark")
    lines.append("")
    lines.append("This file is the canonical benchmark report for the repository. It is regenerated by `tools/benchmark_model_matrix.py`.")
    lines.append("")
    lines.append("## Benchmark requirements")
    lines.append("")
    lines.append("- Every benchmark run must append or refresh all relevant performance data here.")
    lines.append("- The mega chart below must keep all primary variables in one table for quick comparison.")
    lines.append("- The detailed access section must preserve each run's raw performance numbers, paths, and failure reason when applicable.")
    lines.append("- Supporting machine-readable artifacts remain under the external output root in `reports/`, `logs/`, and `outputs/`.")
    lines.append("")
    lines.append("## Run context")
    lines.append("")
    lines.append(f"- Repository root: `{repo_root}`")
    lines.append(f"- Benchmark clip: `{clip_path}`")
    lines.append(f"- External output root: `{output_root}`")
    lines.append(f"- Total recorded entries: {len(results)}")
    lines.append("")
    lines.append("## Mega chart")
    lines.append("")
    lines.append("| Model | Stage | Backend | Format | Scale | Status | Runtime s | Clip s | Input | Output | Input FPS | Output FPS | Throughput FPS | Realtime x | Input MP/s | Output MP/s | Output file | Log | Notes |")
    lines.append("| --- | --- | --- | --- | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |")
    for result in results:
        input_dims = f"{result.input_width}x{result.input_height}" if result.input_width and result.input_height else "—"
        output_dims = f"{result.width}x{result.height}" if result.width and result.height else "—"
        output_file = f"`{result.output_path}`" if result.output_path else "—"
        log_file = f"`{result.log_path}`" if result.log_path else "—"
        notes = result.reason or ""
        lines.append(
            f"| `{result.model_id}` | {result.stage} | {result.backend} | {result.fmt} | {result.scale} | {result.status} | "
            f"{format_metric(result.seconds)} | {format_metric(result.clip_seconds)} | {input_dims} | {output_dims} | "
            f"{format_metric(result.input_fps)} | {format_metric(result.output_fps)} | {format_metric(result.throughput_fps)} | "
            f"{format_metric(result.realtime_factor)} | {format_metric(result.input_mp_per_sec)} | {format_metric(result.output_mp_per_sec)} | "
            f"{output_file} | {log_file} | {notes} |"
        )
    lines.append("")
    lines.append("## Detailed access")
    lines.append("")
    lines.append("Use this section to access each benchmark's individual performance numbers without leaving the repo root report.")
    lines.append("")
    for result in results:
        lines.append(f"### {result.model_id}")
        lines.append("")
        lines.append(f"- Stage: `{result.stage}`")
        lines.append(f"- Backend: `{result.backend}`")
        lines.append(f"- Format: `{result.fmt}`")
        lines.append(f"- Scale: `{result.scale}`")
        lines.append(f"- Status: `{result.status}`")
        if result.reason:
            lines.append(f"- Reason: {result.reason}")
        lines.append(f"- Runtime seconds: {format_metric(result.seconds)}")
        lines.append(f"- Clip seconds: {format_metric(result.clip_seconds)}")
        lines.append(f"- Input resolution: {format_metric(result.input_width, 0)}x{format_metric(result.input_height, 0)}" if result.input_width and result.input_height else "- Input resolution: —")
        lines.append(f"- Output resolution: {format_metric(result.width, 0)}x{format_metric(result.height, 0)}" if result.width and result.height else "- Output resolution: —")
        lines.append(f"- Input FPS: {format_metric(result.input_fps)}")
        lines.append(f"- Output FPS: {format_metric(result.output_fps)}")
        lines.append(f"- Throughput FPS: {format_metric(result.throughput_fps)}")
        lines.append(f"- Realtime factor: {format_metric(result.realtime_factor)}")
        lines.append(f"- Input MP/s: {format_metric(result.input_mp_per_sec)}")
        lines.append(f"- Output MP/s: {format_metric(result.output_mp_per_sec)}")
        lines.append(f"- Output file: `{result.output_path}`" if result.output_path else "- Output file: —")
        lines.append(f"- Log file: `{result.log_path}`" if result.log_path else "- Log file: —")
        lines.append("")

    (repo_root / "benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_results(repo_root: Path, output_root: Path, clip_path: Path, results: Iterable[RunResult]) -> None:
    results = list(results)
    (output_root / "reports").mkdir(parents=True, exist_ok=True)
    json_path = output_root / "reports" / "benchmark_results.json"
    csv_path = output_root / "reports" / "benchmark_results.csv"
    json_path.write_text(json.dumps([asdict(result) for result in results], indent=2), encoding="utf-8")
    fieldnames = list(RunResult.__dataclass_fields__.keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()) if results else fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))
    write_markdown_report(repo_root, output_root, clip_path, results)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    binary = (repo_root / args.binary).resolve() if not Path(args.binary).is_absolute() else Path(args.binary)
    catalog = (repo_root / args.catalog).resolve() if not Path(args.catalog).is_absolute() else Path(args.catalog)
    source = Path(args.input).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    clip_path = output_root / "inputs" / "benchmark_10s.mp4"
    clip_info = ensure_benchmark_clip(source, clip_path, args.clip_duration)
    width = int(clip_info["width"])
    height = int(clip_info["height"])
    fps_num, fps_den = [int(part) for part in clip_info["r_frame_rate"].split("/")]
    input_fps = parse_rate(clip_info.get("avg_frame_rate") or clip_info.get("r_frame_rate") or "0")
    input_frames = parse_int_maybe(clip_info.get("nb_frames"))
    clip_seconds = parse_float_maybe(clip_info.get("duration")) or args.clip_duration

    env = os.environ.copy()
    if args.gpu_index is not None:
        env["AVE_GPU_INDEX"] = str(args.gpu_index)

    results: list[RunResult] = []
    actual_runs = 0
    for entry in parse_catalog(catalog):
        backend, skip_reason = compatible_backend(entry)
        if backend is None:
            results.append(
                RunResult(
                    model_id=entry.model_id,
                    stage=entry.stage,
                    backend="skipped",
                    fmt=entry.fmt,
                    scale=entry.scale,
                    status="skipped",
                    reason=skip_reason or "skipped",
                    seconds=None,
                    input_width=width,
                    input_height=height,
                    input_fps=input_fps,
                    input_frames=input_frames,
                    clip_seconds=clip_seconds,
                    output_path=None,
                    width=None,
                    height=None,
                    output_fps=None,
                    throughput_fps=None,
                    realtime_factor=None,
                    input_mp_per_sec=None,
                    output_mp_per_sec=None,
                    log_path=None,
                )
            )
            continue

        if args.max_runs and actual_runs >= args.max_runs:
            results.append(
                RunResult(
                    model_id=entry.model_id,
                    stage=entry.stage,
                    backend=backend,
                    fmt=entry.fmt,
                    scale=entry.scale,
                    status="skipped",
                    reason=f"max-runs limit {args.max_runs} reached",
                    seconds=None,
                    input_width=width,
                    input_height=height,
                    input_fps=input_fps,
                    input_frames=input_frames,
                    clip_seconds=clip_seconds,
                    output_path=None,
                    width=None,
                    height=None,
                    output_fps=None,
                    throughput_fps=None,
                    realtime_factor=None,
                    input_mp_per_sec=None,
                    output_mp_per_sec=None,
                    log_path=None,
                )
            )
            continue

        print(f"[benchmark] {entry.model_id} ({entry.stage}) via {backend}", flush=True)
        result = run_benchmark(binary, clip_path, output_root, entry, backend, width, height, fps_num, fps_den, input_fps, input_frames, clip_seconds, env)
        results.append(result)
        actual_runs += 1
        write_results(repo_root, output_root, clip_path, results)
        print(f"[benchmark] -> {result.status} {result.reason}".rstrip(), flush=True)

    write_results(repo_root, output_root, clip_path, results)
    ok = sum(1 for result in results if result.status == "ok")
    failed = sum(1 for result in results if result.status == "failed")
    skipped = sum(1 for result in results if result.status == "skipped")
    print(f"[benchmark] complete: ok={ok} failed={failed} skipped={skipped}")
    print(output_root / "reports" / "benchmark_results.json")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
