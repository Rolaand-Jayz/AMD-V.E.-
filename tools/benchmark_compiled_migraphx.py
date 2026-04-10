#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path


TARGETS_480P = [
    (1280, 720),
    (1920, 1080),
    (2560, 1440),
    (3840, 2160),
]

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


@dataclass
class ModelEntry:
    model_id: str
    stage: str
    fmt: str
    scale: int
    fps_mul: float


@dataclass
class RunResult:
    model_id: str
    stage: str
    backend: str
    input_profile: str
    target_width: int
    target_height: int
    status: str
    reason: str
    seconds: float | None
    input_frames: int | None
    clip_seconds: float | None
    throughput_fps: float | None
    realtime_factor: float | None
    actual_width: int | None
    actual_height: int | None
    output_path: str | None
    log_path: str | None
    artifact_path: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark only already-compiled MiGraphX models.")
    parser.add_argument("--binary", default="build/ave")
    parser.add_argument("--catalog", default="src/model_catalog.cpp")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--duration", type=int, default=30)
    parser.add_argument("--max-runs", type=int, default=0, help="Limit runs for debugging (0 = all).")
    parser.add_argument("--model-regex", default="", help="Only benchmark model IDs matching this regex.")
    return parser.parse_args()


def parse_catalog(catalog_path: Path) -> list[ModelEntry]:
    text = catalog_path.read_text(encoding="utf-8")
    blocks = re.findall(r"\{\n(.*?)\n\s*\},", text, re.S)
    entries: list[ModelEntry] = []

    def str_field(block: str, label: str) -> str:
        match = re.search(rf"/\*\s*{re.escape(label)}\s*\*/\s*\"(?P<value>[^\"]*)\"", block)
        return match.group("value") if match else ""

    def enum_field(block: str, label: str, prefix: str) -> str:
        match = re.search(rf"/\*\s*{re.escape(label)}\s*\*/\s*{re.escape(prefix)}(?P<value>[A-Za-z0-9_]+)", block)
        return match.group("value") if match else ""

    def num_field(block: str, label: str, fallback: str) -> str:
        match = re.search(rf"/\*\s*{re.escape(label)}\s*\*/\s*(?P<value>[0-9]+(?:\.[0-9]+)?)", block)
        return match.group("value") if match else fallback

    for block in blocks:
        model_id = str_field(block, "id")
        if not model_id:
            continue
        entries.append(
            ModelEntry(
                model_id=model_id,
                stage=enum_field(block, "stage", "StageKind::"),
                fmt=enum_field(block, "format", "ModelFormat::"),
                scale=int(num_field(block, "scale", "1")),
                fps_mul=float(num_field(block, "fpsMul", "1.0")),
            )
        )
    return entries


def compiled_artifacts(migraphx_dir: Path) -> dict[str, Path]:
    grouped: dict[str, list[Path]] = {}
    for path in migraphx_dir.glob("*.mxr"):
        base = path.stem
        base = re.sub(r"_(\d+)x(\d+)(_b\d+)?_fp16$", "", base)
        base = re.sub(r"_fp16$", "", base)
        grouped.setdefault(base, []).append(path)

    chosen: dict[str, Path] = {}
    for model_id, paths in grouped.items():
        def score(path: Path) -> tuple[int, int]:
            name = path.stem
            batch_match = re.search(r"_b(\d+)_fp16$", name)
            batch = int(batch_match.group(1)) if batch_match else 1
            dims_match = re.search(r"_(\d+)x(\d+)", name)
            area = int(dims_match.group(1)) * int(dims_match.group(2)) if dims_match else 0
            return (batch, area)

        chosen[model_id] = max(paths, key=score)
    return chosen


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
    if "/" in rate:
        num_s, den_s = rate.split("/", 1)
        den = float(den_s)
        return float(num_s) / den if den else 0.0
    return float(rate)


def make_clip(path: Path, width: int, height: int, duration: int) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    filter_chain = (
        f"format=yuv420p,"
        f"drawgrid=width=64:height=64:thickness=1:color=white@0.18,"
        f"drawbox=x='40 + 120*sin(2*PI*t/5)':y='h-180':w=180:h=90:color=yellow@0.25:t=fill,"
        f"drawbox=x='(w-220)/2 + (w-220)/4*sin(2*PI*t/6)':y='h*0.18':w=220:h=120:color=black@0.32:t=fill"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size={width}x{height}:rate=30:duration={duration}",
        "-vf",
        filter_chain,
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
        str(path),
    ]
    subprocess.run(cmd, check=True)


def make_stage_spec(entry: ModelEntry, artifact_path: Path, input_fps: float, target_width: int, target_height: int) -> str:
    cli_stage = STAGE_TO_CLI[entry.stage]
    params = [
        f"model_path={artifact_path}",
        "model_path_explicit=1",
    ]
    if entry.stage == "Upscale":
        params.append(f"width={target_width}")
        params.append(f"height={target_height}")
    elif entry.stage == "Interpolate":
        target_fps = max(1, round(input_fps * entry.fps_mul))
        params.append(f"fps={target_fps}")
    elif entry.stage == "Stereo3D":
        params.append("format=full_sbs")
    return f"{cli_stage}:{','.join(params)}"


def output_suffix(entry: ModelEntry) -> str:
    return ".mkv" if entry.stage == "Stereo3D" else ".mp4"


def run_benchmark(
    binary: Path,
    entry: ModelEntry,
    artifact_path: Path,
    clip_path: Path,
    input_profile: str,
    target_width: int,
    target_height: int,
    output_root: Path,
) -> RunResult:
    clip_info = ffprobe_stream(clip_path)
    input_frames = int(clip_info.get("nb_frames") or "0")
    clip_seconds = float(clip_info.get("duration") or "0")
    input_fps = parse_rate(clip_info.get("avg_frame_rate") or clip_info.get("r_frame_rate") or "30/1")

    stage_spec = make_stage_spec(entry, artifact_path, input_fps, target_width, target_height)
    name = f"{entry.model_id}_{input_profile}_{target_height}p"
    log_path = output_root / "logs" / f"{name}.log"
    output_path = output_root / "outputs" / entry.stage / f"{name}{output_suffix(entry)}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = "0"
    env["ROCR_VISIBLE_DEVICES"] = "0"
    env["AVE_MIGRAPHX_VISIBLE_DEVICES"] = "0"
    env["MIOPEN_FIND_MODE"] = "FAST"
    env["MIOPEN_COMPILE_PARALLEL_LEVEL"] = "1"

    cmd = [
        str(binary),
        "--input",
        str(clip_path),
        "--output",
        str(output_path),
        "--backend",
        "migraphx",
        "--preview",
        "--preview-duration",
        str(int(clip_seconds)),
        "--stage",
        stage_spec,
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
        tail = (proc.stderr or proc.stdout).strip().splitlines()
        reason = tail[-1] if tail else f"exit code {proc.returncode}"
        return RunResult(
            model_id=entry.model_id,
            stage=entry.stage,
            backend="migraphx",
            input_profile=input_profile,
            target_width=target_width,
            target_height=target_height,
            status="failed",
            reason=reason,
            seconds=elapsed,
            input_frames=input_frames or None,
            clip_seconds=clip_seconds or None,
            throughput_fps=(input_frames / elapsed) if input_frames and elapsed > 0 else None,
            realtime_factor=(clip_seconds / elapsed) if clip_seconds > 0 and elapsed > 0 else None,
            actual_width=None,
            actual_height=None,
            output_path=None,
            log_path=str(log_path),
            artifact_path=str(artifact_path),
        )

    actual_width = None
    actual_height = None
    status = "ok"
    reason = ""
    try:
        output_info = ffprobe_stream(output_path)
        actual_width = int(output_info["width"])
        actual_height = int(output_info["height"])
        if actual_width != target_width or actual_height != target_height:
            status = "mismatch"
            reason = f"actual output {actual_width}x{actual_height}"
    except Exception as exc:  # pragma: no cover - best-effort reporting
        status = "failed"
        reason = f"ffprobe failed: {exc}"

    return RunResult(
        model_id=entry.model_id,
        stage=entry.stage,
        backend="migraphx",
        input_profile=input_profile,
        target_width=target_width,
        target_height=target_height,
        status=status,
        reason=reason,
        seconds=elapsed,
        input_frames=input_frames or None,
        clip_seconds=clip_seconds or None,
        throughput_fps=(input_frames / elapsed) if input_frames and elapsed > 0 else None,
        realtime_factor=(clip_seconds / elapsed) if clip_seconds > 0 and elapsed > 0 else None,
        actual_width=actual_width,
        actual_height=actual_height,
        output_path=str(output_path),
        log_path=str(log_path),
        artifact_path=str(artifact_path),
    )


def write_reports(output_root: Path, results: list[RunResult]) -> None:
    reports = output_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    json_path = reports / "compiled_migraphx_results.json"
    csv_path = reports / "compiled_migraphx_results.csv"
    md_path = reports / "compiled_migraphx_results.md"

    json_path.write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RunResult.__dataclass_fields__.keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))

    lines = [
        "# Compiled MiGraphX Benchmark",
        "",
        "| Model | Stage | Input | Target | Actual | Status | Runtime s | Throughput FPS | Realtime x | Artifact | Log | Notes |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for result in results:
        lines.append(
            f"| `{result.model_id}` | {result.stage} | {result.input_profile} | "
            f"{result.target_width}x{result.target_height} | "
            f"{result.actual_width if result.actual_width else '—'}x{result.actual_height if result.actual_height else '—'} | "
            f"{result.status} | "
            f"{fmt(result.seconds)} | "
            f"{fmt(result.throughput_fps)} | "
            f"{fmt(result.realtime_factor)} | "
            f"`{result.artifact_path}` | `{result.log_path}` | {result.reason or '—'} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "—"


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    binary = (repo_root / args.binary).resolve() if not Path(args.binary).is_absolute() else Path(args.binary)
    catalog_path = (repo_root / args.catalog).resolve() if not Path(args.catalog).is_absolute() else Path(args.catalog)
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    migraphx_dir = Path.home() / ".local/share/ave/models/migraphx"
    artifacts = compiled_artifacts(migraphx_dir)
    catalog = {entry.model_id: entry for entry in parse_catalog(catalog_path)}

    clip_480 = output_root / "inputs" / "benchmark_854x480_30s.mp4"
    clip_720 = output_root / "inputs" / "benchmark_1280x720_30s.mp4"
    make_clip(clip_480, 854, 480, args.duration)
    make_clip(clip_720, 1280, 720, args.duration)

    results: list[RunResult] = []
    actual_runs = 0

    compiled_entries = [catalog[model_id] for model_id in artifacts if model_id in catalog]
    if args.model_regex:
        pattern = re.compile(args.model_regex)
        compiled_entries = [entry for entry in compiled_entries if pattern.search(entry.model_id)]

    stage_order = {
        "Upscale": 0,
        "RestoreCompression": 1,
        "RemoveArtifacts": 2,
        "Denoise": 3,
        "Deblur": 4,
        "Dehalo": 5,
        "ColorFix": 6,
        "Stereo3D": 7,
        "Interpolate": 8,
    }
    compiled_entries.sort(key=lambda item: (stage_order.get(item.stage, 99), item.model_id))

    for entry in compiled_entries:
        artifact_path = artifacts[entry.model_id]
        if entry.stage == "Upscale":
            run_targets = TARGETS_480P
            clip_path = clip_480
            input_profile = "480p30"
        else:
            run_targets = [(1280, 720)]
            clip_path = clip_720
            input_profile = "720p30"

        for target_width, target_height in run_targets:
            if args.max_runs and actual_runs >= args.max_runs:
                break
            print(
                f"[benchmark] {entry.model_id} ({entry.stage}) {input_profile} -> {target_width}x{target_height}",
                flush=True,
            )
            result = run_benchmark(binary, entry, artifact_path, clip_path, input_profile, target_width, target_height, output_root)
            results.append(result)
            actual_runs += 1
            write_reports(output_root, results)
            print(
                f"[benchmark] -> {result.status} runtime={fmt(result.seconds)}s fps={fmt(result.throughput_fps)} rt={fmt(result.realtime_factor)}x",
                flush=True,
            )
        if args.max_runs and actual_runs >= args.max_runs:
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
