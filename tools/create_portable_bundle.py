#!/usr/bin/env python3

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile


def run(*command: str) -> None:
    subprocess.run(command, check=True)


def create_archive(stage_dir: pathlib.Path, output_dir: pathlib.Path, archive_path: pathlib.Path, bundle_name: str) -> None:
    tar_executable = shutil.which("tar")
    if tar_executable is not None:
        run(
            tar_executable,
            "-C",
            str(output_dir),
            "-czf",
            str(archive_path),
            bundle_name,
        )
        return

    with tarfile.open(archive_path, "w:gz", dereference=False) as archive:
        archive.add(stage_dir, arcname=bundle_name, recursive=False)
        for root, dirnames, filenames in os.walk(stage_dir, followlinks=False):
            root_path = pathlib.Path(root)
            relative_root = root_path.relative_to(stage_dir)
            for entry_name in sorted([*dirnames, *filenames]):
                entry_path = root_path / entry_name
                arcname = pathlib.Path(bundle_name) / relative_root / entry_name
                archive.add(entry_path, arcname=str(arcname), recursive=False)


def write_portable_runtime_notes(stage_dir: pathlib.Path) -> None:
    notes_path = stage_dir / "PORTABLE_RUNTIME_NOTES.txt"
    notes = """AMD Video Enhancer portable bundle
=================================

This folder is structured to run in-place after extraction.

Run:
  bin/ave
  bin/ave_gui

Compiler/driver helpers bundled with this package:
  bin/ave_model_compile_sweep
  lib/ave/migraphx/bin/ave-migraphx-driver
  lib/ave/migraphx/bin/migraphx-driver
  lib/ave/migraphx/bin/migraphx-hiprtc-driver

What is bundled here:
  - App executables and private runtime libraries
  - Qt plugins needed by the GUI
  - Bundled MiGraphX runtime/compiler userspace libraries
  - The custom MiGraphX driver entrypoints used by this app

What is not bundled because it must come from the host system:
  - AMD kernel driver stack (/dev/kfd, amdgpu kernel modules)
  - A working GPU-visible ROCm-capable environment
  - System services/devices required to access the GPU

If the app starts but ROCm is unavailable, use:
  bin/ave --list-backends

The bundle is verified during packaging with:
  - ldd closure checks
  - bin/ave --list-backends
  - bin/ave_model_compile_sweep --help
  - lib/ave/migraphx/bin/ave-migraphx-driver --help
"""
    notes_path.write_text(notes, encoding="utf-8")


def cleaned_launch_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in (
        "LD_LIBRARY_PATH",
        "QT_PLUGIN_PATH",
        "QT_QPA_PLATFORM_PLUGIN_PATH",
        "AVE_APP_INSTALL_PREFIX",
        "AVE_BUNDLED_MODELS_DIR",
        "AVE_BUNDLED_MIGRAPHX_PREFIX",
    ):
        env.pop(key, None)
    return env


def prepend_env_paths(env: dict[str, str], name: str, entries: list[pathlib.Path]) -> None:
    existing = env.get(name, "")
    values = [str(entry) for entry in entries if entry.exists()]
    if existing:
        values.append(existing)
    if values:
        env[name] = ":".join(values)


def build_app_launch_env(stage_dir: pathlib.Path) -> dict[str, str]:
    env = cleaned_launch_env()

    runtime_dir = stage_dir / "lib" / "ave" / "runtime"
    qt_plugin_dir = stage_dir / "lib" / "ave" / "qt-plugins"
    bundled_models = stage_dir / "share" / "ave" / "models"
    bundled_migraphx = stage_dir / "lib" / "ave" / "migraphx"
    bundled_tools = stage_dir / "libexec" / "ave" / "tools"
    bundled_migraphx_lib = bundled_migraphx / "lib"
    bundled_migraphx_nested_lib = bundled_migraphx_lib / "migraphx" / "lib"

    prepend_env_paths(
        env,
        "LD_LIBRARY_PATH",
        [runtime_dir, bundled_migraphx_lib, bundled_migraphx_nested_lib],
    )
    prepend_env_paths(env, "PATH", [bundled_tools])
    prepend_env_paths(env, "QT_PLUGIN_PATH", [qt_plugin_dir])

    platform_dir = qt_plugin_dir / "platforms"
    if platform_dir.exists():
        env["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(platform_dir)
    if bundled_models.exists():
        env["AVE_BUNDLED_MODELS_DIR"] = str(bundled_models)
    if bundled_migraphx.exists():
        env["AVE_BUNDLED_MIGRAPHX_PREFIX"] = str(bundled_migraphx)
    return env


def build_migraphx_tool_env(stage_dir: pathlib.Path) -> dict[str, str]:
    env = build_app_launch_env(stage_dir)
    migraphx_root = stage_dir / "lib" / "ave" / "migraphx"
    migraphx_bin_dir = migraphx_root / "bin"
    migraphx_lib_dir = migraphx_root / "lib"
    migraphx_nested_lib_dir = migraphx_lib_dir / "migraphx" / "lib"

    prepend_env_paths(env, "PATH", [migraphx_bin_dir])
    prepend_env_paths(env, "LD_LIBRARY_PATH", [migraphx_lib_dir, migraphx_nested_lib_dir])
    if migraphx_root.exists():
        env["AVE_BUNDLED_MIGRAPHX_PREFIX"] = str(migraphx_root)
    return env


def run_smoke_command(target: pathlib.Path, args: list[str], env: dict[str, str]) -> str | None:
    if not target.exists():
        return None
    result = subprocess.run(
        [str(target), *args],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode == 0:
        return None
    output = (result.stdout or "") + (result.stderr or "")
    return f"{target} {' '.join(args)}:\n{output.strip()}"


def verify_portable_bundle(stage_dir: pathlib.Path) -> None:
    if shutil.which("ldd") is None:
        print("[portable-bundle] skipping dependency verification because 'ldd' is not available")
        return

    app_env = build_app_launch_env(stage_dir)
    tool_env = build_migraphx_tool_env(stage_dir)
    app_targets = [
        stage_dir / "libexec" / "ave" / "ave",
        stage_dir / "libexec" / "ave" / "ave_gui",
    ]
    tool_targets = [
        stage_dir / "libexec" / "ave" / "tools" / "ffmpeg",
        stage_dir / "libexec" / "ave" / "tools" / "ffprobe",
        stage_dir / "lib" / "ave" / "migraphx" / "bin" / "migraphx-driver",
        stage_dir / "lib" / "ave" / "migraphx" / "bin" / "migraphx-hiprtc-driver",
    ]
    runtime_targets = sorted((stage_dir / "lib" / "ave" / "runtime").glob("libmigraphx*.so*"))
    bundled_tool_runtime_targets = sorted(
        (stage_dir / "lib" / "ave" / "migraphx").rglob("libmigraphx*.so*")
    )

    failures: list[str] = []
    for target, env in (
        (app_targets, app_env),
        (tool_targets, tool_env),
        (runtime_targets, app_env),
        (bundled_tool_runtime_targets, tool_env),
    ):
        for entry in target:
            if not entry.exists():
                continue
            result = subprocess.run(
                ("ldd", str(entry)),
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
            output = (result.stdout or "") + (result.stderr or "")
            missing = [line.strip() for line in output.splitlines() if "=> not found" in line]
            if result.returncode != 0 or missing:
                details = "\n".join(missing) if missing else output.strip()
                failures.append(f"{entry}:\n{details}")

    smoke_failures = [
        run_smoke_command(stage_dir / "bin" / "ave", ["--list-backends"], app_env),
        run_smoke_command(stage_dir / "bin" / "ave_model_compile_sweep", ["--help"], app_env),
        run_smoke_command(
            stage_dir / "lib" / "ave" / "migraphx" / "bin" / "ave-migraphx-driver",
            ["--help"],
            tool_env,
        ),
    ]
    failures.extend(entry for entry in smoke_failures if entry is not None)

    if failures:
        raise RuntimeError(
            "[portable-bundle] unresolved runtime dependencies detected:\n"
            + "\n\n".join(failures)
        )

    print(f"[portable-bundle] verified runtime dependency closure for {stage_dir}")


def build_bundle(build_dir: pathlib.Path, output_dir: pathlib.Path, bundle_name: str, skip_archive: bool) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_dir = output_dir / bundle_name

    if stage_dir.exists():
        shutil.rmtree(stage_dir)

    run("cmake", "--install", str(build_dir), "--prefix", str(stage_dir))
    write_portable_runtime_notes(stage_dir)
    verify_portable_bundle(stage_dir)

    if skip_archive:
        print(f"[portable-bundle] staged install tree at {stage_dir}")
        return 0

    archive_path = output_dir / f"{bundle_name}.tar.gz"
    if archive_path.exists():
        archive_path.unlink()

    create_archive(stage_dir, output_dir, archive_path, bundle_name)

    print(f"[portable-bundle] staged install tree at {stage_dir}")
    print(f"[portable-bundle] wrote archive {archive_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a self-contained portable AMD Video Enhancer install tree and archive."
    )
    parser.add_argument("--build-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--bundle-name", required=True)
    parser.add_argument("--skip-archive", action="store_true")
    args = parser.parse_args()

    try:
        return build_bundle(args.build_dir.resolve(),
                            args.output_dir.resolve(),
                            args.bundle_name,
                            args.skip_archive)
    except subprocess.CalledProcessError as exc:
        print(f"[portable-bundle] command failed with exit code {exc.returncode}", file=sys.stderr)
        return exc.returncode
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
