#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile


PROJECT_SLUG = "amd-video-enhancer"
INSTALL_PREFIX = pathlib.Path("/opt/amd-video-enhancer")
EXPERIMENTAL_NOTICE = (
    "Open alpha. Only tested on Arch Linux on Ryzen 7 7800X3D + Radeon RX 7900 GRE. "
    "All other distro packages are experimental."
)


def run(*command: str, cwd: pathlib.Path | None = None, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, check=True, cwd=cwd, env=env)


def capture(*command: str, suppress_stderr: bool = False) -> str:
    stderr = subprocess.DEVNULL if suppress_stderr else None
    return subprocess.check_output(command, text=True, stderr=stderr).strip()


def file_digest(path: pathlib.Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_version() -> str:
    try:
        tag = capture("git", "describe", "--tags", "--match", "v*", "--abbrev=0", suppress_stderr=True)
        if tag.startswith("v"):
            return tag[1:]
        return tag
    except (FileNotFoundError, subprocess.CalledProcessError):
        try:
            short = capture("git", "rev-parse", "--short", "HEAD")
        except (FileNotFoundError, subprocess.CalledProcessError):
            short = os.environ.get("AVE_VERSION_FALLBACK_SHA", "nogit")
        date = dt.datetime.now(dt.UTC).strftime("%Y%m%d")
        return f"0.1.0-alpha.1.git{date}.{short}"


def ensure_payload_root(staged_root: pathlib.Path) -> pathlib.Path:
    payload_prefix = staged_root / INSTALL_PREFIX.relative_to("/")
    if not payload_prefix.exists():
        raise RuntimeError(
            f"Expected staged install prefix at {payload_prefix}. "
            "Create it first with tools/stage_release_root.sh."
        )

    usr_bin = staged_root / "usr" / "bin"
    usr_bin.mkdir(parents=True, exist_ok=True)
    for entry in ("ave", "ave_gui", "ave_model_compile_sweep"):
        if not (payload_prefix / "bin" / entry).exists():
            continue
        target = usr_bin / entry
        if target.exists() or target.is_symlink():
            target.unlink()
        write_launcher_script(target, entry)

    if (payload_prefix / "bin" / "ave_gui").exists():
        app_dir = staged_root / "usr" / "share" / "applications"
        app_dir.mkdir(parents=True, exist_ok=True)
        desktop_src = pathlib.Path(__file__).resolve().parent.parent / "packaging" / "common" / "amd-video-enhancer.desktop"
        shutil.copy2(desktop_src, app_dir / "amd-video-enhancer.desktop")

    return payload_prefix


def deb_version(version: str) -> str:
    return version.replace("-alpha.", "~alpha").replace("-beta.", "~beta")


def rpm_version_release(version: str) -> tuple[str, str]:
    if "-alpha." in version:
        base, suffix = version.split("-alpha.", 1)
        return base, f"0.alpha.{suffix}.1"
    if "-beta." in version:
        base, suffix = version.split("-beta.", 1)
        return base, f"0.beta.{suffix}.1"
    return version, "1"


def arch_pkgver(version: str) -> str:
    return version.replace("-", "")


def write_text(path: pathlib.Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_launcher_script(path: pathlib.Path, command_name: str) -> None:
    write_text(
        path,
        (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n\n"
            f'exec "{INSTALL_PREFIX}/bin/{command_name}" "$@"\n'
        ),
    )
    path.chmod(0o755)


def render_template(template_path: pathlib.Path, values: dict[str, str]) -> str:
    content = template_path.read_text(encoding="utf-8")
    for key, value in values.items():
        content = content.replace(f"@{key}@", value)
    return content


def create_deb(staged_root: pathlib.Path, output_dir: pathlib.Path, version: str, distro_label: str, maintainer: str) -> pathlib.Path:
    if shutil.which("dpkg-deb") is None:
        raise RuntimeError("dpkg-deb is required to build Debian/Ubuntu packages.")

    package_root = pathlib.Path(tempfile.mkdtemp(prefix="ave-deb-root."))
    try:
        shutil.copytree(staged_root, package_root, dirs_exist_ok=True, symlinks=True)
        control_dir = package_root / "DEBIAN"
        control_dir.mkdir(parents=True, exist_ok=True)
        template = pathlib.Path(__file__).resolve().parent.parent / "packaging" / "debian" / "control.in"
        control = render_template(template, {
            "DEB_VERSION": deb_version(version),
            "PACKAGE_MAINTAINER": maintainer,
        })
        write_text(control_dir / "control", control)

        md5_lines: list[str] = []
        for file_path in sorted(package_root.rglob("*")):
            if not file_path.is_file():
                continue
            if control_dir in file_path.parents:
                continue
            rel = file_path.relative_to(package_root)
            md5_lines.append(f"{file_digest(file_path, 'md5')}  {rel}")
        write_text(control_dir / "md5sums", "\n".join(md5_lines) + "\n")

        artifact = output_dir / f"{PROJECT_SLUG}_{deb_version(version)}_{distro_label}_amd64.deb"
        if artifact.exists():
            artifact.unlink()
        run(
            "dpkg-deb",
            "--build",
            "--uniform-compression",
            "-Zzstd",
            "-z10",
            str(package_root),
            str(artifact),
        )
        return artifact
    finally:
        shutil.rmtree(package_root, ignore_errors=True)


def create_rpm(staged_root: pathlib.Path, output_dir: pathlib.Path, version: str, distro_label: str, maintainer: str) -> pathlib.Path:
    if shutil.which("rpmbuild") is None:
        raise RuntimeError("rpmbuild is required to build RPM packages.")

    topdir = pathlib.Path(tempfile.mkdtemp(prefix="ave-rpm-topdir."))
    try:
        for name in ("BUILD", "BUILDROOT", "RPMS", "SOURCES", "SPECS", "SRPMS"):
            (topdir / name).mkdir(parents=True, exist_ok=True)

        rpm_ver, rpm_rel = rpm_version_release(version)
        filelist_name = f"{PROJECT_SLUG}-{distro_label}.files"
        filelist_path = topdir / "SOURCES" / filelist_name

        file_lines: list[str] = []
        for path in sorted(staged_root.rglob("*")):
            if path.is_dir():
                continue
            rel = pathlib.Path("/") / path.relative_to(staged_root)
            file_lines.append(str(rel))
        write_text(filelist_path, "\n".join(file_lines) + "\n")

        template = pathlib.Path(__file__).resolve().parent.parent / "packaging" / "rpm" / "amd-video-enhancer.spec.in"
        spec = render_template(template, {
            "RPM_VERSION": rpm_ver,
            "RPM_RELEASE": rpm_rel,
            "RPM_FILELIST_NAME": filelist_name,
            "RPM_CHANGELOG_DATE": dt.datetime.now(dt.UTC).strftime("%a %b %d %Y"),
            "PACKAGE_MAINTAINER": maintainer,
        })
        spec_path = topdir / "SPECS" / "amd-video-enhancer.spec"
        write_text(spec_path, spec)

        run(
            "rpmbuild",
            "--define",
            f"_topdir {topdir}",
            "--define",
            f"staged_root {staged_root}",
            "--define",
            "_binary_payload w3.zstdio",
            "--define",
            "__brp_strip_comment_note %{nil}",
            "-bb",
            str(spec_path),
        )
        produced = next((topdir / "RPMS").rglob("*.rpm"))
        artifact = output_dir / f"{PROJECT_SLUG}-{rpm_ver}-{rpm_rel}.{distro_label}.x86_64.rpm"
        shutil.copy2(produced, artifact)
        return artifact
    finally:
        shutil.rmtree(topdir, ignore_errors=True)


def create_arch(staged_root: pathlib.Path, output_dir: pathlib.Path, version: str, maintainer: str) -> pathlib.Path:
    if shutil.which("makepkg") is None:
        raise RuntimeError("makepkg is required to build Arch packages.")

    workdir = pathlib.Path(tempfile.mkdtemp(prefix="ave-arch-build."))
    try:
        template = pathlib.Path(__file__).resolve().parent.parent / "packaging" / "arch" / "PKGBUILD.in"
        pkgbuild = render_template(template, {
            "ARCH_PKGVER": arch_pkgver(version),
            "ARCH_PKGREL": "1",
        })
        write_text(workdir / "PKGBUILD", pkgbuild)
        env = os.environ.copy()
        env.setdefault("PACKAGER", maintainer)
        env["AVE_STAGED_ROOT"] = str(staged_root)
        run("makepkg", "--nodeps", "--skipinteg", "--force", "--cleanbuild", "--noconfirm", cwd=workdir, env=env)
        produced = next(workdir.glob("*.pkg.tar.zst"))
        artifact = output_dir / produced.name
        shutil.copy2(produced, artifact)
        return artifact
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build native distro packages from a staged release root.")
    parser.add_argument("--staged-root", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--targets",
        default="archlinux,ubuntu-24.04,ubuntu-22.04,debian-12,fedora-41,opensuse-leap-15.6,opensuse-tumbleweed,rocky-9,almalinux-9",
        help="Comma-separated distro targets to emit.",
    )
    parser.add_argument("--version")
    parser.add_argument("--maintainer", default="Rolaand-Jayz <opensource@rolaandjayz.invalid>")
    args = parser.parse_args()
    if args.version is None:
        args.version = git_version()

    staged_root = args.staged_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_payload_root(staged_root)

    artifacts: list[pathlib.Path] = []
    targets = [entry.strip() for entry in args.targets.split(",") if entry.strip()]
    for target in targets:
        if target == "archlinux":
            artifacts.append(create_arch(staged_root, output_dir, args.version, args.maintainer))
            continue
        if target.startswith("ubuntu-") or target.startswith("debian-"):
            artifacts.append(create_deb(staged_root, output_dir, args.version, target, args.maintainer))
            continue
        if target.startswith("fedora-") or target.startswith("opensuse-") or target.startswith("rocky-") or target.startswith("almalinux-"):
            artifacts.append(create_rpm(staged_root, output_dir, args.version, target, args.maintainer))
            continue
        raise RuntimeError(f"Unsupported target '{target}'.")

    print(EXPERIMENTAL_NOTICE)
    for artifact in artifacts:
        print(artifact)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
