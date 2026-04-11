#!/usr/bin/env python3

import argparse
import datetime as dt
import hashlib
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Dict, List, Optional, Tuple


PROJECT_SLUG = "amd-video-enhancer"
AUR_PKGNAME = "amd-video-enhancer-bin"
INSTALL_PREFIX = pathlib.Path("/opt/amd-video-enhancer")
RELEASE_NOTICE = (
    "Beta release. Primarily verified on Arch Linux on a Ryzen 7 7800X3D + Radeon RX 7900 GRE. "
    "Other distro packages remain preview builds and should be validated on the target system."
)
DEFAULT_RELEASE_BASE_URL = "https://github.com/Rolaand-Jayz/AMD-V.E.-/releases/download/{tag}"


def run(*command: str, cwd: Optional[pathlib.Path] = None, env: Optional[Dict[str, str]] = None) -> None:
    subprocess.run(command, check=True, cwd=cwd, env=env)


def capture(
    *command: str,
    suppress_stderr: bool = False,
    cwd: Optional[pathlib.Path] = None,
    env: Optional[Dict[str, str]] = None,
) -> str:
    stderr = subprocess.DEVNULL if suppress_stderr else None
    return subprocess.check_output(
        command,
        universal_newlines=True,
        stderr=stderr,
        cwd=cwd,
        env=env,
    ).strip()


def file_digest(path: pathlib.Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_sha256_sidecar(path: pathlib.Path) -> pathlib.Path:
    digest = file_digest(path, "sha256")
    sidecar = path.with_name(path.name + ".sha256")
    write_text(sidecar, f"{digest}  {path.name}\n")
    return sidecar


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
        date = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d")
        return f"0.1.0-beta.1.git{date}.{short}"


def create_tar_gz_from_directory_contents(source_dir: pathlib.Path, archive_path: pathlib.Path) -> None:
    tar_executable = shutil.which("tar")
    if tar_executable is not None:
        run(tar_executable, "-C", str(source_dir), "-czf", str(archive_path), ".")
        return

    with tarfile.open(archive_path, "w:gz", dereference=False) as archive:
        for entry in sorted(source_dir.iterdir(), key=lambda path: path.name):
            archive.add(entry, arcname=entry.name, recursive=True)


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


def rpm_version_release(version: str) -> Tuple[str, str]:
    if "-alpha." in version:
        base, suffix = version.split("-alpha.", 1)
        return base, f"0.alpha.{suffix}.1"
    if "-beta." in version:
        base, suffix = version.split("-beta.", 1)
        return base, f"0.beta.{suffix}.1"
    return version, "1"


def arch_pkgver(version: str) -> str:
    return version.replace("-", "")


def release_tag_for(version: str) -> str:
    return f"v{version}"


def render_release_base_url(template: str, tag: str) -> str:
    return template.format(tag=tag).rstrip("/")


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


def render_template(template_path: pathlib.Path, values: Dict[str, str]) -> str:
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

        md5_lines = []  # type: List[str]
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

        file_lines = []  # type: List[str]
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
            "RPM_CHANGELOG_DATE": dt.datetime.now(dt.timezone.utc).strftime("%a %b %d %Y"),
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


def create_arch_aur_handoff(
    staged_root: pathlib.Path,
    output_dir: pathlib.Path,
    version: str,
    maintainer: str,
    release_tag: str,
    release_base_url: str,
) -> List[pathlib.Path]:
    if shutil.which("makepkg") is None:
        raise RuntimeError("makepkg is required to generate AUR handoff metadata.")

    payload_archive = output_dir / f"{PROJECT_SLUG}-{version}-archlinux-x86_64-rootfs.tar.gz"
    if payload_archive.exists():
        payload_archive.unlink()
    create_tar_gz_from_directory_contents(staged_root, payload_archive)
    payload_sha256 = file_digest(payload_archive, "sha256")

    workdir = pathlib.Path(tempfile.mkdtemp(prefix="ave-aur-handoff."))
    try:
        template = pathlib.Path(__file__).resolve().parent.parent / "packaging" / "arch" / "PKGBUILD.aur-bin.in"
        pkgbuild = render_template(template, {
            "AUR_PKGNAME": AUR_PKGNAME,
            "ARCH_PKGVER": arch_pkgver(version),
            "ARCH_PKGREL": "1",
            "PACKAGE_MAINTAINER": maintainer,
            "UPSTREAM_VERSION": version,
            "RELEASE_TAG": release_tag,
            "PAYLOAD_ARCHIVE_NAME": payload_archive.name,
            "PAYLOAD_URL": f"{release_base_url}/{payload_archive.name}",
            "PAYLOAD_SHA256": payload_sha256,
        })
        write_text(workdir / "PKGBUILD", pkgbuild)
        srcinfo = capture("makepkg", "--printsrcinfo", cwd=workdir)
        write_text(workdir / ".SRCINFO", srcinfo + "\n")
        write_text(
            workdir / "AUR_SUBMIT_NOTES.txt",
            (
                f"Submit the contents of this bundle to the AUR as '{AUR_PKGNAME}'.\n"
                f"The PKGBUILD expects the release asset '{payload_archive.name}' to be attached to tag '{release_tag}'.\n"
                "This is a binary AUR handoff: it installs the app-private payload under /opt/amd-video-enhancer\n"
                "and keeps the public entrypoints thin under /usr/bin.\n"
            ),
        )

        aur_bundle = output_dir / f"{AUR_PKGNAME}-{arch_pkgver(version)}-aur.tar.gz"
        if aur_bundle.exists():
            aur_bundle.unlink()
        create_tar_gz_from_directory_contents(workdir, aur_bundle)
        return [payload_archive, aur_bundle]
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
    parser.add_argument("--release-tag")
    parser.add_argument(
        "--release-base-url",
        default=DEFAULT_RELEASE_BASE_URL,
        help="Release download URL template or absolute base URL used for generated AUR metadata. Use {tag} in the template if desired.",
    )
    parser.add_argument("--maintainer", default="Rolaand-Jayz <opensource@rolaandjayz.invalid>")
    args = parser.parse_args()
    if args.version is None:
        args.version = git_version()
    if args.release_tag is None:
        args.release_tag = release_tag_for(args.version)
    release_base_url = render_release_base_url(args.release_base_url, args.release_tag)

    staged_root = args.staged_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_payload_root(staged_root)

    artifacts = []  # type: List[pathlib.Path]
    targets = [entry.strip() for entry in args.targets.split(",") if entry.strip()]
    for target in targets:
        if target == "archlinux":
            artifacts.append(create_arch(staged_root, output_dir, args.version, args.maintainer))
            continue
        if target in ("archlinux-aur", "aur"):
            artifacts.extend(
                create_arch_aur_handoff(
                    staged_root,
                    output_dir,
                    args.version,
                    args.maintainer,
                    args.release_tag,
                    release_base_url,
                )
            )
            continue
        if target.startswith("ubuntu-") or target.startswith("debian-"):
            artifacts.append(create_deb(staged_root, output_dir, args.version, target, args.maintainer))
            continue
        if target.startswith("fedora-") or target.startswith("opensuse-") or target.startswith("rocky-") or target.startswith("almalinux-"):
            artifacts.append(create_rpm(staged_root, output_dir, args.version, target, args.maintainer))
            continue
        raise RuntimeError(f"Unsupported target '{target}'.")

    sidecars = []  # type: List[pathlib.Path]
    for artifact in artifacts:
        sidecars.append(write_sha256_sidecar(artifact))

    print(RELEASE_NOTICE)
    for artifact in [*artifacts, *sidecars]:
        print(artifact)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
