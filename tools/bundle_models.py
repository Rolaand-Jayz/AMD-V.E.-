#!/usr/bin/env python3

import argparse
import hashlib
import pathlib
import re
import shutil
import sys
import tempfile
import urllib.request
import zipfile


FIELD_RE = re.compile(r'/\*\s*([A-Za-z0-9_]+)\s*\*/\s*"([^"]*)"')


def parse_catalog(path: pathlib.Path):
    entries = []
    current = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line == "{":
            current = {}
            continue
        if current is None:
            continue

        match = FIELD_RE.search(line)
        if match:
            key, value = match.groups()
            current[key] = value

        if line.startswith("},") or line == "}":
            if current:
                entries.append(current)
            current = None

    return entries


def request_with_user_agent(url: str):
    return urllib.request.Request(url, headers={"User-Agent": "amd-video-enhancer-bundler/1.0"})


def download_file(url: str, destination: pathlib.Path, force: bool):
    if destination.exists() and destination.stat().st_size > 0 and not force:
        print(f"[bundle-models] reuse {destination.name}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temp_file:
        temp_path = pathlib.Path(temp_file.name)
    try:
        print(f"[bundle-models] download {url} -> {destination}")
        with urllib.request.urlopen(request_with_user_agent(url)) as response:
            with temp_path.open("wb") as output:
                shutil.copyfileobj(response, output)
        temp_path.replace(destination)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def cache_path_for_url(cache_dir: pathlib.Path, url: str):
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    suffix = pathlib.Path(url).suffix or ".bin"
    return cache_dir / f"{digest}{suffix}"


def extract_from_archive(archive_path: pathlib.Path, member_name: str, destination: pathlib.Path, force: bool):
    if destination.exists() and destination.stat().st_size > 0 and not force:
        print(f"[bundle-models] reuse {destination.name}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"[bundle-models] extract {member_name} -> {destination}")
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(member_name) as source, tempfile.NamedTemporaryFile(
            dir=destination.parent, delete=False
        ) as temp_file:
            temp_path = pathlib.Path(temp_file.name)
            with temp_path.open("wb") as output:
                shutil.copyfileobj(source, output)
    try:
        temp_path.replace(destination)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def bundle_entries(entries, output_dir: pathlib.Path, cache_dir: pathlib.Path, force: bool, dry_run: bool):
    downloaded_dir = output_dir / "downloaded"
    seen_downloads = set()

    for entry in entries:
        download_url = entry.get("downloadUrl", "")
        filename = entry.get("filename", "")
        if not download_url or not filename:
            continue

        archive_sub_path = entry.get("archiveSubPath", "")
        archive_sub_path_aux = entry.get("archiveSubPathAux", "")
        aux_url = entry.get("dlUrlAux", "") or entry.get("downloadUrlAux", "")
        filename_aux = entry.get("filenameAux", "")

        primary_destination = downloaded_dir / filename
        aux_destination = downloaded_dir / filename_aux if filename_aux else None

        if dry_run:
            print(f"[bundle-models] would stage {entry.get('id', filename)}")
            continue

        if archive_sub_path:
            archive_path = cache_path_for_url(cache_dir, download_url)
            if download_url not in seen_downloads:
                download_file(download_url, archive_path, force)
                seen_downloads.add(download_url)
            extract_from_archive(archive_path, archive_sub_path, primary_destination, force)
            if archive_sub_path_aux and aux_destination is not None:
                extract_from_archive(archive_path, archive_sub_path_aux, aux_destination, force)
            continue

        download_file(download_url, primary_destination, force)
        if aux_url and aux_destination is not None:
            download_file(aux_url, aux_destination, force)


def main():
    parser = argparse.ArgumentParser(description="Bundle all app models into an install tree.")
    parser.add_argument("--catalog", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--cache-dir", type=pathlib.Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    entries = parse_catalog(args.catalog)
    if not entries:
        print("[bundle-models] no catalog entries parsed", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    bundle_entries(entries, args.output_dir, args.cache_dir, args.force, args.dry_run)
    print(f"[bundle-models] processed {len(entries)} catalog entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
