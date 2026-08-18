#!/usr/bin/env python3
from pathlib import Path

path = Path("CMakeLists.txt")
text = path.read_text()

replacements = [
    (
        "include(GNUInstallDirs)\n\nset(CMAKE_CXX_STANDARD 20)",
        "include(GNUInstallDirs)\ninclude(cmake/ave_rocm_discovery.cmake)\n\n"
        "set(AVE_ROCM_ROOT_HINTS \"\" CACHE STRING\n"
        "    \"Additional ROCm installation prefixes to search (semicolon-separated)\")\n"
        "ave_discover_rocm_prefixes()\n"
        "if(AVE_ROCM_DISCOVERED_PREFIXES)\n"
        "    message(STATUS \"ROCm prefixes available to CMake: ${AVE_ROCM_DISCOVERED_PREFIXES}\")\n"
        "endif()\n\n"
        "set(CMAKE_CXX_STANDARD 20)",
    ),
    (
        "    find_package(hiprtc QUIET)\n    find_package(migraphx QUIET)",
        "    find_package(hiprtc CONFIG QUIET)\n    find_package(migraphx CONFIG QUIET)",
    ),
    (
        "    find_package(hip QUIET)\n",
        "    find_package(hip CONFIG QUIET)\n",
    ),
]

changed = False
for old, new in replacements:
    if new in text:
        continue
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"refusing to patch: expected exactly one match, found {count}: {old!r}"
        )
    text = text.replace(old, new, 1)
    changed = True

if changed:
    path.write_text(text)
    print("CMakeLists.txt patched")
else:
    print("CMakeLists.txt already patched")
