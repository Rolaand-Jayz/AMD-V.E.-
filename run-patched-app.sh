#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="${script_dir}"
build_dir="${AVE_BUILD_DIR:-${repo_dir}/build-migraphx-patched}"

usage() {
    cat <<'EOF'
Usage:
  ./run-patched-app.sh [app args...]
  ./run-patched-app.sh --gui [gui args...]

Environment overrides:
  AVE_PATCHED_MIGRAPHX_PREFIX  Override the patched MiGraphX install prefix
  AVE_BUILD_DIR                Override the patched app build directory
  AVE_REBUILD=1               Force a reconfigure and rebuild before launch
EOF
}

find_migraphx_prefix() {
    if [[ -n "${AVE_PATCHED_MIGRAPHX_PREFIX:-}" ]]; then
        if [[ -x "${AVE_PATCHED_MIGRAPHX_PREFIX}/bin/migraphx-driver" ]]; then
            printf '%s\n' "${AVE_PATCHED_MIGRAPHX_PREFIX}"
            return 0
        fi
        printf 'Configured AVE_PATCHED_MIGRAPHX_PREFIX is missing migraphx-driver: %s\n' \
            "${AVE_PATCHED_MIGRAPHX_PREFIX}" >&2
        return 1
    fi

    local candidates=(
        "${HOME}/.local/opt/migraphx-codex"
        "/tmp/AMDMIGraphX/install-codex"
    )
    local prefix
    for prefix in "${candidates[@]}"; do
        if [[ -x "${prefix}/bin/migraphx-driver" ]]; then
            printf '%s\n' "${prefix}"
            return 0
        fi
    done

    printf 'Patched MiGraphX install not found. Checked: %s and %s\n' \
        "${candidates[0]}" "${candidates[1]}" >&2
    return 1
}

configure_and_build() {
    local migraphx_prefix="$1"

    cmake -S "${repo_dir}" -B "${build_dir}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="${migraphx_prefix};/opt/rocm" \
        -DAVE_HAVE_CURL=ON \
        -DAVE_HAVE_MIGRAPHX=ON \
        -DAVE_HAVE_HIP=ON \
        -DAVE_HAVE_VULKAN=ON \
        -DAVE_HAVE_NCNN=ON
    cmake --build "${build_dir}" -j
}

target="ave"
if [[ "${1:-}" == "--gui" ]]; then
    target="ave_gui"
    shift
elif [[ "${1:-}" == "--help-wrapper" ]]; then
    usage
    exit 0
fi

migraphx_prefix="$(find_migraphx_prefix)"

if [[ ! -x "${build_dir}/${target}" || "${AVE_REBUILD:-0}" == "1" ]]; then
    configure_and_build "${migraphx_prefix}"
fi

if [[ ! -x "${build_dir}/${target}" ]]; then
    printf 'Expected target not found after build: %s\n' "${build_dir}/${target}" >&2
    exit 1
fi

export PATH="${migraphx_prefix}/bin:${PATH}"
export LD_LIBRARY_PATH="${migraphx_prefix}/lib:${migraphx_prefix}/lib/migraphx/lib:/opt/rocm/lib:/opt/rocm/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

exec "${build_dir}/${target}" "$@"
