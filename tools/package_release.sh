#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:-runtime-portable}"
build_dir="${BUILD_DIR:-}"
install_prefix="${INSTALL_PREFIX:-}"
archive_mode="${ARCHIVE_MODE:-stage}"
extra_migraphx_dirs="${EXTRA_MIGRAPHX_LIBRARY_DIRS:-}"
bundled_migraphx_prefix=""
compiler_vendor_dir=""

usage() {
    cat <<'EOF'
Usage:
  tools/package_release.sh [runtime-portable|compiler-portable|install-tree]

Environment:
  BUILD_DIR                  Override the build directory used for packaging.
  INSTALL_PREFIX             Prefix for install-tree mode.
  ARCHIVE_MODE               "stage" (default) or "archive" for portable modes.
  AVE_BUNDLED_MIGRAPHX_PREFIX
                             Explicit custom MiGraphX prefix used for
                             compiler-portable release packaging.
  EXTRA_MIGRAPHX_LIBRARY_DIRS
                             Semicolon-separated extra directories searched when
                             closing custom MiGraphX compiler dependencies.

Examples:
  tools/package_release.sh runtime-portable
  ARCHIVE_MODE=archive tools/package_release.sh runtime-portable
  AVE_BUNDLED_MIGRAPHX_PREFIX=/path/to/custom/migraphx EXTRA_MIGRAPHX_LIBRARY_DIRS="/custom/lib;/vendor/lib" tools/package_release.sh compiler-portable
  INSTALL_PREFIX="$PWD/dist/install-root" tools/package_release.sh install-tree
EOF
}

if [[ "${mode}" == "--help" || "${mode}" == "-h" ]]; then
    usage
    exit 0
fi

common_cmake_args=(
  -DCMAKE_BUILD_TYPE=Release
  -DAVE_HAVE_CURL=ON
  -DAVE_HAVE_MIGRAPHX=ON
  -DAVE_HAVE_HIP=ON
  -DAVE_HAVE_VULKAN=ON
  -DAVE_HAVE_NCNN=ON
)

configure_and_build() {
    local mode_build_dir="$1"
    shift
    cmake -S "${root_dir}" -B "${mode_build_dir}" "${common_cmake_args[@]}" "$@"
    cmake --build "${mode_build_dir}" -j
}

stage_compiler_migraphx_deps() {
    local vendor_root="$1"
    local vendor_dir="${vendor_root}/lib"

    "${root_dir}/tools/stage_migraphx_compiler_deps.sh" \
        --vendor-dir "${vendor_dir}"
    compiler_vendor_dir="${vendor_dir}"
}

stage_bundled_migraphx_overlay() {
    local overlay_root="$1"
    local source_prefix="$2"

    rm -rf "${overlay_root}"
    mkdir -p "${overlay_root}"
    cp -a "${source_prefix}/." "${overlay_root}/"

    "${root_dir}/tools/stage_migraphx_runtime_overlay.sh" \
        --prefix "${overlay_root}"
    bundled_migraphx_prefix="${overlay_root}"
}

case "${mode}" in
  runtime-portable)
    if [[ -z "${build_dir}" ]]; then
      build_dir="${root_dir}/build_portable_runtime_ready"
    fi
    configure_args=(
      -DAVE_INSTALL_BUNDLED_MIGRAPHX=OFF
      -DAVE_INSTALL_BUNDLED_MIGRAPHX_COMPILER=OFF
      -DAVE_INSTALL_BUNDLED_MODELS=OFF
    )
    configure_and_build "${build_dir}" "${configure_args[@]}"
    if [[ "${archive_mode}" == "archive" ]]; then
      cmake --build "${build_dir}" -j --target portable_bundle
    else
      cmake --build "${build_dir}" -j --target portable_stage
    fi
    ;;
  compiler-portable)
    if [[ -z "${build_dir}" ]]; then
      build_dir="${root_dir}/build_portable_compiler"
    fi
    rm -rf "${build_dir}"
    if [[ -n "${AVE_BUNDLED_MIGRAPHX_PREFIX:-}" ]]; then
      bundled_migraphx_prefix="${AVE_BUNDLED_MIGRAPHX_PREFIX}"
    fi
    if [[ -z "${bundled_migraphx_prefix}" ]]; then
      echo "compiler-portable requires AVE_BUNDLED_MIGRAPHX_PREFIX=/path/to/custom/migraphx" >&2
      exit 1
    fi
    compiler_overlay_root="$(mktemp -d "${TMPDIR:-/tmp}/ave-migraphx-compiler-overlay.XXXXXX")"
    trap 'rm -rf "${compiler_vendor_root:-}" "${compiler_overlay_root:-}"' EXIT
    stage_bundled_migraphx_overlay "${compiler_overlay_root}" "${bundled_migraphx_prefix}"
    compiler_vendor_root="$(mktemp -d "${TMPDIR:-/tmp}/ave-migraphx-compiler-vendor.XXXXXX")"
    stage_compiler_migraphx_deps "${compiler_vendor_root}"
    configure_args=(
      -DAVE_INSTALL_BUNDLED_MIGRAPHX=ON
      -DAVE_INSTALL_BUNDLED_MIGRAPHX_COMPILER=ON
      -DAVE_INSTALL_BUNDLED_MODELS=OFF
      -DAVE_PREFER_BUNDLED_MIGRAPHX_FOR_BUILD=ON
      -DAVE_BUNDLED_MIGRAPHX_PREFIX=${compiler_overlay_root}
    )
    if [[ -n "${compiler_vendor_dir}" ]]; then
      if [[ -n "${extra_migraphx_dirs}" ]]; then
        configure_args+=("-DAVE_BUNDLED_MIGRAPHX_EXTRA_LIBRARY_DIRS=${compiler_vendor_dir};${extra_migraphx_dirs}")
      else
        configure_args+=("-DAVE_BUNDLED_MIGRAPHX_EXTRA_LIBRARY_DIRS=${compiler_vendor_dir}")
      fi
    elif [[ -n "${extra_migraphx_dirs}" ]]; then
      configure_args+=("-DAVE_BUNDLED_MIGRAPHX_EXTRA_LIBRARY_DIRS=${extra_migraphx_dirs}")
    fi
    configure_and_build "${build_dir}" "${configure_args[@]}"
    if [[ "${archive_mode}" == "archive" ]]; then
      cmake --build "${build_dir}" -j --target portable_bundle
    else
      cmake --build "${build_dir}" -j --target portable_stage
    fi
    ;;
  install-tree)
    if [[ -z "${build_dir}" ]]; then
      build_dir="${root_dir}/build_install_release"
    fi
    if [[ -z "${install_prefix}" ]]; then
      install_prefix="${root_dir}/dist/install-root"
    fi
    configure_and_build "${build_dir}"
    cmake --install "${build_dir}" --prefix "${install_prefix}"
    ;;
  *)
    echo "Unknown packaging mode: ${mode}" >&2
    usage >&2
    exit 2
    ;;
esac
