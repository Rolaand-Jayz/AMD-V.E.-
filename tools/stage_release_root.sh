#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-${root_dir}/build_release_root}"
stage_root="${STAGE_ROOT:-${root_dir}/dist/stage-root}"
install_prefix="${INSTALL_PREFIX:-/opt/amd-video-enhancer}"
extra_migraphx_dirs="${EXTRA_MIGRAPHX_LIBRARY_DIRS:-}"
bundled_migraphx_prefix="${AVE_BUNDLED_MIGRAPHX_PREFIX:-}"
compiler_vendor_dir=""
compiler_overlay_root=""
compiler_vendor_root=""

sanitize_standard_runpaths() {
  local payload_root="$1"

  while IFS= read -r -d '' candidate; do
    local dynamic=""
    dynamic="$(readelf -d "${candidate}" 2>/dev/null || true)"
    [[ -n "${dynamic}" ]] || continue

    local runpath=""
    runpath="$(printf '%s\n' "${dynamic}" | sed -nE 's/.*Library runpath: \[(.*)\].*/\1/p')"
    [[ -n "${runpath}" ]] || runpath="$(printf '%s\n' "${dynamic}" | sed -nE 's/.*Library rpath: \[(.*)\].*/\1/p')"
    [[ -n "${runpath}" ]] || continue

    if [[ "${runpath}" == *'$ORIGIN'* ]]; then
      continue
    fi

    if [[ "${runpath}" == /usr/lib* ]]; then
      patchelf --remove-rpath "${candidate}"
    fi
  done < <(find "${payload_root}" -type f -print0)
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

if [[ -z "${bundled_migraphx_prefix}" && -d "$HOME/.local/opt/migraphx-codex" ]]; then
  bundled_migraphx_prefix="$HOME/.local/opt/migraphx-codex"
fi

trap 'rm -rf "${compiler_vendor_root}" "${compiler_overlay_root}"' EXIT

cmake_args=(
  -DCMAKE_BUILD_TYPE=Release
  -DAVE_HAVE_CURL=ON
  -DAVE_HAVE_MIGRAPHX=ON
  -DAVE_HAVE_HIP=ON
  -DAVE_HAVE_VULKAN=ON
  -DAVE_HAVE_NCNN=ON
  -DAVE_INSTALL_BUNDLED_MODELS=OFF
)

if [[ -n "${bundled_migraphx_prefix}" ]]; then
  compiler_overlay_root="$(mktemp -d "${TMPDIR:-/tmp}/ave-release-root-overlay.XXXXXX")"
  compiler_vendor_root="$(mktemp -d "${TMPDIR:-/tmp}/ave-release-root-vendor.XXXXXX")"
  stage_bundled_migraphx_overlay "${compiler_overlay_root}" "${bundled_migraphx_prefix}"
  stage_compiler_migraphx_deps "${compiler_vendor_root}"
  cmake_args+=(
    -DAVE_BUNDLED_MIGRAPHX_PREFIX="${compiler_overlay_root}"
    -DAVE_PREFER_BUNDLED_MIGRAPHX_FOR_BUILD=ON
  )
  if [[ -n "${compiler_vendor_dir}" && -n "${extra_migraphx_dirs}" ]]; then
    cmake_args+=("-DAVE_BUNDLED_MIGRAPHX_EXTRA_LIBRARY_DIRS=${compiler_vendor_dir};${extra_migraphx_dirs}")
  elif [[ -n "${compiler_vendor_dir}" ]]; then
    cmake_args+=("-DAVE_BUNDLED_MIGRAPHX_EXTRA_LIBRARY_DIRS=${compiler_vendor_dir}")
  elif [[ -n "${extra_migraphx_dirs}" ]]; then
    cmake_args+=("-DAVE_BUNDLED_MIGRAPHX_EXTRA_LIBRARY_DIRS=${extra_migraphx_dirs}")
  fi
fi

cmake -S "${root_dir}" -B "${build_dir}" "${cmake_args[@]}"

cmake --build "${build_dir}" -j
ctest --test-dir "${build_dir}" --output-on-failure

rm -rf "${stage_root}"
mkdir -p "${stage_root}"
cmake --install "${build_dir}" --prefix "${stage_root}${install_prefix}"
sanitize_standard_runpaths "${stage_root}${install_prefix}"

echo "Staged release root at ${stage_root}"
