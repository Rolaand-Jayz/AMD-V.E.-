#!/usr/bin/env bash
set -euo pipefail

overlay_prefix=""

usage() {
    cat <<'EOF'
Usage:
  tools/stage_migraphx_runtime_overlay.sh --prefix <dir>

Stages the exact cached MiGraphX runtime SONAME set into an existing bundled
MiGraphX prefix so compiler-portable packaging matches the app's linked ABI.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix)
            overlay_prefix="${2:-}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "${overlay_prefix}" ]]; then
    echo "Missing --prefix" >&2
    usage >&2
    exit 2
fi

cache_dir="/var/cache/pacman/pkg"
pattern="${cache_dir}/migraphx-*.pkg.tar.zst"
runtime_regex='libmigraphx\.so\.2015000$'

find_matching_archive() {
    local archive=""
    while IFS= read -r candidate; do
        [[ -n "${candidate}" ]] || continue
        if bsdtar -tf "${candidate}" | rg -q "${runtime_regex}"; then
            archive="${candidate}"
        fi
    done < <(compgen -G "${pattern}" | sort -V)

    if [[ -z "${archive}" ]]; then
        return 1
    fi
    printf '%s\n' "${archive}"
}

archive="$(find_matching_archive)" || {
    cat >&2 <<EOF
Unable to find a cached MiGraphX runtime archive that provides libmigraphx.so.2015000.
Expected a package matching:
  ${pattern}
EOF
    exit 1
}

bsdtar -xpf "${archive}" -C "${overlay_prefix}" --strip-components 2 \
    opt/rocm/lib/libmigraphx_c.so \
    opt/rocm/lib/libmigraphx_c.so.3 \
    opt/rocm/lib/libmigraphx_c.so.3.0 \
    opt/rocm/lib/migraphx/lib

for compat_pair in \
    "lib/migraphx/lib/libmigraphx_device.so.2015000:lib/migraphx/lib/libmigraphx_device.so.2016000.0" \
    "lib/migraphx/lib/libmigraphx_onnx.so.2015000:lib/migraphx/lib/libmigraphx_onnx.so.2016000.0" \
    "lib/migraphx/lib/libmigraphx_tf.so.2015000:lib/migraphx/lib/libmigraphx_tf.so.2016000.0"
do
    compat_target="${compat_pair%%:*}"
    compat_source="${compat_pair##*:}"
    if [[ ! -e "${overlay_prefix}/${compat_target}" && -e "${overlay_prefix}/${compat_source}" ]]; then
        ln -s "$(basename "${compat_source}")" "${overlay_prefix}/${compat_target}"
    fi
done

missing_after_stage=""
for soname in \
    lib/migraphx/lib/libmigraphx.so.2015000 \
    lib/migraphx/lib/libmigraphx_device.so.2015000 \
    lib/migraphx/lib/libmigraphx_gpu.so.2015000 \
    lib/migraphx/lib/libmigraphx_onnx.so.2015000 \
    lib/migraphx/lib/libmigraphx_ref.so.2015000 \
    lib/migraphx/lib/libmigraphx_tf.so.2015000
do
    if [[ ! -e "${overlay_prefix}/${soname}" ]]; then
        missing_after_stage+="${soname}"$'\n'
    fi
done

if [[ -n "${missing_after_stage}" ]]; then
    echo "Staging completed, but the following required MiGraphX runtime sonames were still missing:" >&2
    printf '%s' "${missing_after_stage}" >&2
    exit 1
fi

echo "Staged cached MiGraphX runtime overlay into ${overlay_prefix}"
