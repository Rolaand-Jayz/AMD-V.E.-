#!/usr/bin/env bash
set -euo pipefail

vendor_dir=""

usage() {
    cat <<'EOF'
Usage:
  tools/stage_migraphx_compiler_deps.sh --vendor-dir <dir>

Stages ABI-matched MiGraphX compiler-side shared libraries from the local
package cache into the requested vendor directory.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --vendor-dir)
            vendor_dir="${2:-}"
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

if [[ -z "${vendor_dir}" ]]; then
    echo "Missing --vendor-dir" >&2
    usage >&2
    exit 2
fi

cache_dir="/var/cache/pacman/pkg"
abseil_pattern="${cache_dir}/abseil-cpp-*.pkg.tar.zst"
protobuf_pattern="${cache_dir}/protobuf-33.1-*.pkg.tar.zst"

find_matching_archive() {
    local pattern="$1"
    local regex="$2"
    local archive=""

    while IFS= read -r candidate; do
        [[ -n "${candidate}" ]] || continue
        if bsdtar -tf "${candidate}" | rg -q "${regex}"; then
            archive="${candidate}"
        fi
    done < <(compgen -G "${pattern}" | sort -V)

    if [[ -z "${archive}" ]]; then
        return 1
    fi
    printf '%s\n' "${archive}"
}

rm -rf "${vendor_dir}"
mkdir -p "${vendor_dir}"

abseil_archive="$(find_matching_archive "${abseil_pattern}" 'libabsl_.*\.so\.2508\.0\.0$')"
protobuf_archive="$(find_matching_archive "${protobuf_pattern}" 'libprotobuf\.so\.33\.1\.0$|libutf8_validity\.so\.33\.1\.0$')"

if [[ -z "${abseil_archive}" || -z "${protobuf_archive}" ]]; then
    cat >&2 <<EOF
Unable to stage the MiGraphX compiler-side dependencies from the local package cache.
Expected archives matching:
  ${abseil_pattern}
  ${protobuf_pattern}
EOF
    exit 1
fi

bsdtar -xpf "${abseil_archive}" -C "${vendor_dir}" --strip-components 2 usr/lib
bsdtar -xpf "${protobuf_archive}" -C "${vendor_dir}" --strip-components 2 usr/lib

missing_after_stage=""
for soname in \
    libprotobuf.so.33.1.0 \
    libutf8_validity.so.33.1.0 \
    libabsl_log_internal_check_op.so.2508.0.0 \
    libabsl_die_if_null.so.2508.0.0 \
    libabsl_log_internal_conditions.so.2508.0.0 \
    libabsl_log_internal_message.so.2508.0.0 \
    libabsl_log_internal_nullguard.so.2508.0.0
do
    if [[ ! -e "${vendor_dir}/${soname}" ]]; then
        missing_after_stage+="${soname}"$'\n'
    fi
done

if [[ -n "${missing_after_stage}" ]]; then
    echo "Staging completed, but the following required sonames were still missing:" >&2
    printf '%s' "${missing_after_stage}" >&2
    exit 1
fi

echo "Staged MiGraphX compiler-side dependencies into ${vendor_dir}"
