#!/usr/bin/env bash
set -euo pipefail

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"

SCRIPT_SOURCE="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_SOURCE")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ARTIFACT_LEGAL_MANIFEST="$REPO_ROOT/packaging/legal/macos-universal.tsv"
AUDITED_PUBLIC_LOCK="$REPO_ROOT/packaging/legal/public-Cargo.lock"

ARM_TARGET="aarch64-apple-darwin"
X86_TARGET="x86_64-apple-darwin"
UNIVERSAL_TARGET="universal2-apple-darwin"
MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-11.0}"
export MACOSX_DEPLOYMENT_TARGET
CLI_WRAPPER_NAME="mcraw4vulkan"
GUI_WRAPPER_NAME="mcraw4vulkan-gui"
CLI_PACKAGED_BINARY_NAME="mcraw4vulkan-macos"
GUI_PACKAGED_BINARY_NAME="mcraw4vulkan-gui-macos"
GUI_PREFLIGHT_PACKAGED_BINARY_NAME="mcraw4vulkan-preflight-macos"
OLD_PACKAGED_BINARY_SUFFIX="real"

binary=""
gui_binary=""
preflight_binary=""
out_dir=""
run_build=0
discovery_only=0
print_project_version=0
project_version=""
source_revision="unavailable"
source_tree_state="unavailable"
cargo_lock_sha256=""
artifact_legal_unresolved_count=0
discovery_native_evidence_count=0
runtime_roots=("")
install_name_tool_changes=()
vulkan_install_name_tool_changes=()
vulkan_string_sanitizations=()
universal_binary_outputs=()
modified_macho_files=()
ad_hoc_signed_files=()
ad_hoc_sign_failures=()
ad_hoc_signed_frameworks=()
ad_hoc_framework_sign_failures=()
codesign_verified_files=()
codesign_verified_frameworks=()
package_homebrew_leaks=()
package_unexpected_usr_local_deps=()
package_external_framework_deps=()
package_allowed_macfuse_deps=()
package_forbidden_string_leaks=()
packaged_framework_binaries=()
framework_xattr_cleanups=()
framework_copy_tool=""
sdl2_src=""
sdl2_package_path=""
sdl2_image_src=""
sdl2_image_package_path=""
sdl2_image_inclusion_reason="not needed; no packaged Mach-O dependency references SDL2_image.framework"
sdl2_forbidden_string_check="not run"
package_xattr_cleanup_status="not run"
package_quarantine_check_status="not run"
codesign_status="not run"
legal_input_components=()
legal_input_kinds=()
legal_input_paths=()
legal_input_basenames=()
legal_input_casefold_basenames=()
component_version_names=()
component_version_values=()

usage() {
    printf '%s\n' "Usage: $0 [options]"
    printf '%s\n' ""
    printf '%s\n' "Options:"
    printf '%s\n' "  --binary PATH       Path to compiled mcraw4vulkan binary"
    printf '%s\n' "  --out-dir PATH      Output package directory"
    printf '%s\n' "  --runtime-root PATH Vulkan SDK or MoltenVK install root to search first"
    printf '%s\n' "  --component-version COMPONENT=VERSION"
    printf '%s\n' "                       Exact version of SDL2, SDL2_image, Vulkan-Loader, or MoltenVK"
    printf '%s\n' "  --license-file COMPONENT=PATH"
    printf '%s\n' "                       Required license text for a bundled native component"
    printf '%s\n' "  --notice-file COMPONENT=PATH"
    printf '%s\n' "                       Additional notice text for a bundled native component"
    printf '%s\n' "  --discovery-only    Permit unresolved legal-view rows; stage PASS mappings only"
    printf '%s\n' "                       and mark the package output NOT FOR RELEASE"
    printf '%s\n' "  --print-project-version"
    printf '%s\n' "                       Print the current Cargo package version and exit"
    printf '%s\n' "  --build             Accepted for compatibility; Universal release targets are always built"
    printf '%s\n' "  --help              Show this help"
}

die() {
    printf '%s\n' "error: $*" >&2
    exit 1
}

note() {
    printf '%s\n' "$*"
}

abs_path() {
    python3 - "$1" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve(strict=False))
PY
}

append_value() {
    local value="${1:-}"
    local target="$2"
    [ -n "$value" ] || return 0

    case "$target" in
        loader)
            loader_candidates+=("$value")
            ;;
        vulkan_dylib)
            vulkan_dylib_candidates+=("$value")
            ;;
        molten)
            molten_candidates+=("$value")
            ;;
        icd)
            icd_candidates+=("$value")
            ;;
        *)
            die "internal error: unknown candidate target $target"
            ;;
    esac
}

append_root_candidates() {
    local root="$1"
    [ -n "$root" ] || return 0

    loader_candidates+=(
        "$root/lib/libvulkan.1.dylib"
        "$root/macOS/lib/libvulkan.1.dylib"
    )
    vulkan_dylib_candidates+=(
        "$root/lib/libvulkan.dylib"
        "$root/macOS/lib/libvulkan.dylib"
    )
    molten_candidates+=(
        "$root/lib/libMoltenVK.dylib"
        "$root/macOS/lib/libMoltenVK.dylib"
    )
    icd_candidates+=(
        "$root/share/vulkan/icd.d/MoltenVK_icd.json"
        "$root/macOS/share/vulkan/icd.d/MoltenVK_icd.json"
        "$root/../share/vulkan/icd.d/MoltenVK_icd.json"
    )
}

append_pkg_config_root() {
    local package_name="$1"
    local prefix
    if ! command -v pkg-config >/dev/null 2>&1; then
        return 0
    fi
    prefix="$(pkg-config --variable=prefix "$package_name" 2>/dev/null || true)"
    if [ -n "$prefix" ]; then
        append_root_candidates "$prefix"
    fi
}

append_find_results() {
    local start_dir="$1"
    local pattern="$2"
    local target="$3"

    if [ ! -d "$start_dir" ]; then
        return 0
    fi

    while IFS= read -r candidate; do
        append_value "$candidate" "$target"
    done < <(find "$start_dir" -maxdepth 8 -iname "$pattern" 2>/dev/null | head -100)
}

find_first_existing() {
    local label="$1"
    shift
    local candidate
    for candidate in "$@"; do
        if [ -n "$candidate" ] && [ -e "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    die "could not find $label"
}

safe_prepare_out_dir() {
    local output_dir="$1"
    local package_output_root
    package_output_root="$(abs_path "$REPO_ROOT/target/macos-cli-package")"

    case "$output_dir" in
        "$package_output_root"/*)
            ;;
        *)
            die "refusing to write package output outside $package_output_root: $output_dir"
            ;;
    esac

    case "$output_dir" in
        /|/usr|/usr/*|/opt|/opt/homebrew|/opt/homebrew/*|/Applications|/Applications/*|/Library|/Library/*)
            die "refusing to write package output under protected path: $output_dir"
            ;;
    esac
    rm -rf "$output_dir"
    mkdir -p "$output_dir/bin" "$output_dir/lib" "$output_dir/vulkan/icd.d" "$output_dir/Frameworks"
}

copy_required() {
    local src="$1"
    local dst="$2"
    [ -e "$src" ] || die "missing required file: $src"
    cp -p "$src" "$dst"
}

is_legal_component() {
    case "$1" in
        SDL2|SDL2_image|Vulkan-Loader|MoltenVK)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

record_component_version() {
    local spec="$1"
    local component
    local version
    local existing

    case "$spec" in
        *=*)
            component="${spec%%=*}"
            version="${spec#*=}"
            ;;
        *)
            die "--component-version requires COMPONENT=VERSION"
            ;;
    esac
    is_legal_component "$component" || die "unknown bundled component in --component-version"
    [ -n "$version" ] || die "--component-version requires a nonempty version"
    case "$version" in
        *[!A-Za-z0-9._:+-]*)
            die "component version contains unsupported characters"
            ;;
    esac

    if [ "${#component_version_names[@]}" -gt 0 ]; then
        for existing in "${component_version_names[@]}"; do
            [ "$existing" != "$component" ] || die "duplicate --component-version for $component"
        done
    fi
    component_version_names+=("$component")
    component_version_values+=("$version")
}

component_version() {
    local component="$1"
    local index

    if [ "${#component_version_names[@]}" -gt 0 ]; then
        for ((index = 0; index < ${#component_version_names[@]}; index++)); do
            if [ "${component_version_names[$index]}" = "$component" ]; then
                printf '%s\n' "${component_version_values[$index]}"
                return 0
            fi
        done
    fi
    return 1
}

record_legal_input() {
    local kind="$1"
    local spec="$2"
    local component
    local source_path
    local source_basename
    local source_casefold
    local index

    case "$spec" in
        *=*)
            component="${spec%%=*}"
            source_path="${spec#*=}"
            ;;
        *)
            die "--$kind-file requires COMPONENT=PATH"
            ;;
    esac
    is_legal_component "$component" || die "unknown bundled component in --$kind-file"
    [ -n "$source_path" ] || die "--$kind-file requires a nonempty path"
    [ -f "$source_path" ] || die "bundled-component legal input is not a regular file"
    [ ! -L "$source_path" ] || die "bundled-component legal input must not be a symlink"
    [ -s "$source_path" ] || die "bundled-component legal input is empty"
    python3 - "$source_path" <<'PY'
from pathlib import Path
import sys
Path(sys.argv[1]).read_text(encoding="utf-8")
PY

    source_basename="$(basename "$source_path")"
    case "$source_basename" in
        ""|.|..|*[!A-Za-z0-9._+-]*)
            die "bundled-component legal filename is not release-safe"
            ;;
    esac
    source_casefold="$(printf '%s' "$source_basename" | LC_ALL=C tr 'A-Z' 'a-z')"

    if [ "${#legal_input_components[@]}" -gt 0 ]; then
        for ((index = 0; index < ${#legal_input_components[@]}; index++)); do
            if [ "${legal_input_components[$index]}" = "$component" ] &&
                [ "${legal_input_casefold_basenames[$index]}" = "$source_casefold" ]; then
                die "duplicate bundled-component legal destination for $component"
            fi
        done
    fi
    legal_input_components+=("$component")
    legal_input_kinds+=("$kind")
    legal_input_paths+=("$source_path")
    legal_input_basenames+=("$source_basename")
    legal_input_casefold_basenames+=("$source_casefold")
}

sha256_file() {
    python3 - "$1" <<'PY'
from pathlib import Path
import hashlib
import sys
print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
}

resolve_project_version() {
    (
        cd "$REPO_ROOT"
        cargo metadata --manifest-path Cargo.toml --no-deps --format-version 1 --locked
    ) | python3 -c '
import json
from pathlib import Path
import sys

metadata = json.load(sys.stdin)
wanted = (Path(sys.argv[1]) / "crates/mcraw4vulkan/Cargo.toml").resolve()
matches = [
    package["version"]
    for package in metadata["packages"]
    if Path(package["manifest_path"]).resolve() == wanted
]
if len(matches) != 1:
    raise SystemExit("Cargo metadata did not identify exactly one mcraw4vulkan package")
print(matches[0])
' "$REPO_ROOT"
}

validate_current_dependency_closure() {
    [ -f "$REPO_ROOT/Cargo.lock" ] && [ ! -L "$REPO_ROOT/Cargo.lock" ] ||
        die "release source must contain a regular Cargo.lock"
    [ -f "$AUDITED_PUBLIC_LOCK" ] && [ ! -L "$AUDITED_PUBLIC_LOCK" ] ||
        die "missing audited public dependency closure"

    if ! python3 -c '
import json
from pathlib import Path
import sys

current_lock_path = Path(sys.argv[1])
audited_lock_path = Path(sys.argv[2])

def lock_packages(path):
    packages = []
    current = None
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if line == "[[package]]":
            if current is not None:
                packages.append(current)
            current = {}
            continue
        if current is None or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        if key in {"name", "version", "source", "checksum"} and value.startswith(chr(34)):
            current[key] = json.loads(value)
    if current is not None:
        packages.append(current)
    return packages

current_packages = lock_packages(current_lock_path)
audited_packages = lock_packages(audited_lock_path)

def lock_identity(package):
    return (
        package["name"],
        package["version"],
        package.get("source", ""),
        package.get("checksum", ""),
    )

current_external = {
    lock_identity(package)
    for package in current_packages
    if "source" in package
}
audited_external = {
    lock_identity(package)
    for package in audited_packages
    if "source" in package
}
if current_external != audited_external:
    missing = sorted(audited_external - current_external)
    added = sorted(current_external - audited_external)
    if missing:
        print("audited third-party dependencies missing or changed:", file=sys.stderr)
        for identity in missing:
            print(f"  {identity[0]} {identity[1]} {identity[2]}", file=sys.stderr)
    if added:
        print("current unaudited third-party dependencies:", file=sys.stderr)
        for identity in added:
            print(f"  {identity[0]} {identity[1]} {identity[2]}", file=sys.stderr)
    raise SystemExit(1)
' "$REPO_ROOT/Cargo.lock" "$AUDITED_PUBLIC_LOCK"; then
        die "current third-party dependency closure does not match the audited legal closure"
    fi
}

require_component_legal_inputs() {
    local component="$1"
    local index
    local license_count=0

    component_version "$component" >/dev/null || die "missing --component-version for bundled $component"
    if [ "${#legal_input_components[@]}" -gt 0 ]; then
        for ((index = 0; index < ${#legal_input_components[@]}; index++)); do
            if [ "${legal_input_components[$index]}" = "$component" ] &&
                [ "${legal_input_kinds[$index]}" = "license" ]; then
                license_count=$((license_count + 1))
            fi
        done
    fi
    if [ "$discovery_only" -eq 1 ]; then
        return 0
    fi
    [ "$license_count" -gt 0 ] || die "missing --license-file for bundled $component"
}

validate_artifact_legal_manifest() {
    local manifest="$1"
    local selection="$2"
    local selection_output="/dev/null"
    local expected_header
    local actual_header
    local metrics_dir="$cargo_target_root/macos-cli-package"
    local metrics_file="$metrics_dir/.artifact-legal-metrics.$$"
    local validation_failed=0

    [ -f "$manifest" ] || die "missing macOS artifact legal manifest"
    [ ! -L "$manifest" ] || die "macOS artifact legal manifest must not be a symlink"

    expected_header=$'component_name\tcomponent_version\tpackage_or_native_component\tupstream_source\tlicense_expression\tselected_license_branch\tdistribution_kind\tpublic_legal_file_path\tlegal_file_sha256\tnotice_path\tverification_status'
    IFS= read -r actual_header < "$manifest" || die "macOS artifact legal manifest is empty"
    [ "$actual_header" = "$expected_header" ] || die "macOS artifact legal manifest has an unexpected header"
    if [ -n "$selection" ]; then
        selection_output="$selection"
    fi
    mkdir -p "$metrics_dir"

    if ! LC_ALL=C awk -F '\t' -v discovery="$discovery_only" -v metrics="$metrics_file" '
        function safe_legal_path(path) {
            return path ~ /^licenses\// &&
                path !~ /\\/ &&
                path !~ /\/\// &&
                path !~ /(^|\/)\.\.?($|\/)/ &&
                substr(path, length(path), 1) != "/"
        }
        NR == 1 { next }
        NF != 11 {
            printf "macOS artifact legal manifest row %d has %d fields; expected 11\n", NR, NF > "/dev/stderr"
            failed = 1
            next
        }
        {
            component = $1
            version = $2
            public_path = $8
            digest = $9
            notice_path = $10
            status = $11

            if (component == "" || version == "") {
                printf "macOS artifact legal manifest row %d lacks component identity\n", NR > "/dev/stderr"
                failed = 1
            }
            if (status == "") {
                printf "macOS artifact legal manifest row %d lacks verification status\n", NR > "/dev/stderr"
                failed = 1
            } else if (status != "PASS") {
                printf "macOS artifact legal manifest unresolved row %d: %s %s (%s)\n", NR, component, version, status > "/dev/stderr"
                unresolved += 1
                if (!discovery) {
                    failed = 1
                }
            }
            if ((public_path == "") != (digest == "")) {
                printf "macOS artifact legal manifest row %d must pair legal path and SHA-256\n", NR > "/dev/stderr"
                failed = 1
            }
            if (public_path != "" && !safe_legal_path(public_path)) {
                printf "macOS artifact legal manifest row %d has an unsafe public legal path\n", NR > "/dev/stderr"
                failed = 1
            }
            if (notice_path != "" && !safe_legal_path(notice_path)) {
                printf "macOS artifact legal manifest row %d has an unsafe notice path\n", NR > "/dev/stderr"
                failed = 1
            }
            if (digest != "" && (length(digest) != 64 || digest ~ /[^0-9a-f]/)) {
                printf "macOS artifact legal manifest row %d has an invalid SHA-256\n", NR > "/dev/stderr"
                failed = 1
            }
            if (public_path != "") {
                if (public_path in hashes && hashes[public_path] != digest) {
                    printf "macOS artifact legal manifest maps one public path to multiple hashes\n" > "/dev/stderr"
                    failed = 1
                } else {
                    hashes[public_path] = digest
                }
                if (status == "PASS" && !(public_path in selected_hashes)) {
                    selected_hashes[public_path] = digest
                    selected += 1
                    print public_path "\t" digest
                }
            }
        }
        END {
            printf "%d\n", unresolved + 0 > metrics
            if (selected == 0) {
                print "macOS artifact legal manifest selects no legal files" > "/dev/stderr"
                failed = 1
            }
            exit failed
        }
    ' "$manifest" | LC_ALL=C sort -u > "$selection_output"; then
        validation_failed=1
    fi
    if [ ! -s "$metrics_file" ]; then
        if [ -e "$metrics_file" ]; then
            rm -- "$metrics_file"
        fi
        if [ -n "$selection" ]; then
            rm -f -- "$selection"
        fi
        die "macOS artifact legal manifest validation did not report unresolved-row metrics"
    fi
    IFS= read -r artifact_legal_unresolved_count < "$metrics_file"
    rm -- "$metrics_file"
    case "$artifact_legal_unresolved_count" in
        ''|*[!0-9]*)
            if [ -n "$selection" ]; then
                rm -f -- "$selection"
            fi
            die "macOS artifact legal manifest reported invalid unresolved-row metrics"
            ;;
    esac
    if [ "$validation_failed" -ne 0 ]; then
        if [ -n "$selection" ]; then
            rm -f -- "$selection"
        fi
        die "macOS artifact legal manifest validation failed"
    fi
}

stage_release_legal_files() {
    local index
    local component
    local source_path
    local destination
    local source_hash
    local input_kind
    local version
    local relative_path
    local manifest_selection="$out_dir/.macos-legal-selection.tsv"
    local expected_hash

    discovery_native_evidence_count=0
    validate_artifact_legal_manifest "$ARTIFACT_LEGAL_MANIFEST" "$manifest_selection"

    require_component_legal_inputs SDL2
    require_component_legal_inputs Vulkan-Loader
    require_component_legal_inputs MoltenVK
    if [ -n "$sdl2_image_package_path" ]; then
        require_component_legal_inputs SDL2_image
    elif component_version SDL2_image >/dev/null ||
        { [ "${#legal_input_components[@]}" -gt 0 ] && printf '%s\n' "${legal_input_components[@]}" | grep -Fxq SDL2_image; }; then
        die "SDL2_image legal inputs were supplied but SDL2_image is not bundled"
    fi

    [ -s "$REPO_ROOT/LICENSE" ] || die "missing nonempty project LICENSE"
    [ -s "$REPO_ROOT/NOTICE" ] || die "missing nonempty project NOTICE"
    [ -d "$REPO_ROOT/licenses" ] || die "missing retained Rust dependency license directory"

    mkdir -p "$out_dir/licenses"
    copy_required "$REPO_ROOT/LICENSE" "$out_dir/LICENSE"
    copy_required "$REPO_ROOT/NOTICE" "$out_dir/NOTICE"
    copy_required "$ARTIFACT_LEGAL_MANIFEST" "$out_dir/THIRD_PARTY_NOTICES.tsv"
    [ "$(sha256_file "$ARTIFACT_LEGAL_MANIFEST")" = "$(sha256_file "$out_dir/THIRD_PARTY_NOTICES.tsv")" ] ||
        die "artifact legal manifest copy failed identity check"

    while IFS=$'\t' read -r relative_path expected_hash; do
        source_path="$REPO_ROOT/$relative_path"
        destination="$out_dir/$relative_path"
        [ -f "$source_path" ] || die "artifact manifest references a missing legal file: $relative_path"
        [ ! -L "$source_path" ] || die "artifact manifest legal file must not be a symlink: $relative_path"
        [ "$(sha256_file "$source_path")" = "$expected_hash" ] ||
            die "artifact manifest hash does not match legal file: $relative_path"
        mkdir -p "$(dirname "$destination")"
        copy_required "$source_path" "$destination"
        [ "$(sha256_file "$destination")" = "$expected_hash" ] ||
            die "artifact legal file copy failed identity check: $relative_path"
    done < "$manifest_selection"
    rm -f -- "$manifest_selection"

    if [ "${#legal_input_components[@]}" -gt 0 ]; then
        for ((index = 0; index < ${#legal_input_components[@]}; index++)); do
            component="${legal_input_components[$index]}"
            input_kind="${legal_input_kinds[$index]}"
            source_path="${legal_input_paths[$index]}"
            source_hash="$(sha256_file "$source_path")"
            version="$(component_version "$component")"
            if LC_ALL=C awk -F '\t' -v component="$component" -v version="$version" \
                -v digest="$source_hash" -v kind="$input_kind" '
                NR > 1 && $1 == component && $2 == version && $8 != "" &&
                    $9 == digest && $11 == "PASS" &&
                    ((kind == "notice" && $10 == $8) ||
                     (kind == "license" && $10 == "")) {
                    found = 1
                }
                END { exit !found }
            ' "$ARTIFACT_LEGAL_MANIFEST"; then
                :
            elif [ "$discovery_only" -eq 1 ]; then
                discovery_native_evidence_count=$((discovery_native_evidence_count + 1))
                note "discovery evidence recorded as unresolved and not staged: $component $version $input_kind $source_hash"
            else
                die "bundled-component legal input is not an exact manifest-owned file: $component"
            fi
        done
    fi

    {
        printf '%s\n' "mcraw4vulkan package legal-file inventory"
        printf '%s\n' "project_license=GPL-3.0-or-later"
        printf '%s\n' "legal_view_unresolved_rows=$artifact_legal_unresolved_count"
        if [ "$discovery_only" -eq 1 ]; then
            printf '%s\n' "discovery_only=yes"
            printf '%s\n' "release_ready=no"
        else
            printf '%s\n' "discovery_only=no"
            printf '%s\n' "release_ready=yes"
        fi
        for component in SDL2 SDL2_image Vulkan-Loader MoltenVK; do
            if component_version "$component" >/dev/null; then
                printf 'component_version\t%s\t%s\n' "$component" "$(component_version "$component")"
            fi
        done
        for destination in "$out_dir/LICENSE" "$out_dir/NOTICE" "$out_dir/THIRD_PARTY_NOTICES.tsv"; do
            relative_path="${destination#"$out_dir/"}"
            printf 'file\t%s\t%s\n' "$relative_path" "$(sha256_file "$destination")"
        done
        while IFS= read -r destination; do
            relative_path="${destination#"$out_dir/"}"
            printf 'file\t%s\t%s\n' "$relative_path" "$(sha256_file "$destination")"
        done < <(find "$out_dir/licenses" -type f ! -name README.txt -print | sort)
} > "$out_dir/licenses/README.txt"

    if [ "$discovery_only" -eq 1 ]; then
        cat > "$out_dir/NOT-FOR-RELEASE.txt" <<'MARKER'
NOT FOR RELEASE

This discovery-only package preserves unresolved public legal-view rows and
contains only legal files selected by unique PASS path/hash mappings.
MARKER
    fi
}

release_rustflags() {
    local flags="${RUSTFLAGS:-}"
    local mapping
    local cargo_home
    local rustup_home

    [ -z "${CARGO_ENCODED_RUSTFLAGS+x}" ] ||
        die "CARGO_ENCODED_RUSTFLAGS must be unset so release path remapping cannot be bypassed"
    cargo_home="$(abs_path "${CARGO_HOME:-$HOME/.cargo}")"
    rustup_home="$(abs_path "${RUSTUP_HOME:-$HOME/.rustup}")"
    for mapping in \
        "$HOME=/usr/src/user" \
        "$cargo_home=/usr/src/cargo" \
        "$rustup_home=/usr/src/rustup" \
        "$REPO_ROOT=/usr/src/mcraw4vulkan"; do
        flags="${flags:+$flags }--remap-path-prefix=$mapping"
    done
    printf '%s\n' "$flags"
}

append_unique_modified_macho_file() {
    local value="$1"
    local existing
    if [ "${#modified_macho_files[@]}" -gt 0 ]; then
        for existing in "${modified_macho_files[@]}"; do
            if [ "$existing" = "$value" ]; then
                return 0
            fi
        done
    fi
    modified_macho_files+=("$value")
}

append_unique_allowed_macfuse_dep() {
    local value="$1"
    local existing
    if [ "${#package_allowed_macfuse_deps[@]}" -gt 0 ]; then
        for existing in "${package_allowed_macfuse_deps[@]}"; do
            if [ "$existing" = "$value" ]; then
                return 0
            fi
        done
    fi
    package_allowed_macfuse_deps+=("$value")
}

record_install_name_tool_change() {
    local change="$1"
    install_name_tool_changes+=("$change")
    if [ "${install_name_tool_change_scope:-}" = "vulkan" ]; then
        vulkan_install_name_tool_changes+=("$change")
    fi
}

require_command() {
    local name="$1"
    command -v "$name" >/dev/null 2>&1 || die "$name is required for macOS packaging"
}

require_installed_rust_target() {
    local target="$1"

    if ! command -v rustup >/dev/null 2>&1; then
        note "warning: rustup is unavailable; cargo will report if target $target cannot be built"
        return 0
    fi

    if ! rustup target list --installed | grep -Fx "$target" >/dev/null 2>&1; then
        die "Rust target is not installed: $target"
    fi
}

assert_macho_archs_exact() {
    local image="$1"
    shift
    local archs
    local arch
    local expected
    local found
    local found_count=0
    local expected_count="$#"

    archs="$(lipo -archs "$image" 2>/dev/null)" || die "could not inspect Mach-O architectures: $image"

    for arch in $archs; do
        found=0
        for expected in "$@"; do
            if [ "$arch" = "$expected" ]; then
                found=1
                break
            fi
        done
        [ "$found" -eq 1 ] || die "$image contains unexpected architecture '$arch' (found: $archs)"
        found_count=$((found_count + 1))
    done

    [ "$found_count" -eq "$expected_count" ] || die "$image has architectures '$archs'; expected $*"

    for expected in "$@"; do
        found=0
        for arch in $archs; do
            if [ "$arch" = "$expected" ]; then
                found=1
                break
            fi
        done
        [ "$found" -eq 1 ] || die "$image is missing architecture '$expected' (found: $archs)"
    done
}

fuse_pc_dir_for_x86_build() {
    local pc_dir

    pc_dir="$(pkg-config --variable=pcfiledir fuse 2>/dev/null || true)"
    [ -n "$pc_dir" ] || die "pkg-config did not report a fuse pcfiledir"
    [ -d "$pc_dir" ] || die "fuse pkg-config directory does not exist: $pc_dir"
    [ -f "$pc_dir/fuse.pc" ] || die "fuse.pc was not found in $pc_dir"

    printf '%s\n' "$pc_dir"
}

prepend_path_if_missing() {
    local value="$1"
    local existing="${2:-}"

    if [ -z "$existing" ]; then
        printf '%s\n' "$value"
        return 0
    fi

    case ":$existing:" in
        *":$value:"*)
            printf '%s\n' "$existing"
            ;;
        *)
            printf '%s:%s\n' "$value" "$existing"
            ;;
    esac
}

make_universal_binary() {
    local name="$1"
    local output="$2"
    local arm_binary="$ARM_RELEASE_DIR/$name"
    local x86_binary="$X86_RELEASE_DIR/$name"

    [ -f "$arm_binary" ] || die "missing arm64 release binary: $arm_binary"
    [ -x "$arm_binary" ] || die "arm64 release binary is not executable: $arm_binary"
    [ -f "$x86_binary" ] || die "missing x86_64 release binary: $x86_binary"
    [ -x "$x86_binary" ] || die "x86_64 release binary is not executable: $x86_binary"

    assert_macho_archs_exact "$arm_binary" arm64
    assert_macho_archs_exact "$x86_binary" x86_64

    mkdir -p "$(dirname "$output")"
    rm -f "$output"
    lipo -create "$arm_binary" "$x86_binary" -output "$output"
    chmod +x "$output"
    assert_macho_archs_exact "$output" arm64 x86_64
    universal_binary_outputs+=("$output")
}

build_universal_rust_binaries() {
    local fuse_pc_dir
    local x86_pkg_config_path

    require_command cargo
    require_command lipo
    require_command pkg-config
    require_installed_rust_target "$ARM_TARGET"
    require_installed_rust_target "$X86_TARGET"

    fuse_pc_dir="$(fuse_pc_dir_for_x86_build)"
    x86_pkg_config_path="$(prepend_path_if_missing "$fuse_pc_dir" "${PKG_CONFIG_PATH_x86_64_apple_darwin:-}")"

    note "building $ARM_TARGET release binaries"
    (cd "$REPO_ROOT" && RUSTFLAGS="$release_rustflags_value" cargo build --release --locked --target "$ARM_TARGET")

    note "building $X86_TARGET release binaries"
    (
        cd "$REPO_ROOT"
        PKG_CONFIG_ALLOW_CROSS_x86_64_apple_darwin=1 \
            PKG_CONFIG_PATH_x86_64_apple_darwin="$x86_pkg_config_path" \
            RUSTFLAGS="$release_rustflags_value" cargo build --release --locked --target "$X86_TARGET"
    )

    make_universal_binary "mcraw4vulkan" "$UNIVERSAL_RELEASE_DIR/mcraw4vulkan"
    make_universal_binary "mcraw4vulkan-gui" "$UNIVERSAL_RELEASE_DIR/mcraw4vulkan-gui"
    make_universal_binary "mcraw4vulkan-preflight-check" "$UNIVERSAL_RELEASE_DIR/mcraw4vulkan-preflight-check"
}

otool_deps() {
    local image="$1"
    otool -L "$image" | sed -n '2,$s/^[[:space:]]*\(.*\) (compatibility version.*$/\1/p'
}

otool_id() {
    local image="$1"
    otool_deps "$image" | sed -n '1p'
}

mach_o_rpaths() {
    local image="$1"
    otool -l "$image" | awk '
        $1 == "cmd" && $2 == "LC_RPATH" { in_rpath = 1; next }
        in_rpath && $1 == "path" {
            sub(/^[[:space:]]*path /, "")
            sub(/ \(offset [0-9]+\)$/, "")
            print
            in_rpath = 0
        }
    '
}

expand_dyld_path() {
    local dyld_ref="$1"
    local image="$2"
    local image_dir
    image_dir="$(cd "$(dirname "$image")" && pwd)"

    case "$dyld_ref" in
        @loader_path/*)
            printf '%s/%s\n' "$image_dir" "${dyld_ref#@loader_path/}"
            ;;
        @executable_path/*)
            printf '%s/%s\n' "$out_dir/bin" "${dyld_ref#@executable_path/}"
            ;;
        /*)
            printf '%s\n' "$dyld_ref"
            ;;
        *)
            printf '%s/%s\n' "$image_dir" "$dyld_ref"
            ;;
    esac
}

resolve_dylib_reference() {
    local ref="$1"
    local image="$2"
    local label="$3"
    local candidate
    local expanded_rpath
    local rpath
    local suffix

    case "$ref" in
        @loader_path/*|@executable_path/*)
            candidate="$(expand_dyld_path "$ref" "$image")"
            if [ -e "$candidate" ]; then
                abs_path "$candidate"
                return 0
            fi
            ;;
        @rpath/*)
            suffix="${ref#@rpath/}"
            while IFS= read -r rpath; do
                [ -n "$rpath" ] || continue
                expanded_rpath="$(expand_dyld_path "$rpath" "$image")"
                candidate="$expanded_rpath/$suffix"
                if [ -e "$candidate" ]; then
                    abs_path "$candidate"
                    return 0
                fi
            done < <(mach_o_rpaths "$image")

            candidate="$(cd "$(dirname "$image")" && pwd)/$suffix"
            if [ -e "$candidate" ]; then
                abs_path "$candidate"
                return 0
            fi
            ;;
        /*)
            if [ -e "$ref" ]; then
                abs_path "$ref"
                return 0
            fi
            ;;
    esac

    die "could not resolve $label dependency $ref from $image"
}

is_system_dylib_reference() {
    local ref="$1"
    case "$ref" in
        /usr/lib/*|/System/Library/*)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

framework_install_name() {
    local framework_name="$1"
    printf '@rpath/%s.framework/Versions/A/%s\n' "$framework_name" "$framework_name"
}

framework_binary_path() {
    local framework_dir="$1"
    local framework_name="$2"

    if [ -f "$framework_dir/Versions/A/$framework_name" ]; then
        printf '%s\n' "$framework_dir/Versions/A/$framework_name"
        return 0
    fi
    if [ -f "$framework_dir/$framework_name" ]; then
        printf '%s\n' "$framework_dir/$framework_name"
        return 0
    fi

    die "could not find $framework_name framework binary in $framework_dir"
}

find_framework_dependency_reference() {
    local image="$1"
    local framework_name="$2"
    local dep

    while IFS= read -r dep; do
        case "$dep" in
            @rpath/"$framework_name".framework/"$framework_name"|\
            @rpath/"$framework_name".framework/Versions/*/"$framework_name"|\
            */"$framework_name".framework/"$framework_name"|\
            */"$framework_name".framework/Versions/*/"$framework_name")
                printf '%s\n' "$dep"
                return 0
                ;;
        esac

        case "$framework_name:$(basename "$dep")" in
            SDL2:libSDL2.dylib|SDL2:libSDL2-2.0.0.dylib)
                die "$image links to SDL2 as a dylib ($dep); expected SDL2 Classic framework from /Library/Frameworks/SDL2.framework"
                ;;
            SDL2_image:libSDL2_image.dylib|SDL2_image:libSDL2_image-2.0.0.dylib)
                die "$image links to SDL2_image as a dylib ($dep); expected SDL2_image.framework if SDL2_image is needed"
                ;;
        esac
    done < <(otool_deps "$image")

    return 1
}

find_required_framework_dependency_reference() {
    local image="$1"
    local framework_name="$2"

    if find_framework_dependency_reference "$image" "$framework_name"; then
        return 0
    fi

    die "could not detect $framework_name.framework dependency from otool -L: $image"
}

framework_source_from_library_frameworks() {
    local framework_name="$1"
    local src="/Library/Frameworks/$framework_name.framework"

    [ -d "$src" ] || die "$framework_name.framework is required at $src"

    case "$src" in
        *sdl2-compat*)
            die "$framework_name source path uses sdl2-compat: $src. Use SDL2 Classic framework from /Library/Frameworks/SDL2.framework"
            ;;
    esac

    printf '%s\n' "$src"
}

append_unique_packaged_framework_binary() {
    local value="$1"
    local existing
    if [ "${#packaged_framework_binaries[@]}" -gt 0 ]; then
        for existing in "${packaged_framework_binaries[@]}"; do
            if [ "$existing" = "$value" ]; then
                return 0
            fi
        done
    fi
    packaged_framework_binaries+=("$value")
}

copy_framework_required() {
    local src="$1"
    local dst="$2"

    [ -d "$src" ] || die "missing required framework: $src"
    rm -rf "$dst"

    if command -v ditto >/dev/null 2>&1; then
        ditto "$src" "$dst"
        framework_copy_tool="ditto"
    else
        cp -R "$src" "$dst"
        framework_copy_tool="cp -R"
    fi

    remove_framework_launch_blocking_xattrs "$dst" "after copy"
}

remove_framework_launch_blocking_xattrs() {
    local framework="$1"
    local phase="$2"
    local xattrs

    xattrs="$(xattr -lr "$framework" 2>/dev/null || true)"

    if printf '%s\n' "$xattrs" | grep -q 'com.apple.quarantine'; then
        xattr -dr com.apple.quarantine "$framework" 2>/dev/null || die "failed to remove quarantine xattrs from $framework"
        framework_xattr_cleanups+=("$framework: removed com.apple.quarantine $phase")
    fi
    if printf '%s\n' "$xattrs" | grep -q 'com.apple.provenance'; then
        xattr -dr com.apple.provenance "$framework" 2>/dev/null || true
        framework_xattr_cleanups+=("$framework: attempted com.apple.provenance removal $phase")
    fi
}

assert_no_forbidden_sdl_strings() {
    local image="$1"
    local label="$2"
    local matches=()
    local line

    while IFS= read -r line; do
        case "$line" in
            *sdl2-compat*|*SDL3_DYNAMIC_API*|*"Failed loading SDL3 library"*)
                matches+=("$line")
                ;;
        esac
    done < <(strings "$image" 2>/dev/null || true)

    if [ "${#matches[@]}" -gt 0 ]; then
        printf '%s\n' "$label contains forbidden SDL compatibility strings:" >&2
        printf '  %s\n' "${matches[@]}" >&2
        die "use SDL2 Classic framework from /Library/Frameworks/SDL2.framework; do not bundle sdl2-compat or SDL3"
    fi

    sdl2_forbidden_string_check="passed"
}

rewrite_framework_dependency_if_present() {
    local image="$1"
    local framework_name="$2"
    local dep

    if dep="$(find_framework_dependency_reference "$image" "$framework_name")"; then
        rewrite_dependency "$image" "$dep" "$(framework_install_name "$framework_name")"
    fi
}

detect_sdl2_image_requirement() {
    local image
    local dep

    for image in "$@"; do
        if dep="$(find_framework_dependency_reference "$image" "SDL2_image")"; then
            sdl2_image_inclusion_reason="$image depends on $dep"
            return 0
        fi
    done

    return 1
}

bundle_sdl2_image_framework_if_needed() {
    local cli_real="$1"
    local gui_real="$2"
    local sdl2_framework_binary="$3"
    local sdl2_image_binary
    local sdl2_image_install_name

    if ! detect_sdl2_image_requirement "$cli_real" "$gui_real" "$sdl2_framework_binary"; then
        return 0
    fi

    sdl2_image_src="$(framework_source_from_library_frameworks "SDL2_image")"
    sdl2_image_package_path="$out_dir/Frameworks/SDL2_image.framework"
    sdl2_image_install_name="$(framework_install_name "SDL2_image")"

    copy_framework_required "$sdl2_image_src" "$sdl2_image_package_path"
    sdl2_image_binary="$(framework_binary_path "$sdl2_image_package_path" "SDL2_image")"
    append_unique_packaged_framework_binary "$sdl2_image_binary"

    set_dylib_id "$sdl2_image_binary" "$sdl2_image_install_name"
    rewrite_framework_dependency_if_present "$sdl2_image_binary" "SDL2"
    rewrite_framework_dependency_if_present "$cli_real" "SDL2_image"
    rewrite_framework_dependency_if_present "$gui_real" "SDL2_image"
    rewrite_framework_dependency_if_present "$sdl2_framework_binary" "SDL2_image"
}

bundle_sdl2_runtime() {
    local cli_real="$1"
    local gui_real="$2"
    local cli_sdl2_ref
    local gui_sdl2_ref
    local sdl2_binary
    local sdl2_install_name

    cli_sdl2_ref="$(find_required_framework_dependency_reference "$cli_real" "SDL2")"
    gui_sdl2_ref="$(find_required_framework_dependency_reference "$gui_real" "SDL2")"
    sdl2_src="$(framework_source_from_library_frameworks "SDL2")"
    sdl2_package_path="$out_dir/Frameworks/SDL2.framework"
    sdl2_install_name="$(framework_install_name "SDL2")"

    case "$cli_sdl2_ref $gui_sdl2_ref" in
        *sdl2-compat*)
            die "detected sdl2-compat in SDL2 dependency references: $cli_sdl2_ref $gui_sdl2_ref"
            ;;
    esac

    copy_framework_required "$sdl2_src" "$sdl2_package_path"
    sdl2_binary="$(framework_binary_path "$sdl2_package_path" "SDL2")"
    append_unique_packaged_framework_binary "$sdl2_binary"

    set_dylib_id "$sdl2_binary" "$sdl2_install_name"
    add_rpath_if_missing "$cli_real" "@loader_path/../Frameworks"
    add_rpath_if_missing "$gui_real" "@loader_path/../Frameworks"
    rewrite_dependency "$cli_real" "$cli_sdl2_ref" "$sdl2_install_name"
    rewrite_dependency "$gui_real" "$gui_sdl2_ref" "$sdl2_install_name"

    assert_no_forbidden_sdl_strings "$sdl2_binary" "copied SDL2.framework binary"
    bundle_sdl2_image_framework_if_needed "$cli_real" "$gui_real" "$sdl2_binary"
}

rewrite_dependency() {
    local image="$1"
    local old_ref="$2"
    local new_ref="$3"

    if [ "$old_ref" = "$new_ref" ]; then
        return 0
    fi

    chmod u+w "$image"
    install_name_tool -change "$old_ref" "$new_ref" "$image"
    append_unique_modified_macho_file "$image"
    record_install_name_tool_change "install_name_tool -change $old_ref $new_ref $image"
}

set_dylib_id() {
    local image="$1"
    local new_id="$2"

    if [ "$(otool_id "$image")" = "$new_id" ]; then
        return 0
    fi

    chmod u+w "$image"
    install_name_tool -id "$new_id" "$image"
    append_unique_modified_macho_file "$image"
    record_install_name_tool_change "install_name_tool -id $new_id $image"
}

add_rpath_if_missing() {
    local image="$1"
    local rpath="$2"
    local existing

    while IFS= read -r existing; do
        if [ "$existing" = "$rpath" ]; then
            return 0
        fi
    done < <(mach_o_rpaths "$image")

    chmod u+w "$image"
    install_name_tool -add_rpath "$rpath" "$image"
    append_unique_modified_macho_file "$image"
    record_install_name_tool_change "install_name_tool -add_rpath $rpath $image"
}

sanitize_vulkan_loader_search_paths() {
    local image
    local count

    for image in "$@"; do
        [ -e "$image" ] || die "missing packaged Vulkan loader dylib: $image"
        count="$(strings "$image" 2>/dev/null | grep -c '^/usr/local/share:/usr/share$' || true)"
        if [ "$count" -eq 0 ]; then
            continue
        fi

        chmod u+w "$image"
        perl -0pi -e 's#/usr/local/share:/usr/share#"/usr/share:/usr/share" . ("\0" x 6)#ge' "$image"
        append_unique_modified_macho_file "$image"
        vulkan_string_sanitizations+=("$image: replaced $count embedded /usr/local/share:/usr/share fallback string(s) with /usr/share:/usr/share")
    done
}

rewrite_vulkan_loader_dependency_refs() {
    local image="$1"
    local dep
    local dep_basename

    while IFS= read -r dep; do
        [ -n "$dep" ] || continue
        dep_basename="$(basename "$dep")"

        case "$dep" in
            /opt/homebrew/*|/usr/local/*|*/Cellar/*|*/Homebrew/*)
                if [ "$dep_basename" = "libvulkan.1.dylib" ] || [ "$dep_basename" = "libvulkan.dylib" ]; then
                    rewrite_dependency "$image" "$dep" "@loader_path/libvulkan.1.dylib"
                fi
                ;;
        esac
    done < <(otool_deps "$image")
}

assert_no_vulkan_loader_leaks() {
    local image
    local dep

    for image in "$@"; do
        [ -e "$image" ] || die "missing packaged Vulkan loader dylib: $image"
        while IFS= read -r dep; do
            case "$dep" in
                /opt/homebrew/*|/usr/local/*|*/Cellar/*|*/Homebrew/*)
                    die "packaged Vulkan loader still references non-package dependency: $image -> $dep"
                    ;;
            esac
        done < <(otool_deps "$image")
    done
}

fix_vulkan_loader_install_names() {
    local vulkan_1="$out_dir/lib/libvulkan.1.dylib"
    local vulkan="$out_dir/lib/libvulkan.dylib"
    local old_scope="${install_name_tool_change_scope:-}"

    [ -e "$vulkan_1" ] || die "missing packaged Vulkan loader dylib: $vulkan_1"
    [ -e "$vulkan" ] || die "missing packaged Vulkan loader dylib: $vulkan"

    install_name_tool_change_scope="vulkan"
    set_dylib_id "$vulkan_1" "@loader_path/libvulkan.1.dylib"
    rewrite_vulkan_loader_dependency_refs "$vulkan_1"

    if [ ! -L "$vulkan" ]; then
        set_dylib_id "$vulkan" "@loader_path/libvulkan.dylib"
        rewrite_vulkan_loader_dependency_refs "$vulkan"
    fi
    install_name_tool_change_scope="$old_scope"

    assert_no_vulkan_loader_leaks "$vulkan_1" "$vulkan"
}

clean_package_xattrs_before_signing() {
    local remaining_quarantine

    xattr -cr "$out_dir" 2>/dev/null || die "failed to clear xattrs from staged package: $out_dir"
    package_xattr_cleanup_status="xattr -cr applied to staged package"

    remaining_quarantine="$(xattr -lr "$out_dir" 2>/dev/null | grep 'com.apple.quarantine' || true)"
    if [ -n "$remaining_quarantine" ]; then
        printf '%s\n' "staged package still contains quarantine xattrs:" >&2
        printf '%s\n' "$remaining_quarantine" >&2
        die "failed to remove quarantine xattrs from staged package"
    fi
    package_quarantine_check_status="passed"
}

ad_hoc_sign_regular_macho_file() {
    local file="$1"

    [ -f "$file" ] || die "missing Mach-O file to sign: $file"
    [ ! -L "$file" ] || return 0

    if codesign --force --sign - "$file" >/dev/null 2>&1; then
        ad_hoc_signed_files+=("$file")
    else
        ad_hoc_sign_failures+=("$file")
        return 1
    fi

    if codesign --verify --strict --verbose=4 "$file" >/dev/null 2>&1; then
        codesign_verified_files+=("$file")
    else
        ad_hoc_sign_failures+=("$file verification")
        return 1
    fi
}

ad_hoc_sign_framework_bundle() {
    local framework="$1"

    [ -d "$framework" ] || return 0

    if codesign --force --sign - "$framework" >/dev/null 2>&1; then
        ad_hoc_signed_frameworks+=("$framework")
    else
        ad_hoc_framework_sign_failures+=("$framework")
        return 1
    fi

    if codesign --verify --deep --strict --verbose=4 "$framework" >/dev/null 2>&1; then
        codesign_verified_frameworks+=("$framework")
    else
        ad_hoc_framework_sign_failures+=("$framework verification")
        return 1
    fi
}

ad_hoc_sign_final_package_code() {
    local file
    local framework

    require_command codesign
    codesign_status="attempted"

    while IFS= read -r file; do
        ad_hoc_sign_regular_macho_file "$file" || true
    done < <(find "$out_dir/lib" -maxdepth 1 -type f -name '*.dylib' -print | sort)

    for framework in "$out_dir/Frameworks/SDL2.framework" "$out_dir/Frameworks/SDL2_image.framework"; do
        ad_hoc_sign_framework_bundle "$framework" || true
    done

    ad_hoc_sign_regular_macho_file "$out_dir/bin/$CLI_PACKAGED_BINARY_NAME" || true
    ad_hoc_sign_regular_macho_file "$out_dir/bin/$GUI_PACKAGED_BINARY_NAME" || true
    ad_hoc_sign_regular_macho_file "$out_dir/bin/$GUI_PREFLIGHT_PACKAGED_BINARY_NAME" || true

    if [ "${#ad_hoc_sign_failures[@]}" -gt 0 ]; then
        printf '%s\n' "ad-hoc sign or verification failures:" >&2
        printf '  %s\n' "${ad_hoc_sign_failures[@]}" >&2
        die "failed to sign final staged Mach-O files"
    fi

    if [ "${#ad_hoc_framework_sign_failures[@]}" -gt 0 ]; then
        printf '%s\n' "framework ad-hoc sign or verification failures:" >&2
        printf '  %s\n' "${ad_hoc_framework_sign_failures[@]}" >&2
        die "failed to sign final staged framework bundle"
    fi
}

scan_package_runtime_deps() {
    local image
    local dep
    local line

    package_homebrew_leaks=()
    package_unexpected_usr_local_deps=()
    package_external_framework_deps=()
    package_allowed_macfuse_deps=()
    package_forbidden_string_leaks=()

    while IFS= read -r image; do
        if ! otool -L "$image" >/dev/null 2>&1; then
            continue
        fi

        while IFS= read -r dep; do
            case "$dep" in
                *sdl2-compat*|*SDL3_DYNAMIC_API*|*"Failed loading SDL3 library"*|*"$CLI_WRAPPER_NAME-$OLD_PACKAGED_BINARY_SUFFIX"*|*"$GUI_WRAPPER_NAME-$OLD_PACKAGED_BINARY_SUFFIX"*)
                    package_forbidden_string_leaks+=("$image -> $dep")
                    ;;
                /usr/lib/*|/System/Library/*)
                    ;;
                /usr/local/lib/libfuse.2.dylib)
                    append_unique_allowed_macfuse_dep "$image -> $dep"
                    ;;
                /opt/homebrew/*|*/Cellar/*|*/Homebrew/*)
                    package_homebrew_leaks+=("$image -> $dep")
                    ;;
                /Library/Frameworks/SDL2.framework/*|/Library/Frameworks/SDL2_image.framework/*)
                    package_external_framework_deps+=("$image -> $dep")
                    ;;
                /usr/local/*)
                    package_unexpected_usr_local_deps+=("$image -> $dep")
                    ;;
            esac
        done < <(otool_deps "$image")

        while IFS= read -r line; do
            case "$line" in
                *"/usr/local/lib/libfuse.2.dylib"*)
                    append_unique_allowed_macfuse_dep "$image -> /usr/local/lib/libfuse.2.dylib"
                    ;;
                *"/opt/homebrew"*|*"Cellar"*|*"Homebrew"*|*"sdl2-compat"*|*"SDL3_DYNAMIC_API"*|*"Failed loading SDL3 library"*|*"$CLI_WRAPPER_NAME-$OLD_PACKAGED_BINARY_SUFFIX"*|*"$GUI_WRAPPER_NAME-$OLD_PACKAGED_BINARY_SUFFIX"*|*"/Library/Frameworks/SDL2.framework"*|*"/Library/Frameworks/SDL2_image.framework"*)
                    package_forbidden_string_leaks+=("$image -> $line")
                    ;;
                *"/usr/local"*)
                    package_unexpected_usr_local_deps+=("$image -> $line")
                    ;;
            esac
        done < <(strings "$image" 2>/dev/null || true)
    done < <(
        {
            printf '%s\n' "$out_dir/bin/$CLI_PACKAGED_BINARY_NAME"
            printf '%s\n' "$out_dir/bin/$GUI_PACKAGED_BINARY_NAME"
            printf '%s\n' "$out_dir/bin/$GUI_PREFLIGHT_PACKAGED_BINARY_NAME"
            find "$out_dir/lib" -maxdepth 1 \( -type f -o -type l \) -name '*.dylib' -print
            find "$out_dir/Frameworks" \( -type f -o -type l \) -print
        } | sort -u
    )

    if [ "${#package_homebrew_leaks[@]}" -gt 0 ]; then
        printf '%s\n' "Homebrew runtime dependency leaks:" >&2
        printf '  %s\n' "${package_homebrew_leaks[@]}" >&2
        die "package contains Homebrew runtime dependency leaks"
    fi

    if [ "${#package_external_framework_deps[@]}" -gt 0 ]; then
        printf '%s\n' "external /Library/Frameworks runtime dependencies:" >&2
        printf '  %s\n' "${package_external_framework_deps[@]}" >&2
        die "package contains external SDL framework runtime dependencies"
    fi

    if [ "${#package_unexpected_usr_local_deps[@]}" -gt 0 ]; then
        printf '%s\n' "unexpected /usr/local runtime dependencies:" >&2
        printf '  %s\n' "${package_unexpected_usr_local_deps[@]}" >&2
        die "package contains unexpected /usr/local runtime dependencies"
    fi

    if [ "${#package_forbidden_string_leaks[@]}" -gt 0 ]; then
        printf '%s\n' "forbidden package strings or paths:" >&2
        printf '  %s\n' "${package_forbidden_string_leaks[@]}" >&2
        die "package contains forbidden Homebrew, sdl2-compat, SDL3, or external SDL framework strings"
    fi
}

assert_wrappers_no_sdl_dynamic_api() {
    local wrapper
    local matches

    for wrapper in "$out_dir/bin/mcraw4vulkan" "$out_dir/bin/mcraw4vulkan-gui"; do
        matches="$(grep -nE 'SDL3_DYNAMIC_API|SDL_DYNAMIC_API' "$wrapper" 2>/dev/null || true)"
        if [ -n "$matches" ]; then
            printf '%s\n' "wrapper contains forbidden SDL dynamic API environment:" >&2
            printf '%s\n' "$matches" >&2
            die "wrappers must not set SDL3_DYNAMIC_API or SDL_DYNAMIC_API"
        fi
    done
}

write_runtime_wrapper() {
    local wrapper_path="$1"
    local real_binary_name="$2"

    cat > "$wrapper_path" <<WRAPPER
#!/usr/bin/env bash
set -e

BIN_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="\$(cd "\$BIN_DIR/.." && pwd)"

export WGPU_BACKEND=vulkan

export VK_ICD_FILENAMES="\$PACKAGE_ROOT/vulkan/icd.d/MoltenVK_icd.json"

if [ -n "\${DYLD_LIBRARY_PATH:-}" ]; then
    export DYLD_LIBRARY_PATH="\$PACKAGE_ROOT/lib:\$DYLD_LIBRARY_PATH"
else
    export DYLD_LIBRARY_PATH="\$PACKAGE_ROOT/lib"
fi

if [ -n "\${DYLD_FRAMEWORK_PATH:-}" ]; then
    export DYLD_FRAMEWORK_PATH="\$PACKAGE_ROOT/Frameworks:\$DYLD_FRAMEWORK_PATH"
else
    export DYLD_FRAMEWORK_PATH="\$PACKAGE_ROOT/Frameworks"
fi

exec "\$PACKAGE_ROOT/bin/$real_binary_name" "\$@"
WRAPPER
    chmod +x "$wrapper_path"
}

write_gui_runtime_wrapper() {
    local wrapper_path="$1"
    local launcher_binary_name="$2"
    local real_gui_binary_name="$3"

    cat > "$wrapper_path" <<WRAPPER
#!/usr/bin/env bash
set -e

BIN_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="\$(cd "\$BIN_DIR/.." && pwd)"

export WGPU_BACKEND=vulkan

export VK_ICD_FILENAMES="\$PACKAGE_ROOT/vulkan/icd.d/MoltenVK_icd.json"

if [ -n "\${DYLD_LIBRARY_PATH:-}" ]; then
    export DYLD_LIBRARY_PATH="\$PACKAGE_ROOT/lib:\$DYLD_LIBRARY_PATH"
else
    export DYLD_LIBRARY_PATH="\$PACKAGE_ROOT/lib"
fi

if [ -n "\${DYLD_FRAMEWORK_PATH:-}" ]; then
    export DYLD_FRAMEWORK_PATH="\$PACKAGE_ROOT/Frameworks:\$DYLD_FRAMEWORK_PATH"
else
    export DYLD_FRAMEWORK_PATH="\$PACKAGE_ROOT/Frameworks"
fi

exec "\$PACKAGE_ROOT/bin/$launcher_binary_name" --gui "\$PACKAGE_ROOT/bin/$real_gui_binary_name" "\$@"
WRAPPER
    chmod +x "$wrapper_path"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --binary)
            [ "$#" -ge 2 ] || die "--binary requires a path"
            binary="$2"
            shift 2
            ;;
        --out-dir)
            [ "$#" -ge 2 ] || die "--out-dir requires a path"
            out_dir="$2"
            shift 2
            ;;
        --runtime-root)
            [ "$#" -ge 2 ] || die "--runtime-root requires a path"
            runtime_roots+=("$2")
            shift 2
            ;;
        --component-version)
            [ "$#" -ge 2 ] || die "--component-version requires COMPONENT=VERSION"
            record_component_version "$2"
            shift 2
            ;;
        --license-file)
            [ "$#" -ge 2 ] || die "--license-file requires COMPONENT=PATH"
            record_legal_input license "$2"
            shift 2
            ;;
        --notice-file)
            [ "$#" -ge 2 ] || die "--notice-file requires COMPONENT=PATH"
            record_legal_input notice "$2"
            shift 2
            ;;
        --discovery-only)
            discovery_only=1
            shift
            ;;
        --print-project-version)
            print_project_version=1
            shift
            ;;
        --build)
            run_build=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

for required_tool in cargo python3; do
    require_command "$required_tool"
done
if ! project_version="$(resolve_project_version)"; then
    die "failed to resolve the current mcraw4vulkan package version from Cargo metadata"
fi
case "$project_version" in
    ''|*[!0-9A-Za-z.+-]*)
        die "Cargo metadata returned an unsafe project version: $project_version"
        ;;
esac
if [ "$print_project_version" -eq 1 ]; then
    printf '%s\n' "$project_version"
    exit 0
fi

if [ -n "${CARGO_TARGET_DIR:-}" ]; then
    case "$CARGO_TARGET_DIR" in
        /*) cargo_target_root="$CARGO_TARGET_DIR" ;;
        *) cargo_target_root="$REPO_ROOT/$CARGO_TARGET_DIR" ;;
    esac
else
    cargo_target_root="$REPO_ROOT/target"
fi
package_name="mcraw4vulkan-macos-$project_version"
ARM_RELEASE_DIR="$cargo_target_root/$ARM_TARGET/release"
X86_RELEASE_DIR="$cargo_target_root/$X86_TARGET/release"
UNIVERSAL_RELEASE_DIR="$cargo_target_root/$UNIVERSAL_TARGET/release"
binary="${binary:-$UNIVERSAL_RELEASE_DIR/mcraw4vulkan}"
gui_binary="${gui_binary:-$UNIVERSAL_RELEASE_DIR/mcraw4vulkan-gui}"
preflight_binary="${preflight_binary:-$UNIVERSAL_RELEASE_DIR/mcraw4vulkan-preflight-check}"
out_dir="${out_dir:-$cargo_target_root/macos-cli-package/$package_name}"

# Validate the complete legal view before spending time on universal release builds.
# Final mode refuses unresolved rows; discovery mode records them without promotion.
validate_current_dependency_closure
validate_artifact_legal_manifest "$ARTIFACT_LEGAL_MANIFEST" ""

cargo_lock_sha256="$(sha256_file "$REPO_ROOT/Cargo.lock")"
if command -v git >/dev/null 2>&1 &&
    git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    source_revision="$(git -C "$REPO_ROOT" rev-parse HEAD)"
    if [ -n "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=normal)" ]; then
        source_tree_state="dirty"
    else
        source_tree_state="clean"
    fi
fi

if [ "$(uname -s)" != "Darwin" ]; then
    note "warning: this package builder is intended for Darwin hosts"
fi

for required_tool in cargo lipo pkg-config; do
    require_command "$required_tool"
done

if [ "$run_build" -eq 1 ]; then
    note "warning: --build is no longer needed; Universal release targets are built by default"
fi

release_rustflags_value="$(release_rustflags)"
build_universal_rust_binaries

binary="$(abs_path "$binary")"
gui_binary="$(abs_path "$gui_binary")"
preflight_binary="$(abs_path "$preflight_binary")"
out_dir="$(abs_path "$out_dir")"

[ -f "$binary" ] || die "compiled binary not found: $binary"
[ -x "$binary" ] || die "compiled binary is not executable: $binary"
[ -f "$gui_binary" ] || die "compiled GUI binary not found: $gui_binary"
[ -x "$gui_binary" ] || die "compiled GUI binary is not executable: $gui_binary"
[ -f "$preflight_binary" ] || die "compiled GUI preflight launcher not found: $preflight_binary"
[ -x "$preflight_binary" ] || die "compiled GUI preflight launcher is not executable: $preflight_binary"
require_command otool
require_command install_name_tool
for required_tool in rm mkdir cp chmod grep strings perl xattr codesign; do
    require_command "$required_tool"
done

loader_candidates=()
vulkan_dylib_candidates=()
molten_candidates=()
icd_candidates=()

append_value "${MCRAW4VULKAN_VULKAN_LOADER:-}" loader
append_value "${MCRAW4VULKAN_LIBVULKAN_1_DYLIB:-}" loader
append_value "${MCRAW4VULKAN_VULKAN_DYLIB:-}" vulkan_dylib
append_value "${MCRAW4VULKAN_LIBVULKAN_DYLIB:-}" vulkan_dylib
append_value "${MCRAW4VULKAN_MOLTENVK_DYLIB:-}" molten
append_value "${MCRAW4VULKAN_LIBMOLTENVK_DYLIB:-}" molten
append_value "${MCRAW4VULKAN_MOLTENVK_ICD_JSON:-}" icd

if [ -n "${VK_ICD_FILENAMES:-}" ]; then
    old_ifs="$IFS"
    IFS=':'
    for icd_path in $VK_ICD_FILENAMES; do
        if [ -e "$icd_path" ]; then
            append_value "$icd_path" icd
        fi
    done
    IFS="$old_ifs"
fi

if [ -n "${VULKAN_SDK:-}" ]; then
    append_root_candidates "$VULKAN_SDK"
fi

for runtime_root in "${runtime_roots[@]}"; do
    append_root_candidates "$runtime_root"
done

append_pkg_config_root "vulkan"
append_pkg_config_root "MoltenVK"

append_root_candidates "/usr/local"
append_root_candidates "/opt/homebrew"

append_find_results "/Applications" "libvulkan.1.dylib" loader
append_find_results "/Applications" "libvulkan.dylib" vulkan_dylib
append_find_results "/Applications" "libMoltenVK.dylib" molten
append_find_results "/Applications" "MoltenVK_icd.json" icd

libvulkan_1_src="$(find_first_existing "libvulkan.1.dylib" "${loader_candidates[@]}")"
libvulkan_src=""
for candidate in "${vulkan_dylib_candidates[@]}"; do
    if [ -n "$candidate" ] && [ -e "$candidate" ]; then
        libvulkan_src="$candidate"
        break
    fi
done
libmoltenvk_src="$(find_first_existing "libMoltenVK.dylib" "${molten_candidates[@]}")"
icd_src="$(find_first_existing "MoltenVK_icd.json" "${icd_candidates[@]}")"

safe_prepare_out_dir "$out_dir"

cli_packaged_binary="$out_dir/bin/$CLI_PACKAGED_BINARY_NAME"
gui_packaged_binary="$out_dir/bin/$GUI_PACKAGED_BINARY_NAME"
gui_preflight_packaged_binary="$out_dir/bin/$GUI_PREFLIGHT_PACKAGED_BINARY_NAME"
cli_wrapper="$out_dir/bin/$CLI_WRAPPER_NAME"
gui_wrapper="$out_dir/bin/$GUI_WRAPPER_NAME"

copy_required "$binary" "$cli_packaged_binary"
chmod +x "$cli_packaged_binary"
copy_required "$gui_binary" "$gui_packaged_binary"
chmod +x "$gui_packaged_binary"
copy_required "$preflight_binary" "$gui_preflight_packaged_binary"
chmod +x "$gui_preflight_packaged_binary"
assert_macho_archs_exact "$cli_packaged_binary" arm64 x86_64
assert_macho_archs_exact "$gui_packaged_binary" arm64 x86_64
assert_macho_archs_exact "$gui_preflight_packaged_binary" arm64 x86_64
bundle_sdl2_runtime "$cli_packaged_binary" "$gui_packaged_binary"

copy_required "$libvulkan_1_src" "$out_dir/lib/libvulkan.1.dylib"
if [ -n "$libvulkan_src" ]; then
    copy_required "$libvulkan_src" "$out_dir/lib/libvulkan.dylib"
else
    (cd "$out_dir/lib" && ln -s libvulkan.1.dylib libvulkan.dylib)
fi
fix_vulkan_loader_install_names
sanitize_vulkan_loader_search_paths "$out_dir/lib/libvulkan.1.dylib" "$out_dir/lib/libvulkan.dylib"
copy_required "$libmoltenvk_src" "$out_dir/lib/libMoltenVK.dylib"

icd_strategy="$(python3 - "$icd_src" "$out_dir/vulkan/icd.d/MoltenVK_icd.json" "../../lib/libMoltenVK.dylib" "$out_dir/lib/libMoltenVK.dylib" <<'PY'
from pathlib import Path
import json
import sys

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
desired_rel = sys.argv[3]
packaged_molten = Path(sys.argv[4]).resolve(strict=False)

data = json.loads(src.read_text())
icd = data.setdefault("ICD", {})
old_path = str(icd.get("library_path", ""))

def old_path_is_package_local(value: str) -> bool:
    if not value:
        return False
    path = Path(value)
    if path.is_absolute():
        return False
    resolved = (dst.parent / path).resolve(strict=False)
    return resolved == packaged_molten

if old_path_is_package_local(old_path):
    strategy = "copied unchanged"
else:
    icd["library_path"] = desired_rel
    strategy = "rewritten to package relative path"

dst.write_text(json.dumps(data, indent=4) + "\n")
print(strategy)
PY
)"

write_runtime_wrapper "$cli_wrapper" "$CLI_PACKAGED_BINARY_NAME"
write_gui_runtime_wrapper "$gui_wrapper" "$GUI_PREFLIGHT_PACKAGED_BINARY_NAME" "$GUI_PACKAGED_BINARY_NAME"

assert_wrappers_no_sdl_dynamic_api
stage_release_legal_files
clean_package_xattrs_before_signing
ad_hoc_sign_final_package_code
scan_package_runtime_deps

{
    printf '%s\n' "mcraw4vulkan macOS package"
    printf '%s\n' "project_version=$project_version"
    printf '%s\n' "package_name=$package_name"
    printf '%s\n' "source_revision=$source_revision"
    printf '%s\n' "source_tree_state=$source_tree_state"
    printf '%s\n' "cargo_lock_sha256=$cargo_lock_sha256"
    printf '%s\n' "copyright=Copyright (C) 2026 nate808-gh"
    printf '%s\n' "project_license=GPL-3.0-or-later"
    printf '%s\n' "repository=https://github.com/nate808-gh/mcraw4vulkan"
    if [ "$discovery_only" -eq 1 ]; then
        printf '%s\n' "discovery_only=yes"
        printf '%s\n' "release_ready=no"
        printf '%s\n' "disposition=NOT FOR RELEASE"
    else
        printf '%s\n' "discovery_only=no"
        printf '%s\n' "release_ready=yes"
    fi
    printf '%s\n' "legal_view_unresolved_rows=$artifact_legal_unresolved_count"
    printf '%s\n' "discovery_unowned_native_evidence_count=$discovery_native_evidence_count"
    if [ "${#legal_input_components[@]}" -gt 0 ]; then
        for ((index = 0; index < ${#legal_input_components[@]}; index++)); do
            printf 'native_evidence\t%s\t%s\t%s\t%s\t%s\n' \
                "${legal_input_components[$index]}" \
                "$(component_version "${legal_input_components[$index]}")" \
                "${legal_input_kinds[$index]}" \
                "${legal_input_basenames[$index]}" \
                "$(sha256_file "${legal_input_paths[$index]}")"
        done
    fi
    printf '%s\n' "macos_deployment_target=$MACOSX_DEPLOYMENT_TARGET"
    printf '%s\n' "rust_targets=$ARM_TARGET,$X86_TARGET"
    printf '%s\n' "rust_architectures=arm64,x86_64"
    printf '%s\n' "cli_wrapper=bin/$CLI_WRAPPER_NAME"
    printf '%s\n' "gui_wrapper=bin/$GUI_WRAPPER_NAME"
    printf '%s\n' "cli_binary=bin/$CLI_PACKAGED_BINARY_NAME"
    printf '%s\n' "gui_binary=bin/$GUI_PACKAGED_BINARY_NAME"
    printf '%s\n' "preflight_binary=bin/$GUI_PREFLIGHT_PACKAGED_BINARY_NAME"
    printf '%s\n' "sdl2_framework=Frameworks/SDL2.framework"
    printf '%s\n' "sdl2_forbidden_string_check=$sdl2_forbidden_string_check"
    printf '%s\n' "framework_copy_tool=$framework_copy_tool"
    if [ -n "$sdl2_image_package_path" ]; then
        printf '%s\n' "sdl2_image_included=yes"
        printf '%s\n' "sdl2_image_framework=Frameworks/SDL2_image.framework"
    else
        printf '%s\n' "sdl2_image_included=no"
    fi
    printf '%s\n' "packaged_framework_count=${#packaged_framework_binaries[@]}"
    printf '%s\n' "framework_xattr_cleanup_count=${#framework_xattr_cleanups[@]}"
    printf '%s\n' "install_name_change_count=${#install_name_tool_changes[@]}"
    printf '%s\n' "vulkan_install_name_change_count=${#vulkan_install_name_tool_changes[@]}"
    printf '%s\n' "vulkan_string_sanitization_count=${#vulkan_string_sanitizations[@]}"
    printf '%s\n' "package_xattr_cleanup_status=$package_xattr_cleanup_status"
    printf '%s\n' "package_quarantine_check_status=$package_quarantine_check_status"
    printf '%s\n' "signing_mode=ad-hoc"
    printf '%s\n' "notarized=no"
    printf '%s\n' "codesign_status=$codesign_status"
    printf '%s\n' "ad_hoc_signed_file_count=${#ad_hoc_signed_files[@]}"
    printf '%s\n' "ad_hoc_sign_failure_count=${#ad_hoc_sign_failures[@]}"
    printf '%s\n' "ad_hoc_signed_framework_count=${#ad_hoc_signed_frameworks[@]}"
    printf '%s\n' "codesign_verified_file_count=${#codesign_verified_files[@]}"
    printf '%s\n' "codesign_verified_framework_count=${#codesign_verified_frameworks[@]}"
    printf '%s\n' "ad_hoc_framework_sign_failure_count=${#ad_hoc_framework_sign_failures[@]}"
    printf '%s\n' "external_macfuse_dependency_count=${#package_allowed_macfuse_deps[@]}"
    printf '%s\n' "homebrew_runtime_dependency_leaks=none"
    printf '%s\n' "external_sdl_framework_runtime_dependencies=none"
    printf '%s\n' "unexpected_usr_local_runtime_dependencies=none"
    printf '%s\n' "forbidden_package_strings=none"
    printf '%s\n' "vulkan_loader=lib/libvulkan.1.dylib"
    if [ -n "$libvulkan_src" ]; then
        printf '%s\n' "vulkan_loader_alias=lib/libvulkan.dylib"
    else
        printf '%s\n' "vulkan_loader_alias=lib/libvulkan.dylib -> libvulkan.1.dylib"
    fi
    printf '%s\n' "moltenvk=lib/libMoltenVK.dylib"
    printf '%s\n' "moltenvk_icd=vulkan/icd.d/MoltenVK_icd.json"
    printf '%s\n' "MoltenVK_icd_strategy=$icd_strategy"
    printf '%s\n' "SDL2_version=$(component_version SDL2)"
    if [ -n "$sdl2_image_package_path" ]; then
        printf '%s\n' "SDL2_image_version=$(component_version SDL2_image)"
    fi
    printf '%s\n' "Vulkan-Loader_version=$(component_version Vulkan-Loader)"
    printf '%s\n' "MoltenVK_version=$(component_version MoltenVK)"
    printf '%s\n' "project_license_file=LICENSE"
    printf '%s\n' "project_notice=NOTICE"
    printf '%s\n' "third_party_notice=THIRD_PARTY_NOTICES.tsv"
    printf '%s\n' "legal_file_inventory=licenses/README.txt"
    printf '%s\n' "rust_source_path_remapping=enabled"
    printf '%s\n' "wrapper_sets_WGPU_BACKEND=always, value vulkan"
    printf '%s\n' "wrapper_sets_VK_ICD_FILENAMES=package-local MoltenVK_icd.json"
    printf '%s\n' "wrapper_sets_DYLD_LIBRARY_PATH=package lib directory prepended"
    printf '%s\n' "wrapper_sets_DYLD_FRAMEWORK_PATH=package Frameworks directory prepended"
    printf '%s\n' "wrappers=$CLI_WRAPPER_NAME, $GUI_WRAPPER_NAME"
    printf '%s\n' "gui_wrapper_launch_sequence=$GUI_WRAPPER_NAME -> $GUI_PREFLIGHT_PACKAGED_BINARY_NAME --gui $GUI_PACKAGED_BINARY_NAME"
    printf '%s\n' "layout=bin wrappers and binaries, lib dylibs, Frameworks, vulkan icd.d JSON, licenses"
} > "$out_dir/package-info.txt"

if grep -Fq "$REPO_ROOT" "$out_dir/package-info.txt" ||
    grep -Fq "$out_dir" "$out_dir/package-info.txt" ||
    grep -Eq '(^|[=[:space:]])(/Users/|/home/|/private/|[A-Za-z]:\\|\\\\)' "$out_dir/package-info.txt" ||
    grep -Eq '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}' "$out_dir/package-info.txt"; then
    die "package-info.txt contains a private or absolute build value"
fi

note "package created: $out_dir"
