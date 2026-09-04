# Linux packaging contract

This directory contains the shared desktop and icon inputs for native Linux
packages. It does not contain an Arch `PKGBUILD`, an RPM spec, or Debian package
metadata, so this source tree does not itself produce those package formats.

## Package identity

Use these project-owned values in a native recipe:

| Format | Field | Value |
| --- | --- | --- |
| Arch Linux | `license` | `GPL-3.0-or-later` |
| Arch Linux | `url` | `https://github.com/nate808-gh/mcraw4vulkan` |
| RPM | `License` | `GPL-3.0-or-later` |
| RPM | `URL` | `https://github.com/nate808-gh/mcraw4vulkan` |
| Debian control | `Homepage` | `https://github.com/nate808-gh/mcraw4vulkan` |
| Debian copyright | `License` | `GPL-3+` (GNU GPL version 3 or later) |
| Format requiring a maintainer email | maintainer | `nate808-gh <166332048+nate808-gh@users.noreply.github.com>` |
| Project copyright | copyright | `Copyright (C) 2026 nate808-gh` |

The package version must come from Cargo metadata. Do not encode a separate
version in shared packaging files.

## Build and payload

Build the locked workspace with compiler path remapping so release images do
not retain builder source, Cargo-home, Rustup-home, or user-home prefixes:

The RNA export derives current local workspace versions in `Cargo.lock` while
preserving its audited third-party dependency closure. A project source or
version change does not require a new legal seal; a third-party dependency
change still requires a new closure audit and legal-view update.

```sh
if [ -n "${CARGO_ENCODED_RUSTFLAGS+x}" ]; then
    echo "CARGO_ENCODED_RUSTFLAGS must be unset for the release build" >&2
    exit 1
fi
release_rustflags="${RUSTFLAGS:+$RUSTFLAGS }--remap-path-prefix=${HOME}=/usr/src/user --remap-path-prefix=${CARGO_HOME:-$HOME/.cargo}=/usr/src/cargo --remap-path-prefix=${RUSTUP_HOME:-$HOME/.rustup}=/usr/src/rustup --remap-path-prefix=${PWD}=/usr/src/mcraw4vulkan"
RUSTFLAGS="$release_rustflags" cargo build --workspace --release --locked
```

Install one product containing these colocated executables:

```text
${MC4V_EXEC_DIR}/mcraw4vulkan
${MC4V_EXEC_DIR}/mcraw4vulkan-gui
${MC4V_EXEC_DIR}/mcraw4vulkan-preflight-check
${BINDIR}/mcraw4vulkan -> ${MC4V_EXEC_DIR}/mcraw4vulkan
```

The GUI first looks for the CLI beside its own executable before falling back
to `PATH`, so the executable layout is functional rather than cosmetic.

Render `mcraw4vulkan.desktop.in` to
`${DATADIR}/applications/mcraw4vulkan.desktop`, replacing
`@MC4V_LIBEXECDIR@` with the absolute installed `MC4V_EXEC_DIR`. Install the
scalable icon as
`${DATADIR}/icons/hicolor/scalable/apps/mcraw4vulkan.svg`.

Install the project files and the target distribution's artifact-specific
legal view in the distribution's normal documentation or license location:

```text
LICENSE
NOTICE
THIRD_PARTY_NOTICES.tsv
licenses/<only paths selected by the artifact legal view>
```

Use exactly one of these generated views:

| Artifact | Legal view |
| --- | --- |
| Arch x86_64 | `packaging/legal/arch-x86_64.tsv` |
| Fedora 44 x86_64 | `packaging/legal/fedora44-x86_64.tsv` |
| EL10 x86_64 | `packaging/legal/el10-x86_64.tsv` |
| Ubuntu amd64 | `packaging/legal/ubuntu-amd64.tsv` |

Require the exact manifest header and a `PASS` verification status on every
row. Copy each unique, nonempty `public_legal_file_path`, verify its SHA-256
against `legal_file_sha256`, and install the selected view itself as
`THIRD_PARTY_NOTICES.tsv`. Do not copy the whole private or public `licenses/`
directory. The project-owned source grant is GNU GPL version 3 or later;
selected third-party files retain their upstream terms and text.

## Native dependencies

Map these capabilities to the target distribution's native package names:

- Rust and Cargo compatible with the declared minimum Rust version;
- a C/C++ linker and `pkg-config`;
- SDL2 development files;
- FUSE3 development files;
- SDL2, FUSE3, and the Vulkan loader at runtime;
- `fusermount3`, `/dev/fuse`, kernel FUSE support, and a working GPU-driver ICD.

Linux packages should use system native libraries and package-manager
dependencies. Do not bundle SDL2, FUSE3, the Vulkan loader, GPU drivers, Rust,
Cargo, compiler toolchains, fixture media, build logs, or `target/`.

## Validation handoff

Before distributing a native package:

- build in the target distribution's established isolated environment;
- inspect native dependency metadata and package owner/group fields;
- confirm the three executables remain colocated and the public CLI symlink
  resolves to the package-owned CLI;
- run `mcraw4vulkan --help`;
- validate the rendered desktop file when `desktop-file-validate` is available;
- inspect and privacy-scan every extracted regular file and executable string;
- confirm the project license, artifact-specific notice index, and exactly the
  manifest-selected third-party legal texts are included, with no extras;
- confirm no source path, private identity, email other than the approved
  GitHub noreply address, signing material, PDB, fixture, or build log is
  present.

No native Linux package should be described as available until its recipe,
artifact, extracted payload, metadata, and privacy checks have passed on the
target distribution.
