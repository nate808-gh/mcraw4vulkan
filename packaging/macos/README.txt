mcraw4vulkan macOS package

package-cli.sh builds locked arm64 and x86_64 release workspaces, combines the
three product executables into universal Mach-O files, and stages a relocatable
directory. The default deployment target is macOS 11.0.

Builder requirements:

- installed aarch64-apple-darwin and x86_64-apple-darwin Rust targets
- cargo, lipo, pkg-config, install_name_tool, otool, xattr, and codesign
- SDL2 Classic at /Library/Frameworks/SDL2.framework
- macFUSE development metadata discoverable as fuse.pc
- Vulkan Loader, MoltenVK, and a MoltenVK ICD JSON
- reviewed version-matched license and notice files for every bundled native
  component

This builder is intentionally present in an RNA public export. It derives the
current mcraw4vulkan version from Cargo metadata, validates the reachable
third-party dependency closure against packaging/legal/public-Cargo.lock, and
records the current lock and source identities. Local workspace versions and
source revisions are not frozen packaging prerequisites. A third-party
dependency change still requires a fresh license-closure audit and legal-view
update before distribution.

Run:

  cargo build --release
  ./packaging/macos/package-cli.sh \
    --component-version SDL2=VERSION \
    --component-version Vulkan-Loader=VERSION \
    --component-version MoltenVK=VERSION \
    --license-file SDL2=PATH \
    --license-file Vulkan-Loader=PATH \
    --license-file MoltenVK=PATH

Add --notice-file COMPONENT=PATH for any additional required notice or
copyright file. If a staged executable requires SDL2_image, both
--component-version SDL2_image=VERSION and at least one
--license-file SDL2_image=PATH are also required.

The default output, using the current Cargo project version, is:

  target/macos-cli-package/mcraw4vulkan-macos-VERSION

Entry points:

  bin/mcraw4vulkan
  bin/mcraw4vulkan-gui

The CLI wrapper executes bin/mcraw4vulkan-macos. The GUI wrapper launches
bin/mcraw4vulkan-preflight-macos with --gui and the path to
bin/mcraw4vulkan-gui-macos. Both wrappers select Vulkan, point the loader at the
package-local MoltenVK ICD JSON, and prepend package-local library and framework
directories for the launched process.

The package includes SDL2, the Vulkan Loader, MoltenVK, and the MoltenVK ICD
JSON. SDL2_image is included only when a staged Mach-O dependency requires it.
macFUSE remains separately installed and is not bundled. Install macFUSE using
its vendor instructions; the preflight check verifies installation artifacts
but does not perform a test mount.

All staged Mach-O code is ad-hoc signed and structurally verified. The package
is not signed with an Apple Developer ID and is not notarized. macOS may require
explicit user approval before first launch.

Project LICENSE and NOTICE remain at the package root. The package consumes the
single macOS legal view at packaging/legal/macos-universal.tsv, installs that
view as THIRD_PARTY_NOTICES.tsv, and copies only its unique, nonempty
public_legal_file_path entries under licenses/. Packaging stops if any manifest
row is not PASS or if a selected file does not match its recorded SHA-256.
Reviewed native-component legal inputs remain mandatory identity checks against
the manifest-owned public legal files; they do not create additional package
files or override a non-PASS row.
package-info.txt uses only package-relative paths and
contains no builder filesystem paths or signing identity. Release compilation
remaps source, Cargo-home, Rustup-home, and user-home prefixes before they can
enter the Mach-O images.
