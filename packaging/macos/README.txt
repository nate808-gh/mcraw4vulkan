mcraw4vulkan macOS source and runtime notes

The public source builds the macOS CLI and GUI through Cargo. Native developer
requirements are:

- installed aarch64-apple-darwin and/or x86_64-apple-darwin Rust targets
- SDL2 Classic at /Library/Frameworks/SDL2.framework
- a Vulkan Loader and MoltenVK implementation

Provide the SDL2 framework search path to the build command. For example:

  cargo rustc --locked --release \
      -p mcraw4vulkan --bin mcraw4vulkan -- \
      -L framework=/Library/Frameworks

The target-specific mcraw4vulkan-macfuse-ffi crate is the application's narrow
native boundary. It uses macFUSE's unmodified FUSE3 library; FUSE2 is not a
fallback. A compatible
macFUSE installation is required for DNG mounting, but is optional for application
startup, Preview, PIPE and movie export. Missing mount support is shown on the GUI
splash with an OK acknowledgment; other essential GUI prerequisites still apply.
macFUSE is not bundled.

Preflight and explicit mounting share one side-effect-free location policy. In
order, automatic discovery supports:

- /usr/local/lib/libfuse3.4.dylib with the registered filesystem bundle at
  /Library/Filesystems/macfuse.fs (upstream installer and Homebrew macfuse cask).
- /opt/local/lib/libfuse3.4.dylib with MacPorts' relocated bundle at
  /opt/local/Library/Filesystems/macfuse.fs. The registered system-bundle path
  must resolve to this bundle, as required by MacPorts' fs_link variant/notes.

The bundle and its Contents/Resources/mount_macfuse helper must exist. Discovery
follows ordinary symlinks and requires library/helper files, not directories.
The application never creates registration links or changes OS permissions.
Homebrew's cask uses the upstream installer; Apple Silicon does not imply a
separate Homebrew-prefix macFUSE library. No FUSE2 library is a fallback.

For an intentionally nonstandard installation, MCRAW4VULKAN_MACFUSE_LIBRARY may
contain one absolute library filename. This optional process-environment value
has precedence over both defaults. It is used literally, preserving spaces and
Unicode; no trimming, shell/variable expansion, path list or directory search is
performed. Empty, relative or unavailable explicit choices fail without falling
back. Known paths retain their bundle pairing; a custom file uses the installed
system-registered bundle/helper, without guessing companion paths from its name.
Moving a dylib alone does not relocate its framework/helper dependencies or
satisfy macFUSE registration and permission requirements.

The headless launcher and its GUI child inherit the same environment. A Finder
launch does not automatically inherit an override set in an unrelated terminal.
No preference, installer, elevated helper or global DYLD setting is required.
Preflight checks artifacts only; native loading and symbol errors occur only on
explicit mounting. A successful API stays loaded for the process lifetime, so
changing installations after success requires a new application launch. Failed
acquisition remains retryable. Presence does not guarantee a successful mount.

Public product assets, the macOS legal view, and required notices remain under
packaging/ and licenses/. Distribution assembly, selected SDK versions and
hashes, signing choices, staging locations, and publication decisions belong
to the maintainer's private native-host packaging tools and are not exported by
RNA-polymerase.

Packaged end users do not need Rust, pkg-config, a Vulkan SDK, Homebrew, or
other compilation tools. Consult each package's included build information for
its measured native inputs, architectures, signature state, and test status.
