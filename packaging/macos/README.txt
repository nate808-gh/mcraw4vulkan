mcraw4vulkan macOS source and runtime notes

The public source builds the macOS CLI and GUI through Cargo. Native developer
requirements are:

- installed aarch64-apple-darwin and/or x86_64-apple-darwin Rust targets
- SDL2 Classic at /Library/Frameworks/SDL2.framework
- macFUSE development metadata for FUSE3, with its supplied libfuse3 ABI dylib
  available to the linker (the standard installation uses /usr/local/lib)
- a Vulkan Loader and MoltenVK implementation

For the standard macFUSE installation, provide the native search paths only to
the build command. For example:

  LIBRARY_PATH=/usr/local/lib cargo rustc --locked --release \
      -p mcraw4vulkan --bin mcraw4vulkan -- \
      -L framework=/Library/Frameworks

The target-specific mcraw4vulkan-macfuse-ffi crate is the application's narrow
native boundary. It uses macFUSE's unmodified FUSE3 library; FUSE2 is not a
fallback. The CLI and GUI link libfuse3 at launch, so a compatible macFUSE
installation is an application prerequisite. macFUSE is not bundled.

Public product assets, the macOS legal view, and required notices remain under
packaging/ and licenses/. Distribution assembly, selected SDK versions and
hashes, signing choices, staging locations, and publication decisions belong
to the maintainer's private native-host packaging tools and are not exported by
RNA-polymerase.

Packaged end users do not need Rust, pkg-config, a Vulkan SDK, Homebrew, or
other compilation tools. Consult each package's included build information for
its measured native inputs, architectures, signature state, and test status.
