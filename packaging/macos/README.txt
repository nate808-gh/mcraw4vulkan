mcraw4vulkan 0.9.5 macOS — FSKit DNG mounts

End-user setup
Install the official macFUSE 5.4.0 package from https://macfuse.github.io/, then
mcraw4vulkan.pkg. Enable the normal macFUSE File System Extension if macOS requests
it. Mount / Mount All creates real per-clip directories under $HOME/mcraw4vulkan.
FFmpeg is optional and checked only for movie export; preview and DNG mounting do
not require it. No kernel-extension setup, developer tools, shell exports, client
selection or package-manager dependency recipe is required by the packaged app.

The application uses one macFUSE FUSE3 callback bridge with backend=fskit always
specified. Linux keeps fuser/FUSE3; Windows keeps ProjFS. Existing DNG geometry,
correction, handles, range reads, cache and prefetch owners remain shared. Each
clip keeps one owned CLI process and its bounded GPU/cache resources. There is no
fallback transport or backend chooser. A mounting failure leaves the GUI usable.

The real home parent may contain user files; it is never a volume or cleanup target.
Clip names use the existing stable source-identity suffix and collision policy.
After detachment and native cleanup, empty app-owned clip folders are removed.
The same unchanged source recreates the same path unless a new collision exists.
Replaced, unowned or nonempty folders are retained; child files are never deleted.
Explicit CLI --mountpoint paths are honored exactly by FSKit, or return an error.
No live path migration, /Volumes alias or editor-project rewrite occurs. Close
applications using DNG files before normal Unmount / Quit; busy cleanup retains
its owner and reports the reason. Unknown/replaced volumes are never unmounted.

Compatibility
Launch/preview/export retain the existing macOS 12 application floor. DNG mounting
is separate: macFUSE 5.4.0 / FUSE3 API318 (3.18.2), real home mounts and descriptor-
bound unmount have been validated on macOS 26.6.2. General FSKit availability from
15.4 does not certify every older OS's home-path or unmount behavior. The vendor
wiki's older /Volumes restriction differs from observed 5.4.0 behavior; no universal
support promise is made. Missing/inactive/unsupported mounting remains nonfatal.

Building CLI / GUI / launcher
Use the workspace's locked Rust dependencies and MSRV 1.87 (edition 2024). Native
build inputs remain Apple's compiler/linker/macOS SDK, SDL2 Classic, Vulkan Loader
and MoltenVK. Existing macOS builds use /Library/Frameworks for SDL2. The callback
bridge uses Rust ABI declarations and dynamic loading: it does not compile macFUSE,
include its headers, link -lfuse3, or link Apple's FSKit framework. The vendor's
own Xcode/SDK requirements do not impose a new Xcode version on this application.
For an existing compatible Apple SDK/toolchain and external BUILD_TARGET:

cargo rustc --release --locked --offline -p mcraw4vulkan --bin mcraw4vulkan --target-dir "$BUILD_TARGET" -- -L framework=/Library/Frameworks
cargo rustc --release --locked --offline -p mcraw4vulkan-gui --bin mcraw4vulkan-gui --target-dir "$BUILD_TARGET" -- -L framework=/Library/Frameworks
cargo rustc --release --locked --offline -p mcraw4vulkan-preflight --bin mcraw4vulkan-preflight-check --target-dir "$BUILD_TARGET" -- -L framework=/Library/Frameworks
