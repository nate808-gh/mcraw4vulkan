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

Client build input
Use a verified official macfuse-5.4.0 package (the installer includes libfuse3), or
its previously verified extracted artifact. Keep the unmodified genuine client,
matching include/fuse3 headers, notices and provenance in a durable maintainer
build-input directory outside source and disposable reports. It is not libfuse2,
macFUSE.framework's Objective-C wrapper or a separate "macFUSE FSKit SDK" download.
Do not copy an unverified shared /usr/local/lib client, which another vendor may
have replaced. The supported input is signed by Team3T5GSNBU6W, arm64+x86_64, and
binds /Library/Filesystems/macfuse.fs/Contents/Frameworks/MFMount.framework.

Package producer integration (external producer/stub source is not in this tree)
1. Use RNA-produced CLI, GUI and preflight binaries in the existing Contents/Helpers
   layout. Retain stable application identity, native launcher, GPU wrappers/resources
   and SDL2/Vulkan/MoltenVK framework arrangement. Set app/package version 0.9.5.
2. Assemble a fresh app and run the builder-only stager:
   python3 packaging/macos/stage-runtime-inputs.py --app "$CANDIDATE_APP" --inputs "$MACFUSE_INPUT_JSON" --development-only
   The single input object declares macfuse_release="5.4.0", fuse_abi=318,
   version="5.4.0 / FUSE3 3.18.2", arrangement="private-client-external-components",
   client, client_sha256, notices, and distribution_approval_reference for releases.
   The stager checks vendor signatures, slices, exact framework dependency and
   installed component identities, and records hashes as packaging provenance.
   It stages only Contents/Frameworks/macFUSE/libfuse3.4.dylib and macFUSE notices.
   It adds the application-owned @loader_path/../Frameworks rpath before signing.
   otool/lipo/Python are packaging tools only, never application runtime commands.
3. Runtime discovery prefers the signed app-private client, then verified standard
   or registered MacPorts clients. The developer override, if used deliberately,
   is strict; it is not an end-user requirement. Foreign same-named libraries are
   rejected before ABI use. Deferred native loading checks API318, pins the verified
   MFMount framework, and checks actual native image addresses. A compatible5.4.0
   reinstall is accepted without matching incidental framework/helper bytes; other
   releases/layouts require validation. No native test mount occurs at GUI launch.
4. The user-installed official package owns MFMount, helpers and FSKit extensions.
   Our app/pkg never installs, replaces, registers or removes those components,
   shared dylibs, headers, pkg-config files or vendor receipts. The client alone
   is not a complete FSKit runtime. Omitting --inputs stages no client; the app still
   launches and may use a genuine supported separately installed client.
5. Keep the normal Network Volumes usage description for user-selected video/media
   on network volumes. No privacy reset, enablement-record edit, daemon restart or
   security repair script belongs in this package. Normal consent remains the user's.
6. Preserve vendor signatures. Sign application-owned companions with the intended
   Developer ID and the existing scoped CLI mount-library.entitlements where needed.
   Ad-hoc signing is local development evidence, not notarization or redistribution
   approval. Maintainer macFUSE distribution permission remains a release input.
7. Include packaging/macos/preinstall in the existing package producer. It refuses
   replacement while app/clip owners or pending cleanup use the application. Never
   kill processes or unmount volumes from installer scripts. BundleIsRelocatable=false;
   payload is only Applications/mcraw4vulkan.app. Removal touches only our app and
   leaves vendor installations, user clips/exports and the home mount parent alone.
