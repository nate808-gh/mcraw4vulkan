# mcraw4vulkan

mcraw4vulkan is a Rust application for displaying and processing MotionCam RAW
`.mcraw` recordings.

## Public Beta status

This source tree is a public Beta. Preserve original recordings and validate
outputs before relying on them in a production workflow. Interfaces and package
details may change as Beta feedback is incorporated.

## What mcraw4vulkan does

The primary production path decodes MotionCam RAW video through wgpu's Vulkan
backend. The command-line interface can display clips, mount virtual DNG files,
write PIPE raw-video output, and run the optimizer. A graphical interface built
with SDL2, wgpu, and egui provides Quick Preview, Display, PIPE, DNG, and
optimizer workflows.

## Main features

- Vulkan/wgpu accelerated `.mcraw` decoding, with a CPU parity path available
  as an explicit manual fallback.
- CLI and GUI applications.
- Quick Preview and full decoded Display playback.
- Direct planar `yuv444p12le` PIPE output with TV range, BT.2020 primaries,
  linear transfer, BT.2020 NCL matrix coefficients, no alpha, and a fixed 1/2
  scene-linear signal scale. Supported input audio is extracted through the
  existing WAV sidecar path.
- Virtual DNG mounting with the platform's native filesystem adapter.
- Audio playback and virtual audio files where the input and platform support
  them.
- An optimizer that selects between the two supported production payload
  profiles.

## Supported operating systems

- Linux uses FUSE3 for virtual DNG mounts.
- macOS uses macFUSE for virtual DNG mounts. GPU work follows the
  MoltenVK-compatible Vulkan path.
- Windows uses Projected File System (ProjFS) for virtual DNG mounts.

The adapters share the virtual-filesystem model, but native mount lifecycle and
runtime requirements differ by operating system.

## Runtime requirements

The production decode path requires a working Vulkan loader and driver. SDL2 is
required for the graphical application, display windows, and supported audio
playback. DNG mounting additionally requires FUSE3 on Linux, macFUSE on macOS,
or ProjFS on Windows.

The preflight checker reports the platform capabilities needed by the GUI.
Package-specific runtime details are documented under `packaging/`.

## Installation

Build the three release executables from this source tree as described below,
or install them through a platform package that includes `LICENSE`,
`THIRD_PARTY_NOTICES.md`, and its bundled-component license files. Keep
`mcraw4vulkan`, `mcraw4vulkan-gui`, and
`mcraw4vulkan-preflight-check` together so the GUI can resolve its companion
processes.

## Quick start

Display a clip with the default GPU path:

```sh
mcraw4vulkan display FILE.mcraw
```

Launch the graphical interface through the preflight checker:

```sh
mcraw4vulkan-preflight-check --gui mcraw4vulkan-gui
```

Write the direct-YUV byte stream and version-3 metadata beside an output file:

```sh
mcraw4vulkan pipe --output clip.yuv444p12le FILE.mcraw
```

PIPE is a scene-linear editing derivative intended for grading. It is not
display-ready and may appear dark until exposure and a display transform are
applied in the editor; mcraw4vulkan does not add display exposure, an OETF, or
tone mapping. The GUI Pipe Example feeds these bytes to FFmpeg as-is and uses
Vulkan ProRes 4444 profile 4 with no color-conversion filter and no alpha.
ProRes remains lossy. Virtual DNG remains the camera-domain preservation
output.

Run `mcraw4vulkan --help` for the complete current command syntax.

## DNG mounting

Mount one clip as a virtual DNG and audio directory:

```sh
mcraw4vulkan dng FILE.mcraw
```

The command serves a foreground, per-clip mount under the mcraw4vulkan folder
in the user's home or profile directory. Press Ctrl-C to stop it. Unmount one
clip with `mcraw4vulkan dng unmount FILE.mcraw`, or use
`mcraw4vulkan dng unmount all` for mcraw4vulkan-owned mounts.

The macFUSE VFS supports a maximum of 64 simultaneous mounts system-wide.
Other macFUSE volumes consume the same mount slots; this is not an
mcraw4vulkan memory limit.

## Optimizer

The optimizer compares the complete `default_chunked64` and `offset_prefetch`
payload profiles using Display and PIPE measurements, applies the production
no-loss and throughput policy, and stores only the selected payload profile.

```sh
mcraw4vulkan optimizer FILE.mcraw
```

Restore the default selection with `mcraw4vulkan optimizer
--restore-defaults`.

## Build from source

The workspace declares Rust 1.87 as its minimum supported Rust version and
selects the stable toolchain with rustfmt and Clippy. Install the native SDL2,
Vulkan, and mount-backend development files required by your operating system,
then run:

```sh
cargo build --workspace --release --locked
```

The user-facing executables are written to `target/release/`. Validate the
workspace with:

```sh
cargo test --workspace --locked
```

These commands build the current checked-out source. Ordinary source edits do
not require regenerating or resealing a distribution manifest, and
`cargo build --release` is also the normal optimized build used for direct
testing.

The public macOS source workflow continues with:

```sh
./packaging/macos/package-cli.sh
```

That intentionally public script derives the current project version from
Cargo metadata and packages the current build without fixed project-source,
legal-manifest, RNA-export, or package-version identity constants.

## Beta limitations

- Decoder support is limited to the container and payload encodings implemented
  by this Beta; compatibility with every `.mcraw` recording is not guaranteed.
- The CPU decoder is a parity/manual fallback, not the optimizer-selected
  production default.
- Mount behavior and cleanup details differ across FUSE3, macFUSE, and ProjFS.
- A working platform Vulkan stack is required for the primary decode path.
- Native package installation and first-launch approval behavior depends on the
  operating system and package format.

## macOS signing status

The initial macOS Beta packages use ad-hoc code signing and are not notarized
with an Apple Developer ID. macOS may require explicit user approval before
the first launch.

## Reporting bugs and support

Report reproducible problems through
[GitHub Issues](https://github.com/nate808-gh/mcraw4vulkan/issues). Include the
operating system, command or GUI workflow, and non-sensitive diagnostic output.

## License

Copyright (C) 2026 nate808-gh

mcraw4vulkan is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version (`GPL-3.0-or-later`). See `LICENSE`.

Third-party components retain their upstream licenses. See
`THIRD_PARTY_NOTICES.md` and `licenses/`.
