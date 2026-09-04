use std::io::IsTerminal;
use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow, bail};
use mcraw4vulkan_dngwriter::DngSinkVignetteMode;
use mcraw4vulkan_mcrawcontainer::McrawContainer;

use crate::app_policy::{DngAppBackendPolicy, DngAppPolicy, DngMountPolicy};
use crate::app_runner::run_display_cli;
use crate::display_window::{
    DisplayCliBackend, DisplayCliOverlay, DisplayCliRunConfig, DisplayCliSettings, DisplayCliSound,
    DisplayCliVignette, DisplayCliVsync,
};
use crate::measurement_cli::{
    InternalMeasurementCommand, InternalMeasurementConfig, InternalMeasurementFlagState,
    consume_internal_measurement_flag, finalize_internal_measurement,
    is_internal_measurement_option,
};
use crate::pipe_cli::{
    PipeCliBackend, PipeCliOutput, PipeCliRunConfig, PipeCliVignette, run_pipe_cli,
    write_pipe_metadata_json_for_config,
};

pub struct Cli;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Mcraw4VulkanCommand {
    Help,
    Display(DisplayCommand),
    Dng(DngCommand),
    Pipe(PipeCommand),
    PipeHelp,
    PipeMetadata(PipeMetadataCommand),
    Optimizer(OptimizerCommand),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DisplayCommand {
    pub input: PathBuf,
    pub backend: ComputeBackendChoice,
    pub vignette: VignetteCorrectionChoice,
    pub vsync: VsyncChoice,
    pub settings: SettingsSourceChoice,
    pub overlay: DisplayOverlayChoice,
    pub with_sound: bool,
    pub startup_timing: bool,
    pub(crate) internal_measurement: Option<InternalMeasurementConfig>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DngCommand {
    Mount(DngMountCommand),
    UnmountFile(DngUnmountFileCommand),
    UnmountAll,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DngMountCommand {
    pub input: PathBuf,
    pub additional_inputs: Vec<PathBuf>,
    pub backend: ComputeBackendChoice,
    pub vignette: VignetteCorrectionChoice,
    pub settings: SettingsSourceChoice,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DngUnmountFileCommand {
    pub input: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipeCommand {
    pub input: PathBuf,
    pub backend: ComputeBackendChoice,
    pub vignette: VignetteCorrectionChoice,
    pub settings: SettingsSourceChoice,
    pub output: PipeOutputChoice,
    pub(crate) internal_measurement: Option<InternalMeasurementConfig>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipeMetadataCommand {
    pub input: PathBuf,
    pub backend: ComputeBackendChoice,
    pub vignette: VignetteCorrectionChoice,
    pub settings: SettingsSourceChoice,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptimizerCommand {
    Run { input: PathBuf },
    RestoreDefaults,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeBackendChoice {
    Gpu,
    Cpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VignetteCorrectionChoice {
    NoCorrection,
    WithCorrection,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VsyncChoice {
    Vsync,
    NoVsync,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SettingsSourceChoice {
    Default,
    Optimized,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisplayOverlayChoice {
    WithOverlay,
    NoOverlay,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PipeOutputChoice {
    StdoutIfPiped,
    File(PathBuf),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DngMountNaming {
    pub mounted_folder: PathBuf,
    pub first_dng_path: PathBuf,
    pub audio_path: PathBuf,
}

impl Cli {
    pub fn parse_production_env() -> Result<Mcraw4VulkanCommand> {
        Self::parse_command_from(std::env::args().skip(1))
    }

    pub fn parse_command_from<I, S>(args: I) -> Result<Mcraw4VulkanCommand>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let args: Vec<String> = args.into_iter().map(Into::into).collect();
        let Some((command, rest)) = args.split_first() else {
            bail!("missing command\n\n{}", Self::usage());
        };

        match command.as_str() {
            "-h" | "--help" | "help" => Ok(Mcraw4VulkanCommand::Help),
            "display" if is_single_help_arg(rest) => Ok(Mcraw4VulkanCommand::Help),
            "display" => parse_display(rest).map(Mcraw4VulkanCommand::Display),
            "dng" if is_single_help_arg(rest) => Ok(Mcraw4VulkanCommand::Help),
            "dng" => parse_dng(rest).map(Mcraw4VulkanCommand::Dng),
            "pipe" => parse_pipe_command(rest),
            "optimizer" if is_single_help_arg(rest) => Ok(Mcraw4VulkanCommand::Help),
            "optimizer" => parse_optimizer(rest).map(Mcraw4VulkanCommand::Optimizer),
            other if other.starts_with('-') => {
                bail!("unknown option {other:?}\n\n{}", Self::usage())
            }
            other => bail!("unknown command {other:?}\n\n{}", Self::usage()),
        }
    }

    pub fn usage() -> &'static str {
        "mcraw4vulkan\n\nUsage:\n  mcraw4vulkan display [--gpu|--cpu] [--no-vig-correction|--with-vig-correction] [--vsync|--no-vsync] [--with-sound] [--default|--optimized] [--with-overlay|--no-overlay] FILE.mcraw\n  mcraw4vulkan dng [--gpu|--cpu] [--no-vig-correction|--with-vig-correction] [--default|--optimized] FILE.mcraw\n  mcraw4vulkan dng unmount FILE.mcraw\n  mcraw4vulkan dng unmount all\n  mcraw4vulkan pipe [--gpu|--cpu] [--no-vig-correction|--with-vig-correction] [--default|--optimized] [--output FILE] FILE.mcraw\n  mcraw4vulkan pipe --metadata FILE.mcraw\n  mcraw4vulkan optimizer FILE.mcraw\n  mcraw4vulkan optimizer --restore-defaults\n\nCommands:\n  display     show the 8-bit preview in a window; press F to toggle fullscreen\n  dng         mount or unmount mcraw4vulkan-owned virtual DNG plus audio folders\n  pipe        write direct yuv444p12le rawvideo or print PIPE metadata JSON\n  optimizer   measure and optionally save optimized settings\n\nDefaults:\n  display:   --gpu --no-vig-correction --vsync --default --with-overlay\n  dng:       --gpu --with-vig-correction --default\n  pipe:      --gpu --with-vig-correction --default\n\nPIPE output contract:\n  stdout is planar yuv444p12le: full Y, Cb, then Cr planes; six bytes per pixel.\n  Samples are little-endian 16-bit words with 12 meaningful low bits.\n  Color is TV-range BT.2020 primaries, linear transfer, BT.2020 NCL matrix, no alpha.\n  A fixed 1/2 scene-linear signal scale preserves headroom. The result is not\n  display-ready and may require exposure and a display transform in the editor.\n  Diagnostics, progress, and warnings use stderr only.\n\nDisplay audio:\n  display --with-sound plays synced clip audio and is valid only with Vsync display.\n  display --with-sound cannot be combined with --no-vsync.\n\nDNG mounts:\n  dng FILE.mcraw mounts one virtual DNG and audio folder; it does not export DNGs directly.\n  The command serves one foreground per-file mountpoint under the mcraw4vulkan folder in your home directory.\n  Start one dng command per mounted clip; press Ctrl-C to stop a foreground mount.\n\nDNG unmounts:\n  dng unmount FILE.mcraw  unmount the per-file DNG mount for FILE.mcraw\n  dng unmount all         unmount all mcraw4vulkan-owned DNG mounts\n\nExamples:\n  mcraw4vulkan display --gpu --no-vig-correction --vsync --with-overlay --default FILE.mcraw\n  mcraw4vulkan display --with-sound FILE.mcraw\n  mcraw4vulkan display --cpu --no-vsync --no-overlay FILE.mcraw\n  mcraw4vulkan dng --gpu --with-vig-correction --default FILE.mcraw\n  mcraw4vulkan dng unmount FILE.mcraw\n  mcraw4vulkan dng unmount all\n  mcraw4vulkan pipe --gpu --with-vig-correction --output clip.yuv444p12le FILE.mcraw\n  mcraw4vulkan pipe --gpu --with-vig-correction FILE.mcraw > clip.yuv444p12le\n  mcraw4vulkan pipe --metadata FILE.mcraw\n  mcraw4vulkan optimizer FILE.mcraw\n  mcraw4vulkan optimizer --restore-defaults\n\nPIPE stdout safety:\n  pipe without --output writes raw bytes to stdout only when stdout is not a terminal.\n  Diagnostics, progress, and warnings use stderr."
    }

    pub fn pipe_usage() -> &'static str {
        "mcraw4vulkan pipe\n\nUsage:\n  mcraw4vulkan pipe [--gpu|--cpu] [--no-vig-correction|--with-vig-correction] [--default|--optimized] [--output FILE] FILE.mcraw\n  mcraw4vulkan pipe --metadata FILE.mcraw\n\nOptions:\n  --metadata   print strict PIPE sidecar version 3 JSON without rendering video\n  --output     write raw yuv444p12le video bytes to FILE\n\nOutput contract:\n  stdout is six bytes/pixel: planar Y, Cb, Cr little-endian words with 12 meaningful low bits.\n  The signal is TV-range BT.2020/linear/BT.2020 NCL with no alpha and fixed scale 1/2.\n  It is a scene-linear editing derivative, not a display-ready image, and may look dark\n  until exposure and a display transform are applied downstream.\n\nPIPE stdout safety:\n  pipe without --output writes raw bytes to stdout only when stdout is not a terminal.\n  Diagnostics, progress, and warnings use stderr."
    }
}

pub fn run_from_env_args() -> Result<()> {
    run_production_command(Cli::parse_production_env()?)
}

pub fn run_production_command(command: Mcraw4VulkanCommand) -> Result<()> {
    match command {
        Mcraw4VulkanCommand::Help => {
            println!("{}", Cli::usage());
            Ok(())
        }
        Mcraw4VulkanCommand::Display(command) => {
            if command.internal_measurement.is_some() {
                crate::measurement_cli::run_display_internal_measurement(command)
            } else {
                let effective = effective_settings_for_choice(command.settings);
                warn_if_optimized_state_fallback(&effective);
                run_display_cli(command.into_display_run_config_with_settings(&effective))
            }
        }
        Mcraw4VulkanCommand::Dng(DngCommand::Mount(command)) => {
            let effective = effective_settings_for_choice(command.settings);
            warn_if_optimized_state_fallback(&effective);
            crate::dng_mount::run_dng_mount(
                command.into_dng_mount_run_config_with_settings(&effective)?,
            )
        }
        Mcraw4VulkanCommand::Dng(DngCommand::UnmountFile(command)) => {
            crate::app_runner::run_dng_unmount_file(&command.input)
        }
        Mcraw4VulkanCommand::Dng(DngCommand::UnmountAll) => {
            crate::app_runner::run_dng_unmount_all()
        }
        Mcraw4VulkanCommand::Pipe(command) => {
            if command.internal_measurement.is_some() {
                crate::measurement_cli::run_pipe_internal_measurement(command)
            } else {
                let effective = effective_settings_for_choice(command.settings);
                warn_if_optimized_state_fallback(&effective);
                let output = resolve_pipe_output_for_stdout(
                    &command.output,
                    std::io::stdout().is_terminal(),
                )?;
                run_pipe_cli(command.into_pipe_run_config_with_settings(output, &effective))
            }
        }
        Mcraw4VulkanCommand::PipeHelp => {
            println!("{}", Cli::pipe_usage());
            Ok(())
        }
        Mcraw4VulkanCommand::PipeMetadata(command) => {
            let effective = effective_settings_for_choice(command.settings);
            warn_if_optimized_state_fallback(&effective);
            let config = command.into_pipe_run_config_with_settings(&effective);
            let stdout = std::io::stdout();
            let mut writer = stdout.lock();
            write_pipe_metadata_json_for_config(&config, &mut writer)
        }
        Mcraw4VulkanCommand::Optimizer(command) => run_optimizer_command(command),
    }
}

pub fn run_optimizer_command(command: OptimizerCommand) -> Result<()> {
    if let OptimizerCommand::Run { input } = &command {
        let container = McrawContainer::open_for_display(input)
            .map_err(|error| anyhow!("failed to inspect optimizer input: {error}"))?;
        mcraw4vulkan_optimizer::shellout::validate_optimizer_frame_count(container.frame_count())
            .map_err(anyhow::Error::new)?;
    }

    match command {
        OptimizerCommand::Run { .. } => {
            let config = optimizer_run_config_for_command(command)?;
            mcraw4vulkan_optimizer::run_optimizer(config)
                .map(|_| ())
                .map_err(|error| anyhow!("optimizer shell-out runner failed: {error}"))
        }
        OptimizerCommand::RestoreDefaults => run_optimizer_restore_defaults(),
    }
}

pub fn optimizer_run_config_for_command(
    command: OptimizerCommand,
) -> Result<mcraw4vulkan_optimizer::OptimizerRunConfig> {
    let total_ram_bytes = mcraw4vulkan_preflight::collect_system_ram().installed_bytes;
    optimizer_run_config_for_command_with_total_ram(command, total_ram_bytes)
}

fn optimizer_run_config_for_command_with_total_ram(
    command: OptimizerCommand,
    total_ram_bytes: Option<u64>,
) -> Result<mcraw4vulkan_optimizer::OptimizerRunConfig> {
    let OptimizerCommand::Run { input } = command else {
        bail!("optimizer --restore-defaults does not run measurements");
    };
    let exe = std::env::current_exe()
        .map_err(|error| anyhow!("failed to locate current mcraw4vulkan executable: {error}"))?;
    Ok(mcraw4vulkan_optimizer::OptimizerRunConfig::user_facing(
        input,
        exe,
        mcraw4vulkan_optimizer::OptimizerRunConfig::keep_reports_from_env(),
        total_ram_bytes,
    ))
}

pub fn run_optimizer_restore_defaults() -> Result<()> {
    match mcraw4vulkan_optimizer::restore_optimized_state()
        .map_err(|error| anyhow!("failed to restore built-in defaults: {error}"))?
    {
        mcraw4vulkan_optimizer::RestoreOptimizedStateOutcome::Deleted { path } => {
            println!("removed optimized settings file: {}", path.display());
            println!("built-in defaults are active");
        }
        mcraw4vulkan_optimizer::RestoreOptimizedStateOutcome::Missing { path } => {
            println!("no optimized settings file found at {}", path.display());
            println!("built-in defaults are active");
        }
        mcraw4vulkan_optimizer::RestoreOptimizedStateOutcome::PathUnavailable { error } => {
            println!("optimized settings path is unavailable: {error}");
            println!("built-in defaults are active");
        }
    }
    Ok(())
}

// Raw YUV444P12LE may use stdout only as a byte sink. Refusing terminals keeps
// binary output separate from diagnostics and protects interactive consoles.
fn resolve_pipe_output_for_stdout(
    output: &PipeOutputChoice,
    stdout_is_terminal: bool,
) -> Result<PipeCliOutput> {
    match output {
        PipeOutputChoice::File(path) => Ok(PipeCliOutput::File(path.clone())),
        PipeOutputChoice::StdoutIfPiped if stdout_is_terminal => bail!(
            "refusing to write raw yuv444p12le bytes to a terminal; pass --output FILE or pipe stdout to a file"
        ),
        PipeOutputChoice::StdoutIfPiped => Ok(PipeCliOutput::Stdout),
    }
}

pub fn dng_mount_naming_for_input(input: &Path) -> Result<DngMountNaming> {
    let stem = path_stem(input)?;
    let mounted_folder = PathBuf::from(stem.clone());
    let first_dng_path = mounted_folder.join(format!("{stem}_000000.dng"));
    let audio_path = mounted_folder.join(format!("{stem}.wav"));

    Ok(DngMountNaming {
        mounted_folder,
        first_dng_path,
        audio_path,
    })
}

impl DisplayCommand {
    #[cfg(test)]
    fn into_display_run_config(self) -> DisplayCliRunConfig {
        self.into_display_run_config_with_settings(
            &mcraw4vulkan_optimizer::built_in_default_settings(),
        )
    }

    fn into_display_run_config_with_settings(
        self,
        effective: &mcraw4vulkan_optimizer::EffectiveOptimizerSettings,
    ) -> DisplayCliRunConfig {
        DisplayCliRunConfig {
            input_path: self.input,
            backend: match self.backend {
                ComputeBackendChoice::Gpu => DisplayCliBackend::Gpu,
                ComputeBackendChoice::Cpu => DisplayCliBackend::Cpu,
            },
            vignette: match self.vignette {
                VignetteCorrectionChoice::NoCorrection => DisplayCliVignette::NoCorrection,
                VignetteCorrectionChoice::WithCorrection => DisplayCliVignette::WithCorrection,
            },
            vsync: match self.vsync {
                VsyncChoice::Vsync => DisplayCliVsync::Vsync,
                VsyncChoice::NoVsync => DisplayCliVsync::NoVsync,
            },
            settings: match self.settings {
                SettingsSourceChoice::Default => DisplayCliSettings::Default,
                SettingsSourceChoice::Optimized => DisplayCliSettings::Optimized,
            },
            overlay: match self.overlay {
                DisplayOverlayChoice::WithOverlay => DisplayCliOverlay::WithOverlay,
                DisplayOverlayChoice::NoOverlay => DisplayCliOverlay::NoOverlay,
            },
            sound: if self.with_sound {
                DisplayCliSound::WithSound
            } else {
                DisplayCliSound::Silent
            },
            payload_feeder_options: effective.payload_profile.payload_feeder_options(),
            startup_timing: self.startup_timing,
        }
    }
}

impl DngMountCommand {
    fn input_paths(&self) -> Vec<PathBuf> {
        let mut input_paths = Vec::with_capacity(self.additional_inputs.len() + 1);
        input_paths.push(self.input.clone());
        input_paths.extend(self.additional_inputs.iter().cloned());
        input_paths
    }

    fn into_dng_mount_run_config_with_settings(
        self,
        _effective: &mcraw4vulkan_optimizer::EffectiveOptimizerSettings,
    ) -> Result<crate::dng_mount::DngMountRunConfig> {
        let input_paths = self.input_paths();
        let policy = DngAppPolicy {
            backend: match self.backend {
                ComputeBackendChoice::Gpu => DngAppBackendPolicy::Gpu,
                ComputeBackendChoice::Cpu => DngAppBackendPolicy::Cpu,
            },
            vignette: match self.vignette {
                VignetteCorrectionChoice::NoCorrection => DngSinkVignetteMode::None,
                VignetteCorrectionChoice::WithCorrection => DngSinkVignetteMode::LumaPlane0,
            },
            mount: DngMountPolicy::platform_default(),
        };
        crate::dng_mount::DngMountRunConfig::from_app_policy_inputs(&input_paths, &policy)
    }
}

impl PipeCommand {
    #[cfg(test)]
    fn into_pipe_run_config(self, output: PipeCliOutput) -> PipeCliRunConfig {
        self.into_pipe_run_config_with_settings(
            output,
            &mcraw4vulkan_optimizer::built_in_default_settings(),
        )
    }

    fn into_pipe_run_config_with_settings(
        self,
        output: PipeCliOutput,
        effective: &mcraw4vulkan_optimizer::EffectiveOptimizerSettings,
    ) -> PipeCliRunConfig {
        PipeCliRunConfig {
            input_path: self.input,
            backend: match self.backend {
                ComputeBackendChoice::Gpu => PipeCliBackend::Gpu,
                ComputeBackendChoice::Cpu => PipeCliBackend::Cpu,
            },
            vignette: match self.vignette {
                VignetteCorrectionChoice::NoCorrection => PipeCliVignette::NoCorrection,
                VignetteCorrectionChoice::WithCorrection => PipeCliVignette::WithCorrection,
            },
            output,
            payload_feeder_options: effective.payload_profile.payload_feeder_options(),
        }
    }
}

impl PipeMetadataCommand {
    fn into_pipe_run_config_with_settings(
        self,
        effective: &mcraw4vulkan_optimizer::EffectiveOptimizerSettings,
    ) -> PipeCliRunConfig {
        PipeCliRunConfig {
            input_path: self.input,
            backend: match self.backend {
                ComputeBackendChoice::Gpu => PipeCliBackend::Gpu,
                ComputeBackendChoice::Cpu => PipeCliBackend::Cpu,
            },
            vignette: match self.vignette {
                VignetteCorrectionChoice::NoCorrection => PipeCliVignette::NoCorrection,
                VignetteCorrectionChoice::WithCorrection => PipeCliVignette::WithCorrection,
            },
            output: PipeCliOutput::Stdout,
            payload_feeder_options: effective.payload_profile.payload_feeder_options(),
        }
    }
}

fn parse_display(args: &[String]) -> Result<DisplayCommand> {
    let mut backend = FlagChoice::new(ComputeBackendChoice::Gpu);
    let mut vignette = FlagChoice::new(VignetteCorrectionChoice::NoCorrection);
    let mut vsync = FlagChoice::new(VsyncChoice::Vsync);
    let mut settings = FlagChoice::new(SettingsSourceChoice::Default);
    let mut overlay = FlagChoice::new(DisplayOverlayChoice::WithOverlay);
    let mut internal_measurement = InternalMeasurementFlagState::default();
    let mut startup_timing = false;
    let mut with_sound = false;
    let mut input: Option<PathBuf> = None;

    let mut index = 0usize;
    while index < args.len() {
        if is_internal_measurement_option(args[index].as_str()) {
            index = consume_internal_measurement_flag(args, index, &mut internal_measurement)?;
            index += 1;
            continue;
        }
        match args[index].as_str() {
            "--gpu" => backend.set("--gpu", ComputeBackendChoice::Gpu)?,
            "--cpu" => backend.set("--cpu", ComputeBackendChoice::Cpu)?,
            "--no-vig-correction" => vignette.set(
                "--no-vig-correction",
                VignetteCorrectionChoice::NoCorrection,
            )?,
            "--with-vig-correction" => vignette.set(
                "--with-vig-correction",
                VignetteCorrectionChoice::WithCorrection,
            )?,
            "--vsync" => vsync.set("--vsync", VsyncChoice::Vsync)?,
            "--no-vsync" => vsync.set("--no-vsync", VsyncChoice::NoVsync)?,
            "--with-sound" => {
                if with_sound {
                    bail!("--with-sound was supplied more than once");
                }
                with_sound = true;
            }
            "--default" => settings.set("--default", SettingsSourceChoice::Default)?,
            "--optimized" => settings.set("--optimized", SettingsSourceChoice::Optimized)?,
            "--with-overlay" => overlay.set("--with-overlay", DisplayOverlayChoice::WithOverlay)?,
            "--no-overlay" => overlay.set("--no-overlay", DisplayOverlayChoice::NoOverlay)?,
            "--internal-startup-timing" => {
                if startup_timing {
                    bail!("--internal-startup-timing was supplied more than once");
                }
                startup_timing = true;
            }
            "--output" => bail!("--output is only valid with pipe"),
            value if value.starts_with("--output=") => {
                let _ = value;
                bail!("--output is only valid with pipe")
            }
            value if value.starts_with('-') => {
                bail!("unknown display option {value:?}")
            }
            value => set_input(&mut input, value, "display")?,
        }
        index += 1;
    }
    let internal_measurement =
        finalize_internal_measurement(InternalMeasurementCommand::Display, internal_measurement)?;
    if internal_measurement.is_some() && overlay.flag.is_some() {
        bail!(
            "display --internal-measure uses a no-window runner; overlay flags are not applicable"
        );
    }
    if internal_measurement.is_some() && startup_timing {
        bail!("display --internal-startup-timing is not supported with --internal-measure");
    }
    if internal_measurement.is_some() && with_sound {
        bail!("display --with-sound is not supported with --internal-measure");
    }
    if with_sound && vsync.value == VsyncChoice::NoVsync {
        bail!("--with-sound requires Vsync display and cannot be used with --no-vsync");
    }

    Ok(DisplayCommand {
        input: required_input(input, "display")?,
        backend: backend.value,
        vignette: vignette.value,
        vsync: vsync.value,
        settings: settings.value,
        overlay: overlay.value,
        with_sound,
        startup_timing,
        internal_measurement,
    })
}

fn is_single_help_arg(args: &[String]) -> bool {
    matches!(args, [arg] if matches!(arg.as_str(), "-h" | "--help" | "help"))
}

fn parse_dng(args: &[String]) -> Result<DngCommand> {
    if args.first().is_some_and(|value| value == "unmount") {
        return parse_dng_unmount(&args[1..]);
    }

    let mut backend = FlagChoice::new(ComputeBackendChoice::Gpu);
    let mut vignette = FlagChoice::new(VignetteCorrectionChoice::WithCorrection);
    let mut settings = FlagChoice::new(SettingsSourceChoice::Default);
    let mut input_paths = Vec::<PathBuf>::new();

    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "export" | "mount" => {
                bail!("dng has no export or mount subcommand; use: mcraw4vulkan dng FILE.mcraw")
            }
            "--gpu" => backend.set("--gpu", ComputeBackendChoice::Gpu)?,
            "--cpu" => backend.set("--cpu", ComputeBackendChoice::Cpu)?,
            "--no-vig-correction" => vignette.set(
                "--no-vig-correction",
                VignetteCorrectionChoice::NoCorrection,
            )?,
            "--with-vig-correction" => vignette.set(
                "--with-vig-correction",
                VignetteCorrectionChoice::WithCorrection,
            )?,
            "--default" => settings.set("--default", SettingsSourceChoice::Default)?,
            "--optimized" => settings.set("--optimized", SettingsSourceChoice::Optimized)?,
            "--vsync" | "--no-vsync" | "--with-sound" => {
                bail!("{} is only valid with display", args[index])
            }
            "--with-overlay" | "--no-overlay" => {
                bail!("{} is only valid with display", args[index])
            }
            "--output" => bail!("--output is only valid with pipe"),
            value if value.starts_with("--output=") => {
                let _ = value;
                bail!("--output is only valid with pipe")
            }
            value if value.starts_with('-') => {
                bail!("unknown dng option {value:?}")
            }
            value => input_paths.push(PathBuf::from(value)),
        }
        index += 1;
    }
    if input_paths.is_empty() {
        bail!("dng requires FILE.mcraw")
    }
    if input_paths.len() > 1 {
        bail!("dng currently accepts one FILE per mount process; start one process per file");
    }
    let input = input_paths.remove(0);

    Ok(DngCommand::Mount(DngMountCommand {
        input,
        additional_inputs: input_paths,
        backend: backend.value,
        vignette: vignette.value,
        settings: settings.value,
    }))
}

fn parse_dng_unmount(args: &[String]) -> Result<DngCommand> {
    for value in args {
        if value.starts_with('-') {
            bail!(
                "dng unmount does not accept mount flags; use: mcraw4vulkan dng unmount FILE.mcraw"
            );
        }
    }

    let [target] = args else {
        if args.is_empty() {
            bail!("dng unmount requires FILE.mcraw or all");
        }
        bail!("dng unmount accepts exactly one target: FILE.mcraw or all");
    };

    if target == "all" {
        Ok(DngCommand::UnmountAll)
    } else {
        Ok(DngCommand::UnmountFile(DngUnmountFileCommand {
            input: PathBuf::from(target),
        }))
    }
}

fn parse_pipe_command(args: &[String]) -> Result<Mcraw4VulkanCommand> {
    if args.len() == 1 && matches!(args[0].as_str(), "-h" | "--help" | "help") {
        return Ok(Mcraw4VulkanCommand::PipeHelp);
    }

    if args.iter().any(|arg| arg == "--metadata") {
        return parse_pipe_metadata(args).map(Mcraw4VulkanCommand::PipeMetadata);
    }

    parse_pipe(args).map(Mcraw4VulkanCommand::Pipe)
}

fn parse_pipe_metadata(args: &[String]) -> Result<PipeMetadataCommand> {
    let mut metadata_seen = false;
    let mut backend = FlagChoice::new(ComputeBackendChoice::Gpu);
    let mut vignette = FlagChoice::new(VignetteCorrectionChoice::WithCorrection);
    let mut settings = FlagChoice::new(SettingsSourceChoice::Default);
    let mut input: Option<PathBuf> = None;
    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "--metadata" => {
                if metadata_seen {
                    bail!("--metadata was supplied more than once");
                }
                metadata_seen = true;
                index += 1;
                let path = next_value(args, index, "--metadata")?;
                set_input(&mut input, path, "pipe --metadata")?;
            }
            "-h" | "--help" | "help" => {
                bail!("pipe --help accepts no other arguments")
            }
            "--gpu" => backend.set("--gpu", ComputeBackendChoice::Gpu)?,
            "--cpu" => backend.set("--cpu", ComputeBackendChoice::Cpu)?,
            "--no-vig-correction" => vignette.set(
                "--no-vig-correction",
                VignetteCorrectionChoice::NoCorrection,
            )?,
            "--with-vig-correction" => vignette.set(
                "--with-vig-correction",
                VignetteCorrectionChoice::WithCorrection,
            )?,
            "--default" => settings.set("--default", SettingsSourceChoice::Default)?,
            "--optimized" => settings.set("--optimized", SettingsSourceChoice::Optimized)?,
            "--output" | "--vsync" | "--no-vsync" | "--with-sound" | "--with-overlay"
            | "--no-overlay" => {
                bail!("{} cannot be combined with pipe --metadata", args[index])
            }
            value if value.starts_with("--output=") => {
                let _ = value;
                bail!("--output cannot be combined with pipe --metadata")
            }
            value if value.starts_with('-') => {
                bail!("unknown pipe metadata option {value:?}")
            }
            value => set_input(&mut input, value, "pipe --metadata")?,
        }
        index += 1;
    }

    if !metadata_seen {
        bail!("pipe metadata mode requires --metadata FILE.mcraw");
    }

    Ok(PipeMetadataCommand {
        input: required_input(input, "pipe --metadata")?,
        backend: backend.value,
        vignette: vignette.value,
        settings: settings.value,
    })
}

fn parse_pipe(args: &[String]) -> Result<PipeCommand> {
    let mut backend = FlagChoice::new(ComputeBackendChoice::Gpu);
    let mut vignette = FlagChoice::new(VignetteCorrectionChoice::WithCorrection);
    let mut settings = FlagChoice::new(SettingsSourceChoice::Default);
    let mut output = PipeOutputChoice::StdoutIfPiped;
    let mut output_flag_seen = false;
    let mut internal_measurement = InternalMeasurementFlagState::default();
    let mut input: Option<PathBuf> = None;

    let mut index = 0usize;
    while index < args.len() {
        if is_internal_measurement_option(args[index].as_str()) {
            index = consume_internal_measurement_flag(args, index, &mut internal_measurement)?;
            index += 1;
            continue;
        }
        match args[index].as_str() {
            "--gpu" => backend.set("--gpu", ComputeBackendChoice::Gpu)?,
            "--cpu" => backend.set("--cpu", ComputeBackendChoice::Cpu)?,
            "--no-vig-correction" => vignette.set(
                "--no-vig-correction",
                VignetteCorrectionChoice::NoCorrection,
            )?,
            "--with-vig-correction" => vignette.set(
                "--with-vig-correction",
                VignetteCorrectionChoice::WithCorrection,
            )?,
            "--default" => settings.set("--default", SettingsSourceChoice::Default)?,
            "--optimized" => settings.set("--optimized", SettingsSourceChoice::Optimized)?,
            "--vsync" | "--no-vsync" | "--with-sound" => {
                bail!("{} is only valid with display", args[index])
            }
            "--with-overlay" | "--no-overlay" => {
                bail!("{} is only valid with display", args[index])
            }
            "--output" => {
                if output_flag_seen {
                    bail!("--output was supplied more than once");
                }
                output_flag_seen = true;
                index += 1;
                let path = next_value(args, index, "--output")?;
                output = PipeOutputChoice::File(PathBuf::from(path));
            }
            value if value.starts_with("--output=") => {
                if output_flag_seen {
                    bail!("--output was supplied more than once");
                }
                output_flag_seen = true;
                let path = value.trim_start_matches("--output=");
                if path.is_empty() {
                    bail!("--output requires a path");
                }
                output = PipeOutputChoice::File(PathBuf::from(path));
            }
            value if value.starts_with('-') => {
                bail!("unknown pipe option {value:?}")
            }
            value => set_input(&mut input, value, "pipe")?,
        }
        index += 1;
    }
    let internal_measurement =
        finalize_internal_measurement(InternalMeasurementCommand::Pipe, internal_measurement)?;
    if internal_measurement.is_some() && output_flag_seen {
        bail!(
            "pipe --internal-measure uses internal discard/count output and does not accept --output"
        );
    }

    let input = required_input(input, "pipe")?;
    if let PipeOutputChoice::File(path) = &output {
        if path == &input {
            bail!("pipe --output path must not equal the input path");
        }
    }

    Ok(PipeCommand {
        input,
        backend: backend.value,
        vignette: vignette.value,
        settings: settings.value,
        output,
        internal_measurement,
    })
}

fn parse_optimizer(args: &[String]) -> Result<OptimizerCommand> {
    if args
        .first()
        .is_some_and(|value| value == "--restore-defaults")
    {
        if args.len() == 1 {
            return Ok(OptimizerCommand::RestoreDefaults);
        }
        bail!("optimizer --restore-defaults accepts no FILE.mcraw or extra options");
    }

    let mut input: Option<PathBuf> = None;

    for arg in args {
        match arg.as_str() {
            "--with-overlay" | "--no-overlay" | "--with-sound" => {
                bail!("{arg} is only valid with display")
            }
            value if value.starts_with('-') => {
                bail!("optimizer accepts only FILE.mcraw; unknown option {value:?}")
            }
            value => set_input(&mut input, value, "optimizer")?,
        }
    }

    Ok(OptimizerCommand::Run {
        input: required_input(input, "optimizer")?,
    })
}

#[derive(Debug, Clone, Copy)]
struct FlagChoice<T> {
    value: T,
    flag: Option<&'static str>,
}

impl<T> FlagChoice<T>
where
    T: Copy + PartialEq,
{
    fn new(default: T) -> Self {
        Self {
            value: default,
            flag: None,
        }
    }

    fn set(&mut self, flag: &'static str, value: T) -> Result<()> {
        if let Some(existing) = self.flag {
            if self.value == value {
                bail!("{flag} was supplied more than once");
            }
            bail!("{flag} conflicts with {existing}");
        }

        self.value = value;
        self.flag = Some(flag);
        Ok(())
    }
}

fn set_input(target: &mut Option<PathBuf>, value: &str, command: &str) -> Result<()> {
    if target.is_some() {
        bail!("{command} accepts exactly one input file; extra positional argument {value:?}");
    }
    *target = Some(PathBuf::from(value));
    Ok(())
}

fn required_input(input: Option<PathBuf>, command: &str) -> Result<PathBuf> {
    input.ok_or_else(|| anyhow!("{command} requires FILE.mcraw"))
}

fn next_value<'a>(args: &'a [String], index: usize, flag: &str) -> Result<&'a str> {
    let Some(value) = args.get(index) else {
        bail!("{flag} requires a path")
    };
    if value.starts_with('-') {
        bail!("{flag} requires a path, got option {value:?}")
    }
    Ok(value)
}

fn path_stem(path: &Path) -> Result<String> {
    let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
        bail!("path {:?} does not have a valid UTF-8 file stem", path);
    };
    if stem.is_empty() {
        bail!("path {:?} does not have a non-empty file stem", path);
    }
    Ok(stem.to_string())
}

fn effective_settings_for_choice(
    settings: SettingsSourceChoice,
) -> mcraw4vulkan_optimizer::EffectiveOptimizerSettings {
    match settings {
        SettingsSourceChoice::Default => mcraw4vulkan_optimizer::resolve_effective_settings(
            mcraw4vulkan_optimizer::SettingsSourceSelection::Default,
        ),
        SettingsSourceChoice::Optimized => mcraw4vulkan_optimizer::resolve_effective_settings(
            mcraw4vulkan_optimizer::SettingsSourceSelection::Optimized,
        ),
    }
}

fn warn_if_optimized_state_fallback(
    effective: &mcraw4vulkan_optimizer::EffectiveOptimizerSettings,
) {
    if let Some(warning) = effective.warning() {
        eprintln!("warning: {warning}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_ok(args: &[&str]) -> Mcraw4VulkanCommand {
        Cli::parse_command_from(args.iter().copied()).expect("CLI parses")
    }

    fn parse_err(args: &[&str]) -> String {
        Cli::parse_command_from(args.iter().copied())
            .expect_err("CLI rejects")
            .to_string()
    }

    fn optimized_effective_settings() -> mcraw4vulkan_optimizer::EffectiveOptimizerSettings {
        mcraw4vulkan_optimizer::resolve_effective_settings_from_load(
            mcraw4vulkan_optimizer::OptimizedStateLoadOutcome::Loaded {
                path: PathBuf::from("optimized-state.json"),
                state: mcraw4vulkan_optimizer::OptimizedState::new(
                    mcraw4vulkan_optimizer::PayloadProfile::OffsetPrefetch,
                ),
            },
        )
    }

    #[test]
    fn display_defaults_parse() {
        let command = parse_ok(&["display", "FILE.mcraw"]);
        let Mcraw4VulkanCommand::Display(display) = command else {
            panic!("expected display command");
        };

        assert_eq!(display.input, PathBuf::from("FILE.mcraw"));
        assert_eq!(display.backend, ComputeBackendChoice::Gpu);
        assert_eq!(display.vignette, VignetteCorrectionChoice::NoCorrection);
        assert_eq!(display.vsync, VsyncChoice::Vsync);
        assert_eq!(display.settings, SettingsSourceChoice::Default);
        assert_eq!(display.overlay, DisplayOverlayChoice::WithOverlay);
        assert!(!display.startup_timing);
        assert_eq!(display.internal_measurement, None);
    }

    #[test]
    fn display_help_routes_to_usage() {
        assert_eq!(parse_ok(&["display", "--help"]), Mcraw4VulkanCommand::Help);
        assert!(Cli::usage().contains("display --with-sound"));
    }

    #[test]
    fn display_accepts_hidden_startup_timing_flag() {
        let command = parse_ok(&["display", "--internal-startup-timing", "FILE.mcraw"]);
        let Mcraw4VulkanCommand::Display(display) = command else {
            panic!("expected display command");
        };

        assert!(display.startup_timing);
        let config = display.into_display_run_config();
        assert!(config.startup_timing);
    }

    #[test]
    fn display_accepts_hidden_measurement_flags() {
        let command = parse_ok(&[
            "display",
            "--gpu",
            "--no-vig-correction",
            "--no-vsync",
            "--default",
            "--internal-measure",
            "--internal-measure-report",
            "display.json",
            "--internal-measure-frames",
            "600",
            "--internal-measure-warmup-frames",
            "30",
            "--internal-payload-profile",
            "default_chunked64",
            "FILE.mcraw",
        ]);
        let Mcraw4VulkanCommand::Display(display) = command else {
            panic!("expected display command");
        };
        let config = display.internal_measurement.expect("internal config");

        assert_eq!(config.report_path, PathBuf::from("display.json"));
        assert_eq!(config.frames, 600);
        assert_eq!(config.warmup_frames, 30);
        assert_eq!(
            config.payload_profile,
            Some(crate::measurement_cli::InternalPayloadProfile::DefaultChunked64)
        );
    }

    #[test]
    fn pipe_accepts_hidden_measurement_flags() {
        let command = parse_ok(&[
            "pipe",
            "--gpu",
            "--with-vig-correction",
            "--default",
            "--internal-measure",
            "--internal-measure-report",
            "pipe.json",
            "--internal-measure-frames",
            "600",
            "--internal-payload-profile",
            "default_chunked64",
            "FILE.mcraw",
        ]);
        let Mcraw4VulkanCommand::Pipe(pipe) = command else {
            panic!("expected pipe command");
        };
        let config = pipe.internal_measurement.expect("internal config");

        assert_eq!(config.report_path, PathBuf::from("pipe.json"));
        assert_eq!(config.frames, 600);
        assert_eq!(
            config.payload_profile,
            Some(crate::measurement_cli::InternalPayloadProfile::DefaultChunked64)
        );
    }

    #[test]
    fn internal_measurement_frames_accept_smoke_and_benchmark_counts() {
        for frames in ["1", "600", "1200"] {
            let command = parse_ok(&[
                "display",
                "--internal-measure",
                "--internal-measure-report",
                "report.json",
                "--internal-measure-frames",
                frames,
                "FILE.mcraw",
            ]);
            let Mcraw4VulkanCommand::Display(display) = command else {
                panic!("expected display command");
            };
            assert_eq!(
                display
                    .internal_measurement
                    .expect("internal config")
                    .frames,
                frames.parse::<usize>().unwrap()
            );
        }
    }

    #[test]
    fn omitted_internal_measurement_frames_default_to_600() {
        let command = parse_ok(&[
            "display",
            "--internal-measure",
            "--internal-measure-report",
            "report.json",
            "FILE.mcraw",
        ]);
        let Mcraw4VulkanCommand::Display(display) = command else {
            panic!("expected display command");
        };

        assert_eq!(
            display
                .internal_measurement
                .expect("internal config")
                .frames,
            crate::measurement_cli::INTERNAL_MEASUREMENT_DEFAULT_FRAMES
        );
    }

    #[test]
    fn zero_internal_measurement_frames_reject() {
        assert!(
            parse_err(&[
                "display",
                "--internal-measure",
                "--internal-measure-report",
                "report.json",
                "--internal-measure-frames",
                "0",
                "FILE.mcraw",
            ])
            .contains("must be positive")
        );
    }

    #[test]
    fn hidden_measurement_requires_report_path() {
        assert!(
            parse_err(&["display", "--internal-measure", "FILE.mcraw"])
                .contains("requires --internal-measure-report")
        );
    }

    #[test]
    fn report_path_requires_hidden_measurement() {
        assert!(
            parse_err(&[
                "display",
                "--internal-measure-report",
                "report.json",
                "FILE.mcraw",
            ])
            .contains("require --internal-measure")
        );
    }

    #[test]
    fn warmup_frames_may_be_zero() {
        let command = parse_ok(&[
            "display",
            "--internal-measure",
            "--internal-measure-report",
            "report.json",
            "--internal-measure-warmup-frames",
            "0",
            "FILE.mcraw",
        ]);
        let Mcraw4VulkanCommand::Display(display) = command else {
            panic!("expected display command");
        };

        assert_eq!(
            display
                .internal_measurement
                .expect("internal config")
                .warmup_frames,
            0
        );
    }

    #[test]
    fn invalid_internal_profiles_reject() {
        assert!(
            parse_err(&[
                "display",
                "--internal-measure",
                "--internal-measure-report",
                "report.json",
                "--internal-payload-profile",
                "bad",
                "FILE.mcraw",
            ])
            .contains("invalid --internal-payload-profile")
        );
    }

    #[test]
    fn rejected_commands_do_not_accept_hidden_measurement_flags() {
        assert!(
            parse_err(&["dng", "--internal-measure", "FILE.mcraw"]).contains("unknown dng option")
        );
        assert!(
            parse_err(&["optimizer", "--internal-measure", "FILE.mcraw"])
                .contains("unknown option")
        );
        assert!(
            parse_err(&["dng", "unmount", "--internal-measure", "FILE.mcraw"])
                .contains("does not accept mount flags")
        );
        assert!(
            parse_err(&["dng", "unmount", "all", "--internal-measure"])
                .contains("does not accept mount flags")
        );
    }

    #[test]
    fn pipe_internal_measurement_rejects_output() {
        assert!(
            parse_err(&[
                "pipe",
                "--internal-measure",
                "--internal-measure-report",
                "report.json",
                "--output",
                "out.yuv444p12le",
                "FILE.mcraw",
            ])
            .contains("does not accept --output")
        );
    }

    #[test]
    fn display_internal_measurement_rejects_explicit_overlay_flags() {
        for overlay in ["--with-overlay", "--no-overlay"] {
            assert!(
                parse_err(&[
                    "display",
                    "--internal-measure",
                    "--internal-measure-report",
                    "report.json",
                    overlay,
                    "FILE.mcraw",
                ])
                .contains("overlay flags are not applicable")
            );
        }
    }

    #[test]
    fn display_with_overlay_parses() {
        let command = parse_ok(&["display", "--with-overlay", "FILE.mcraw"]);
        let Mcraw4VulkanCommand::Display(display) = command else {
            panic!("expected display command");
        };

        assert_eq!(display.overlay, DisplayOverlayChoice::WithOverlay);
    }

    #[test]
    fn display_no_overlay_parses() {
        let command = parse_ok(&["display", "--no-overlay", "FILE.mcraw"]);
        let Mcraw4VulkanCommand::Display(display) = command else {
            panic!("expected display command");
        };

        assert_eq!(display.overlay, DisplayOverlayChoice::NoOverlay);
    }

    #[test]
    fn display_rejects_overlay_conflict() {
        assert!(
            parse_err(&["display", "--with-overlay", "--no-overlay", "FILE.mcraw"])
                .contains("conflicts")
        );
    }

    #[test]
    fn display_rejects_backend_conflict() {
        assert!(parse_err(&["display", "--gpu", "--cpu", "FILE.mcraw"]).contains("conflicts"));
    }

    #[test]
    fn display_rejects_vignette_conflict() {
        assert!(
            parse_err(&[
                "display",
                "--no-vig-correction",
                "--with-vig-correction",
                "FILE.mcraw"
            ])
            .contains("conflicts")
        );
    }

    #[test]
    fn display_rejects_vsync_conflict() {
        assert!(
            parse_err(&["display", "--vsync", "--no-vsync", "FILE.mcraw"]).contains("conflicts")
        );
    }

    #[test]
    fn display_with_sound_parses_for_default_vsync() {
        let command = parse_ok(&["display", "--with-sound", "FILE.mcraw"]);
        let Mcraw4VulkanCommand::Display(display) = command else {
            panic!("expected display command");
        };

        assert!(display.with_sound);
        assert_eq!(display.vsync, VsyncChoice::Vsync);
        let config = display.into_display_run_config();
        assert_eq!(config.sound, DisplayCliSound::WithSound);
        assert_eq!(config.vsync, DisplayCliVsync::Vsync);
    }

    #[test]
    fn display_without_sound_uses_silent_config() {
        let command = parse_ok(&["display", "FILE.mcraw"]);
        let Mcraw4VulkanCommand::Display(display) = command else {
            panic!("expected display command");
        };

        assert!(!display.with_sound);
        assert_eq!(
            display.into_display_run_config().sound,
            DisplayCliSound::Silent
        );
    }

    #[test]
    fn display_no_vsync_with_sound_rejects_regardless_of_order() {
        for args in [
            ["display", "--no-vsync", "--with-sound", "FILE.mcraw"],
            ["display", "--with-sound", "--no-vsync", "FILE.mcraw"],
        ] {
            let error = parse_err(&args);
            assert!(
                error.contains("--with-sound requires Vsync display"),
                "{error}"
            );
            assert!(error.contains("--no-vsync"), "{error}");
        }
    }

    #[test]
    fn display_rejects_duplicate_with_sound() {
        assert!(
            parse_err(&["display", "--with-sound", "--with-sound", "FILE.mcraw"])
                .contains("supplied more than once")
        );
    }

    #[test]
    fn display_optimized_parses() {
        let command = parse_ok(&["display", "--optimized", "FILE.mcraw"]);
        let Mcraw4VulkanCommand::Display(display) = command else {
            panic!("expected display command");
        };
        assert_eq!(display.settings, SettingsSourceChoice::Optimized);
    }

    #[test]
    fn display_command_maps_gpu_to_run_config() {
        let command = parse_ok(&["display", "--gpu", "FILE.mcraw"]);
        let Mcraw4VulkanCommand::Display(display) = command else {
            panic!("expected display command");
        };
        let config = display.into_display_run_config();
        assert_eq!(config.backend, DisplayCliBackend::Gpu);
    }

    #[test]
    fn display_command_maps_cpu_to_run_config() {
        let command = parse_ok(&["display", "--cpu", "FILE.mcraw"]);
        let Mcraw4VulkanCommand::Display(display) = command else {
            panic!("expected display command");
        };
        let config = display.into_display_run_config();
        assert_eq!(config.backend, DisplayCliBackend::Cpu);
    }

    #[test]
    fn display_command_maps_vignette_and_vsync_to_run_config() {
        let command = parse_ok(&[
            "display",
            "--with-vig-correction",
            "--no-vsync",
            "FILE.mcraw",
        ]);
        let Mcraw4VulkanCommand::Display(display) = command else {
            panic!("expected display command");
        };
        let config = display.into_display_run_config();
        assert_eq!(config.vignette, DisplayCliVignette::WithCorrection);
        assert_eq!(config.vsync, DisplayCliVsync::NoVsync);
        assert_eq!(config.overlay, DisplayCliOverlay::WithOverlay);
    }

    #[test]
    fn display_command_maps_no_overlay_to_run_config() {
        let command = parse_ok(&["display", "--no-overlay", "FILE.mcraw"]);
        let Mcraw4VulkanCommand::Display(display) = command else {
            panic!("expected display command");
        };
        let config = display.into_display_run_config();

        assert_eq!(config.overlay, DisplayCliOverlay::NoOverlay);
    }

    #[test]
    fn optimized_effective_settings_apply_display_payload_profile() {
        let command = DisplayCommand {
            input: PathBuf::from("missing.mcraw"),
            backend: ComputeBackendChoice::Gpu,
            vignette: VignetteCorrectionChoice::NoCorrection,
            vsync: VsyncChoice::Vsync,
            settings: SettingsSourceChoice::Optimized,
            overlay: DisplayOverlayChoice::WithOverlay,
            with_sound: false,
            startup_timing: false,
            internal_measurement: None,
        };
        let effective = optimized_effective_settings();
        let config = command.into_display_run_config_with_settings(&effective);

        assert_eq!(
            config.payload_feeder_options,
            mcraw4vulkan_optimizer::PayloadProfile::OffsetPrefetch.payload_feeder_options()
        );
    }

    #[test]
    fn display_rejects_output() {
        assert!(parse_err(&["display", "--output", "out", "FILE.mcraw"]).contains("pipe"));
    }

    #[test]
    fn dng_defaults_parse() {
        let command = parse_ok(&["dng", "FILE.mcraw"]);
        let Mcraw4VulkanCommand::Dng(DngCommand::Mount(dng)) = command else {
            panic!("expected dng mount command");
        };

        assert_eq!(dng.input, PathBuf::from("FILE.mcraw"));
        assert!(dng.additional_inputs.is_empty());
        assert_eq!(dng.backend, ComputeBackendChoice::Gpu);
        assert_eq!(dng.vignette, VignetteCorrectionChoice::WithCorrection);
        assert_eq!(dng.settings, SettingsSourceChoice::Default);
    }

    #[test]
    fn dng_help_routes_to_usage() {
        assert_eq!(parse_ok(&["dng", "--help"]), Mcraw4VulkanCommand::Help);
    }

    #[test]
    fn dng_multiple_inputs_reject() {
        let error = parse_err(&["dng", "FILE1.mcraw", "FILE2.mcraw", "FILE3.mcraw"]);

        assert!(error.contains("one FILE per mount process"));
        assert!(error.contains("start one process per file"));
    }

    #[test]
    fn dng_unmount_file_parses() {
        let command = parse_ok(&["dng", "unmount", "FILE.mcraw"]);
        let Mcraw4VulkanCommand::Dng(DngCommand::UnmountFile(unmount)) = command else {
            panic!("expected dng unmount file command");
        };

        assert_eq!(unmount.input, PathBuf::from("FILE.mcraw"));
    }

    #[test]
    fn dng_unmount_all_parses() {
        let command = parse_ok(&["dng", "unmount", "all"]);
        let Mcraw4VulkanCommand::Dng(DngCommand::UnmountAll) = command else {
            panic!("expected dng unmount all command");
        };
    }

    #[test]
    fn dng_mount_treats_all_mcraw_as_file_path() {
        let command = parse_ok(&["dng", "all.mcraw"]);
        let Mcraw4VulkanCommand::Dng(DngCommand::Mount(dng)) = command else {
            panic!("expected dng mount command");
        };

        assert_eq!(dng.input, PathBuf::from("all.mcraw"));
        assert!(dng.additional_inputs.is_empty());
    }

    #[test]
    fn dng_unmount_missing_target_rejects() {
        assert!(parse_err(&["dng", "unmount"]).contains("requires FILE.mcraw or all"));
    }

    #[test]
    fn dng_unmount_all_extra_rejects() {
        assert!(parse_err(&["dng", "unmount", "all", "EXTRA"]).contains("exactly one target"));
    }

    #[test]
    fn dng_unmount_file_extra_rejects() {
        assert!(
            parse_err(&["dng", "unmount", "FILE.mcraw", "EXTRA"]).contains("exactly one target")
        );
    }

    #[test]
    fn dng_unmount_rejects_mount_flags() {
        assert!(
            parse_err(&["dng", "unmount", "--gpu", "FILE.mcraw"])
                .contains("does not accept mount flags")
        );
    }

    #[test]
    fn dng_export_rejects() {
        assert!(parse_err(&["dng", "export", "FILE.mcraw"]).contains("no export"));
    }

    #[test]
    fn dng_mount_rejects() {
        assert!(parse_err(&["dng", "mount", "FILE.mcraw"]).contains("no export"));
    }

    #[test]
    fn dng_no_vignette_parses() {
        let command = parse_ok(&["dng", "--no-vig-correction", "FILE.mcraw"]);
        let Mcraw4VulkanCommand::Dng(DngCommand::Mount(dng)) = command else {
            panic!("expected dng mount command");
        };
        assert_eq!(dng.vignette, VignetteCorrectionChoice::NoCorrection);
    }

    #[test]
    fn optimized_settings_do_not_override_production_dng_mount_cache_policy() {
        let effective = optimized_effective_settings();
        let input = std::env::current_exe().expect("test executable path");
        let command = DngMountCommand {
            input,
            additional_inputs: Vec::new(),
            backend: ComputeBackendChoice::Gpu,
            vignette: VignetteCorrectionChoice::WithCorrection,
            settings: SettingsSourceChoice::Optimized,
        };
        let config = command
            .into_dng_mount_run_config_with_settings(&effective)
            .expect("DNG config");

        assert_eq!(
            config.cache_frame_capacity,
            mcraw4vulkan_fuse::DEFAULT_DNG_CACHE_FRAME_CAPACITY
        );
        assert_eq!(
            config.prefetch_forward_frames,
            mcraw4vulkan_fuse::DEFAULT_PREFETCH_FORWARD_FRAMES
        );
    }

    #[test]
    fn dng_mount_rejects_conflicting_flags() {
        assert!(parse_err(&["dng", "--gpu", "--cpu", "FILE.mcraw"]).contains("conflicts"));
        assert!(
            parse_err(&[
                "dng",
                "--no-vig-correction",
                "--with-vig-correction",
                "FILE.mcraw"
            ])
            .contains("conflicts")
        );
    }

    #[test]
    fn dng_rejects_vsync() {
        assert!(parse_err(&["dng", "--vsync", "FILE.mcraw"]).contains("display"));
    }

    #[test]
    fn dng_rejects_overlay_flags() {
        assert!(parse_err(&["dng", "--no-overlay", "FILE.mcraw"]).contains("display"));
    }

    #[test]
    fn pipe_defaults_to_stdout_mode() {
        let command = parse_ok(&["pipe", "FILE.mcraw"]);
        let Mcraw4VulkanCommand::Pipe(pipe) = command else {
            panic!("expected pipe command");
        };

        assert_eq!(pipe.input, PathBuf::from("FILE.mcraw"));
        assert_eq!(pipe.backend, ComputeBackendChoice::Gpu);
        assert_eq!(pipe.vignette, VignetteCorrectionChoice::WithCorrection);
        assert_eq!(pipe.settings, SettingsSourceChoice::Default);
        assert_eq!(pipe.output, PipeOutputChoice::StdoutIfPiped);
    }

    #[test]
    fn pipe_output_helper_rejects_terminal_stdout() {
        let error = resolve_pipe_output_for_stdout(&PipeOutputChoice::StdoutIfPiped, true)
            .expect_err("terminal stdout rejected");
        assert!(error.to_string().contains("terminal"));
    }

    #[test]
    fn pipe_output_helper_allows_piped_stdout() {
        assert_eq!(
            resolve_pipe_output_for_stdout(&PipeOutputChoice::StdoutIfPiped, false)
                .expect("piped stdout allowed"),
            PipeCliOutput::Stdout
        );
    }

    #[test]
    fn pipe_output_file_parses() {
        let command = parse_ok(&["pipe", "--output", "clip.yuv444p12le", "FILE.mcraw"]);
        let Mcraw4VulkanCommand::Pipe(pipe) = command else {
            panic!("expected pipe command");
        };
        assert_eq!(
            pipe.output,
            PipeOutputChoice::File(PathBuf::from("clip.yuv444p12le"))
        );
    }

    #[test]
    fn pipe_rejects_ouput_typo() {
        assert!(
            parse_err(&["pipe", "--ouput", "clip.yuv444p12le", "FILE.mcraw"])
                .contains("unknown pipe option")
        );
    }

    #[test]
    fn pipe_command_maps_defaults_to_production_run_config() {
        let command = parse_ok(&["pipe", "--output", "clip.yuv444p12le", "FILE.mcraw"]);
        let Mcraw4VulkanCommand::Pipe(pipe) = command else {
            panic!("expected pipe command");
        };
        let config =
            pipe.into_pipe_run_config(PipeCliOutput::File(PathBuf::from("clip.yuv444p12le")));

        assert_eq!(config.input_path, PathBuf::from("FILE.mcraw"));
        assert_eq!(config.backend, PipeCliBackend::Gpu);
        assert_eq!(config.vignette, PipeCliVignette::WithCorrection);
        assert_eq!(
            config.output,
            PipeCliOutput::File(PathBuf::from("clip.yuv444p12le"))
        );
    }

    #[test]
    fn pipe_command_maps_stdout_to_production_run_config() {
        let command = parse_ok(&["pipe", "FILE.mcraw"]);
        let Mcraw4VulkanCommand::Pipe(pipe) = command else {
            panic!("expected pipe command");
        };
        let config = pipe.into_pipe_run_config(PipeCliOutput::Stdout);

        assert_eq!(config.output, PipeCliOutput::Stdout);
    }

    #[test]
    fn pipe_metadata_parses() {
        let command = parse_ok(&["pipe", "--metadata", "FILE.mcraw"]);
        let Mcraw4VulkanCommand::PipeMetadata(metadata) = command else {
            panic!("expected pipe metadata command");
        };

        assert_eq!(metadata.input, PathBuf::from("FILE.mcraw"));
        assert_eq!(metadata.backend, ComputeBackendChoice::Gpu);
        assert_eq!(metadata.vignette, VignetteCorrectionChoice::WithCorrection);
        assert_eq!(metadata.settings, SettingsSourceChoice::Default);
    }

    #[test]
    fn pipe_metadata_accepts_and_preserves_production_route_flags() {
        let command = parse_ok(&[
            "pipe",
            "--cpu",
            "--optimized",
            "--no-vig-correction",
            "--metadata",
            "FILE.mcraw",
        ]);
        let Mcraw4VulkanCommand::PipeMetadata(metadata) = command else {
            panic!("expected pipe metadata command");
        };

        assert_eq!(metadata.backend, ComputeBackendChoice::Cpu);
        assert_eq!(metadata.vignette, VignetteCorrectionChoice::NoCorrection);
        assert_eq!(metadata.settings, SettingsSourceChoice::Optimized);
    }

    #[test]
    fn pipe_metadata_rejects_conflicting_production_route_flags() {
        assert!(
            parse_err(&["pipe", "--metadata", "FILE.mcraw", "--gpu", "--cpu"])
                .contains("conflicts")
        );
        assert!(
            parse_err(&[
                "pipe",
                "--metadata",
                "FILE.mcraw",
                "--with-vig-correction",
                "--no-vig-correction"
            ])
            .contains("conflicts")
        );
        assert!(
            parse_err(&[
                "pipe",
                "--metadata",
                "FILE.mcraw",
                "--default",
                "--optimized"
            ])
            .contains("conflicts")
        );
    }

    #[test]
    fn pipe_help_parses() {
        assert_eq!(parse_ok(&["pipe", "--help"]), Mcraw4VulkanCommand::PipeHelp);
    }

    #[test]
    fn pipe_metadata_requires_input() {
        assert!(parse_err(&["pipe", "--metadata"]).contains("--metadata requires a path"));
    }

    #[test]
    fn pipe_metadata_rejects_output() {
        assert!(
            parse_err(&[
                "pipe",
                "--metadata",
                "FILE.mcraw",
                "--output",
                "clip.yuv444p12le"
            ])
            .contains("cannot be combined")
        );
        assert!(
            parse_err(&[
                "pipe",
                "--output",
                "clip.yuv444p12le",
                "--metadata",
                "FILE.mcraw"
            ])
            .contains("cannot be combined")
        );
    }

    #[test]
    fn pipe_metadata_rejects_extra_input() {
        assert!(
            parse_err(&["pipe", "--metadata", "FILE.mcraw", "EXTRA.mcraw"])
                .contains("exactly one input")
        );
    }

    #[test]
    fn render_metadata_is_not_public() {
        assert!(parse_err(&["render", "--metadata", "FILE.mcraw"]).contains("unknown command"));
    }

    #[test]
    fn render_help_is_not_public() {
        assert!(parse_err(&["render", "--help"]).contains("unknown command"));
    }

    #[test]
    fn pipe_metadata_route_uses_metadata_writer() {
        let source = include_str!("cli.rs");
        let old_writer_name = ["write_render", "_metadata_json"].concat();

        assert!(source.contains("Mcraw4VulkanCommand::PipeMetadata(command)"));
        assert!(source.contains("write_pipe_metadata_json_for_config(&config"));
        assert!(!source.contains(&old_writer_name));
    }

    #[test]
    fn top_level_cli_dispatch_routes_known_commands() {
        let source = include_str!("cli.rs");

        assert!(source.contains("Mcraw4VulkanCommand::Display(command)"));
        assert!(source.contains("Mcraw4VulkanCommand::Pipe(command)"));
        assert!(source.contains("Mcraw4VulkanCommand::PipeMetadata(command)"));
        assert!(source.contains("Mcraw4VulkanCommand::Dng(DngCommand::Mount(command))"));
        assert!(source.contains("Mcraw4VulkanCommand::Optimizer(command)"));
    }

    #[test]
    fn pipe_rejects_output_without_path() {
        assert!(parse_err(&["pipe", "--output"]).contains("requires a path"));
    }

    #[test]
    fn pipe_rejects_output_equal_to_input() {
        assert!(
            parse_err(&["pipe", "--output", "FILE.mcraw", "FILE.mcraw"]).contains("must not equal")
        );
    }

    #[test]
    fn pipe_rejects_vsync() {
        assert!(parse_err(&["pipe", "--vsync", "FILE.mcraw"]).contains("display"));
    }

    #[test]
    fn pipe_rejects_overlay_flags() {
        assert!(parse_err(&["pipe", "--no-overlay", "FILE.mcraw"]).contains("display"));
        assert!(parse_err(&["pipe", "--with-overlay", "FILE.mcraw"]).contains("display"));
    }

    #[test]
    fn pipe_rejects_conflicting_backend_and_policy_flags() {
        assert!(parse_err(&["pipe", "--gpu", "--cpu", "FILE.mcraw"]).contains("conflicts"));
        assert!(
            parse_err(&[
                "pipe",
                "--no-vig-correction",
                "--with-vig-correction",
                "FILE.mcraw"
            ])
            .contains("conflicts")
        );
        assert!(
            parse_err(&["pipe", "--default", "--optimized", "FILE.mcraw"]).contains("conflicts")
        );
    }

    #[test]
    fn optimized_effective_settings_apply_pipe_payload_profile() {
        let command = PipeCommand {
            input: PathBuf::from("missing.mcraw"),
            backend: ComputeBackendChoice::Gpu,
            vignette: VignetteCorrectionChoice::WithCorrection,
            settings: SettingsSourceChoice::Optimized,
            output: PipeOutputChoice::File(PathBuf::from("clip.yuv444p12le")),
            internal_measurement: None,
        };
        let effective = optimized_effective_settings();
        let config = command.into_pipe_run_config_with_settings(
            PipeCliOutput::File(PathBuf::from("clip.yuv444p12le")),
            &effective,
        );

        assert_eq!(
            config.payload_feeder_options,
            mcraw4vulkan_optimizer::PayloadProfile::OffsetPrefetch.payload_feeder_options()
        );
    }

    #[test]
    fn pipe_cpu_run_reaches_input_open() {
        let command = Mcraw4VulkanCommand::Pipe(PipeCommand {
            input: PathBuf::from("missing.mcraw"),
            backend: ComputeBackendChoice::Cpu,
            vignette: VignetteCorrectionChoice::WithCorrection,
            settings: SettingsSourceChoice::Default,
            output: PipeOutputChoice::File(PathBuf::from("clip.yuv444p12le")),
            internal_measurement: None,
        });
        let error = run_production_command(command)
            .expect_err("missing input still fails")
            .to_string();

        assert!(error.contains("failed to open input"));
        assert!(error.contains("missing.mcraw"));
    }

    #[test]
    fn optimizer_parses() {
        let command = parse_ok(&["optimizer", "FILE.mcraw"]);
        let Mcraw4VulkanCommand::Optimizer(optimizer) = command else {
            panic!("expected optimizer command");
        };
        assert_eq!(
            optimizer,
            OptimizerCommand::Run {
                input: PathBuf::from("FILE.mcraw")
            }
        );
    }

    #[test]
    fn optimizer_help_routes_to_usage() {
        assert_eq!(
            parse_ok(&["optimizer", "--help"]),
            Mcraw4VulkanCommand::Help
        );
    }

    #[test]
    fn optimizer_restore_defaults_parses() {
        let command = parse_ok(&["optimizer", "--restore-defaults"]);
        let Mcraw4VulkanCommand::Optimizer(optimizer) = command else {
            panic!("expected optimizer command");
        };
        assert_eq!(optimizer, OptimizerCommand::RestoreDefaults);
    }

    #[test]
    fn optimizer_restore_defaults_rejects_file_or_extra_options() {
        assert!(
            parse_err(&["optimizer", "--restore-defaults", "FILE.mcraw"])
                .contains("accepts no FILE.mcraw")
        );
        assert!(
            parse_err(&["optimizer", "--restore-defaults", "--anything"])
                .contains("accepts no FILE.mcraw")
        );
    }

    #[test]
    fn optimizer_rejects_flags() {
        assert!(parse_err(&["optimizer", "--default", "FILE.mcraw"]).contains("unknown option"));
    }

    #[test]
    fn optimizer_rejects_overlay_flags() {
        assert!(parse_err(&["optimizer", "--no-overlay", "FILE.mcraw"]).contains("display"));
    }

    #[test]
    fn optimizer_route_builds_shellout_config_with_current_exe() {
        let config = optimizer_run_config_for_command(OptimizerCommand::Run {
            input: PathBuf::from("FILE.mcraw"),
        })
        .expect("optimizer config");

        assert_eq!(config.input_path, PathBuf::from("FILE.mcraw"));
        assert_eq!(
            config.frames,
            mcraw4vulkan_optimizer::shellout::OPTIMIZER_DECISION_FRAMES
        );
        assert_eq!(config.mcraw4vulkan_exe, std::env::current_exe().unwrap());
    }

    #[test]
    fn optimizer_config_helper_passes_known_total_ram() {
        let total_ram_bytes = 16 * 1024 * 1024 * 1024;
        let config = optimizer_run_config_for_command_with_total_ram(
            OptimizerCommand::Run {
                input: PathBuf::from("FILE.mcraw"),
            },
            Some(total_ram_bytes),
        )
        .expect("optimizer config");

        assert_eq!(config.total_ram_bytes, Some(total_ram_bytes));
    }

    #[test]
    fn optimizer_config_helper_passes_unavailable_total_ram() {
        let config = optimizer_run_config_for_command_with_total_ram(
            OptimizerCommand::Run {
                input: PathBuf::from("FILE.mcraw"),
            },
            None,
        )
        .expect("optimizer config");

        assert_eq!(config.total_ram_bytes, None);
    }

    #[test]
    fn optimizer_route_source_calls_shellout_runner() {
        let source = include_str!("cli.rs");

        let metadata_check = source.find("McrawContainer::open_for_display").unwrap();
        let frame_gate = source
            .find("mcraw4vulkan_optimizer::shellout::validate_optimizer_frame_count")
            .unwrap();
        let runner = source
            .find("mcraw4vulkan_optimizer::run_optimizer")
            .unwrap();
        assert!(metadata_check < frame_gate);
        assert!(frame_gate < runner);
        assert!(source.contains("optimizer_run_config_for_command"));
    }

    #[test]
    fn optimizer_short_input_error_reaches_cli_with_exact_wording() {
        let error = mcraw4vulkan_optimizer::shellout::validate_optimizer_frame_count(599)
            .expect_err("599 frames must be rejected");
        let cli_error = anyhow::Error::new(error);

        assert_eq!(
            cli_error.to_string(),
            "Optimization testing requires at least 600 frames for accuracy."
        );
    }

    #[test]
    fn unknown_subcommand_rejects() {
        assert!(parse_err(&["unknown", "FILE.mcraw"]).contains("unknown command"));
    }

    #[test]
    fn missing_input_rejects() {
        assert!(parse_err(&["display"]).contains("requires FILE.mcraw"));
    }

    #[test]
    fn dng_missing_input_rejects() {
        assert!(parse_err(&["dng"]).contains("requires FILE.mcraw"));
    }

    #[test]
    fn extra_positional_rejects() {
        assert!(parse_err(&["display", "a.mcraw", "b.mcraw"]).contains("extra positional"));
    }

    #[test]
    fn help_contains_final_commands() {
        let help = Cli::usage();
        for expected in [
            "mcraw4vulkan display [--gpu|--cpu] [--no-vig-correction|--with-vig-correction] [--vsync|--no-vsync] [--with-sound] [--default|--optimized] [--with-overlay|--no-overlay] FILE.mcraw",
            "mcraw4vulkan dng",
            "mcraw4vulkan dng unmount FILE.mcraw",
            "mcraw4vulkan dng unmount all",
            "display --with-sound plays synced clip audio and is valid only with Vsync display.",
            "DNG mounts:",
            "dng FILE.mcraw mounts one virtual DNG and audio folder; it does not export DNGs directly.",
            "The command serves one foreground per-file mountpoint under the mcraw4vulkan folder in your home directory.",
            "Start one dng command per mounted clip; press Ctrl-C to stop a foreground mount.",
            "dng unmount FILE.mcraw  unmount the per-file DNG mount for FILE.mcraw",
            "dng unmount all         unmount all mcraw4vulkan-owned DNG mounts",
            "mcraw4vulkan pipe",
            "mcraw4vulkan pipe --metadata FILE.mcraw",
            "mcraw4vulkan optimizer FILE.mcraw",
            "mcraw4vulkan optimizer --restore-defaults",
        ] {
            assert!(help.contains(expected), "missing {expected}");
        }
        assert!(!help.contains("mcraw4vulkan render --metadata"));
        assert!(!help.contains("\n  render"));
    }

    #[test]
    fn dng_help_omits_os_and_mount_backend_words() {
        let help = Cli::usage();
        for forbidden in [
            "Linux",
            "linux",
            "macOS",
            "MacOS",
            "Windows",
            "windows",
            "FUSE",
            "fuse",
            "ProjFS",
            "macFUSE",
            "WinFsp",
            "XDG",
            "XDG_VIDEOS_DIR",
            "Videos",
            "Movies",
            "USERPROFILE",
        ] {
            assert!(!help.contains(forbidden), "DNG help contains {forbidden}");
        }
    }

    #[test]
    fn help_documents_overlay_only_for_display() {
        let help = Cli::usage();

        assert!(help.contains("[--with-overlay|--no-overlay] FILE.mcraw"));
        assert!(
            help.contains("display:   --gpu --no-vig-correction --vsync --default --with-overlay")
        );
        assert!(!help.contains("mcraw4vulkan dng [--gpu|--cpu] [--no-vig-correction|--with-vig-correction] [--default|--optimized] [--with-overlay|--no-overlay]"));
        assert!(!help.contains("mcraw4vulkan pipe [--gpu|--cpu] [--no-vig-correction|--with-vig-correction] [--default|--optimized] [--with-overlay|--no-overlay]"));
    }

    #[test]
    fn help_omits_hidden_internal_measurement_flags() {
        let help = Cli::usage();
        for hidden in [
            "--internal-measure",
            "--internal-measure-report",
            "--internal-measure-frames",
            "--internal-measure-warmup-frames",
            "--internal-payload-profile",
        ] {
            assert!(!help.contains(hidden), "help contains {hidden}");
        }
    }

    #[test]
    fn help_omits_direct_dng_export_command() {
        let help = Cli::usage();
        assert!(!help.contains("dng export"));
        assert!(!help.contains("dng mount FILE"));
    }

    #[test]
    fn dng_mount_naming_maps_stem() {
        let naming = dng_mount_naming_for_input(Path::new("260518_211252_VIDEO_25mm.mcraw"))
            .expect("naming");
        assert_eq!(
            naming.mounted_folder,
            PathBuf::from("260518_211252_VIDEO_25mm")
        );
        assert_eq!(
            naming.first_dng_path,
            PathBuf::from("260518_211252_VIDEO_25mm").join("260518_211252_VIDEO_25mm_000000.dng")
        );
        assert_eq!(
            naming.audio_path,
            PathBuf::from("260518_211252_VIDEO_25mm").join("260518_211252_VIDEO_25mm.wav")
        );
    }

    #[test]
    fn help_omits_external_tool_words() {
        let help = Cli::usage();
        let forbidden_words = [
            ["ff", "mpeg"].concat(),
            ["FF", "mpeg"].concat(),
            ["lib", "av"].concat(),
            ["Pro", "Res"].concat(),
            ["HE", "VC"].concat(),
            ["NV", "ENC"].concat(),
            ["lib", "placebo"].concat(),
            ["x", "264"].concat(),
            ["x", "265"].concat(),
        ];
        for forbidden in forbidden_words {
            assert!(!help.contains(&forbidden), "help contains {forbidden}");
        }
    }

    #[test]
    fn render_command_is_not_public() {
        assert!(parse_err(&["render", "FILE.mcraw"]).contains("unknown command"));
    }

    #[test]
    fn pipe_help_documents_metadata() {
        let help = Cli::pipe_usage();

        assert!(help.contains("mcraw4vulkan pipe --metadata FILE.mcraw"));
        assert!(help.contains("--metadata"));
        assert!(!help.contains("render --metadata"));
    }
}
