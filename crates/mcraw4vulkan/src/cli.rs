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
        "mcraw4vulkan\n\nUsage:\n  mcraw4vulkan display [--gpu|--cpu] [--no-vig-correction|--with-vig-correction] [--vsync|--no-vsync] [--with-sound] [--default|--optimized] [--with-overlay|--no-overlay] FILE.mcraw\n  mcraw4vulkan dng [--gpu|--cpu] [--no-vig-correction|--with-vig-correction] [--default|--optimized] FILE.mcraw\n  mcraw4vulkan dng unmount FILE.mcraw\n  mcraw4vulkan dng unmount all\n  mcraw4vulkan pipe [--gpu|--cpu] [--no-vig-correction|--with-vig-correction] [--default|--optimized] [--output FILE] FILE.mcraw\n  mcraw4vulkan pipe --metadata FILE.mcraw\n  mcraw4vulkan optimizer FILE.mcraw\n  mcraw4vulkan optimizer --restore-defaults\n\nCommands:\n  display     show the 8-bit preview in a window; press F to toggle fullscreen\n  dng         mount or unmount mcraw4vulkan-owned virtual DNG plus audio folders\n  pipe        write direct yuv444p12le rawvideo or print PIPE metadata JSON\n  optimizer   measure and optionally save optimized settings\n\nDefaults:\n  display:   --gpu --no-vig-correction --vsync --default --with-overlay\n  dng:       --gpu --with-vig-correction --default\n  pipe:      --gpu --with-vig-correction --default\n\nPIPE output contract:\n  stdout is planar yuv444p12le: full Y, Cb, then Cr planes; six bytes per pixel. Samples are little-endian 16-bit words with 12 meaningful low bits. Color is TV-range BT.2020/D65, original Apple Log, BT.2020 NCL, no alpha channel. To correctly display the output file in another app, you may have to manually assign BT2020 AppleLog in that app\n  Diagnostics, progress, and warnings use stderr only.\n\nDisplay audio:\n  display --with-sound plays synced clip audio and is valid only with Vsync display.\n  display --with-sound cannot be combined with --no-vsync.\n\nDNG mounts:\n  dng FILE.mcraw mounts one virtual DNG and audio folder; it does not export DNGs directly.\n  The command serves one foreground per-file mountpoint under the mcraw4vulkan folder in your home directory.\n  Start one dng command per mounted clip; press Ctrl-C to stop a foreground mount.\n\nDNG unmounts:\n  dng unmount FILE.mcraw  unmount the per-file DNG mount for FILE.mcraw\n  dng unmount all         unmount all mcraw4vulkan-owned DNG mounts\n\nExamples:\n  mcraw4vulkan display --gpu --no-vig-correction --vsync --with-overlay --default FILE.mcraw\n  mcraw4vulkan display --with-sound FILE.mcraw\n  mcraw4vulkan display --cpu --no-vsync --no-overlay FILE.mcraw\n  mcraw4vulkan dng --gpu --with-vig-correction --default FILE.mcraw\n  mcraw4vulkan dng unmount FILE.mcraw\n  mcraw4vulkan dng unmount all\n  mcraw4vulkan pipe --gpu --with-vig-correction --output clip.yuv444p12le FILE.mcraw\n  mcraw4vulkan pipe --metadata FILE.mcraw\n  mcraw4vulkan optimizer FILE.mcraw\n  mcraw4vulkan optimizer --restore-defaults\n\nPIPE stdout safety:\n  pipe without --output writes raw bytes to stdout only when stdout is not a terminal.\n  Diagnostics, progress, and warnings use stderr."
    }

    pub fn pipe_usage() -> &'static str {
        "mcraw4vulkan pipe\n\nUsage:\n  mcraw4vulkan pipe [--gpu|--cpu] [--no-vig-correction|--with-vig-correction] [--default|--optimized] [--output FILE] FILE.mcraw\n  mcraw4vulkan pipe --metadata FILE.mcraw\n\nOptions:\n  --metadata   print strict PIPE sidecar version 4 JSON without rendering video\n  --output     write raw yuv444p12le video bytes to FILE\n\nOutput contract:\n  stdout is planar yuv444p12le: full Y, Cb, then Cr planes; six bytes per pixel. Samples are little-endian 16-bit words with 12 meaningful low bits. Color is TV-range BT.2020/D65, original Apple Log, BT.2020 NCL, no alpha channel. To correctly display the output file in another app, you may have to manually assign BT2020 AppleLog in that app\n\nPIPE stdout safety:\n  pipe without --output writes raw bytes to stdout only when stdout is not a terminal.\n  Diagnostics, progress, and warnings use stderr."
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
