use std::fmt;
use std::path::{Path, PathBuf};

use mcraw4vulkan::{
    PIPE_AUDIO_FILE_SUFFIX, PIPE_METADATA_FILE_SUFFIX,
    PIPE_PRORES_FILE_SUFFIX as PRORES_OUTPUT_FILE_SUFFIX,
    PIPE_PRORES_SIDECAR_SUFFIX as PRORES_OUTPUT_SIDECAR_SUFFIX, PipeExampleFacts,
    pipe_example_facts_for_input,
};

use crate::gui_settings::{DecodeMode, OptimizerProfile};

pub const HEADER: &str = "mcraw4vulkan Pipe";
pub const BODY: &str = "Using the command line, mcraw4vulkan Pipe decodes a MotionCam RAW file directly into a 12 bit yuv444p12le bytestream with BT.2020 primaries and the AppleLog transfer function. Audio and metadata sidecars are also created.";
pub const USEFUL_TEXT: &str = "The bytestream is formatted for rapid processing in the GPU. To correctly display the .mov file in another app, you may have to manually assign BT2020 AppleLog in that app";
pub const NO_SELECTED_FILE_MESSAGE: &str = "Select a playlist file to generate a pipe example.";
pub const PRORES_OUTPUT_COLOR_TEXT: &str = "The macOS VideoToolbox example converts the 12-bit producer output to p410le: 10 meaningful bits in 16-bit storage. ProRes is lossy. Apple Log input assignment remains manual; a sidecar or filename does not ensure automatic recognition.";
pub const VULKAN_FFMPEG_8_1_NOTICE: &str =
    "FFmpeg 8.1 or later is required for GPU encoding of ProRes files with Vulkan";
pub const SIMPLE_COMMAND_PIPE: &str = "mcraw4vulkan pipe";
pub const SIMPLE_COMMAND: &str = "mcraw4vulkan pipe FILE_NAME.mcraw > FILE_NAME.yuv444p12le";
pub const COMPLICATED_EXAMPLE_LABEL: &str = "A complicated example:";
pub const COPY_COMMAND_TEXT: &str = "Copy and Paste this whole command into the terminal";
const PRORES_OUTPUT_COLOR_ARGS: &str = "-colorspace bt2020nc -color_primaries bt2020 -color_trc 2 -color_range tv -movflags +write_colr";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PipeExamplePanel {
    Message(String),
    Error(String),
    Example(PipeExample),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipeExample {
    pub input_path: PathBuf,
    pub facts: PipeExampleFacts,
    pub decode_mode: DecodeMode,
    pub optimizer_profile: OptimizerProfile,
    pub vignette_correction: bool,
    pub simple_command: String,
    pub command: String,
    pub output_file_name: String,
    pub target: CommandTarget,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandTarget {
    Linux,
    Windows,
    Macos,
}

impl CommandTarget {
    pub fn terminal_label(self) -> &'static str {
        "A simple example using the terminal to create a video bytestream"
    }

    pub fn hardware_text(self) -> &'static str {
        match self {
            Self::Linux | Self::Windows => {
                "With this single command, use the GPU to decode the mcraw file and pass it directly to FFmpeg to GPU-encode (Vulkan) a 12 bit ProRes 4444 .mov file"
            }
            Self::Macos => {
                "With this single command, use the GPU to decode the .mcraw file and pass it directly to FFmpeg to hardware-encode (VideoToolbox) a 10 bit ProRes 4444 .mov file."
            }
        }
    }

    pub fn ffmpeg_prerequisite_text(self) -> Option<&'static str> {
        match self {
            Self::Linux | Self::Windows => Some(VULKAN_FFMPEG_8_1_NOTICE),
            Self::Macos => Some(PRORES_OUTPUT_COLOR_TEXT),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipeExampleError {
    message: String,
}

impl PipeExampleError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for PipeExampleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for PipeExampleError {}

pub fn example_for_selected_file(
    selected: Option<&Path>,
    decode_mode: DecodeMode,
    optimizer_profile: OptimizerProfile,
    vignette_correction: bool,
) -> PipeExamplePanel {
    example_for_selected_file_with_facts(
        selected,
        decode_mode,
        optimizer_profile,
        vignette_correction,
        |input_path| {
            pipe_example_facts_for_input(input_path)
                .map_err(|error| PipeExampleError::new(error.to_string()))
        },
    )
}

pub fn example_for_selected_file_with_facts(
    selected: Option<&Path>,
    decode_mode: DecodeMode,
    optimizer_profile: OptimizerProfile,
    vignette_correction: bool,
    facts_source: impl FnOnce(&Path) -> Result<PipeExampleFacts, PipeExampleError>,
) -> PipeExamplePanel {
    let Some(input_path) = selected else {
        return PipeExamplePanel::Message(NO_SELECTED_FILE_MESSAGE.to_string());
    };

    let facts = match facts_source(input_path) {
        Ok(facts) => facts,
        Err(error) => {
            return PipeExamplePanel::Error(format!("Could not read PIPE example facts: {error}"));
        }
    };

    PipeExamplePanel::Example(build_pipe_example(
        input_path,
        facts,
        decode_mode,
        optimizer_profile,
        vignette_correction,
        current_command_target(),
    ))
}

pub fn build_pipe_example(
    input_path: &Path,
    facts: PipeExampleFacts,
    decode_mode: DecodeMode,
    optimizer_profile: OptimizerProfile,
    vignette_correction: bool,
    target: CommandTarget,
) -> PipeExample {
    let output_file_name = output_file_name_for_input(input_path);
    let simple_command = SIMPLE_COMMAND.to_string();
    let command = command_for_target(
        input_path,
        &facts,
        decode_mode,
        optimizer_profile,
        vignette_correction,
        target,
        &output_file_name,
    );

    PipeExample {
        input_path: input_path.to_path_buf(),
        facts,
        decode_mode,
        optimizer_profile,
        vignette_correction,
        simple_command,
        command,
        output_file_name,
        target,
    }
}

fn current_command_target() -> CommandTarget {
    if cfg!(target_os = "windows") {
        CommandTarget::Windows
    } else if cfg!(target_os = "macos") {
        CommandTarget::Macos
    } else {
        CommandTarget::Linux
    }
}

fn command_for_target(
    input_path: &Path,
    facts: &PipeExampleFacts,
    decode_mode: DecodeMode,
    optimizer_profile: OptimizerProfile,
    vignette_correction: bool,
    target: CommandTarget,
    output_file_name: &str,
) -> String {
    match target {
        CommandTarget::Linux => linux_command(
            input_path,
            facts,
            decode_mode,
            optimizer_profile,
            vignette_correction,
            output_file_name,
        ),
        CommandTarget::Windows => windows_command(
            input_path,
            facts,
            decode_mode,
            optimizer_profile,
            vignette_correction,
            output_file_name,
        ),
        CommandTarget::Macos => macos_command(
            input_path,
            facts,
            decode_mode,
            optimizer_profile,
            vignette_correction,
            output_file_name,
        ),
    }
}

fn linux_command(
    input_path: &Path,
    facts: &PipeExampleFacts,
    decode_mode: DecodeMode,
    optimizer_profile: OptimizerProfile,
    vignette_correction: bool,
    output_file_name: &str,
) -> String {
    let settings = pipe_settings_flags(decode_mode, optimizer_profile, vignette_correction);
    let stem = file_stem_text(input_path);
    let aspect = display_aspect_ratio_text(facts);
    let filter = unix_quote(&vulkan_prores_filter(&aspect));
    let timescale_arg = ffmpeg_timescale_arg(facts);
    let mux_command = unix_mux_command(&timescale_arg);
    let output_sidecar_file_name = output_sidecar_file_name_for_input(input_path);

    format!(
        concat!(
            "( set -o pipefail; in={input}; base={base}; out={output}; sidecar={output_sidecar} ",
            "&& outdir=$(dirname \"$out\") && [[ ! -e \"$out\" && ! -e \"$sidecar\" ]] || {{ echo \"Output already exists: $out or $sidecar\" >&2; exit 1; }}; ",
            "in_abs=\"$in\"; case \"$in_abs\" in /*) ;; *) in_abs=\"$(pwd)/$in_abs\";; esac; ",
            "tmpdir=$(mktemp -d \"${{outdir}}/${{base}}-tmp.XXXXXX\") && trap 'rm -fr \"$tmpdir\"' EXIT ",
            "&& tmp_video=\"${{tmpdir}}/${{base}}-video.mov\" && tmp_mux=\"${{tmpdir}}/${{base}}-mux.mov\" && producer_sidecar=\"${{tmpdir}}/${{base}}{producer_sidecar_suffix}\" && wav=\"${{tmpdir}}/${{base}}{audio_suffix}\" ",
            "&& ( cd \"$tmpdir\" && mcraw4vulkan pipe {settings} \"$in_abs\" ) | ffmpeg -nostdin -hide_banner -y -noauto_conversion_filters -init_hw_device vulkan=vk:0 -filter_hw_device vk -f rawvideo -pixel_format yuv444p12le -video_size {width}x{height} -framerate {fps_num}/{fps_den} -color_range tv -color_primaries bt2020 -color_trc 2 -colorspace bt2020nc -i - -map 0:v:0 -vf {filter} -an -c:v prores_ks_vulkan -profile:v 4 -alpha_bits 0 {output_color_args}{timescale_arg} \"$tmp_video\" ",
            "&& {mux_command} && mv \"$producer_sidecar\" \"$sidecar\" && mv \"$tmp_mux\" \"$out\" )"
        ),
        input = unix_quote(&path_text(input_path)),
        base = unix_quote(&stem),
        output = unix_quote(&format!("./{output_file_name}")),
        output_sidecar = unix_quote(&format!("./{output_sidecar_file_name}")),
        width = facts.width,
        height = facts.height,
        fps_num = facts.cadence.fps_num,
        fps_den = facts.cadence.fps_den,
        settings = settings,
        filter = filter,
        producer_sidecar_suffix = PIPE_METADATA_FILE_SUFFIX,
        audio_suffix = PIPE_AUDIO_FILE_SUFFIX,
        output_color_args = PRORES_OUTPUT_COLOR_ARGS,
        timescale_arg = timescale_arg,
        mux_command = mux_command,
    )
}

fn windows_command(
    input_path: &Path,
    facts: &PipeExampleFacts,
    decode_mode: DecodeMode,
    optimizer_profile: OptimizerProfile,
    vignette_correction: bool,
    output_file_name: &str,
) -> String {
    let settings = pipe_settings_flags(decode_mode, optimizer_profile, vignette_correction);
    let stem = file_stem_text(input_path);
    let aspect = display_aspect_ratio_text(facts);
    let filter = vulkan_prores_filter(&aspect);
    let cmd_switches = windows_cmd_switches();
    let timescale_arg = ffmpeg_timescale_arg(facts);
    let mux_command = windows_mux_command(&timescale_arg);
    let output_sidecar_file_name = output_sidecar_file_name_for_input(input_path);

    format!(
        concat!(
            "$ErrorActionPreference = 'Stop'; $in = {input}; $inAbs = [System.IO.Path]::GetFullPath($in); $base = {base}; ",
            "$out = Join-Path (Get-Location) {output}; $sidecar = Join-Path (Get-Location) {output_sidecar}; ",
            "if ((Test-Path -LiteralPath $out) -or (Test-Path -LiteralPath $sidecar)) {{ throw \"Output already exists: $out or $sidecar\" }}; ",
            "$outDir = Split-Path -Parent $out; if ([string]::IsNullOrEmpty($outDir)) {{ $outDir = (Get-Location).Path }}; $tmpDir = $null; ",
            "try {{ ",
            "$tmpDir = New-Item -ItemType Directory -Path (Join-Path $outDir ($base + '-tmp.' + [System.Guid]::NewGuid().ToString('N').Substring(0, 6))); ",
            "$tmpVideo = Join-Path $tmpDir.FullName ($base + '-video.mov'); $tmpMux = Join-Path $tmpDir.FullName ($base + '-mux.mov'); $producerSidecar = Join-Path $tmpDir.FullName ($base + '{producer_sidecar_suffix}'); $wav = Join-Path $tmpDir.FullName ($base + '{audio_suffix}'); ",
            "Push-Location -LiteralPath $tmpDir.FullName; try {{ cmd.exe{cmd_switches} \"mcraw4vulkan.exe pipe {settings} `\"$inAbs`\" | ffmpeg.exe -nostdin -hide_banner -y -noauto_conversion_filters -init_hw_device vulkan=vk:0 -filter_hw_device vk -f rawvideo -pixel_format yuv444p12le -video_size {width}x{height} -framerate {fps_num}/{fps_den} -color_range tv -color_primaries bt2020 -color_trc 2 -colorspace bt2020nc -i - -map 0:v:0 -vf {filter} -an -c:v prores_ks_vulkan -profile:v 4 -alpha_bits 0 {output_color_args}{timescale_arg} `\"$tmpVideo`\"\"; if ($LASTEXITCODE -ne 0) {{ throw \"PIPE or FFmpeg video stage failed with exit code $LASTEXITCODE\" }} }} finally {{ Pop-Location }}; ",
            "{mux_command}; Move-Item -LiteralPath $producerSidecar -Destination $sidecar; Move-Item -LiteralPath $tmpMux -Destination $out ",
            "}} finally {{ if ($tmpDir) {{ Remove-Item -LiteralPath $tmpDir.FullName -Recurse -Force -ErrorAction SilentlyContinue }} }}"
        ),
        input = powershell_quote(&path_text(input_path)),
        base = powershell_quote(&stem),
        output = powershell_quote(output_file_name),
        output_sidecar = powershell_quote(&output_sidecar_file_name),
        cmd_switches = cmd_switches,
        width = facts.width,
        height = facts.height,
        fps_num = facts.cadence.fps_num,
        fps_den = facts.cadence.fps_den,
        settings = settings,
        filter = filter,
        producer_sidecar_suffix = PIPE_METADATA_FILE_SUFFIX,
        audio_suffix = PIPE_AUDIO_FILE_SUFFIX,
        output_color_args = PRORES_OUTPUT_COLOR_ARGS,
        timescale_arg = timescale_arg,
        mux_command = mux_command,
    )
}

fn windows_cmd_switches() -> String {
    [" /", "d", " /", "s", " /", "c"].concat()
}

fn macos_command(
    input_path: &Path,
    facts: &PipeExampleFacts,
    decode_mode: DecodeMode,
    optimizer_profile: OptimizerProfile,
    vignette_correction: bool,
    output_file_name: &str,
) -> String {
    let settings = pipe_settings_flags(decode_mode, optimizer_profile, vignette_correction);
    let stem = file_stem_text(input_path);
    let aspect = display_aspect_ratio_text(facts);
    let filter = unix_quote(&videotoolbox_prores_filter(&aspect));
    let timescale_arg = ffmpeg_timescale_arg(facts);
    let mux_command = unix_mux_command(&timescale_arg);
    let output_sidecar_file_name = output_sidecar_file_name_for_input(input_path);

    format!(
        concat!(
            "( set -o pipefail; in={input}; base={base}; out={output}; sidecar={output_sidecar} ",
            "&& outdir=$(dirname \"$out\") && [[ ! -e \"$out\" && ! -e \"$sidecar\" ]] || {{ echo \"Output already exists: $out or $sidecar\" >&2; exit 1; }}; ",
            "in_abs=\"$in\"; case \"$in_abs\" in /*) ;; *) in_abs=\"$(pwd)/$in_abs\";; esac; ",
            "tmpdir=$(mktemp -d \"${{outdir}}/${{base}}-tmp.XXXXXX\") && trap 'rm -fr \"$tmpdir\"' EXIT ",
            "&& tmp_video=\"${{tmpdir}}/${{base}}-video.mov\" && tmp_mux=\"${{tmpdir}}/${{base}}-mux.mov\" && producer_sidecar=\"${{tmpdir}}/${{base}}{producer_sidecar_suffix}\" && wav=\"${{tmpdir}}/${{base}}{audio_suffix}\" ",
            "&& ( cd \"$tmpdir\" && mcraw4vulkan pipe {settings} \"$in_abs\" ) | ffmpeg -nostdin -hide_banner -y -f rawvideo -pixel_format yuv444p12le -video_size {width}x{height} -framerate {fps_num}/{fps_den} -color_range tv -color_primaries bt2020 -color_trc 2 -colorspace bt2020nc -i - -map 0:v:0 -vf {filter} -an -c:v prores_videotoolbox -profile:v 4 -pix_fmt p410le {output_color_args}{timescale_arg} \"$tmp_video\" ",
            "&& {mux_command} && mv \"$producer_sidecar\" \"$sidecar\" && mv \"$tmp_mux\" \"$out\" )"
        ),
        input = unix_quote(&path_text(input_path)),
        base = unix_quote(&stem),
        output = unix_quote(&format!("./{output_file_name}")),
        output_sidecar = unix_quote(&format!("./{output_sidecar_file_name}")),
        width = facts.width,
        height = facts.height,
        fps_num = facts.cadence.fps_num,
        fps_den = facts.cadence.fps_den,
        settings = settings,
        filter = filter,
        producer_sidecar_suffix = PIPE_METADATA_FILE_SUFFIX,
        audio_suffix = PIPE_AUDIO_FILE_SUFFIX,
        output_color_args = PRORES_OUTPUT_COLOR_ARGS,
        timescale_arg = timescale_arg,
        mux_command = mux_command,
    )
}

fn vulkan_prores_filter(aspect: &str) -> String {
    format!(
        "setsar=1,setdar={aspect},setparams=range=tv:color_primaries=bt2020:color_trc=2:colorspace=bt2020nc,format=yuv444p12le,hwupload"
    )
}

fn videotoolbox_prores_filter(aspect: &str) -> String {
    format!(
        "setsar=1,setdar={aspect},setparams=range=tv:color_primaries=bt2020:color_trc=2:colorspace=bt2020nc,format=p410le"
    )
}

fn ffmpeg_timescale_arg(facts: &PipeExampleFacts) -> String {
    format!(
        " -video_track_timescale {}",
        facts.cadence.video_track_timescale
    )
}

fn unix_mux_command(timescale_arg: &str) -> String {
    format!(
        "{{ if [[ -e \"$wav\" ]]; then ffmpeg -nostdin -hide_banner -y -i \"$tmp_video\" -i \"$wav\" -map '0:v:0' -map '1:a:0' -c:v copy -c:a copy {output_color_args}{timescale_arg} \"$tmp_mux\"; else mv \"$tmp_video\" \"$tmp_mux\"; fi; }}",
        output_color_args = PRORES_OUTPUT_COLOR_ARGS,
    )
}

fn windows_mux_command(timescale_arg: &str) -> String {
    format!(
        "if (Test-Path -LiteralPath $wav) {{ ffmpeg.exe -nostdin -hide_banner -y -i $tmpVideo -i $wav -map '0:v:0' -map '1:a:0' -c:v copy -c:a copy {output_color_args}{timescale_arg} $tmpMux; if ($LASTEXITCODE -ne 0) {{ throw \"FFmpeg audio mux failed with exit code $LASTEXITCODE\" }} }} else {{ Move-Item -LiteralPath $tmpVideo -Destination $tmpMux }}",
        output_color_args = PRORES_OUTPUT_COLOR_ARGS,
    )
}

fn display_aspect_ratio_text(facts: &PipeExampleFacts) -> String {
    format!(
        "{}/{}",
        facts.display_aspect_ratio.numerator, facts.display_aspect_ratio.denominator
    )
}

pub fn optimizer_profile_flag(profile: OptimizerProfile) -> &'static str {
    profile.cli_flag()
}

fn pipe_settings_flags(
    decode_mode: DecodeMode,
    profile: OptimizerProfile,
    vignette_correction: bool,
) -> String {
    format!(
        "{} {} {}",
        decode_mode_flag(decode_mode),
        optimizer_profile_flag(profile),
        vignette_correction_flag(vignette_correction)
    )
}

fn decode_mode_flag(mode: DecodeMode) -> &'static str {
    match mode {
        DecodeMode::Gpu => "--gpu",
        DecodeMode::Cpu => "--cpu",
    }
}

fn vignette_correction_flag(enabled: bool) -> &'static str {
    if enabled {
        "--with-vig-correction"
    } else {
        "--no-vig-correction"
    }
}

fn output_file_name_for_input(input_path: &Path) -> String {
    format!("{}{PRORES_OUTPUT_FILE_SUFFIX}", file_stem_text(input_path))
}

fn output_sidecar_file_name_for_input(input_path: &Path) -> String {
    format!(
        "{}{PRORES_OUTPUT_SIDECAR_SUFFIX}",
        file_stem_text(input_path)
    )
}

fn file_stem_text(input_path: &Path) -> String {
    input_path
        .file_stem()
        .map(|stem| stem.to_string_lossy().trim().to_string())
        .filter(|stem| !stem.is_empty())
        .unwrap_or_else(|| "output".to_string())
}

fn path_text(path: &Path) -> String {
    path.to_string_lossy().to_string()
}

fn unix_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

fn powershell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}
