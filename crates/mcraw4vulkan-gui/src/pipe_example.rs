use std::fmt;
use std::path::{Path, PathBuf};

use mcraw4vulkan::{PipeExampleFacts, pipe_example_facts_for_input};

use crate::gui_settings::{DecodeMode, OptimizerProfile};

pub const HEADER: &str = "mcraw4vulkan Pipe";
pub const BODY: &str = "Using the command line, mcraw4vulkan Pipe decodes a MotionCam RAW file directly into a 12 bit yuv444p12le bytestream with BT.2020 primaries and a linear transfer coefficient. Audio and metadata sidecars are also created.";
pub const USEFUL_TEXT: &str = "The bytestream is formatted for rapid processing in the GPU. It is intentionally not display-ready and may appear dark until adjusted in an editing app.";
pub const NO_SELECTED_FILE_MESSAGE: &str = "Select a playlist file to generate a pipe example.";
pub const PRORES_OUTPUT_COLOR_TEXT: &str = "On macOS, ProRes hardware encoding with VideoToolbox requires the bytestream to first be converted to 10-bit 444. This is still high-quality. For full-quality lossless video, use the DNG method instead.";
pub const VULKAN_FFMPEG_8_1_NOTICE: &str =
    "FFmpeg 8.1 or later is required for GPU encoding of ProRes files with Vulkan";
pub const SIMPLE_COMMAND_PIPE: &str = "mcraw4vulkan pipe";
pub const SIMPLE_COMMAND: &str = "mcraw4vulkan pipe FILE_NAME.mcraw > FILE_NAME.yuv444p12le";
pub const COMPLICATED_EXAMPLE_LABEL: &str = "A complicated example:";
pub const COPY_COMMAND_TEXT: &str = "Copy and Paste this whole command into the terminal";
const PIPE_METADATA_FILE_SUFFIX: &str = "-BT2020-linear-tv.json";
const PIPE_AUDIO_FILE_SUFFIX: &str = "-audio.wav";
const PRORES_OUTPUT_FILE_SUFFIX: &str = "_prores4444_bt2020_linear.mov";
const PRORES_OUTPUT_SIDECAR_SUFFIX: &str = "_prores4444_bt2020_linear.json";
const PRORES_OUTPUT_COLOR_ARGS: &str =
    "-colorspace bt2020nc -color_primaries bt2020 -color_trc linear -color_range tv";

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
        match self {
            Self::Linux => "A simple example using the terminal to create a video bytestream",
            Self::Windows => {
                "A simple example using Windows PowerShell to create a video bytestream"
            }
            Self::Macos => "A simple example using MacOS Terminal to create a video bytestream",
        }
    }

    pub fn hardware_text(self) -> &'static str {
        match self {
            Self::Linux | Self::Windows => {
                "With this single command, use the GPU to decode the .mcraw file and pass it directly to FFmpeg to GPU-encode (Vulkan) a 12 bit ProRes 4444 .mov file."
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
            "&& ( cd \"$tmpdir\" && mcraw4vulkan pipe {settings} \"$in_abs\" ) | ffmpeg -nostdin -hide_banner -y -noauto_conversion_filters -init_hw_device vulkan=vk:0 -filter_hw_device vk -f rawvideo -pixel_format yuv444p12le -video_size {width}x{height} -framerate {fps_num}/{fps_den} -color_range tv -color_primaries bt2020 -color_trc linear -colorspace bt2020nc -i - -map 0:v:0 -vf {filter} -an -c:v prores_ks_vulkan -profile:v 4 -alpha_bits 0 {output_color_args}{timescale_arg} \"$tmp_video\" ",
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
            "Push-Location -LiteralPath $tmpDir.FullName; try {{ cmd.exe{cmd_switches} \"mcraw4vulkan.exe pipe {settings} `\"$inAbs`\" | ffmpeg.exe -nostdin -hide_banner -y -noauto_conversion_filters -init_hw_device vulkan=vk:0 -filter_hw_device vk -f rawvideo -pixel_format yuv444p12le -video_size {width}x{height} -framerate {fps_num}/{fps_den} -color_range tv -color_primaries bt2020 -color_trc linear -colorspace bt2020nc -i - -map 0:v:0 -vf {filter} -an -c:v prores_ks_vulkan -profile:v 4 -alpha_bits 0 {output_color_args}{timescale_arg} `\"$tmpVideo`\"\"; if ($LASTEXITCODE -ne 0) {{ throw \"PIPE or FFmpeg video stage failed with exit code $LASTEXITCODE\" }} }} finally {{ Pop-Location }}; ",
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
            "&& ( cd \"$tmpdir\" && mcraw4vulkan pipe {settings} \"$in_abs\" ) | ffmpeg -nostdin -hide_banner -y -f rawvideo -pixel_format yuv444p12le -video_size {width}x{height} -framerate {fps_num}/{fps_den} -color_range tv -color_primaries bt2020 -color_trc linear -colorspace bt2020nc -i - -map 0:v:0 -vf {filter} -an -c:v prores_videotoolbox -profile:v 4 -pix_fmt p410le {output_color_args}{timescale_arg} \"$tmp_video\" ",
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
        "setsar=1,setdar={aspect},setparams=range=tv:color_primaries=bt2020:color_trc=linear:colorspace=bt2020nc,format=yuv444p12le,hwupload"
    )
}

fn videotoolbox_prores_filter(aspect: &str) -> String {
    format!(
        "setsar=1,setdar={aspect},setparams=range=tv:color_primaries=bt2020:color_trc=linear:colorspace=bt2020nc,format=p410le"
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

#[cfg(test)]
mod tests {
    use super::*;
    use mcraw4vulkan::{PipeAspectRatio, PipeMovCadence};

    #[cfg(unix)]
    use std::process::Command;

    const OCEAN_PATH: &str = concat!("/", "fixtures/ocean.mcraw");

    fn facts_for(
        width: u32,
        height: u32,
        source_fps_num: u64,
        source_fps_den: u64,
        dar_num: u64,
        dar_den: u64,
    ) -> PipeExampleFacts {
        PipeExampleFacts {
            width,
            height,
            cadence: PipeMovCadence::from_source_rate(source_fps_num, source_fps_den)
                .expect("valid test cadence"),
            sample_aspect_ratio: PipeAspectRatio::square_pixels(),
            display_aspect_ratio: PipeAspectRatio::new(dar_num, dar_den)
                .expect("valid test display aspect ratio"),
        }
    }

    fn standard_facts() -> PipeExampleFacts {
        facts_for(4096, 2160, 30_000, 1_001, 256, 135)
    }

    fn ocean_facts() -> PipeExampleFacts {
        facts_for(3840, 2160, 500_000_000, 16_668_971, 16, 9)
    }

    fn example(target: CommandTarget, facts: PipeExampleFacts, path: &str) -> PipeExample {
        build_pipe_example(
            Path::new(path),
            facts,
            DecodeMode::Gpu,
            OptimizerProfile::Default,
            true,
            target,
        )
    }

    fn expected_vulkan_filter(facts: PipeExampleFacts) -> String {
        format!(
            "setsar=1,setdar={}/{},setparams=range=tv:color_primaries=bt2020:color_trc=linear:colorspace=bt2020nc,format=yuv444p12le,hwupload",
            facts.display_aspect_ratio.numerator, facts.display_aspect_ratio.denominator
        )
    }

    fn expected_videotoolbox_filter(facts: PipeExampleFacts) -> String {
        format!(
            "setsar=1,setdar={}/{},setparams=range=tv:color_primaries=bt2020:color_trc=linear:colorspace=bt2020nc,format=p410le",
            facts.display_aspect_ratio.numerator, facts.display_aspect_ratio.denominator
        )
    }

    fn assert_direct_yuv_contract(command: &str, facts: PipeExampleFacts, target: CommandTarget) {
        let producer = match target {
            CommandTarget::Windows => "mcraw4vulkan.exe pipe",
            CommandTarget::Linux | CommandTarget::Macos => "mcraw4vulkan pipe",
        };
        let ffmpeg = match target {
            CommandTarget::Windows => "ffmpeg.exe",
            CommandTarget::Linux | CommandTarget::Macos => "ffmpeg",
        };

        assert!(command.contains(producer));
        assert!(command.contains(ffmpeg));
        assert!(command.contains("-f rawvideo"));
        assert!(command.contains("-pixel_format yuv444p12le"));
        assert!(command.contains(&format!("-video_size {}x{}", facts.width, facts.height)));
        assert!(command.contains(&format!(
            "-framerate {}/{}",
            facts.cadence.fps_num, facts.cadence.fps_den
        )));
        assert!(command.contains("-color_range tv"));
        assert!(command.contains("-color_primaries bt2020"));
        assert!(command.contains("-color_trc linear"));
        assert!(command.contains("-colorspace bt2020nc"));
        assert!(command.contains("-i -"));
        assert!(command.contains("-map 0:v:0"));
        assert!(command.contains("-an"));
        assert!(command.contains(&format!(
            "-video_track_timescale {}",
            facts.cadence.video_track_timescale
        )));
        assert!(command.contains(PRORES_OUTPUT_COLOR_ARGS));

        match target {
            CommandTarget::Linux | CommandTarget::Windows => {
                assert!(command.contains("-noauto_conversion_filters"));
                assert!(command.contains("-init_hw_device vulkan=vk:0"));
                assert!(command.contains("-filter_hw_device vk"));
                assert!(command.contains(&expected_vulkan_filter(facts)));
                assert!(command.contains("-c:v prores_ks_vulkan"));
                assert!(command.contains("-profile:v 4"));
                assert!(command.contains("-alpha_bits 0"));
                assert!(!command.contains("prores_videotoolbox"));
                assert!(!command.contains("p410le"));
            }
            CommandTarget::Macos => {
                assert!(!command.contains("-noauto_conversion_filters"));
                assert!(!command.contains("prores_ks_vulkan"));
                assert!(!command.contains("init_hw_device"));
                assert!(!command.contains("filter_hw_device"));
                assert!(!command.contains("vulkan=vk:"));
                assert!(!command.contains("hwupload"));
                assert!(command.contains(&expected_videotoolbox_filter(facts)));
                assert!(command.contains("-c:v prores_videotoolbox"));
                assert!(command.contains("-profile:v 4"));
                assert!(command.contains("-pix_fmt p410le"));
                assert!(!command.contains("-alpha_bits"));
            }
        }
    }

    fn assert_forbidden_tokens_absent(command: &str) {
        let private_validator = ["validate", "direct_yuv12_pipeline"].join("_");
        let task_gate = ["TASK", "ID"].join("_");
        let old_rgb_format = ["gbrp", "16le"].concat();
        let old_float_rgb_format = ["gbrpf", "16le"].concat();
        let old_timescale = 500_000_000_u64.to_string();

        for forbidden in [
            private_validator.as_str(),
            task_gate.as_str(),
            old_rgb_format.as_str(),
            old_float_rgb_format.as_str(),
            old_timescale.as_str(),
            "scale=",
            ",colorspace=",
            "zscale",
            "libplacebo",
            "qscale",
            "bits_per_mb",
            "quant_mat",
            "linear_images",
            "cleanup()",
            "owner_marker",
            "sidecar_published",
            "movie_published",
            " -ef ",
            " HardLink ",
            "fsutil.exe",
            "Remove-OwnedPublication",
        ] {
            assert!(
                !command.contains(forbidden),
                "generated command contains forbidden token {forbidden:?}: {command}"
            );
        }
    }

    #[test]
    fn simple_is_exact_and_complex_preserves_direct_yuv_contract_on_every_platform() {
        let facts = standard_facts();
        for target in [
            CommandTarget::Linux,
            CommandTarget::Windows,
            CommandTarget::Macos,
        ] {
            let example = example(target, facts, "clips/clip.mcraw");
            assert_eq!(example.simple_command, SIMPLE_COMMAND);
            assert_direct_yuv_contract(&example.command, facts, target);
            assert_forbidden_tokens_absent(&example.simple_command);
            assert_forbidden_tokens_absent(&example.command);
            assert!(
                example
                    .output_file_name
                    .ends_with(PRORES_OUTPUT_FILE_SUFFIX)
            );
            assert!(example.command.contains(PRORES_OUTPUT_SIDECAR_SUFFIX));
            assert!(example.command.contains(PIPE_METADATA_FILE_SUFFIX));
            assert!(example.command.contains(PIPE_AUDIO_FILE_SUFFIX));
        }
    }

    #[test]
    fn ocean_uses_exact_bounded_timing_in_every_platform_command() {
        let facts = ocean_facts();
        assert_eq!(facts.cadence.fps_num, 57_862);
        assert_eq!(facts.cadence.fps_den, 1_929);
        assert_eq!(facts.cadence.timebase_num, 1_929);
        assert_eq!(facts.cadence.timebase_den, 57_862);
        assert_eq!(facts.cadence.video_track_timescale, 57_862);

        for target in [
            CommandTarget::Linux,
            CommandTarget::Windows,
            CommandTarget::Macos,
        ] {
            let example = example(target, facts, OCEAN_PATH);
            assert_eq!(example.simple_command, SIMPLE_COMMAND);
            assert!(example.command.contains("-video_size 3840x2160"));
            assert!(example.command.contains("-framerate 57862/1929"));
            assert!(example.command.contains("setsar=1,setdar=16/9"));
            assert!(!example.command.contains(&500_000_000_u64.to_string()));
            assert!(!example.command.contains(&16_668_971_u64.to_string()));
            assert_eq!(
                example
                    .command
                    .matches("-video_track_timescale 57862")
                    .count(),
                2
            );
        }
    }

    #[test]
    fn linux_complex_command_has_exact_ponytail_structure() {
        let command = example(CommandTarget::Linux, standard_facts(), "clips/clip.mcraw").command;

        assert!(command.starts_with("( set -o pipefail;"));
        assert_eq!(command.matches("function ").count(), 0);
        assert_eq!(command.matches("cleanup()").count(), 0);
        assert_eq!(command.matches("trap ").count(), 1);
        assert_eq!(command.matches("trap 'rm -fr \"$tmpdir\"' EXIT").count(), 1);
        for signal in [" HUP", " INT", " TERM"] {
            assert!(!command.contains(signal));
        }
        assert_eq!(command.matches("HardLink").count(), 0);
        assert_eq!(command.matches(" link ").count(), 0);
        assert_eq!(command.matches("-ef").count(), 0);
        assert_eq!(command.matches("mktemp -d").count(), 1);
        assert_eq!(command.matches("mcraw4vulkan pipe ").count(), 1);
        assert_eq!(command.matches("ffmpeg -nostdin").count(), 2);
        assert_eq!(command.matches("if [[ -e \"$wav\" ]]").count(), 1);
        assert_eq!(command.matches("mv \"$tmp_video\" \"$tmp_mux\"").count(), 1);
        assert_eq!(
            command
                .matches("mv \"$producer_sidecar\" \"$sidecar\"")
                .count(),
            1
        );
        assert_eq!(command.matches("mv \"$tmp_mux\" \"$out\"").count(), 1);
        assert!(command.contains("[[ ! -e \"$out\" && ! -e \"$sidecar\" ]]"));
        assert!(command.contains("tmpdir=$(mktemp -d"));
    }

    #[test]
    fn simple_command_is_platform_independent_and_has_no_derived_metadata() {
        for target in [
            CommandTarget::Linux,
            CommandTarget::Windows,
            CommandTarget::Macos,
        ] {
            let example = example(target, standard_facts(), "clips/clip.mcraw");
            assert_eq!(example.simple_command, SIMPLE_COMMAND);
            assert!(example.simple_command.len() < example.command.len());
            for derived in [
                "clips/clip.mcraw",
                "4096x2160",
                "30000/1001",
                "-video_track_timescale",
                PIPE_AUDIO_FILE_SUFFIX,
                PIPE_METADATA_FILE_SUFFIX,
                "ffmpeg",
                "--gpu",
                "--default",
                "--with-vig-correction",
            ] {
                assert!(!example.simple_command.contains(derived));
            }
        }
    }

    #[test]
    fn platform_commands_keep_their_native_master_shaped_dialects() {
        let linux = example(CommandTarget::Linux, standard_facts(), "clips/clip.mcraw");
        assert_eq!(linux.simple_command, SIMPLE_COMMAND);
        assert!(linux.command.contains("mktemp -d"));
        assert!(linux.command.contains("case \"$in_abs\""));
        assert!(linux.command.contains("trap 'rm -fr \"$tmpdir\"' EXIT"));
        assert!(!linux.command.contains("PowerShell"));
        assert!(!linux.command.contains("New-Item"));

        let windows = example(
            CommandTarget::Windows,
            standard_facts(),
            "clips\\clip.mcraw",
        );
        assert_eq!(windows.simple_command, SIMPLE_COMMAND);
        assert!(
            windows
                .command
                .starts_with("$ErrorActionPreference = 'Stop';")
        );
        assert!(
            windows
                .command
                .contains(&format!("cmd.exe{}", windows_cmd_switches()))
        );
        assert!(windows.command.contains("New-Item -ItemType Directory"));
        assert!(windows.command.contains("Push-Location"));
        assert!(windows.command.contains("Move-Item"));
        assert!(windows.command.contains("Remove-Item"));
        assert!(!windows.command.contains("mktemp"));
        assert!(!windows.command.contains("trap "));

        let macos = example(CommandTarget::Macos, standard_facts(), "clips/clip.mcraw");
        assert_eq!(macos.simple_command, SIMPLE_COMMAND);
        assert!(macos.command.contains("mktemp -d"));
        assert!(
            CommandTarget::Macos
                .hardware_text()
                .contains("VideoToolbox")
        );
        assert!(
            CommandTarget::Macos
                .hardware_text()
                .contains("10 bit ProRes 4444")
        );
        assert_eq!(
            CommandTarget::Macos.ffmpeg_prerequisite_text(),
            Some(PRORES_OUTPUT_COLOR_TEXT)
        );
    }

    #[test]
    fn macos_commands_use_explicit_videotoolbox_boundary_without_vulkan() {
        let example = example(CommandTarget::Macos, ocean_facts(), OCEAN_PATH);
        assert_eq!(example.simple_command, SIMPLE_COMMAND);
        let command = &example.command;
        assert!(command.contains("-pixel_format yuv444p12le"));
        assert!(command.contains("format=p410le"));
        assert!(command.contains("-c:v prores_videotoolbox"));
        assert!(command.contains("-profile:v 4"));
        assert!(command.contains("-pix_fmt p410le"));
        for forbidden in [
            "prores_ks_vulkan",
            "init_hw_device",
            "filter_hw_device",
            "vulkan=vk:",
            "hwupload",
            "alpha_bits",
            "zscale",
            "libplacebo",
        ] {
            assert!(
                !command.contains(forbidden),
                "macOS command contains forbidden token {forbidden:?}: {command}"
            );
        }
    }

    #[test]
    fn settings_vignette_and_quoting_are_applied_to_complex_examples() {
        for target in [
            CommandTarget::Linux,
            CommandTarget::Windows,
            CommandTarget::Macos,
        ] {
            let path = match target {
                CommandTarget::Windows => "clips\\clip one's.mcraw",
                CommandTarget::Linux | CommandTarget::Macos => "clips/clip one's.mcraw",
            };
            let example = build_pipe_example(
                Path::new(path),
                standard_facts(),
                DecodeMode::Cpu,
                OptimizerProfile::Optimized,
                false,
                target,
            );
            assert_eq!(example.simple_command, SIMPLE_COMMAND);
            assert!(example.command.contains("--cpu"));
            assert!(example.command.contains("--optimized"));
            assert!(example.command.contains("--no-vig-correction"));
            assert!(!example.command.contains("--with-vig-correction"));
            match target {
                CommandTarget::Windows => {
                    assert!(example.command.contains("$in = 'clips\\clip one''s.mcraw'"));
                }
                CommandTarget::Linux | CommandTarget::Macos => {
                    assert!(example.command.contains("in='clips/clip one'\\''s.mcraw'"));
                }
            }
        }
    }

    #[test]
    fn with_vignette_and_default_gpu_flags_are_preserved_in_complex_examples() {
        for target in [
            CommandTarget::Linux,
            CommandTarget::Windows,
            CommandTarget::Macos,
        ] {
            let example = example(target, standard_facts(), "clip.mcraw");
            assert_eq!(example.simple_command, SIMPLE_COMMAND);
            assert!(
                example
                    .command
                    .contains("--gpu --default --with-vig-correction")
            );
            assert!(!example.command.contains("--no-vig-correction"));
        }
    }

    #[test]
    fn output_names_use_the_input_stem_on_every_platform() {
        for target in [
            CommandTarget::Linux,
            CommandTarget::Windows,
            CommandTarget::Macos,
        ] {
            let example = example(target, standard_facts(), "My Clip.mcraw");
            assert_eq!(
                example.output_file_name,
                "My Clip_prores4444_bt2020_linear.mov"
            );
            assert_eq!(example.simple_command, SIMPLE_COMMAND);
            assert!(
                example
                    .command
                    .contains("My Clip_prores4444_bt2020_linear.json")
            );
            for forbidden in ["_linear_tv", "scale-1of2", "1of2"] {
                assert!(!example.command.contains(forbidden));
            }
        }
    }

    #[test]
    fn source_display_aspect_ratio_is_used_without_hardcoding_ocean() {
        let facts = facts_for(4080, 3072, 24, 1, 85, 64);
        let example = example(CommandTarget::Linux, facts, "clip.mcraw");
        assert_eq!(example.simple_command, SIMPLE_COMMAND);
        assert!(example.command.contains("-video_size 4080x3072"));
        assert!(example.command.contains("setsar=1,setdar=85/64"));
        assert!(!example.command.contains("setdar=16/9"));
    }

    #[test]
    fn generated_commands_are_one_physical_line() {
        for target in [
            CommandTarget::Linux,
            CommandTarget::Windows,
            CommandTarget::Macos,
        ] {
            let example = example(target, standard_facts(), "clip.mcraw");
            for command in [&example.simple_command, &example.command] {
                assert!(!command.contains('\n'));
                assert!(!command.contains('\r'));
            }
        }
    }

    #[cfg(unix)]
    #[test]
    fn linux_and_macos_shell_commands_pass_bash_syntax_validation() {
        for target in [CommandTarget::Linux, CommandTarget::Macos] {
            let example = example(target, standard_facts(), "clips/clip one's.mcraw");
            for command in [&example.simple_command, &example.command] {
                let output = Command::new("bash")
                    .arg("-n")
                    .arg("-c")
                    .arg(command)
                    .output()
                    .expect("bash is available for command syntax validation");
                assert!(
                    output.status.success(),
                    "generated {target:?} command failed bash -n: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
            }
        }
    }

    #[test]
    fn selected_file_uses_injected_facts_provider() {
        let expected = standard_facts();
        let selected = Path::new("clips/clip.mcraw");
        let panel = example_for_selected_file_with_facts(
            Some(selected),
            DecodeMode::Cpu,
            OptimizerProfile::Optimized,
            false,
            |path| {
                assert_eq!(path, selected);
                Ok(expected)
            },
        );

        let PipeExamplePanel::Example(example) = panel else {
            panic!("expected generated Pipe Example")
        };
        assert_eq!(example.facts, expected);
        assert_eq!(example.decode_mode, DecodeMode::Cpu);
        assert_eq!(example.optimizer_profile, OptimizerProfile::Optimized);
        assert!(!example.vignette_correction);
        assert!(example.command.contains("-video_size 4096x2160"));
        assert!(example.command.contains("-framerate 30000/1001"));
        assert!(
            example
                .command
                .contains("--cpu --optimized --no-vig-correction")
        );
    }

    #[test]
    fn injected_facts_provider_error_is_reported() {
        let panel = example_for_selected_file_with_facts(
            Some(Path::new("clip.mcraw")),
            DecodeMode::Gpu,
            OptimizerProfile::Default,
            true,
            |_| Err(PipeExampleError::new("synthetic facts failure")),
        );

        assert_eq!(
            panel,
            PipeExamplePanel::Error(
                "Could not read PIPE example facts: synthetic facts failure".to_string()
            )
        );
    }

    #[test]
    fn no_selection_does_not_call_facts_provider() {
        let panel = example_for_selected_file_with_facts(
            None,
            DecodeMode::Gpu,
            OptimizerProfile::Default,
            true,
            |_| panic!("facts provider must not run without a selected file"),
        );

        assert_eq!(
            panel,
            PipeExamplePanel::Message(NO_SELECTED_FILE_MESSAGE.to_string())
        );
    }

    #[test]
    fn requested_gui_text_and_platform_notices_are_exact() {
        assert_eq!(HEADER, "mcraw4vulkan Pipe");
        assert_eq!(
            BODY,
            "Using the command line, mcraw4vulkan Pipe decodes a MotionCam RAW file directly into a 12 bit yuv444p12le bytestream with BT.2020 primaries and a linear transfer coefficient. Audio and metadata sidecars are also created."
        );
        assert_eq!(
            USEFUL_TEXT,
            "The bytestream is formatted for rapid processing in the GPU. It is intentionally not display-ready and may appear dark until adjusted in an editing app."
        );
        assert_eq!(
            NO_SELECTED_FILE_MESSAGE,
            "Select a playlist file to generate a pipe example."
        );
        assert_eq!(
            PRORES_OUTPUT_COLOR_TEXT,
            "On macOS, ProRes hardware encoding with VideoToolbox requires the bytestream to first be converted to 10-bit 444. This is still high-quality. For full-quality lossless video, use the DNG method instead."
        );
        assert_eq!(
            VULKAN_FFMPEG_8_1_NOTICE,
            "FFmpeg 8.1 or later is required for GPU encoding of ProRes files with Vulkan"
        );
        assert_eq!(
            SIMPLE_COMMAND,
            "mcraw4vulkan pipe FILE_NAME.mcraw > FILE_NAME.yuv444p12le"
        );
        assert_eq!(COMPLICATED_EXAMPLE_LABEL, "A complicated example:");
        assert_eq!(
            COPY_COMMAND_TEXT,
            "Copy and Paste this whole command into the terminal"
        );
        assert_eq!(
            CommandTarget::Linux.terminal_label(),
            "A simple example using the terminal to create a video bytestream"
        );
        assert_eq!(
            CommandTarget::Windows.terminal_label(),
            "A simple example using Windows PowerShell to create a video bytestream"
        );
        assert_eq!(
            CommandTarget::Macos.terminal_label(),
            "A simple example using MacOS Terminal to create a video bytestream"
        );
        assert_eq!(
            CommandTarget::Linux.ffmpeg_prerequisite_text(),
            Some(VULKAN_FFMPEG_8_1_NOTICE)
        );
        assert_eq!(
            CommandTarget::Windows.ffmpeg_prerequisite_text(),
            Some(VULKAN_FFMPEG_8_1_NOTICE)
        );
        assert_eq!(
            CommandTarget::Macos.ffmpeg_prerequisite_text(),
            Some(PRORES_OUTPUT_COLOR_TEXT)
        );
    }
}
