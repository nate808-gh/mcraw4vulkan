//! Project-only preparation through the shared export scheduler and child owner.
//! Gyroflow owns GCSV interpretation, synchronization and stabilization math.
use super::*;
use serde_json::{Value, json};

pub const MISSING_GCSV: &str = "File skipped. GCSV file not found";

#[derive(Debug, Clone)]
pub struct ProjectRequest {
    pub inputs: ExportInputs,
    pub lens_profile: PathBuf,
    pub rolling_shutter_ms: f64,
    pub producer: ProducerLocation,
    pub settings: SettingsSourceChoice,
    pub ffmpeg: Option<PathBuf>,
}

impl ProjectRequest {
    fn export_options(&self) -> ExportOptions {
        ExportOptions {
            backend: ComputeBackendChoice::Gpu,
            settings: self.settings,
            vignette: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectStage {
    Checking,
    Encoding,
    Preparing,
    Creating,
    Created,
}

#[derive(Debug, Clone)]
pub struct ProjectEvent {
    pub row: usize,
    pub stage: ProjectStage,
}

pub(super) fn event(snapshot: &mut ExportSnapshot, row: usize, stage: ProjectStage) {
    snapshot.project_events.push(ProjectEvent { row, stage });
}

#[derive(Debug)]
pub(super) struct ProjectJob {
    pub destination: PathBuf,
    motion: PathBuf,
}

#[derive(Clone)]
pub(super) struct ProjectSettings {
    lens: Value,
    readout: f64,
    ffprobe: PathBuf,
    gyroflow: PathBuf,
}

pub fn validate_readout(value: f64) -> Result<()> {
    ensure!(
        value.is_finite() && value >= 0.0,
        "--rolling-shutter must be a finite, nonnegative number of milliseconds"
    );
    Ok(())
}

pub fn sibling_motion(source: &Path) -> PathBuf {
    source.with_extension("gcsv")
}

fn readable_motion(source: &Path) -> Result<PathBuf> {
    let path = sibling_motion(source);
    let metadata = match fs::metadata(&path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => bail!(MISSING_GCSV),
        result => result.with_context(|| format!("cannot inspect GCSV {}", path.display()))?,
    };
    ensure!(
        metadata.is_file(),
        "GCSV is not a regular file: {}",
        path.display()
    );
    ensure!(metadata.len() > 0, "GCSV is empty: {}", path.display());
    let file =
        fs::File::open(&path).with_context(|| format!("GCSV is unreadable: {}", path.display()))?;
    let mut header = Vec::new();
    file.take(65536).read_to_end(&mut header)?;
    // A bounded format envelope check, never a telemetry decoder. The standard
    // reader must still produce real motion, offsets and processed data below.
    let header = std::str::from_utf8(&header).context("GCSV header is not UTF-8")?;
    let mut lines = header.lines();
    let signature = lines.next() == Some("GYROFLOW IMU LOG");
    let columns = lines.find(|s| s.starts_with("t,gx,gy,gz"));
    ensure!(
        signature && columns.is_some(),
        "invalid GCSV header: {}",
        path.display()
    );
    ensure!(
        lines.any(|s| !s.trim().is_empty()),
        "GCSV has no sample rows: {}",
        path.display()
    );
    Ok(path)
}

fn read_json(path: &Path, limit: u64) -> Result<Value> {
    let file = fs::File::open(path).with_context(|| format!("cannot read {}", path.display()))?;
    ensure!(
        file.metadata()?.is_file() && file.metadata()?.len() <= limit,
        "JSON is not a regular file within the {limit}-byte limit: {}",
        path.display()
    );
    serde_json::from_reader(file).with_context(|| format!("invalid JSON: {}", path.display()))
}

fn output_available(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e).context("cannot inspect project destination"),
        Ok(_) => bail!("output already exists: {}", path.display()),
    }
}

// LensProfile stores an optional enum; stabilization exports a concrete enum.
// Null/absent is no direction request, not a request to export JSON null. At
// zero readout there is no row/column time offset; keep per-frame correction.
fn requested_readout_direction(lens: &Value, readout: f64) -> Result<Option<&str>> {
    let direction = match lens.get("frame_readout_direction") {
        None | Some(Value::Null) => None,
        Some(Value::String(value))
            if matches!(
                value.as_str(),
                "TopToBottom" | "BottomToTop" | "LeftToRight" | "RightToLeft"
            ) =>
        {
            Some(value.as_str())
        }
        _ => bail!(
            "invalid lens-profile frame_readout_direction: expected null, absent, or TopToBottom, BottomToTop, LeftToRight, RightToLeft"
        ),
    };
    ensure!(
        readout == 0.0 || direction.is_some(),
        "positive readout requires a valid lens-profile readout direction"
    );
    Ok(direction)
}

fn check_geometry(facts: &PipeExampleFacts, lens: &Value, readout: f64) -> Result<()> {
    ensure!(
        facts.width % 2 == 0 && facts.height % 2 == 0,
        "GPU Main10 4:2:0 requires an even raster; no crop or padding will be applied"
    );
    validate_readout(readout)?;
    // No guessed mapping from calibration coordinates to a recovered/cropped raster.
    let dimension = lens
        .get("orig_dimension")
        .or_else(|| lens.get("calib_dimension"))
        .context("lens profile has no calibration dimensions")?;
    let same_raster = number_is(&dimension["w"], facts.width as f64, 0.0)
        && number_is(&dimension["h"], facts.height as f64, 0.0);
    // Gyroflow's compatible_settings explicitly declares alternate capture modes.
    // Permit its uniform-scale mapping only when that declaration changes no
    // crop, stretch, lens or readout parameters. Aspect ratio alone proves nothing.
    let declared_scale = lens["compatible_settings"].as_array().is_some_and(|modes| {
        modes.iter().any(|mode| {
            mode.as_object().is_some_and(|o| {
                o.keys()
                    .all(|k| matches!(k.as_str(), "width" | "height" | "fps"))
            }) && number_is(&mode["width"], facts.width as f64, 0.0)
                && number_is(&mode["height"], facts.height as f64, 0.0)
                && dimensions_equal(dimension, &lens["calib_dimension"])
                && dimension["w"]
                    .as_f64()
                    .zip(dimension["h"].as_f64())
                    .is_some_and(|(w, h)| {
                        w > 0.0
                            && h > 0.0
                            && (facts.width as f64 / w - facts.height as f64 / h).abs() < 1e-12
                    })
        })
    });
    ensure!(
        same_raster || declared_scale,
        "lens/source raster mapping is not established: use a matching calibration or an explicit uniform-scale compatible_settings mode; crop/rotation cannot be inferred"
    );
    ensure!(
        facts.sample_aspect_ratio == crate::PipeAspectRatio::square_pixels(),
        "analysis requires the existing square-pixel PIPE geometry"
    );
    requested_readout_direction(lens, readout)?;
    Ok(())
}

pub(super) fn run_batch(
    request: ProjectRequest,
    cancel: Arc<AtomicBool>,
    state: Arc<Mutex<ExportSnapshot>>,
) -> ExportResult {
    let mut result = ExportResult::default();
    let mut pending = VecDeque::new();
    let setup = (|| -> Result<RunnerConfig> {
        validate_readout(request.rolling_shutter_ms)?;
        let cwd = std::env::current_dir()?;
        let profile = absolute(&request.lens_profile, &cwd);
        ensure!(
            profile
                .extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("json")),
            "--lensprofile must be a JSON file"
        );
        let lens = read_json(&profile, 4 * 1024 * 1024)?;
        ensure!(
            lens.is_object() && lens["fisheye_params"].is_object(),
            "lens JSON has no fisheye calibration"
        );
        let dest = absolute(
            &crate::dng_mount::default_dng_shared_mount_root()?.join("gyroflow_projects"),
            &cwd,
        );
        result.snapshot.destination = dest.clone();
        let options = request.export_options();
        let paths = discover(request.inputs, &cancel)?;
        result.snapshot.total = paths.len();
        ensure!(!paths.is_empty(), "no direct .mcraw files to prepare");
        let mut seen = std::collections::HashSet::new();
        let mut names = HashMap::<String, usize>::new();
        for path in &paths {
            if let Some(stem) = path.file_stem() {
                *names
                    .entry(stem.to_string_lossy().to_uppercase())
                    .or_default() += 1;
            }
        }
        for path in paths {
            let source = absolute(&path, &cwd);
            let row = result.add_source(source.clone());
            event(&mut result.snapshot, row, ProjectStage::Checking);
            *lock(&state) = result.snapshot.clone();
            let admission = (|| -> Result<ProjectJob> {
                ensure!(!cancel.load(Ordering::Relaxed), "Export canceled.");
                ensure!(
                    is_mcraw(&source) && fs::metadata(&source)?.is_file(),
                    "source must be a regular .mcraw file"
                );
                let motion = readable_motion(&source)?;
                let stem = source
                    .file_stem()
                    .filter(|s| !s.is_empty())
                    .context("source has no filename stem")?;
                ensure!(
                    source.to_str().is_some(),
                    "producer CLI requires a UTF-8 source path"
                );
                ensure!(
                    seen.insert(source.canonicalize()?),
                    "Duplicate source selection."
                );
                ensure!(
                    names[&stem.to_string_lossy().to_uppercase()] == 1,
                    "distinct sources have colliding output stems"
                );
                let mut name = stem.to_os_string();
                name.push(".gyroflow");
                let destination = dest.join(name);
                output_available(&destination)?;
                Ok(ProjectJob {
                    destination,
                    motion,
                })
            })();
            match admission {
                Ok(project) => pending.push_back((row, source, project)),
                Err(e) => {
                    let reason = format!("{e:#}");
                    let outcome = if reason == MISSING_GCSV
                        || reason.starts_with("output already exists")
                        || reason.contains("colliding")
                    {
                        Outcome::Skipped
                    } else {
                        Outcome::Failed
                    };
                    result.record(row, outcome, reason);
                }
            }
        }
        *lock(&state) = result.snapshot.clone();
        ensure!(!pending.is_empty(), "no eligible clips to prepare");
        // These optional dependencies are queried only after cheap admission.
        let gyroflow = resolve_gyroflow(&cwd, &cancel)?;
        let producer = producer_path(request.producer)?;
        let ffmpeg = discover_ffmpeg(request.ffmpeg.as_deref(), &cwd)?;
        let ffprobe = select_ffprobe(
            &ffmpeg,
            &producer,
            std::env::var_os("PATH").as_deref(),
            &cancel,
        )?;
        check_hevc(&ffmpeg, &producer, &cancel)?;
        prepare_destination(&dest)?;
        Ok(RunnerConfig {
            producer,
            ffmpeg,
            encoder: VideoEncoding::AnalysisHevc,
            project: Some(ProjectSettings {
                lens,
                readout: request.rolling_shutter_ms,
                ffprobe,
                gyroflow,
            }),
            cancel: Arc::clone(&cancel),
            cwd,
            options,
            overlap_remux: false,
        })
    })();
    let config = match setup {
        Ok(config) => config,
        Err(error) => {
            result.problem = Some(format!("{error:#}"));
            for (row, _, _) in pending {
                result.record(
                    row,
                    Outcome::Failed,
                    "Project preparation did not start.".into(),
                );
            }
            result.cancellation_requested = cancel.load(Ordering::Relaxed);
            *lock(&state) = result.snapshot.clone();
            return result;
        }
    };
    let mut jobs = VecDeque::new();
    for (row, source, project) in pending {
        let prepared = (|| -> Result<Job> {
            let facts = crate::pipe_cli::pipe_export_facts_for_input(&source, || {
                cancel.load(Ordering::Relaxed)
            })?;
            let settings = config.project.as_ref().unwrap();
            check_geometry(&facts.facts, &settings.lens, settings.readout)?;
            Ok(Job {
                output: JobOutput::Project(project),
                source,
                row,
                facts: facts.facts,
                frames: facts.frames,
                has_audio: facts.has_audio,
                geometry_handoff: facts.geometry_handoff,
            })
        })();
        match prepared {
            Ok(job) => jobs.push_back(job),
            Err(e) => result.record(row, Outcome::Failed, format!("{e:#}")),
        }
    }
    run_schedule(config, jobs, result, cancel, state)
}

// The stock macOS CLI permits its Application Support subtree. Allocate the
// original PIPE job there; publication still uses ProjectJob::destination.
#[cfg(target_os = "macos")]
pub(super) fn owned_job_directory() -> Result<PathBuf> {
    let home = crate::dng_mount::default_dng_home_env().context("HOME is not set")?;
    owned_job_directory_from_home(Path::new(&home), &std::env::current_dir()?)
}

#[cfg(target_os = "macos")]
fn owned_job_directory_from_home(home: &Path, cwd: &Path) -> Result<PathBuf> {
    ensure!(!home.as_os_str().is_empty(), "HOME is empty");
    let root = absolute(home, cwd).join("Library/Application Support/Gyroflow");
    ensure!(root.is_absolute(), "Gyroflow scratch root must be absolute");
    prepare_destination(&root)?;
    owned_directory(&root)
}

// Gyroflow's sandboxed output_path handler cannot infer an unbookmarked folder.
// Encode native path bytes once for its explicit output_folder file URL.
#[cfg(target_os = "macos")]
fn output_directory_url(directory: &Path) -> String {
    use std::os::unix::ffi::OsStrExt;

    // The allocator establishes this invariant before starting either producer.
    assert!(
        directory.is_absolute(),
        "Gyroflow output directory must be absolute"
    );
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut url = String::from("file://");
    for &byte in directory.as_os_str().as_bytes() {
        if byte.is_ascii_alphanumeric() || b"/-._~".contains(&byte) {
            url.push(char::from(byte));
        } else {
            url.push('%');
            url.push(char::from(HEX[(byte >> 4) as usize]));
            url.push(char::from(HEX[(byte & 15) as usize]));
        }
    }
    if !url.ends_with('/') {
        url.push('/');
    }
    url
}

fn gyroflow_command(executable: &Path) -> Command {
    let mut command = Command::new(executable);
    configure_process(&mut command);
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    command
}

// A candidate may be rejected only after its child and drains are accounted for.
// Keep this distinction local; FFmpeg/FFprobe selection is unchanged.
#[derive(Debug)]
struct CheckOwnershipFailure;

impl std::fmt::Display for CheckOwnershipFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("external CLI check ownership or teardown failed")
    }
}

fn checked(mut command: Command, cancel: &AtomicBool, timeout: Duration) -> Result<String> {
    ensure!(!cancel.load(Ordering::Relaxed), "Export canceled.");
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let process = command.spawn().context("could not launch external CLI")?;
    drop(command);
    let mut process =
        ExportProcess::capture(process, true, false).context(CheckOwnershipFailure)?;
    let start = Instant::now();
    loop {
        if cancel.load(Ordering::Relaxed) || start.elapsed() >= timeout {
            process.finish(true).context(CheckOwnershipFailure)?;
            if cancel.load(Ordering::Relaxed) {
                bail!("Export canceled.");
            }
            bail!("external CLI check timed out");
        }
        let status = match process.poll() {
            Ok(status) => status,
            Err(error) => {
                process
                    .finish(true)
                    .context(format!("{error:#}; external CLI check teardown failed"))
                    .context(CheckOwnershipFailure)?;
                return Err(error);
            }
        };
        if let Some(status) = status {
            process.finish(false).context(CheckOwnershipFailure)?;
            ensure!(
                status.success(),
                "external CLI failed ({status}): {}",
                process.diagnostics()
            );
            let stdout = process.stdout.as_ref().unwrap();
            ensure!(
                lock(&stdout.state).bytes <= LOG_LIMIT,
                "external CLI response exceeded its bound"
            );
            return Ok(stdout.text());
        }
        thread::sleep(POLL);
    }
}

fn check_gyroflow(executable: &Path, cancel: &AtomicBool) -> Result<()> {
    let mut command = gyroflow_command(executable);
    command.arg("--help");
    let help = checked(command, cancel, Duration::from_secs(15))?;
    let missing: Vec<_> = [
        "--export-project",
        "--optical-per-frame",
        "--gyro-file",
        "--when-done",
    ]
    .into_iter()
    .filter(|flag| !help.contains(*flag))
    .collect();
    ensure!(
        missing.is_empty(),
        "Gyroflow help lacks required CLI capabilities: {}",
        missing.join(", ")
    );
    Ok(())
}

fn gyroflow_path_candidates(
    target: CommandTarget,
    path: Option<&OsStr>,
    cwd: &Path,
) -> Vec<PathBuf> {
    let names: &[&str] = match target {
        CommandTarget::Windows => &["Gyroflow.exe"],
        // Preserve an existing working lowercase command before trying the
        // uppercase name used by Linux manual installations.
        CommandTarget::Linux => &["gyroflow", "Gyroflow"],
        CommandTarget::Macos => &["gyroflow"],
    };
    let mut candidates = Vec::new();
    if let Some(path) = path {
        for name in names {
            candidates
                .extend(std::env::split_paths(path).map(|dir| absolute(&dir, cwd).join(name)));
        }
    }
    candidates
}

fn append_gyroflow_installations(
    target: CommandTarget,
    candidates: &mut Vec<PathBuf>,
    home: Option<&Path>,
    program_files: Option<&Path>,
    local_app_data: Option<&Path>,
) {
    let home = home.filter(|path| path.is_absolute());
    match target {
        CommandTarget::Macos => {
            candidates.push(PathBuf::from(
                "/Applications/Gyroflow.app/Contents/MacOS/gyroflow",
            ));
            if let Some(home) = home {
                candidates.push(home.join("Applications/Gyroflow.app/Contents/MacOS/gyroflow"));
                candidates.push(home.join("Desktop/Gyroflow.app/Contents/MacOS/gyroflow"));
            }
        }
        CommandTarget::Windows => {
            if let Some(root) = program_files.filter(|path| path.is_absolute()) {
                candidates.push(root.join("Gyroflow/Gyroflow.exe"));
            }
            if let Some(root) = local_app_data.filter(|path| path.is_absolute()) {
                candidates.push(root.join("Programs/Gyroflow/Gyroflow.exe"));
            }
            // Exact conventional Desktop location, not a folder scan or a
            // claim to support redirected/OneDrive Desktops automatically.
            if let Some(home) = home {
                candidates.push(home.join("Desktop/Gyroflow/Gyroflow.exe"));
            }
        }
        CommandTarget::Linux => {}
    }
}

fn resolve_gyroflow(cwd: &Path, cancel: &AtomicBool) -> Result<PathBuf> {
    let target = pipe_example::current_command_target();
    let path = std::env::var_os("PATH");
    let home = crate::dng_mount::default_dng_home_env().map(PathBuf::from);
    let program_files = std::env::var_os("ProgramFiles").map(PathBuf::from);
    let local_app_data = std::env::var_os("LOCALAPPDATA").map(PathBuf::from);
    let mut candidates = gyroflow_path_candidates(target, path.as_deref(), cwd);
    append_gyroflow_installations(
        target,
        &mut candidates,
        home.as_deref(),
        program_files.as_deref(),
        local_app_data.as_deref(),
    );
    resolve_gyroflow_candidates(candidates, cwd, cancel)
}

fn resolve_gyroflow_candidates(
    candidates: Vec<PathBuf>,
    cwd: &Path,
    cancel: &AtomicBool,
) -> Result<PathBuf> {
    let mut seen = std::collections::HashSet::new();
    let mut failures = String::new();
    for candidate in candidates {
        ensure!(!cancel.load(Ordering::Relaxed), "Export canceled.");
        let candidate = absolute(&candidate, cwd);
        // Canonicalize only the duplicate identity; invoke the selected absolute
        // pathname so ordinary symlink-based launchers retain their location.
        if !seen.insert(
            candidate
                .canonicalize()
                .unwrap_or_else(|_| candidate.clone()),
        ) {
            continue;
        }
        let result = match fs::metadata(&candidate) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => Err(error.into()),
            Ok(metadata) if !metadata.is_file() => Err(anyhow!("not a regular executable file")),
            Ok(_) => check_gyroflow(&candidate, cancel),
        };
        match result {
            Ok(()) => {
                ensure!(!cancel.load(Ordering::Relaxed), "Export canceled.");
                if !failures.is_empty() {
                    tracing::debug!("Earlier Gyroflow candidates rejected:\n{failures}");
                }
                tracing::info!("Using Gyroflow: {}", candidate.display());
                return Ok(candidate);
            }
            Err(error) if error.downcast_ref::<CheckOwnershipFailure>().is_some() => {
                return Err(error).with_context(|| {
                    format!(
                        "Gyroflow check could not finish safely: {}",
                        candidate.display()
                    )
                });
            }
            Err(error) => {
                ensure!(!cancel.load(Ordering::Relaxed), "Export canceled.");
                let reason = format!("{}: {error:#}\n", candidate.display());
                let remaining = LOG_LIMIT.saturating_sub(failures.len());
                if reason.len() <= remaining {
                    failures.push_str(&reason);
                } else if remaining > 4 {
                    failures.push_str(&truncate_text(&reason, remaining - 4));
                    failures.push_str("...\n");
                }
            }
        }
    }
    ensure!(!cancel.load(Ordering::Relaxed), "Export canceled.");
    if !failures.is_empty() {
        bail!("Gyroflow CLI candidates were found but none was usable:\n{failures}");
    }
    let guidance = match pipe_example::current_command_target() {
        CommandTarget::Macos => {
            "Place Gyroflow.app in Applications or your home Desktop, or make its CLI available on PATH."
        }
        CommandTarget::Windows => {
            "Place the Gyroflow folder in Program Files, LocalAppData\\Programs or your home Desktop, or make Gyroflow.exe available on PATH."
        }
        CommandTarget::Linux => "Make gyroflow or Gyroflow available on PATH.",
    };
    bail!("Gyroflow CLI was not found.\n{guidance}")
}

fn select_ffprobe(
    ffmpeg: &Path,
    producer: &Path,
    path: Option<&std::ffi::OsStr>,
    cancel: &AtomicBool,
) -> Result<PathBuf> {
    let name = if cfg!(windows) {
        "ffprobe.exe"
    } else {
        "ffprobe"
    };
    let mut candidates = vec![ffmpeg.with_file_name(name)];
    if let Some(path) = path {
        candidates.extend(std::env::split_paths(path).map(|p| p.join(name)));
    }
    let mut failures = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for candidate in candidates {
        if !seen.insert(candidate.clone()) {
            continue;
        }
        match fs::metadata(&candidate) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => {
                failures.push(format!("{}: {e}", candidate.display()));
                continue;
            }
            Ok(meta) if !meta.is_file() => continue,
            Ok(_) => {}
        }
        let mut command = ffmpeg_command(&candidate, producer);
        command.arg("-version");
        match checked(command, cancel, Duration::from_secs(10)) {
            Ok(text) if text.starts_with("ffprobe version ") => {
                let selected = absolute(&candidate, &std::env::current_dir()?);
                tracing::info!(path = %selected.display(), "Gyroflow proxy FFprobe selected");
                return Ok(selected);
            }
            Ok(_) => failures.push(format!(
                "{}: does not identify itself as FFprobe",
                candidate.display()
            )),
            Err(e) => failures.push(format!("{}: {e:#}", candidate.display())),
        }
        ensure!(!cancel.load(Ordering::Relaxed), "Export canceled.");
    }
    let reason = if failures.is_empty() {
        "FFprobe not found".to_owned()
    } else {
        format!("FFprobe found but not usable: {}", failures.join("; "))
    };
    bail!(
        "{reason}. Gyroflow project validation requires FFprobe executable beside the selected FFmpeg or through PATH; verify ffprobe -version."
    )
}

fn number_is(value: &Value, expected: f64, tolerance: f64) -> bool {
    value
        .as_f64()
        .is_some_and(|n| n.is_finite() && (n - expected).abs() <= tolerance)
}

fn dimensions_equal(a: &Value, b: &Value) -> bool {
    ["w", "h"].iter().all(|k| {
        b[k].as_f64()
            .is_some_and(|n| n > 0.0 && number_is(&a[k], n, 0.0))
    })
}

fn rational(value: &Value, separator: char) -> Option<(u64, u64)> {
    let (n, d) = value.as_str()?.split_once(separator)?;
    let (n, d) = (n.parse().ok()?, d.parse().ok()?);
    (d != 0).then_some((n, d))
}

fn rational_is(value: &Value, separator: char, n: u64, d: u64) -> bool {
    rational(value, separator).is_some_and(|(a, b)| a as u128 * d as u128 == n as u128 * b as u128)
}

fn check_hevc(ffmpeg: &Path, producer: &Path, cancel: &AtomicBool) -> Result<()> {
    let facts = PipeExampleFacts {
        // Above Vulkan Video's observed 160-pixel minimum width; this is
        // synthetic readiness media, never a resize of the source recording.
        width: 256,
        height: 128,
        cadence: crate::PipeMovCadence::from_source_rate(30, 1)?,
        sample_aspect_ratio: crate::PipeAspectRatio::square_pixels(),
        display_aspect_ratio: crate::PipeAspectRatio::new(2, 1)?,
    };
    let mut args = pipe_example::analysis_hevc_args(&facts, pipe_example::current_command_target());
    let input = args.iter().position(|s| s == "-f").unwrap();
    let map = args.iter().position(|s| s == "-map").unwrap();
    args.splice(
        input..map,
        ["-f", "lavfi", "-i", "color=size=256x128:rate=30"].map(str::to_owned),
    );
    args.extend(["-frames:v", "2", "-f", "null", "-"].map(str::to_owned));
    let mut command = ffmpeg_command(ffmpeg, producer);
    command.args(args);
    checked(command, cancel, Duration::from_secs(30)).context("GPU Main10 HEVC capability check failed; this project requires hevc_vulkan (Windows/Linux) or hardware hevc_videotoolbox (macOS)")?;
    Ok(())
}

fn validate_proxy(
    files: &JobFiles,
    settings: &ProjectSettings,
    config: &RunnerConfig,
) -> Result<()> {
    let mut command = ffmpeg_command(&settings.ffprobe, &config.producer);
    command
        .args(["-v", "error", "-show_streams", "-of", "json"])
        .arg(&files.video);
    let text = checked(command, &config.cancel, Duration::from_secs(20)).with_context(|| {
        format!(
            "FFprobe {} could not inspect this proxy",
            settings.ffprobe.display()
        )
    })?;
    let value: Value =
        serde_json::from_str(&text).context("FFprobe returned invalid inspection JSON")?;
    validate_proxy_streams(&value, files)
        .context("FFprobe inspection found incompatible proxy media")?;
    tracing::info!(
        frames = files.job.frames,
        proxy_bytes = fs::metadata(&files.video)?.len(),
        fps_num = files.job.facts.cadence.fps_num,
        fps_den = files.job.facts.cadence.fps_den,
        "Gyroflow proxy verified"
    );
    Ok(())
}

fn validate_proxy_streams(value: &Value, files: &JobFiles) -> Result<()> {
    let streams = value["streams"]
        .as_array()
        .context("ffprobe has no streams")?;
    ensure!(
        streams.len() == 1,
        "analysis proxy must contain one silent video stream"
    );
    let s = &streams[0];
    let f = &files.job.facts;
    let c = &f.cadence;
    ensure!(
        s["codec_name"] == "hevc" && s["profile"] == "Main 10" && s["pix_fmt"] == "yuv420p10le",
        "proxy is not Main10 HEVC / 10-bit 4:2:0"
    );
    ensure!(
        number_is(&s["width"], f.width as f64, 0.0)
            && number_is(&s["height"], f.height as f64, 0.0)
            && rational_is(&s["sample_aspect_ratio"], ':', 1, 1),
        "proxy raster/aspect mismatch"
    );
    ensure!(
        s["nb_frames"].as_str().and_then(|s| s.parse::<u64>().ok()) == Some(files.job.frames),
        "proxy frame count differs from the producer"
    );
    ensure!(
        rational_is(
            &s["avg_frame_rate"],
            '/',
            c.fps_num as u64,
            c.fps_den as u64
        ) && rational_is(&s["time_base"], '/', 1, c.video_track_timescale as u64)
            && number_is(&s["start_pts"], 0.0, 0.0),
        "proxy cadence/time origin mismatch"
    );
    let expected_ticks =
        (files.job.frames as u128 * c.fps_den as u128 * c.video_track_timescale as u128)
            / c.fps_num as u128;
    ensure!(
        s["duration_ts"].as_u64().map(u128::from) == Some(expected_ticks),
        "proxy duration mismatch"
    );
    ensure!(
        s["side_data_list"]
            .as_array()
            .is_none_or(|a| a.iter().all(|v| v["rotation"].as_i64().unwrap_or(0) == 0)),
        "unexpected proxy rotation"
    );
    Ok(())
}

fn preset(files: &JobFiles, settings: &ProjectSettings) -> Result<Value> {
    let mut value = json!({
        "version":4,
        "calibration_data":settings.lens, "offsets":{}, "keyframes":{}, "trim_ranges_ms":[],
        "gyro_source":{"ignore_file_motion":false,"integration_method":2,"lpf":0.0,"mf":0,"rotation":null,"acc_rotation":null,"gyro_bias":null,"optical_removed_tracks":[]},
        "stabilization":{
            "frame_readout_time":settings.readout, "method":"Default",
            "smoothing_params":[{"name":"smoothness","value":0.5},{"name":"smoothness_pitch","value":0.5},{"name":"smoothness_yaw","value":0.5},{"name":"smoothness_roll","value":0.5},{"name":"per_axis","value":0.0},{"name":"trim_range_only","value":1.0},{"name":"max_smoothness","value":1.0},{"name":"alpha_0_1s","value":0.1}],
            "fov":1.0,"adaptive_zoom_window":4.0,"adaptive_zoom_center_offset":[0,0],"adaptive_zoom_method":1,"max_zoom":130.0,"max_zoom_iterations":5,"fixed_zoom":false,"zoom_limit_whole_clip":false,"lens_correction_amount":1.0,"output_projection":"rectilinear","horizon_lock_amount":0.0,"horizon_lock_roll":0.0,"video_speed":1.0,"frame_offset":0,"additional_rotation":[0,0,0],"additional_translation":[0,0,0]
        },
        "output":{"output_width":files.job.facts.width,"output_height":files.job.facts.height}
    });
    if let Some(direction) = requested_readout_direction(&settings.lens, settings.readout)? {
        value["stabilization"]["frame_readout_direction"] = json!(direction);
    }
    // Pin the project filename so saved defaults cannot redirect this job.
    let mut name = files.job.source.file_stem().unwrap().to_os_string();
    name.push("_stabilized.mp4");
    #[cfg(target_os = "macos")]
    {
        value["output"]["output_folder"] = json!(output_directory_url(&files.directory));
        value["output"]["output_filename"] = json!(Path::new(&name));
    }
    #[cfg(not(target_os = "macos"))]
    {
        value["output"]["output_path"] = json!(files.directory.join(name));
    }
    value["stabilization"]["automatic_lock"] = json!(false);
    value["stabilization"]["horizon_lock_pitch_enabled"] = json!(false);
    value["stabilization"]["use_gravity_vectors"] = json!(false);
    Ok(value)
}

fn sync_params(readout: f64) -> Value {
    // CLI setup_defaults merges these overrides with its stable sync defaults.
    // A synchronization object in the preset would REPLACE that merged object.
    // Explicit false is necessary: absence must not inherit per-frame correction.
    // Use Gyroflow's evenly spaced windows. Its experimental motion-ranked
    // placement can select windows with no usable optical tracks.
    json!({"do_autosync":true,"auto_sync_points":false,"ignore_file_motion":false,
        "optical_correction_per_frame":readout == 0.0,"processing_resolution":720})
}

pub(super) fn start(
    files: &JobFiles,
    settings: &ProjectSettings,
    config: &RunnerConfig,
) -> Result<ExportProcess> {
    validate_proxy(files, settings, config)?;
    let project = files.job.project().unwrap();
    // Recheck the exact sibling before copying; never substitute a neighbor.
    readable_motion(&files.job.source)?;
    fs::copy(&project.motion, files.directory.join("motion.gcsv"))?;
    fs::write(
        files.directory.join("lens.json"),
        serde_json::to_vec(&settings.lens)?,
    )?;
    fs::write(
        files.directory.join("settings.gyroflow"),
        serde_json::to_vec(&preset(files, settings)?)?,
    )?;
    let mut command = project_command(files, settings)?;
    let child = command.spawn().with_context(|| {
        format!(
            "cannot launch selected Gyroflow CLI for project preparation: {}",
            settings.gyroflow.display()
        )
    })?;
    drop(command);
    ExportProcess::capture(child, true, false)
}

fn project_command(files: &JobFiles, settings: &ProjectSettings) -> Result<Command> {
    let mut command = gyroflow_command(&settings.gyroflow);
    command
        .arg(&files.video)
        .arg(files.directory.join("lens.json"))
        .arg("-g")
        .arg(files.directory.join("motion.gcsv"))
        .arg("--preset")
        .arg(files.directory.join("settings.gyroflow"))
        .arg("--sync-params")
        .arg(serde_json::to_string(&sync_params(settings.readout))?)
        .args([
            "--parallel-renders",
            "1",
            "--export-project",
            "3",
            "--when-done",
            "0",
            "--suffix",
            "_stabilized",
            "--stdout-progress",
        ]);
    if settings.readout == 0.0 {
        command.arg("--optical-per-frame");
    }
    command.args(["--optical-correction", "50", "--optical-tracking", "klt"]);
    command.current_dir(&files.directory);
    Ok(command)
}

fn validate_calibration(actual: &Value, supplied: &Value) -> Result<()> {
    // Metadata (name, RMS error, derived radial limit, author, sync options) is
    // not lens geometry. Compare the calibrated transform, not its serialization.
    let a = &actual["calib_dimension"];
    let b = &supplied["calib_dimension"];
    for key in ["w", "h"] {
        ensure!(
            a[key].as_f64().is_some_and(|v| v > 0.0) && b[key].as_f64().is_some_and(|v| v > 0.0),
            "missing lens calibration dimensions"
        );
    }
    for row in 0..3 {
        for col in 0..3 {
            let divisor = |d: &Value| {
                if row == 0 {
                    d["w"].as_f64().unwrap()
                } else if row == 1 {
                    d["h"].as_f64().unwrap()
                } else {
                    1.0
                }
            };
            let expected = supplied["fisheye_params"]["camera_matrix"][row][col]
                .as_f64()
                .context("supplied lens camera matrix is incomplete")?
                / divisor(b);
            let measured = actual["fisheye_params"]["camera_matrix"][row][col]
                .as_f64()
                .context("project lens camera matrix is incomplete")?
                / divisor(a);
            // 1e-10 normalized coordinates is < 0.000001 pixel at an 8K raster.
            ensure!(
                measured.is_finite() && (measured - expected).abs() <= 1e-10,
                "effective lens camera matrix changed"
            );
        }
    }
    let expected = supplied["fisheye_params"]["distortion_coeffs"]
        .as_array()
        .context("supplied lens has no distortion coefficients")?;
    let measured = actual["fisheye_params"]["distortion_coeffs"]
        .as_array()
        .context("project lens has no distortion coefficients")?;
    ensure!(
        !expected.is_empty() && expected.len() == measured.len(),
        "effective lens distortion coefficients changed"
    );
    for (a, b) in measured.iter().zip(expected) {
        ensure!(
            b.as_f64().is_some_and(|v| number_is(a, v, 1e-10)),
            "effective lens distortion coefficients changed"
        );
    }
    // Other models derive this limit from their coefficients during load. GoPro
    // retains the supplied value, so changing it would change the usable field.
    if supplied["distortion_model"] == "gopro" {
        let a = &actual["fisheye_params"]["radial_distortion_limit"];
        let b = &supplied["fisheye_params"]["radial_distortion_limit"];
        ensure!(
            b.as_f64()
                .map_or(a.is_null() && b.is_null(), |v| number_is(a, v, 1e-10)),
            "effective lens radial distortion limit changed"
        );
    }
    for key in ["input_horizontal_stretch", "input_vertical_stretch"] {
        // FrameTransform uses unity for absent/disabled stretch (<= 0.01).
        let stretch = |lens: &Value| lens[key].as_f64().filter(|v| *v > 0.01).unwrap_or(1.0);
        ensure!(
            (stretch(actual) - stretch(supplied)).abs() <= 1e-10,
            "effective lens stretch changed ({key})"
        );
    }
    ensure!(
        actual["asymmetrical"].as_bool().unwrap_or(false)
            == supplied["asymmetrical"].as_bool().unwrap_or(false),
        "effective lens asymmetry changed"
    );
    ensure!(
        actual["distortion_model"]
            .as_str()
            .unwrap_or("opencv_fisheye")
            == supplied["distortion_model"]
                .as_str()
                .unwrap_or("opencv_fisheye"),
        "effective lens distortion model changed"
    );
    // These values affect the source area/model, unlike descriptive profile fields.
    for key in ["crop", "digital_lens", "interpolations"] {
        if let Some(n) = supplied[key].as_f64() {
            ensure!(
                number_is(&actual[key], n, 1e-10),
                "effective lens model changed ({key})"
            );
        } else {
            ensure!(
                actual[key] == supplied[key],
                "effective lens model changed ({key})"
            );
        }
    }
    match (
        actual["digital_lens_params"].as_array(),
        supplied["digital_lens_params"].as_array(),
    ) {
        (Some(a), Some(b)) => ensure!(
            a.len() == b.len()
                && a.iter()
                    .zip(b)
                    .all(|(a, b)| b.as_f64().is_some_and(|v| number_is(a, v, 1e-10))),
            "effective digital lens parameters changed"
        ),
        _ => ensure!(
            actual["digital_lens_params"].is_null() && supplied["digital_lens_params"].is_null(),
            "effective digital lens parameters changed"
        ),
    }
    Ok(())
}

pub(super) fn validate_project(files: &JobFiles, settings: &ProjectSettings) -> Result<()> {
    let value = read_json(&files.video.with_extension("gyroflow"), 128 * 1024 * 1024)?;
    let f = &files.job.facts;
    let video = &value["video_info"];
    // The stock reader supports this envelope across versions 2-4 (its pre-v4
    // timing exception is RED-specific, not paired GCSV). Regardless of version,
    // require the processed-data semantics below; this is not a migration claim.
    ensure!(
        [2.0, 3.0, 4.0]
            .iter()
            .any(|v| number_is(&value["version"], *v, 0.0)),
        "unsupported Gyroflow project representation: expected the known v2-v4 JSON envelope with processed-data fields; future schemas require review"
    );
    ensure!(
        number_is(&video["num_frames"], files.job.frames as f64, 0.0)
            && number_is(&video["width"], f.width as f64, 0.0)
            && number_is(&video["height"], f.height as f64, 0.0)
            && number_is(&video["rotation"], 0.0, 0.0)
            && (video["fps_scale"].is_null() || number_is(&video["fps_scale"], 1.0, 0.0)),
        "project frame count/raster mismatch"
    );
    let fps = f.cadence.fps_num as f64 / f.cadence.fps_den as f64;
    ensure!(
        video["fps"]
            .as_f64()
            .is_some_and(|v| (v - fps).abs() < 1e-8)
            && video["duration_ms"].as_f64().is_some_and(|v| (v - 1000.0
                * files.job.frames as f64
                / fps)
                .abs()
                < 0.01),
        "project cadence/duration mismatch"
    );
    check_geometry(f, &value["calibration_data"], settings.readout)?;
    validate_calibration(&value["calibration_data"], &settings.lens)?;
    let stab = &value["stabilization"];
    let gyro = &value["gyro_source"];
    let requested = preset(files, settings)?;
    for key in [
        "method",
        "fixed_zoom",
        "zoom_limit_whole_clip",
        "output_projection",
        "automatic_lock",
        "horizon_lock_pitch_enabled",
        "use_gravity_vectors",
    ] {
        ensure!(
            stab[key] == requested["stabilization"][key],
            "effective stabilization setting changed ({key})"
        );
    }
    for key in [
        "fov",
        "adaptive_zoom_window",
        "max_zoom",
        "lens_correction_amount",
        "horizon_lock_amount",
    ] {
        ensure!(
            number_is(
                &stab[key],
                requested["stabilization"][key].as_f64().unwrap(),
                1e-9
            ),
            "effective stabilization setting changed ({key})"
        );
    }
    for param in requested["stabilization"]["smoothing_params"]
        .as_array()
        .unwrap()
    {
        let actual = stab["smoothing_params"]
            .as_array()
            .and_then(|a| a.iter().find(|p| p["name"] == param["name"]));
        ensure!(
            actual.is_some_and(|p| number_is(&p["value"], param["value"].as_f64().unwrap(), 1e-9)),
            "effective smoothing parameter changed ({})",
            param["name"]
        );
    }
    ensure!(
        number_is(&stab["frame_readout_time"], settings.readout, 1e-6),
        "effective rolling-shutter time changed"
    );
    if let Some(direction) = requested["stabilization"]["frame_readout_direction"].as_str() {
        ensure!(
            stab["frame_readout_direction"].as_str() == Some(direction),
            "effective stabilization.frame_readout_direction changed: expected {direction}, actual {}",
            stab["frame_readout_direction"]
                .as_str()
                .unwrap_or("<missing or non-string>")
        );
    }
    ensure!(
        number_is(&stab["video_speed"], 1.0, 0.0)
            && number_is(&stab["frame_offset"], 0.0, 0.0)
            && value["trim_ranges_ms"]
                .as_array()
                .is_some_and(Vec::is_empty)
            && value["keyframes"]
                .as_object()
                .is_some_and(serde_json::Map::is_empty),
        "project contains unexpected retiming, trim or keyframes"
    );
    ensure!(
        gyro["ignore_file_motion"] == false
            && gyro["optical_correction_enabled"] == true
            && gyro["optical_correction_per_frame"] == (settings.readout == 0.0)
            && gyro["optical_tracking"] == "klt"
            && number_is(&gyro["optical_correction_strength"], 0.5, 1e-9),
        "project did not retain source gyro and requested optical correction"
    );
    ensure!(
        value["offsets"].as_object().is_some_and(
            |o| !o.is_empty() && o.values().all(|v| v.as_f64().is_some_and(f64::is_finite))
        ),
        "Gyroflow did not produce synchronization offsets"
    );
    // Stock mode 3's compressed schema fields. No second gyro decoder or analysis.
    // Reload/deletion and OFX acceptance are covered separately, not claimed here.
    for key in [
        "file_metadata",
        "optical_correction",
        "synced_imu_timestamps",
        "integrated_quaternions",
        "smoothed_quaternions",
        "adaptive_zoom_fovs",
    ] {
        ensure!(
            gyro[key].as_str().is_some_and(|s| s.len() >= 32),
            "project is missing embedded processed data: {key}"
        );
    }
    Ok(())
}

pub(super) fn publish(files: &JobFiles) -> Result<()> {
    let destination = &files.job.project().unwrap().destination;
    output_available(destination)?;
    crate::pipe_cli::publish_movie_no_replace(&files.video.with_extension("gyroflow"), destination)
        .context("cannot publish Gyroflow project without replacement")
}

#[cfg(windows)]
pub(super) fn guard_directory(path: &Path) -> Result<fs::File> {
    use std::os::windows::fs::{MetadataExt, OpenOptionsExt};
    // Do not follow reparse points and deny rename/deletion for the lifetime of
    // the job. NTFS can reuse creation timestamps, so they are not identity.
    let file = fs::OpenOptions::new()
        .read(true)
        .share_mode(0x1 | 0x2) // FILE_SHARE_READ | FILE_SHARE_WRITE, no DELETE
        .custom_flags(0x02000000 | 0x00200000) // BACKUP_SEMANTICS | OPEN_REPARSE_POINT
        .open(path)
        .context("cannot guard owned project directory")?;
    let meta = file.metadata()?;
    ensure!(
        meta.is_dir() && meta.file_attributes() & 0x400 == 0,
        "owned project directory is not an ordinary directory"
    );
    Ok(file)
}

pub(super) fn cleanup(files: &JobFiles) -> Result<()> {
    #[cfg(windows)]
    let mut guard = lock(&files.directory_guard);
    #[cfg(windows)]
    ensure!(
        guard.is_some(),
        "cleanup failed: owned directory identity guard was lost; inspect {}",
        files.directory.display()
    );
    let meta =
        fs::symlink_metadata(&files.directory).context("owned project directory disappeared")?;
    ensure!(
        meta.is_dir() && !meta.file_type().is_symlink(),
        "cleanup failed: owned project directory was replaced; inspect {}",
        files.directory.display()
    );
    #[cfg(windows)]
    ensure!(
        files.directory_created.is_some() && meta.created().ok() == files.directory_created,
        "cleanup failed: owned directory identity changed; inspect {}",
        files.directory.display()
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        ensure!(
            (meta.dev(), meta.ino()) == files.directory_node,
            "cleanup failed: owned directory identity changed; inspect {}",
            files.directory.display()
        );
    }
    let allowed = [
        files.video.clone(),
        files.video.with_extension("gyroflow"),
        files.wav.clone(),
        files.sidecar.clone(),
        files.directory.join("lens.json"),
        files.directory.join("motion.gcsv"),
        files.directory.join("settings.gyroflow"),
    ];
    // Refuse unknown entries, directories and links. Never recursively sweep.
    let entries = fs::read_dir(&files.directory)?.collect::<std::io::Result<Vec<_>>>()?;
    for entry in &entries {
        ensure!(
            allowed.contains(&entry.path()) && entry.file_type()?.is_file(),
            "cleanup failed: unexpected owned-directory entry {}; left for inspection",
            entry.path().display()
        );
        #[cfg(windows)]
        {
            use std::os::windows::fs::MetadataExt;
            ensure!(
                fs::symlink_metadata(entry.path())?.file_attributes() & 0x400 == 0,
                "cleanup failed: reparse entry {}; left for inspection",
                entry.path().display()
            );
        }
    }
    for entry in entries {
        fs::remove_file(entry.path())
            .with_context(|| format!("cleanup failed: {}", entry.path().display()))?;
    }
    // Release the Windows rename guard only after all validated files are gone.
    // The final operation removes an empty leaf only, never traversing contents.
    #[cfg(windows)]
    drop(guard.take());
    fs::remove_dir(&files.directory)
        .with_context(|| format!("cleanup failed: {}", files.directory.display()))
}
