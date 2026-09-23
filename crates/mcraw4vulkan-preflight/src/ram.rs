#[cfg(any(target_os = "macos", target_os = "windows"))]
use crate::child_process::configure_preflight_child_process;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SystemRamSnapshot {
    // Both facts use bytes. None preserves probe unavailability and is never
    // collapsed into a zero-capacity measurement.
    pub installed_bytes: Option<u64>,
    pub available_bytes: Option<u64>,
}

impl SystemRamSnapshot {
    pub fn unknown() -> Self {
        Self {
            installed_bytes: None,
            available_bytes: None,
        }
    }
}

pub fn collect_system_ram() -> SystemRamSnapshot {
    collect_system_ram_platform()
}

pub fn format_system_ram_line(snapshot: &SystemRamSnapshot) -> String {
    format!(
        "Installed RAM {} | Available RAM {}",
        format_ram_value(snapshot.installed_bytes),
        format_ram_value(snapshot.available_bytes)
    )
}

fn format_ram_value(bytes: Option<u64>) -> String {
    match bytes {
        Some(bytes) => format!("{} GB", bytes_to_rounded_gb(bytes)),
        None => "unknown".to_string(),
    }
}

fn bytes_to_rounded_gb(bytes: u64) -> u64 {
    const GIB: u64 = 1024 * 1024 * 1024;
    bytes.saturating_add(GIB / 2) / GIB
}

#[cfg(target_os = "linux")]
fn collect_system_ram_platform() -> SystemRamSnapshot {
    std::fs::read_to_string(crate::platform::absolute_path(&["proc", "meminfo"]))
        .ok()
        .map(|input| parse_linux_meminfo(&input))
        .unwrap_or_else(SystemRamSnapshot::unknown)
}

#[cfg(target_os = "macos")]
fn collect_system_ram_platform() -> SystemRamSnapshot {
    let installed_bytes = command_stdout("sysctl", &["-n", "hw.memsize"])
        .and_then(|output| parse_macos_hw_memsize(&output));
    let available_bytes =
        command_stdout("vm_stat", &[]).and_then(|output| parse_macos_vm_stat_available(&output));

    SystemRamSnapshot {
        installed_bytes,
        available_bytes,
    }
}

#[cfg(target_os = "windows")]
fn collect_system_ram_platform() -> SystemRamSnapshot {
    collect_windows_powershell_ram()
        .or_else(collect_windows_wmic_ram)
        .unwrap_or_else(SystemRamSnapshot::unknown)
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
fn collect_system_ram_platform() -> SystemRamSnapshot {
    SystemRamSnapshot::unknown()
}

#[cfg(any(target_os = "macos", target_os = "windows"))]
fn command_stdout(program: &str, args: &[&str]) -> Option<String> {
    let mut command = std::process::Command::new(program);
    configure_preflight_child_process(&mut command);
    let output = command.args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).into_owned())
}

#[cfg(target_os = "windows")]
fn collect_windows_powershell_ram() -> Option<SystemRamSnapshot> {
    command_stdout(
        "powershell.exe",
        &[
            "-NoProfile",
            "-Command",
            "$os=Get-CimInstance Win32_OperatingSystem; Write-Output $os.TotalVisibleMemorySize; Write-Output $os.FreePhysicalMemory",
        ],
    )
    .and_then(|output| parse_windows_powershell_memory(&output))
}

#[cfg(target_os = "windows")]
fn collect_windows_wmic_ram() -> Option<SystemRamSnapshot> {
    let value_arg = format!("{}Value", char::from(47));
    command_stdout(
        "wmic",
        &[
            "OS",
            "get",
            "TotalVisibleMemorySize,FreePhysicalMemory",
            value_arg.as_str(),
        ],
    )
    .and_then(|output| parse_windows_wmic_memory(&output))
}

#[cfg(any(target_os = "linux", test))]
fn parse_linux_meminfo(input: &str) -> SystemRamSnapshot {
    let mut installed_bytes = None;
    let mut available_bytes = None;
    let mut free_bytes = None;

    for line in input.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        match key.trim() {
            "MemTotal" => installed_bytes = parse_meminfo_value_bytes(value),
            "MemAvailable" => available_bytes = parse_meminfo_value_bytes(value),
            "MemFree" => free_bytes = parse_meminfo_value_bytes(value),
            _ => {}
        }
    }

    SystemRamSnapshot {
        installed_bytes,
        available_bytes: available_bytes.or(free_bytes),
    }
}

#[cfg(any(target_os = "linux", test))]
fn parse_meminfo_value_bytes(value: &str) -> Option<u64> {
    let mut parts = value.split_whitespace();
    let amount = parts.next()?.parse::<u64>().ok()?;
    match parts.next().unwrap_or("kB") {
        "kB" | "KB" | "kb" | "KiB" => amount.checked_mul(1024),
        "B" | "bytes" => Some(amount),
        _ => None,
    }
}

#[cfg(any(target_os = "macos", test))]
fn parse_macos_hw_memsize(input: &str) -> Option<u64> {
    input.trim().parse().ok()
}

#[cfg(any(target_os = "macos", test))]
fn parse_macos_vm_stat_available(input: &str) -> Option<u64> {
    let page_size = parse_macos_vm_stat_page_size(input)?;
    let mut free_pages = None;
    let mut inactive_pages = None;
    let mut speculative_pages = None;

    for line in input.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        match key.trim() {
            "Pages free" => free_pages = parse_count(value),
            "Pages inactive" => inactive_pages = parse_count(value),
            "Pages speculative" => speculative_pages = parse_count(value),
            _ => {}
        }
    }

    // macOS exposes page classes rather than one available-byte value; the
    // shared fact counts free, inactive, and speculative pages with checked math.
    let pages = free_pages?
        .checked_add(inactive_pages.unwrap_or(0))?
        .checked_add(speculative_pages.unwrap_or(0))?;
    pages.checked_mul(page_size)
}

#[cfg(any(target_os = "macos", test))]
fn parse_macos_vm_stat_page_size(input: &str) -> Option<u64> {
    let marker = "page size of ";
    let (_, rest) = input.split_once(marker)?;
    parse_count(rest)
}

#[cfg(any(target_os = "windows", test))]
fn parse_windows_powershell_memory(input: &str) -> Option<SystemRamSnapshot> {
    let mut lines = input.lines().map(str::trim).filter(|line| !line.is_empty());
    let installed_kib = lines.next()?.parse::<u64>().ok()?;
    let available_kib = lines.next()?.parse::<u64>().ok()?;
    // Win32 reports these values in KiB even though the shared snapshot is bytes.
    Some(SystemRamSnapshot {
        installed_bytes: installed_kib.checked_mul(1024),
        available_bytes: available_kib.checked_mul(1024),
    })
}

#[cfg(any(target_os = "windows", test))]
fn parse_windows_wmic_memory(input: &str) -> Option<SystemRamSnapshot> {
    let mut installed_bytes = None;
    let mut available_bytes = None;

    for line in input.lines() {
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        match key.trim() {
            "TotalVisibleMemorySize" => {
                installed_bytes = value.trim().parse::<u64>().ok()?.checked_mul(1024)
            }
            "FreePhysicalMemory" => {
                available_bytes = value.trim().parse::<u64>().ok()?.checked_mul(1024)
            }
            _ => {}
        }
    }

    (installed_bytes.is_some() || available_bytes.is_some()).then_some(SystemRamSnapshot {
        installed_bytes,
        available_bytes,
    })
}

#[cfg(any(target_os = "macos", test))]
fn parse_count(value: &str) -> Option<u64> {
    value
        .split_whitespace()
        .next()?
        .trim_end_matches('.')
        .parse()
        .ok()
}
