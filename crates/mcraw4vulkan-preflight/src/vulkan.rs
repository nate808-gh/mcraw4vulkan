use std::env;
use std::ffi::OsStr;
#[cfg(any(target_os = "macos", test))]
use std::path::Path;
use std::path::PathBuf;

use crate::{PreflightRequirementStatus, VulkanSummary};

pub(crate) fn probe_vulkan(enabled: bool) -> VulkanSummary {
    if !enabled {
        // Disabled is Unknown rather than Missing because no availability fact
        // was collected.
        return VulkanSummary {
            status: PreflightRequirementStatus::Unknown,
            adapter_name: None,
            detail: "Vulkan probe disabled".to_string(),
        };
    }

    // Platform probes establish loader/runtime discoverability without creating a
    // window or a graphics device.
    probe_platform_vulkan()
}

#[cfg(target_os = "macos")]
fn probe_platform_vulkan() -> VulkanSummary {
    if let Some(package_root) = env::current_exe()
        .ok()
        .and_then(|exe| package_root_from_exe_path(&exe))
    {
        return probe_macos_package_vulkan(
            &package_root,
            env::var_os("VK_ICD_FILENAMES").as_deref(),
        );
    }

    probe_macos_developer_vulkan(
        env::var_os("VK_ICD_FILENAMES").as_deref(),
        env::var_os("VULKAN_SDK").as_deref(),
        env::var_os("DYLD_LIBRARY_PATH").as_deref(),
    )
}

#[cfg(target_os = "linux")]
fn probe_platform_vulkan() -> VulkanSummary {
    probe_linux_vulkan(
        env::var_os("VULKAN_SDK").as_deref(),
        env::var_os("LD_LIBRARY_PATH").as_deref(),
    )
}

#[cfg(target_os = "windows")]
fn probe_platform_vulkan() -> VulkanSummary {
    probe_windows_vulkan(env::var_os("SystemRoot").as_deref())
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
fn probe_platform_vulkan() -> VulkanSummary {
    VulkanSummary {
        status: PreflightRequirementStatus::Missing,
        adapter_name: None,
        detail: "No headless Vulkan library check is implemented for this platform".to_string(),
    }
}

#[cfg(any(target_os = "macos", test))]
fn probe_macos_package_vulkan(
    package_root: &Path,
    vk_icd_filenames: Option<&OsStr>,
) -> VulkanSummary {
    let required = macos_package_vulkan_paths(package_root);
    let missing = missing_paths(&required);
    let expected_icd = package_root.join("vulkan/icd.d/MoltenVK_icd.json");
    let icd_env_ok =
        vk_icd_filenames.is_some_and(|value| env_path_list_contains_existing(value, &expected_icd));

    if missing.is_empty() && icd_env_ok {
        VulkanSummary {
            status: PreflightRequirementStatus::Available,
            adapter_name: None,
            detail: format!(
                "Package-local Vulkan/MoltenVK files are present under {}",
                package_root.display()
            ),
        }
    } else {
        let mut details = Vec::new();
        if !missing.is_empty() {
            details.push(format!(
                "missing package-local files: {}",
                missing.join(", ")
            ));
        }
        if !icd_env_ok {
            details.push(format!(
                "VK_ICD_FILENAMES does not point to {}",
                expected_icd.display()
            ));
        }
        VulkanSummary {
            status: PreflightRequirementStatus::Missing,
            adapter_name: None,
            detail: details.join("; "),
        }
    }
}

#[cfg(target_os = "macos")]
fn probe_macos_developer_vulkan(
    vk_icd_filenames: Option<&OsStr>,
    vulkan_sdk: Option<&OsStr>,
    dyld_library_path: Option<&OsStr>,
) -> VulkanSummary {
    let loader_paths = candidate_paths(
        &[
            absolute_path(&["usr", "local", "lib", "libvulkan.1.dylib"]),
            absolute_path(&["usr", "local", "lib", "libvulkan.dylib"]),
            absolute_path(&["opt", "homebrew", "lib", "libvulkan.1.dylib"]),
            absolute_path(&["opt", "homebrew", "lib", "libvulkan.dylib"]),
        ],
        dyld_library_path,
        &["libvulkan.1.dylib", "libvulkan.dylib"],
    );
    let molten_paths = candidate_paths(
        &[
            absolute_path(&["usr", "local", "lib", "libMoltenVK.dylib"]),
            absolute_path(&["opt", "homebrew", "lib", "libMoltenVK.dylib"]),
        ],
        dyld_library_path,
        &["libMoltenVK.dylib"],
    );
    let sdk_paths = vulkan_sdk
        .map(|value| PathBuf::from(value.to_os_string()))
        .into_iter()
        .flat_map(|root| {
            [
                root.join("lib/libvulkan.1.dylib"),
                root.join("macOS/lib/libvulkan.1.dylib"),
                root.join("lib/libvulkan.dylib"),
                root.join("macOS/lib/libvulkan.dylib"),
                root.join("lib/libMoltenVK.dylib"),
                root.join("macOS/lib/libMoltenVK.dylib"),
                root.join("share/vulkan/icd.d/MoltenVK_icd.json"),
                root.join("macOS/share/vulkan/icd.d/MoltenVK_icd.json"),
            ]
        })
        .collect::<Vec<_>>();
    let icd_exists = vk_icd_filenames.is_some_and(env_path_list_has_existing);
    let sdk_available = sdk_paths.iter().any(|path| path.exists());
    let loader_available = loader_paths.iter().any(|path| path.exists()) || sdk_available;
    let molten_available =
        molten_paths.iter().any(|path| path.exists()) || icd_exists || sdk_available;

    if loader_available && molten_available {
        VulkanSummary {
            status: PreflightRequirementStatus::Available,
            adapter_name: None,
            detail: "Vulkan/MoltenVK runtime files are discoverable without creating a window"
                .to_string(),
        }
    } else {
        VulkanSummary {
            status: PreflightRequirementStatus::Missing,
            adapter_name: None,
            detail: "Vulkan/MoltenVK runtime files were not found in package, environment, VULKAN_SDK, or standard developer locations".to_string(),
        }
    }
}

#[cfg(any(target_os = "linux", test))]
fn probe_linux_vulkan(
    vulkan_sdk: Option<&OsStr>,
    ld_library_path: Option<&OsStr>,
) -> VulkanSummary {
    let sdk_paths = vulkan_sdk
        .map(|value| PathBuf::from(value.to_os_string()))
        .into_iter()
        .flat_map(|root| {
            [
                root.join("lib/libvulkan.so.1"),
                root.join("lib64/libvulkan.so.1"),
            ]
        })
        .collect::<Vec<_>>();
    let candidates = candidate_paths(
        &[
            absolute_path(&["usr", "lib", "libvulkan.so.1"]),
            absolute_path(&["usr", "lib64", "libvulkan.so.1"]),
            absolute_path(&["usr", "lib", "x86_64-linux-gnu", "libvulkan.so.1"]),
            absolute_path(&["usr", "local", "lib", "libvulkan.so.1"]),
        ],
        ld_library_path,
        &["libvulkan.so.1"],
    );
    let available = candidates.iter().any(|path| path.exists())
        || sdk_paths.iter().any(|path| path.exists())
        || ldconfig_lists_library("libvulkan.so.1");

    if available {
        VulkanSummary {
            status: PreflightRequirementStatus::Available,
            adapter_name: None,
            detail: "Vulkan loader library is discoverable without creating a window".to_string(),
        }
    } else {
        VulkanSummary {
            status: PreflightRequirementStatus::Missing,
            adapter_name: None,
            detail: "libvulkan.so.1 was not found".to_string(),
        }
    }
}

#[cfg(any(target_os = "windows", test))]
fn probe_windows_vulkan(system_root: Option<&OsStr>) -> VulkanSummary {
    let vulkan_dll = system_root
        .map(|value| PathBuf::from(value.to_os_string()))
        .unwrap_or_else(default_windows_system_root)
        .join("System32")
        .join("vulkan-1.dll");

    if vulkan_dll.exists() {
        VulkanSummary {
            status: PreflightRequirementStatus::Available,
            adapter_name: None,
            detail: format!("{} is present", vulkan_dll.display()),
        }
    } else {
        VulkanSummary {
            status: PreflightRequirementStatus::Missing,
            adapter_name: None,
            detail: format!("{} was not found", vulkan_dll.display()),
        }
    }
}

#[cfg(any(target_os = "windows", test))]
fn default_windows_system_root() -> PathBuf {
    PathBuf::from(["C:", "Windows"].join("\\"))
}

#[cfg(any(target_os = "macos", test))]
fn package_root_from_exe_path(exe: &Path) -> Option<PathBuf> {
    let bin_dir = exe.parent()?;
    if bin_dir.file_name()? != "bin" {
        return None;
    }
    let root = bin_dir.parent()?;
    (root.join("lib").is_dir() || root.join("vulkan").is_dir()).then(|| root.to_path_buf())
}

#[cfg(any(target_os = "macos", test))]
fn macos_package_vulkan_paths(package_root: &Path) -> [PathBuf; 4] {
    [
        package_root.join("lib/libvulkan.1.dylib"),
        package_root.join("lib/libvulkan.dylib"),
        package_root.join("lib/libMoltenVK.dylib"),
        package_root.join("vulkan/icd.d/MoltenVK_icd.json"),
    ]
}

#[cfg(any(target_os = "linux", test))]
fn ldconfig_lists_library(library_name: &str) -> bool {
    std::process::Command::new("ldconfig")
        .arg("-p")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| {
            String::from_utf8_lossy(&output.stdout)
                .lines()
                .any(|line| line.contains(library_name))
        })
        .unwrap_or(false)
}

#[cfg(any(target_os = "linux", target_os = "macos", test))]
fn candidate_paths(
    absolute_paths: &[PathBuf],
    library_path_env: Option<&OsStr>,
    library_names: &[&str],
) -> Vec<PathBuf> {
    let mut candidates = absolute_paths.to_vec();
    if let Some(value) = library_path_env {
        candidates.extend(
            env::split_paths(value)
                .flat_map(|dir| library_names.iter().map(move |name| dir.join(name))),
        );
    }
    candidates
}

#[cfg(any(target_os = "linux", target_os = "macos", test))]
fn absolute_path(parts: &[&str]) -> PathBuf {
    let mut path = PathBuf::from(std::path::MAIN_SEPARATOR.to_string());
    for part in parts {
        path.push(part);
    }
    path
}

#[cfg(any(target_os = "macos", test))]
fn env_path_list_contains_existing(value: &OsStr, expected: &Path) -> bool {
    env::split_paths(value).any(|path| {
        if path == expected {
            return path.exists();
        }

        let Ok(path_canonical) = path.canonicalize() else {
            return false;
        };
        let Ok(expected_canonical) = expected.canonicalize() else {
            return false;
        };
        path_canonical == expected_canonical
    })
}

#[cfg(target_os = "macos")]
fn env_path_list_has_existing(value: &OsStr) -> bool {
    env::split_paths(value).any(|path| path.exists())
}

#[cfg(any(target_os = "macos", test))]
fn missing_paths(paths: &[PathBuf]) -> Vec<String> {
    paths
        .iter()
        .filter(|path| !path.exists())
        .map(|path| path.display().to_string())
        .collect()
}
