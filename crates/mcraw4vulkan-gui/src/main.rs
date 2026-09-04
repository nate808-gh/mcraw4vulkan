#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]
#![forbid(unsafe_code)]

fn main() {
    if let Err(error) = mcraw4vulkan_gui::run() {
        eprintln!("mcraw4vulkan-gui: {error}");
        std::process::exit(1);
    }
}
