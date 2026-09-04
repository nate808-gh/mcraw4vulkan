use std::process::Command;

#[cfg(windows)]
pub(crate) fn configure_preflight_child_process(command: &mut Command) {
    use std::os::windows::process::CommandExt;

    const CREATE_NO_WINDOW: u32 = 0x08000000;
    // Fact-probe helpers are non-interactive and must not open a transient console.
    command.creation_flags(CREATE_NO_WINDOW);
}

#[cfg(not(windows))]
pub(crate) fn configure_preflight_child_process(_command: &mut Command) {}
