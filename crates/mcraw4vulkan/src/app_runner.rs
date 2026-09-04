use std::path::Path;

use anyhow::Result;

use crate::display_window::{DisplayCliRunConfig, run_display_window};
use crate::dng_mount::{
    run_dng_unmount_all as run_registered_dng_unmount_all,
    run_dng_unmount_file as run_registered_dng_unmount_file,
};

pub fn run_display_cli(config: DisplayCliRunConfig) -> Result<()> {
    run_display_window(config)
}

pub fn run_dng_unmount_file(input: &Path) -> Result<()> {
    run_registered_dng_unmount_file(input).map(|_| ())
}

pub fn run_dng_unmount_all() -> Result<()> {
    run_registered_dng_unmount_all().map(|_| ())
}
