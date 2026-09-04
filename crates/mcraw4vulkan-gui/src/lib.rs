#![forbid(unsafe_code)]

pub mod app;
pub mod dng_actions;
pub mod file_chooser;
mod gui_child_process;
pub mod gui_settings;
pub mod lazy_loop;
pub mod main_view;
pub mod optimizer_actions;
pub mod pipe_example;
pub mod playlist;
pub mod playlist_store;
pub mod preflight_view;
mod preview;
pub mod style;

pub use app::{GuiError, run};
