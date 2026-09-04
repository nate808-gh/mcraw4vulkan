use tracing::info;

fn main() {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .init();
    info!("starting mcraw4vulkan");

    if let Err(error) = mcraw4vulkan::run_from_env_args() {
        eprintln!("{error:#}");
        std::process::exit(1);
    }
}
