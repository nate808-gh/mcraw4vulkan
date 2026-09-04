#![forbid(unsafe_code)]

use std::error::Error;

use mcraw4vulkan_sdl2_wgpu_surface::{
    RenderFrameContext, RenderFrameStatus, Sdl2WgpuSurface, Sdl2WgpuSurfaceConfig,
};
use sdl2::event::{Event, WindowEvent};
use sdl2::keyboard::Keycode;

fn main() -> Result<(), Box<dyn Error>> {
    let mut surface = Sdl2WgpuSurface::new(Sdl2WgpuSurfaceConfig::default())?;
    println!(
        "SDL current video driver: {}",
        surface.current_video_driver()
    );
    let adapter = surface.adapter_info();
    println!("wgpu adapter: {} ({:?})", adapter.name, adapter.backend);
    println!(
        "surface: {:?}, present: {:?}, alpha: {:?}, size: {:?}, scale: {:.2}",
        surface.surface_format(),
        surface.present_mode(),
        surface.alpha_mode(),
        surface.size(),
        surface.scale_factor()
    );

    let mut event_pump = surface.event_pump()?;
    let mut running = true;

    while running {
        if let Some(event) = event_pump.wait_event_timeout(16) {
            running = handle_event(&mut surface, event)?;
        }
        for event in event_pump.poll_iter() {
            if !handle_event(&mut surface, event)? {
                running = false;
                break;
            }
        }

        match surface.render_frame(clear_frame)? {
            RenderFrameStatus::Submitted { .. }
            | RenderFrameStatus::SkippedZeroSize
            | RenderFrameStatus::SurfaceChanged
            | RenderFrameStatus::Timeout => {}
        }
    }

    Ok(())
}

fn handle_event(surface: &mut Sdl2WgpuSurface, event: Event) -> Result<bool, Box<dyn Error>> {
    match event {
        Event::Quit { .. }
        | Event::KeyDown {
            keycode: Some(Keycode::Escape),
            ..
        }
        | Event::Window {
            win_event: WindowEvent::Close,
            ..
        } => Ok(false),
        Event::Window {
            win_event: WindowEvent::Resized(_, _) | WindowEvent::SizeChanged(_, _),
            ..
        } => {
            let _ = surface.reconfigure()?;
            Ok(true)
        }
        _ => Ok(true),
    }
}

fn clear_frame(frame: RenderFrameContext<'_>) {
    let color_attachment = Some(wgpu::RenderPassColorAttachment {
        view: frame.view,
        resolve_target: None,
        ops: wgpu::Operations {
            load: wgpu::LoadOp::Clear(wgpu::Color {
                r: 0.02,
                g: 0.04,
                b: 0.06,
                a: 1.0,
            }),
            store: wgpu::StoreOp::Store,
        },
    });
    let color_attachments = [color_attachment];
    let _pass = frame
        .encoder
        .begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("SDL2 wgpu surface boundary smoke clear pass"),
            color_attachments: &color_attachments,
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
}
