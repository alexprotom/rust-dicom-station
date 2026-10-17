//! The graphics backend choice ([`crate::gfx::Backend`]) in the window
//! library's `wgpu` types.
//!
//! The choice itself, its spellings, the platform rules and the fallback
//! order live with the settings in the domain crate, which does not link the
//! window library; the two conversions that need `eframe::wgpu` are here, in
//! the crate that draws the window.

use crate::gfx::Backend;
use eframe::wgpu;

/// Which [`Backend`] a running `wgpu` adapter is.
///
/// The menu says what the program is drawing with *now*, which is not
/// always what was asked for: a machine whose Vulkan driver does not start
/// falls through to the next backend by itself.
pub fn from_wgpu(backend: wgpu::Backend) -> Backend {
    match backend {
        wgpu::Backend::Vulkan => Backend::Vulkan,
        wgpu::Backend::Dx12 => Backend::Dx12,
        wgpu::Backend::Metal => Backend::Metal,
        wgpu::Backend::Gl => Backend::OpenGl,
        _ => Backend::Auto,
    }
}

/// The `wgpu` backends `backend` allows. `Auto` allows all of them, which is
/// what `wgpu` does when left alone.
pub fn bits(backend: Backend) -> wgpu::Backends {
    use wgpu::Backends;
    match backend {
        Backend::Auto => Backends::all(),
        Backend::Vulkan => Backends::VULKAN,
        Backend::Dx12 => Backends::DX12,
        Backend::Metal => Backends::METAL,
        Backend::OpenGl => Backends::GL,
    }
}
