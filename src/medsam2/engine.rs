//! Picking a backend, and the one call the rest of the program makes.
//!
//! The network below this module is generic over a `burn` backend, which is
//! what lets it run on the GPU and on the CPU from a single implementation —
//! but a generic type is awkward to hold in a UI struct or hand to a
//! background thread. [`Engine`] erases it: it owns whichever backend was
//! chosen and exposes a prompt-in, mask-out call that knows nothing about
//! tensors.
//!
//! The choice is made once, when the weights are loaded. With the `gpu`
//! feature (on by default) a wgpu adapter is tried first and the CPU backend
//! is the fallback; without it there is only the CPU backend, which is still
//! pure Rust.

use std::sync::Mutex;

use anyhow::Result;
use burn::tensor::backend::Backend;

use crate::nn::device::DevicePref;
use crate::nn::params::Params;
use crate::progress::ProgressSink;
use crate::volume::Volume;

use super::config;
use super::infer::{self, Config, Segmentation, Slices};
use super::model::{Medsam2, SliceFeatures};
use super::ops;
use super::preprocess::Prepared;
use super::prompt::Point;
use super::resample::{self, Filter};
use super::track::Prompt;

/// The pure-Rust CPU backend.
pub type Cpu = burn::backend::NdArray;
/// Vulkan / DX12 / Metal, with no CUDA toolkit involved.
#[cfg(feature = "gpu")]
pub type Gpu = burn::backend::Wgpu;

/// A prompt, in terms the caller can produce without touching a tensor.
pub enum EnginePrompt {
    /// Clicks or box corners, in **prepared** pixel coordinates
    /// (`row`, `column` of the oriented slice).
    Points(Vec<PixelPrompt>),
    /// A binary mask over one prepared slice, `rows * columns` bytes — an
    /// existing contour, propagated.
    Mask(Vec<u8>),
}

/// One click or box corner, in prepared pixel coordinates.
#[derive(Clone, Copy, Debug)]
pub struct PixelPrompt {
    pub row: f32,
    pub column: f32,
    pub label: i32,
}

impl PixelPrompt {
    pub fn positive(row: f32, column: f32) -> PixelPrompt {
        PixelPrompt {
            row,
            column,
            label: super::prompt::LABEL_POSITIVE,
        }
    }

    pub fn negative(row: f32, column: f32) -> PixelPrompt {
        PixelPrompt {
            row,
            column,
            label: super::prompt::LABEL_NEGATIVE,
        }
    }

    /// The two corners of a box, in prepared pixel coordinates.
    pub fn box_corners(row0: f32, col0: f32, row1: f32, col1: f32) -> Vec<PixelPrompt> {
        vec![
            PixelPrompt {
                row: row0.min(row1),
                column: col0.min(col1),
                label: super::prompt::LABEL_BOX_MIN,
            },
            PixelPrompt {
                row: row0.max(row1),
                column: col0.max(col1),
                label: super::prompt::LABEL_BOX_MAX,
            },
        ]
    }
}

/// The last encoded slice, kept so that adjusting a prompt on it — the whole
/// point of an interactive box — costs no encoder pass at all.
struct Cache<B: Backend> {
    slice: usize,
    features: SliceFeatures<B>,
}

enum Inner {
    Cpu(Box<Medsam2<Cpu>>, Mutex<Option<Cache<Cpu>>>),
    #[cfg(feature = "gpu")]
    Gpu(Box<Medsam2<Gpu>>, Mutex<Option<Cache<Gpu>>>),
}

/// The loaded network, on whichever backend was chosen.
pub struct Engine {
    inner: Inner,
    device: String,
}

impl Engine {
    /// Build the network from a loaded state dict, on the GPU when `device`
    /// allows it and a usable adapter exists (see [`DevicePref::resolve`]).
    pub fn load(params: &Params, device: DevicePref) -> Result<Engine> {
        let gpu = device.resolve()?;
        match gpu {
            #[cfg(feature = "gpu")]
            Some(ctx) => Ok(Engine {
                inner: Inner::Gpu(
                    Box::new(Medsam2::<Gpu>::load(params, ctx.device())?),
                    Mutex::new(None),
                ),
                device: ctx.describe(),
            }),
            #[cfg(not(feature = "gpu"))]
            Some(ctx) => ctx.unreachable(),
            None => {
                let device = burn::tensor::Device::<Cpu>::default();
                Ok(Engine {
                    inner: Inner::Cpu(
                        Box::new(Medsam2::<Cpu>::load(params, &device)?),
                        Mutex::new(None),
                    ),
                    device: crate::nn::device::describe_cpu(),
                })
            }
        }
    }

    /// What to show the user: which backend is actually running.
    pub fn device(&self) -> &str {
        &self.device
    }

    /// Segment one structure: prompt one slice, propagate through the stack.
    pub fn propagate(
        &self,
        prepared: &Prepared,
        slice: usize,
        prompt: &EnginePrompt,
        config: &Config,
        hooks: &dyn ProgressSink,
    ) -> Result<Segmentation> {
        match &self.inner {
            Inner::Cpu(model, cache) => run(model, cache, prepared, slice, prompt, config, hooks),
            #[cfg(feature = "gpu")]
            Inner::Gpu(model, cache) => run(model, cache, prepared, slice, prompt, config, hooks),
        }
    }

    /// Segment **one** slice, for the interactive loop: draw, look, adjust.
    ///
    /// The mask comes back at the prepared slice's own size, `rows * columns`
    /// bytes. Repeated calls on the same slice reuse its encoded features, so
    /// only the prompt path — a few milliseconds of it — runs again.
    pub fn preview(
        &self,
        prepared: &Prepared,
        slice: usize,
        prompt: &EnginePrompt,
        config: &Config,
    ) -> Result<Vec<u8>> {
        match &self.inner {
            Inner::Cpu(model, cache) => run_preview(model, cache, prepared, slice, prompt, config),
            #[cfg(feature = "gpu")]
            Inner::Gpu(model, cache) => run_preview(model, cache, prepared, slice, prompt, config),
        }
    }

    /// Slices the image encoder has processed since this engine was loaded.
    ///
    /// Re-prompting a cached slice must leave this unchanged — that is what
    /// makes the interactive loop interactive, and the one honest way to
    /// assert it, since how *long* a re-prompt takes depends on the machine.
    pub fn encode_count(&self) -> usize {
        match &self.inner {
            Inner::Cpu(model, _) => model.encode_count(),
            #[cfg(feature = "gpu")]
            Inner::Gpu(model, _) => model.encode_count(),
        }
    }

    /// Forget the cached slice — call this whenever the prepared stack itself
    /// changes (a different study, or a different intensity window).
    pub fn clear_cache(&self) {
        match &self.inner {
            Inner::Cpu(_, cache) => *cache.lock().unwrap() = None,
            #[cfg(feature = "gpu")]
            Inner::Gpu(_, cache) => *cache.lock().unwrap() = None,
        }
    }

    /// Segment, and land the mask on the volume's own grid.
    pub fn propagate_to_volume(
        &self,
        prepared: &Prepared,
        volume: &Volume,
        slice: usize,
        prompt: &EnginePrompt,
        config: &Config,
        hooks: &dyn ProgressSink,
    ) -> Result<(Vec<u8>, Segmentation)> {
        let seg = self.propagate(prepared, slice, prompt, config, hooks)?;
        let grid = prepared.mask_to_volume_grid(&seg.masks, volume);
        Ok((grid, seg))
    }
}

/// The prompt, in the network's coordinates and on its device.
fn to_prompt<B: Backend>(
    prepared: &Prepared,
    prompt: &EnginePrompt,
    device: &B::Device,
) -> Prompt<B> {
    match prompt {
        EnginePrompt::Points(points) => Prompt::Points(
            points
                .iter()
                .map(|p| {
                    let (x, y) = prepared.to_network(p.row, p.column);
                    Point {
                        x,
                        y,
                        label: p.label,
                    }
                })
                .collect(),
        ),
        EnginePrompt::Mask(mask) => {
            // The reference takes a mask at the video's resolution and lets
            // the prompt encoder shrink it; here it arrives on the study's
            // grid, so it is resampled to the network's first.
            let bytes: Vec<f32> = mask.iter().map(|v| f32::from(*v)).collect();
            let size = config::IMAGE_SIZE;
            let scaled = resample::resize(
                &bytes,
                prepared.size(),
                [size, size],
                Filter::Triangle,
                true,
            );
            let binary: Vec<f32> = scaled.into_iter().map(|v| f32::from(v > 0.5)).collect();
            Prompt::Mask(ops::from_slice(&binary, [1, 1, size, size], device))
        }
    }
}

/// The encoded prompted slice, from the cache when it is the same one.
fn anchor<B: Backend>(
    model: &Medsam2<B>,
    cache: &Mutex<Option<Cache<B>>>,
    stack: &dyn Slices<B>,
    slice: usize,
) -> SliceFeatures<B> {
    if let Some(hit) = cache.lock().unwrap().as_ref() {
        if hit.slice == slice {
            return hit.features.clone();
        }
    }
    let features = model.encode_slice(stack.slice(slice));
    *cache.lock().unwrap() = Some(Cache {
        slice,
        features: features.clone(),
    });
    features
}

fn run<B: Backend>(
    model: &Medsam2<B>,
    cache: &Mutex<Option<Cache<B>>>,
    prepared: &Prepared,
    slice: usize,
    prompt: &EnginePrompt,
    config: &Config,
    hooks: &dyn ProgressSink,
) -> Result<Segmentation> {
    let device = model.device().clone();
    let prompt = to_prompt::<B>(prepared, prompt, &device);
    let stack = prepared.stack::<B>(device);
    if slice >= stack.len() {
        anyhow::bail!("slice {slice} is outside a stack of {}", stack.len());
    }
    hooks.report(0.0, "Encoding the prompted slice");
    let anchor = anchor(model, cache, &stack, slice);
    infer::propagate_from(model, &stack, slice, &anchor, &prompt, config, hooks)
}

fn run_preview<B: Backend>(
    model: &Medsam2<B>,
    cache: &Mutex<Option<Cache<B>>>,
    prepared: &Prepared,
    slice: usize,
    prompt: &EnginePrompt,
    config: &Config,
) -> Result<Vec<u8>> {
    let device = model.device().clone();
    let prompt = to_prompt::<B>(prepared, prompt, &device);
    let stack = prepared.stack::<B>(device);
    if slice >= stack.len() {
        anyhow::bail!("slice {slice} is outside a stack of {}", stack.len());
    }
    let anchor = anchor(model, cache, &stack, slice);
    Ok(infer::preview(
        model,
        &anchor,
        &prompt,
        stack.out_size(),
        config.threshold,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_box_prompt_is_ordered_and_labelled() {
        let c = PixelPrompt::box_corners(30.0, 40.0, 10.0, 20.0);
        assert_eq!((c[0].row, c[0].column), (10.0, 20.0));
        assert_eq!((c[1].row, c[1].column), (30.0, 40.0));
        assert_eq!(c[0].label, super::super::prompt::LABEL_BOX_MIN);
        assert_eq!(c[1].label, super::super::prompt::LABEL_BOX_MAX);
        assert_eq!(
            PixelPrompt::positive(1.0, 2.0).label,
            super::super::prompt::LABEL_POSITIVE
        );
    }
}
