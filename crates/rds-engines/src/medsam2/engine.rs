//! Picking a backend, and the one call the rest of the program makes.
//!
//! The network below this module is generic over a `burn` backend, which is
//! what lets it run on the GPU and on the CPU from a single implementation -
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
    /// A binary mask over one prepared slice, `rows * columns` bytes - an
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

/// Encoded slices kept for the next prompt on the same stack, at most
/// [`KEEP_SLICES`] of them.
///
/// The prompted slice is always kept, so adjusting a prompt on it - the
/// whole point of an interactive box - costs no encoder pass at all. So are
/// the slices a propagation went through, those nearest the prompted slice
/// first: propagating a corrected prompt from the same slice then re-encodes
/// only the slices beyond them. (A plain least-recently-used rule would keep
/// the far end of the last pass, exactly the slices the next pass reaches
/// last.)
struct Cache<B: Backend> {
    /// The slice of the last prompt, what "nearest" is measured from.
    centre: usize,
    kept: Vec<(usize, SliceFeatures<B>)>,
}

/// Slices kept with their features: about 4 MB each at MedSAM2's 512-pixel
/// input.
pub const KEEP_SLICES: usize = 32;

impl<B: Backend> Default for Cache<B> {
    fn default() -> Self {
        Cache {
            centre: 0,
            kept: Vec::new(),
        }
    }
}

impl<B: Backend> Cache<B> {
    fn get(&self, slice: usize) -> Option<SliceFeatures<B>> {
        self.kept
            .iter()
            .find(|(s, _)| *s == slice)
            .map(|(_, f)| f.clone())
    }

    /// Keep `features` of `slice` unless every kept slice is nearer the
    /// centre; then the farthest one makes room. The centre's own slice is
    /// always kept.
    fn put(&mut self, slice: usize, features: SliceFeatures<B>) {
        if let Some(e) = self.kept.iter_mut().find(|(s, _)| *s == slice) {
            e.1 = features;
            return;
        }
        let dist = |s: usize| s.abs_diff(self.centre);
        if self.kept.len() >= KEEP_SLICES {
            let (far, _) = self
                .kept
                .iter()
                .enumerate()
                .max_by_key(|(_, (s, _))| (dist(*s), *s))
                .expect("full");
            if dist(self.kept[far].0) <= dist(slice) && slice != self.centre {
                return;
            }
            self.kept.swap_remove(far);
        }
        self.kept.push((slice, features));
    }
}

enum Inner {
    Cpu(Box<Medsam2<Cpu>>, Mutex<Cache<Cpu>>),
    #[cfg(feature = "gpu")]
    Gpu(Box<Medsam2<Gpu>>, Mutex<Cache<Gpu>>),
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
                    Mutex::new(Cache::default()),
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
                        Mutex::new(Cache::default()),
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
    /// only the prompt path - a few milliseconds of it - runs again.
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
    /// Re-prompting a cached slice must leave this unchanged - that is what
    /// makes the interactive loop interactive, and the one honest way to
    /// assert it, since how *long* a re-prompt takes depends on the machine.
    pub fn encode_count(&self) -> usize {
        match &self.inner {
            Inner::Cpu(model, _) => model.encode_count(),
            #[cfg(feature = "gpu")]
            Inner::Gpu(model, _) => model.encode_count(),
        }
    }

    /// Forget the cached slices - call this whenever the prepared stack
    /// itself changes (a different study, or a different intensity window).
    pub fn clear_cache(&self) {
        match &self.inner {
            Inner::Cpu(_, cache) => *lock(cache) = Cache::default(),
            #[cfg(feature = "gpu")]
            Inner::Gpu(_, cache) => *lock(cache) = Cache::default(),
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

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

/// A slice's features, from the cache when it holds them.
fn features<B: Backend>(
    model: &Medsam2<B>,
    cache: &Mutex<Cache<B>>,
    stack: &dyn Slices<B>,
    slice: usize,
) -> SliceFeatures<B> {
    if let Some(hit) = lock(cache).get(slice) {
        return hit;
    }
    let features = model.encode_slice(stack.slice(slice));
    lock(cache).put(slice, features.clone());
    features
}

/// The encoded prompted slice, which becomes the cache's centre.
fn anchor<B: Backend>(
    model: &Medsam2<B>,
    cache: &Mutex<Cache<B>>,
    stack: &dyn Slices<B>,
    slice: usize,
) -> SliceFeatures<B> {
    lock(cache).centre = slice;
    features(model, cache, stack, slice)
}

fn run<B: Backend>(
    model: &Medsam2<B>,
    cache: &Mutex<Cache<B>>,
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
    let tracked = |index: usize| features(model, cache, &stack, index);
    infer::propagate_with(
        model, &stack, slice, &anchor, &prompt, config, hooks, &tracked,
    )
}

fn run_preview<B: Backend>(
    model: &Medsam2<B>,
    cache: &Mutex<Cache<B>>,
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

    fn feats(v: f32) -> SliceFeatures<Cpu> {
        let dev = burn::tensor::Device::<Cpu>::default();
        SliceFeatures {
            pix_feat: ops::from_slice(&[v], [1, 1, 1, 1], &dev),
            high_res: None,
        }
    }

    fn value(f: &SliceFeatures<Cpu>) -> f32 {
        ops::to_vec(f.pix_feat.clone())[0]
    }

    #[test]
    fn the_slices_nearest_the_prompt_are_kept() {
        let mut c: Cache<Cpu> = Cache {
            centre: 50,
            kept: Vec::new(),
        };
        // A propagation from slice 50 up to 99, then back down to 0.
        for s in (50..100).chain((0..50).rev()) {
            c.put(s, feats(s as f32));
        }
        assert_eq!(c.kept.len(), KEEP_SLICES);
        let mut kept: Vec<usize> = c.kept.iter().map(|(s, _)| *s).collect();
        kept.sort();
        // The 32 nearest: 35..=65, and of 34 and 66 (both 16 away) the one
        // already kept when the other came.
        assert_eq!(kept, (35..67).collect::<Vec<_>>());
        assert_eq!(value(&c.get(40).unwrap()), 40.0);
        assert!(c.get(70).is_none());
        // A prompt elsewhere moves the centre; what is near it now wins.
        c.centre = 90;
        c.put(90, feats(90.0));
        c.put(91, feats(91.0));
        assert!(c.get(90).is_some() && c.get(91).is_some());
        assert!(c.get(35).is_none() && c.get(36).is_none() && c.get(37).is_some());
        assert_eq!(c.kept.len(), KEEP_SLICES);
    }

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
