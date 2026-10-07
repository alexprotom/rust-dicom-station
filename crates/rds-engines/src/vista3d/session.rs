//! A VISTA-3D point session: the volume prepared once, the clicks given so
//! far, and the windows already answered.
//!
//! Every click opens its own 128-cubed window ([`super::points::stitch`]),
//! and a window's answer depends only on the image under it and the clicks
//! inside it; so a new click re-runs the windows it falls into and nothing
//! else, and the result is the one a run over all the clicks at once gives.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Result};
use burn::tensor::backend::Backend;

use super::points::{keep_with_positive, stitch, Click, PointHead};
use super::{load, Prepared};
use crate::medsam2::ops;
use crate::nn::device::DevicePref;
use crate::nn::params::Params;
use crate::progress::{Progress, ProgressSink, CANCELLED};
use crate::segresnet::net::{NormKind, SegResNetDs};
use crate::volume::Volume;

/// The window size of the bundle.
pub const ROI: [usize; 3] = [128; 3];
/// Answered windows kept for later clicks (8 MB each).
const CACHE: usize = 24;

/// The point half of `vista3d132` on one device.
pub struct PointNet<B: Backend> {
    encoder: SegResNetDs<B>,
    head: PointHead<B>,
    device: B::Device,
}

impl<B: Backend> PointNet<B> {
    /// `vista3d132`'s point half; the width from the checkpoint.
    pub fn load(p: &Params, dev: &B::Device) -> Result<Self> {
        Self::load_with(p, &[1, 2, 2, 4, 4], dev)
    }

    pub fn load_with(p: &Params, blocks_down: &[usize], dev: &B::Device) -> Result<Self> {
        let width = p
            .shape("point_head.mask_tokens.weight")
            .and_then(|s| s.get(1).copied())
            .ok_or_else(|| anyhow::anyhow!("checkpoint has no point head"))?;
        Ok(PointNet {
            encoder: SegResNetDs::load(
                p,
                "image_encoder.",
                width,
                blocks_down,
                width,
                NormKind::Instance,
                Some("up_layers"),
                None,
                true,
                dev,
            )?,
            head: PointHead::load(p, width, dev)?,
            device: dev.clone(),
        })
    }

    /// Logits for one window with the clicks inside it.
    pub fn window(&self, patch: &[f32], p: [usize; 3], clicks: &[Click]) -> Vec<f32> {
        let x = ops::from_slice::<B, 5>(patch, [1, 1, p[0], p[1], p[2]], &self.device);
        ops::to_vec(self.head.apply(self.encoder.forward_point(x), clicks))
    }
}

enum Net {
    Cpu(Box<PointNet<crate::medsam2::engine::Cpu>>),
    #[cfg(feature = "gpu")]
    Gpu(Box<PointNet<crate::medsam2::engine::Gpu>>),
}

/// The loaded network.
pub struct PointModel {
    net: Net,
    /// Where it runs, as the interface shows it.
    pub device: String,
}

impl PointModel {
    /// Load VISTA-3D's point half, downloading the weights on first use.
    pub fn load(root: &Path, device: DevicePref, sink: &dyn ProgressSink) -> Result<PointModel> {
        let params = load(root, sink)?;
        let gpu = device.resolve()?;
        Ok(match gpu {
            None => PointModel {
                net: Net::Cpu(Box::new(PointNet::load(&params, &Default::default())?)),
                device: crate::nn::device::describe_cpu(),
            },
            #[cfg(feature = "gpu")]
            Some(ctx) => PointModel {
                net: Net::Gpu(Box::new(PointNet::load(&params, ctx.device())?)),
                device: ctx.describe(),
            },
            #[cfg(not(feature = "gpu"))]
            Some(ctx) => ctx.unreachable(),
        })
    }

    pub fn window(&self, patch: &[f32], clicks: &[Click]) -> Result<Vec<f32>> {
        Ok(match &self.net {
            Net::Cpu(n) => n.window(patch, ROI, clicks),
            #[cfg(feature = "gpu")]
            Net::Gpu(n) => crate::nn::device::guarded(|| Ok(n.window(patch, ROI, clicks)))?,
        })
    }
}

type Key = ([usize; 3], Vec<(u32, u32, u32, i32)>);

fn key(lo: [usize; 3], clicks: &[Click]) -> Key {
    (
        lo,
        clicks
            .iter()
            .map(|c| {
                (
                    c.at[0].to_bits(),
                    c.at[1].to_bits(),
                    c.at[2].to_bits(),
                    c.label,
                )
            })
            .collect(),
    )
}

/// The session.
pub struct PointSession {
    prep: Prepared,
    clicks: Vec<Click>,
    cache: HashMap<Key, Vec<f32>>,
    order: Vec<Key>,
}

impl PointSession {
    pub fn new(volume: &Volume) -> PointSession {
        PointSession {
            prep: Prepared::new(volume),
            clicks: Vec::new(),
            cache: HashMap::new(),
            order: Vec::new(),
        }
    }

    pub fn n_clicks(&self) -> usize {
        self.clicks.len()
    }

    /// A click at a voxel of the volume (`[x, y, z]`, fractional).
    pub fn add(&mut self, at: [f64; 3], include: bool) -> Result<()> {
        let p = self.prep.point(at);
        if (0..3).any(|a| p[a] < 0.0 || p[a] > (self.prep.dims[a] - 1) as f32) {
            bail!("the click is outside the part of the scan VISTA-3D looks at (the body)");
        }
        self.clicks.push(Click {
            at: p,
            label: i32::from(include),
        });
        Ok(())
    }

    /// Take back the last click.
    pub fn undo(&mut self) -> bool {
        self.clicks.pop().is_some()
    }

    pub fn reset(&mut self) {
        self.clicks.clear();
    }

    /// The structure the clicks describe, on the volume's own grid.
    pub fn segment(
        &mut self,
        model: &PointModel,
        volume: &Volume,
        progress: &Progress,
    ) -> Result<Vec<u8>> {
        if self.clicks.is_empty() {
            return Ok(vec![0; volume.data.len()]);
        }
        let n_windows = self.clicks.len();
        let mut done = 0usize;
        let (cache, order) = (&mut self.cache, &mut self.order);
        let logits = stitch(
            &self.prep.data,
            self.prep.dims,
            ROI,
            &self.clicks,
            &mut |lo, patch, inside| {
                done += 1;
                if progress.cancelled() {
                    bail!(CANCELLED);
                }
                let k = key(lo, inside);
                if let Some(v) = cache.get(&k) {
                    return Ok(v.clone());
                }
                progress.report(
                    (done - 1) as f32 / n_windows as f32,
                    &format!("Segmenting (VISTA-3D points): window {done}/{n_windows}"),
                );
                let v = model.window(patch, inside)?;
                cache.insert(k.clone(), v.clone());
                order.push(k);
                if order.len() > CACHE {
                    let old = order.remove(0);
                    cache.remove(&old);
                }
                Ok(v)
            },
        )?;
        let mask = keep_with_positive(&logits, self.prep.dims, &self.clicks);
        Ok(self.prep.to_volume(&mask, volume))
    }
}
