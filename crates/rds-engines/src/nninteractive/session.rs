//! The interactive session: one image, the prompts given so far, and the
//! segmentation they have produced - a port of
//! `nnInteractiveInferenceSession` (nninteractive 2.6.0) in its default
//! configuration (AutoZoom on, constant padding, the legacy channel layout
//! of the v1.0 checkpoint).
//!
//! What happens on every prompt:
//!
//! 1. The prompt is drawn into its channel at an intensity that grows by
//!    `1 / decay` per prompt, so the newest one weighs most once the
//!    channels are divided by the current intensity for the network
//!    (`_prepare_new_interaction_intensity`).
//! 2. A patch around the prompt is segmented ([`Session::predict`]). If the
//!    answer changed along the patch border, the view zooms out by 1.5
//!    (up to 4x): a larger box, resampled to the patch (trilinear for the
//!    image, area averages for the prompts, point and scribble channels
//!    dilated first so they survive the averaging) and run again.
//! 3. After a zoomed pass, the changed region is covered by patch-sized
//!    boxes ([`ops::generate_bounding_boxes`]) and each is re-run at full
//!    resolution with the coarse answer as its previous segmentation.
//!
//! The image is used in voxel space - no resampling - and z-scored with the
//! mean and standard deviation of its nonzero bounding box. Arrays are
//! C-order `[d0][d1][d2]`; prompts take voxel coordinates in that order.
//!
//! Prompts are kept as a list and drawn into a patch when one is needed,
//! instead of as seven full-size float16 channels; the values are the
//! float16 values upstream stores, so the network sees the same numbers.

use anyhow::{bail, Result};

use super::ops::{self, h, round_py, BBox, Lcg, Stride};
use crate::progress::{Progress, CANCELLED};

/// The prompt channels, after the image: the previous segmentation, then a
/// positive and a negative channel per prompt kind (the v1.0 layout:
/// `prev_seg` 0, `bbox2d` / `lasso` (1, 2), `points` (3, 4), `scribble`
/// (5, 6)).
pub const N_PROMPT_CHANNELS: usize = 7;
const PREV_SEG: usize = 0;
/// Channels that are dilated before a zoomed-out resampling.
const THIN: [usize; 4] = [3, 4, 5, 6];
const MAX_ZOOM: f64 = 4.0;
const ZOOM_STEP: f64 = 1.5;

/// The kinds of prompt.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    /// A 2-D box: one voxel thick along one axis.
    Box,
    /// A filled outline on one slice.
    Lasso,
    Point,
    /// A drawn stroke.
    Scribble,
}

impl Kind {
    fn channels(self) -> (usize, usize) {
        match self {
            Kind::Box | Kind::Lasso => (1, 2),
            Kind::Point => (3, 4),
            Kind::Scribble => (5, 6),
        }
    }
}

/// Per-model settings, from the checkpoint's `inference_session_class.json`
/// and `plans.json`.
#[derive(Clone, Debug)]
pub struct Settings {
    pub patch: [usize; 3],
    pub point_radius: usize,
    /// How much an older prompt counts relative to the next one.
    pub decay: f64,
    /// Zoom out when the answer reaches the patch border.
    pub autozoom: bool,
    /// The part of a refinement box not counted as covered (24 voxels
    /// upstream, sized for the 192-cubed patch).
    pub refine_margin: usize,
}

impl Settings {
    pub fn v1(patch: [usize; 3]) -> Settings {
        Settings {
            patch,
            point_radius: 4,
            decay: 0.98,
            autozoom: true,
            refine_margin: 24,
        }
    }
}

/// One network: the image and the prompt channels in
/// (`[1 + N_PROMPT_CHANNELS, p0, p1, p2]`, C-order), foreground labels out.
pub trait Predictor {
    fn predict(&self, input: &[f32], patch: [usize; 3]) -> Result<Vec<u8>>;
}

enum Shape {
    /// The point's structuring element centred here.
    Point([i64; 3]),
    /// A constant over `[lo, hi)`.
    Fill([usize; 3], [usize; 3]),
    /// A mask over `[lo, lo + dims)`.
    Mask {
        lo: [usize; 3],
        dims: [usize; 3],
        bits: Vec<u8>,
    },
}

struct Prompt {
    channel: usize,
    intensity: f64,
    shape: Shape,
}

struct Undo {
    prompts: usize,
    intensity: f64,
    pending: Option<([i64; 3], f64)>,
    mask: Vec<u8>,
}

/// What one prediction did.
#[derive(Clone, Debug, Default)]
pub struct Outcome {
    /// Network passes, coarse and refining.
    pub passes: usize,
    /// The last zoom factor.
    pub zoom: f64,
    /// The region whose labels were written, clipped to the image.
    pub changed: Option<BBox>,
}

pub struct Session {
    pub dims: [usize; 3],
    image: Vec<f32>,
    /// The current segmentation - upstream's `prev_seg` channel and target
    /// buffer, which always agree.
    pub mask: Vec<u8>,
    prompts: Vec<Prompt>,
    intensity: f64,
    pub settings: Settings,
    strel: (usize, Vec<f64>),
    /// Centre and zoom factor of the newest prompt not yet predicted.
    pending: Option<([i64; 3], f64)>,
    rng: Lcg,
    undo: Option<Undo>,
}

impl Session {
    /// A session on `image` (C-order, `dims`), normalized here.
    pub fn new(image: Vec<f32>, dims: [usize; 3], settings: Settings) -> Result<Session> {
        if image.len() != dims[0] * dims[1] * dims[2] {
            bail!("the image does not match its dimensions");
        }
        let image = normalize(image, dims)?;
        let strel = ops::point_strel(settings.point_radius);
        Ok(Session {
            dims,
            mask: vec![0; image.len()],
            image,
            prompts: Vec::new(),
            intensity: 1.0,
            settings,
            strel,
            pending: None,
            rng: Lcg::default(),
            undo: None,
        })
    }

    pub fn n_prompts(&self) -> usize {
        self.prompts.len()
    }

    /// Forget every prompt and the segmentation.
    pub fn reset(&mut self) {
        self.prompts.clear();
        self.intensity = 1.0;
        self.pending = None;
        self.mask.fill(0);
        self.undo = None;
    }

    /// Take back the last prompt and what its prediction wrote. One level,
    /// as upstream.
    pub fn undo(&mut self) -> bool {
        let Some(u) = self.undo.take() else {
            return false;
        };
        self.prompts.truncate(u.prompts);
        self.intensity = u.intensity;
        self.pending = u.pending;
        self.mask = u.mask;
        true
    }

    fn begin(&mut self) {
        self.undo = Some(Undo {
            prompts: self.prompts.len(),
            intensity: self.intensity,
            pending: self.pending,
            mask: self.mask.clone(),
        });
    }

    fn bump_intensity(&mut self) {
        let d = self.settings.decay;
        if d > 0.0 && d < 1.0 {
            self.intensity /= d;
            // Keep the stored float16 values finite: rescale every prompt
            // when the running intensity passes float16's range.
            if self.intensity > ops::F16_MAX {
                let target = ops::F16_MAX / 10.0;
                let s = target / self.intensity;
                for p in &mut self.prompts {
                    p.intensity *= s;
                }
                self.intensity = target;
            }
        }
    }

    /// A click at voxel `at`; `include` says whether it is on the structure.
    pub fn add_point(&mut self, at: [f64; 3], include: bool) -> Result<()> {
        self.begin();
        let c: [i64; 3] = std::array::from_fn(|a| round_py(at[a]));
        self.pending = Some((c, 1.0));
        self.bump_intensity();
        let (pos, neg) = Kind::Point.channels();
        let n = self.strel.0 as i64;
        let reach: BBox = std::array::from_fn(|a| [c[a] - n / 2, c[a] + n / 2 + n % 2]);
        if (0..3).any(|a| reach[a][1] < 0 || reach[a][0] > self.dims[a] as i64) {
            // Outside the image: upstream draws nothing.
            return Ok(());
        }
        self.prompts.push(Prompt {
            channel: if include { pos } else { neg },
            intensity: self.intensity,
            shape: Shape::Point(c),
        });
        Ok(())
    }

    /// A box `[lo, hi)` in voxels, one voxel thick along at least one axis
    /// (the v1.0 model takes 2-D boxes only).
    pub fn add_box(&mut self, lo: [f64; 3], hi: [f64; 3], include: bool) -> Result<()> {
        let size: [f64; 3] = std::array::from_fn(|a| hi[a] - lo[a]);
        if size.contains(&0.0) {
            bail!("the box has no extent along one axis");
        }
        if !size.contains(&1.0) {
            bail!("the model takes 2-D boxes only: one voxel thick along one axis");
        }
        self.begin();
        let mut b: BBox = std::array::from_fn(|a| [round_py(lo[a]), round_py(hi[a])]);
        for (a, r) in b.iter_mut().enumerate() {
            let n = self.dims[a] as i64;
            r[0] = r[0].max(0);
            r[1] = r[1].min(n);
            if r[1] <= r[0] {
                if r[0] == 0 {
                    r[1] = n.min(1);
                } else {
                    r[0] = (r[0] - 1).max(0);
                }
            }
        }
        let p = self.settings.patch;
        let centre: [i64; 3] = std::array::from_fn(|a| round_py((b[a][0] + b[a][1]) as f64 / 2.0));
        let zoom = (0..3)
            .map(|a| ((b[a][1] - b[a][0]) as f64 + (p[a] / 3) as f64) / p[a] as f64)
            .fold(1.0, f64::max);
        self.pending = Some((centre, zoom));
        self.bump_intensity();
        let (pos, neg) = Kind::Box.channels();
        self.prompts.push(Prompt {
            channel: if include { pos } else { neg },
            intensity: self.intensity,
            shape: Shape::Fill(
                std::array::from_fn(|a| b[a][0] as usize),
                std::array::from_fn(|a| b[a][1] as usize),
            ),
        });
        Ok(())
    }

    /// A scribble or a lasso: the nonzero voxels of `bits` (dims `mdims`)
    /// placed at `lo`.
    pub fn add_mask(
        &mut self,
        kind: Kind,
        lo: [usize; 3],
        mdims: [usize; 3],
        bits: Vec<u8>,
        include: bool,
    ) -> Result<()> {
        if !matches!(kind, Kind::Scribble | Kind::Lasso) {
            bail!("only scribbles and lassos are masks");
        }
        if bits.len() != mdims[0] * mdims[1] * mdims[2] || mdims.contains(&0) {
            bail!("the mask does not match its dimensions");
        }
        if (0..3).any(|a| lo[a] + mdims[a] > self.dims[a]) {
            bail!("the mask reaches outside the image");
        }
        self.begin();
        if let Some(nz) = ops::nonzero_bbox(&bits, mdims) {
            let p = self.settings.patch;
            let roi: BBox =
                std::array::from_fn(|a| [nz[a][0] + lo[a] as i64, nz[a][1] + lo[a] as i64]);
            let centre = std::array::from_fn(|a| round_py((roi[a][0] + roi[a][1]) as f64 / 2.0));
            let zoom = (0..3)
                .map(|a| ((roi[a][1] - roi[a][0]) as f64 + (p[a] / 3) as f64) / p[a] as f64)
                .fold(1.0, f64::max);
            self.pending = Some((centre, zoom));
        }
        self.bump_intensity();
        let (pos, neg) = kind.channels();
        self.prompts.push(Prompt {
            channel: if include { pos } else { neg },
            intensity: self.intensity,
            shape: Shape::Mask {
                lo,
                dims: mdims,
                bits,
            },
        });
        Ok(())
    }

    /// Start from an existing segmentation (`mask`, the session's dims):
    /// every prompt is forgotten. With `queue`, the next
    /// [`Session::predict`] (with `force_full_refine`) runs the network over
    /// all of it; without, it is kept as it is until the next prompt.
    pub fn set_initial(&mut self, mask: &[u8], queue: bool) -> Result<()> {
        if mask.len() != self.mask.len() {
            bail!("the segmentation does not match the image");
        }
        self.begin();
        self.prompts.clear();
        self.intensity = 1.0;
        self.pending = None;
        for (m, &v) in self.mask.iter_mut().zip(mask) {
            *m = u8::from(v != 0);
        }
        if !queue {
            return Ok(());
        }
        if let Some(nz) = ops::nonzero_bbox(&self.mask, self.dims) {
            let p = self.settings.patch;
            let centre = std::array::from_fn(|a| round_py((nz[a][0] + nz[a][1]) as f64 / 2.0));
            let zoom = (0..3)
                .map(|a| ((nz[a][1] - nz[a][0]) as f64 + (p[a] / 3) as f64) / p[a] as f64)
                .fold(1.0, f64::max);
            self.pending = Some((centre, zoom));
        }
        Ok(())
    }

    /// Run the network for the newest prompt. `force_full_refine` re-runs
    /// the whole current segmentation as well (upstream uses it after an
    /// initial segmentation).
    pub fn predict(
        &mut self,
        net: &dyn Predictor,
        force_full_refine: bool,
        progress: &Progress,
    ) -> Result<Outcome> {
        let Some((centre, zoom0)) = self.pending else {
            return Ok(Outcome::default());
        };
        let p = self.settings.patch;
        let mut passes = 0usize;
        let mut pass = |input: &[f32], what: &str| -> Result<Vec<u8>> {
            if progress.cancelled() {
                bail!(CANCELLED);
            }
            passes += 1;
            progress.set(format!("{what} (network pass {passes})"));
            net.predict(input, p)
        };
        let mut zoom = zoom0.min(MAX_ZOOM);
        let (input, mut scaled_bbox, mut previous) = self.network_input(centre, zoom);
        let mut pred = pass(&input, "Segmenting around the prompt")?;
        drop(input);
        let mut change = border_change(&pred, &previous, p);
        while change && self.settings.autozoom {
            if zoom >= MAX_ZOOM {
                break;
            }
            zoom = (zoom * ZOOM_STEP).min(MAX_ZOOM);
            let (input, b, prev) = self.network_input(centre, zoom);
            scaled_bbox = b;
            previous = prev;
            pred = pass(&input, &format!("Zooming out x{zoom:.2}"))?;
            change = border_change(&pred, &previous, p);
        }
        drop(previous);
        let mut changed = None;
        if zoom == 1.0 {
            ops::paste(&mut self.mask, self.dims, &pred, p, &scaled_bbox);
            changed = ops::clip(&scaled_bbox, self.dims);
        } else if let Some(seen) = ops::clip(&scaled_bbox, self.dims) {
            let coarse = self.upsample(&pred, &scaled_bbox, &seen);
            let boxes = self.plan_refinement(&coarse, &seen, centre, force_full_refine);
            // The coarse answer is context for the refinement passes only:
            // it lives in a cache over the boxes and reaches the
            // segmentation through them alone.
            if let Some(cache_bbox) = ops::union(&boxes) {
                let cd = ops::bbox_size(&cache_bbox);
                let mut cache = vec![0u8; cd[0] * cd[1] * cd[2]];
                if let Some(inside) = ops::clip(&cache_bbox, self.dims) {
                    let part = crop(&self.mask, self.dims, &inside);
                    ops::paste(
                        &mut cache,
                        cd,
                        &part,
                        ops::bbox_size(&inside),
                        &ops::to_local(&inside, &cache_bbox),
                    );
                }
                ops::paste(
                    &mut cache,
                    cd,
                    &coarse,
                    ops::bbox_size(&seen),
                    &ops::to_local(&seen, &cache_bbox),
                );
                let n = boxes.len();
                for (bi, b) in boxes.iter().enumerate() {
                    let local = ops::to_local(b, &cache_bbox);
                    let input = self.patch_input(b, Some((&cache, cd, &local)));
                    let refined = pass(&input, &format!("Refining {} of {n}", bi + 1))?;
                    ops::paste(&mut cache, cd, &refined, p, &local);
                }
                for b in &boxes {
                    let part = crop(&cache, cd, &ops::to_local(b, &cache_bbox));
                    ops::paste(&mut self.mask, self.dims, &part, p, b);
                }
                changed = ops::clip(&cache_bbox, self.dims);
            }
        }
        self.pending = None;
        Ok(Outcome {
            passes,
            zoom,
            changed,
        })
    }

    // ------------------------------------------------------- prompt channels --

    /// Channel `c` over `region` (image coordinates, may reach outside),
    /// as stored upstream: 0 / 1 for the segmentation, float16 values for
    /// the prompts, 0 outside the image.
    fn render(&self, c: usize, region: &BBox) -> Vec<f32> {
        let rd = ops::bbox_size(region);
        let mut out = vec![0f32; rd[0] * rd[1] * rd[2]];
        let Some(inside) = ops::clip(region, self.dims) else {
            return out;
        };
        let at = |i: i64, j: i64, k: i64| -> usize {
            ops::idx(
                rd,
                (i - region[0][0]) as usize,
                (j - region[1][0]) as usize,
                (k - region[2][0]) as usize,
            )
        };
        if c == PREV_SEG {
            for i in inside[0][0]..inside[0][1] {
                for j in inside[1][0]..inside[1][1] {
                    for k in inside[2][0]..inside[2][1] {
                        out[at(i, j, k)] = f32::from(
                            self.mask[ops::idx(self.dims, i as usize, j as usize, k as usize)],
                        );
                    }
                }
            }
            return out;
        }
        for pr in self.prompts.iter().filter(|p| p.channel == c) {
            match &pr.shape {
                Shape::Point(centre) => {
                    let (n, s) = (&self.strel.0, &self.strel.1);
                    let n = *n as i64;
                    let lo: [i64; 3] = std::array::from_fn(|a| centre[a] - n / 2);
                    for i in lo[0].max(inside[0][0])..(lo[0] + n).min(inside[0][1]) {
                        for j in lo[1].max(inside[1][0])..(lo[1] + n).min(inside[1][1]) {
                            for k in lo[2].max(inside[2][0])..(lo[2] + n).min(inside[2][1]) {
                                let si =
                                    (((i - lo[0]) * n + (j - lo[1])) * n + (k - lo[2])) as usize;
                                let v = h((s[si] * pr.intensity) as f32);
                                let o = &mut out[at(i, j, k)];
                                *o = o.max(v);
                            }
                        }
                    }
                }
                Shape::Fill(lo, hi) => {
                    let v = h(pr.intensity as f32);
                    for i in (lo[0] as i64).max(inside[0][0])..(hi[0] as i64).min(inside[0][1]) {
                        for j in (lo[1] as i64).max(inside[1][0])..(hi[1] as i64).min(inside[1][1])
                        {
                            for k in
                                (lo[2] as i64).max(inside[2][0])..(hi[2] as i64).min(inside[2][1])
                            {
                                let o = &mut out[at(i, j, k)];
                                *o = o.max(v);
                            }
                        }
                    }
                }
                Shape::Mask { lo, dims, bits } => {
                    let v = h(pr.intensity as f32);
                    let hi: [i64; 3] = std::array::from_fn(|a| (lo[a] + dims[a]) as i64);
                    for i in (lo[0] as i64).max(inside[0][0])..hi[0].min(inside[0][1]) {
                        for j in (lo[1] as i64).max(inside[1][0])..hi[1].min(inside[1][1]) {
                            for k in (lo[2] as i64).max(inside[2][0])..hi[2].min(inside[2][1]) {
                                let m = ops::idx(
                                    *dims,
                                    (i - lo[0] as i64) as usize,
                                    (j - lo[1] as i64) as usize,
                                    (k - lo[2] as i64) as usize,
                                );
                                if bits[m] != 0 {
                                    let o = &mut out[at(i, j, k)];
                                    *o = o.max(v);
                                }
                            }
                        }
                    }
                }
            }
        }
        out
    }

    /// Divide a prompt channel by the running intensity, in float16, as
    /// `_normalize_interaction_channels_for_network_` does.
    fn normalize_channel(&self, values: &mut [f32]) {
        if self.intensity == 1.0 || self.intensity == 0.0 {
            return;
        }
        let i = self.intensity as f32;
        values.iter_mut().for_each(|v| *v = h(*v / i));
    }

    /// The image over `region`, 0 outside.
    fn image_region(&self, region: &BBox) -> Vec<f32> {
        let rd = ops::bbox_size(region);
        let mut out = vec![0f32; rd[0] * rd[1] * rd[2]];
        if let Some(inside) = ops::clip(region, self.dims) {
            for i in inside[0][0]..inside[0][1] {
                for j in inside[1][0]..inside[1][1] {
                    let k0 = inside[2][0];
                    let len = (inside[2][1] - k0) as usize;
                    let src = ops::idx(self.dims, i as usize, j as usize, k0 as usize);
                    let dst = ops::idx(
                        rd,
                        (i - region[0][0]) as usize,
                        (j - region[1][0]) as usize,
                        (k0 - region[2][0]) as usize,
                    );
                    out[dst..dst + len].copy_from_slice(&self.image[src..src + len]);
                }
            }
        }
        out
    }

    /// The network input for a patch-sized box at full resolution; the
    /// segmentation channel comes from `prev` (the refinement cache) when
    /// given.
    fn patch_input(&self, b: &BBox, prev: Option<(&[u8], [usize; 3], &BBox)>) -> Vec<f32> {
        let mut input = self.image_region(b);
        for c in 0..N_PROMPT_CHANNELS {
            let mut ch = match (c, prev) {
                (PREV_SEG, Some((cache, cd, local))) => crop(cache, cd, local)
                    .iter()
                    .map(|&v| f32::from(v))
                    .collect(),
                _ => self.render(c, b),
            };
            if c != PREV_SEG {
                self.normalize_channel(&mut ch);
            }
            input.extend_from_slice(&ch);
        }
        input
    }

    /// `_build_network_input`: the input around `centre` at `zoom`, the box
    /// it covers, and the previous segmentation as the network's patch sees
    /// it (for the border test).
    fn network_input(&self, centre: [i64; 3], zoom: f64) -> (Vec<f32>, BBox, Vec<f32>) {
        let p = self.settings.patch;
        let scaled: [usize; 3] = std::array::from_fn(|a| round_py(p[a] as f64 * zoom) as usize);
        let bbox: BBox = std::array::from_fn(|a| {
            let s = scaled[a] as i64;
            [centre[a] - s / 2, centre[a] + s / 2 + s % 2]
        });
        if scaled == p {
            let input = self.patch_input(&bbox, None);
            let n = p[0] * p[1] * p[2];
            let previous = input[n..2 * n].to_vec();
            return (input, bbox, previous);
        }
        let n = p[0] * p[1] * p[2];
        let mut input = Vec::with_capacity(n * (1 + N_PROMPT_CHANNELS));
        let valid = ops::clip(&bbox, self.dims);
        // the image: trilinear from the zero-padded crop
        {
            let img = |i: usize, j: usize, k: usize| -> f32 {
                let (x, y, z) = (
                    bbox[0][0] + i as i64,
                    bbox[1][0] + j as i64,
                    bbox[2][0] + k as i64,
                );
                if x < 0
                    || y < 0
                    || z < 0
                    || x >= self.dims[0] as i64
                    || y >= self.dims[1] as i64
                    || z >= self.dims[2] as i64
                {
                    0.0
                } else {
                    self.image[ops::idx(self.dims, x as usize, y as usize, z as usize)]
                }
            };
            input.extend(ops::trilinear_region(
                &img,
                scaled,
                p,
                [[0, p[0]], [0, p[1]], [0, p[2]]],
            ));
        }
        // the previous segmentation, nearest, for the border test
        let mut previous = vec![0f32; n];
        for o0 in 0..p[0] {
            let i = ops::nearest_index(o0, scaled[0], p[0]) as i64 + bbox[0][0];
            for o1 in 0..p[1] {
                let j = ops::nearest_index(o1, scaled[1], p[1]) as i64 + bbox[1][0];
                for o2 in 0..p[2] {
                    let k = ops::nearest_index(o2, scaled[2], p[2]) as i64 + bbox[2][0];
                    if i >= 0
                        && j >= 0
                        && k >= 0
                        && i < self.dims[0] as i64
                        && j < self.dims[1] as i64
                        && k < self.dims[2] as i64
                    {
                        previous[ops::idx(p, o0, o1, o2)] = f32::from(
                            self.mask[ops::idx(self.dims, i as usize, j as usize, k as usize)],
                        );
                    }
                }
            }
        }
        let ks = ops::round_to_nearest_odd(zoom * 2.0 - 1.0);
        for c in 0..N_PROMPT_CHANNELS {
            let Some(valid) = valid else {
                input.extend(std::iter::repeat_n(0.0, n));
                continue;
            };
            // The channel over the in-image part of the box, grown by the
            // dilation's reach for the thin channels: outside it the padded
            // buffer is zero, and with values >= 0 a max over a window that
            // also holds zeros is the max over the rest.
            let r = if ks > 1 && THIN.contains(&c) {
                ((ks - 1) / 2) as i64
            } else {
                0
            };
            let ext: BBox = std::array::from_fn(|a| {
                [
                    (valid[a][0] - r).max(bbox[a][0]),
                    (valid[a][1] + r).min(bbox[a][1]),
                ]
            });
            let ed = ops::bbox_size(&ext);
            let mut ch = self.render(c, &ext);
            if r > 0 {
                ops::pool_extreme(&mut ch, ed, ks, false);
            }
            let off: [i64; 3] = std::array::from_fn(|a| ext[a][0] - bbox[a][0]);
            let at = |i: usize, j: usize, k: usize| -> f32 {
                let (x, y, z) = (i as i64 - off[0], j as i64 - off[1], k as i64 - off[2]);
                if x < 0
                    || y < 0
                    || z < 0
                    || x >= ed[0] as i64
                    || y >= ed[1] as i64
                    || z >= ed[2] as i64
                {
                    0.0
                } else {
                    ch[ops::idx(ed, x as usize, y as usize, z as usize)]
                }
            };
            let mut pooled = ops::area_f16(&at, scaled, p);
            if c != PREV_SEG {
                self.normalize_channel(&mut pooled);
            }
            input.extend_from_slice(&pooled);
        }
        (input, bbox, previous)
    }

    /// The coarse answer brought back to the zoomed box's own size (a
    /// trilinear resampling thresholded at 0.5), on its in-image part
    /// `seen` only - the rest is never read.
    fn upsample(&self, pred: &[u8], scaled_bbox: &BBox, seen: &BBox) -> Vec<u8> {
        let p = self.settings.patch;
        let scaled = ops::bbox_size(scaled_bbox);
        let local = ops::to_local(seen, scaled_bbox);
        if scaled == p {
            return crop(pred, p, &local);
        }
        let at = |i: usize, j: usize, k: usize| f32::from(pred[ops::idx(p, i, j, k)]);
        let region: [[usize; 2]; 3] =
            std::array::from_fn(|a| [local[a][0] as usize, local[a][1] as usize]);
        ops::trilinear_region(&at, p, scaled, region)
            .into_iter()
            .map(|v| u8::from(v >= 0.5))
            .collect()
    }

    /// `_plan_refinement_bboxes`: patch-sized boxes over where the coarse
    /// answer differs from the segmentation (opened, so specks do not each
    /// earn a pass), or one box on the prompt when nothing changed.
    fn plan_refinement(
        &mut self,
        coarse: &[u8],
        seen: &BBox,
        centre: [i64; 3],
        force_full: bool,
    ) -> Vec<BBox> {
        let p = self.settings.patch;
        let fallback = || -> Vec<BBox> {
            vec![std::array::from_fn(|a| {
                let lo = centre[a] - (p[a] / 2) as i64;
                [lo, lo + p[a] as i64]
            })]
        };
        let mut planning = *seen;
        if force_full {
            if let Some(nz) = ops::nonzero_bbox(&self.mask, self.dims) {
                planning = ops::union(&[planning, nz]).unwrap_or(planning);
            }
        }
        let pd = ops::bbox_size(&planning);
        let mut diff = vec![0u8; pd[0] * pd[1] * pd[2]];
        {
            let sd = ops::bbox_size(seen);
            let mut d = vec![0f32; sd[0] * sd[1] * sd[2]];
            for i in 0..sd[0] {
                for j in 0..sd[1] {
                    for k in 0..sd[2] {
                        let g = ops::idx(
                            self.dims,
                            i + seen[0][0] as usize,
                            j + seen[1][0] as usize,
                            k + seen[2][0] as usize,
                        );
                        let l = ops::idx(sd, i, j, k);
                        d[l] = f32::from(coarse[l] != self.mask[g]);
                    }
                }
            }
            ops::pool_extreme(&mut d, sd, 5, true);
            ops::pool_extreme(&mut d, sd, 5, false);
            let bits: Vec<u8> = d.iter().map(|&v| u8::from(v != 0.0)).collect();
            ops::paste(&mut diff, pd, &bits, sd, &ops::to_local(seen, &planning));
        }
        if force_full {
            for i in 0..pd[0] {
                for j in 0..pd[1] {
                    for k in 0..pd[2] {
                        let g = ops::idx(
                            self.dims,
                            i + planning[0][0] as usize,
                            j + planning[1][0] as usize,
                            k + planning[2][0] as usize,
                        );
                        if self.mask[g] != 0 {
                            diff[ops::idx(pd, i, j, k)] = 1;
                        }
                    }
                }
            }
        }
        let m = self.settings.refine_margin;
        let local = ops::generate_bounding_boxes(
            &mut diff,
            pd,
            p,
            Stride::Auto,
            [m, m, m],
            3,
            0,
            &mut self.rng,
        );
        if local.is_empty() {
            return fallback();
        }
        local
            .iter()
            .map(|b| std::array::from_fn(|a| [b[a][0] + planning[a][0], b[a][1] + planning[a][0]]))
            .collect()
    }
}

/// `src` (dims `sd`) over `b` (in its own coordinates, inside it).
fn crop(src: &[u8], sd: [usize; 3], b: &BBox) -> Vec<u8> {
    let d = ops::bbox_size(b);
    let mut out = Vec::with_capacity(d[0] * d[1] * d[2]);
    for i in b[0][0]..b[0][1] {
        for j in b[1][0]..b[1][1] {
            let s = ops::idx(sd, i as usize, j as usize, b[2][0] as usize);
            out.extend_from_slice(&src[s..s + d[2]]);
        }
    }
    out
}

/// `_detect_change_at_border`: did the answer change along any face of the
/// patch - more than 1500 voxels, or more than 100 and by more than 20 %?
fn border_change(pred: &[u8], previous: &[f32], p: [usize; 3]) -> bool {
    for dim in 0..3 {
        for &at in &[0, p[dim] - 1] {
            let (mut sp, mut sc, mut sd) = (0f64, 0f64, 0f64);
            for i in 0..p[0] {
                for j in 0..p[1] {
                    for k in 0..p[2] {
                        if [i, j, k][dim] != at {
                            continue;
                        }
                        let v = ops::idx(p, i, j, k);
                        let a = f64::from(previous[v]);
                        let b = f64::from(pred[v]);
                        sp += a;
                        sc += b;
                        sd += f64::from(u8::from(a != b));
                    }
                }
            }
            let rel = sp.max(sc) / sp.min(sc).max(1e-5) - 1.0;
            if sd > 1500.0 || (sd > 100.0 && rel > 0.2) {
                return true;
            }
        }
    }
    false
}

/// z-score with the statistics of the nonzero bounding box: upstream
/// normalizes the whole image but takes mean and (unbiased) standard
/// deviation where nnU-Net's training crop would have been.
fn normalize(mut image: Vec<f32>, dims: [usize; 3]) -> Result<Vec<f32>> {
    let Some(b) = ops::nonzero_bbox(&image, dims) else {
        bail!("the image is entirely zero");
    };
    let (mut sum, mut n) = (0f64, 0usize);
    for i in b[0][0]..b[0][1] {
        for j in b[1][0]..b[1][1] {
            for k in b[2][0]..b[2][1] {
                sum += f64::from(image[ops::idx(dims, i as usize, j as usize, k as usize)]);
                n += 1;
            }
        }
    }
    let mean = sum / n as f64;
    let mut ss = 0f64;
    for i in b[0][0]..b[0][1] {
        for j in b[1][0]..b[1][1] {
            for k in b[2][0]..b[2][1] {
                let d = f64::from(image[ops::idx(dims, i as usize, j as usize, k as usize)]) - mean;
                ss += d * d;
            }
        }
    }
    let std = (ss / (n.max(2) - 1) as f64).sqrt();
    let (m, s) = (mean as f32, (std as f32).max(f32::MIN_POSITIVE));
    for v in &mut image {
        *v = (*v - m) / s;
    }
    Ok(image)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stand-in network: label 1 wherever the point channel is set.
    struct PointsOnly;
    impl Predictor for PointsOnly {
        fn predict(&self, input: &[f32], p: [usize; 3]) -> Result<Vec<u8>> {
            let n = p[0] * p[1] * p[2];
            Ok((0..n).map(|v| u8::from(input[4 * n + v] > 0.0)).collect())
        }
    }

    fn session() -> Session {
        let dims = [20, 22, 18];
        let image: Vec<f32> = (0..dims[0] * dims[1] * dims[2])
            .map(|v| (v % 7) as f32 + 1.0)
            .collect();
        Session::new(image, dims, Settings::v1([8, 8, 8])).unwrap()
    }

    #[test]
    fn a_click_is_drawn_at_the_newest_intensity() {
        let mut s = session();
        s.add_point([10.0, 11.0, 9.0], true).unwrap();
        let ch = s.render(3, &[[6, 15], [7, 16], [5, 14]]);
        let peak = ch.iter().cloned().fold(0.0, f32::max);
        assert_eq!(peak, h((1.0 / 0.98) as f32));
        let mut n = ch.clone();
        s.normalize_channel(&mut n);
        let top = n.iter().cloned().fold(0.0, f32::max);
        assert!((top - 1.0).abs() < 2e-3, "{top}");
        // the negative channel is untouched
        assert!(s
            .render(4, &[[0, 20], [0, 22], [0, 18]])
            .iter()
            .all(|&v| v == 0.0));
    }

    #[test]
    fn a_prediction_lands_and_undo_takes_it_back() {
        let mut s = session();
        s.settings.autozoom = false;
        s.add_point([10.0, 11.0, 9.0], true).unwrap();
        let out = s.predict(&PointsOnly, false, &Progress::default()).unwrap();
        assert_eq!(out.passes, 1);
        let set = s.mask.iter().filter(|&&v| v != 0).count();
        // the ball of radius 4 minus its zero-valued rim
        assert!(set > 100 && set < 400, "{set}");
        assert!(s.undo());
        assert!(s.mask.iter().all(|&v| v == 0));
        assert_eq!(s.n_prompts(), 0);
    }

    #[test]
    fn boxes_must_be_flat_and_inside() {
        let mut s = session();
        assert!(s.add_box([2.0, 2.0, 2.0], [6.0, 6.0, 6.0], true).is_err());
        s.add_box([2.0, 2.0, 4.0], [12.0, 30.0, 5.0], true).unwrap();
        // clipped to the image along axis 1
        let ch = s.render(1, &[[0, 20], [0, 22], [0, 18]]);
        assert_eq!(ch.iter().filter(|&&v| v > 0.0).count(), 10 * 20);
    }

    #[test]
    fn normalization_uses_the_nonzero_box() {
        let dims = [4, 4, 4];
        let mut image = vec![0f32; 64];
        for i in 1..3 {
            for j in 0..4 {
                for k in 0..4 {
                    image[ops::idx(dims, i, j, k)] = if j < 2 { 1.0 } else { 3.0 };
                }
            }
        }
        let s = Session::new(image, dims, Settings::v1([4, 4, 4])).unwrap();
        // inside the box: mean 2, unbiased std over 32 voxels
        let std = (32.0f64 / 31.0).sqrt() as f32;
        assert!((s.image[ops::idx(dims, 1, 0, 0)] - (-1.0 / std)).abs() < 1e-6);
        // outside it the zeros are normalized with the same constants
        assert!((s.image[0] - (-2.0 / std)).abs() < 1e-6);
    }
}
