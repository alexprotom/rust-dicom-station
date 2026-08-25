//! Turning a study into what the network was trained on.
//!
//! MedSAM2's preprocessing is short, and every step of it matters because it
//! defines the input distribution the weights were fitted to:
//!
//! 1. clip the volume to a **HU window**;
//! 2. min-max the *whole clipped volume* to `[0, 255]` and **quantize to
//!    `u8`** — this is not a formality, the network never saw anything finer;
//! 3. resize each slice to 512 x 512 with `PIL.Image.resize`, whose default is
//!    a bicubic kernel with `a = -0.5` (see [`super::resample`]);
//! 4. divide by 255 and normalize with the ImageNet statistics.
//!
//! Two things are deliberately *not* done, because the reference does not do
//! them: no resampling to a target spacing, and no foreground cropping. The
//! nnU-Net-style pipeline in [`crate::autoseg`] and the statistics-based one
//! in [`crate::segvol`] would both quietly change the distribution.
//!
//! The window is the one thing this port sources differently. The reference
//! reads a per-lesion window out of a CSV; RDS has one on screen already, so
//! [`Window`] takes the viewport's window/level — what you see is what the
//! model sees — with the paper's presets available by name.
//!
//! ## Geometry
//!
//! Slices are taken along the patient's superior axis and oriented the way a
//! radiologist reads them: rows run anterior to posterior, columns right to
//! left. That is [`crate::autoseg::preprocess::canonical_axes`]'s `[S, A, R]`
//! with the last two axes flipped, and for an ordinary head-first-supine CT it
//! is exactly the acquisition order, so nothing is moved at all.

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use rayon::prelude::*;

use crate::autoseg::preprocess::canonical_axes;
use crate::volume::Volume;

use super::config;
use super::infer::Slices;
use super::ops;
use super::resample::{self, Filter};


/// An intensity window, in the volume's own units (HU for CT).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Window {
    pub lower: f32,
    pub upper: f32,
}

impl Window {
    pub fn new(lower: f32, upper: f32) -> Window {
        Window { lower, upper }
    }

    /// From the viewer's window width and level.
    pub fn from_width_level(width: f32, level: f32) -> Window {
        Window {
            lower: level - width / 2.0,
            upper: level + width / 2.0,
        }
    }

    /// The windows the MedSAM2 paper used to build its CT training corpus,
    /// as `(name, width, level)`.
    pub const PRESETS: [(&'static str, f32, f32); 5] = [
        ("Brain", 80.0, 40.0),
        ("Abdomen", 400.0, 40.0),
        ("Bone", 1800.0, 400.0),
        ("Lung", 1500.0, -600.0),
        ("Mediastinum", 400.0, 40.0),
    ];

    pub fn preset(name: &str) -> Option<Window> {
        Self::PRESETS
            .iter()
            .find(|(n, _, _)| *n == name)
            .map(|(_, w, l)| Window::from_width_level(*w, *l))
    }
}

/// A study, windowed and oriented, ready to be sliced for the network.
pub struct Prepared {
    /// One byte per pixel per slice, `dims[0]` slices of `dims[1] x dims[2]`.
    pub slices: Vec<Vec<u8>>,
    /// `[slices, rows, columns]` of the oriented stack.
    pub dims: [usize; 3],
    /// Voxel spacing along those same axes, in millimetres.
    pub spacing: [f64; 3],
    /// Oriented axis -> volume axis.
    perm: [usize; 3],
    /// Oriented axis runs opposite to the volume axis.
    flip: [bool; 3],
    pub window: Window,
}

/// Axial axes: the superior axis for slices, then anterior-to-posterior rows
/// and right-to-left columns.
/// Where a volume voxel lands in the oriented stack.
fn reorient_index(
    voxel: [usize; 3],
    perm: [usize; 3],
    flip: [bool; 3],
    dims: [usize; 3],
) -> [usize; 3] {
    let mut out = [0usize; 3];
    for a in 0..3 {
        let v = voxel[perm[a]];
        out[a] = if flip[a] { dims[a] - 1 - v } else { v };
    }
    out
}

/// The same, for a volume that has not been prepared yet.
///
/// The user interface needs this to turn a box drawn on screen into slice and
/// pixel numbers *before* paying for [`Prepared::prepare`] — and it agrees
/// with [`Prepared::from_volume_index`] by construction, since both are
/// [`reorient_index`].
pub fn volume_index_to_prepared(vol: &Volume, voxel: [usize; 3]) -> [usize; 3] {
    let (perm, flip) = axial_axes(vol);
    let dims = [vol.dims[perm[0]], vol.dims[perm[1]], vol.dims[perm[2]]];
    reorient_index(voxel, perm, flip, dims)
}

/// Which volume axis becomes which prepared axis, and whether it is reversed.
///
/// Exposed because the user interface needs to know *before* anything is
/// prepared which of the three views a prompt has to be drawn in: the one
/// whose slices are the ones the network will propagate through.
pub fn axial_axes(vol: &Volume) -> ([usize; 3], [bool; 3]) {
    let (perm, mut flip) = canonical_axes(vol);
    // `canonical_axes` targets [S, A, R]; reading order is [S, P, L].
    flip[1] = !flip[1];
    flip[2] = !flip[2];
    (perm, flip)
}

impl Prepared {
    /// Window, quantize and orient. This is the whole of steps 1 and 2.
    pub fn prepare(vol: &Volume, window: Window) -> Prepared {
        let (perm, flip) = axial_axes(vol);
        let dims = [
            vol.dims[perm[0]],
            vol.dims[perm[1]],
            vol.dims[perm[2]],
        ];

        // The reference min-maxes the clipped volume, which is not quite the
        // window itself: a study that never reaches the window's ends maps its
        // own extremes to 0 and 255.
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for v in &vol.data {
            let v = (*v as f32).clamp(window.lower, window.upper);
            lo = lo.min(v);
            hi = hi.max(v);
        }
        let span = (hi - lo).max(1e-6);

        let (nx, ny) = (vol.dims[0], vol.dims[1]);
        let [d0, d1, d2] = dims;
        let mut slices: Vec<Vec<u8>> = (0..d0).map(|_| vec![0u8; d1 * d2]).collect();
        slices.par_iter_mut().enumerate().for_each(|(a0, slab)| {
            let mut idx = [0usize; 3];
            idx[perm[0]] = if flip[0] { d0 - 1 - a0 } else { a0 };
            for a1 in 0..d1 {
                idx[perm[1]] = if flip[1] { d1 - 1 - a1 } else { a1 };
                for a2 in 0..d2 {
                    idx[perm[2]] = if flip[2] { d2 - 1 - a2 } else { a2 };
                    let src = idx[2] * nx * ny + idx[1] * nx + idx[0];
                    let v = (vol.data[src] as f32).clamp(window.lower, window.upper);
                    // `np.uint8(x)` truncates; the values are already in
                    // [0, 255] so this is the reference's own rounding.
                    slab[a1 * d2 + a2] = ((v - lo) / span * 255.0) as u8;
                }
            }
        });

        Prepared {
            slices,
            dims,
            spacing: [
                vol.spacing[perm[0]],
                vol.spacing[perm[1]],
                vol.spacing[perm[2]],
            ],
            perm,
            flip,
            window,
        }
    }

    /// Read one oriented slice out of a mask that lives on the volume's grid
    /// — the inverse of [`Self::mask_to_volume_grid`], for one slice.
    ///
    /// This is what turns "the contour I already drew" into a mask prompt.
    pub fn slice_from_volume_mask(
        &self,
        mask: &[u8],
        vol: &Volume,
        slice: usize,
    ) -> Vec<u8> {
        let (nx, ny) = (vol.dims[0], vol.dims[1]);
        let [d0, d1, d2] = self.dims;
        let mut out = vec![0u8; d1 * d2];
        let mut idx = [0usize; 3];
        idx[self.perm[0]] = if self.flip[0] { d0 - 1 - slice } else { slice };
        for a1 in 0..d1 {
            idx[self.perm[1]] = if self.flip[1] { d1 - 1 - a1 } else { a1 };
            for a2 in 0..d2 {
                idx[self.perm[2]] = if self.flip[2] { d2 - 1 - a2 } else { a2 };
                out[a1 * d2 + a2] = mask[idx[2] * nx * ny + idx[1] * nx + idx[0]];
            }
        }
        out
    }

    /// In-plane size of a slice, `[rows, columns]`.
    pub fn size(&self) -> [usize; 2] {
        [self.dims[1], self.dims[2]]
    }

    /// Where a prepared pixel lands in the network's 512 x 512 input.
    pub fn to_network(&self, row: f32, column: f32) -> (f32, f32) {
        let size = config::IMAGE_SIZE as f32;
        (
            column * size / self.dims[2] as f32,
            row * size / self.dims[1] as f32,
        )
    }

    /// The oriented index of a volume voxel `[i, j, k]`.
    pub fn from_volume_index(&self, voxel: [usize; 3]) -> [usize; 3] {
        reorient_index(voxel, self.perm, self.flip, self.dims)
    }

    /// Scatter per-slice masks back onto the volume's own grid.
    pub fn mask_to_volume_grid(&self, masks: &[Vec<u8>], vol: &Volume) -> Vec<u8> {
        let (nx, ny, nz) = (vol.dims[0], vol.dims[1], vol.dims[2]);
        let mut out = vec![0u8; nx * ny * nz];
        let [d0, d1, d2] = self.dims;
        for (a0, mask) in masks.iter().enumerate().take(d0) {
            if mask.is_empty() {
                continue;
            }
            let mut idx = [0usize; 3];
            idx[self.perm[0]] = if self.flip[0] { d0 - 1 - a0 } else { a0 };
            for a1 in 0..d1 {
                idx[self.perm[1]] = if self.flip[1] { d1 - 1 - a1 } else { a1 };
                for a2 in 0..d2 {
                    if mask[a1 * d2 + a2] == 0 {
                        continue;
                    }
                    idx[self.perm[2]] = if self.flip[2] { d2 - 1 - a2 } else { a2 };
                    out[idx[2] * nx * ny + idx[1] * nx + idx[0]] = 1;
                }
            }
        }
        out
    }

    /// Bind the stack to a device so it can be propagated through.
    pub fn stack<B: Backend>(&self, device: B::Device) -> Stack<'_, B> {
        Stack {
            prepared: self,
            device,
        }
    }
}

/// One windowed slice as the network wants it: resized, scaled and
/// normalized, `[1, 3, 512, 512]`.
///
/// Split out from [`Prepared`] so it can be tested against the reference
/// pipeline at a size that fits in a fixture.
pub fn slice_to_network<B: Backend>(
    slice: &[u8],
    size: [usize; 2],
    target: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    // PIL resizes the *8-bit* image, so the result is quantized back to
    // whole values before anything else touches it.
    let resized = resample::resize_u8(slice, size, [target, target], Filter::PIL_BICUBIC, true);
    let n = target * target;
    let mut data = vec![0f32; 3 * n];
    for c in 0..3 {
        let (mean, std) = (config::IMAGENET_MEAN[c], config::IMAGENET_STD[c]);
        for i in 0..n {
            data[c * n + i] = (f32::from(resized[i]) / 255.0 - mean) / std;
        }
    }
    ops::from_slice(&data, [1, 3, target, target], device)
}

/// A [`Prepared`] study bound to a device.
pub struct Stack<'a, B: Backend> {
    prepared: &'a Prepared,
    device: B::Device,
}

impl<B: Backend> Slices<B> for Stack<'_, B> {
    fn len(&self) -> usize {
        self.prepared.dims[0]
    }

    fn slice(&self, index: usize) -> Tensor<B, 4> {
        slice_to_network(
            &self.prepared.slices[index],
            self.prepared.size(),
            config::IMAGE_SIZE,
            &self.device,
        )
    }

    fn out_size(&self) -> [usize; 2] {
        self.prepared.size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Vec3;
    use crate::nn::cache::load_safetensors;
    use std::path::Path;

    type Bk = burn::backend::NdArray;

    fn volume(dims: [usize; 3], data: Vec<i16>) -> Volume {
        Volume {
            data,
            dims,
            spacing: [1.0, 1.0, 1.0],
            origin: Vec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            row_dir: Vec3 {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
            col_dir: Vec3 {
                x: 0.0,
                y: 1.0,
                z: 0.0,
            },
            normal: Vec3 {
                x: 0.0,
                y: 0.0,
                z: 1.0,
            },
            frame_of_reference_uid: String::new(),
            min_value: 0,
            max_value: 0,
        }
    }

    #[test]
    fn a_head_first_supine_study_is_not_reordered_at_all() {
        // row_dir +x, col_dir +y, normal +z is the ordinary case, and the
        // reading orientation is then the acquisition one.
        let v = volume([4, 3, 2], vec![0; 24]);
        let (perm, flip) = axial_axes(&v);
        assert_eq!(perm, [2, 1, 0], "slices along k, rows along j, columns along i");
        assert_eq!(flip, [false, false, false]);
    }

    #[test]
    fn windowing_maps_the_clipped_extremes_onto_the_whole_byte_range() {
        let data: Vec<i16> = vec![-1000, -100, 0, 100, 200, 3000];
        let v = volume([6, 1, 1], data);
        let p = Prepared::prepare(&v, Window::new(-100.0, 200.0));
        assert_eq!(p.dims, [1, 1, 6]);
        let row = &p.slices[0];
        assert_eq!(row[0], 0, "below the window");
        assert_eq!(row[1], 0, "the window's floor");
        assert_eq!(row[5], 255, "above the window");
        assert!(row[2] > 0 && row[2] < row[3] && row[3] < row[4]);
    }

    #[test]
    fn a_window_can_come_from_width_and_level() {
        assert_eq!(
            Window::from_width_level(400.0, 40.0),
            Window::new(-160.0, 240.0)
        );
        assert_eq!(Window::preset("Lung"), Some(Window::new(-1350.0, 150.0)));
        assert!(Window::preset("nonesuch").is_none());
    }

    #[test]
    fn a_mask_round_trips_through_the_volume_grid() {
        let v = volume([4, 3, 2], vec![0; 24]);
        let p = Prepared::prepare(&v, Window::new(0.0, 1.0));
        assert_eq!(p.dims, [2, 3, 4]);
        let mut masks: Vec<Vec<u8>> = (0..2).map(|_| vec![0u8; 12]).collect();
        // one voxel, at oriented (slice 1, row 2, column 3)
        masks[1][2 * 4 + 3] = 1;
        let grid = p.mask_to_volume_grid(&masks, &v);
        assert_eq!(grid.iter().filter(|x| **x != 0).count(), 1);
        // and back again
        assert_eq!(p.slice_from_volume_mask(&grid, &v, 1), masks[1]);
        assert!(p
            .slice_from_volume_mask(&grid, &v, 0)
            .iter()
            .all(|x| *x == 0));
        // which for this identity orientation is volume voxel (3, 2, 1)
        // k * nx * ny + j * nx + i, with k = 1, j = 2, i = 3
        assert_eq!(grid[12 + 8 + 3], 1);
        assert_eq!(p.from_volume_index([3, 2, 1]), [1, 2, 3]);
    }

    #[test]
    fn the_prompt_scales_with_the_slice_size() {
        let v = volume([256, 128, 1], vec![0; 256 * 128]);
        let p = Prepared::prepare(&v, Window::new(0.0, 1.0));
        assert_eq!(p.size(), [128, 256]);
        let (x, y) = p.to_network(64.0, 128.0);
        assert_eq!((x, y), (256.0, 256.0), "the centre stays the centre");
    }

    #[test]
    fn the_pipeline_matches_the_reference_preprocessing() {
        let dev: burn::tensor::Device<Bk> = Default::default();
        let f = load_safetensors(Path::new("tests/data/medsam2-ops.safetensors")).unwrap();
        let u8s = f.get("preprocess.u8").expect("fixture");
        let want = f.get("preprocess.y").expect("fixture");
        let (h, w) = (u8s.shape[0], u8s.shape[1]);
        let target = want.shape[want.shape.len() - 1];
        let slice: Vec<u8> = u8s.data.iter().map(|v| *v as u8).collect();
        let got = ops::to_vec(slice_to_network::<Bk>(&slice, [h, w], target, &dev));
        assert_eq!(got.len(), want.data.len());
        let worst = got
            .iter()
            .zip(want.data.iter())
            .map(|(a, b)| (a - b).abs() / (1.0 + b.abs()))
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-5, "relative error {worst:e}");
    }
}
