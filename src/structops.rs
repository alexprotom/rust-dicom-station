//! Structure algebra: combining contours and segmentations into new ones.
//!
//! Every planning department does this a dozen times a day. `Lungs` is
//! `Lung_L ∪ Lung_R`. `PTV` is `CTV` grown by a margin. `PTV_eval` is that
//! same PTV cropped 5 mm inside the body. A ring for an optimiser objective
//! is one expansion minus a smaller one. None of it is difficult; all of it
//! is tedious and error-prone by hand, and none of it should depend on
//! whether the thing you want to combine happens to be a contour or a
//! painted mask.
//!
//! So this module works in one currency - a binary mask on one lattice - and
//! the caller converts on the way in and out. An RT structure is rasterized
//! onto the grid ([`segmentation::rasterize_roi`]), a segmentation is already
//! there, and the result goes back as either kind. Mixing them is then not a
//! special case but the normal one.
//!
//! A recipe is read left to right:
//!
//! ```text
//! (A ± margin)  op  (B ± margin)  op  …   →  ± margin  →  cleanup
//! ```
//!
//! The per-operand margin is what makes the whole thing expressive rather
//! than merely convenient: *crop to* is an intersection whose second operand
//! was shrunk first, and a ring is a subtraction between two expansions of
//! the same structure.

use anyhow::{bail, Result};
use rayon::prelude::*;

use crate::morphology::{self as morph, Radii};
use crate::volume::Grid;

/// How two masks are put together.
///
/// Applied left to right over the operand list, so three operands under
/// [`BoolOp::Subtract`] mean `A − B − C`, which is what everyone expects and
/// is why the order of the list is worth showing in the interface.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoolOp {
    /// In any of them.
    Union,
    /// In all of them.
    Intersect,
    /// In the first and none of the rest.
    Subtract,
    /// In an odd number of them.
    Xor,
}

impl BoolOp {
    pub const ALL: [BoolOp; 4] = [
        BoolOp::Union,
        BoolOp::Intersect,
        BoolOp::Subtract,
        BoolOp::Xor,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            BoolOp::Union => "Union (A ∪ B)",
            BoolOp::Intersect => "Intersection (A ∩ B)",
            BoolOp::Subtract => "Subtraction (A − B)",
            BoolOp::Xor => "Symmetric difference (A ⊕ B)",
        }
    }

    /// How the operand list reads, for the line above the buttons.
    pub fn joiner(&self) -> &'static str {
        match self {
            BoolOp::Union => "∪",
            BoolOp::Intersect => "∩",
            BoolOp::Subtract => "−",
            BoolOp::Xor => "⊕",
        }
    }

    fn apply(&self, acc: &mut [u8], rhs: &[u8], first: bool) {
        if first {
            acc.copy_from_slice(rhs);
            return;
        }
        match self {
            BoolOp::Union => acc
                .par_iter_mut()
                .zip(rhs.par_iter())
                .for_each(|(a, &b)| *a |= b),
            BoolOp::Intersect => acc
                .par_iter_mut()
                .zip(rhs.par_iter())
                .for_each(|(a, &b)| *a &= b),
            BoolOp::Subtract => acc
                .par_iter_mut()
                .zip(rhs.par_iter())
                .for_each(|(a, &b)| *a &= 1 - b),
            BoolOp::Xor => acc
                .par_iter_mut()
                .zip(rhs.par_iter())
                .for_each(|(a, &b)| *a ^= b),
        }
    }
}

/// A margin in millimetres, in **patient** directions rather than array axes.
///
/// That distinction is the whole reason this type exists. "8 mm superiorly"
/// has to mean the same thing on an axial CT, a coronal MR and an obliquely
/// acquired series; the direction cosines decide which array axis that is and
/// which way along it. Positive grows, negative shrinks, and the two may be
/// mixed - the expansion runs first, then the contraction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Margin {
    pub right: f64,
    pub left: f64,
    pub anterior: f64,
    pub posterior: f64,
    pub superior: f64,
    pub inferior: f64,
}

impl Default for Margin {
    fn default() -> Self {
        Margin::NONE
    }
}

impl Margin {
    pub const NONE: Margin = Margin::uniform(0.0);

    pub const fn uniform(mm: f64) -> Margin {
        Margin {
            right: mm,
            left: mm,
            anterior: mm,
            posterior: mm,
            superior: mm,
            inferior: mm,
        }
    }

    pub fn is_none(&self) -> bool {
        self.all().iter().all(|v| v.abs() < 1e-9)
    }

    /// True when one number describes it, which is what the interface offers
    /// until the user asks for more.
    pub fn is_uniform(&self) -> bool {
        let v = self.all();
        v.iter().all(|x| (x - v[0]).abs() < 1e-9)
    }

    /// Right, left, anterior, posterior, superior, inferior.
    pub fn all(&self) -> [f64; 6] {
        [
            self.right,
            self.left,
            self.anterior,
            self.posterior,
            self.superior,
            self.inferior,
        ]
    }

    /// `+5.0 mm`, or the six values when they differ.
    pub fn describe(&self) -> String {
        if self.is_none() {
            "none".to_string()
        } else if self.is_uniform() {
            format!("{:+.1} mm", self.right)
        } else {
            format!(
                "R{:+.0} L{:+.0} A{:+.0} P{:+.0} S{:+.0} I{:+.0}",
                self.right, self.left, self.anterior, self.posterior, self.superior, self.inferior
            )
        }
    }

    /// Split into the part that grows and the part that shrinks, each mapped
    /// onto the lattice's own axes and directions.
    ///
    /// Canonical axis 0 is superior, 1 anterior, 2 right (see
    /// [`Grid::canonical_axes`]); `flip` says that the canonical direction
    /// runs *against* the array index, which swaps the pair.
    fn to_radii(self, grid: &Grid) -> (Radii, Radii) {
        let (perm, flip) = grid.canonical_axes();
        // Per canonical axis: (toward +canonical, toward −canonical).
        let pairs = [
            (self.superior, self.inferior),
            (self.anterior, self.posterior),
            (self.right, self.left),
        ];
        let mut grow: Radii = [[0.0; 2]; 3];
        let mut shrink: Radii = [[0.0; 2]; 3];
        for (c, (plus, minus)) in pairs.into_iter().enumerate() {
            let axis = perm[c];
            // radii[axis] = [toward decreasing index, toward increasing index]
            let (lo, hi) = if flip[c] {
                (plus, minus)
            } else {
                (minus, plus)
            };
            grow[axis] = [lo.max(0.0), hi.max(0.0)];
            shrink[axis] = [(-lo).max(0.0), (-hi).max(0.0)];
        }
        (grow, shrink)
    }

    /// Grow, then shrink. Both are no-ops when their half of the margin is
    /// zero, so the common uniform case costs one distance transform.
    pub fn apply(&self, mask: &[u8], grid: &Grid, sink: &dyn ProgressSink) -> Vec<u8> {
        if self.is_none() {
            return mask.to_vec();
        }
        let (grow, shrink) = self.to_radii(grid);
        let mut out = mask.to_vec();
        if morph_any(&grow) {
            sink.report(0.0, "Expanding");
            out = morph::dilate_radii(&out, grid.dims, grid.spacing, &grow);
        }
        if morph_any(&shrink) {
            sink.report(0.5, "Contracting");
            out = morph::erode_radii(&out, grid.dims, grid.spacing, &shrink);
        }
        out
    }
}

fn morph_any(r: &Radii) -> bool {
    r.iter().flatten().any(|v| *v > 0.0)
}

/// Tidying applied to the finished mask.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cleanup {
    /// Close interior cavities, slice by slice - the same reasoning as the
    /// body contour: a lung drains through the trachea, so a three-
    /// dimensional fill would leave it open.
    pub fill_holes: bool,
    /// Morphological closing, to take the staircase off a surface.
    pub close_mm: f64,
    /// Discard everything but the largest connected piece.
    pub keep_largest: bool,
    /// …or, less bluntly, discard pieces below this volume. Ignored when
    /// `keep_largest` is set.
    pub min_volume_cm3: f64,
}

impl Default for Cleanup {
    fn default() -> Self {
        Cleanup {
            fill_holes: false,
            close_mm: 0.0,
            keep_largest: false,
            min_volume_cm3: 0.0,
        }
    }
}

impl Cleanup {
    pub fn is_none(&self) -> bool {
        !self.fill_holes && self.close_mm <= 0.0 && !self.keep_largest && self.min_volume_cm3 <= 0.0
    }

    pub fn apply(&self, mask: &mut Vec<u8>, grid: &Grid, sink: &dyn ProgressSink) {
        if self.close_mm > 0.0 {
            sink.report(0.0, "Smoothing");
            *mask = morph::close_mm(mask, grid.dims, grid.spacing, self.close_mm);
        }
        if self.fill_holes {
            sink.report(0.4, "Filling cavities");
            morph::fill_holes_2d(mask, grid.dims, grid.canonical_axes().0[0]);
        }
        if self.keep_largest || self.min_volume_cm3 > 0.0 {
            sink.report(0.7, "Dropping small pieces");
            let voxel_cm3 = grid.spacing[0] * grid.spacing[1] * grid.spacing[2] / 1000.0;
            let comps = morph::components(mask, grid.dims);
            let keep: Vec<&morph::Component> = if self.keep_largest {
                comps.iter().take(1).collect()
            } else {
                let min = (self.min_volume_cm3 / voxel_cm3).max(1.0) as usize;
                comps.iter().filter(|c| c.len() >= min).collect()
            };
            mask.fill(0);
            for c in keep {
                for &v in &c.voxels {
                    mask[v as usize] = 1;
                }
            }
        }
    }
}

/// One input to a recipe: a mask on the recipe's lattice, and the margin
/// applied to it before it is combined with the others.
pub struct Operand {
    /// Shown in messages and in the summary line.
    pub name: String,
    pub mask: Vec<u8>,
    pub margin: Margin,
}

/// What to compute.
pub struct Recipe {
    pub op: BoolOp,
    pub operands: Vec<Operand>,
    /// Applied to the combined result.
    pub margin: Margin,
    pub cleanup: Cleanup,
}

use crate::progress::ProgressSink;

/// What a finished recipe hands back. The mask is named rather than printed
/// in the `Debug` output, which is otherwise tens of megabytes of ones.
pub struct Combined {
    pub mask: Vec<u8>,
    pub voxels: u64,
    pub cm3: f64,
    /// Separate pieces in the result - worth saying, because a subtraction
    /// that cuts a structure in two is rarely what was intended.
    pub pieces: usize,
}

/// Evaluate a recipe on `grid`.
///
/// Every operand must already be a mask on that lattice; converting a
/// contour is the caller's job, since only it knows where the contour came
/// from.
impl std::fmt::Debug for Combined {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Combined")
            .field("mask", &format_args!("<{} voxels>", self.mask.len()))
            .field("voxels", &self.voxels)
            .field("cm3", &self.cm3)
            .field("pieces", &self.pieces)
            .finish()
    }
}

pub fn combine(recipe: &Recipe, grid: &Grid, sink: &dyn ProgressSink) -> Result<Combined> {
    let n = grid.dims[0] * grid.dims[1] * grid.dims[2];
    if recipe.operands.is_empty() {
        bail!("nothing to combine - add at least one structure");
    }
    if recipe.op == BoolOp::Subtract && recipe.operands.len() < 2 {
        bail!("a subtraction needs something to subtract");
    }
    for o in &recipe.operands {
        if o.mask.len() != n {
            bail!(
                "'{}' is on a different lattice ({} voxels, expected {n})",
                o.name,
                o.mask.len()
            );
        }
    }
    let steps = recipe.operands.len() as f32 + 2.0;
    let mut acc = vec![0u8; n];
    for (i, operand) in recipe.operands.iter().enumerate() {
        sink.report(i as f32 / steps, &format!("Preparing '{}'", operand.name));
        if sink.cancelled() {
            bail!(crate::progress::CANCELLED);
        }
        // Normalize to 0/1 first: a mask that arrived as a label map would
        // otherwise make `^` and `&` mean something else entirely.
        let mut m: Vec<u8> = operand.mask.par_iter().map(|&v| u8::from(v != 0)).collect();
        if !operand.margin.is_none() {
            m = operand.margin.apply(&m, grid, &crate::progress::Quiet);
        }
        recipe.op.apply(&mut acc, &m, i == 0);
    }
    if sink.cancelled() {
        bail!(crate::progress::CANCELLED);
    }
    if !recipe.margin.is_none() {
        sink.report(recipe.operands.len() as f32 / steps, "Applying the margin");
        acc = recipe.margin.apply(&acc, grid, &crate::progress::Quiet);
    }
    if !recipe.cleanup.is_none() {
        sink.report((recipe.operands.len() as f32 + 1.0) / steps, "Cleaning up");
        recipe
            .cleanup
            .apply(&mut acc, grid, &crate::progress::Quiet);
    }
    let voxels: u64 = acc.par_iter().map(|&v| u64::from(v != 0)).sum();
    let pieces = if voxels == 0 {
        0
    } else {
        morph::components(&acc, grid.dims).len()
    };
    let voxel_cm3 = grid.spacing[0] * grid.spacing[1] * grid.spacing[2] / 1000.0;
    sink.report(1.0, "Done");
    Ok(Combined {
        mask: acc,
        voxels,
        cm3: voxels as f64 * voxel_cm3,
        pieces,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Vec3;
    use crate::progress::Quiet;

    fn grid(dims: [usize; 3], spacing: [f64; 3]) -> Grid {
        Grid {
            dims,
            spacing,
            origin: Vec3::new(0.0, 0.0, 0.0),
            row_dir: Vec3::new(1.0, 0.0, 0.0),
            col_dir: Vec3::new(0.0, 1.0, 0.0),
            normal: Vec3::new(0.0, 0.0, 1.0),
            frame_of_reference_uid: "1.2.3".into(),
        }
    }

    fn operand(name: &str, mask: Vec<u8>) -> Operand {
        Operand {
            name: name.into(),
            mask,
            margin: Margin::NONE,
        }
    }

    fn run(op: BoolOp, masks: Vec<Vec<u8>>, g: &Grid) -> Vec<u8> {
        let recipe = Recipe {
            op,
            operands: masks
                .into_iter()
                .enumerate()
                .map(|(i, m)| operand(&format!("s{i}"), m))
                .collect(),
            margin: Margin::NONE,
            cleanup: Cleanup::default(),
        };
        combine(&recipe, g, &Quiet).expect("a result").mask
    }

    #[test]
    fn the_four_operations_are_the_four_operations() {
        let g = grid([4, 1, 1], [1.0; 3]);
        let a = vec![1, 1, 0, 0];
        let b = vec![0, 1, 1, 0];
        assert_eq!(
            run(BoolOp::Union, vec![a.clone(), b.clone()], &g),
            [1, 1, 1, 0]
        );
        assert_eq!(
            run(BoolOp::Intersect, vec![a.clone(), b.clone()], &g),
            [0, 1, 0, 0]
        );
        assert_eq!(
            run(BoolOp::Subtract, vec![a.clone(), b.clone()], &g),
            [1, 0, 0, 0]
        );
        assert_eq!(run(BoolOp::Xor, vec![a, b], &g), [1, 0, 1, 0]);
    }

    #[test]
    fn more_than_two_operands_fold_left_to_right() {
        let g = grid([5, 1, 1], [1.0; 3]);
        let a = vec![1, 1, 1, 1, 1];
        let b = vec![0, 1, 0, 0, 0];
        let c = vec![0, 0, 0, 1, 0];
        // A − B − C, not A − (B − C).
        assert_eq!(run(BoolOp::Subtract, vec![a, b, c], &g), [1, 0, 1, 0, 1]);
    }

    #[test]
    fn a_label_map_operand_is_read_as_a_mask_not_as_numbers() {
        // Values other than 0/1 must not survive into `^` or `&`.
        let g = grid([3, 1, 1], [1.0; 3]);
        let a = vec![7, 0, 3];
        let b = vec![9, 9, 0];
        assert_eq!(run(BoolOp::Xor, vec![a, b], &g), [0, 1, 1]);
    }

    #[test]
    fn a_margin_is_measured_in_patient_directions_not_array_axes() {
        // Two lattices of the same anatomy, the second with its rows running
        // the other way. A 4 mm superior margin has to land on the same side
        // of the patient in both.
        let dims = [3, 3, 9];
        let mut mask = vec![0u8; 81];
        let at = |i: usize, j: usize, k: usize| k * 9 + j * 3 + i;
        mask[at(1, 1, 4)] = 1;
        let mut up = grid(dims, [2.0, 2.0, 2.0]);
        up.normal = Vec3::new(0.0, 0.0, 1.0); // +k is superior
        let mut down = up.clone();
        down.normal = Vec3::new(0.0, 0.0, -1.0); // +k is inferior

        let m = Margin {
            superior: 4.0,
            ..Margin::NONE
        };
        let a = m.apply(&mask, &up, &Quiet);
        let b = m.apply(&mask, &down, &Quiet);
        assert_eq!(a[at(1, 1, 6)], 1, "grew toward +k when +k is superior");
        assert_eq!(a[at(1, 1, 2)], 0, "and not the other way");
        assert_eq!(b[at(1, 1, 2)], 1, "grew toward −k when −k is superior");
        assert_eq!(b[at(1, 1, 6)], 0, "and not the other way");
    }

    #[test]
    fn a_negative_margin_shrinks_and_a_mixed_one_does_both() {
        let dims = [21, 21, 1];
        let g = grid(dims, [1.0, 1.0, 1.0]);
        let at = |i: usize, j: usize| j * 21 + i;
        let mut mask = vec![0u8; 441];
        for j in 5..16 {
            for i in 5..16 {
                mask[at(i, j)] = 1;
            }
        }
        let shrunk = Margin::uniform(-2.0).apply(&mask, &g, &Quiet);
        assert_eq!(shrunk[at(10, 10)], 1, "the middle survives");
        assert_eq!(shrunk[at(6, 10)], 0, "the edge moved in");
        // Grow right, shrink left: the two halves move independently.
        let mixed = Margin {
            right: 3.0,
            left: -3.0,
            ..Margin::NONE
        }
        .apply(&mask, &g, &Quiet);
        // +x is Left in LPS, so "right" grows toward decreasing i.
        assert_eq!(mixed[at(3, 10)], 1, "grew to the patient's right");
        assert_eq!(mixed[at(15, 10)], 0, "shrank on the patient's left");
    }

    #[test]
    fn cleanup_fills_closes_and_prunes() {
        let dims = [30, 30, 1];
        let g = grid(dims, [1.0, 1.0, 1.0]);
        let at = |i: usize, j: usize| j * 30 + i;
        let mut mask = vec![0u8; 900];
        // A ring, plus a speck far away.
        for j in 3..18 {
            for i in 3..18 {
                mask[at(i, j)] = 1;
            }
        }
        for j in 8..13 {
            for i in 8..13 {
                mask[at(i, j)] = 0;
            }
        }
        mask[at(27, 27)] = 1;
        let mut filled = mask.clone();
        Cleanup {
            fill_holes: true,
            ..Cleanup::default()
        }
        .apply(&mut filled, &g, &Quiet);
        assert_eq!(filled[at(10, 10)], 1, "the hole closed");
        assert_eq!(filled[at(27, 27)], 1, "the speck is still there");

        let mut pruned = mask.clone();
        Cleanup {
            keep_largest: true,
            ..Cleanup::default()
        }
        .apply(&mut pruned, &g, &Quiet);
        assert_eq!(pruned[at(4, 4)], 1, "the ring survived");
        assert_eq!(pruned[at(27, 27)], 0, "the speck did not");
    }

    #[test]
    fn a_crop_is_an_intersection_with_a_shrunken_operand() {
        // PTV_eval = PTV ∩ (BODY − 5 mm), the everyday use of a per-operand
        // margin.
        let dims = [41, 5, 1];
        let g = grid(dims, [1.0, 1.0, 1.0]);
        let at = |i: usize| 2 * 41 + i;
        let mut body = vec![0u8; 205];
        let mut ptv = vec![0u8; 205];
        for j in 0..5 {
            for i in 5..36 {
                body[j * 41 + i] = 1;
            }
            for i in 3..20 {
                ptv[j * 41 + i] = 1;
            }
        }
        let out = combine(
            &Recipe {
                op: BoolOp::Intersect,
                operands: vec![
                    operand("PTV", ptv),
                    Operand {
                        name: "BODY".into(),
                        mask: body,
                        margin: Margin::uniform(-5.0),
                    },
                ],
                margin: Margin::NONE,
                cleanup: Cleanup::default(),
            },
            &g,
            &Quiet,
        )
        .expect("a result");
        assert_eq!(out.mask[at(9)], 0, "outside the shrunken body");
        assert_eq!(out.mask[at(11)], 1, "inside both");
        assert_eq!(out.mask[at(25)], 0, "outside the PTV");
        assert_eq!(out.pieces, 1);
    }

    #[test]
    fn a_ring_is_the_difference_of_two_expansions() {
        let dims = [41, 41, 1];
        let g = grid(dims, [1.0, 1.0, 1.0]);
        let at = |i: usize, j: usize| j * 41 + i;
        let mut core = vec![0u8; 1681];
        for j in 18..23 {
            for i in 18..23 {
                core[at(i, j)] = 1;
            }
        }
        let out = combine(
            &Recipe {
                op: BoolOp::Subtract,
                operands: vec![
                    Operand {
                        name: "outer".into(),
                        mask: core.clone(),
                        margin: Margin::uniform(10.0),
                    },
                    Operand {
                        name: "inner".into(),
                        mask: core,
                        margin: Margin::uniform(4.0),
                    },
                ],
                margin: Margin::NONE,
                cleanup: Cleanup::default(),
            },
            &g,
            &Quiet,
        )
        .expect("a result");
        assert_eq!(out.mask[at(20, 20)], 0, "hollow in the middle");
        assert_eq!(out.mask[at(20, 12)], 1, "solid in the ring");
        assert_eq!(out.mask[at(20, 4)], 0, "and nothing beyond it");
    }

    #[test]
    fn an_empty_or_mismatched_recipe_is_refused_not_guessed() {
        let g = grid([4, 4, 1], [1.0; 3]);
        let empty = Recipe {
            op: BoolOp::Union,
            operands: Vec::new(),
            margin: Margin::NONE,
            cleanup: Cleanup::default(),
        };
        assert!(combine(&empty, &g, &Quiet).is_err());
        let lone = Recipe {
            op: BoolOp::Subtract,
            operands: vec![operand("A", vec![0u8; 16])],
            margin: Margin::NONE,
            cleanup: Cleanup::default(),
        };
        assert!(combine(&lone, &g, &Quiet).is_err());
        let wrong = Recipe {
            op: BoolOp::Union,
            operands: vec![operand("A", vec![0u8; 9])],
            margin: Margin::NONE,
            cleanup: Cleanup::default(),
        };
        let err = format!("{:#}", combine(&wrong, &g, &Quiet).unwrap_err());
        assert!(err.contains("lattice"), "{err}");
    }
}
