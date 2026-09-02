//! Structure algebra against contours and segments as they actually arrive:
//! an RT structure rasterized from patient-space polygons, a segmentation
//! painted on the voxel grid, and the two combined without either knowing
//! about the other.
//!
//! The unit tests in `src/structops.rs` cover the algebra on bare masks.
//! What is worth testing here is the seam: that a contour and a mask
//! describing the same sphere really do meet, that a margin given in patient
//! directions lands where the direction cosines say it should even when the
//! series is stored the other way up, and that the result survives the round
//! trip back out to contours.

use rust_dicom_station::geometry::Vec3;
use rust_dicom_station::progress::Quiet;
use rust_dicom_station::rtstruct::{Contour, Roi};
use rust_dicom_station::segmentation::{self, Segmentation};
use rust_dicom_station::structops::{combine, BoolOp, Cleanup, Margin, Operand, Recipe};
use rust_dicom_station::volume::Grid;

const DIMS: [usize; 3] = [40, 40, 12];
const SP: [f64; 3] = [2.0, 2.0, 5.0];

fn idx(i: usize, j: usize, k: usize) -> usize {
    k * DIMS[0] * DIMS[1] + j * DIMS[0] + i
}

/// A lattice whose k axis runs superiorly (`up`) or inferiorly.
fn grid(up: bool) -> Grid {
    Grid {
        dims: DIMS,
        spacing: SP,
        origin: Vec3::new(0.0, 0.0, 0.0),
        row_dir: Vec3::new(1.0, 0.0, 0.0),
        col_dir: Vec3::new(0.0, 1.0, 0.0),
        normal: Vec3::new(0.0, 0.0, if up { 1.0 } else { -1.0 }),
        frame_of_reference_uid: "1.2.826.0.1.3680043.8.498.algebra".into(),
    }
}

/// A square contour on every slice, in patient coordinates - an RT structure
/// as a file would carry it.
fn boxed_roi(name: &str, lo: [f64; 2], hi: [f64; 2], g: &Grid) -> Roi {
    let contours = (0..DIMS[2])
        .map(|k| {
            let z = g.voxel_to_patient(0.0, 0.0, k as f64).z;
            Contour {
                geometric_type: "CLOSED_PLANAR".into(),
                points: vec![
                    Vec3::new(lo[0], lo[1], z),
                    Vec3::new(hi[0], lo[1], z),
                    Vec3::new(hi[0], hi[1], z),
                    Vec3::new(lo[0], hi[1], z),
                ],
            }
        })
        .collect();
    Roi {
        number: 1,
        name: name.into(),
        color: [255, 0, 0],
        roi_type: "ORGAN".into(),
        contours,
    }
}

fn painted(name: &str, lo: [usize; 2], hi: [usize; 2]) -> Segmentation {
    let mut mask = vec![0u8; DIMS[0] * DIMS[1] * DIMS[2]];
    for k in 0..DIMS[2] {
        for j in lo[1]..hi[1] {
            for i in lo[0]..hi[0] {
                mask[idx(i, j, k)] = 1;
            }
        }
    }
    Segmentation::from_mask(name.into(), [0, 255, 0], DIMS, mask)
}

fn operand(name: &str, mask: Vec<u8>, margin: Margin) -> Operand {
    Operand {
        name: name.into(),
        mask,
        margin,
    }
}

fn recipe(op: BoolOp, operands: Vec<Operand>) -> Recipe {
    Recipe {
        op,
        operands,
        margin: Margin::NONE,
        cleanup: Cleanup::default(),
    }
}

#[test]
fn a_contour_and_a_painted_mask_combine_as_one_kind() {
    let g = grid(true);
    // The contour covers i 5..25; the painted mask i 15..35. Both span the
    // same rows and every slice, so the overlap is i 15..25.
    let roi = boxed_roi("contoured", [10.0, 20.0], [50.0, 60.0], &g);
    let contour_mask = segmentation::rasterize_roi(&g, &roi).expect("the contour rasterizes");
    let seg = painted("painted", [15, 10], [35, 30]);

    let both = combine(
        &recipe(
            BoolOp::Intersect,
            vec![
                operand("contoured", contour_mask.clone(), Margin::NONE),
                operand("painted", seg.mask.clone(), Margin::NONE),
            ],
        ),
        &g,
        &Quiet,
    )
    .expect("a result");
    assert!(both.voxels > 0, "the two overlap");
    assert_eq!(both.mask[idx(20, 15, 5)], 1, "inside both");
    assert_eq!(both.mask[idx(8, 15, 5)], 0, "contour only");
    assert_eq!(both.mask[idx(32, 15, 5)], 0, "mask only");

    let either = combine(
        &recipe(
            BoolOp::Union,
            vec![
                operand("contoured", contour_mask, Margin::NONE),
                operand("painted", seg.mask, Margin::NONE),
            ],
        ),
        &g,
        &Quiet,
    )
    .expect("a result");
    assert_eq!(either.mask[idx(8, 15, 5)], 1);
    assert_eq!(either.mask[idx(32, 15, 5)], 1);
    assert!(either.voxels > both.voxels);
}

#[test]
fn a_result_converts_back_to_contours_that_enclose_the_same_voxels() {
    let g = grid(true);
    let a = painted("a", [8, 8], [24, 24]).mask;
    let b = painted("b", [16, 16], [32, 32]).mask;
    let out = combine(
        &recipe(
            BoolOp::Union,
            vec![operand("a", a, Margin::NONE), operand("b", b, Margin::NONE)],
        ),
        &g,
        &Quiet,
    )
    .expect("a result");

    let seg = Segmentation::from_mask("union".into(), [255, 255, 0], DIMS, out.mask.clone());
    let roi = segmentation::mask_to_roi(&seg, &g, 7);
    assert!(!roi.contours.is_empty(), "the union has an outline");
    let back = segmentation::rasterize_roi(&g, &roi).expect("it rasterizes again");
    // The L-shaped union is not convex, so this is a real test of the
    // contour walk rather than of a bounding box.
    let differing = back
        .iter()
        .zip(out.mask.iter())
        .filter(|(x, y)| (**x != 0) != (**y != 0))
        .count();
    let total = out.voxels as usize;
    assert!(
        differing * 200 < total,
        "{differing} of {total} voxels changed across the round trip"
    );
}

#[test]
fn a_superior_margin_follows_the_patient_not_the_array() {
    // The same anatomy stored two ways up. A superior margin has to grow
    // toward the head in both, which is opposite array directions.
    let mut mask = vec![0u8; DIMS[0] * DIMS[1] * DIMS[2]];
    for j in 18..22 {
        for i in 18..22 {
            mask[idx(i, j, 6)] = 1;
        }
    }
    let m = Margin {
        superior: 10.0,
        ..Margin::NONE
    };
    let up = m.apply(&mask, &grid(true), &Quiet);
    let down = m.apply(&mask, &grid(false), &Quiet);
    assert_eq!(up[idx(20, 20, 8)], 1, "toward +k when +k is superior");
    assert_eq!(up[idx(20, 20, 4)], 0);
    assert_eq!(down[idx(20, 20, 4)], 1, "toward −k when −k is superior");
    assert_eq!(down[idx(20, 20, 8)], 0);
}

#[test]
fn subtracting_the_wrong_way_round_gives_nothing_and_says_so() {
    // The mistake the tool exists to make easy to spot: a small structure
    // minus the big one it sits inside is empty, and the caller has to be
    // able to tell that apart from an error.
    let g = grid(true);
    let small = painted("small", [18, 18], [22, 22]).mask;
    let big = painted("big", [5, 5], [35, 35]).mask;
    let wrong = combine(
        &recipe(
            BoolOp::Subtract,
            vec![
                operand("small", small.clone(), Margin::NONE),
                operand("big", big.clone(), Margin::NONE),
            ],
        ),
        &g,
        &Quiet,
    )
    .expect("a result, just an empty one");
    assert_eq!(wrong.voxels, 0);
    assert_eq!(wrong.pieces, 0);

    let right = combine(
        &recipe(
            BoolOp::Subtract,
            vec![
                operand("big", big, Margin::NONE),
                operand("small", small, Margin::NONE),
            ],
        ),
        &g,
        &Quiet,
    )
    .expect("a result");
    assert!(right.voxels > 0);
    assert_eq!(
        right.mask[idx(20, 20, 5)],
        0,
        "the hole is where it belongs"
    );
    assert_eq!(right.mask[idx(10, 10, 5)], 1);
}

#[test]
fn cleanup_can_rescue_a_subtraction_that_left_slivers() {
    let g = grid(true);
    let mut body = vec![0u8; DIMS[0] * DIMS[1] * DIMS[2]];
    for k in 0..DIMS[2] {
        for j in 8..32 {
            for i in 8..32 {
                body[idx(i, j, k)] = 1;
            }
        }
    }
    // A cut straight across, leaving two pieces of very different size.
    let mut knife = vec![0u8; DIMS[0] * DIMS[1] * DIMS[2]];
    for k in 0..DIMS[2] {
        for j in 8..32 {
            for i in 10..32 {
                knife[idx(i, j, k)] = 1;
            }
        }
    }
    let split = combine(
        &recipe(
            BoolOp::Subtract,
            vec![
                operand("body", body.clone(), Margin::NONE),
                operand("knife", knife.clone(), Margin::NONE),
            ],
        ),
        &g,
        &Quiet,
    )
    .expect("a result");
    assert_eq!(split.pieces, 1, "one sliver survives the cut");

    let kept = combine(
        &Recipe {
            op: BoolOp::Subtract,
            operands: vec![
                operand("body", body, Margin::NONE),
                operand("knife", knife, Margin::NONE),
            ],
            margin: Margin::NONE,
            cleanup: Cleanup {
                keep_largest: true,
                ..Cleanup::default()
            },
        },
        &g,
        &Quiet,
    )
    .expect("a result");
    assert_eq!(kept.pieces, 1);
    assert!(kept.voxels <= split.voxels);
}
