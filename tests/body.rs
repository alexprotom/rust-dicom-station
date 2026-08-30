//! The body contour, tested against a phantom whose ground truth is known
//! by construction.
//!
//! The point of a synthetic phantom is that every failure mode of the real
//! problem can be built into it deliberately and then asserted on: a couch
//! skin and a rail under the back, a moulded mask shell hugging the face
//! with a real air gap *and* a real contact patch, ears that a plain opening
//! would shave off, lungs draining through an airway, a cable. Real scans
//! differ in a hundred ways, but a method that gets these wrong is broken,
//! and one that gets them right is worth taking to real data.
//!
//! The voxels are deliberately anisotropic (2 × 2 × 5 mm), because that is
//! where a voxel-counting implementation goes wrong and a millimetre-aware
//! one does not.

use rust_dicom_station::bodymask::{contour_body, BodyParams, Foreground, Method};
use rust_dicom_station::geometry::Vec3;
use rust_dicom_station::progress::Progress;
use rust_dicom_station::volume::Volume;

const NX: usize = 100;
const NY: usize = 70;
const NZ: usize = 40;
const SP: [f64; 3] = [2.0, 2.0, 5.0];
/// The body: an elliptical cylinder 160 × 104 mm, centred in the field.
const CX: f64 = 100.0;
const CY: f64 = 62.0;
const RX: f64 = 80.0;
const RY: f64 = 52.0;
/// Where the moulded shell is pressed against the skin instead of standing
/// off it — the one thing geometry cannot undo.
const CONTACT: std::ops::Range<usize> = 46..55;

fn idx(i: usize, j: usize, k: usize) -> usize {
    k * NX * NY + j * NX + i
}

/// The first row index at or below the body's upper surface at column `i`.
fn body_top(i: usize) -> Option<usize> {
    let x = i as f64 * SP[0];
    let t = 1.0 - ((x - CX) / RX).powi(2);
    if t < 0.0 {
        return None;
    }
    Some(((CY - RY * t.sqrt()) / SP[1]).ceil() as usize)
}

/// The row the mask shell occupies at column `i`: two voxels clear of the
/// skin, except over the contact patch, where it touches.
fn shell_row(i: usize) -> Option<usize> {
    let top = body_top(i)?;
    let off = if CONTACT.contains(&i) { 1 } else { 2 };
    top.checked_sub(off)
}

/// The phantom, plus the body mask it was built from.
struct Phantom {
    volume: Volume,
    truth: Vec<u8>,
}

fn phantom() -> Phantom {
    let mut data = vec![-1000i16; NX * NY * NZ];
    let mut truth = vec![0u8; NX * NY * NZ];
    for k in 0..NZ {
        for j in 0..NY {
            for i in 0..NX {
                let (x, y) = (i as f64 * SP[0], j as f64 * SP[1]);
                if ((x - CX) / RX).powi(2) + ((y - CY) / RY).powi(2) <= 1.0 {
                    data[idx(i, j, k)] = 40;
                    truth[idx(i, j, k)] = 1;
                }
            }
        }
    }
    // Ears: 6 mm plates just outside each flank, over 25 mm of the stack.
    for k in 8..13 {
        for j in 26..36 {
            // Overlapping the flank by two voxels, so they are attached to
            // the body rather than to the knife edge of a perfect ellipse.
            for i in [7, 8, 9, 10, 11, 88, 89, 90, 91, 92] {
                data[idx(i, j, k)] = 40;
                truth[idx(i, j, k)] = 1;
            }
        }
    }
    // Lungs, a comfortable wall away from the skin.
    for k in 0..NZ {
        for j in 22..43 {
            for i in (30..46).chain(55..71) {
                data[idx(i, j, k)] = -850;
            }
        }
    }
    // Airways reaching both lungs — on the first two slices only, so that
    // exactly those two slices have a cavity open to the outside.
    for k in 0..2 {
        for j in 0..23 {
            for i in (40..44).chain(57..61) {
                data[idx(i, j, k)] = -850;
                truth[idx(i, j, k)] = 0;
            }
        }
    }
    // The couch: a carbon skin under the back with a 2 mm air gap, and a
    // rail 6 mm below it. The foam between them is below the threshold.
    for k in 0..NZ {
        for i in 10..90 {
            data[idx(i, 59, k)] = 300;
            data[idx(i, 62, k)] = 300;
        }
    }
    // The moulded mask shell.
    for k in 0..NZ {
        for i in 15..86 {
            if let Some(j) = shell_row(i) {
                data[idx(i, j, k)] = 120;
            }
        }
    }
    // A cable: thin, free-standing, running the whole length.
    for k in 0..NZ {
        data[idx(95, 65, k)] = 200;
    }
    Phantom {
        volume: volume(data),
        truth,
    }
}

fn volume(data: Vec<i16>) -> Volume {
    Volume {
        data,
        dims: [NX, NY, NZ],
        spacing: SP,
        origin: Vec3::new(0.0, 0.0, 0.0),
        row_dir: Vec3::new(1.0, 0.0, 0.0),
        col_dir: Vec3::new(0.0, 1.0, 0.0),
        normal: Vec3::new(0.0, 0.0, 1.0),
        frame_of_reference_uid: "1.2.826.0.1.3680043.8.498.phantom".into(),
        min_value: -1000,
        max_value: 300,
    }
}

fn params() -> BodyParams {
    BodyParams {
        method: Method::Classical,
        foreground: Foreground::Hu(-300.0),
        open_mm: 8.0,
        // The phantom is 200 mm long; a 100 mm window still tells the
        // extruded equipment from the 25 mm ears with room to spare.
        persist_window_mm: 100.0,
        min_volume_cm3: 20.0,
        ..BodyParams::default()
    }
}

/// No model folder is ever reached by the classical method.
fn nowhere() -> &'static std::path::Path {
    std::path::Path::new("/nonexistent")
}

fn dice(a: &[u8], b: &[u8]) -> f64 {
    let inter = a
        .iter()
        .zip(b.iter())
        .filter(|(x, y)| **x != 0 && **y != 0)
        .count();
    let na = a.iter().filter(|v| **v != 0).count();
    let nb = b.iter().filter(|v| **v != 0).count();
    2.0 * inter as f64 / (na + nb).max(1) as f64
}

const VOXEL_CM3: f64 = SP[0] * SP[1] * SP[2] / 1000.0;

#[test]
fn the_classical_method_finds_the_patient_and_leaves_the_equipment_out() {
    let p = phantom();
    let r = contour_body(&p.volume, &params(), nowhere(), &Progress::default()).expect("a body");

    // The residual is the two slices whose airway is open to the outside,
    // which is the behaviour the slice-wise fill is chosen for.
    let d = dice(&r.mask, &p.truth);
    assert!(d > 0.99, "Dice against the built body is {d:.4}");
    assert_eq!(r.pieces.len(), 1, "one body");
    assert!(r.removed_voxels > 0, "equipment was reported as removed");

    for k in [0, NZ / 2, NZ - 1] {
        for i in 15..85 {
            assert_eq!(r.mask[idx(i, 59, k)], 0, "couch skin kept at i={i} k={k}");
            assert_eq!(r.mask[idx(i, 62, k)], 0, "couch rail kept at i={i} k={k}");
        }
        assert_eq!(r.mask[idx(95, 65, k)], 0, "cable kept at k={k}");
    }

    // The mask shell goes wherever it stands clear of the skin.
    for k in [0, NZ / 2, NZ - 1] {
        for i in 15..86 {
            if CONTACT.contains(&i) {
                continue;
            }
            if let Some(j) = shell_row(i) {
                assert_eq!(r.mask[idx(i, j, k)], 0, "mask shell kept at i={i} k={k}");
            }
        }
    }

    // The ears — thin, and exactly what a plain opening shaves off.
    assert_eq!(r.mask[idx(8, 30, 10)], 1, "left ear");
    assert_eq!(r.mask[idx(91, 30, 10)], 1, "right ear");
    assert!(r.recovered_voxels > 0, "thin anatomy was recovered");

    // The lungs are inside the body on every slice past the airway.
    for k in 2..NZ {
        assert_eq!(r.mask[idx(35, 30, k)], 1, "left lung filled on k={k}");
        assert_eq!(r.mask[idx(65, 30, k)], 1, "right lung filled on k={k}");
    }
}

#[test]
fn a_shell_pressed_against_the_skin_is_kept_and_the_cost_is_bounded() {
    // The documented limitation, pinned as a test rather than left to be
    // rediscovered: where a shell touches with no air gap, it is locally
    // indistinguishable from a slightly thicker patient, so it stays. What
    // must not happen is the error growing beyond the contact patch.
    let p = phantom();
    let r = contour_body(&p.volume, &params(), nowhere(), &Progress::default()).expect("a body");
    let kept = CONTACT
        .filter_map(|i| shell_row(i).map(|j| (i, j)))
        .flat_map(|(i, j)| (0..NZ).map(move |k| (i, j, k)))
        .filter(|&(i, j, k)| r.mask[idx(i, j, k)] != 0)
        .count();
    assert!(kept > 0, "the contact patch is expected to survive");
    let extra = r
        .mask
        .iter()
        .zip(p.truth.iter())
        .filter(|(m, t)| **m != 0 && **t == 0)
        .count() as f64
        * VOXEL_CM3;
    assert!(
        extra < 10.0,
        "the error beyond the patient is {extra:.1} cm³ — the contact patch \
         alone should be about 7"
    );
}

#[test]
fn the_chest_wall_over_a_lung_is_not_mistaken_for_a_couch_skin() {
    // The failure real data taught, reduced to its essentials: a hollow
    // cylinder — a 6 mm wall around a big cavity — beside a 2 mm couch
    // skin. A threshold does not see a body here, it sees a thin shell
    // that repeats slice after slice, which is the exact signature the
    // equipment test looks for. Both are thin, both are extruded; only one
    // of them encloses the patient.
    let (nx, ny, nz) = (70usize, 70usize, 40usize);
    let at = |i: usize, j: usize, k: usize| k * nx * ny + j * nx + i;
    let (cx, cy) = (70.0f64, 70.0f64);
    let mut data = vec![-1000i16; nx * ny * nz];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let r = ((i as f64 * SP[0] - cx).powi(2) + (j as f64 * SP[1] - cy).powi(2)).sqrt();
                if r <= 50.0 {
                    data[at(i, j, k)] = 40;
                }
                if r <= 44.0 {
                    data[at(i, j, k)] = -850;
                }
            }
        }
        for i in 5..65 {
            data[at(i, 63, k)] = 300;
        }
    }
    let mut vol = volume(vec![0i16; nx * ny * nz]);
    vol.data = data;
    vol.dims = [nx, ny, nz];
    let r = contour_body(&vol, &params(), nowhere(), &Progress::default()).expect("a body");
    for k in [0, nz / 2, nz - 1] {
        for deg in (0..360).step_by(30) {
            let a = (deg as f64).to_radians();
            let i = ((cx + 47.0 * a.cos()) / SP[0]).round() as usize;
            let j = ((cy + 47.0 * a.sin()) / SP[1]).round() as usize;
            assert_eq!(r.mask[at(i, j, k)], 1, "wall deleted at {deg}° on k={k}");
        }
        // The cavity is inside the body…
        assert_eq!(r.mask[at(35, 35, k)], 1, "cavity not filled on k={k}");
        // …and the couch is not.
        for i in 10..60 {
            assert_eq!(r.mask[at(i, 63, k)], 0, "couch skin kept at i={i} k={k}");
        }
    }
}

#[test]
fn two_legs_are_two_bodies_not_the_larger_one() {
    let (nx, ny, nz) = (60usize, 40usize, 20usize);
    let at = |i: usize, j: usize, k: usize| k * nx * ny + j * nx + i;
    let mut data = vec![-1000i16; nx * ny * nz];
    for k in 0..nz {
        for j in 8..32 {
            for i in (4..24).chain(34..50) {
                data[at(i, j, k)] = 40;
            }
        }
    }
    let mut vol = volume(vec![0i16; nx * ny * nz]);
    vol.data = data;
    vol.dims = [nx, ny, nz];
    let mut p = params();
    p.remove_devices = false;
    let r = contour_body(&vol, &p, nowhere(), &Progress::default()).expect("two bodies");
    assert_eq!(r.pieces.len(), 2, "both legs kept");
    assert_eq!(r.mask[at(14, 20, 10)], 1);
    assert_eq!(r.mask[at(42, 20, 10)], 1);
}

#[test]
fn an_mr_phantom_with_a_coil_gradient_still_comes_out_whole() {
    // The same body, but with MR intensities and a receive profile that
    // falls off across the field — the case a fixed threshold cannot serve.
    let p = phantom();
    let mut data = vec![0i16; NX * NY * NZ];
    for k in 0..NZ {
        for j in 0..NY {
            for i in 0..NX {
                let gain = (-1.6 * i as f32 / NX as f32).exp();
                let raw = if p.truth[idx(i, j, k)] == 0 {
                    5.0
                } else if p.volume.data[idx(i, j, k)] < -300 {
                    20.0 // lung: dark on MR too
                } else {
                    900.0
                };
                data[idx(i, j, k)] = (raw * gain) as i16;
            }
        }
    }
    let mut pr = params();
    pr.foreground = Foreground::MrRelative {
        fraction: 0.12,
        // Well above any anatomy, well below the 160 mm body — the window a
        // bias estimate has to sit in.
        sigma_mm: 25.0,
    };
    pr.remove_devices = false;
    let r = contour_body(&volume(data), &pr, nowhere(), &Progress::default()).expect("a body");
    let d = dice(&r.mask, &p.truth);
    assert!(d > 0.97, "Dice on the MR phantom is {d:.4}");
    assert_eq!(r.mask[idx(12, 31, 20)], 1, "the bright end of the field");
    assert_eq!(r.mask[idx(87, 31, 20)], 1, "the dim end of the field");
}

#[test]
fn a_model_folder_that_cannot_exist_is_an_error_not_a_panic() {
    // A regular file where the model folder should be: the download cannot
    // even begin, whatever the network is doing, which is what makes this
    // deterministic on a build machine that happens to be online.
    let file = std::env::temp_dir().join("rds-body-not-a-folder");
    std::fs::write(&file, b"not a folder").expect("write the blocker");
    let p = phantom();
    let mut pr = params();
    pr.method = Method::ModelAssisted;
    let err = contour_body(&p.volume, &pr, &file, &Progress::default())
        .expect_err("no weights, no answer");
    let text = format!("{err:#}");
    assert!(
        text.to_lowercase().contains("body")
            || text.contains("create")
            || text.contains("os error"),
        "the error should name what failed: {text}"
    );
    let _ = std::fs::remove_file(&file);
}

/// The real thing, end to end, against the published weights. Ignored by
/// default because it downloads 124 MB on first run:
///
/// ```text
/// RDS_BODY_MODELS=path/to/models/totalsegmentator \
///   cargo test --release --test body -- --ignored
/// ```
#[test]
#[ignore]
fn the_model_assisted_method_runs_the_published_network() {
    let dir = std::env::var("RDS_BODY_MODELS").expect("set RDS_BODY_MODELS");
    let p = phantom();
    let mut pr = params();
    pr.method = Method::ModelAssisted;
    let r = contour_body(
        &p.volume,
        &pr,
        std::path::Path::new(&dir),
        &Progress::default(),
    )
    .expect("a body");
    // The network sees a featureless ellipse rather than a person, so this
    // asserts that the hybrid holds together — a body of a plausible size,
    // with the skin still placed by the threshold — not a Dice figure that
    // would only be meaningful on real anatomy.
    assert!(r.voxels > 0, "the network found a patient");
    assert!(!r.device.is_empty(), "the device was reported");
    let d = dice(&r.mask, &p.truth);
    assert!(d > 0.9, "Dice {d:.4}");
}
