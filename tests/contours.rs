//! The seam between the contour engine and the rest of the viewer: the two
//! representations of one structure have to agree, and a drawing session has
//! to come out with the volume it looks like it drew.

use rust_dicom_station::contours::{Poly, Stack};
use rust_dicom_station::geometry::Vec3;
use rust_dicom_station::rtstruct::Roi;
use rust_dicom_station::segmentation;
use rust_dicom_station::volume::Grid;

/// A lattice stored feet-first (the slice axis running the other way), so
/// nothing can quietly assume a head-first stack.
fn grid(dims: [usize; 3], spacing: [f64; 3]) -> Grid {
    Grid {
        dims,
        spacing,
        origin: Vec3::new(-120.0, -80.0, 40.0),
        row_dir: Vec3::new(1.0, 0.0, 0.0),
        col_dir: Vec3::new(0.0, 1.0, 0.0),
        normal: Vec3::new(0.0, 0.0, -1.0),
        frame_of_reference_uid: "1.2.826.0.1.3680043.2.1125.1".into(),
    }
}

fn roi(contours: Vec<rust_dicom_station::rtstruct::Contour>) -> Roi {
    Roi {
        number: 1,
        name: "test".into(),
        color: [255, 128, 0],
        roi_type: "ORGAN".into(),
        contours,
    }
}

/// A ball with a spherical cavity: two nested contours a slice, which is the
/// case even-odd filling exists for.
fn shell(dims: [usize; 3], outer: f64, inner: f64) -> Vec<u8> {
    let [nx, ny, nz] = dims;
    let c = [
        (nx - 1) as f64 / 2.0,
        (ny - 1) as f64 / 2.0,
        (nz - 1) as f64 / 2.0,
    ];
    let mut m = vec![0u8; nx * ny * nz];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let d = ((i as f64 - c[0]).powi(2)
                    + (j as f64 - c[1]).powi(2)
                    + (k as f64 - c[2]).powi(2))
                .sqrt();
                if d <= outer && d >= inner {
                    m[k * nx * ny + j * nx + i] = 1;
                }
            }
        }
    }
    m
}

/// The two ways of filling a structure - the one the viewer has always used
/// on RTSTRUCT contours, and the stack's own - must agree voxel for voxel.
#[test]
fn the_stack_and_the_scanline_filler_agree() {
    let dims = [48, 48, 24];
    let g = grid(dims, [1.2, 1.2, 2.5]);
    let mask = shell(dims, 16.0, 7.0);
    let st = Stack::from_mask(&mask, dims, 2);
    let r = roi(st.to_contours(&g));

    let a = segmentation::rasterize_roi(&g, &r).expect("the ROI fills");
    let b = st.rasterize(dims);
    let diff = a.iter().zip(&b).filter(|(x, y)| x != y).count();
    assert_eq!(diff, 0, "{diff} voxels differ between the two fillers");
    // And the cavity really is a cavity.
    let mid = dims[2] / 2 * dims[0] * dims[1] + dims[1] / 2 * dims[0] + dims[0] / 2;
    assert_eq!(a[mid], 0, "the shell is hollow");
}

/// Reading a structure set back gives the same geometry, on a lattice whose
/// slice axis runs towards the feet.
#[test]
fn a_structure_survives_the_trip_through_patient_space() {
    let dims = [48, 48, 24];
    let g = grid(dims, [1.2, 1.2, 2.5]);
    let st = Stack::from_mask(&shell(dims, 16.0, 7.0), dims, 2);
    let r = roi(st.to_contours(&g));
    let back = Stack::from_roi(&r, &g);
    assert_eq!(back.axis, 2);
    assert_eq!(back.occupied(), st.occupied());
    let (v0, v1) = (st.volume_cm3(g.spacing), back.volume_cm3(g.spacing));
    assert!((v0 - v1).abs() < 1e-9 * v0.max(1.0), "{v0} vs {v1}");
    // The mask agrees too, which is what every voxel feature downstream sees.
    let m0 = st.rasterize(dims);
    let m1 = back.rasterize(dims);
    assert_eq!(m0.iter().zip(&m1).filter(|(a, b)| a != b).count(), 0);
}

/// Drawing on a few slices and interpolating between them gives the volume
/// the drawing implies - the workflow the interpolation tool exists for.
#[test]
fn drawing_and_interpolating_a_cylinder_gives_a_cylinder() {
    let dims = [64, 64, 21];
    let g = grid(dims, [1.0, 1.0, 3.0]);
    let mut st = Stack::empty(2);
    // A 12-voxel circle drawn on every fifth slice of a 21-slice stack.
    for k in (0..21).step_by(5) {
        st.region_mut(k)
            .add_ring(&Poly::circle([32.0, 32.0], 12.0, 96));
    }
    assert_eq!(st.occupied(), 5);
    let drawn = st.volume_cm3(g.spacing);
    let one_slice = std::f64::consts::PI * 144.0 * 1.0 * 1.0 * 3.0 / 1000.0;
    assert!((drawn - 5.0 * one_slice).abs() < 0.02 * drawn, "{drawn}");

    let filled = st.interpolated(dims);
    assert_eq!(filled.occupied(), 16, "every gap slice");
    for s in &filled.slices {
        *st.region_mut(s.level) = s.region.clone();
    }
    let whole = st.volume_cm3(g.spacing);
    let want = 21.0 * one_slice;
    // The interpolated slices are re-traced from a blended distance field on
    // the lattice, so they carry the lattice's own accuracy - a little under
    // the analytic circle, by about half a voxel of outline.
    assert!(
        (whole - want).abs() < 0.04 * want,
        "interpolated {whole} vs the cylinder's {want}"
    );
    assert!(whole < want, "and it errs on the inside, not the outside");
}

/// A stroke in *subtract* mode takes a bite of exactly its own size, and the
/// contours the tool did not touch keep their own points.
#[test]
fn a_subtract_stroke_takes_the_bite_it_looks_like() {
    let dims = [64, 64, 5];
    let g = grid(dims, [1.0, 1.0, 3.0]);
    let mut st = Stack::empty(2);
    for k in 0..5 {
        st.region_mut(k)
            .add_ring(&Poly::circle([32.0, 32.0], 15.0, 128));
    }
    let untouched = st.region_at(0).unwrap().rings[0].clone();
    let before = st.volume_cm3(g.spacing);

    // A circle of radius 6 centred on the outline: half of it is inside.
    let bite = Poly::circle([47.0, 32.0], 6.0, 96);
    st.region_mut(2).subtract_ring(&bite);
    let after = st.volume_cm3(g.spacing);
    let removed = before - after;
    let expect = 0.5 * std::f64::consts::PI * 36.0 * 3.0 / 1000.0;
    assert!(
        (removed - expect).abs() < 0.12 * expect,
        "removed {removed} cm³, expected about {expect}"
    );
    assert_eq!(
        st.region_at(0).unwrap().rings[0],
        untouched,
        "the other slices are not even re-sampled"
    );
    // The result still round-trips as a structure set.
    let r = roi(st.to_contours(&g));
    let back = Stack::from_roi(&r, &g);
    assert!((back.volume_cm3(g.spacing) - after).abs() < 1e-9 * after);
}

/// The tidying operations do what their buttons say, on a geometry with a
/// hole, a speck and a staircase.
#[test]
fn the_tidy_operations_do_what_the_buttons_say() {
    let dims = [64, 64, 6];
    let g = grid(dims, [1.0, 1.0, 3.0]);
    let mut st = Stack::empty(2);
    for k in 1..5 {
        let r = st.region_mut(k);
        r.add_ring(&Poly::circle([30.0, 30.0], 14.0, 128));
        r.subtract_ring(&Poly::circle([30.0, 30.0], 4.0, 64));
        r.add_ring(&Poly::circle([56.0, 56.0], 1.5, 32));
    }
    let full = st.volume_cm3(g.spacing);

    let mut holes = st.clone();
    holes.remove_holes();
    let gained = holes.volume_cm3(g.spacing) - full;
    let cavity = std::f64::consts::PI * 16.0 * 4.0 * 3.0 / 1000.0;
    assert!(
        (gained - cavity).abs() < 0.05 * cavity,
        "{gained} vs {cavity}"
    );

    let mut small = st.clone();
    small.drop_smaller_than(20.0);
    let speck = std::f64::consts::PI * 2.25 * 4.0 * 3.0 / 1000.0;
    assert!(
        (full - small.volume_cm3(g.spacing) - speck).abs() < 0.2 * speck,
        "the speck goes and nothing else"
    );

    let mut capped = st.clone();
    capped.limit_points(20);
    for s in &capped.slices {
        for r in &s.region.rings {
            assert!(r.len() <= 20);
        }
    }
    assert!((capped.volume_cm3(g.spacing) - full).abs() < 0.05 * full);
}

/// Drawing in a sagittal view produces a sagittal stack, and writing it into
/// a structure set converts it to the axial contours every other system
/// expects - keeping the shape.
#[test]
fn a_sagittal_drawing_lands_as_axial_contours() {
    let dims = [40, 40, 40];
    let g = grid(dims, [1.5, 1.5, 1.5]);
    let mut st = Stack::empty(0);
    for i in 14..26 {
        st.region_mut(i)
            .add_ring(&Poly::circle([20.0, 20.0], 9.0, 96));
    }
    let drawn = st.volume_cm3(g.spacing);
    let mut r = roi(Vec::new());
    st.apply_to_roi(&mut r, &g);
    let back = Stack::from_roi(&r, &g);
    assert_eq!(back.axis, 2, "stored axially");
    let kept = back.volume_cm3(g.spacing);
    assert!((kept - drawn).abs() < 0.03 * drawn, "{kept} vs {drawn}");
    // And it fills the same voxels the sagittal stack did, to a voxel skin.
    let a = st.rasterize(dims);
    let b = back.rasterize(dims);
    let differ = a.iter().zip(&b).filter(|(x, y)| x != y).count();
    let set = a.iter().filter(|&&v| v != 0).count();
    assert!(differ < set / 20, "{differ} of {set} voxels differ");
}
