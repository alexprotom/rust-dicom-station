//! Registration engine tests against analytically known transforms.
//!
//! Run with `--release` (the optimizer loops are slow in debug builds).

use rust_dicom_viewer::geometry::Vec3;
use rust_dicom_viewer::registration::{
    register, RegKind, RegParams, RegProgress, RigidTransform,
};
use rust_dicom_viewer::volume::Volume;

/// Smoothly-edged multi-feature phantom (values in HU-like units).
/// A *finite* ellipsoid body (so every rigid DOF is constrained by the
/// surface) plus several asymmetric blobs for internal structure.
fn phantom(p: Vec3) -> f32 {
    #[inline]
    fn blob(p: Vec3, c: Vec3, r: f64, edge: f64) -> f64 {
        let d = (p - c).length();
        // 1 inside, 0 outside, smooth over `edge` mm.
        (0.5 - (d - r) / edge).clamp(0.0, 1.0)
    }
    // Ellipsoid body, semi-axes (75, 65, 82) mm, ~6 mm smooth edge.
    let e = ((p.x / 75.0).powi(2) + (p.y / 65.0).powi(2) + (p.z / 82.0).powi(2)).sqrt();
    let body = (0.5 - (e - 1.0) / 0.09).clamp(0.0, 1.0);
    if body <= 0.0 {
        return -1000.0;
    }
    let mut v = -1000.0 + 1000.0 * body; // air → water
    v += 100.0 * blob(p, Vec3::new(0.0, 0.0, 0.0), 20.0, 6.0);
    v += 60.0 * blob(p, Vec3::new(30.0, 20.0, 10.0), 14.0, 6.0);
    v += 140.0 * blob(p, Vec3::new(-28.0, 12.0, -14.0), 11.0, 6.0);
    v += -80.0 * blob(p, Vec3::new(-5.0, -32.0, 8.0), 12.0, 6.0);
    v += 90.0 * blob(p, Vec3::new(12.0, 8.0, 42.0), 13.0, 6.0);
    v as f32
}

/// Build an axis-aligned volume sampling `f` at voxel centers.
fn make_volume(n: usize, spacing: f64, f: impl Fn(Vec3) -> f32 + Sync) -> Volume {
    let half = (n as f64 - 1.0) * 0.5 * spacing;
    let origin = Vec3::new(-half, -half, -half);
    let mut data = vec![0i16; n * n * n];
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let p = origin
                    + Vec3::new(i as f64 * spacing, j as f64 * spacing, k as f64 * spacing);
                data[k * n * n + j * n + i] = f(p).round().clamp(-32768.0, 32767.0) as i16;
            }
        }
    }
    Volume {
        data,
        dims: [n, n, n],
        spacing: [spacing; 3],
        origin,
        row_dir: Vec3::new(1.0, 0.0, 0.0),
        col_dir: Vec3::new(0.0, 1.0, 0.0),
        normal: Vec3::new(0.0, 0.0, 1.0),
        frame_of_reference_uid: String::new(),
        min_value: -1000,
        max_value: 300,
    }
}

#[test]
fn rigid_recovers_known_transform() {
    let n = 64;
    let spacing = 3.0;

    // Ground truth: T_true maps fixed → moving.
    let t_true = RigidTransform::new(
        [
            2.0f64.to_radians(),
            -1.5f64.to_radians(),
            3.0f64.to_radians(),
            6.0,
            -4.0,
            3.0,
        ],
        Vec3::ZERO,
    );

    let fixed = make_volume(n, spacing, phantom);
    // M(q) = F(T_true⁻¹ q)  ⇒  M(T_true x) = F(x): optimum is exactly T_true.
    let moving = make_volume(n, spacing, |p| phantom(t_true.unmap(p)));

    let params = RegParams {
        kind: RegKind::Rigid,
        levels: 3,
        iterations: 400,
        samples: 4000,
        grid_spacing_mm: 32.0,
        fixed_threshold: -500.0,
    };
    let progress = RegProgress::default();
    let t0 = std::time::Instant::now();
    let res = register(&fixed, &moving, &params, &progress).expect("registration runs");
    eprintln!(
        "rigid: MSD {:.1} → {:.1} in {:?} ({} iters)",
        res.initial_metric,
        res.final_metric,
        t0.elapsed(),
        res.iterations_run
    );

    assert!(
        res.final_metric < 0.1 * res.initial_metric,
        "metric should drop by >90% (got {:.1} → {:.1})",
        res.initial_metric,
        res.final_metric
    );

    // Compare the recovered mapping to the ground truth at body points.
    let probes = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(40.0, 10.0, 20.0),
        Vec3::new(-30.0, 25.0, -25.0),
        Vec3::new(10.0, -40.0, 30.0),
        Vec3::new(-15.0, -20.0, -35.0),
    ];
    let mut max_err = 0.0f64;
    for p in probes {
        let err = (res.transform.map(p) - t_true.map(p)).length();
        max_err = max_err.max(err);
    }
    eprintln!("rigid: max mapping error {max_err:.2} mm");
    assert!(max_err < 1.5, "max mapping error {max_err:.2} mm >= 1.5 mm");

    // Inverse must round-trip.
    for p in probes {
        let rt = (res.transform.unmap(res.transform.map(p)) - p).length();
        assert!(rt < 1e-6, "rigid inverse round-trip {rt}");
    }
}

/// Ground-truth smooth displacement (fixed → moving), a Gaussian bump.
fn true_disp(p: Vec3) -> Vec3 {
    let p0 = Vec3::new(10.0, 5.0, 0.0);
    let sigma = 25.0f64;
    let d = p - p0;
    let a = 7.0 * (-d.dot(d) / (2.0 * sigma * sigma)).exp();
    Vec3::new(0.0, a, 0.0)
}

#[test]
fn bspline_recovers_gaussian_bump() {
    let n = 64;
    let spacing = 3.0;

    let fixed = make_volume(n, spacing, phantom);
    // Want optimum T(x) = x + d(x). Build M with M(x + d(x)) = F(x):
    // M(q) = F(g(q)) where g inverts x ↦ x + d(x) (fixed-point iteration).
    let moving = make_volume(n, spacing, |q| {
        let mut x = q;
        for _ in 0..10 {
            x = q - true_disp(x);
        }
        phantom(x)
    });

    let params = RegParams {
        kind: RegKind::Deformable,
        levels: 3,
        iterations: 400,
        samples: 5000,
        grid_spacing_mm: 24.0,
        fixed_threshold: -500.0,
    };
    let progress = RegProgress::default();
    let t0 = std::time::Instant::now();
    let res = register(&fixed, &moving, &params, &progress).expect("registration runs");
    eprintln!(
        "bspline: MSD {:.1} → {:.1} in {:?} ({} iters)",
        res.initial_metric,
        res.final_metric,
        t0.elapsed(),
        res.iterations_run
    );

    assert!(
        res.final_metric < 0.4 * res.initial_metric,
        "deformable metric should drop by >60% (got {:.1} → {:.1})",
        res.initial_metric,
        res.final_metric
    );

    // Recovered displacement near the bump center should match the truth.
    let probes = [
        Vec3::new(10.0, 5.0, 0.0),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(25.0, 15.0, 5.0),
    ];
    let mut max_err = 0.0f64;
    for p in probes {
        let expected = p + true_disp(p);
        let err = (res.transform.map(p) - expected).length();
        max_err = max_err.max(err);
    }
    eprintln!("bspline: max mapping error at probes {max_err:.2} mm");
    assert!(max_err < 3.0, "max deformable mapping error {max_err:.2} mm >= 3 mm");

    // Approximate inverse should round-trip within a fraction of a mm.
    let p = Vec3::new(12.0, 8.0, 3.0);
    let rt = (res.transform.unmap(res.transform.map(p)) - p).length();
    assert!(rt < 0.1, "deformable inverse round-trip {rt:.4} mm");
}
