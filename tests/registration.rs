//! Registration engine tests against analytically known transforms.
//!
//! Run with `--release` (the optimizer loops are slow in debug builds).
//!
//! Every engine is asked the same kind of question: recover a transform that
//! is known exactly, and land within a tolerance of it at probe points. The
//! two intensity engines are held to the same phantom so their results are
//! comparable; the landmark warp and the local runs are checked on what is
//! specific to them - exactness at the landmarks, and leaving the rest of
//! the volume alone.

use rust_dicom_station::geometry::Vec3;
use rust_dicom_station::progress::Progress;
use rust_dicom_station::registration::{
    register, LandmarkKernel, LandmarkPair, LandmarkParams, Metric, RegMethod, RegParams,
    RegionMask, RigidTransform, VectorField,
};
use rust_dicom_station::volume::Volume;

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
                let p =
                    origin + Vec3::new(i as f64 * spacing, j as f64 * spacing, k as f64 * spacing);
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
fn elastix_rigid_recovers_known_transform() {
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
        method: RegMethod::ElastixRigid,
        levels: 3,
        iterations: 400,
        samples: 4000,
        ..RegParams::default()
    };
    let progress = Progress::default();
    let t0 = std::time::Instant::now();
    let res = register(&fixed, &moving, &params, &progress).expect("registration runs");
    eprintln!(
        "elastix rigid: MSD {:.1} → {:.1} in {:?} ({} iters)",
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
    eprintln!("elastix rigid: max mapping error {max_err:.2} mm");
    assert!(max_err < 1.5, "max mapping error {max_err:.2} mm >= 1.5 mm");

    // Inverse must round-trip.
    for p in probes {
        let rt = (res.transform.unmap(res.transform.map(p)) - p).length();
        assert!(rt < 1e-6, "rigid inverse round-trip {rt}");
    }

    // The analysis must report exactly the transform that was recovered:
    // a rigid result is explained by six numbers with no residual left.
    let a = &res.analysis;
    assert!(
        a.dof.residual_mm < 1e-3,
        "a rigid result left a residual of {:.4} mm",
        a.dof.residual_mm
    );
    // Checked against the *recovered* transform, not the ground truth: how
    // close the optimizer got is the mapping-error assertion above, and a
    // small rotation about a near-symmetry axis of the phantom is the one
    // parameter that stays ill-conditioned even when the mapping is right to
    // half a millimetre. What is asserted here is that the analysis reports
    // the transform it was given.
    let recovered = res.transform.rigid.params();
    for (got, want) in a
        .dof
        .rotation_deg
        .iter()
        .zip(recovered[..3].iter().map(|r| r.to_degrees()))
    {
        assert!(
            (got - want).abs() < 1e-3,
            "the fit says {got:.4}° where the transform says {want:.4}°"
        );
    }
    let t = Vec3::new(recovered[3], recovered[4], recovered[5]);
    assert!(
        (a.mean_vector - t).length() < 2.0,
        "mean displacement {:?} against a translation of {t:?}",
        a.mean_vector
    );
    assert!(
        (a.jacobian.mean - 1.0).abs() < 1e-3,
        "a rigid body preserves volume"
    );
    assert_eq!(a.jacobian.folded, 0.0);
    assert!(a.samples > 1000);
}

/// Ground-truth smooth displacement (fixed → moving), a Gaussian bump.
fn true_disp(p: Vec3) -> Vec3 {
    let p0 = Vec3::new(10.0, 5.0, 0.0);
    let sigma = 25.0f64;
    let d = p - p0;
    let a = 7.0 * (-d.dot(d) / (2.0 * sigma * sigma)).exp();
    Vec3::new(0.0, a, 0.0)
}

/// A moving image whose optimum against the phantom is `x ↦ x + d(x)`.
fn bumped_moving(n: usize, spacing: f64) -> Volume {
    // Want M(x + d(x)) = F(x), i.e. M(q) = F(g(q)) with g inverting the map.
    make_volume(n, spacing, |q| {
        let mut x = q;
        for _ in 0..10 {
            x = q - true_disp(x);
        }
        phantom(x)
    })
}

#[test]
fn elastix_bspline_recovers_gaussian_bump() {
    let n = 64;
    let spacing = 3.0;
    let fixed = make_volume(n, spacing, phantom);
    let moving = bumped_moving(n, spacing);

    let params = RegParams {
        method: RegMethod::ElastixBSpline,
        levels: 3,
        iterations: 400,
        samples: 5000,
        grid_spacing_mm: 24.0,
        ..RegParams::default()
    };
    let progress = Progress::default();
    let t0 = std::time::Instant::now();
    let res = register(&fixed, &moving, &params, &progress).expect("registration runs");
    eprintln!(
        "elastix bspline: MSD {:.1} → {:.1} in {:?} ({} iters)",
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

    let probes = [
        Vec3::new(10.0, 5.0, 0.0),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(25.0, 15.0, 5.0),
    ];
    let mut max_err = 0.0f64;
    for p in probes {
        let expected = p + true_disp(p);
        max_err = max_err.max((res.transform.map(p) - expected).length());
    }
    eprintln!("elastix bspline: max mapping error at probes {max_err:.2} mm");
    assert!(
        max_err < 3.0,
        "max deformable mapping error {max_err:.2} mm >= 3 mm"
    );

    // Approximate inverse should round-trip within a fraction of a mm.
    let p = Vec3::new(12.0, 8.0, 3.0);
    let rt = (res.transform.unmap(res.transform.map(p)) - p).length();
    assert!(rt < 0.1, "deformable inverse round-trip {rt:.4} mm");

    // A 7 mm bump is a deformation, so the rigid fit must not explain it,
    // and a smooth bump must not fold the tissue anywhere.
    assert!(res.analysis.dof.residual_mm > 0.3);
    assert_eq!(res.analysis.jacobian.folded, 0.0);
    assert!(res.analysis.displacement.max > 3.0);
}

#[test]
fn plastimatch_bspline_recovers_gaussian_bump() {
    let n = 64;
    let spacing = 3.0;
    let fixed = make_volume(n, spacing, phantom);
    let moving = bumped_moving(n, spacing);

    let params = RegParams {
        method: RegMethod::PlastimatchBSpline,
        levels: 3,
        // A dense exact gradient converges in tens of iterations, not
        // hundreds - that is the whole trade against the stochastic engine.
        iterations: 60,
        grid_spacing_mm: 24.0,
        regularization: 0.01,
        ..RegParams::default()
    };
    let progress = Progress::default();
    let t0 = std::time::Instant::now();
    let res = register(&fixed, &moving, &params, &progress).expect("registration runs");
    eprintln!(
        "plastimatch bspline: MSD {:.1} → {:.1} in {:?} ({} evals)",
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
    let probes = [
        Vec3::new(10.0, 5.0, 0.0),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(25.0, 15.0, 5.0),
    ];
    let mut max_err = 0.0f64;
    for p in probes {
        let expected = p + true_disp(p);
        max_err = max_err.max((res.transform.map(p) - expected).length());
    }
    eprintln!("plastimatch bspline: max mapping error at probes {max_err:.2} mm");
    assert!(
        max_err < 3.0,
        "max deformable mapping error {max_err:.2} mm >= 3 mm"
    );
    // The bending-energy penalty exists to keep the field invertible.
    assert_eq!(
        res.analysis.jacobian.folded, 0.0,
        "a regularized B-spline must not fold"
    );
}

#[test]
fn plastimatch_mutual_information_survives_an_inverted_contrast() {
    let n = 56;
    let spacing = 3.5;
    let fixed = make_volume(n, spacing, phantom);
    // The same anatomy, deformed by the same bump, but with the soft-tissue
    // contrast inverted - air untouched, so the body mask still works. Mean
    // squares has no minimum at the truth here; mutual information does.
    let moving = make_volume(n, spacing, |q| {
        let mut x = q;
        for _ in 0..10 {
            x = q - true_disp(x);
        }
        let v = phantom(x);
        if v <= -900.0 {
            v
        } else {
            200.0 - v
        }
    });

    let params = RegParams {
        method: RegMethod::PlastimatchBSpline,
        metric: Metric::MutualInformation,
        levels: 2,
        iterations: 40,
        grid_spacing_mm: 30.0,
        regularization: 0.02,
        ..RegParams::default()
    };
    let progress = Progress::default();
    let t0 = std::time::Instant::now();
    let res = register(&fixed, &moving, &params, &progress).expect("registration runs");
    eprintln!(
        "plastimatch MI: −MI {:.4} → {:.4} in {:?} ({} evals)",
        res.initial_metric,
        res.final_metric,
        t0.elapsed(),
        res.iterations_run
    );
    // −MI is minimized, so it must go down.
    assert!(
        res.final_metric < res.initial_metric - 1e-4,
        "mutual information did not improve ({:.4} → {:.4})",
        res.initial_metric,
        res.final_metric
    );
    let p = Vec3::new(10.0, 5.0, 0.0);
    let err = (res.transform.map(p) - (p + true_disp(p))).length();
    eprintln!("plastimatch MI: mapping error at the bump centre {err:.2} mm");
    assert!(err < 5.0, "MI mapping error {err:.2} mm >= 5 mm");
}

#[test]
fn the_landmark_warp_lands_exactly_on_its_pairs() {
    let n = 24;
    let spacing = 6.0;
    let fixed = make_volume(n, spacing, phantom);
    let moving = make_volume(n, spacing, phantom);

    // Eight corners of a cube plus the centre, each shifted by a known
    // amount that varies over the volume, so the warp is not a global shift.
    let mut landmarks = Vec::new();
    let mut idx = 0;
    for x in [-50.0f64, 50.0] {
        for y in [-40.0f64, 40.0] {
            for z in [-60.0f64, 60.0] {
                let p = Vec3::new(x, y, z);
                let d = Vec3::new(0.03 * y, 0.04 * z, -0.02 * x);
                idx += 1;
                landmarks.push(LandmarkPair::new(format!("L{idx}"), p, p + d));
            }
        }
    }
    let params = RegParams {
        method: RegMethod::PlastimatchLandmark,
        landmark: LandmarkParams {
            kernel: LandmarkKernel::ThinPlate,
            stiffness: 0.0,
            radius_mm: 60.0,
        },
        landmarks: landmarks.clone(),
        ..RegParams::default()
    };
    let progress = Progress::default();
    let res = register(&fixed, &moving, &params, &progress).expect("landmark warp runs");
    for l in &landmarks {
        let err = (res.transform.map(l.fixed) - l.moving).length();
        assert!(err < 1e-4, "{}: landed {err:.5} mm off", l.name);
    }
    eprintln!(
        "landmarks: residual {:.5} mm, displacement {}",
        res.final_metric,
        res.analysis.displacement.line()
    );
    assert!(res.final_metric < 1e-4);
    assert!(res.analysis.displacement.max > 0.5);

    // The compactly supported kernel must leave everything far away alone.
    let local = RegParams {
        landmark: LandmarkParams {
            kernel: LandmarkKernel::Wendland,
            stiffness: 0.0,
            radius_mm: 20.0,
        },
        ..params.clone()
    };
    let res = register(&fixed, &moving, &local, &progress).expect("wendland warp runs");
    let far = Vec3::new(0.0, 0.0, 0.0);
    assert_eq!(res.transform.map(far), far, "the centre is out of reach");
    for l in &landmarks {
        let err = (res.transform.map(l.fixed) - l.moving).length();
        assert!(err < 1e-4, "{}: landed {err:.5} mm off", l.name);
    }
}

#[test]
fn a_local_registration_leaves_the_rest_of_the_volume_alone() {
    let n = 48;
    let spacing = 4.0;
    let centre = Vec3::new(30.0, 20.0, 10.0);
    // The moving image differs from the fixed one only inside one blob,
    // which is displaced by 5 mm.
    let shift = Vec3::new(5.0, 0.0, 0.0);
    let fixed = make_volume(n, spacing, phantom);
    let moving = make_volume(n, spacing, |q| {
        if (q - centre).length() < 22.0 {
            phantom(q - shift)
        } else {
            phantom(q)
        }
    });

    // A mask over that blob, as a segmentation would give.
    let mut mask = vec![0u8; n * n * n];
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let p = fixed.voxel_to_patient(i as f64, j as f64, k as f64);
                if (p - centre).length() < 16.0 {
                    mask[k * n * n + j * n + i] = 1;
                }
            }
        }
    }
    let region = std::sync::Arc::new(
        RegionMask::from_mask(&fixed, &mask, "blob".into(), 8.0).expect("a region"),
    );
    assert!(region.voxels() > 100);

    let params = RegParams {
        method: RegMethod::ElastixBSpline,
        levels: 2,
        iterations: 300,
        samples: 4000,
        grid_spacing_mm: 12.0,
        region: Some(region.clone()),
        ..RegParams::default()
    };
    let progress = Progress::default();
    let res = register(&fixed, &moving, &params, &progress).expect("local registration runs");
    eprintln!(
        "local: {} · {}",
        res.metric_line(),
        res.analysis.displacement.line()
    );
    assert_eq!(res.region.as_deref(), Some("blob"));

    // Inside: the blob's own displacement is recovered.
    let inside = res.transform.displacement(centre);
    eprintln!("local: displacement at the blob centre {inside:?}");
    assert!(
        (inside - shift).length() < 3.0,
        "inside the region: {inside:?} vs {shift:?}"
    );
    // Outside: the lattice does not reach, so nothing moved at all.
    for p in [
        Vec3::new(-40.0, -30.0, -40.0),
        Vec3::new(0.0, 0.0, -60.0),
        Vec3::new(-50.0, 30.0, 20.0),
    ] {
        let d = res.transform.displacement(p).length();
        assert!(d < 1e-9, "a local run moved {p:?} by {d} mm");
    }

    // The analytics are measured inside the region, not over the volume.
    assert!(res.analysis.samples > 0);
    assert!(res.analysis.displacement.max > 1.0);

    // Refining adds to an existing result rather than replacing it.
    let global = RegParams {
        method: RegMethod::ElastixRigid,
        levels: 2,
        iterations: 200,
        ..RegParams::default()
    };
    let base = register(&fixed, &moving, &global, &progress).expect("global runs");
    let refine = RegParams {
        method: RegMethod::ElastixBSpline,
        levels: 2,
        iterations: 200,
        samples: 4000,
        grid_spacing_mm: 12.0,
        region: Some(region),
        start: Some(base.transform.clone()),
        ..RegParams::default()
    };
    let refined = register(&fixed, &moving, &refine, &progress).expect("refinement runs");
    for p in [Vec3::new(-40.0, -30.0, -40.0), Vec3::new(0.0, 0.0, -60.0)] {
        let a = base.transform.map(p);
        let b = refined.transform.map(p);
        assert!(
            (a - b).length() < 1e-9,
            "the refinement changed {p:?} outside its region"
        );
    }
}

#[test]
fn the_vector_field_reproduces_the_transform_it_was_sampled_from() {
    let n = 40;
    let spacing = 4.0;
    let fixed = make_volume(n, spacing, phantom);
    let t_true = RigidTransform::new([0.0, 0.0, 0.05, 4.0, -2.0, 1.0], Vec3::ZERO);
    let moving = make_volume(n, spacing, |p| phantom(t_true.unmap(p)));
    let params = RegParams {
        method: RegMethod::ElastixRigid,
        levels: 2,
        iterations: 250,
        ..RegParams::default()
    };
    let progress = Progress::default();
    let res = register(&fixed, &moving, &params, &progress).expect("registration runs");

    let field = VectorField::sample(&fixed, &res.transform, None, 8.0);
    eprintln!("field: {}", field.describe());
    assert!(field.len() > 100);
    assert!(field.max_mag > 1.0);
    // Interpolating the lattice must agree with evaluating the transform.
    let mut worst = 0.0f64;
    for p in [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(21.0, -13.0, 7.0),
        Vec3::new(-34.0, 22.0, -19.0),
    ] {
        let d = (field.sample_patient(p) - res.transform.displacement(p)).length();
        worst = worst.max(d);
    }
    eprintln!("field: worst interpolation error {worst:.4} mm");
    assert!(worst < 0.05, "field interpolation off by {worst:.4} mm");
}
