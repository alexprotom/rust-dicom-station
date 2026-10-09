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
    register, register_cached, Init, LandmarkKernel, LandmarkPair, LandmarkParams, Metric,
    PyramidCache, RegMethod, RegParams, RegionMask, RigidTransform, VectorField,
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

/// The same phantom filed in another frame of reference: the moving volume
/// sits `offset` mm away in patient coordinates, with the object in the
/// middle of it, so at the identity the two do not overlap at all. That is
/// what a cardiac CT and a 4DCT of one patient look like to the engine.
fn make_volume_at(n: usize, spacing: f64, offset: Vec3, f: impl Fn(Vec3) -> f32 + Sync) -> Volume {
    // `f` takes patient coordinates of the moving frame.
    let mut v = make_volume(n, spacing, |p| f(p + offset));
    v.origin = v.origin + offset;
    v
}

#[test]
fn two_volumes_that_do_not_overlap_are_initialised_before_the_search() {
    let n = 64;
    let spacing = 3.0;
    // Nothing overlaps at the identity: the moving volume is a whole
    // volume's width away along y and z.
    let offset = Vec3::new(20.0, -400.0, 700.0);
    let small = RigidTransform::new(
        [
            1.5f64.to_radians(),
            -1.0f64.to_radians(),
            2.0f64.to_radians(),
            3.0,
            -2.0,
            4.0,
        ],
        offset,
    );
    // T_true maps fixed → moving: shift into the other frame, then the
    // small motion about the object's centre there.
    let t_true = |p: Vec3| small.map(p + offset);
    let fixed = make_volume(n, spacing, phantom);
    let moving = make_volume_at(n, spacing, offset, |q| {
        // q is in the moving frame; the object there is the fixed phantom
        // carried by T_true, so M(T_true x) = F(x).
        phantom(small.unmap(q) - offset)
    });
    let base = RegParams {
        method: RegMethod::ElastixRigid,
        levels: 3,
        iterations: 300,
        samples: 4000,
        ..RegParams::default()
    };
    let probes = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(40.0, 10.0, 20.0),
        Vec3::new(-30.0, 25.0, -25.0),
        Vec3::new(10.0, -40.0, 30.0),
    ];
    let max_err = |res: &rust_dicom_station::registration::RegistrationResult| {
        probes
            .iter()
            .map(|&p| (res.transform.map(p) - t_true(p)).length())
            .fold(0.0, f64::max)
    };

    // From the identity there is nothing to follow, and the engine says so
    // rather than handing the identity back as a result.
    let e = match register(
        &fixed,
        &moving,
        &RegParams {
            init: Init::Identity,
            ..base.clone()
        },
        &Progress::default(),
    ) {
        Ok(_) => panic!("no overlap at the identity, yet a result came back"),
        Err(e) => e,
    };
    assert!(
        format!("{e:#}").contains("do not overlap"),
        "the error explains itself: {e:#}"
    );

    // Automatic: the images do not overlap, so the centres of gravity are
    // matched and the search starts within a few millimetres of the answer.
    let res = register(&fixed, &moving, &base, &Progress::default()).expect("auto init runs");
    let err = max_err(&res);
    eprintln!(
        "auto init: MSD {:.1} → {:.1}, max mapping error {err:.2} mm",
        res.initial_metric, res.final_metric
    );
    assert!(
        res.initial_metric.is_finite(),
        "the report starts from the initialisation"
    );
    assert!(err < 1.5, "max mapping error {err:.2} mm >= 1.5 mm");

    // Two matched points (the centroids of a structure contoured on both):
    // the object's centre in each frame.
    let res = register(
        &fixed,
        &moving,
        &RegParams {
            init: Init::Points {
                fixed: Vec3::ZERO,
                moving: small.map(offset),
            },
            ..base.clone()
        },
        &Progress::default(),
    )
    .expect("point init runs");
    let err = max_err(&res);
    eprintln!("point init: max mapping error {err:.2} mm");
    assert!(err < 1.5, "max mapping error {err:.2} mm >= 1.5 mm");
}

#[test]
fn a_same_frame_pair_still_starts_from_the_identity() {
    // The automatic initialisation must not disturb the ordinary case: two
    // volumes of one frame that overlap start exactly where they used to.
    let n = 32;
    let fixed = make_volume(n, 4.0, phantom);
    let moving = make_volume(n, 4.0, |p| phantom(p - Vec3::new(5.0, 0.0, 0.0)));
    let params = RegParams {
        method: RegMethod::ElastixRigid,
        levels: 1,
        iterations: 1,
        samples: 2000,
        ..RegParams::default()
    };
    let res = register(&fixed, &moving, &params, &Progress::default()).unwrap();
    let auto = res.initial_metric;
    let res = register(
        &fixed,
        &moving,
        &RegParams {
            init: Init::Identity,
            ..params
        },
        &Progress::default(),
    )
    .unwrap();
    // The two are the same 2000-sample mean; only the parallel summation
    // order differs between runs, which is a rounding error, not a start.
    assert!(
        (auto - res.initial_metric).abs() <= 1e-9 * auto.abs().max(1.0),
        "auto is the identity when the images overlap: {auto} vs {}",
        res.initial_metric
    );
}

#[test]
fn a_kept_pyramid_registers_exactly_as_a_fresh_one() {
    // A workflow hands one pyramid cache to all its registrations; what it
    // saves is building the pyramids, never a digit of the result - in
    // either role, and after a third volume has come and gone.
    use std::sync::Arc;
    let n = 32;
    let fixed = Arc::new(make_volume(n, 4.0, phantom));
    let moving = Arc::new(make_volume(n, 4.0, |p| {
        phantom(p - Vec3::new(3.0, -2.0, 1.0))
    }));
    let other = Arc::new(make_volume(n, 4.0, |p| phantom(p * 1.02)));
    let params = RegParams {
        method: RegMethod::ElastixRigid,
        levels: 2,
        iterations: 40,
        samples: 2000,
        ..RegParams::default()
    };
    let probes = [Vec3::new(10.0, -20.0, 5.0), Vec3::new(-30.0, 15.0, 25.0)];
    let bits = |r: &rust_dicom_station::registration::RegistrationResult| -> Vec<u64> {
        probes
            .iter()
            .flat_map(|&p| {
                let q = r.transform.map(p);
                [q.x.to_bits(), q.y.to_bits(), q.z.to_bits()]
            })
            .chain([r.final_metric.to_bits()])
            .collect()
    };
    let pr = Progress::default();
    let fresh = bits(&register(&fixed, &moving, &params, &pr).unwrap());
    let back = bits(&register(&moving, &fixed, &params, &pr).unwrap());
    assert_eq!(
        fresh,
        bits(&register(&fixed, &moving, &params, &pr).unwrap()),
        "a registration is repeatable to the bit"
    );
    let mut cache = PyramidCache::default();
    for _ in 0..2 {
        let r = register_cached(&fixed, &moving, &params, &mut cache, &pr).unwrap();
        assert_eq!(bits(&r), fresh, "cached, fixed and moving as given");
        let r = register_cached(&moving, &fixed, &params, &mut cache, &pr).unwrap();
        assert_eq!(bits(&r), back, "cached, the roles swapped");
        register_cached(&other, &moving, &params, &mut cache, &pr).unwrap();
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

// ---------------------------------------------------------------------------
// Registration by structures (registration::shape)
// ---------------------------------------------------------------------------
//
// The images below hold nothing but zeros: whatever the engine recovers, it
// recovered from the structures, which is the point of it.

mod shapes {
    use super::*;
    use rust_dicom_station::registration::shape;
    pub use rust_dicom_station::registration::{ShapeDof, ShapePair, ShapeRequest, Transform3};

    /// An image of zeros: `n³` voxels at `spacing`, voxel 0 at `origin`.
    pub fn blank(n: usize, spacing: f64, origin: Vec3) -> Volume {
        Volume {
            data: vec![0i16; n * n * n],
            dims: [n, n, n],
            spacing: [spacing; 3],
            origin,
            row_dir: Vec3::new(1.0, 0.0, 0.0),
            col_dir: Vec3::new(0.0, 1.0, 0.0),
            normal: Vec3::new(0.0, 0.0, 1.0),
            frame_of_reference_uid: String::new(),
            min_value: 0,
            max_value: 0,
        }
    }

    /// A mask on `vol`'s lattice: the voxels whose centre `inside` accepts.
    pub fn mask(vol: &Volume, inside: impl Fn(Vec3) -> bool) -> Vec<u8> {
        let [nx, ny, nz] = vol.dims;
        let mut m = vec![0u8; nx * ny * nz];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    m[k * nx * ny + j * nx + i] =
                        inside(vol.voxel_to_patient(i as f64, j as f64, k as f64)) as u8;
                }
            }
        }
        m
    }

    /// An ellipsoid: centre and semi-axes, mm.
    #[derive(Clone, Copy)]
    pub struct Ell {
        pub c: Vec3,
        pub r: Vec3,
    }

    impl Ell {
        pub fn holds(&self, p: Vec3) -> bool {
            let d = p - self.c;
            (d.x / self.r.x).powi(2) + (d.y / self.r.y).powi(2) + (d.z / self.r.z).powi(2) <= 1.0
        }
        /// Its centre and the six ends of its axes: where the error of a
        /// transform is measured.
        pub fn probes(&self) -> Vec<Vec3> {
            let mut out = vec![self.c];
            for s in [-1.0, 1.0] {
                out.push(self.c + Vec3::new(s * self.r.x, 0.0, 0.0));
                out.push(self.c + Vec3::new(0.0, s * self.r.y, 0.0));
                out.push(self.c + Vec3::new(0.0, 0.0, s * self.r.z));
            }
            out
        }
    }

    pub const A: Ell = Ell {
        c: Vec3 {
            x: 5.0,
            y: -3.0,
            z: 4.0,
        },
        r: Vec3 {
            x: 22.0,
            y: 15.0,
            z: 18.0,
        },
    };
    pub const B: Ell = Ell {
        c: Vec3 {
            x: -24.0,
            y: 20.0,
            z: -16.0,
        },
        r: Vec3 {
            x: 9.0,
            y: 8.0,
            z: 10.0,
        },
    };
    pub const C: Ell = Ell {
        c: Vec3 {
            x: 24.0,
            y: 22.0,
            z: -22.0,
        },
        r: Vec3 {
            x: 7.0,
            y: 7.0,
            z: 7.0,
        },
    };

    /// The transform the moving image was made with (fixed → moving).
    pub fn truth(shift: Vec3) -> RigidTransform {
        RigidTransform::new([0.05, -0.04, 0.08, shift.x, shift.y, shift.z], Vec3::ZERO)
    }

    /// The fixed image, a moving image on another lattice, and a pair per
    /// ellipsoid: the fixed one as it is, the moving one carried by `t`,
    /// with `extra[i]` added to where ellipsoid `i` sits on the moving side.
    pub fn scene(
        ells: &[Ell],
        t: &RigidTransform,
        moving_origin: Vec3,
        extra: &[Vec3],
    ) -> (Volume, Volume, Vec<ShapePair>) {
        let fixed = blank(64, 1.5, Vec3::new(-47.25, -47.25, -47.25));
        let moving = blank(76, 1.3, moving_origin);
        let pairs = ells
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let off = extra.get(i).copied().unwrap_or(Vec3::ZERO);
                ShapePair {
                    name: format!("s{i}"),
                    color: [200, 80, 40],
                    weight: 1.0,
                    fixed: mask(&fixed, |p| e.holds(p)),
                    moving: mask(&moving, |q| e.holds(t.unmap(q - off))),
                }
            })
            .collect();
        (fixed, moving, pairs)
    }

    /// The largest distance between where `got` and `want` send the probes.
    pub fn tre(got: &Transform3, want: &RigidTransform, probes: &[Vec3]) -> f64 {
        probes
            .iter()
            .map(|&p| (got.map(p) - want.map(p)).length())
            .fold(0.0, f64::max)
    }

    /// The angle between two rotations, rad.
    pub fn angle_between(got: &Transform3, want: &RigidTransform) -> f64 {
        let a = got.rigid.matrix();
        let b = want.matrix();
        // trace(A Bᵀ) = Σ a_ij b_ij.
        let tr: f64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        ((tr - 1.0) * 0.5).clamp(-1.0, 1.0).acos()
    }

    pub fn request(pairs: Vec<ShapePair>) -> ShapeRequest {
        ShapeRequest {
            pairs,
            ..ShapeRequest::default()
        }
    }

    pub fn run(fixed: &Volume, moving: &Volume, req: &ShapeRequest) -> shape::ShapeOutcome {
        shape::register(fixed, moving, req, &Progress::default()).expect("registration runs")
    }
}

#[test]
fn structures_alone_recover_a_rigid_transform() {
    use shapes::*;
    let t = truth(Vec3::new(6.0, -4.0, 5.0));
    let (fixed, moving, pairs) = scene(&[A, B], &t, Vec3::new(-50.0, -52.0, -46.0), &[]);
    let out = run(&fixed, &moving, &request(pairs));
    let res = &out.result;
    let probes: Vec<Vec3> = [A, B].iter().flat_map(|e| e.probes()).collect();
    let err = tre(&res.transform, &t, &probes);
    let ang = angle_between(&res.transform, &t).to_degrees();
    eprintln!(
        "structures: {} · TRE {err:.3} mm · rotation off by {ang:.3} deg",
        res.metric_line()
    );
    for l in &out.report.lines {
        eprintln!("  {}", l.line());
    }
    assert_eq!(res.method, RegMethod::ShapeRigid);
    assert_eq!(res.metric, Metric::SurfaceDistance);
    assert!(err < 0.5, "probes land {err:.3} mm off");
    assert!(ang < 0.3, "rotation off by {ang:.3} deg");
    // What is left is the voxels: each surface is a staircase of 1.3 and
    // 1.5 mm steps, so the two never meet closer than about half a voxel.
    assert!(
        res.final_metric < 0.9,
        "surfaces meet: {}",
        res.final_metric
    );
    assert!(res.final_metric < res.initial_metric);
    assert!(res.transform.warp.is_none());
    for l in &out.report.lines {
        assert!(l.mean_after_mm < l.mean_before_mm, "{}", l.line());
        assert!(l.dice_after.unwrap() > 0.9, "{}", l.line());
    }
    assert_eq!(res.region.as_deref(), Some("s0, s1"));
}

#[test]
fn the_weights_decide_which_structures_the_fit_follows() {
    use shapes::*;
    let t = truth(Vec3::new(3.0, 2.0, -4.0));
    // The third structure is 8 mm out of place on the moving image: a
    // contour that disagrees with the other two.
    let extra = [Vec3::ZERO, Vec3::ZERO, Vec3::new(8.0, 0.0, 0.0)];
    let (fixed, moving, mut pairs) = scene(&[A, B, C], &t, Vec3::new(-50.0, -50.0, -50.0), &extra);
    let probes: Vec<Vec3> = [A, B].iter().flat_map(|e| e.probes()).collect();

    let equal = run(&fixed, &moving, &request(pairs.clone()));
    let e_equal = tre(&equal.result.transform, &t, &probes);
    pairs[2].weight = 0.1;
    let light = run(&fixed, &moving, &request(pairs));
    let e_light = tre(&light.result.transform, &t, &probes);
    eprintln!("weights: equal {e_equal:.2} mm, the stray one at 0.1 {e_light:.2} mm");
    assert!(
        e_light < 1.0,
        "the two heavy structures are followed: {e_light:.2}"
    );
    assert!(
        e_light * 2.5 < e_equal,
        "a light weight takes the stray structure out of it: {e_light:.2} vs {e_equal:.2}"
    );
}

#[test]
fn translation_only_leaves_the_rotations_at_zero() {
    use shapes::*;
    let t = truth(Vec3::new(6.0, -4.0, 5.0));
    let (fixed, moving, pairs) = scene(&[A, B], &t, Vec3::new(-50.0, -52.0, -46.0), &[]);
    let mut req = request(pairs);
    req.dof = ShapeDof::Translation;
    let out = run(&fixed, &moving, &req);
    let prm = out.result.transform.rigid.params();
    eprintln!("translation only: {:?} · {}", prm, out.result.metric_line());
    assert_eq!(&prm[..3], &[0.0, 0.0, 0.0]);
    assert_eq!(out.report.dof, ShapeDof::Translation);
    // What is left is about the shift of the structures' centre.
    let c = out.report.rigid_center.unwrap();
    let want = t.map(c) - c;
    let got = Vec3::new(prm[3], prm[4], prm[5]);
    assert!(
        (got - want).length() < 2.0,
        "translation {got:?} against the centre's shift {want:?}"
    );
    assert!(out.result.final_metric < out.result.initial_metric);
}

#[test]
fn structures_in_two_frames_of_reference_find_each_other() {
    use shapes::*;
    // 200 mm apart in patient coordinates: the moving image's lattice sits
    // there too, so the two images do not overlap at all.
    let t = truth(Vec3::new(206.0, -4.0, 5.0));
    let (fixed, moving, pairs) = scene(&[A, B], &t, Vec3::new(150.0, -52.0, -46.0), &[]);
    let probes: Vec<Vec3> = [A, B].iter().flat_map(|e| e.probes()).collect();
    let out = run(&fixed, &moving, &request(pairs));
    let err = tre(&out.result.transform, &t, &probes);
    eprintln!("two frames: {} · TRE {err:.3} mm", out.result.metric_line());
    assert!(err < 0.5, "found from the centroids: {err:.3} mm");
    // The centroids closed the 206 mm before the search took a step.
    assert!(
        out.result.initial_metric < 5.0,
        "the search started next to the answer: {}",
        out.result.initial_metric
    );
}

#[test]
fn a_registration_by_structures_reproduces_itself_on_any_thread_count() {
    use shapes::*;
    let t = truth(Vec3::new(6.0, -4.0, 5.0));
    let (fixed, moving, pairs) = scene(&[A, B], &t, Vec3::new(-50.0, -52.0, -46.0), &[]);
    let mut req = request(pairs);
    req.robust_mm = Some(2.0);
    let on = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| run(&fixed, &moving, &req))
    };
    let one = on(1);
    let three = on(3);
    assert_eq!(
        one.result.transform.rigid.params(),
        three.result.transform.rigid.params()
    );
    assert_eq!(one.result.final_metric, three.result.final_metric);
    assert_eq!(one.report.rigid_iterations, three.report.rigid_iterations);
}

#[test]
fn a_refinement_by_structures_follows_a_local_deformation() {
    use shapes::*;
    let t = truth(Vec3::new(4.0, -3.0, 2.0));
    let fixed = blank(64, 1.5, Vec3::new(-47.25, -47.25, -47.25));
    let moving = blank(76, 1.3, Vec3::new(-50.0, -52.0, -46.0));
    // The moving structure is the fixed one carried by `t` with a 5 mm bump
    // pushed out of its top: no rigid body fits that.
    let bump_at = A.c + Vec3::new(6.0, 0.0, A.r.z);
    let bump = |x: Vec3| Vec3::new(0.0, 0.0, 5.0) * (-(x - bump_at).length().powi(2) / 128.0).exp();
    let pair = ShapePair {
        name: "s0".into(),
        color: [0, 0, 0],
        weight: 1.0,
        fixed: mask(&fixed, |p| A.holds(p)),
        moving: mask(&moving, |q| {
            let x = t.unmap(q);
            A.holds(x - bump(x))
        }),
    };
    let rigid = run(&fixed, &moving, &request(vec![pair.clone()]));
    let mut req = request(vec![pair]);
    req.refine = Some(RegParams {
        method: RegMethod::ElastixBSpline,
        levels: 2,
        iterations: 200,
        samples: 3000,
        grid_spacing_mm: 10.0,
        ..RegParams::default()
    });
    req.margin_mm = 8.0;
    let out = run(&fixed, &moving, &req);
    let (r, d) = (&rigid.report.lines[0], &out.report.lines[0]);
    eprintln!(
        "rigid: {}\nrefined: {} ({:?})",
        r.line(),
        d.line(),
        d.refine_line
    );
    assert_eq!(out.result.method, RegMethod::ShapeDeformable);
    assert!(d.refine_line.is_some());
    // Down to the voxel floor the rigid case also reaches (about 0.6 mm on
    // these lattices), which no rigid body gets to with the bump there.
    assert!(
        d.mean_after_mm < r.mean_after_mm - 0.25 && d.mean_after_mm < 0.7,
        "the refinement lays the bump down: {:.2} against {:.2} mm",
        d.mean_after_mm,
        r.mean_after_mm
    );
    assert!(d.dice_after.unwrap() > r.dice_after.unwrap());
    assert!(d.folded_fraction.unwrap() < 0.01);
    // Far from the structure the lattice does not reach: the rigid result.
    let tr = &out.result.transform;
    for p in [Vec3::new(-44.0, -44.0, -44.0), Vec3::new(40.0, 40.0, -40.0)] {
        assert!((tr.map(p) - tr.rigid.map(p)).length() < 1e-9);
    }
    // And a refinement of that result on its own (no rigid stage) starts
    // where it ended.
    let mut again = req.clone();
    again.start = Some(out.result.transform.clone());
    let twice = run(&fixed, &moving, &again);
    assert_eq!(twice.report.rigid_iterations, 0);
    assert!(twice.report.lines[0].mean_after_mm <= d.mean_after_mm + 0.05);
}

#[test]
fn a_registration_by_structures_refuses_what_it_cannot_do() {
    use rust_dicom_station::registration::shape;
    use shapes::*;
    let t = truth(Vec3::ZERO);
    let (fixed, moving, mut pairs) = scene(&[A], &t, Vec3::new(-50.0, -50.0, -50.0), &[]);
    let p = Progress::default();
    let err = |req: &ShapeRequest| {
        format!(
            "{:#}",
            shape::register(&fixed, &moving, req, &p)
                .err()
                .expect("refused")
        )
    };
    assert!(err(&request(Vec::new())).contains("at least one structure"));
    let mut req = request(pairs.clone());
    req.start = Some(std::sync::Arc::new(Transform3::rigid_only(
        RigidTransform::identity(Vec3::ZERO),
    )));
    assert!(err(&req).contains("deformable stage"));
    let mut req = request(pairs.clone());
    req.refine = Some(RegParams {
        method: RegMethod::ElastixRigid,
        ..RegParams::default()
    });
    assert!(err(&req).contains("B-spline"));
    pairs[0].moving.iter_mut().for_each(|v| *v = 0);
    assert!(err(&request(pairs)).contains("empty on the moving image"));
}
