//! The posture model against analytic ground truth.
//!
//! The articulated phantom (`gen_test_data::articulated_pair`) is two rigid
//! "bones" joined at a hinge in soft tissue with an organ that grows, and a
//! second state made by a known hinge angle and placed elsewhere in its
//! scanner. Everything a table reports about it is known exactly: the
//! anchor's placement, the lower bone's screw, the joint angle, the organ's
//! volume change. These tests build the model the way the window, the CLI
//! and the MCP tool do (`workflow::posture::build`) and hold it to that.
//!
//! The mathematics on its own (exp / log of SE(3), the flow of a velocity
//! field, the poly-rigid blend) is tested to machine precision in the unit
//! tests of `crates/rds-domain/src/posture/`.
//!
//! Three tests are `#[ignore]`d: the model fitted back onto a known
//! transform of the bundled 4DCT phase (`data-test/`, or `RDS_POSTURE_DATA`
//! naming a CT folder with a structure set); a real pair of one patient in
//! two positions (`RDS_POSTURE_PAIR`, two states separated by `;`, each a
//! folder or `SERIES_DIR::STRUCTURE_SET_FILE` as `posture_cli` takes them,
//! and `RDS_POSTURE_ANCHOR`, a structure on both, else the whole image),
//! held to what any real pair must give - no folding, a consistent inverse,
//! bones rigid away from their joints, bodies that fit; and the live slices
//! on the GPU against the CPU's, with
//!
//! ```text
//! cargo test --release --test posture -- --ignored real
//! RDS_GPU_TESTS=1 cargo test --release --test posture -- --ignored gpu
//! ```
//!
//! No real data is in the repository: the paths come from the environment.

mod common;

use std::sync::Arc;

use rust_dicom_station::gen_test_data::{self, ArticulatedParams};
use rust_dicom_station::loader::{self, LoadedStudy};
use rust_dicom_station::posture::{self, apply, file, Anchor, Mode};
use rust_dicom_station::progress::Progress;
use rust_dicom_station::workflow::posture::{
    build, Bodies, PostureOutcome, PostureRequest, StateInput, WeplBeam,
};

fn states<'a>(pair: &'a [LoadedStudy]) -> Vec<StateInput<'a>> {
    pair.iter()
        .enumerate()
        .map(|(i, s)| StateInput {
            label: ["A", "B", "C"][i].to_string(),
            study: s,
            structure_set: None,
            posture: None,
        })
        .collect()
}

fn request(mode: Mode) -> PostureRequest {
    PostureRequest {
        anchor: Anchor::Structure("BONE_UPPER".into()),
        bodies: Bodies::Bones,
        mode,
        lattice_mm: 6.0,
        analysis_mm: 2.0,
        ..PostureRequest::default()
    }
}

fn run(pair: &[LoadedStudy], mode: Mode) -> PostureOutcome {
    build(&states(pair), &request(mode), &Progress::default()).expect("the model builds")
}

fn body<'a>(o: &'a PostureOutcome, name: &str) -> &'a posture::measure::BodyRow {
    o.tables
        .bodies
        .iter()
        .find(|b| b.name == name)
        .unwrap_or_else(|| panic!("no body {name}"))
}

#[test]
fn the_articulated_phantom_is_recovered() {
    let params = ArticulatedParams::default();
    let pair = gen_test_data::articulated_pair(&params);
    let o = run(&pair, Mode::Diffeomorphic);
    let t = &o.tables;

    // The anchor laid B's upper bone onto A's: the placement, read back.
    // Its centroid's shift is pinned by the whole surface; the placement's
    // turn is about the bone's own long axis, which a box of 12 x 7 voxels
    // pins only to a couple of degrees (a 6 degree roll moves its corners
    // by under a voxel), so that is held loosely.
    let into = &t.states[1].to_anchor;
    let upper_c = rust_dicom_station::geometry::Vec3::new(0.0, 0.0, 52.0);
    let want = params.placement().map(upper_c) - upper_c;
    assert!(
        (into.translation - want).length() < 1.5,
        "anchor shift {:?}, want {want:?}",
        into.translation
    );
    assert!(
        (into.rotation_deg[2] - params.placement_deg[2]).abs() < 3.0
            && into.rotation_deg[0].abs() < 1.0
            && into.rotation_deg[1].abs() < 1.0,
        "anchor rotation {:?}",
        into.rotation_deg
    );
    // Both bodies, the anchor still.
    assert_eq!(o.model.body_names().len(), 2, "{:?}", o.model.body_names());
    let upper = body(&o, "BONE_UPPER");
    assert!(upper.anchor);
    assert!(upper.screw.angle_deg < 0.5, "{}", upper.screw.angle_deg);

    // The lower bone turned on the hinge: angle, axis, no travel.
    let lower = body(&o, "BONE_LOWER");
    assert!(
        (lower.screw.angle_deg - params.angle_deg).abs() < 2.0,
        "lower bone {} deg",
        lower.screw.angle_deg
    );
    assert!(
        lower.screw.axis.x.abs() > 0.99,
        "axis {:?}",
        lower.screw.axis
    );
    assert!(
        lower.screw.travel_mm.abs() < 2.0,
        "travel {}",
        lower.screw.travel_mm
    );
    // The pivot is the hinge (the axis through the origin, along x).
    let pivot = lower.screw.point;
    assert!(pivot.y.hypot(pivot.z) < 4.0, "pivot {pivot:?}");
    // Halfway along, halfway round: the path is a screw.
    let mid = t.nearest(0.5);
    assert!(
        (lower.at[mid].angle_deg - params.angle_deg * t.times[mid]).abs() < 1.5,
        "at t = {}: {} deg",
        t.times[mid],
        lower.at[mid].angle_deg
    );

    // The joint, about RL.
    let joint = t.joints.first().expect("the two bones make a joint");
    let end = joint.at.last().unwrap();
    assert!(
        (end[0].abs() - params.angle_deg).abs() < 2.0,
        "joint {end:?}"
    );
    assert!(end[1].abs() < 2.0 && end[2].abs() < 2.0, "joint {end:?}");

    // Bones keep their shape at every time, and nothing folds.
    assert!(
        lower.max_stretch < 0.02,
        "lower bone stretches {}",
        lower.max_stretch
    );
    assert!(
        t.model.jacobian_min > 0.0,
        "Jacobian {}",
        t.model.jacobian_min
    );
    assert_eq!(t.model.folded_fraction, 0.0);
    // The way back undoes the way there.
    assert!(
        t.model.inverse_mm[1][1] < 0.5,
        "inverse {:?}",
        t.model.inverse_mm
    );

    // The organ grew as it was told to (contoured volumes), and the model
    // carries the lower bone onto its contour on B.
    let organ = t.structures.iter().find(|s| s.name == "ORGAN").unwrap();
    let want = params.organ_cm3();
    let (dv, _) = organ.volume_change();
    assert!(
        (dv - (want[1] - want[0])).abs() < 0.35 * (want[1] - want[0]),
        "organ change {dv} cm3, want {}",
        want[1] - want[0]
    );
    assert!(!organ.used);
    // The bones keep their volume as contoured: the phantom's own check.
    for name in ["BONE_UPPER", "BONE_LOWER"] {
        let b = t.structures.iter().find(|s| s.name == name).unwrap();
        let (_, pct) = b.volume_change();
        assert!(pct.abs() < 5.0, "{name} changes by {pct} %");
    }
    // The anchor bone does not move or stretch.
    assert!(
        upper.max_stretch < 0.02,
        "upper bone stretches {}",
        upper.max_stretch
    );
    let lb = t
        .structures
        .iter()
        .find(|s| s.name == "BONE_LOWER")
        .unwrap();
    let check = lb.check.expect("a model check");
    assert!(check.dice > 0.8, "lower bone carried: Dice {}", check.dice);
    assert!(lb.used);

    // The CSV names every table.
    let csv = t.csv();
    for part in [
        "States",
        "Structures",
        "Bones and joints",
        "Model",
        "BONE_LOWER",
    ] {
        assert!(csv.contains(part), "no {part} in the CSV");
    }
}

#[test]
fn the_linear_path_shares_the_end_points_and_shrinks_the_turning_bone() {
    let params = ArticulatedParams::default();
    let pair = gen_test_data::articulated_pair(&params);
    let diffeo = run(&pair, Mode::Diffeomorphic);
    let linear = run(&pair, Mode::Linear);
    // The same end point.
    let p = rust_dicom_station::geometry::Vec3::new(0.0, 0.0, -60.0);
    let a = diffeo.model.frame(1.0).forward.map(p);
    let b = linear.model.frame(1.0).forward.map(p);
    assert!((a - b).length() < 0.5, "{a:?} vs {b:?}");
    // On the way, the chord: the bone shrinks, which the screw never does.
    let dl = body(&diffeo, "BONE_LOWER").max_stretch;
    let ll = body(&linear, "BONE_LOWER").max_stretch;
    assert!(
        ll > 0.04 && ll > 2.0 * dl,
        "linear {ll}, diffeomorphic {dl}"
    );
    assert_eq!(linear.tables.model.mode, Mode::Linear);
}

#[test]
fn a_model_file_reads_back_the_same_path() {
    let pair = gen_test_data::articulated_pair(&ArticulatedParams::default());
    let o = run(&pair, Mode::Diffeomorphic);
    let dir = common::target_dir("posture-file");
    let path = dir.join(format!("model.{}", file::EXTENSION));
    file::write(&path, &o.model, &o.occupancy).unwrap();
    let back = file::read(&path).unwrap();
    assert_eq!(back.states.len(), 2);
    assert_eq!(back.anchor, o.model.anchor);
    assert_eq!(back.body_names(), o.model.body_names());
    let f0 = o.model.frame(0.6).forward;
    let f1 = back.frame(0.6).forward;
    let worst =
        f0.d.iter()
            .zip(&f1.d)
            .map(|(x, y)| (*x - *y).length())
            .fold(0.0, f64::max);
    assert!(worst < 1e-6, "{worst} mm apart");
    // Reopened onto the same states, the tables come back without a fit.
    let again = rust_dicom_station::workflow::posture::reopen(
        back,
        &states(&pair),
        &request(Mode::Diffeomorphic),
        &Progress::default(),
    )
    .unwrap();
    let a = body(&o, "BONE_LOWER").screw.angle_deg;
    let b = body(&again, "BONE_LOWER").screw.angle_deg;
    assert!((a - b).abs() < 0.05, "{a} vs {b} deg");
    assert_eq!(again.tables.structures.len(), o.tables.structures.len());
}

#[test]
fn the_cross_fade_lands_on_each_scan_and_key_frames_fit_the_budget() {
    let pair = gen_test_data::articulated_pair(&ArticulatedParams::default());
    let o = run(&pair, Mode::Diffeomorphic);
    let imgs: Vec<Arc<_>> = pair.iter().map(|s| s.volume.clone()).collect();
    let grid = o.model.grid.clone();
    let at0 = apply::render(
        &o.model,
        &imgs,
        &o.model.frame(0.0),
        apply::Show::CrossFade,
        &grid,
    )
    .unwrap();
    // At t = 0 the cross-fade is A itself.
    let differ = at0
        .data
        .iter()
        .zip(&pair[0].volume.data)
        .filter(|(a, b)| (**a - **b).abs() > 2)
        .count();
    assert!(
        differ * 1000 < at0.data.len(),
        "{differ} voxels differ at t = 0"
    );
    // At t = 1 the lower bone of B moved back... is where B's is in the
    // anchor frame: compare with B resampled through the anchor alone.
    let at1 = apply::render(
        &o.model,
        &imgs,
        &o.model.frame(1.0),
        apply::Show::CrossFade,
        &grid,
    )
    .unwrap();
    let g = o.model.to_state[1];
    let b_in_a = apply::resample(&pair[1].volume, &grid, |y| g.map(y));
    let bone = |v: &rust_dicom_station::volume::Volume| v.data.iter().filter(|&&x| x > 400).count();
    let both = at1
        .data
        .iter()
        .zip(&b_in_a.data)
        .filter(|(a, b)| **a > 400 && **b > 400)
        .count();
    let dice = 2.0 * both as f64 / (bone(&at1) + bone(&b_in_a)) as f64;
    assert!(dice > 0.8, "bone at t = 1 against B: Dice {dice}");
    // Key frames on a working grid within the budget.
    let budget = 4 << 20;
    let wg = apply::working_grid(&grid, 9, budget);
    assert!(wg.dims.iter().product::<usize>() * 2 * 9 <= budget);
    let kf = apply::key_frames(
        &o.model,
        &imgs,
        9,
        apply::Show::StartForward,
        &wg,
        &apply::Extras {
            dose: None,
            jacobian: true,
        },
        &Progress::default(),
    )
    .unwrap();
    assert!(kf[4].extras.jacobian.is_some());
    assert_eq!(kf.len(), 9);
    assert!((kf[8].t - 1.0).abs() < 1e-12);
    // The structures follow: the lower bone's contour carried to the end.
    let ss = &pair[0].structure_sets[0];
    let carried = apply::carry_structure_set(ss, &kf[8].forward, "t = 1".into());
    assert_eq!(carried.rois.len(), ss.rois.len());
}

#[test]
fn the_phantom_survives_a_dicom_round_trip() {
    let dir = common::target_dir("posture-dicom");
    let params = ArticulatedParams::default();
    let folders = gen_test_data::generate_articulated(&dir, &params, &Progress::default()).unwrap();
    let pair: Vec<LoadedStudy> = folders
        .iter()
        .map(|f| loader::load_directory(f, &Progress::default()).expect("the state loads"))
        .collect();
    for s in &pair {
        assert!(s.has_volume());
        let names: Vec<&str> = s.structure_sets[0]
            .rois
            .iter()
            .map(|r| r.name.as_str())
            .collect();
        for n in ["BODY", "BONE_UPPER", "BONE_LOWER", "ORGAN"] {
            assert!(names.contains(&n), "{n} missing: {names:?}");
        }
    }
    let o = run(&pair, Mode::Diffeomorphic);
    let lower = body(&o, "BONE_LOWER");
    assert!(
        (lower.screw.angle_deg - params.angle_deg).abs() < 2.0,
        "{} deg after the round trip",
        lower.screw.angle_deg
    );
}

#[test]
fn a_chain_of_three_states_runs_over_two_units_of_time() {
    let a = ArticulatedParams {
        angle_deg: 20.0,
        ..ArticulatedParams::default()
    };
    let b = ArticulatedParams::default();
    let first = gen_test_data::articulated_pair(&a);
    let second = gen_test_data::articulated_pair(&b);
    let chain = [first[0].clone(), first[1].clone(), second[1].clone()];
    let mut st = states(&chain);
    for (s, p) in st.iter_mut().zip([0.0, 20.0, 40.0]) {
        s.posture = Some(p);
    }
    let o = build(&st, &request(Mode::Diffeomorphic), &Progress::default()).unwrap();
    assert_eq!(o.model.segments.len(), 2);
    assert_eq!(o.model.span(), 2.0);
    assert_eq!(o.model.param_at(1.5), Some(30.0));
    assert_eq!(o.model.time_of_param(10.0), Some(0.5));
    let lower = body(&o, "BONE_LOWER");
    assert!(
        (lower.screw.angle_deg - 40.0).abs() < 2.5,
        "{}",
        lower.screw.angle_deg
    );
    let at1 = lower.at[o.tables.nearest(1.0)].angle_deg;
    assert!((at1 - 20.0).abs() < 2.0, "at the middle state: {at1}");
}

/// The three planes on screen are the volume's planes, to the bit: the
/// live slices and the exact frame never disagree.
#[test]
fn the_live_planes_are_the_rendered_volume_s_planes() {
    let pair = gen_test_data::articulated_pair(&ArticulatedParams::default());
    let o = run(&pair, Mode::Diffeomorphic);
    let imgs: Vec<Arc<_>> = pair.iter().map(|s| s.volume.clone()).collect();
    let grid = o.model.grid.clone();
    let frame = o.model.frame(0.37);
    let planes = [
        apply::Plane {
            axis: 2,
            index: grid.dims[2] / 2,
        },
        apply::Plane {
            axis: 0,
            index: grid.dims[0] / 3,
        },
        apply::Plane {
            axis: 1,
            index: grid.dims[1] / 2 + 3,
        },
    ];
    for show in apply::Show::ALL {
        let full = apply::render(&o.model, &imgs, &frame, show, &grid).unwrap();
        let vals = apply::render_planes(&o.model, &imgs, &frame, show, &grid, &planes).unwrap();
        // Written into a blank volume on the grid, the planes are the
        // full volume's.
        let mut canvas = (*pair[0].volume).clone();
        canvas.data.iter_mut().for_each(|v| *v = i16::MIN);
        apply::write_planes(&mut canvas, &planes, &vals);
        let [nx, ny, _] = grid.dims;
        for pl in &planes {
            let [nu, nv] = pl.size(&grid);
            for v in 0..nv {
                for u in 0..nu {
                    let [i, j, k] = pl.voxel(u, v);
                    let at = k * nx * ny + j * nx + i;
                    assert_eq!(
                        canvas.data[at], full.data[at],
                        "{show:?} plane {pl:?} at ({i}, {j}, {k})"
                    );
                }
            }
        }
    }
}

/// The device, when `RDS_GPU_TESTS` asks for the GPU tests.
#[cfg(feature = "gpu")]
fn gpu() -> Option<rust_dicom_station::nn::device::GpuContext> {
    std::env::var_os("RDS_GPU_TESTS")?;
    Some(
        rust_dicom_station::nn::device::GpuContext::try_new()
            .expect("RDS_GPU_TESTS is set but no GPU opens"),
    )
}

/// The live slices on the GPU against the CPU's: the same pixels to a
/// fraction of a grey level, but for the odd pixel where a pull-back lands
/// within single-precision rounding of a scan's edge.
#[cfg(feature = "gpu")]
#[test]
#[ignore = "needs a GPU and RDS_GPU_TESTS"]
fn gpu_slices_agree_with_the_cpu() {
    use rust_dicom_station::posture::gpu::GpuSlicer;
    let Some(ctx) = gpu() else { return };
    let pair = gen_test_data::articulated_pair(&ArticulatedParams::default());
    let o = run(&pair, Mode::Diffeomorphic);
    let imgs: Vec<Arc<_>> = pair.iter().map(|s| s.volume.clone()).collect();
    let slicer = GpuSlicer::new(&ctx, &imgs).unwrap();
    let grid = o.model.grid.clone();
    let planes: Vec<apply::Plane> = (0..3)
        .map(|axis| apply::Plane {
            axis,
            index: grid.dims[axis] / 2 + axis,
        })
        .collect();
    for t in [0.0, 0.37, 1.0] {
        let frame = o.model.frame(t);
        for show in apply::Show::ALL {
            let cpu = apply::render_planes(&o.model, &imgs, &frame, show, &grid, &planes).unwrap();
            let got = slicer
                .render_planes(&o.model, &frame, show, &grid, &planes)
                .unwrap();
            assert_eq!(got.len(), cpu.len());
            let (mut n, mut off, mut sum) = (0usize, 0usize, 0.0f64);
            for (a, b) in cpu.iter().zip(&got) {
                assert_eq!(a.len(), b.len());
                for (x, y) in a.iter().zip(b) {
                    let d = (x - y).abs() as f64;
                    n += 1;
                    sum += d;
                    if d > 1.0 {
                        off += 1;
                    }
                }
            }
            assert!(
                off * 500 < n && sum / (n as f64) < 0.1,
                "t {t} {show:?}: {off} of {n} pixels more than one grey level apart, mean {}",
                sum / n as f64
            );
        }
    }
}

/// A chain whose two steps turn at different rates: on curves, the lower
/// bone's angular speed is the same either side of the middle state; as
/// two screws it jumps from one rate to the other. The states are where
/// they were either way, and the bone stays rigid on its curve.
#[test]
fn a_chain_turns_smoothly_through_its_middle_state() {
    let ten = ArticulatedParams {
        angle_deg: 10.0,
        ..ArticulatedParams::default()
    };
    let forty = ArticulatedParams::default();
    let first = gen_test_data::articulated_pair(&ten);
    let second = gen_test_data::articulated_pair(&forty);
    let chain = [first[0].clone(), first[1].clone(), second[1].clone()];
    let smooth = build(
        &states(&chain),
        &request(Mode::Diffeomorphic),
        &Progress::default(),
    )
    .expect("the smooth chain builds");
    let mut req = request(Mode::Diffeomorphic);
    req.smooth_chain = false;
    let plain = build(&states(&chain), &req, &Progress::default()).expect("the plain chain builds");
    assert!(smooth.model.segments.iter().all(|s| s.curves.is_some()));
    assert!(plain.model.segments.iter().all(|s| s.curves.is_none()));
    let speed = |m: &posture::PostureModel, a: f64, b: f64| -> f64 {
        let ga = m.body_pose("BONE_LOWER", a).unwrap();
        let gb = m.body_pose("BONE_LOWER", b).unwrap();
        ga.inverse().then(&gb).angle_deg() / (b - a)
    };
    let h = 0.02;
    let (before, after) = (
        speed(&smooth.model, 1.0 - h, 1.0),
        speed(&smooth.model, 1.0, 1.0 + h),
    );
    assert!(
        (before - after).abs() < 0.1 * before.max(after),
        "smooth: {before:.2} then {after:.2} deg per unit of time"
    );
    let (pb, pa) = (
        speed(&plain.model, 1.0 - h, 1.0),
        speed(&plain.model, 1.0, 1.0 + h),
    );
    assert!(
        pa > 2.0 * pb,
        "two screws: {pb:.2} then {pa:.2} deg per unit"
    );
    // The same poses at the states.
    let p = rust_dicom_station::geometry::Vec3::new(0.0, 0.0, -60.0);
    for t in [1.0, 2.0] {
        let a = smooth.model.body_pose("BONE_LOWER", t).unwrap().map(p);
        let b = plain.model.body_pose("BONE_LOWER", t).unwrap().map(p);
        assert!((a - b).length() < 1e-6, "t {t}: {a:?} vs {b:?}");
        let f = smooth.model.frame(t).forward.map(p);
        assert!((f - a).length() < 0.05, "t {t}: field {f:?}, pose {a:?}");
    }
    // Inside the bone the field moves it as its curve says, between the
    // states too, and the bone keeps its shape.
    for t in [0.5, 1.5] {
        let want = smooth.model.body_pose("BONE_LOWER", t).unwrap().map(p);
        let got = smooth.model.frame(t).forward.map(p);
        assert!((got - want).length() < 0.05, "t {t}: {got:?} vs {want:?}");
    }
    assert!(
        body(&smooth, "BONE_LOWER").max_stretch < 0.02,
        "{}",
        body(&smooth, "BONE_LOWER").max_stretch
    );
    assert!(smooth
        .model
        .provenance
        .iter()
        .any(|l| l.contains("cubic curve")));
    // A model file keeps the curves.
    let dir = common::target_dir("posture-chain-file");
    let path = dir.join(format!("chain.{}", file::EXTENSION));
    file::write(&path, &smooth.model, &smooth.occupancy).unwrap();
    let back = file::read(&path).unwrap();
    let a = smooth.model.frame(0.5).forward.map(p);
    let b = back.frame(0.5).forward.map(p);
    assert!((a - b).length() < 1e-6, "read back: {a:?} vs {b:?}");
}

/// The WEPL to the organ along a beam from the front: its depth in the
/// phantom, organ and soft tissue, on each state.
#[test]
fn the_wepl_to_the_organ_is_its_depth_in_water() {
    use rust_dicom_station::sct::eval::Hlut;
    let params = ArticulatedParams::default();
    let pair = gen_test_data::articulated_pair(&params);
    let mut req = request(Mode::Diffeomorphic);
    req.wepl = Some(WeplBeam {
        gantry_deg: 0.0,
        couch_deg: 0.0,
        hlut: None,
    });
    let o = build(&states(&pair), &req, &Progress::default()).unwrap();
    let organ = o
        .tables
        .structures
        .iter()
        .find(|s| s.name == "ORGAN")
        .unwrap();
    assert_eq!(organ.wepl_mm.len(), o.tables.times.len());
    // Gantry 0, head first supine: the beam enters the front. From the
    // organ's centre (27 mm right of the axis) the ray crosses the organ's
    // radius (10 HU) and then soft tissue (40 HU) to the capsule's surface.
    let h = Hlut::default();
    let surface = (gen_test_data::ART_BODY_R.powi(2) - 27f64.powi(2)).sqrt();
    let want = |r: f64| r * h.rsp(10.0) + (surface - r) * h.rsp(40.0);
    let (first, last) = (organ.wepl_mm[0], *organ.wepl_mm.last().unwrap());
    let [ra, rb] = params.organ_r;
    assert!(
        (first - want(ra)).abs() < 1.5,
        "first state {first:.2} mm, want {:.2}",
        want(ra)
    );
    assert!(
        (last - want(rb)).abs() < 1.5,
        "last state {last:.2} mm, want {:.2}",
        want(rb)
    );
    assert!(o
        .tables
        .model
        .beam
        .as_deref()
        .is_some_and(|b| b.contains("gantry 0")));
    assert!(o.tables.csv().contains("WEPL"));
}

/// A CT folder with its structure set: `RDS_POSTURE_DATA`, else the
/// bundled phase folder, else - in the TCIA download's layout, where the
/// structure sets sit beside the phase folders - phase 0 and its set copied
/// into a folder of their own under `target/`.
fn real_data() -> std::path::PathBuf {
    use std::path::PathBuf;
    if let Ok(d) = std::env::var("RDS_POSTURE_DATA") {
        return PathBuf::from(d);
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data-test");
    let old = root.join("lung_p1_4DCT_phase_000");
    if old.exists() {
        return old;
    }
    let tcia = root.join("TCIA_4D-LUNG/P102/4DFBCT+RTS");
    let out = common::target_dir("posture-real");
    let set = tcia.join("RS_RTS__0.0_A.dcm");
    if set.exists() && !out.join("RS_RTS__0.0_A.dcm").exists() {
        for e in std::fs::read_dir(tcia.join("1_CT_4DFBCT__Gated__0.0_A")).unwrap() {
            let e = e.unwrap();
            std::fs::copy(e.path(), out.join(e.file_name())).unwrap();
        }
        std::fs::copy(&set, out.join("RS_RTS__0.0_A.dcm")).unwrap();
    }
    out
}

/// A real pair of one patient in two positions (supine and upright, say):
/// no ground truth, but what any model of it must be - invertible all the
/// way (no fold, the way back undoing the way there), every rigid body
/// fitting its two contours and staying rigid away from its joints, the
/// anchor still. Prints the tables' main lines.
#[test]
#[ignore = "needs RDS_POSTURE_PAIR: two states of one patient"]
fn real_pair_reads_a_plausible_posture() {
    let Ok(pair) = std::env::var("RDS_POSTURE_PAIR") else {
        eprintln!("RDS_POSTURE_PAIR is not set");
        return;
    };
    let specs: Vec<&str> = pair.split(';').filter(|s| !s.trim().is_empty()).collect();
    assert!(
        specs.len() >= 2,
        "RDS_POSTURE_PAIR names two states or more"
    );
    let studies: Vec<LoadedStudy> = specs
        .iter()
        .map(|s| {
            rust_dicom_station::workflow::posture::load_state(s.trim(), &Progress::default())
                .expect("a state loads")
        })
        .collect();
    let anchor = std::env::var("RDS_POSTURE_ANCHOR")
        .map(|a| Anchor::parse(&a))
        .unwrap_or(Anchor::Image);
    let req = PostureRequest {
        anchor: anchor.clone(),
        bodies: Bodies::Bones,
        wepl: Some(WeplBeam {
            gantry_deg: 90.0,
            couch_deg: 0.0,
            hlut: None,
        }),
        ..PostureRequest::default()
    };
    let t0 = std::time::Instant::now();
    let o = build(&states(&studies), &req, &Progress::default()).expect("the model builds");
    let secs = t0.elapsed().as_secs_f64();
    let t = &o.tables;
    eprintln!("built in {secs:.1} s; {} warning(s)", o.warnings.len());
    for b in &t.bodies {
        eprintln!(
            "  {:<20} {:6.2} deg  fit {:.2} mm  stretch max {:.3}, p95 off the joints {:.3}",
            b.name, b.screw.angle_deg, b.fit_mean_mm, b.max_stretch, b.stretch_p95
        );
    }
    for j in &t.joints {
        let e = j.at.last().copied().unwrap_or_default();
        eprintln!(
            "  joint {}: {:.1} {:.1} {:.1} deg",
            j.name, e[0], e[1], e[2]
        );
    }
    let m = &t.model;
    eprintln!(
        "  Jacobian min {:.3}, folded {}, inverse end max {:.2} mm p95 {:.3} mm",
        m.jacobian_min, m.folded_fraction, m.inverse_mm[1][0], m.inverse_mm[1][1]
    );
    assert!(t.bodies.len() >= 2, "{} bodies", t.bodies.len());
    assert!(
        m.jacobian_min > 0.0 && m.folded_fraction == 0.0,
        "the path folds"
    );
    assert!(
        m.inverse_mm[1][1] < 0.5,
        "inverse p95 {}",
        m.inverse_mm[1][1]
    );
    for b in &t.bodies {
        if b.anchor {
            assert!(b.screw.angle_deg < 0.5, "the anchor turned");
            continue;
        }
        assert!(
            b.fit_mean_mm <= posture::fit::POOR_FIT_MM,
            "{}: fit {} mm",
            b.name,
            b.fit_mean_mm
        );
        assert!(
            b.stretch_p95 < 0.15,
            "{}: stretch {} off the joints",
            b.name,
            b.stretch_p95
        );
        let row = t.structures.iter().find(|s| s.name == b.name).unwrap();
        let check = row.check.expect("a model check");
        assert!(check.dice > 0.75, "{}: carried Dice {}", b.name, check.dice);
    }
    // The WEPL column reads a depth for every structure the image holds.
    assert!(t
        .structures
        .iter()
        .any(|s| s.wepl_mm.first().is_some_and(|w| *w > 0.0)));
}

/// The model fitted back onto a known rigid motion of a real CT: the
/// simulator moves the whole study, every structure moves with it, and the
/// model's bodies must each read the motion back.
#[test]
#[ignore = "needs the bundled 4DCT (data-test/) or RDS_POSTURE_DATA"]
fn real_anatomy_reads_back_a_known_transform() {
    use rust_dicom_station::simulate::{generate_transformed_study, SimParams};
    let dir = real_data();
    let a = loader::load_directory(&dir, &Progress::default()).expect("data");
    let params = SimParams {
        translation: [8.0, -6.0, 4.0],
        rotation_deg: [0.0, 0.0, 3.0],
        ..SimParams::default()
    };
    let b = generate_transformed_study(&a, &params, &Progress::default());
    // The three largest structures of ten slices or more: one drawn on a
    // single slice (a carina mark) has no rotation to read back, and a
    // thin tube along the turn's axis (the cord) hides a turn about it.
    let area = |r: &rust_dicom_station::rtstruct::Roi| -> f64 {
        r.contours
            .iter()
            .map(|c| {
                let p = &c.points;
                (0..p.len())
                    .map(|i| {
                        let (a, b) = (p[i], p[(i + 1) % p.len()]);
                        a.x * b.y - b.x * a.y
                    })
                    .sum::<f64>()
                    .abs()
                    * 0.5
            })
            .sum()
    };
    let mut rois: Vec<&rust_dicom_station::rtstruct::Roi> = a.structure_sets[0]
        .rois
        .iter()
        .filter(|r| !r.is_poi() && r.contours.len() >= 10)
        .collect();
    rois.sort_by(|x, y| area(y).total_cmp(&area(x)));
    let names: Vec<String> = rois.iter().take(3).map(|r| r.name.clone()).collect();
    assert_eq!(names.len(), 3, "three structures of ten slices or more");
    let pair = [a, b];
    let req = PostureRequest {
        anchor: Anchor::Scanner,
        bodies: Bodies::Named(names),
        analysis_mm: 3.0,
        lattice_mm: 8.0,
        ..PostureRequest::default()
    };
    let o = build(&states(&pair), &req, &Progress::default()).unwrap();
    for b in &o.tables.bodies {
        assert!(
            (b.screw.angle_deg - 3.0).abs() < 0.7,
            "{}: {} deg",
            b.name,
            b.screw.angle_deg
        );
    }
}

/// *Save GIF* of the posture window, headless (`posture::film`): the
/// phantom's path as an animated GIF, decoded back - one frame per time of
/// the run (there and back too), the size the layout gives (the three
/// planes side by side, or one alone), the delay asked for, structure
/// outlines in colour over the grey, the frames moving with the hinge, and
/// the same file again when written again.
#[test]
fn the_path_is_written_as_a_gif_that_decodes_back() {
    use image::codecs::gif::GifDecoder;
    use image::{AnimationDecoder, ImageDecoder};
    use rust_dicom_station::posture::film::{self, FilmParams, FilmViews};
    use rust_dicom_station::volume::ViewPlane;

    let pair = gen_test_data::articulated_pair(&ArticulatedParams::default());
    let o = run(&pair, Mode::Diffeomorphic);
    let imgs: Vec<Arc<_>> = pair.iter().map(|s| s.volume.clone()).collect();
    let dir = common::target_dir("test-posture-gif");
    let grid = &o.model.grid;
    for (views, bounce, frames, width) in [
        (FilmViews::All, false, 5, 300),
        (FilmViews::One(ViewPlane::Sagittal), true, 4, 160),
    ] {
        let mut fp = FilmParams::centred(grid, apply::Show::CrossFade, pair[0].default_window);
        fp.views = views;
        fp.bounce = bounce;
        fp.frames = frames;
        fp.width = width;
        fp.fps = 10.0;
        let path = dir.join(format!("film_{}.gif", views.planes().len()));
        let write = || {
            film::write_gif(
                &path,
                &o.model,
                &imgs,
                None,
                pair[0].structure_sets.first(),
                None,
                &fp,
                &Progress::default(),
            )
            .expect("the GIF is written")
        };
        let s = write();
        let times = film::film_times(o.model.span(), frames, bounce).len();
        assert_eq!(s.frames, times);
        assert_eq!([s.width, s.height], film::film_size(grid, &fp, false));
        assert!(!s.scene, "no scene was given");
        assert_eq!(s.width, width, "the width asked for");
        assert!(!s.gpu, "no slicer was given");

        let file = std::io::BufReader::new(std::fs::File::open(&path).unwrap());
        let dec = GifDecoder::new(file).expect("a GIF");
        assert_eq!(dec.dimensions(), (s.width as u32, s.height as u32));
        let decoded = dec.into_frames().collect_frames().expect("its frames");
        assert_eq!(decoded.len(), times, "{views:?}");
        for f in &decoded {
            assert_eq!(f.buffer().dimensions(), (s.width as u32, s.height as u32));
            let (num, den) = f.delay().numer_denom_ms();
            assert_eq!(num / den.max(1), 100, "10 frames a second");
        }
        let first = decoded[0].buffer();
        let middle = decoded[frames / 2].buffer();
        assert_ne!(first.as_raw(), middle.as_raw(), "the hinge turns");
        assert!(
            first.pixels().any(|p| p[0] != p[1] || p[1] != p[2]),
            "no structure outline in colour"
        );
        assert!(
            first
                .pixels()
                .filter(|p| p[0] == p[1] && p[1] == p[2])
                .count()
                > (s.width * s.height) / 2,
            "the image is grey"
        );
        let bytes = std::fs::read(&path).unwrap();
        write();
        assert_eq!(bytes, std::fs::read(&path).unwrap(), "made twice, the same");
    }
}

/// *Save GIF* with the rest of the window (`posture::film`): all views with
/// the 3D scene under the planes and the displacement arrows over them;
/// the sagittal plane alone with the deformed grid; the 3D scene alone.
/// Each has one frame per time and the size its layout gives; the arrows
/// are drawn in colour where the tissue has moved and nowhere at the first
/// state, where nothing has; the scene shows the surfaces and moves with
/// the hinge; the same film twice is the same file.
#[test]
fn the_gif_carries_the_scene_the_field_and_any_one_view() {
    use image::codecs::gif::GifDecoder;
    use image::{AnimationDecoder, RgbaImage};
    use rust_dicom_station::posture::film::{self, FieldOverlay, FilmParams, FilmScene, FilmViews};
    use rust_dicom_station::registration::dvf::FieldStyle;
    use rust_dicom_station::volume::ViewPlane;

    let pair = gen_test_data::articulated_pair(&ArticulatedParams::default());
    let o = run(&pair, Mode::Diffeomorphic);
    let imgs: Vec<Arc<_>> = pair.iter().map(|s| s.volume.clone()).collect();
    let dir = common::target_dir("test-posture-gif-more");
    let grid = &o.model.grid;
    let set = pair[0]
        .structure_sets
        .first()
        .expect("the phantom's structures");
    let meshes = rust_dicom_station::mesh3d::build_meshes(set, &Progress::default());
    assert!(!meshes.is_empty(), "the phantom's structures have surfaces");
    let scene = FilmScene::new(Arc::new(meshes));
    let decode = |path: &std::path::Path| -> Vec<RgbaImage> {
        let file = std::io::BufReader::new(std::fs::File::open(path).unwrap());
        GifDecoder::new(file)
            .expect("a GIF")
            .into_frames()
            .collect_frames()
            .expect("its frames")
            .into_iter()
            .map(|f| f.into_buffer())
            .collect()
    };
    let write = |name: &str, fp: &FilmParams, with_scene: bool| {
        let path = dir.join(name);
        let s = film::write_gif(
            &path,
            &o.model,
            &imgs,
            None,
            Some(set),
            with_scene.then_some(&scene),
            fp,
            &Progress::default(),
        )
        .expect("the GIF is written");
        (path, s)
    };
    let colored = |p: &image::Rgba<u8>| p[0] != p[1] || p[1] != p[2];

    // All views, the scene under the planes, the arrows over them.
    let mut fp = FilmParams::centred(grid, apply::Show::StartForward, pair[0].default_window);
    fp.frames = 4;
    fp.width = 300;
    fp.cell_aspect = Some(1.2);
    fp.scene_aspect = Some(3.0);
    fp.field = Some(FieldOverlay {
        spacing_mm: 6.0,
        ..FieldOverlay::default()
    });
    let (path, s) = write("all_arrows.gif", &fp, true);
    assert!(s.scene, "the scene is in the film");
    assert_eq!(s.frames, 4);
    assert_eq!([s.width, s.height], film::film_size(grid, &fp, true));
    let planes_h = film::film_size(grid, &fp, false)[1];
    assert_eq!(
        s.height,
        planes_h + 2 + 100,
        "the scene's row: 300 / 3.0 under a gap"
    );
    let with = decode(&path);
    assert_eq!(with.len(), 4);
    assert!(with
        .iter()
        .all(|f| f.dimensions() == (s.width as u32, s.height as u32)));
    // The scene's row has the surfaces, in colour, and they move.
    let scene_px = |f: &RgbaImage| -> Vec<[u8; 4]> {
        (planes_h as u32 + 2..f.height())
            .flat_map(|y| (0..f.width()).map(move |x| (x, y)))
            .map(|(x, y)| f.get_pixel(x, y).0)
            .collect()
    };
    let lit = scene_px(&with[0])
        .iter()
        .filter(|p| p[..3] != [0, 0, 0])
        .count();
    assert!(lit > 200, "{lit} pixels of surface in the scene");
    assert_ne!(
        scene_px(&with[0]),
        scene_px(&with[3]),
        "the hinge turns in 3D"
    );
    // The arrows: the coloured pixels over the planes that the same film
    // without them does not have (the two palettes hold different greys,
    // so the greys are not compared).
    let mut bare = fp.clone();
    bare.field = None;
    let (bare_path, _) = write("all_bare.gif", &bare, true);
    let without = decode(&bare_path);
    let colored_on_planes = |f: &RgbaImage| {
        f.rows()
            .take(planes_h)
            .flatten()
            .filter(|p| colored(p))
            .count()
    };
    assert_eq!(
        colored_on_planes(&with[0]),
        colored_on_planes(&without[0]),
        "no arrow at the first state: nothing has moved"
    );
    let arrows = colored_on_planes(&with[3]) as i64 - colored_on_planes(&without[3]) as i64;
    assert!(arrows > 50, "{arrows} arrow pixels at the end");
    let bytes = std::fs::read(&path).unwrap();
    write("all_arrows.gif", &fp, true);
    assert_eq!(bytes, std::fs::read(&path).unwrap(), "made twice, the same");

    // The sagittal plane alone, the deformed grid over it: the plane's own
    // shape at the width asked for, and no scene even when one is given.
    let mut fp = FilmParams::centred(grid, apply::Show::CrossFade, pair[0].default_window);
    fp.views = FilmViews::One(ViewPlane::Sagittal);
    fp.frames = 3;
    fp.width = 160;
    fp.field = Some(FieldOverlay {
        style: FieldStyle::Grid,
        ..FieldOverlay::default()
    });
    let (path, s) = write("sagittal_grid.gif", &fp, true);
    assert!(!s.scene);
    let tall = (160.0 * grid.dims[2] as f64 * grid.spacing[2]
        / (grid.dims[1] as f64 * grid.spacing[1]))
        .round() as usize;
    assert_eq!([s.width, s.height], [160, tall.clamp(16, 4096)]);
    assert_eq!([s.width, s.height], film::film_size(grid, &fp, true));
    let frames = decode(&path);
    assert_eq!(frames.len(), 3);
    let mut bare = fp.clone();
    bare.field = None;
    let (bare_path, _) = write("sagittal_bare.gif", &bare, true);
    let count = |f: &RgbaImage| f.pixels().filter(|p| colored(p)).count();
    assert!(
        count(&frames[2]) > count(&decode(&bare_path)[2]),
        "the grid's lines in colour"
    );

    // The 3D scene alone.
    let mut fp = FilmParams::centred(grid, apply::Show::StartForward, pair[0].default_window);
    fp.views = FilmViews::Scene;
    fp.frames = 3;
    fp.bounce = true;
    fp.width = 200;
    fp.scene_aspect = Some(2.0);
    let (path, s) = write("scene.gif", &fp, true);
    assert!(s.scene);
    assert_eq!([s.width, s.height], [200, 100]);
    let frames = decode(&path);
    assert_eq!(frames.len(), 4, "there and back: 0, 1/2, 1, 1/2");
    assert_ne!(frames[0].as_raw(), frames[2].as_raw(), "the hinge turns");
    assert_eq!(
        frames[1].as_raw(),
        frames[3].as_raw(),
        "half way, both ways"
    );
    // Without a scene there is nothing to film.
    let none = film::write_gif(
        &dir.join("no_scene.gif"),
        &o.model,
        &imgs,
        None,
        Some(set),
        None,
        &fp,
        &Progress::default(),
    );
    assert!(none.is_err(), "a 3D film with no surfaces");
    assert!(!dir.join("no_scene.gif").exists(), "and no file left");
}

/// *Auto* on the phantom: no sacrum and no vertebra, so one of the two
/// bones every scan sees whole is held still; whichever it is, the joint
/// between the two reads the hinge's angle, and the verdict is not poor.
#[test]
fn the_auto_anchor_holds_a_bone_both_scans_see_whole() {
    let params = ArticulatedParams::default();
    let pair = gen_test_data::articulated_pair(&params);
    let mut req = request(Mode::Diffeomorphic);
    req.anchor = Anchor::Auto;
    let o = build(&states(&pair), &req, &Progress::default()).expect("the model builds");
    match &o.model.anchor {
        Anchor::Structure(s) => assert!(s.starts_with("BONE_"), "anchor {s}"),
        a => panic!("Auto resolved to {a:?}"),
    }
    assert!(
        o.model
            .provenance
            .iter()
            .any(|l| l.starts_with("anchor chosen")),
        "{:?}",
        o.model.provenance
    );
    let joint = o.tables.joints.first().expect("the two bones make a joint");
    let end = joint.at.last().unwrap();
    assert!(
        (end[0].abs() - params.angle_deg).abs() < 2.0,
        "joint {end:?}"
    );
    let v = o.tables.verdict();
    assert_ne!(v.level, posture::measure::Level::Poor, "{v:?}");
}

/// The residual on top of the bones keeps the path a deformation: nothing
/// folds, the way back still finds the way there, the bones stay rigid, and
/// the organ that grows lands on its contour no worse than without it.
#[test]
fn the_residual_does_not_fold_and_keeps_the_bones() {
    let params = ArticulatedParams::default();
    let pair = gen_test_data::articulated_pair(&params);
    let plain = run(&pair, Mode::Diffeomorphic);
    let mut req = request(Mode::Diffeomorphic);
    req.residual = Some(rust_dicom_station::workflow::posture::default_residual());
    let o = build(&states(&pair), &req, &Progress::default()).expect("the model builds");
    let m = &o.tables.model;
    assert!(m.jacobian_min > 0.0, "Jacobian {}", m.jacobian_min);
    assert_eq!(m.folded_fraction, 0.0);
    assert!(m.inverse_mm[1][1] < 1.0, "inverse {:?}", m.inverse_mm);
    assert!(
        m.provenance
            .iter()
            .any(|l| l.contains("within the part both scans cover")),
        "{:?}",
        m.provenance
    );
    let lower = body(&o, "BONE_LOWER");
    assert!(lower.stretch_p95 < 0.05, "stretch {}", lower.stretch_p95);
    let organ = |o: &PostureOutcome| {
        o.tables
            .structures
            .iter()
            .find(|s| s.name == "ORGAN")
            .and_then(|s| s.check)
            .map(|c| c.dice)
            .expect("the organ is checked")
    };
    assert!(
        organ(&o) > organ(&plain) - 0.02,
        "organ {} with the residual, {} without",
        organ(&o),
        organ(&plain)
    );
}

/// *Patient only*: outside each state's EXTERNAL structure is air in the
/// images the model draws, inside it the scan's own values; without an
/// EXTERNAL structure the scan is kept as it is. The model's images are
/// these.
#[test]
fn patient_only_clears_what_lies_outside_the_body() {
    use rust_dicom_station::workflow::posture::{patient_only, state_images};
    let mut pair = gen_test_data::articulated_pair(&ArticulatedParams::default());
    // Something that is not the patient: a couch under it.
    let v = Arc::make_mut(&mut pair[0].volume);
    let [nx, ny, nz] = v.dims;
    for k in 0..nz {
        for i in 0..nx {
            v.data[(k * ny + (ny - 2)) * nx + i] = 300;
        }
    }
    let only = patient_only(&pair[0], None).expect("the phantom has a BODY");
    let couch = (0..nx).map(|i| only.data[((nz / 2) * ny + (ny - 2)) * nx + i]);
    assert!(couch.into_iter().all(|x| x <= -1000), "the couch is left");
    let centre = ((nz / 2) * ny + ny / 2) * nx + nx / 2;
    assert_eq!(only.data[centre], pair[0].volume.data[centre]);
    let images = state_images(&states(&pair), true);
    assert!(!Arc::ptr_eq(&images[0], &pair[0].volume));
    let o = run(&pair, Mode::Diffeomorphic);
    assert_eq!(o.images.len(), 2);
    assert!(
        o.model
            .provenance
            .iter()
            .any(|l| l.starts_with("patient only")),
        "{:?}",
        o.model.provenance
    );
    // Without a structure set there is nothing to cut by.
    let mut bare = pair[1].clone();
    bare.structure_sets.clear();
    assert!(patient_only(&bare, None).is_none());
}

/// Another image of the first position carried along the path: laid in
/// the scanner's frame, the first state's own image comes back as it is
/// on its grid, and rendered forward to the end it is the model's own
/// forward image there.
#[test]
fn an_image_of_the_first_position_is_carried_like_its_own() {
    use rust_dicom_station::workflow::posture::{carry_image, CarryAlign};
    let pair = gen_test_data::articulated_pair(&ArticulatedParams::default());
    let o = run(&pair, Mode::Diffeomorphic);
    let (v, line) = carry_image(
        &o.images[0],
        &o.images[0],
        &CarryAlign::Scanner,
        2.0,
        &Progress::default(),
    )
    .expect("carried");
    assert!(line.contains("scanner"), "{line}");
    assert_eq!(v.data, o.images[0].data);
    let mut images = o.images.clone();
    images[0] = Arc::new(v);
    let frame = o.model.frame(1.0);
    let a = apply::render(
        &o.model,
        &images,
        &frame,
        apply::Show::StartForward,
        &o.model.grid,
    )
    .unwrap();
    let b = apply::render(
        &o.model,
        &o.images,
        &frame,
        apply::Show::StartForward,
        &o.model.grid,
    )
    .unwrap();
    assert_eq!(a.data, b.data);
    // Two frames of reference cannot be laid in the scanner's.
    let mut other = (*o.images[0]).clone();
    other.frame_of_reference_uid = "1.2.3".into();
    assert!(carry_image(
        &o.images[0],
        &other,
        &CarryAlign::Scanner,
        2.0,
        &Progress::default()
    )
    .is_err());
}
