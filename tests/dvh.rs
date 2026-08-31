//! Dose–volume histograms against a study that came out of a DICOM file.
//!
//! The unit tests in `src/dvh.rs` check the arithmetic on hand-built grids.
//! What this adds is the whole path: the synthetic RT study is *written* as
//! DICOM, read back through the normal loader, its RTSTRUCT contours are
//! rasterized onto the CT lattice, and the histogram is taken against the
//! RTDOSE as parsed — including its own grid geometry, frame offsets and
//! dose-grid scaling.
//!
//! The phantom is what makes this worth doing. Its dose is an analytic
//! Gaussian, `D(r) = peak · exp(−r² / 2σ²)` with σ = 20 mm, centred on the
//! spherical 25 mm target. So the target's DVH is known in closed form: the
//! volume receiving at least `D` is the ball of radius
//! `r = σ·√(2·ln(peak/D))`, and the fraction of the target inside it is
//! `(r/R)³`. The test compares against that, not against a previous run.

use std::path::PathBuf;
use std::sync::OnceLock;

use rust_dicom_station::dvh::{self, Constraint, DvhParams, Metric};
use rust_dicom_station::gen_test_data::{self, GenParams};
use rust_dicom_station::loader::{self, LoadedStudy};
use rust_dicom_station::progress::Progress;
use rust_dicom_station::segmentation;

const PEAK: f64 = 60.0;
const SIGMA: f64 = 20.0;
const TARGET_R: f64 = 25.0;

fn study() -> &'static LoadedStudy {
    static STUDY: OnceLock<LoadedStudy> = OnceLock::new();
    STUDY.get_or_init(|| {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/test_data_dvh");
        let _ = std::fs::remove_dir_all(&dir);
        gen_test_data::generate(&dir, &GenParams::default(), &Progress::default())
            .expect("the synthetic study is generated");
        loader::load_directory(&dir, &Progress::default()).expect("and loads")
    })
}

/// One named ROI's DVH against the study's dose.
fn curve_for(name: &str) -> dvh::Dvh {
    let st = study();
    let grid = st.volume.grid();
    let roi = st.structure_sets[0]
        .rois
        .iter()
        .find(|r| r.name == name)
        .unwrap_or_else(|| panic!("the phantom has a {name}"));
    let mask = segmentation::rasterize_roi(&grid, roi).expect("it rasterizes");
    dvh::compute(
        &roi.name,
        roi.color,
        &mask,
        &grid,
        &st.doses[0],
        DvhParams::default(),
    )
    .expect("a curve")
}

/// The fraction of a ball of radius `R` centred on the dose peak that
/// receives at least `d` — the closed form the phantom was built to have.
fn analytic_fraction(d: f64, radius: f64) -> f64 {
    if d >= PEAK {
        return 0.0;
    }
    if d <= 0.0 {
        return 1.0;
    }
    let r = SIGMA * (2.0 * (PEAK / d).ln()).sqrt();
    (r / radius).min(1.0).powi(3)
}

#[test]
fn the_phantom_loads_with_a_dose_and_three_structures() {
    let st = study();
    assert_eq!(st.doses.len(), 1, "one RTDOSE");
    let names: Vec<&str> = st.structure_sets[0]
        .rois
        .iter()
        .map(|r| r.name.as_str())
        .collect();
    assert!(names.contains(&"BODY") && names.contains(&"TARGET") && names.contains(&"CORD"));
    assert_eq!(st.doses[0].units.to_uppercase(), "GY");
    assert!(
        (st.doses[0].max_dose as f64 - PEAK).abs() < 0.5,
        "peak {} should be about {PEAK}",
        st.doses[0].max_dose
    );
    assert_eq!(
        st.plans[0].target_prescription_dose,
        Some(PEAK),
        "the plan carries the prescription the percentage axis needs"
    );
}

#[test]
fn the_target_dvh_matches_the_analytic_gaussian() {
    let c = curve_for("TARGET");
    // A 25 mm ball is 65.4 cm³; the contours are polygonal and the lattice
    // is 2 mm, so allow a few per cent.
    let want_volume = 4.0 / 3.0 * std::f64::consts::PI * TARGET_R.powi(3) / 1000.0;
    assert!(
        (c.volume_cm3 - want_volume).abs() / want_volume < 0.08,
        "volume {:.2} cm³, expected about {want_volume:.2}",
        c.volume_cm3
    );
    assert!(c.outside_fraction() < 1e-6, "the target is inside the dose");
    // Close to the peak, but not equal to it, and it should not be: the
    // hottest *CT* voxel centre is up to a millimetre or two off the origin
    // where the Gaussian peaks, and the dose it reads is interpolated
    // between 4 mm dose nodes across a curve that is concave down. Both
    // effects lose about a per cent, and both are correct.
    assert!(
        (PEAK - c.max) > 0.0 && (PEAK - c.max) < 1.5,
        "the hottest voxel {:.2} should approach the {PEAK} Gy peak from below",
        c.max
    );

    // The DVH itself, against the closed form.
    for d in [10.0, 20.0, 30.0, 40.0, 50.0, 55.0] {
        let got = c.volume_fraction_at_dose(d);
        let want = analytic_fraction(d, TARGET_R);
        assert!(
            (got - want).abs() < 0.05,
            "V{d}Gy = {got:.3}, analytic {want:.3}"
        );
    }
    // …and read the other way round.
    for frac in [0.95, 0.5, 0.05] {
        let got = c.dose_at_volume_fraction(frac);
        // Invert: the dose whose ball holds `frac` of the target.
        let r = TARGET_R * frac.cbrt();
        let want = PEAK * (-(r * r) / (2.0 * SIGMA * SIGMA)).exp();
        assert!(
            (got - want).abs() < 2.0,
            "D{}% = {got:.2} Gy, analytic {want:.2} Gy",
            frac * 100.0
        );
    }
}

#[test]
fn a_cumulative_dvh_never_rises() {
    for name in ["BODY", "TARGET", "CORD"] {
        let c = curve_for(name);
        let mut last = f64::INFINITY;
        for (dose, vol) in c.cumulative() {
            assert!(
                vol <= last + 1e-9,
                "{name} rises at {dose:.2} Gy: {vol} after {last}"
            );
            last = vol;
        }
        // The curve starts at the whole structure and ends at nothing.
        assert!((c.cumulative()[0].1 - c.volume_cm3).abs() < 1e-9);
        assert!(c.cumulative().last().unwrap().1 < 1e-9);
        // Statistics bracket each other.
        assert!(c.min <= c.mean + 1e-9 && c.mean <= c.max + 1e-9, "{name}");
    }
}

#[test]
fn a_structure_inside_another_is_nowhere_hotter_in_absolute_volume() {
    // TARGET ⊂ BODY, so at every dose the body has at least as much volume
    // above it. A DVH implementation that mixed up its lattices would fail
    // this long before anyone noticed the numbers were wrong.
    let body = curve_for("BODY");
    let target = curve_for("TARGET");
    assert!(body.volume_cm3 > target.volume_cm3 * 5.0);
    for d in [0.0, 5.0, 15.0, 30.0, 45.0, 58.0] {
        assert!(
            body.volume_at_dose(d) >= target.volume_at_dose(d) - 1e-6,
            "at {d} Gy body {:.3} < target {:.3}",
            body.volume_at_dose(d),
            target.volume_at_dose(d)
        );
    }
    // The target is the hot structure: most of it is above half the peak,
    // while most of the body is not.
    assert!(target.volume_fraction_at_dose(PEAK / 2.0) > 0.6);
    assert!(body.volume_fraction_at_dose(PEAK / 2.0) < 0.2);
}

#[test]
fn the_metrics_table_and_its_csv_agree_with_the_curve() {
    let c = curve_for("TARGET");
    let metrics = vec![
        Metric::Volume,
        Metric::Mean,
        Metric::Max,
        Metric::DoseAtPct(95.0),
        Metric::VolumePctAtDose(30.0),
    ];
    assert!((Metric::Volume.evaluate(&c) - c.volume_cm3).abs() < 1e-9);
    assert!((Metric::Max.evaluate(&c) - c.max).abs() < 1e-9);
    assert!(
        (Metric::VolumePctAtDose(30.0).evaluate(&c) - c.volume_fraction_at_dose(30.0) * 100.0)
            .abs()
            < 1e-9
    );

    let csv = dvh::metrics_csv(std::slice::from_ref(&c), &metrics);
    let mut lines = csv.lines();
    let header = lines.next().expect("a header");
    assert!(
        header.starts_with("Structure,Dose,Volume [cm³]"),
        "{header}"
    );
    assert!(header.contains("D95% [Gy]"), "{header}");
    let row = lines.next().expect("one row");
    assert!(row.starts_with("TARGET,"), "{row}");
    // The last column is V30Gy in per cent, and it must match the curve.
    let last: f64 = row.rsplit(',').next().unwrap().parse().expect("a number");
    assert!((last - c.volume_fraction_at_dose(30.0) * 100.0).abs() < 1e-3);

    let curves = dvh::curves_csv(std::slice::from_ref(&c), true);
    let first = curves.lines().nth(1).expect("a first data row");
    assert!(first.starts_with("0.0000,100.0000"), "{first}");
}

#[test]
fn a_protocol_reads_the_phantom_the_way_a_physicist_would() {
    let curves: Vec<dvh::Dvh> = ["TARGET", "CORD", "BODY"]
        .map(curve_for)
        .into_iter()
        .collect();
    let protocol = "\
# the phantom, as it happens to be
TARGET  D95%  >= 20
TARGET  Dmax  <= 61
CORD    Dmax  <= 1
Ghost   Dmean <= 10
";
    let cs = dvh::parse_protocol(protocol);
    assert_eq!(cs.len(), 4);
    let v = dvh::check(&cs, &curves);
    assert!(v[0].pass, "D95% of the target clears 20 Gy");
    assert!(v[1].pass, "and its maximum is the 60 Gy peak");
    assert!(
        !v[2].pass,
        "the cord sits in the penumbra, so 1 Gy is not met — {:?}",
        v[2].value
    );
    assert!(
        v[3].value.is_none() && !v[3].pass,
        "a missing structure fails"
    );

    // A wildcard finds the target by prefix, which is how real protocols
    // are written against PTV_5400 and friends.
    let wild = Constraint {
        structure: "TARG*".into(),
        metric: Metric::Mean,
        cmp: dvh::Cmp::AtLeast,
        limit: 1.0,
    };
    assert_eq!(dvh::check(&[wild], &curves)[0].structure, "TARGET");
}
