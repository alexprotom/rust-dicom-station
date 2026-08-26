//! Synthetic DICOM RT study generator.
//!
//! Writes a complete, self-consistent test study into a directory:
//!
//! * **CT series** — 40 slices, 96 × 96 px, 2 mm isotropic. Water cylinder
//!   phantom (r = 70 mm), spherical target (r = 25 mm, HU 100) at the origin
//!   and a small "cord" cylinder (r = 8 mm, HU 40) at (0, 60).
//! * **RTSTRUCT** — BODY (EXTERNAL), TARGET (PTV), CORD (ORGAN).
//! * **RTDOSE** — 3D Gaussian, 60 Gy at the isocenter, σ = 20 mm, 32-bit,
//!   4 mm in-plane grid with 2 mm frame steps.
//! * **RTPLAN** — ion (proton) plan, 2 beams, 60 Gy / 30 fx prescription.
//! * **Extras** (optional) — DX radiograph, RTIMAGE (DRR), REG spatial
//!   registration and an RT Ion Beams Treatment Record.
//!
//! The geometry is exact and analytically known, which makes the study usable
//! as ground truth for the viewer's registration, dose and contour code.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use dicom_core::value::{PrimitiveValue, C};
use dicom_core::{DataElement, VR};
use dicom_dictionary_std::tags;
use dicom_object::InMemDicomObject;

use crate::dicom_export::{
    fmt_ds, new_uid, put_ds, put_is, put_seq, put_str, put_strs, put_us, today, write_object,
};
use crate::progress::Progress;
use crate::settings;

// ---------------------------------------------------------------------------
// SOP Class UIDs
// ---------------------------------------------------------------------------

const SOP_CT: &str = "1.2.840.10008.5.1.4.1.1.2";
const SOP_RTSTRUCT: &str = "1.2.840.10008.5.1.4.1.1.481.3";
const SOP_RTDOSE: &str = "1.2.840.10008.5.1.4.1.1.481.2";
const SOP_RTIONPLAN: &str = "1.2.840.10008.5.1.4.1.1.481.8";
const SOP_RTIMAGE: &str = "1.2.840.10008.5.1.4.1.1.481.1";
const SOP_DX: &str = "1.2.840.10008.5.1.4.1.1.1.1";
const SOP_SPATIAL_REG: &str = "1.2.840.10008.5.1.4.1.1.66.1";
const SOP_RT_ION_BEAMS_RECORD: &str = "1.2.840.10008.5.1.4.1.1.481.9";
/// Detached Study Management (referenced from RTSTRUCT).
const SOP_DETACHED_STUDY: &str = "1.2.840.10008.3.1.2.3.1";

// ---------------------------------------------------------------------------
// Phantom geometry (fixed — the integration tests depend on these values)
// ---------------------------------------------------------------------------

/// CT columns / rows.
const NX: usize = 96;
const NY: usize = 96;
/// CT slices.
const NZ: usize = 40;
/// CT voxel size, isotropic (mm).
const SPACING: f64 = 2.0;

/// Body cylinder radius (mm).
const R_BODY: f64 = 70.0;
/// Target sphere radius (mm).
const R_TARGET: f64 = 25.0;
/// Cord cylinder radius (mm) and its Y offset.
const R_CORD: f64 = 8.0;
const CORD_Y: f64 = 60.0;

/// Dose grid: odd counts so that (0, 0, 0) — the peak — is exactly on a node.
const DNX: usize = 47;
const DNY: usize = 47;
const DNZ: usize = 41;
/// Dose in-plane spacing (mm); frame steps use the CT `SPACING`.
const DSP: f64 = 4.0;
/// Gaussian dose sigma (mm).
const SIGMA: f64 = 20.0;
/// Stored-value scaling of the dose grid (Gy per stored unit).
const DOSE_SCALING: f64 = 1.0e-3;

/// Planar (DX / RTIMAGE) detector size in pixels.
const PX_COLS: usize = 512;
const PX_ROWS: usize = 400;

const PATIENT_NAME: &str = "PHANTOM^RT";
const PATIENT_ID: &str = "RTTEST001";
const MACHINE: &str = "SYNTH-PBS";

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

/// Generator options. The defaults reproduce the canonical test study that
/// `tests/synthetic_study.rs` asserts against.
#[derive(Clone, Debug, PartialEq)]
pub struct GenParams {
    /// Target / dose Y shift in mm (moves the target relative to the body).
    pub target_shift_y: f64,
    /// Whole-phantom X shift in mm (for registration tests).
    pub shift_x: f64,
    /// Whole-phantom Y shift in mm (for registration tests).
    pub shift_y: f64,
    /// Dose peak in Gy (also the plan's prescription dose).
    pub peak: f64,
    /// RT Plan Label (SH, truncated to 16 characters on write).
    pub plan_label: String,
    /// Translation (mm) written into the REG object's second matrix.
    pub reg_shift: [f64; 3],
    /// Write the DX / RTIMAGE / REG / RTRECORD extras.
    pub extras: bool,
}

impl Default for GenParams {
    fn default() -> Self {
        GenParams {
            target_shift_y: 0.0,
            shift_x: 0.0,
            shift_y: 0.0,
            peak: 60.0,
            plan_label: "SynthProton".into(),
            reg_shift: [12.0, -9.0, 0.0],
            extras: true,
        }
    }
}

pub fn default_output_dir() -> PathBuf {
    settings::data_dir().join("test_data")
}

/// Names of the files this generator writes (for UI hints and cleanup).
pub fn output_summary(p: &GenParams) -> String {
    if p.extras {
        format!(
            "CT_000…CT_{:03}.dcm, RS/RP/RD_synth.dcm, DX/RI/REG/RT_record_synth.dcm",
            NZ - 1
        )
    } else {
        format!(
            "CT_000…CT_{:03}.dcm, RS_synth.dcm, RP_synth.dcm, RD_synth.dcm",
            NZ - 1
        )
    }
}

// ---------------------------------------------------------------------------
// Shared per-run identifiers
// ---------------------------------------------------------------------------

struct Ids {
    study_uid: String,
    for_uid: String,
    ct_series_uid: String,
    ct_sop_uids: Vec<String>,
    struct_uid: String,
    plan_uid: String,
    date: String,
    time: String,
}

fn base_dataset(ids: &Ids, sop_class: &str, sop_uid: &str, modality: &str) -> InMemDicomObject {
    let mut o = InMemDicomObject::new_empty();
    put_str(&mut o, tags::SPECIFIC_CHARACTER_SET, VR::CS, "ISO_IR 100");
    put_str(&mut o, tags::SOP_CLASS_UID, VR::UI, sop_class);
    put_str(&mut o, tags::SOP_INSTANCE_UID, VR::UI, sop_uid);
    put_str(&mut o, tags::PATIENT_NAME, VR::PN, PATIENT_NAME);
    put_str(&mut o, tags::PATIENT_ID, VR::LO, PATIENT_ID);
    put_str(&mut o, tags::PATIENT_BIRTH_DATE, VR::DA, "19700101");
    put_str(&mut o, tags::PATIENT_SEX, VR::CS, "O");
    put_str(
        &mut o,
        tags::STUDY_INSTANCE_UID,
        VR::UI,
        ids.study_uid.clone(),
    );
    put_str(&mut o, tags::STUDY_DATE, VR::DA, ids.date.clone());
    put_str(&mut o, tags::STUDY_TIME, VR::TM, ids.time.clone());
    put_str(
        &mut o,
        tags::STUDY_DESCRIPTION,
        VR::LO,
        "Synthetic RT study",
    );
    put_str(&mut o, tags::ACCESSION_NUMBER, VR::SH, "1");
    put_str(&mut o, tags::REFERRING_PHYSICIAN_NAME, VR::PN, "");
    put_str(&mut o, tags::MODALITY, VR::CS, modality);
    put_str(
        &mut o,
        tags::MANUFACTURER,
        VR::LO,
        "rust-dicom-station synthetic",
    );
    o
}

/// Reference to one SOP instance (`ReferencedSOPClassUID` + `…InstanceUID`).
fn ref_item(sop_class: &str, sop_uid: &str) -> InMemDicomObject {
    let mut it = InMemDicomObject::new_empty();
    put_str(&mut it, tags::REFERENCED_SOP_CLASS_UID, VR::UI, sop_class);
    put_str(&mut it, tags::REFERENCED_SOP_INSTANCE_UID, VR::UI, sop_uid);
    it
}

/// Store a 16-bit pixel buffer as native little-endian `PixelData` (OW).
fn put_pixels_u16(o: &mut InMemDicomObject, words: Vec<u16>) {
    o.put(DataElement::new(
        tags::PIXEL_DATA,
        VR::OW,
        PrimitiveValue::U16(C::from_vec(words)),
    ));
}

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

/// Generate the synthetic study into `dir`, creating it if necessary.
/// Returns the number of DICOM files written.
pub fn generate(dir: &Path, params: &GenParams, progress: &Progress) -> Result<usize> {
    std::fs::create_dir_all(dir).with_context(|| format!("create directory {}", dir.display()))?;
    let (date, time) = today();
    let mut ids = Ids {
        study_uid: new_uid(),
        for_uid: new_uid(),
        ct_series_uid: new_uid(),
        ct_sop_uids: Vec::with_capacity(NZ),
        struct_uid: new_uid(),
        plan_uid: new_uid(),
        date,
        time,
    };

    let mut n_files = 0usize;
    n_files += write_ct(dir, params, &mut ids, progress)?;
    n_files += write_rtstruct(dir, params, &ids, progress)?;
    n_files += write_rtplan(dir, params, &ids, progress)?;
    n_files += write_rtdose(dir, params, &ids, progress)?;
    if params.extras {
        n_files += write_extras(dir, params, &ids, progress)?;
    }

    progress.set("done");
    Ok(n_files)
}

// -- CT ---------------------------------------------------------------------

/// Grid origin (center of voxel 0) along one in-plane axis.
fn axis_origin(n: usize, spacing: f64) -> f64 {
    -((n as f64 - 1.0) / 2.0) * spacing
}

fn write_ct(dir: &Path, p: &GenParams, ids: &mut Ids, progress: &Progress) -> Result<usize> {
    let x0 = axis_origin(NX, SPACING);
    let y0 = axis_origin(NY, SPACING);
    let z0 = axis_origin(NZ, SPACING);
    let (sx, sy) = (p.shift_x, p.shift_y);

    let mut hu = vec![0i16; NX * NY];
    for k in 0..NZ {
        if k % 8 == 0 {
            progress.set(format!("Writing CT slice {}/{}…", k + 1, NZ));
        }
        let z = z0 + k as f64 * SPACING;

        for j in 0..NY {
            let y = y0 + j as f64 * SPACING;
            for i in 0..NX {
                let x = x0 + i as f64 * SPACING;
                let dx = x - sx;
                // Air by default; then body, target and cord in that order so
                // later (denser) structures win where they overlap.
                let mut v = -1000.0f64;
                let dy_body = y - sy;
                if dx * dx + dy_body * dy_body <= R_BODY * R_BODY {
                    v = 0.0;
                }
                let dy_t = y - p.target_shift_y - sy;
                if dx * dx + dy_t * dy_t + z * z <= R_TARGET * R_TARGET {
                    v = 100.0;
                }
                let dy_c = y - CORD_Y - sy;
                if dx * dx + dy_c * dy_c <= R_CORD * R_CORD {
                    v = 40.0;
                }
                // Stored value = HU − RescaleIntercept(−1024).
                hu[j * NX + i] = (v + 1024.0).round() as i16;
            }
        }

        let sop_uid = new_uid();
        ids.ct_sop_uids.push(sop_uid.clone());
        let mut o = base_dataset(ids, SOP_CT, &sop_uid, "CT");
        put_str(
            &mut o,
            tags::SERIES_INSTANCE_UID,
            VR::UI,
            ids.ct_series_uid.clone(),
        );
        put_is(&mut o, tags::SERIES_NUMBER, 1);
        put_str(&mut o, tags::SERIES_DESCRIPTION, VR::LO, "Synthetic CT");
        put_is(&mut o, tags::INSTANCE_NUMBER, k as i64 + 1);
        put_str(
            &mut o,
            tags::FRAME_OF_REFERENCE_UID,
            VR::UI,
            ids.for_uid.clone(),
        );
        put_str(&mut o, tags::POSITION_REFERENCE_INDICATOR, VR::LO, "");
        put_ds(&mut o, tags::IMAGE_POSITION_PATIENT, &[x0, y0, z]);
        put_ds(
            &mut o,
            tags::IMAGE_ORIENTATION_PATIENT,
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        );
        put_ds(&mut o, tags::PIXEL_SPACING, &[SPACING, SPACING]);
        put_ds(&mut o, tags::SLICE_THICKNESS, &[SPACING]);
        put_str(&mut o, tags::KVP, VR::DS, "120");
        put_us(&mut o, tags::ROWS, NY as u16);
        put_us(&mut o, tags::COLUMNS, NX as u16);
        put_us(&mut o, tags::BITS_ALLOCATED, 16);
        put_us(&mut o, tags::BITS_STORED, 16);
        put_us(&mut o, tags::HIGH_BIT, 15);
        put_us(&mut o, tags::PIXEL_REPRESENTATION, 1); // signed
        put_us(&mut o, tags::SAMPLES_PER_PIXEL, 1);
        put_str(
            &mut o,
            tags::PHOTOMETRIC_INTERPRETATION,
            VR::CS,
            "MONOCHROME2",
        );
        put_ds(&mut o, tags::RESCALE_INTERCEPT, &[-1024.0]);
        put_ds(&mut o, tags::RESCALE_SLOPE, &[1.0]);
        put_str(&mut o, tags::RESCALE_TYPE, VR::LO, "HU");
        put_ds(&mut o, tags::WINDOW_CENTER, &[40.0]);
        put_ds(&mut o, tags::WINDOW_WIDTH, &[400.0]);
        put_pixels_u16(&mut o, hu.iter().map(|&v| v as u16).collect());

        write_object(o, SOP_CT, &dir.join(format!("CT_{k:03}.dcm")))?;
    }
    Ok(NZ)
}

// -- RTSTRUCT ---------------------------------------------------------------

/// `n` evenly spaced points on a circle (first point at angle 0, not closed).
fn circle_points(cx: f64, cy: f64, r: f64, z: f64, n: usize) -> Vec<String> {
    let mut out = Vec::with_capacity(n * 3);
    for a in 0..n {
        let ang = 2.0 * std::f64::consts::PI * a as f64 / n as f64;
        out.push(fmt_ds(cx + r * ang.cos()));
        out.push(fmt_ds(cy + r * ang.sin()));
        out.push(fmt_ds(z));
    }
    out
}

fn write_rtstruct(dir: &Path, p: &GenParams, ids: &Ids, progress: &Progress) -> Result<usize> {
    progress.set("Writing RTSTRUCT…");
    let z0 = axis_origin(NZ, SPACING);
    let (sx, sy) = (p.shift_x, p.shift_y);

    let mut o = base_dataset(ids, SOP_RTSTRUCT, &ids.struct_uid.clone(), "RTSTRUCT");
    put_str(&mut o, tags::SERIES_INSTANCE_UID, VR::UI, new_uid());
    put_is(&mut o, tags::SERIES_NUMBER, 2);
    put_str(&mut o, tags::STRUCTURE_SET_LABEL, VR::SH, "SynthStructs");
    put_str(&mut o, tags::STRUCTURE_SET_DATE, VR::DA, ids.date.clone());
    put_str(&mut o, tags::STRUCTURE_SET_TIME, VR::TM, ids.time.clone());

    // ReferencedFrameOfReference → RTReferencedStudy → RTReferencedSeries.
    let mut rt_ref_series = InMemDicomObject::new_empty();
    put_str(
        &mut rt_ref_series,
        tags::SERIES_INSTANCE_UID,
        VR::UI,
        ids.ct_series_uid.clone(),
    );
    put_seq(
        &mut rt_ref_series,
        tags::CONTOUR_IMAGE_SEQUENCE,
        ids.ct_sop_uids
            .iter()
            .map(|u| ref_item(SOP_CT, u))
            .collect(),
    );
    let mut rt_ref_study = ref_item(SOP_DETACHED_STUDY, &ids.study_uid);
    put_seq(
        &mut rt_ref_study,
        tags::RT_REFERENCED_SERIES_SEQUENCE,
        vec![rt_ref_series],
    );
    let mut ref_frame = InMemDicomObject::new_empty();
    put_str(
        &mut ref_frame,
        tags::FRAME_OF_REFERENCE_UID,
        VR::UI,
        ids.for_uid.clone(),
    );
    put_seq(
        &mut ref_frame,
        tags::RT_REFERENCED_STUDY_SEQUENCE,
        vec![rt_ref_study],
    );
    put_seq(
        &mut o,
        tags::REFERENCED_FRAME_OF_REFERENCE_SEQUENCE,
        vec![ref_frame],
    );

    let rois: [(i64, &str, &str, [i64; 3]); 3] = [
        (1, "BODY", "EXTERNAL", [0, 255, 0]),
        (2, "TARGET", "PTV", [255, 0, 0]),
        (3, "CORD", "ORGAN", [255, 255, 0]),
    ];

    let mut ssr = Vec::new();
    let mut rcs = Vec::new();
    let mut obs = Vec::new();
    for (num, name, typ, color) in rois {
        let mut s = InMemDicomObject::new_empty();
        put_is(&mut s, tags::ROI_NUMBER, num);
        put_str(
            &mut s,
            tags::REFERENCED_FRAME_OF_REFERENCE_UID,
            VR::UI,
            ids.for_uid.clone(),
        );
        put_str(&mut s, tags::ROI_NAME, VR::LO, name);
        put_str(&mut s, tags::ROI_GENERATION_ALGORITHM, VR::CS, "AUTOMATIC");
        ssr.push(s);

        let mut rc = InMemDicomObject::new_empty();
        put_is(&mut rc, tags::REFERENCED_ROI_NUMBER, num);
        put_strs(
            &mut rc,
            tags::ROI_DISPLAY_COLOR,
            VR::IS,
            &color.map(|c| c.to_string()),
        );

        let mut contours = Vec::new();
        for k in 0..NZ {
            let z = z0 + k as f64 * SPACING;
            // Radius of this ROI on this slice; targets taper off as a sphere.
            let (r, cy) = match name {
                "BODY" => (R_BODY, 0.0),
                "CORD" => (R_CORD, CORD_Y),
                _ => {
                    let r2 = R_TARGET * R_TARGET - z * z;
                    if r2 <= 4.0 {
                        continue;
                    }
                    (r2.sqrt(), p.target_shift_y)
                }
            };
            let pts = circle_points(sx, cy + sy, r, z, 64);
            let mut co = InMemDicomObject::new_empty();
            put_str(
                &mut co,
                tags::CONTOUR_GEOMETRIC_TYPE,
                VR::CS,
                "CLOSED_PLANAR",
            );
            put_is(
                &mut co,
                tags::NUMBER_OF_CONTOUR_POINTS,
                (pts.len() / 3) as i64,
            );
            put_strs(&mut co, tags::CONTOUR_DATA, VR::DS, &pts);
            put_seq(
                &mut co,
                tags::CONTOUR_IMAGE_SEQUENCE,
                vec![ref_item(SOP_CT, &ids.ct_sop_uids[k])],
            );
            contours.push(co);
        }
        put_seq(&mut rc, tags::CONTOUR_SEQUENCE, contours);
        rcs.push(rc);

        let mut ob = InMemDicomObject::new_empty();
        put_is(&mut ob, tags::OBSERVATION_NUMBER, num);
        put_is(&mut ob, tags::REFERENCED_ROI_NUMBER, num);
        put_str(&mut ob, tags::RTROI_INTERPRETED_TYPE, VR::CS, typ);
        put_str(&mut ob, tags::ROI_INTERPRETER, VR::PN, "");
        obs.push(ob);
    }
    put_seq(&mut o, tags::STRUCTURE_SET_ROI_SEQUENCE, ssr);
    put_seq(&mut o, tags::ROI_CONTOUR_SEQUENCE, rcs);
    put_seq(&mut o, tags::RTROI_OBSERVATIONS_SEQUENCE, obs);

    write_object(o, SOP_RTSTRUCT, &dir.join("RS_synth.dcm"))?;
    Ok(1)
}

// -- RTPLAN (ion) -----------------------------------------------------------

fn write_rtplan(dir: &Path, p: &GenParams, ids: &Ids, progress: &Progress) -> Result<usize> {
    progress.set("Writing RTPLAN…");
    let mut o = base_dataset(ids, SOP_RTIONPLAN, &ids.plan_uid, "RTPLAN");
    put_str(&mut o, tags::SERIES_INSTANCE_UID, VR::UI, new_uid());
    put_is(&mut o, tags::SERIES_NUMBER, 3);
    put_str(
        &mut o,
        tags::FRAME_OF_REFERENCE_UID,
        VR::UI,
        ids.for_uid.clone(),
    );
    put_str(
        &mut o,
        tags::RT_PLAN_LABEL,
        VR::SH,
        p.plan_label.chars().take(16).collect::<String>(),
    );
    put_str(&mut o, tags::RT_PLAN_NAME, VR::LO, "Synthetic proton plan");
    put_str(&mut o, tags::RT_PLAN_DATE, VR::DA, ids.date.clone());
    put_str(&mut o, tags::RT_PLAN_TIME, VR::TM, ids.time.clone());
    put_str(&mut o, tags::RT_PLAN_GEOMETRY, VR::CS, "PATIENT");
    put_seq(
        &mut o,
        tags::REFERENCED_STRUCTURE_SET_SEQUENCE,
        vec![ref_item(SOP_RTSTRUCT, &ids.struct_uid)],
    );

    let mut dr = InMemDicomObject::new_empty();
    put_is(&mut dr, tags::DOSE_REFERENCE_NUMBER, 1);
    put_str(&mut dr, tags::DOSE_REFERENCE_STRUCTURE_TYPE, VR::CS, "SITE");
    put_str(&mut dr, tags::DOSE_REFERENCE_DESCRIPTION, VR::LO, "TARGET");
    put_str(&mut dr, tags::DOSE_REFERENCE_TYPE, VR::CS, "TARGET");
    put_ds(&mut dr, tags::TARGET_PRESCRIPTION_DOSE, &[p.peak]);
    put_seq(&mut o, tags::DOSE_REFERENCE_SEQUENCE, vec![dr]);

    let beam_specs: [(i64, &str, f64, f64, f64); 2] =
        [(1, "G000", 0.0, 120.5, 1.05), (2, "G090", 90.0, 98.3, 0.95)];

    let mut fg = InMemDicomObject::new_empty();
    put_is(&mut fg, tags::FRACTION_GROUP_NUMBER, 1);
    put_is(&mut fg, tags::NUMBER_OF_FRACTIONS_PLANNED, 30);
    put_is(&mut fg, tags::NUMBER_OF_BEAMS, beam_specs.len() as i64);
    put_is(&mut fg, tags::NUMBER_OF_BRACHY_APPLICATION_SETUPS, 0);
    let rbs = beam_specs
        .iter()
        .map(|&(num, _, _, mset, bdose)| {
            let mut rb = InMemDicomObject::new_empty();
            put_is(&mut rb, tags::REFERENCED_BEAM_NUMBER, num);
            put_ds(&mut rb, tags::BEAM_METERSET, &[mset]);
            put_ds(&mut rb, tags::BEAM_DOSE, &[bdose]);
            rb
        })
        .collect();
    put_seq(&mut fg, tags::REFERENCED_BEAM_SEQUENCE, rbs);
    put_seq(&mut o, tags::FRACTION_GROUP_SEQUENCE, vec![fg]);

    const ENERGIES: [f64; 4] = [180.0, 160.0, 140.0, 120.0];
    let mut beams = Vec::new();
    for (num, name, gantry, _, _) in beam_specs {
        let mut b = InMemDicomObject::new_empty();
        put_is(&mut b, tags::BEAM_NUMBER, num);
        put_str(&mut b, tags::BEAM_NAME, VR::LO, name);
        put_str(&mut b, tags::BEAM_TYPE, VR::CS, "STATIC");
        put_str(&mut b, tags::RADIATION_TYPE, VR::CS, "PROTON");
        put_str(&mut b, tags::SCAN_MODE, VR::CS, "MODULATED");
        put_str(&mut b, tags::TREATMENT_MACHINE_NAME, VR::SH, MACHINE);
        put_str(&mut b, tags::TREATMENT_DELIVERY_TYPE, VR::CS, "TREATMENT");
        put_is(&mut b, tags::NUMBER_OF_WEDGES, 0);
        put_is(&mut b, tags::NUMBER_OF_COMPENSATORS, 0);
        put_is(&mut b, tags::NUMBER_OF_BOLI, 0);
        put_is(&mut b, tags::NUMBER_OF_BLOCKS, 0);
        put_ds(&mut b, tags::FINAL_CUMULATIVE_METERSET_WEIGHT, &[1.0]);
        put_is(
            &mut b,
            tags::NUMBER_OF_CONTROL_POINTS,
            ENERGIES.len() as i64,
        );
        put_is(&mut b, tags::NUMBER_OF_RANGE_SHIFTERS, 0);
        put_is(&mut b, tags::NUMBER_OF_LATERAL_SPREADING_DEVICES, 0);
        put_is(&mut b, tags::NUMBER_OF_RANGE_MODULATORS, 0);
        put_str(&mut b, tags::PATIENT_SUPPORT_TYPE, VR::CS, "TABLE");

        let mut cps = Vec::new();
        for (ci, energy) in ENERGIES.iter().enumerate() {
            let mut cp = InMemDicomObject::new_empty();
            put_is(&mut cp, tags::CONTROL_POINT_INDEX, ci as i64);
            put_ds(&mut cp, tags::NOMINAL_BEAM_ENERGY, &[*energy]);
            put_ds(
                &mut cp,
                tags::CUMULATIVE_METERSET_WEIGHT,
                &[ci as f64 / (ENERGIES.len() as f64 - 1.0)],
            );
            if ci == 0 {
                put_ds(&mut cp, tags::GANTRY_ANGLE, &[gantry]);
                put_str(&mut cp, tags::GANTRY_ROTATION_DIRECTION, VR::CS, "NONE");
                put_ds(&mut cp, tags::PATIENT_SUPPORT_ANGLE, &[0.0]);
                put_str(
                    &mut cp,
                    tags::PATIENT_SUPPORT_ROTATION_DIRECTION,
                    VR::CS,
                    "NONE",
                );
                put_ds(
                    &mut cp,
                    tags::ISOCENTER_POSITION,
                    &[p.shift_x, p.target_shift_y + p.shift_y, 0.0],
                );
            }
            cps.push(cp);
        }
        put_seq(&mut b, tags::ION_CONTROL_POINT_SEQUENCE, cps);
        beams.push(b);
    }
    put_seq(&mut o, tags::ION_BEAM_SEQUENCE, beams);

    write_object(o, SOP_RTIONPLAN, &dir.join("RP_synth.dcm"))?;
    Ok(1)
}

// -- RTDOSE -----------------------------------------------------------------

fn write_rtdose(dir: &Path, p: &GenParams, ids: &Ids, progress: &Progress) -> Result<usize> {
    progress.set("Writing RTDOSE…");
    let dx0 = axis_origin(DNX, DSP);
    let dy0 = axis_origin(DNY, DSP);
    // Frames step by the CT slice spacing, not the in-plane dose spacing.
    let dz0 = axis_origin(DNZ, SPACING);
    let (sx, sy) = (p.shift_x, p.shift_y);

    let mut stored = Vec::with_capacity(DNX * DNY * DNZ * 2);
    let mut offsets = Vec::with_capacity(DNZ);
    let two_sigma2 = 2.0 * SIGMA * SIGMA;
    for f in 0..DNZ {
        let z = dz0 + f as f64 * SPACING;
        offsets.push(z - dz0);
        for j in 0..DNY {
            let y = dy0 + j as f64 * DSP;
            let dy = y - p.target_shift_y - sy;
            for i in 0..DNX {
                let x = dx0 + i as f64 * DSP;
                let dx = x - sx;
                let r2 = dx * dx + dy * dy + z * z;
                let dose = p.peak * (-r2 / two_sigma2).exp();
                let v = (dose / DOSE_SCALING).round().clamp(0.0, u32::MAX as f64) as u32;
                // 32-bit native little-endian: low word first.
                stored.push((v & 0xFFFF) as u16);
                stored.push((v >> 16) as u16);
            }
        }
    }

    let mut o = base_dataset(ids, SOP_RTDOSE, &new_uid(), "RTDOSE");
    put_str(&mut o, tags::SERIES_INSTANCE_UID, VR::UI, new_uid());
    put_is(&mut o, tags::SERIES_NUMBER, 4);
    put_str(
        &mut o,
        tags::FRAME_OF_REFERENCE_UID,
        VR::UI,
        ids.for_uid.clone(),
    );
    put_ds(&mut o, tags::IMAGE_POSITION_PATIENT, &[dx0, dy0, dz0]);
    put_ds(
        &mut o,
        tags::IMAGE_ORIENTATION_PATIENT,
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    );
    put_ds(&mut o, tags::PIXEL_SPACING, &[DSP, DSP]);
    put_str(&mut o, tags::SLICE_THICKNESS, VR::DS, "");
    put_us(&mut o, tags::ROWS, DNY as u16);
    put_us(&mut o, tags::COLUMNS, DNX as u16);
    put_is(&mut o, tags::NUMBER_OF_FRAMES, DNZ as i64);
    o.put(DataElement::new(
        tags::FRAME_INCREMENT_POINTER,
        VR::AT,
        PrimitiveValue::from(tags::GRID_FRAME_OFFSET_VECTOR),
    ));
    put_ds(&mut o, tags::GRID_FRAME_OFFSET_VECTOR, &offsets);
    put_us(&mut o, tags::BITS_ALLOCATED, 32);
    put_us(&mut o, tags::BITS_STORED, 32);
    put_us(&mut o, tags::HIGH_BIT, 31);
    put_us(&mut o, tags::PIXEL_REPRESENTATION, 0);
    put_us(&mut o, tags::SAMPLES_PER_PIXEL, 1);
    put_str(
        &mut o,
        tags::PHOTOMETRIC_INTERPRETATION,
        VR::CS,
        "MONOCHROME2",
    );
    put_str(&mut o, tags::DOSE_UNITS, VR::CS, "GY");
    put_str(&mut o, tags::DOSE_TYPE, VR::CS, "PHYSICAL");
    put_str(&mut o, tags::DOSE_SUMMATION_TYPE, VR::CS, "PLAN");
    put_ds(&mut o, tags::DOSE_GRID_SCALING, &[DOSE_SCALING]);
    put_seq(
        &mut o,
        tags::REFERENCED_RT_PLAN_SEQUENCE,
        vec![ref_item(SOP_RTIONPLAN, &ids.plan_uid)],
    );
    put_pixels_u16(&mut o, stored);

    write_object(o, SOP_RTDOSE, &dir.join("RD_synth.dcm"))?;
    Ok(1)
}

// -- Extras: DX, RTIMAGE, REG, RTRECORD -------------------------------------

fn write_extras(dir: &Path, p: &GenParams, ids: &Ids, progress: &Progress) -> Result<usize> {
    progress.set("Writing DX / RTIMAGE…");
    let raw = ap_radiograph(p);

    // Detector extent: 190 mm across x, 80 mm across z.
    let sp_row = 80.0 / PX_ROWS as f64;
    let sp_col = 190.0 / PX_COLS as f64;

    /// Elements shared by the DX and the RTIMAGE object.
    fn planar_common(o: &mut InMemDicomObject, raw: &[u16]) {
        put_us(o, tags::ROWS, PX_ROWS as u16);
        put_us(o, tags::COLUMNS, PX_COLS as u16);
        put_us(o, tags::BITS_ALLOCATED, 16);
        put_us(o, tags::BITS_STORED, 16);
        put_us(o, tags::HIGH_BIT, 15);
        put_us(o, tags::PIXEL_REPRESENTATION, 0);
        put_us(o, tags::SAMPLES_PER_PIXEL, 1);
        put_str(o, tags::PHOTOMETRIC_INTERPRETATION, VR::CS, "MONOCHROME2");
        put_ds(o, tags::WINDOW_CENTER, &[1500.0]);
        put_ds(o, tags::WINDOW_WIDTH, &[3000.0]);
        put_pixels_u16(o, raw.to_vec());
    }

    // ---- DX radiograph ----
    let mut o = base_dataset(ids, SOP_DX, &new_uid(), "DX");
    put_str(&mut o, tags::SERIES_INSTANCE_UID, VR::UI, new_uid());
    put_is(&mut o, tags::SERIES_NUMBER, 20);
    put_str(&mut o, tags::SERIES_DESCRIPTION, VR::LO, "Synthetic DX AP");
    put_ds(&mut o, tags::IMAGER_PIXEL_SPACING, &[sp_row, sp_col]);
    put_str(&mut o, tags::BODY_PART_EXAMINED, VR::CS, "ABDOMEN");
    put_str(&mut o, tags::VIEW_POSITION, VR::CS, "AP");
    put_str(&mut o, tags::KVP, VR::DS, "120");
    planar_common(&mut o, &raw);
    write_object(o, SOP_DX, &dir.join("DX_synth.dcm"))?;

    // ---- RTIMAGE (DRR-like) ----
    let mut o = base_dataset(ids, SOP_RTIMAGE, &new_uid(), "RTIMAGE");
    put_str(&mut o, tags::SERIES_INSTANCE_UID, VR::UI, new_uid());
    put_is(&mut o, tags::SERIES_NUMBER, 21);
    put_str(&mut o, tags::RT_IMAGE_LABEL, VR::SH, "DRR_G000");
    put_str(
        &mut o,
        tags::RT_IMAGE_DESCRIPTION,
        VR::ST,
        "Synthetic DRR, gantry 0",
    );
    put_str(&mut o, tags::RADIATION_MACHINE_NAME, VR::SH, MACHINE);
    put_ds(&mut o, tags::RADIATION_MACHINE_SAD, &[2000.0]);
    put_ds(&mut o, tags::RT_IMAGE_SID, &[1500.0]);
    put_ds(&mut o, tags::GANTRY_ANGLE, &[0.0]);
    put_ds(&mut o, tags::IMAGE_PLANE_PIXEL_SPACING, &[sp_row, sp_col]);
    put_str(&mut o, tags::RT_IMAGE_PLANE, VR::CS, "NORMAL");
    planar_common(&mut o, &raw);
    write_object(o, SOP_RTIMAGE, &dir.join("RI_synth.dcm"))?;

    // ---- REG: identity for this frame + a rigid translation for another ----
    progress.set("Writing REG / RTRECORD…");
    let [tx, ty, tz] = p.reg_shift;
    let mut o = base_dataset(ids, SOP_SPATIAL_REG, &new_uid(), "REG");
    put_str(&mut o, tags::SERIES_INSTANCE_UID, VR::UI, new_uid());
    put_is(&mut o, tags::SERIES_NUMBER, 22);
    put_str(
        &mut o,
        tags::CONTENT_DESCRIPTION,
        VR::LO,
        "Synthetic spatial registration",
    );
    put_str(
        &mut o,
        tags::FRAME_OF_REFERENCE_UID,
        VR::UI,
        ids.for_uid.clone(),
    );
    put_str(&mut o, tags::CONTENT_DATE, VR::DA, ids.date.clone());
    put_str(&mut o, tags::CONTENT_TIME, VR::TM, ids.time.clone());

    #[rustfmt::skip]
    let identity = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];
    #[rustfmt::skip]
    let shifted = [
        1.0, 0.0, 0.0, tx,
        0.0, 1.0, 0.0, ty,
        0.0, 0.0, 1.0, tz,
        0.0, 0.0, 0.0, 1.0,
    ];
    let reg_item = |item_for: String, matrix: [f64; 16]| {
        let mut ms = InMemDicomObject::new_empty();
        put_str(
            &mut ms,
            tags::FRAME_OF_REFERENCE_TRANSFORMATION_MATRIX_TYPE,
            VR::CS,
            "RIGID",
        );
        put_ds(
            &mut ms,
            tags::FRAME_OF_REFERENCE_TRANSFORMATION_MATRIX,
            &matrix,
        );
        let mut mr = InMemDicomObject::new_empty();
        put_seq(&mut mr, tags::MATRIX_SEQUENCE, vec![ms]);
        let mut it = InMemDicomObject::new_empty();
        put_str(&mut it, tags::FRAME_OF_REFERENCE_UID, VR::UI, item_for);
        put_seq(&mut it, tags::MATRIX_REGISTRATION_SEQUENCE, vec![mr]);
        it
    };
    put_seq(
        &mut o,
        tags::REGISTRATION_SEQUENCE,
        vec![
            reg_item(ids.for_uid.clone(), identity),
            reg_item(new_uid(), shifted),
        ],
    );
    write_object(o, SOP_SPATIAL_REG, &dir.join("REG_synth.dcm"))?;

    // ---- RTRECORD: RT Ion Beams Treatment Record ----
    let mut o = base_dataset(ids, SOP_RT_ION_BEAMS_RECORD, &new_uid(), "RTRECORD");
    put_str(&mut o, tags::SERIES_INSTANCE_UID, VR::UI, new_uid());
    put_is(&mut o, tags::SERIES_NUMBER, 23);
    put_str(&mut o, tags::TREATMENT_DATE, VR::DA, ids.date.clone());
    put_str(&mut o, tags::TREATMENT_TIME, VR::TM, ids.time.clone());
    put_str(&mut o, tags::TREATMENT_MACHINE_NAME, VR::SH, MACHINE);
    let recs = [(1i64, "G000", 120.5, 120.3), (2, "G090", 98.3, 98.3)]
        .iter()
        .map(|&(num, name, spec, deliv)| {
            let mut it = InMemDicomObject::new_empty();
            put_is(&mut it, tags::REFERENCED_BEAM_NUMBER, num);
            put_str(&mut it, tags::BEAM_NAME, VR::LO, name);
            put_is(&mut it, tags::CURRENT_FRACTION_NUMBER, 5);
            put_str(
                &mut it,
                tags::TREATMENT_TERMINATION_STATUS,
                VR::CS,
                "NORMAL",
            );
            put_str(
                &mut it,
                tags::TREATMENT_VERIFICATION_STATUS,
                VR::CS,
                "VERIFIED",
            );
            put_ds(&mut it, tags::SPECIFIED_PRIMARY_METERSET, &[spec]);
            put_ds(&mut it, tags::DELIVERED_PRIMARY_METERSET, &[deliv]);
            put_str(&mut it, tags::TREATMENT_DELIVERY_TYPE, VR::CS, "TREATMENT");
            it
        })
        .collect();
    put_seq(&mut o, tags::TREATMENT_SESSION_ION_BEAM_SEQUENCE, recs);
    write_object(o, SOP_RT_ION_BEAMS_RECORD, &dir.join("RT_record_synth.dcm"))?;

    Ok(4)
}

/// Synthetic AP radiograph: chord lengths through the phantom projected along
/// Y onto an (x, z) detector, normalized to a 0…3000 stored range.
fn ap_radiograph(p: &GenParams) -> Vec<u16> {
    let chord = |r2: f64| 2.0 * r2.max(0.0).sqrt();
    let mut proj = vec![0.0f64; PX_ROWS * PX_COLS];
    let mut max = 0.0f64;
    for j in 0..PX_ROWS {
        // linspace(-40, 40, PX_ROWS), endpoints inclusive.
        let z = -40.0 + 80.0 * j as f64 / (PX_ROWS as f64 - 1.0);
        for i in 0..PX_COLS {
            let x = -95.0 + 190.0 * i as f64 / (PX_COLS as f64 - 1.0);
            let dx = x - p.shift_x;
            let v = 0.02 * chord(R_BODY * R_BODY - dx * dx)
                + 0.05 * chord(R_TARGET * R_TARGET - dx * dx - z * z)
                + 0.08 * chord(R_CORD * R_CORD - dx * dx);
            proj[j * PX_COLS + i] = v;
            max = max.max(v);
        }
    }
    let scale = if max > 0.0 { 3000.0 / max } else { 0.0 };
    proj.iter()
        .map(|&v| (v * scale).round().clamp(0.0, 65535.0) as u16)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phantom_origins_match_the_documented_geometry() {
        assert_eq!(axis_origin(NX, SPACING), -95.0);
        assert_eq!(axis_origin(NZ, SPACING), -39.0);
        assert_eq!(axis_origin(DNX, DSP), -92.0);
        assert_eq!(axis_origin(DNZ, SPACING), -40.0);
    }

    #[test]
    fn circle_points_are_on_the_circle() {
        let pts = circle_points(3.0, -4.0, 25.0, 7.0, 64);
        assert_eq!(pts.len(), 64 * 3);
        for c in pts.as_chunks::<3>().0 {
            let x: f64 = c[0].parse().unwrap();
            let y: f64 = c[1].parse().unwrap();
            let z: f64 = c[2].parse().unwrap();
            assert!(((x - 3.0).hypot(y + 4.0) - 25.0).abs() < 1e-3);
            assert_eq!(z, 7.0);
        }
    }
}
