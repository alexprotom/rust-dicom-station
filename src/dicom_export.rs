//! Export a loaded (or simulated) study as a set of DICOM files:
//! one CT Image Storage file per slice plus RTSTRUCT / RTDOSE / RTPLAN
//! objects, written with `dicom-rs` (Explicit VR Little Endian).
//!
//! The exported objects carry everything this application models (geometry,
//! HU values, contours with colors and types, dose grids with scaling, plan
//! prescription/beams/isocenters). They round-trip through this viewer and
//! standard tooling such as pydicom; they are QA/research exports, not
//! guaranteed-complete clinical IODs.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use dicom_core::value::{PrimitiveValue, Value, C};
use dicom_core::{DataElement, Length, Tag, VR};
use dicom_dictionary_std::tags;
use dicom_object::meta::FileMetaTableBuilder;
use dicom_object::InMemDicomObject;

use crate::loader::{LoadedStudy, Progress};

const SOP_CT: &str = "1.2.840.10008.5.1.4.1.1.2";
const SOP_RTSTRUCT: &str = "1.2.840.10008.5.1.4.1.1.481.3";
const SOP_RTDOSE: &str = "1.2.840.10008.5.1.4.1.1.481.2";
const SOP_RTPLAN: &str = "1.2.840.10008.5.1.4.1.1.481.5";
const SOP_RTIONPLAN: &str = "1.2.840.10008.5.1.4.1.1.481.8";
const EXPLICIT_VR_LE: &str = "1.2.840.10008.1.2.1";

static UID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Generate a unique UID under the UUID-derived `2.25.` root.
pub(crate) fn new_uid() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let n = UID_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("2.25.{}{:04}", nanos, n % 10000)
}

// ---------------------------------------------------------------------------
// Small element helpers (shared with `gen_test_data`)
// ---------------------------------------------------------------------------

pub(crate) fn put_str(o: &mut InMemDicomObject, tag: Tag, vr: VR, v: impl Into<String>) {
    o.put(DataElement::new(tag, vr, PrimitiveValue::Str(v.into())));
}

pub(crate) fn put_strs(o: &mut InMemDicomObject, tag: Tag, vr: VR, vals: &[String]) {
    o.put(DataElement::new(
        tag,
        vr,
        PrimitiveValue::Strs(C::from_vec(vals.to_vec())),
    ));
}

pub(crate) fn put_us(o: &mut InMemDicomObject, tag: Tag, v: u16) {
    o.put(DataElement::new(tag, VR::US, PrimitiveValue::from(v)));
}

pub(crate) fn put_ds(o: &mut InMemDicomObject, tag: Tag, vals: &[f64]) {
    let s: Vec<String> = vals.iter().map(|v| fmt_ds(*v)).collect();
    put_strs(o, tag, VR::DS, &s);
}

pub(crate) fn put_is(o: &mut InMemDicomObject, tag: Tag, v: i64) {
    put_str(o, tag, VR::IS, v.to_string());
}

pub(crate) fn put_seq(o: &mut InMemDicomObject, tag: Tag, items: Vec<InMemDicomObject>) {
    o.put(DataElement::new(
        tag,
        VR::SQ,
        Value::new_sequence(items, Length::UNDEFINED),
    ));
}

/// DICOM DS: max 16 bytes; use a compact fixed-point representation.
pub(crate) fn fmt_ds(v: f64) -> String {
    let s = format!("{v:.6}");
    let s = s.trim_end_matches('0').trim_end_matches('.').to_string();
    if s.is_empty() || s == "-" { "0".into() } else { s }
}

pub(crate) fn today() -> (String, String) {
    // Days since epoch → Y/M/D (proleptic Gregorian, civil algorithm).
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = (secs / 86400) as i64;
    let rem = secs % 86400;
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (
        format!("{:04}{:02}{:02}", y, m, d),
        format!("{:02}{:02}{:02}", rem / 3600, (rem % 3600) / 60, rem % 60),
    )
}

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

struct Ctx {
    study_uid: String,
    for_uid: String,
    date: String,
    time: String,
}

fn common_elements(o: &mut InMemDicomObject, study: &LoadedStudy, ctx: &Ctx, modality: &str) {
    put_str(o, tags::SPECIFIC_CHARACTER_SET, VR::CS, "ISO_IR 100");
    put_str(o, tags::PATIENT_NAME, VR::PN, study.meta.patient_name.clone());
    put_str(o, tags::PATIENT_ID, VR::LO, study.meta.patient_id.clone());
    put_str(o, tags::PATIENT_BIRTH_DATE, VR::DA, "");
    put_str(o, tags::PATIENT_SEX, VR::CS, "O");
    put_str(o, tags::STUDY_INSTANCE_UID, VR::UI, ctx.study_uid.clone());
    put_str(o, tags::STUDY_DATE, VR::DA, ctx.date.clone());
    put_str(o, tags::STUDY_TIME, VR::TM, ctx.time.clone());
    put_str(
        o,
        tags::STUDY_DESCRIPTION,
        VR::LO,
        study.meta.study_description.clone(),
    );
    put_str(o, tags::ACCESSION_NUMBER, VR::SH, "");
    put_str(o, tags::REFERRING_PHYSICIAN_NAME, VR::PN, "");
    put_str(o, tags::MODALITY, VR::CS, modality);
    put_str(o, tags::MANUFACTURER, VR::LO, "rust-dicom-viewer export");
}

pub(crate) fn write_object(
    obj: InMemDicomObject,
    sop_class: &str,
    path: &Path,
) -> Result<()> {
    let file_obj = obj
        .with_meta(
            FileMetaTableBuilder::new()
                .transfer_syntax(EXPLICIT_VR_LE)
                .media_storage_sop_class_uid(sop_class),
        )
        .context("build file meta")?;
    file_obj
        .write_to_file(path)
        .with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

/// Export `study` into `dir` as individual DICOM files.
/// Returns the number of files written.
pub fn export_study(study: &LoadedStudy, dir: &Path, progress: &Progress) -> Result<usize> {
    std::fs::create_dir_all(dir)
        .with_context(|| format!("create directory {}", dir.display()))?;
    let (date, time) = today();
    let vol = &study.volume;
    let ctx = Ctx {
        study_uid: new_uid(),
        for_uid: if vol.frame_of_reference_uid.is_empty() {
            new_uid()
        } else {
            vol.frame_of_reference_uid.clone()
        },
        date,
        time,
    };
    let mut n_files = 0usize;

    // ---- CT series -------------------------------------------------------
    let series_uid = new_uid();
    let [nx, ny, nz] = vol.dims;
    let modality = study
        .series
        .get(study.active_series)
        .map(|s| s.modality.clone())
        .filter(|m| !m.is_empty())
        .unwrap_or_else(|| "CT".into());
    let mut ct_sop_uids = Vec::with_capacity(nz);

    for k in 0..nz {
        if k % 20 == 0 {
            progress.set(format!("Writing CT slice {}/{}…", k + 1, nz));
        }
        let sop_uid = new_uid();
        ct_sop_uids.push(sop_uid.clone());
        let mut o = InMemDicomObject::new_empty();
        common_elements(&mut o, study, &ctx, &modality);
        put_str(&mut o, tags::SOP_CLASS_UID, VR::UI, SOP_CT);
        put_str(&mut o, tags::SOP_INSTANCE_UID, VR::UI, sop_uid);
        put_str(&mut o, tags::SERIES_INSTANCE_UID, VR::UI, series_uid.clone());
        put_is(&mut o, tags::SERIES_NUMBER, 1);
        put_str(
            &mut o,
            tags::SERIES_DESCRIPTION,
            VR::LO,
            study
                .series
                .get(study.active_series)
                .map(|s| s.description.clone())
                .unwrap_or_default(),
        );
        put_is(&mut o, tags::INSTANCE_NUMBER, k as i64 + 1);
        put_str(&mut o, tags::FRAME_OF_REFERENCE_UID, VR::UI, ctx.for_uid.clone());
        put_str(&mut o, tags::POSITION_REFERENCE_INDICATOR, VR::LO, "");
        let ipp = vol.voxel_to_patient(0.0, 0.0, k as f64);
        put_ds(&mut o, tags::IMAGE_POSITION_PATIENT, &[ipp.x, ipp.y, ipp.z]);
        put_ds(
            &mut o,
            tags::IMAGE_ORIENTATION_PATIENT,
            &[
                vol.row_dir.x,
                vol.row_dir.y,
                vol.row_dir.z,
                vol.col_dir.x,
                vol.col_dir.y,
                vol.col_dir.z,
            ],
        );
        // PixelSpacing = [between rows, between columns].
        put_ds(&mut o, tags::PIXEL_SPACING, &[vol.spacing[1], vol.spacing[0]]);
        put_ds(&mut o, tags::SLICE_THICKNESS, &[vol.spacing[2]]);
        put_us(&mut o, tags::ROWS, ny as u16);
        put_us(&mut o, tags::COLUMNS, nx as u16);
        put_us(&mut o, tags::BITS_ALLOCATED, 16);
        put_us(&mut o, tags::BITS_STORED, 16);
        put_us(&mut o, tags::HIGH_BIT, 15);
        put_us(&mut o, tags::PIXEL_REPRESENTATION, 1); // signed (HU stored raw)
        put_us(&mut o, tags::SAMPLES_PER_PIXEL, 1);
        put_str(&mut o, tags::PHOTOMETRIC_INTERPRETATION, VR::CS, "MONOCHROME2");
        put_ds(&mut o, tags::RESCALE_INTERCEPT, &[0.0]);
        put_ds(&mut o, tags::RESCALE_SLOPE, &[1.0]);
        put_str(&mut o, tags::RESCALE_TYPE, VR::LO, "HU");
        put_ds(&mut o, tags::WINDOW_CENTER, &[study.default_window.0 as f64]);
        put_ds(&mut o, tags::WINDOW_WIDTH, &[study.default_window.1 as f64]);

        let base = k * nx * ny;
        let words: Vec<u16> = vol.data[base..base + nx * ny]
            .iter()
            .map(|&v| v as u16)
            .collect();
        o.put(DataElement::new(
            tags::PIXEL_DATA,
            VR::OW,
            PrimitiveValue::U16(C::from_vec(words)),
        ));

        write_object(o, SOP_CT, &dir.join(format!("CT_{k:04}.dcm")))?;
        n_files += 1;
    }

    // ---- RTSTRUCT ---------------------------------------------------------
    if let Some(ss) = &study.structures {
        progress.set("Writing RTSTRUCT…");
        let mut o = InMemDicomObject::new_empty();
        common_elements(&mut o, study, &ctx, "RTSTRUCT");
        put_str(&mut o, tags::SOP_CLASS_UID, VR::UI, SOP_RTSTRUCT);
        put_str(&mut o, tags::SOP_INSTANCE_UID, VR::UI, new_uid());
        put_str(&mut o, tags::SERIES_INSTANCE_UID, VR::UI, new_uid());
        put_is(&mut o, tags::SERIES_NUMBER, 2);
        put_str(&mut o, tags::STRUCTURE_SET_LABEL, VR::SH, truncate(&ss.label, 16));
        put_str(&mut o, tags::STRUCTURE_SET_DATE, VR::DA, ctx.date.clone());
        put_str(&mut o, tags::STRUCTURE_SET_TIME, VR::TM, ctx.time.clone());

        // Referenced frame of reference.
        let mut rfr = InMemDicomObject::new_empty();
        put_str(&mut rfr, tags::FRAME_OF_REFERENCE_UID, VR::UI, ctx.for_uid.clone());
        put_seq(&mut o, tags::REFERENCED_FRAME_OF_REFERENCE_SEQUENCE, vec![rfr]);

        let mut ssr = Vec::new();
        let mut rcs = Vec::new();
        let mut obs = Vec::new();
        for roi in &ss.rois {
            let mut s = InMemDicomObject::new_empty();
            put_is(&mut s, tags::ROI_NUMBER, roi.number as i64);
            put_str(
                &mut s,
                tags::REFERENCED_FRAME_OF_REFERENCE_UID,
                VR::UI,
                ctx.for_uid.clone(),
            );
            put_str(&mut s, tags::ROI_NAME, VR::LO, roi.name.clone());
            put_str(&mut s, tags::ROI_GENERATION_ALGORITHM, VR::CS, "AUTOMATIC");
            ssr.push(s);

            let mut rc = InMemDicomObject::new_empty();
            put_is(&mut rc, tags::REFERENCED_ROI_NUMBER, roi.number as i64);
            put_strs(
                &mut rc,
                tags::ROI_DISPLAY_COLOR,
                VR::IS,
                &[
                    roi.color[0].to_string(),
                    roi.color[1].to_string(),
                    roi.color[2].to_string(),
                ],
            );
            let mut contours = Vec::with_capacity(roi.contours.len());
            for c in &roi.contours {
                let mut co = InMemDicomObject::new_empty();
                put_str(
                    &mut co,
                    tags::CONTOUR_GEOMETRIC_TYPE,
                    VR::CS,
                    c.geometric_type.clone(),
                );
                put_is(&mut co, tags::NUMBER_OF_CONTOUR_POINTS, c.points.len() as i64);
                let data: Vec<String> = c
                    .points
                    .iter()
                    .flat_map(|p| [fmt_ds(p.x), fmt_ds(p.y), fmt_ds(p.z)])
                    .collect();
                put_strs(&mut co, tags::CONTOUR_DATA, VR::DS, &data);
                contours.push(co);
            }
            put_seq(&mut rc, tags::CONTOUR_SEQUENCE, contours);
            rcs.push(rc);

            let mut ob = InMemDicomObject::new_empty();
            put_is(&mut ob, tags::OBSERVATION_NUMBER, roi.number as i64);
            put_is(&mut ob, tags::REFERENCED_ROI_NUMBER, roi.number as i64);
            put_str(&mut ob, tags::RTROI_INTERPRETED_TYPE, VR::CS, roi.roi_type.clone());
            put_str(&mut ob, tags::ROI_INTERPRETER, VR::PN, "");
            obs.push(ob);
        }
        put_seq(&mut o, tags::STRUCTURE_SET_ROI_SEQUENCE, ssr);
        put_seq(&mut o, tags::ROI_CONTOUR_SEQUENCE, rcs);
        put_seq(&mut o, tags::RTROI_OBSERVATIONS_SEQUENCE, obs);

        write_object(o, SOP_RTSTRUCT, &dir.join("RS_export.dcm"))?;
        n_files += 1;
    }

    // ---- RTDOSE (16-bit, rescaled) ----------------------------------------
    for (di, d) in study.doses.iter().enumerate() {
        progress.set(format!("Writing RTDOSE {}/{}…", di + 1, study.doses.len()));
        let [dnx, dny, dnf] = d.dims;
        let scaling = (d.max_dose as f64 / 60000.0).max(1e-9);
        let mut o = InMemDicomObject::new_empty();
        common_elements(&mut o, study, &ctx, "RTDOSE");
        put_str(&mut o, tags::SOP_CLASS_UID, VR::UI, SOP_RTDOSE);
        put_str(&mut o, tags::SOP_INSTANCE_UID, VR::UI, new_uid());
        put_str(&mut o, tags::SERIES_INSTANCE_UID, VR::UI, new_uid());
        put_is(&mut o, tags::SERIES_NUMBER, 3 + di as i64);
        put_str(&mut o, tags::FRAME_OF_REFERENCE_UID, VR::UI, ctx.for_uid.clone());
        put_str(&mut o, tags::POSITION_REFERENCE_INDICATOR, VR::LO, "");
        put_ds(
            &mut o,
            tags::IMAGE_POSITION_PATIENT,
            &[d.origin.x, d.origin.y, d.origin.z],
        );
        put_ds(
            &mut o,
            tags::IMAGE_ORIENTATION_PATIENT,
            &[
                d.row_dir.x, d.row_dir.y, d.row_dir.z, d.col_dir.x, d.col_dir.y, d.col_dir.z,
            ],
        );
        put_ds(&mut o, tags::PIXEL_SPACING, &[d.spacing[1], d.spacing[0]]);
        put_us(&mut o, tags::ROWS, dny as u16);
        put_us(&mut o, tags::COLUMNS, dnx as u16);
        put_is(&mut o, tags::NUMBER_OF_FRAMES, dnf as i64);
        o.put(DataElement::new(
            tags::FRAME_INCREMENT_POINTER,
            VR::AT,
            PrimitiveValue::from(tags::GRID_FRAME_OFFSET_VECTOR),
        ));
        put_ds(&mut o, tags::GRID_FRAME_OFFSET_VECTOR, &d.offsets);
        put_us(&mut o, tags::BITS_ALLOCATED, 16);
        put_us(&mut o, tags::BITS_STORED, 16);
        put_us(&mut o, tags::HIGH_BIT, 15);
        put_us(&mut o, tags::PIXEL_REPRESENTATION, 0);
        put_us(&mut o, tags::SAMPLES_PER_PIXEL, 1);
        put_str(&mut o, tags::PHOTOMETRIC_INTERPRETATION, VR::CS, "MONOCHROME2");
        put_str(&mut o, tags::DOSE_UNITS, VR::CS, d.units.clone());
        put_str(&mut o, tags::DOSE_TYPE, VR::CS, "PHYSICAL");
        put_str(
            &mut o,
            tags::DOSE_SUMMATION_TYPE,
            VR::CS,
            if d.summation_type.is_empty() { "PLAN" } else { &d.summation_type },
        );
        put_ds(&mut o, tags::DOSE_GRID_SCALING, &[scaling]);
        let words: Vec<u16> = d
            .data
            .iter()
            .map(|&v| ((v as f64 / scaling).round().clamp(0.0, 65535.0)) as u16)
            .collect();
        o.put(DataElement::new(
            tags::PIXEL_DATA,
            VR::OW,
            PrimitiveValue::U16(C::from_vec(words)),
        ));

        write_object(o, SOP_RTDOSE, &dir.join(format!("RD_export_{di}.dcm")))?;
        n_files += 1;
    }

    // ---- RTPLAN (skeleton with prescription, fractionation, beams) --------
    for (pi, plan) in study.plans.iter().enumerate() {
        progress.set(format!("Writing RTPLAN {}/{}…", pi + 1, study.plans.len()));
        let ion = plan.plan_kind == "Ion"
            || plan
                .beams
                .iter()
                .any(|b| b.radiation_type == "PROTON" || b.radiation_type == "ION");
        let sop_class = if ion { SOP_RTIONPLAN } else { SOP_RTPLAN };
        let mut o = InMemDicomObject::new_empty();
        common_elements(&mut o, study, &ctx, "RTPLAN");
        put_str(&mut o, tags::SOP_CLASS_UID, VR::UI, sop_class);
        put_str(&mut o, tags::SOP_INSTANCE_UID, VR::UI, new_uid());
        put_str(&mut o, tags::SERIES_INSTANCE_UID, VR::UI, new_uid());
        put_is(&mut o, tags::SERIES_NUMBER, 10 + pi as i64);
        put_str(&mut o, tags::FRAME_OF_REFERENCE_UID, VR::UI, ctx.for_uid.clone());
        put_str(&mut o, tags::RT_PLAN_LABEL, VR::SH, truncate(&plan.label, 16));
        put_str(&mut o, tags::RT_PLAN_NAME, VR::LO, plan.name.clone());
        put_str(&mut o, tags::RT_PLAN_DATE, VR::DA, ctx.date.clone());
        put_str(&mut o, tags::RT_PLAN_TIME, VR::TM, ctx.time.clone());
        put_str(&mut o, tags::RT_PLAN_GEOMETRY, VR::CS, "PATIENT");

        if let Some(rx) = plan.target_prescription_dose {
            let mut dr = InMemDicomObject::new_empty();
            put_is(&mut dr, tags::DOSE_REFERENCE_NUMBER, 1);
            put_str(&mut dr, tags::DOSE_REFERENCE_STRUCTURE_TYPE, VR::CS, "SITE");
            put_str(&mut dr, tags::DOSE_REFERENCE_TYPE, VR::CS, "TARGET");
            put_ds(&mut dr, tags::TARGET_PRESCRIPTION_DOSE, &[rx]);
            put_seq(&mut o, tags::DOSE_REFERENCE_SEQUENCE, vec![dr]);
        }

        let mut fg = InMemDicomObject::new_empty();
        put_is(&mut fg, tags::FRACTION_GROUP_NUMBER, 1);
        put_is(
            &mut fg,
            tags::NUMBER_OF_FRACTIONS_PLANNED,
            plan.n_fractions.unwrap_or(1) as i64,
        );
        put_is(&mut fg, tags::NUMBER_OF_BEAMS, plan.beams.len() as i64);
        put_is(&mut fg, tags::NUMBER_OF_BRACHY_APPLICATION_SETUPS, 0);
        let mut rbs = Vec::new();
        for b in &plan.beams {
            let mut rb = InMemDicomObject::new_empty();
            put_is(&mut rb, tags::REFERENCED_BEAM_NUMBER, b.number as i64);
            if let Some(m) = b.meterset {
                put_ds(&mut rb, tags::BEAM_METERSET, &[m]);
            }
            if let Some(bd) = b.beam_dose {
                put_ds(&mut rb, tags::BEAM_DOSE, &[bd]);
            }
            rbs.push(rb);
        }
        put_seq(&mut fg, tags::REFERENCED_BEAM_SEQUENCE, rbs);
        put_seq(&mut o, tags::FRACTION_GROUP_SEQUENCE, vec![fg]);

        let mut beams = Vec::new();
        for b in &plan.beams {
            let mut bo = InMemDicomObject::new_empty();
            put_is(&mut bo, tags::BEAM_NUMBER, b.number as i64);
            put_str(&mut bo, tags::BEAM_NAME, VR::LO, b.name.clone());
            put_str(&mut bo, tags::BEAM_TYPE, VR::CS, "STATIC");
            put_str(
                &mut bo,
                tags::RADIATION_TYPE,
                VR::CS,
                if b.radiation_type.is_empty() { "PHOTON" } else { &b.radiation_type },
            );
            if ion && !b.scan_mode.is_empty() {
                put_str(&mut bo, tags::SCAN_MODE, VR::CS, b.scan_mode.clone());
            }
            put_str(
                &mut bo,
                tags::TREATMENT_DELIVERY_TYPE,
                VR::CS,
                if b.delivery_type.is_empty() { "TREATMENT" } else { &b.delivery_type },
            );
            put_str(&mut bo, tags::TREATMENT_MACHINE_NAME, VR::SH, "EXPORT");
            put_is(&mut bo, tags::NUMBER_OF_WEDGES, 0);
            put_is(&mut bo, tags::NUMBER_OF_COMPENSATORS, 0);
            put_is(&mut bo, tags::NUMBER_OF_BOLI, 0);
            put_is(&mut bo, tags::NUMBER_OF_BLOCKS, 0);
            put_ds(&mut bo, tags::FINAL_CUMULATIVE_METERSET_WEIGHT, &[1.0]);
            put_is(&mut bo, tags::NUMBER_OF_CONTROL_POINTS, 1);

            let mut cp = InMemDicomObject::new_empty();
            put_is(&mut cp, tags::CONTROL_POINT_INDEX, 0);
            if let Some(e) = b.energy_max.or(b.energy_min) {
                put_ds(&mut cp, tags::NOMINAL_BEAM_ENERGY, &[e]);
            }
            put_ds(&mut cp, tags::CUMULATIVE_METERSET_WEIGHT, &[0.0]);
            if let Some(g) = b.gantry_angle {
                put_ds(&mut cp, tags::GANTRY_ANGLE, &[g]);
                put_str(&mut cp, tags::GANTRY_ROTATION_DIRECTION, VR::CS, "NONE");
            }
            if let Some(c) = b.couch_angle {
                put_ds(&mut cp, tags::PATIENT_SUPPORT_ANGLE, &[c]);
                put_str(
                    &mut cp,
                    tags::PATIENT_SUPPORT_ROTATION_DIRECTION,
                    VR::CS,
                    "NONE",
                );
            }
            if let Some(iso) = b.isocenter {
                put_ds(&mut cp, tags::ISOCENTER_POSITION, &[iso.x, iso.y, iso.z]);
            }
            let cp_tag = if ion {
                tags::ION_CONTROL_POINT_SEQUENCE
            } else {
                tags::CONTROL_POINT_SEQUENCE
            };
            put_seq(&mut bo, cp_tag, vec![cp]);
            beams.push(bo);
        }
        let beam_tag = if ion { tags::ION_BEAM_SEQUENCE } else { tags::BEAM_SEQUENCE };
        put_seq(&mut o, beam_tag, beams);

        write_object(o, sop_class, &dir.join(format!("RP_export_{pi}.dcm")))?;
        n_files += 1;
    }

    progress.set("done");
    Ok(n_files)
}

fn truncate(s: &str, max: usize) -> String {
    s.chars().take(max).collect()
}
