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

use crate::dicomseg;
use crate::loader::LoadedStudy;
use crate::progress::Progress;

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
    if s.is_empty() || s == "-" {
        "0".into()
    } else {
        s
    }
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
// Export parameters (the editable DICOM attributes of *File ▶ Export*)
// ---------------------------------------------------------------------------

/// One patient / study / equipment attribute written into every exported
/// object. Values are pre-filled from the loaded study and can be edited in
/// the export dialog before the files are written.
#[derive(Clone)]
pub struct ExportField {
    pub tag: Tag,
    pub name: &'static str,
    pub vr: VR,
    /// The value that will be written (empty ⇒ zero-length element).
    pub value: String,
    /// What the study itself carried — the dialog's “↺” restores it.
    pub suggested: String,
    /// Unchecked rows are skipped entirely (the tag is not written).
    pub enabled: bool,
}

/// Everything the export dialog can influence.
#[derive(Clone)]
pub struct ExportParams {
    pub fields: Vec<ExportField>,
    /// Keep the study's Frame of Reference UID, so the export stays spatially
    /// linked to its source; otherwise a fresh one is generated.
    pub keep_frame_of_reference: bool,
}

impl ExportParams {
    /// Defaults for `study`: identity and description tags taken from the
    /// study itself, dates/times from the current clock where the study has
    /// none, equipment tags naming this application.
    pub fn for_study(study: &LoadedStudy) -> Self {
        let (date, time) = today();
        let f = |tag, name, vr, value: String| ExportField {
            tag,
            name,
            vr,
            suggested: value.clone(),
            value,
            enabled: true,
        };
        let series_desc = study
            .series
            .get(study.active_series)
            .map(|s| s.description.clone())
            .unwrap_or_default();
        let study_date = if study.meta.study_date.trim().is_empty() {
            date
        } else {
            study.meta.study_date.trim().to_string()
        };
        ExportParams {
            fields: vec![
                f(
                    tags::PATIENT_NAME,
                    "PatientName",
                    VR::PN,
                    study.meta.patient_name.clone(),
                ),
                f(
                    tags::PATIENT_ID,
                    "PatientID",
                    VR::LO,
                    study.meta.patient_id.clone(),
                ),
                f(
                    tags::PATIENT_BIRTH_DATE,
                    "PatientBirthDate",
                    VR::DA,
                    String::new(),
                ),
                f(tags::PATIENT_SEX, "PatientSex", VR::CS, "O".into()),
                f(tags::STUDY_ID, "StudyID", VR::SH, "1".into()),
                f(
                    tags::STUDY_DESCRIPTION,
                    "StudyDescription",
                    VR::LO,
                    study.meta.study_description.clone(),
                ),
                f(tags::STUDY_DATE, "StudyDate", VR::DA, study_date),
                f(tags::STUDY_TIME, "StudyTime", VR::TM, time),
                f(
                    tags::ACCESSION_NUMBER,
                    "AccessionNumber",
                    VR::SH,
                    String::new(),
                ),
                f(
                    tags::REFERRING_PHYSICIAN_NAME,
                    "ReferringPhysicianName",
                    VR::PN,
                    String::new(),
                ),
                f(
                    tags::SERIES_DESCRIPTION,
                    "SeriesDescription",
                    VR::LO,
                    series_desc,
                ),
                f(
                    tags::INSTITUTION_NAME,
                    "InstitutionName",
                    VR::LO,
                    String::new(),
                ),
                f(tags::STATION_NAME, "StationName", VR::SH, String::new()),
                f(
                    tags::MANUFACTURER,
                    "Manufacturer",
                    VR::LO,
                    "rust-dicom-station".into(),
                ),
                f(
                    tags::MANUFACTURER_MODEL_NAME,
                    "ManufacturerModelName",
                    VR::LO,
                    "DICOM export".into(),
                ),
            ],
            keep_frame_of_reference: true,
        }
    }

    /// The value that will be written for `tag` (`None` when the row is off).
    pub fn value(&self, tag: Tag) -> Option<&str> {
        self.fields
            .iter()
            .find(|f| f.tag == tag && f.enabled)
            .map(|f| f.value.trim())
    }

    /// Write every enabled field except SeriesDescription, which is a
    /// per-series attribute and only belongs on the image series.
    pub(crate) fn write_common(&self, o: &mut InMemDicomObject) {
        for f in self.fields.iter().filter(|f| f.enabled) {
            if f.tag == tags::SERIES_DESCRIPTION {
                continue;
            }
            put_str(o, f.tag, f.vr, f.value.trim());
        }
    }
}

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

struct Ctx<'a> {
    study_uid: String,
    for_uid: String,
    /// Date / time stamped on the RT objects — the StudyDate / StudyTime
    /// fields of `params` when set, today otherwise.
    date: String,
    time: String,
    params: &'a ExportParams,
}

fn common_elements(o: &mut InMemDicomObject, ctx: &Ctx, modality: &str) {
    put_str(o, tags::SPECIFIC_CHARACTER_SET, VR::CS, "ISO_IR 100");
    ctx.params.write_common(o);
    put_str(o, tags::STUDY_INSTANCE_UID, VR::UI, ctx.study_uid.clone());
    put_str(o, tags::MODALITY, VR::CS, modality);
}

pub(crate) fn write_object(obj: InMemDicomObject, sop_class: &str, path: &Path) -> Result<()> {
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

/// Build the RTSTRUCT object for one structure set.
///
/// Split out of [`export_study`] so the archive can write the same object
/// against the study it already belongs to (`export_derived`) instead of the
/// fresh one a full export invents.
fn build_rtstruct(
    ss: &crate::rtstruct::StructureSet,
    ctx: &Ctx,
    series_number: i64,
    sop_uid: &str,
) -> InMemDicomObject {
    let mut o = InMemDicomObject::new_empty();
    common_elements(&mut o, ctx, "RTSTRUCT");
    put_str(&mut o, tags::SOP_CLASS_UID, VR::UI, SOP_RTSTRUCT);
    put_str(&mut o, tags::SOP_INSTANCE_UID, VR::UI, sop_uid.to_string());
    put_str(&mut o, tags::SERIES_INSTANCE_UID, VR::UI, new_uid());
    put_is(&mut o, tags::SERIES_NUMBER, series_number);
    put_str(
        &mut o,
        tags::STRUCTURE_SET_LABEL,
        VR::SH,
        truncate(&ss.label, 16),
    );
    put_str(&mut o, tags::STRUCTURE_SET_DATE, VR::DA, ctx.date.clone());
    put_str(&mut o, tags::STRUCTURE_SET_TIME, VR::TM, ctx.time.clone());

    // Referenced frame of reference.
    let mut rfr = InMemDicomObject::new_empty();
    put_str(
        &mut rfr,
        tags::FRAME_OF_REFERENCE_UID,
        VR::UI,
        ctx.for_uid.clone(),
    );
    put_seq(
        &mut o,
        tags::REFERENCED_FRAME_OF_REFERENCE_SEQUENCE,
        vec![rfr],
    );

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
            put_is(
                &mut co,
                tags::NUMBER_OF_CONTOUR_POINTS,
                c.points.len() as i64,
            );
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
        put_str(
            &mut ob,
            tags::RTROI_INTERPRETED_TYPE,
            VR::CS,
            roi.roi_type.clone(),
        );
        put_str(&mut ob, tags::ROI_INTERPRETER, VR::PN, "");
        obs.push(ob);
    }
    put_seq(&mut o, tags::STRUCTURE_SET_ROI_SEQUENCE, ssr);
    put_seq(&mut o, tags::ROI_CONTOUR_SEQUENCE, rcs);
    put_seq(&mut o, tags::RTROI_OBSERVATIONS_SEQUENCE, obs);

    o
}

/// Export `study` into `dir` as individual DICOM files.
/// Returns the number of files written.
pub fn export_study(
    study: &LoadedStudy,
    dir: &Path,
    params: &ExportParams,
    progress: &Progress,
) -> Result<usize> {
    std::fs::create_dir_all(dir).with_context(|| format!("create directory {}", dir.display()))?;
    let (today_date, today_time) = today();
    let vol = &study.volume;
    let pick = |tag, fallback: &str| {
        params
            .value(tag)
            .filter(|v| !v.is_empty())
            .unwrap_or(fallback)
            .to_string()
    };
    let ctx = Ctx {
        study_uid: new_uid(),
        for_uid: if params.keep_frame_of_reference && !vol.frame_of_reference_uid.is_empty() {
            vol.frame_of_reference_uid.clone()
        } else {
            new_uid()
        },
        date: pick(tags::STUDY_DATE, &today_date),
        time: pick(tags::STUDY_TIME, &today_time),
        params,
    };
    let mut n_files = 0usize;

    // Stable SOP Instance UIDs so the exported objects keep their DICOM
    // cross-references (dose ▶ plan ▶ structure set).
    let rs_uids: Vec<String> = study.structure_sets.iter().map(|_| new_uid()).collect();
    let plan_uids: Vec<String> = study.plans.iter().map(|_| new_uid()).collect();

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
            progress.set(format!("Writing CT slice {}/{}", k + 1, nz));
        }
        let sop_uid = new_uid();
        ct_sop_uids.push(sop_uid.clone());
        let mut o = InMemDicomObject::new_empty();
        common_elements(&mut o, &ctx, &modality);
        put_str(&mut o, tags::SOP_CLASS_UID, VR::UI, SOP_CT);
        put_str(&mut o, tags::SOP_INSTANCE_UID, VR::UI, sop_uid);
        put_str(
            &mut o,
            tags::SERIES_INSTANCE_UID,
            VR::UI,
            series_uid.clone(),
        );
        put_is(&mut o, tags::SERIES_NUMBER, 1);
        if let Some(d) = params.value(tags::SERIES_DESCRIPTION) {
            put_str(&mut o, tags::SERIES_DESCRIPTION, VR::LO, d);
        }
        put_is(&mut o, tags::INSTANCE_NUMBER, k as i64 + 1);
        put_str(
            &mut o,
            tags::FRAME_OF_REFERENCE_UID,
            VR::UI,
            ctx.for_uid.clone(),
        );
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
        put_ds(
            &mut o,
            tags::PIXEL_SPACING,
            &[vol.spacing[1], vol.spacing[0]],
        );
        put_ds(&mut o, tags::SLICE_THICKNESS, &[vol.spacing[2]]);
        put_us(&mut o, tags::ROWS, ny as u16);
        put_us(&mut o, tags::COLUMNS, nx as u16);
        put_us(&mut o, tags::BITS_ALLOCATED, 16);
        put_us(&mut o, tags::BITS_STORED, 16);
        put_us(&mut o, tags::HIGH_BIT, 15);
        put_us(&mut o, tags::PIXEL_REPRESENTATION, 1); // signed (HU stored raw)
        put_us(&mut o, tags::SAMPLES_PER_PIXEL, 1);
        put_str(
            &mut o,
            tags::PHOTOMETRIC_INTERPRETATION,
            VR::CS,
            "MONOCHROME2",
        );
        put_ds(&mut o, tags::RESCALE_INTERCEPT, &[0.0]);
        put_ds(&mut o, tags::RESCALE_SLOPE, &[1.0]);
        put_str(&mut o, tags::RESCALE_TYPE, VR::LO, "HU");
        put_ds(
            &mut o,
            tags::WINDOW_CENTER,
            &[study.default_window.0 as f64],
        );
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

    // ---- RTSTRUCT (one file per structure set) ----------------------------
    for (si, ss) in study.structure_sets.iter().enumerate() {
        progress.set(format!(
            "Writing RTSTRUCT {}/{}",
            si + 1,
            study.structure_sets.len()
        ));
        let o = build_rtstruct(ss, &ctx, 2 + si as i64, &rs_uids[si]);
        write_object(o, SOP_RTSTRUCT, &dir.join(format!("RS_export_{si}.dcm")))?;
        n_files += 1;
    }

    // ---- SEG (one Segmentation Storage object per segmentation series) ----
    for (gi, ser) in study.seg_series.iter().enumerate() {
        if ser.segs.iter().all(|s| s.count == 0) {
            continue;
        }
        progress.set(format!("Writing SEG {}/{}", gi + 1, study.seg_series.len()));
        // The exported image slices may only be claimed as the source of a
        // series that actually sits on their lattice.
        let same_grid = ser.grid.matches(&vol.grid());
        let seg_ctx = dicomseg::SegWriteCtx {
            study_uid: &ctx.study_uid,
            for_uid: &ctx.for_uid,
            date: &ctx.date,
            time: &ctx.time,
            series_number: 20 + gi as i64,
            image_series_uid: if same_grid { &series_uid } else { "" },
            image_sop_uids: if same_grid { &ct_sop_uids } else { &[] },
            params,
        };
        dicomseg::write(ser, &seg_ctx, &dir.join(format!("SEG_export_{gi}.dcm")))?;
        n_files += 1;
    }

    // ---- RTDOSE (16-bit, rescaled) ----------------------------------------
    for (di, d) in study.doses.iter().enumerate() {
        progress.set(format!("Writing RTDOSE {}/{}", di + 1, study.doses.len()));
        let [dnx, dny, dnf] = d.dims;
        let scaling = (d.max_dose as f64 / 60000.0).max(1e-9);
        let mut o = InMemDicomObject::new_empty();
        common_elements(&mut o, &ctx, "RTDOSE");
        put_str(&mut o, tags::SOP_CLASS_UID, VR::UI, SOP_RTDOSE);
        put_str(&mut o, tags::SOP_INSTANCE_UID, VR::UI, new_uid());
        put_str(&mut o, tags::SERIES_INSTANCE_UID, VR::UI, new_uid());
        put_is(&mut o, tags::SERIES_NUMBER, 3 + di as i64);
        put_str(
            &mut o,
            tags::FRAME_OF_REFERENCE_UID,
            VR::UI,
            ctx.for_uid.clone(),
        );
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
                d.row_dir.x,
                d.row_dir.y,
                d.row_dir.z,
                d.col_dir.x,
                d.col_dir.y,
                d.col_dir.z,
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
        put_str(
            &mut o,
            tags::PHOTOMETRIC_INTERPRETATION,
            VR::CS,
            "MONOCHROME2",
        );
        put_str(&mut o, tags::DOSE_UNITS, VR::CS, d.units.clone());
        put_str(&mut o, tags::DOSE_TYPE, VR::CS, "PHYSICAL");
        put_str(
            &mut o,
            tags::DOSE_SUMMATION_TYPE,
            VR::CS,
            if d.summation_type.is_empty() {
                "PLAN"
            } else {
                &d.summation_type
            },
        );
        put_ds(&mut o, tags::DOSE_GRID_SCALING, &[scaling]);
        if !plan_uids.is_empty() {
            let pidx = study
                .plans
                .iter()
                .position(|p| p.sop_instance_uid == d.referenced_plan_uid)
                .unwrap_or(0);
            let mut rp = InMemDicomObject::new_empty();
            put_str(
                &mut rp,
                tags::REFERENCED_SOP_CLASS_UID,
                VR::UI,
                SOP_RTIONPLAN,
            );
            put_str(
                &mut rp,
                tags::REFERENCED_SOP_INSTANCE_UID,
                VR::UI,
                plan_uids[pidx].clone(),
            );
            put_seq(&mut o, tags::REFERENCED_RT_PLAN_SEQUENCE, vec![rp]);
        }
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
        progress.set(format!("Writing RTPLAN {}/{}", pi + 1, study.plans.len()));
        let ion = plan.plan_kind == "Ion"
            || plan
                .beams
                .iter()
                .any(|b| b.radiation_type == "PROTON" || b.radiation_type == "ION");
        let sop_class = if ion { SOP_RTIONPLAN } else { SOP_RTPLAN };
        let mut o = InMemDicomObject::new_empty();
        common_elements(&mut o, &ctx, "RTPLAN");
        put_str(&mut o, tags::SOP_CLASS_UID, VR::UI, sop_class);
        put_str(
            &mut o,
            tags::SOP_INSTANCE_UID,
            VR::UI,
            plan_uids[pi].clone(),
        );
        put_str(&mut o, tags::SERIES_INSTANCE_UID, VR::UI, new_uid());
        put_is(&mut o, tags::SERIES_NUMBER, 10 + pi as i64);
        put_str(
            &mut o,
            tags::FRAME_OF_REFERENCE_UID,
            VR::UI,
            ctx.for_uid.clone(),
        );
        put_str(
            &mut o,
            tags::RT_PLAN_LABEL,
            VR::SH,
            truncate(&plan.label, 16),
        );
        put_str(&mut o, tags::RT_PLAN_NAME, VR::LO, plan.name.clone());
        put_str(&mut o, tags::RT_PLAN_DATE, VR::DA, ctx.date.clone());
        put_str(&mut o, tags::RT_PLAN_TIME, VR::TM, ctx.time.clone());
        put_str(&mut o, tags::RT_PLAN_GEOMETRY, VR::CS, "PATIENT");
        if !rs_uids.is_empty() {
            let sidx = study
                .structure_sets
                .iter()
                .position(|ss| ss.sop_instance_uid == plan.referenced_structset_uid)
                .unwrap_or(0);
            let mut rs = InMemDicomObject::new_empty();
            put_str(
                &mut rs,
                tags::REFERENCED_SOP_CLASS_UID,
                VR::UI,
                SOP_RTSTRUCT,
            );
            put_str(
                &mut rs,
                tags::REFERENCED_SOP_INSTANCE_UID,
                VR::UI,
                rs_uids[sidx].clone(),
            );
            put_seq(&mut o, tags::REFERENCED_STRUCTURE_SET_SEQUENCE, vec![rs]);
        }

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
                if b.radiation_type.is_empty() {
                    "PHOTON"
                } else {
                    &b.radiation_type
                },
            );
            if ion && !b.scan_mode.is_empty() {
                put_str(&mut bo, tags::SCAN_MODE, VR::CS, b.scan_mode.clone());
            }
            put_str(
                &mut bo,
                tags::TREATMENT_DELIVERY_TYPE,
                VR::CS,
                if b.delivery_type.is_empty() {
                    "TREATMENT"
                } else {
                    &b.delivery_type
                },
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
        let beam_tag = if ion {
            tags::ION_BEAM_SEQUENCE
        } else {
            tags::BEAM_SEQUENCE
        };
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

// ---------------------------------------------------------------------------
// Deformable Spatial Registration
// ---------------------------------------------------------------------------

/// SOP Class UID of the Deformable Spatial Registration IOD.
pub const SOP_DEFORMABLE_REG: &str = "1.2.840.10008.5.1.4.1.1.66.3";

// The IOD's own tags. Written by number rather than by dictionary name so
// the file does not depend on which release of the data dictionary is
// linked — these five have not moved since Supplement 73.
const TAG_DEFORMABLE_REGISTRATION_SEQ: Tag = Tag(0x0064, 0x0002);
const TAG_SOURCE_FRAME_OF_REFERENCE_UID: Tag = Tag(0x0064, 0x0003);
const TAG_DEFORMABLE_REGISTRATION_GRID_SEQ: Tag = Tag(0x0064, 0x0005);
const TAG_GRID_DIMENSIONS: Tag = Tag(0x0064, 0x0007);
const TAG_GRID_RESOLUTION: Tag = Tag(0x0064, 0x0008);
const TAG_VECTOR_GRID_DATA: Tag = Tag(0x0064, 0x0009);
const TAG_PRE_DEFORMATION_MATRIX_SEQ: Tag = Tag(0x0064, 0x000F);
const TAG_POST_DEFORMATION_MATRIX_SEQ: Tag = Tag(0x0064, 0x0010);
const TAG_MATRIX_SEQ: Tag = Tag(0x0070, 0x030A);
const TAG_MATRIX_TYPE: Tag = Tag(0x0070, 0x030C);

fn put_ul(o: &mut InMemDicomObject, tag: Tag, vals: &[u32]) {
    o.put(DataElement::new(
        tag,
        VR::UL,
        PrimitiveValue::U32(C::from_vec(vals.to_vec())),
    ));
}

fn put_fd(o: &mut InMemDicomObject, tag: Tag, vals: &[f64]) {
    o.put(DataElement::new(
        tag,
        VR::FD,
        PrimitiveValue::F64(C::from_vec(vals.to_vec())),
    ));
}

fn put_of(o: &mut InMemDicomObject, tag: Tag, vals: Vec<f32>) {
    o.put(DataElement::new(
        tag,
        VR::OF,
        PrimitiveValue::F32(C::from_vec(vals)),
    ));
}

/// An identity 4 × 4 matrix registration item — the pre- and post-
/// deformation slots of the IOD, which this writer never uses because the
/// grid it writes already carries the *total* displacement.
fn identity_matrix_item() -> InMemDicomObject {
    let mut m = InMemDicomObject::new_empty();
    put_str(&mut m, TAG_MATRIX_TYPE, VR::CS, "RIGID");
    put_ds(
        &mut m,
        tags::FRAME_OF_REFERENCE_TRANSFORMATION_MATRIX,
        &[
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ],
    );
    let mut item = InMemDicomObject::new_empty();
    put_seq(&mut item, TAG_MATRIX_SEQ, vec![m]);
    item
}

/// Everything a Deformable Spatial Registration needs besides the field.
pub struct DvfExport<'a> {
    /// Frame of Reference the field's own lattice lives in — the *fixed*
    /// dataset, since that is the domain a recovered transform is
    /// parameterized on.
    pub source_for_uid: &'a str,
    /// Frame of Reference the displacements point into — the *moving*
    /// dataset.
    pub target_for_uid: &'a str,
    pub study_uid: &'a str,
    pub patient_name: &'a str,
    pub patient_id: &'a str,
    /// Content label, e.g. the method that produced it.
    pub label: &'a str,
    pub description: &'a str,
}

/// Write a recovered deformation as a DICOM Deformable Spatial Registration.
///
/// The IOD applies its grid *after* a pre-deformation matrix and *before* a
/// post-deformation one; both are written as the identity here and the grid
/// carries the whole mapping, `T(p) − p`, exactly as
/// [`crate::registration::VectorField`] holds it. That is the least
/// surprising thing to hand another system: the file says what the transform
/// does with no composition rule to get wrong.
pub fn write_deformable_registration(
    path: &Path,
    field: &crate::registration::VectorField,
    meta: &DvfExport,
) -> Result<()> {
    if field.is_empty() {
        anyhow::bail!("the vector field is empty");
    }
    let mut grid = InMemDicomObject::new_empty();
    put_ds(
        &mut grid,
        tags::IMAGE_POSITION_PATIENT,
        &[field.origin.x, field.origin.y, field.origin.z],
    );
    put_ds(
        &mut grid,
        tags::IMAGE_ORIENTATION_PATIENT,
        &[
            field.axes[0].x,
            field.axes[0].y,
            field.axes[0].z,
            field.axes[1].x,
            field.axes[1].y,
            field.axes[1].z,
        ],
    );
    put_ul(
        &mut grid,
        TAG_GRID_DIMENSIONS,
        &[
            field.dims[0] as u32,
            field.dims[1] as u32,
            field.dims[2] as u32,
        ],
    );
    put_fd(&mut grid, TAG_GRID_RESOLUTION, &field.spacing);
    // Column-fastest, then rows, then planes — the order the lattice is
    // already stored in, and the one the standard prescribes.
    let mut data = Vec::with_capacity(field.data.len() * 3);
    for v in &field.data {
        data.push(v.x as f32);
        data.push(v.y as f32);
        data.push(v.z as f32);
    }
    put_of(&mut grid, TAG_VECTOR_GRID_DATA, data);

    let mut reg = InMemDicomObject::new_empty();
    put_str(
        &mut reg,
        TAG_SOURCE_FRAME_OF_REFERENCE_UID,
        VR::UI,
        meta.source_for_uid,
    );
    put_seq(
        &mut reg,
        TAG_PRE_DEFORMATION_MATRIX_SEQ,
        vec![identity_matrix_item()],
    );
    put_seq(
        &mut reg,
        TAG_POST_DEFORMATION_MATRIX_SEQ,
        vec![identity_matrix_item()],
    );
    put_seq(&mut reg, TAG_DEFORMABLE_REGISTRATION_GRID_SEQ, vec![grid]);

    let mut o = InMemDicomObject::new_empty();
    put_str(&mut o, tags::SPECIFIC_CHARACTER_SET, VR::CS, "ISO_IR 100");
    put_str(&mut o, tags::SOP_CLASS_UID, VR::UI, SOP_DEFORMABLE_REG);
    put_str(&mut o, tags::SOP_INSTANCE_UID, VR::UI, new_uid());
    put_str(&mut o, tags::MODALITY, VR::CS, "REG");
    put_str(&mut o, tags::PATIENT_NAME, VR::PN, meta.patient_name);
    put_str(&mut o, tags::PATIENT_ID, VR::LO, meta.patient_id);
    put_str(&mut o, tags::STUDY_INSTANCE_UID, VR::UI, meta.study_uid);
    put_str(&mut o, tags::SERIES_INSTANCE_UID, VR::UI, new_uid());
    put_is(&mut o, tags::SERIES_NUMBER, 1);
    put_is(&mut o, tags::INSTANCE_NUMBER, 1);
    // The Frame of Reference of the *instance* is where the displacements
    // point: the moving dataset.
    put_str(
        &mut o,
        tags::FRAME_OF_REFERENCE_UID,
        VR::UI,
        meta.target_for_uid,
    );
    put_str(&mut o, tags::CONTENT_LABEL, VR::CS, sanitize_cs(meta.label));
    put_str(&mut o, tags::CONTENT_DESCRIPTION, VR::LO, meta.description);
    put_str(&mut o, tags::SERIES_DESCRIPTION, VR::LO, meta.description);
    let (date, time) = today();
    put_str(&mut o, tags::CONTENT_DATE, VR::DA, date.clone());
    put_str(&mut o, tags::CONTENT_TIME, VR::TM, time.clone());
    put_str(&mut o, tags::SERIES_DATE, VR::DA, date);
    put_str(&mut o, tags::SERIES_TIME, VR::TM, time);
    put_seq(&mut o, TAG_DEFORMABLE_REGISTRATION_SEQ, vec![reg]);

    write_object(o, SOP_DEFORMABLE_REG, path)
}

/// A Content Label is CS: uppercase, 16 characters, a restricted alphabet.
fn sanitize_cs(s: &str) -> String {
    let mut out: String = s
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect();
    out.truncate(16);
    if out.is_empty() {
        out.push_str("REGISTRATION");
    }
    out
}

/// Write only the objects this application produces — RT structure sets and
/// DICOM Segmentation series — into `dir`, keeping the study and frame of
/// reference each already belongs to.
///
/// This is the other half of [`export_study`], and the difference is the
/// whole point: a full export invents a new study so the result stands on
/// its own, whereas contours and segments drawn on a study that already
/// exists must attach *to that study*. Fresh SOP Instance UIDs, original
/// Study Instance UID and Frame of Reference UID — which is exactly what
/// sending derived objects back to an archive means.
///
/// Returns the number of files written.
pub fn export_derived(
    study: &LoadedStudy,
    dir: &Path,
    params: &ExportParams,
    progress: &Progress,
) -> Result<usize> {
    std::fs::create_dir_all(dir).with_context(|| format!("create directory {}", dir.display()))?;
    let (today_date, today_time) = today();
    let vol_for = study.volume.frame_of_reference_uid.clone();
    // The study an object belongs to, from the object itself where it says
    // so and from the series it references otherwise.
    let study_of = |own: &str, referenced: &str| -> String {
        if !own.is_empty() {
            return own.to_string();
        }
        study
            .series
            .iter()
            .find(|se| se.uid == referenced)
            .map(|se| se.study_uid.clone())
            .or_else(|| study.series.first().map(|se| se.study_uid.clone()))
            .unwrap_or_default()
    };
    let mut n_files = 0usize;

    for (si, ss) in study.structure_sets.iter().enumerate() {
        if ss.rois.is_empty() {
            continue;
        }
        progress.set(format!(
            "Writing RTSTRUCT {}/{}",
            si + 1,
            study.structure_sets.len()
        ));
        let ctx = Ctx {
            study_uid: study_of(&ss.study_uid, &ss.referenced_series_uid),
            for_uid: if ss.frame_of_reference_uid.is_empty() {
                vol_for.clone()
            } else {
                ss.frame_of_reference_uid.clone()
            },
            date: today_date.clone(),
            time: today_time.clone(),
            params,
        };
        let o = build_rtstruct(ss, &ctx, 2 + si as i64, &new_uid());
        write_object(o, SOP_RTSTRUCT, &dir.join(format!("RS_derived_{si}.dcm")))?;
        n_files += 1;
    }

    for (gi, ser) in study.seg_series.iter().enumerate() {
        if ser.segs.iter().all(|s| s.count == 0) {
            continue;
        }
        progress.set(format!("Writing SEG {}/{}", gi + 1, study.seg_series.len()));
        let study_uid = study_of(&ser.study_uid, &ser.referenced_series_uid);
        let for_uid = if ser.grid.frame_of_reference_uid.is_empty() {
            vol_for.clone()
        } else {
            ser.grid.frame_of_reference_uid.clone()
        };
        let seg_ctx = dicomseg::SegWriteCtx {
            study_uid: &study_uid,
            for_uid: &for_uid,
            date: &today_date,
            time: &today_time,
            series_number: 20 + gi as i64,
            // The image series it was drawn on is already in the archive, so
            // naming it is a real cross-reference rather than a claim about
            // files written beside it.
            image_series_uid: &ser.referenced_series_uid,
            image_sop_uids: &[],
            params,
        };
        dicomseg::write(ser, &seg_ctx, &dir.join(format!("SEG_derived_{gi}.dcm")))?;
        n_files += 1;
    }
    Ok(n_files)
}
