//! Directory scanning, DICOM classification and volume reconstruction.
//!
//! The loader makes two passes: a fast header-only scan (stops before Pixel
//! Data) to classify every file, then a parallel full read of the selected
//! image series. RT objects (RTSTRUCT / RTDOSE / RTPLAN) are parsed by their
//! dedicated modules.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{bail, Context, Result};
use dicom_dictionary_std::tags;
use dicom_object::{InMemDicomObject, OpenFileOptions};
use dicom_pixeldata::{ConvertOptions, ModalityLutOption, PixelDecoder};
use rayon::prelude::*;

use crate::extras::{self, PlanarImage, SpatialReg, TreatRecord};
use crate::geometry::Vec3;
use crate::rtdose::{self, DoseGrid};
use crate::rtplan::{self, PlanInfo};
use crate::rtstruct::{self, StructureSet};
use crate::volume::Volume;

/// Shared progress indicator for the background loading thread.
#[derive(Default)]
pub struct Progress {
    msg: Mutex<String>,
}

impl Progress {
    pub fn set(&self, s: impl Into<String>) {
        *self.msg.lock().unwrap() = s.into();
    }
    pub fn get(&self) -> String {
        self.msg.lock().unwrap().clone()
    }
}

// ---------------------------------------------------------------------------
// Safe element extraction helpers (missing/malformed tags never panic).
// ---------------------------------------------------------------------------

pub fn str_of(obj: &InMemDicomObject, tag: dicom_core::Tag) -> Option<String> {
    obj.element(tag)
        .ok()
        .and_then(|e| e.to_str().ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

pub fn f64_of(obj: &InMemDicomObject, tag: dicom_core::Tag) -> Option<f64> {
    obj.element(tag).ok().and_then(|e| e.to_float64().ok())
}

pub fn f64s_of(obj: &InMemDicomObject, tag: dicom_core::Tag) -> Option<Vec<f64>> {
    obj.element(tag)
        .ok()
        .and_then(|e| e.to_multi_float64().ok())
}

pub fn i32_of(obj: &InMemDicomObject, tag: dicom_core::Tag) -> Option<i32> {
    obj.element(tag).ok().and_then(|e| e.to_int::<i32>().ok())
}

pub fn items_of<'a>(
    obj: &'a InMemDicomObject,
    tag: dicom_core::Tag,
) -> Option<&'a [InMemDicomObject]> {
    obj.element(tag).ok().and_then(|e| e.items()).map(|v| &v[..])
}

// ---------------------------------------------------------------------------
// Scan results
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct SeriesInfo {
    pub uid: String,
    pub modality: String,
    pub description: String,
    /// Patient this series belongs to (for the patient ▶ study ▶ series tree).
    pub patient_id: String,
    pub patient_name: String,
    /// Study this series belongs to (for the study/series tree).
    pub study_uid: String,
    pub study_date: String,
    pub study_description: String,
    pub files: Vec<PathBuf>,
}

impl SeriesInfo {
    /// Grouping key for the patient level of the data tree.
    pub fn patient_key(&self) -> &str {
        if !self.patient_id.is_empty() {
            &self.patient_id
        } else if !self.patient_name.is_empty() {
            &self.patient_name
        } else {
            "?"
        }
    }
}

#[derive(Default, Clone)]
pub struct PatientMeta {
    pub patient_name: String,
    pub patient_id: String,
    pub study_date: String,
    pub study_description: String,
}

/// Everything found in a directory, with the primary series volume loaded.
#[derive(Clone)]
pub struct LoadedStudy {
    pub meta: PatientMeta,
    pub series: Vec<SeriesInfo>,
    pub active_series: usize,
    pub volume: Volume,
    /// All RT Structure Sets found in the folder (e.g. one per 4DCT phase).
    /// The application selects the active one per study slot.
    pub structure_sets: Vec<StructureSet>,
    pub doses: Vec<DoseGrid>,
    pub plans: Vec<PlanInfo>,
    /// DX / CR radiographs and RTIMAGE (DRR / portal) planar images.
    pub planar_images: Vec<PlanarImage>,
    /// REG spatial registration objects found in the folder.
    pub registrations: Vec<SpatialReg>,
    /// RT (Ion) Beams Treatment Records.
    pub treat_records: Vec<TreatRecord>,
    pub warnings: Vec<String>,
    pub default_window: (f32, f32),
}

const IMAGE_MODALITIES: &[&str] = &["CT", "MR", "PT", "NM", "US", "OT"];
/// Modalities treated as 2D projection images (no volume reconstruction).
const PLANAR_MODALITIES: &[&str] = &["DX", "CR", "RTIMAGE", "MG", "XA", "RF", "PX"];
const SOP_RTIMAGE: &str = "1.2.840.10008.5.1.4.1.1.481.1";
const SOP_DX_PRESENTATION: &str = "1.2.840.10008.5.1.4.1.1.1.1";
const SOP_DX_PROCESSING: &str = "1.2.840.10008.5.1.4.1.1.1.1.1";

/// SOP Class UID fallbacks for RT objects (when Modality is absent).
const SOP_RTSTRUCT: &str = "1.2.840.10008.5.1.4.1.1.481.3";
const SOP_RTDOSE: &str = "1.2.840.10008.5.1.4.1.1.481.2";
const SOP_RTPLAN: &str = "1.2.840.10008.5.1.4.1.1.481.5";
const SOP_RTIONPLAN: &str = "1.2.840.10008.5.1.4.1.1.481.8";

pub fn load_directory(dir: &Path, progress: &Progress) -> Result<LoadedStudy> {
    progress.set("Scanning directory…");

    let files: Vec<PathBuf> = walkdir::WalkDir::new(dir)
        .follow_links(true)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .map(|e| e.into_path())
        .collect();

    if files.is_empty() {
        bail!("No files found in {}", dir.display());
    }
    progress.set(format!("Reading headers of {} files…", files.len()));

    // Parallel header-only scan.
    struct Scanned {
        path: PathBuf,
        modality: String,
        sop_class: String,
        series_uid: String,
        series_desc: String,
        study_uid: String,
        has_geometry: bool,
        meta: PatientMeta,
    }

    let scanned: Vec<Scanned> = files
        .par_iter()
        .filter_map(|path| {
            let obj = OpenFileOptions::new()
                .read_until(tags::PIXEL_DATA)
                .open_file(path)
                .ok()?;
            let modality = str_of(&obj, tags::MODALITY).unwrap_or_default();
            let sop_class = str_of(&obj, tags::SOP_CLASS_UID).unwrap_or_default();
            let series_uid = str_of(&obj, tags::SERIES_INSTANCE_UID).unwrap_or_default();
            let series_desc = str_of(&obj, tags::SERIES_DESCRIPTION).unwrap_or_default();
            let study_uid = str_of(&obj, tags::STUDY_INSTANCE_UID).unwrap_or_default();
            let has_geometry = obj.element(tags::IMAGE_POSITION_PATIENT).is_ok()
                && obj.element(tags::ROWS).is_ok();
            let meta = PatientMeta {
                patient_name: str_of(&obj, tags::PATIENT_NAME).unwrap_or_default(),
                patient_id: str_of(&obj, tags::PATIENT_ID).unwrap_or_default(),
                study_date: str_of(&obj, tags::STUDY_DATE).unwrap_or_default(),
                study_description: str_of(&obj, tags::STUDY_DESCRIPTION).unwrap_or_default(),
            };
            Some(Scanned {
                path: path.clone(),
                modality,
                sop_class,
                series_uid,
                series_desc,
                study_uid,
                has_geometry,
                meta,
            })
        })
        .collect();

    let mut warnings = Vec::new();
    let unreadable = files.len() - scanned.len();
    if unreadable > 0 {
        warnings.push(format!("{unreadable} file(s) were not readable as DICOM and were skipped"));
    }
    if scanned.is_empty() {
        bail!("No DICOM files found in {}", dir.display());
    }

    // Classify.
    let mut image_series: Vec<SeriesInfo> = Vec::new();
    let mut rtstruct_files = Vec::new();
    let mut rtdose_files = Vec::new();
    let mut rtplan_files = Vec::new();
    let mut planar_files = Vec::new();
    let mut reg_files = Vec::new();
    let mut record_files = Vec::new();
    let mut meta = PatientMeta::default();

    for s in &scanned {
        if meta.patient_id.is_empty() && !s.meta.patient_id.is_empty() {
            meta = s.meta.clone();
        }
        let is_rt_struct = s.modality == "RTSTRUCT" || s.sop_class == SOP_RTSTRUCT;
        let is_rt_dose = s.modality == "RTDOSE" || s.sop_class == SOP_RTDOSE;
        let is_rt_plan = s.modality == "RTPLAN"
            || s.modality == "RTIONPLAN"
            || s.sop_class == SOP_RTPLAN
            || s.sop_class == SOP_RTIONPLAN;
        let is_reg = s.modality == "REG" || extras::is_reg_sop(&s.sop_class);
        let is_record = s.modality == "RTRECORD" || extras::is_record_sop(&s.sop_class);
        let is_planar = PLANAR_MODALITIES.contains(&s.modality.as_str())
            || s.sop_class == SOP_RTIMAGE
            || s.sop_class == SOP_DX_PRESENTATION
            || s.sop_class == SOP_DX_PROCESSING;

        if is_rt_struct {
            rtstruct_files.push(s.path.clone());
        } else if is_rt_dose {
            rtdose_files.push(s.path.clone());
        } else if is_rt_plan {
            rtplan_files.push(s.path.clone());
        } else if is_reg {
            reg_files.push(s.path.clone());
        } else if is_record {
            record_files.push(s.path.clone());
        } else if is_planar && !s.has_geometry {
            planar_files.push(s.path.clone());
        } else if s.has_geometry
            && (IMAGE_MODALITIES.contains(&s.modality.as_str())
                || PLANAR_MODALITIES.contains(&s.modality.as_str())
                || s.modality.is_empty())
        {
            match image_series.iter_mut().find(|se| se.uid == s.series_uid) {
                Some(se) => se.files.push(s.path.clone()),
                None => image_series.push(SeriesInfo {
                    uid: s.series_uid.clone(),
                    modality: s.modality.clone(),
                    description: s.series_desc.clone(),
                    patient_id: s.meta.patient_id.clone(),
                    patient_name: s.meta.patient_name.clone(),
                    study_uid: s.study_uid.clone(),
                    study_date: s.meta.study_date.clone(),
                    study_description: s.meta.study_description.clone(),
                    files: vec![s.path.clone()],
                }),
            }
        }
    }

    if image_series.is_empty() {
        bail!("No image series (CT/MR/…) with geometry found — cannot build a volume");
    }

    // Default to the series with the most slices (typically the planning CT).
    image_series.sort_by_key(|s| std::cmp::Reverse(s.files.len()));
    let active_series = 0;

    let (volume, default_window, mut vol_warnings) =
        load_series_volume(&image_series[active_series], progress)?;
    warnings.append(&mut vol_warnings);

    // RT objects — every structure set is loaded (e.g. one per 4DCT phase);
    // the application chooses which one is active.
    let mut structure_sets = Vec::new();
    for p in &rtstruct_files {
        progress.set("Parsing RT Structure Sets…");
        match rtstruct::load(p) {
            Ok(ss) => structure_sets.push(ss),
            Err(e) => warnings.push(format!(
                "RTSTRUCT {} load failed: {e:#}",
                p.file_name().unwrap_or_default().to_string_lossy()
            )),
        }
    }

    let mut doses = Vec::new();
    for p in &rtdose_files {
        progress.set("Parsing RT Dose…");
        match rtdose::load(p) {
            Ok(d) => doses.push(d),
            Err(e) => warnings.push(format!(
                "RTDOSE {} load failed: {e:#}",
                p.file_name().unwrap_or_default().to_string_lossy()
            )),
        }
    }

    let mut plans = Vec::new();
    for p in &rtplan_files {
        progress.set("Parsing RT Plan…");
        match rtplan::load(p) {
            Ok(pl) => plans.push(pl),
            Err(e) => warnings.push(format!(
                "RTPLAN {} load failed: {e:#}",
                p.file_name().unwrap_or_default().to_string_lossy()
            )),
        }
    }

    let mut planar_images = Vec::new();
    for p in &planar_files {
        progress.set("Loading planar images (DX/RTIMAGE)…");
        match extras::load_planar(p) {
            Ok(img) => planar_images.push(img),
            Err(e) => warnings.push(format!(
                "planar image {} load failed: {e:#}",
                p.file_name().unwrap_or_default().to_string_lossy()
            )),
        }
    }

    let mut registrations = Vec::new();
    for p in &reg_files {
        progress.set("Parsing spatial registrations (REG)…");
        match extras::load_reg(p) {
            Ok(r) => {
                if r.deformable {
                    warnings.push(format!(
                        "REG {} is a deformable registration — only its rigid matrices are read",
                        p.file_name().unwrap_or_default().to_string_lossy()
                    ));
                }
                registrations.push(r);
            }
            Err(e) => warnings.push(format!(
                "REG {} load failed: {e:#}",
                p.file_name().unwrap_or_default().to_string_lossy()
            )),
        }
    }

    let mut treat_records = Vec::new();
    for p in &record_files {
        progress.set("Parsing treatment records…");
        match extras::load_record(p) {
            Ok(r) => treat_records.push(r),
            Err(e) => warnings.push(format!(
                "RTRECORD {} load failed: {e:#}",
                p.file_name().unwrap_or_default().to_string_lossy()
            )),
        }
    }

    // Frame-of-reference sanity checks.
    for ss in &structure_sets {
        if !ss.frame_of_reference_uid.is_empty()
            && !volume.frame_of_reference_uid.is_empty()
            && ss.frame_of_reference_uid != volume.frame_of_reference_uid
        {
            warnings.push(format!(
                "RTSTRUCT {} frame of reference differs from the image volume — contours may be misaligned",
                ss.file_name
            ));
        }
    }
    for d in &doses {
        if !d.frame_of_reference_uid.is_empty()
            && !volume.frame_of_reference_uid.is_empty()
            && d.frame_of_reference_uid != volume.frame_of_reference_uid
        {
            warnings.push(
                "RTDOSE frame of reference differs from the image volume — overlay may be misaligned"
                    .into(),
            );
        }
    }

    Ok(LoadedStudy {
        meta,
        series: image_series,
        active_series,
        volume,
        structure_sets,
        doses,
        plans,
        planar_images,
        registrations,
        treat_records,
        warnings,
        default_window,
    })
}

/// Fully load one image series into a `Volume` (parallel decode).
pub fn load_series_volume(
    series: &SeriesInfo,
    progress: &Progress,
) -> Result<(Volume, (f32, f32), Vec<String>)> {
    progress.set(format!(
        "Loading {} slice(s) of series {}…",
        series.files.len(),
        if series.description.is_empty() { &series.modality } else { &series.description }
    ));

    struct SliceRec {
        pos: Vec3,
        proj: f64,
        rows: usize,
        cols: usize,
        row_dir: Vec3,
        col_dir: Vec3,
        spacing: [f64; 2],
        for_uid: String,
        window: Option<(f32, f32)>,
        thickness: Option<f64>,
        data: Vec<i16>,
    }

    let opts = ConvertOptions::new().with_modality_lut(ModalityLutOption::Default);

    let results: Vec<Result<SliceRec>> = series
        .files
        .par_iter()
        .map(|path| -> Result<SliceRec> {
            let obj = dicom_object::open_file(path)
                .with_context(|| format!("open {}", path.display()))?;

            let ipp = f64s_of(&obj, tags::IMAGE_POSITION_PATIENT)
                .filter(|v| v.len() >= 3)
                .with_context(|| format!("missing ImagePositionPatient in {}", path.display()))?;
            let iop = f64s_of(&obj, tags::IMAGE_ORIENTATION_PATIENT)
                .filter(|v| v.len() >= 6)
                .unwrap_or_else(|| vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
            let ps = f64s_of(&obj, tags::PIXEL_SPACING)
                .filter(|v| v.len() >= 2)
                .unwrap_or_else(|| vec![1.0, 1.0]);

            let row_dir = Vec3::from_slice(&iop[0..3]).normalized();
            let col_dir = Vec3::from_slice(&iop[3..6]).normalized();
            let normal = row_dir.cross(col_dir).normalized();
            let pos = Vec3::from_slice(&ipp);

            let window = match (
                f64s_of(&obj, tags::WINDOW_CENTER).and_then(|v| v.first().copied()),
                f64s_of(&obj, tags::WINDOW_WIDTH).and_then(|v| v.first().copied()),
            ) {
                (Some(c), Some(w)) if w > 1.0 => Some((c as f32, w as f32)),
                _ => None,
            };

            let decoded = obj
                .decode_pixel_data()
                .with_context(|| format!("decode pixel data of {}", path.display()))?;
            let rows = decoded.rows() as usize;
            let cols = decoded.columns() as usize;
            if decoded.number_of_frames() > 1 {
                bail!(
                    "multi-frame image {} not supported as part of a series",
                    path.display()
                );
            }
            let f: Vec<f32> = decoded
                .to_vec_with_options(&opts)
                .with_context(|| format!("convert pixels of {}", path.display()))?;
            if f.len() < rows * cols {
                bail!("pixel buffer smaller than Rows×Columns in {}", path.display());
            }
            let data: Vec<i16> = f[..rows * cols]
                .iter()
                .map(|&v| v.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16)
                .collect();

            Ok(SliceRec {
                pos,
                proj: pos.dot(normal),
                rows,
                cols,
                row_dir,
                col_dir,
                spacing: [ps[1], ps[0]], // [along i (columns), along j (rows)]
                for_uid: str_of(&obj, tags::FRAME_OF_REFERENCE_UID).unwrap_or_default(),
                window,
                thickness: f64_of(&obj, tags::SLICE_THICKNESS),
                data,
            })
        })
        .collect();

    let mut warnings = Vec::new();
    let mut slices: Vec<SliceRec> = Vec::with_capacity(results.len());
    for r in results {
        match r {
            Ok(s) => slices.push(s),
            Err(e) => warnings.push(format!("slice skipped: {e:#}")),
        }
    }
    if slices.is_empty() {
        bail!("No slices of the series could be decoded");
    }

    // Consistent in-plane dimensions.
    let (rows, cols) = (slices[0].rows, slices[0].cols);
    let before = slices.len();
    slices.retain(|s| s.rows == rows && s.cols == cols);
    if slices.len() != before {
        warnings.push(format!(
            "{} slice(s) with mismatched dimensions were dropped",
            before - slices.len()
        ));
    }

    // Sort along the slice normal and drop duplicates.
    slices.sort_by(|a, b| a.proj.partial_cmp(&b.proj).unwrap_or(std::cmp::Ordering::Equal));
    slices.dedup_by(|a, b| (a.proj - b.proj).abs() < 0.01);

    let nz = slices.len();
    let slice_spacing = if nz > 1 {
        let mut diffs: Vec<f64> = slices.windows(2).map(|w| w[1].proj - w[0].proj).collect();
        diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = diffs[diffs.len() / 2];
        let max_dev = diffs
            .iter()
            .map(|d| (d - median).abs())
            .fold(0.0_f64, f64::max);
        if median > 1e-6 && max_dev / median > 0.01 {
            warnings.push(format!(
                "Non-uniform slice spacing (median {:.3} mm, max deviation {:.3} mm) — using median",
                median, max_dev
            ));
        }
        median.max(1e-6)
    } else {
        slices[0].thickness.unwrap_or(1.0).max(1e-6)
    };

    let nx = cols;
    let ny = rows;
    let mut data = Vec::with_capacity(nx * ny * nz);
    let mut min_v = i16::MAX;
    let mut max_v = i16::MIN;
    for s in &slices {
        for &v in &s.data {
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }
        data.extend_from_slice(&s.data);
    }

    let first = &slices[0];
    let volume = Volume {
        data,
        dims: [nx, ny, nz],
        spacing: [first.spacing[0], first.spacing[1], slice_spacing],
        origin: first.pos,
        row_dir: first.row_dir,
        col_dir: first.col_dir,
        normal: first.row_dir.cross(first.col_dir).normalized(),
        frame_of_reference_uid: first.for_uid.clone(),
        min_value: min_v,
        max_value: max_v,
    };

    let default_window = first
        .window
        .unwrap_or_else(|| default_window_for(&series.modality, min_v, max_v));

    Ok((volume, default_window, warnings))
}

/// Merge `src` into `dest` (used by *File ▶ Add folder* and the tree
/// copy/move actions). Series and RT objects already present in `dest`
/// (same UID) are skipped; the displayed volume and all active selections
/// of `dest` are left untouched. Returns human-readable notes about
/// anything that was skipped.
pub fn merge_study(dest: &mut LoadedStudy, src: LoadedStudy) -> Vec<String> {
    let mut notes = Vec::new();

    let mut skipped = 0usize;
    for s in src.series {
        if dest.series.iter().any(|d| d.uid == s.uid) {
            skipped += 1;
        } else {
            dest.series.push(s);
        }
    }
    if skipped > 0 {
        notes.push(format!("{skipped} series were already present and were not added again"));
    }

    for ss in src.structure_sets {
        let dup = dest.structure_sets.iter().any(|d| {
            (!ss.sop_instance_uid.is_empty() && d.sop_instance_uid == ss.sop_instance_uid)
                || (ss.sop_instance_uid.is_empty()
                    && !ss.file_name.is_empty()
                    && d.file_name == ss.file_name)
        });
        if !dup {
            dest.structure_sets.push(ss);
        }
    }
    for p in src.plans {
        let dup = !p.sop_instance_uid.is_empty()
            && dest.plans.iter().any(|d| d.sop_instance_uid == p.sop_instance_uid);
        if !dup {
            dest.plans.push(p);
        }
    }
    for d in src.doses {
        let dup = dest.doses.iter().any(|e| {
            e.dims == d.dims
                && e.referenced_plan_uid == d.referenced_plan_uid
                && e.study_uid == d.study_uid
                && e.label == d.label
                && e.max_dose == d.max_dose
        });
        if !dup {
            dest.doses.push(d);
        }
    }
    for img in src.planar_images {
        let dup = dest
            .planar_images
            .iter()
            .any(|e| e.label == img.label && e.rows == img.rows && e.cols == img.cols);
        if !dup {
            dest.planar_images.push(img);
        }
    }
    for r in src.registrations {
        if !dest.registrations.iter().any(|e| e.label == r.label) {
            dest.registrations.push(r);
        }
    }
    for r in src.treat_records {
        if !dest
            .treat_records
            .iter()
            .any(|e| e.label == r.label && e.date == r.date)
        {
            dest.treat_records.push(r);
        }
    }
    dest.warnings.extend(src.warnings);
    notes
}

fn default_window_for(modality: &str, min_v: i16, max_v: i16) -> (f32, f32) {
    match modality {
        "CT" => (40.0, 400.0),
        _ => {
            let c = (min_v as f32 + max_v as f32) * 0.5;
            let w = (max_v as f32 - min_v as f32).max(1.0);
            (c, w)
        }
    }
}
