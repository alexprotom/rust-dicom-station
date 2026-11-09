//! Planar images: 2D contours (projected from the CT's structures, drawn
//! as polygons, carried by a 2D/2D registration), the presentation state
//! they are saved as, 2D/2D registration, and a setup read from two 2D
//! matches. Images are named by number or label as describe_dataset lists
//! them; planar structure sets and 2D registrations by number or label.

use anyhow::{anyhow, bail, Result};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{json, Value};

use super::super::phi::clean_text;
use super::super::session::Setup;
use super::super::Core;
use crate::contours::Poly;
use crate::geometry::Vec3;
use crate::loader::LoadedStudy;
use crate::planar_structs::{self, ContourSource, PlanarStructureSet, PoseChoice, ProjectRequest};
use crate::progress::Progress;
use crate::registration::metric2d::Metric2d;
use crate::registration::planar::{
    self as p2, DeformMetric, Dof2d, Method2d, Optimizer2d, Planar2d, PlanarReg, Request2d,
};

/// A planar image of the study by number (1-based) or label.
fn image_index(study: &LoadedStudy, s: &str) -> Result<usize> {
    let s = s.trim();
    if let Ok(n) = s.parse::<usize>() {
        if n >= 1 && n <= study.planar_images.len() {
            return Ok(n - 1);
        }
        bail!(
            "no planar image {n} (the dataset has {})",
            study.planar_images.len()
        );
    }
    study
        .planar_images
        .iter()
        .position(|p| p.label == s)
        .ok_or_else(|| anyhow!("no planar image labelled '{}'", clean_text(s)))
}

/// A planar structure set by number (1-based) or label.
fn set_index(study: &LoadedStudy, s: &str) -> Result<usize> {
    let s = s.trim();
    if let Ok(n) = s.parse::<usize>() {
        if n >= 1 && n <= study.planar_structs.len() {
            return Ok(n - 1);
        }
    }
    study
        .planar_structs
        .iter()
        .position(|p| p.label == s)
        .ok_or_else(|| anyhow!("no planar structure set '{}'", clean_text(s)))
}

/// A 2D registration by number (1-based) or label.
fn reg_index(study: &LoadedStudy, s: &str) -> Result<usize> {
    let s = s.trim();
    if let Ok(n) = s.parse::<usize>() {
        if n >= 1 && n <= study.planar_regs.len() {
            return Ok(n - 1);
        }
    }
    study
        .planar_regs
        .iter()
        .position(|p| p.label == s)
        .ok_or_else(|| anyhow!("no 2D registration '{}'", clean_text(s)))
}

/// A set as the client sees it: per structure and image, the rings, the
/// area and the box; no UID, images by number.
fn set_json(study: &LoadedStudy, k: usize) -> Value {
    let set = &study.planar_structs[k];
    let image_ref = |uid: &str| -> Value {
        match study.planar_by_uid(uid) {
            Some(i) => json!({"number": i + 1, "label": clean_text(&study.planar_images[i].label)}),
            None => json!({"loaded": false}),
        }
    };
    let structures: Vec<Value> = set
        .structures
        .iter()
        .map(|s| {
            let contours: Vec<Value> = s
                .contours
                .iter()
                .map(|c| {
                    let spacing = study
                        .planar_by_uid(&c.image_uid)
                        .map(|i| study.planar_images[i].spacing)
                        .unwrap_or([1.0, 1.0]);
                    let b = c.region.bbox();
                    json!({
                        "image": image_ref(&c.image_uid),
                        "rings": c.region.rings.len(),
                        "area_mm2": (planar_structs::area_mm2(&c.region, spacing) * 10.0).round() / 10.0,
                        "bbox_px": if c.region.is_empty() { Value::Null } else { json!([b[0], b[1], b[2], b[3]]) },
                        "source": clean_text(&c.source.describe()),
                    })
                })
                .collect();
            json!({
                "name": clean_text(&s.name),
                "color": s.color,
                "contours": contours,
                "points": s.points.len(),
            })
        })
        .collect();
    json!({
        "number": k + 1,
        "label": clean_text(&set.label),
        "locked": set.locked,
        "from_file": set.source_path.is_some(),
        "images": set.images.iter().map(|u| image_ref(u)).collect::<Vec<_>>(),
        "structures": structures,
    })
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ProjectArgs {
    pub dataset: String,
    /// The CT series (number); the displayed one when omitted.
    #[serde(default)]
    pub series: Option<u32>,
    /// Structure names.
    pub structures: Vec<String>,
    /// Planar images by number or label.
    pub images: Vec<String>,
    /// `rendered` / `plan` (the images' own cameras, the default) or a
    /// `setup` handle from register_2d3d or setup_from_planar (the
    /// structures where that setup found the patient).
    #[serde(default)]
    pub pose: Option<String>,
    /// The set's label (after the pose when omitted); an unlocked set of
    /// that label takes the contours.
    #[serde(default)]
    pub set: Option<String>,
}

pub fn project_structures(core: &mut Core, a: ProjectArgs, p: &Progress) -> Result<Value> {
    let ds = core.session.dataset(&a.dataset)?;
    let idx = core.session.series_index(ds, a.series)?;
    let volume = core.session.volume(&a.dataset, idx, p)?;
    let setup = match a.pose.as_deref().map(str::trim) {
        None | Some("") | Some("rendered") | Some("plan") => None,
        Some(h) => Some(core.session.setup(h)?.result.clone()),
    };
    let ds = core.session.dataset(&a.dataset)?;
    let mut study = planar_structs::projection_view(&ds.study);
    study.volume = volume;
    study.active_series = idx;
    let pose = match setup {
        None => PoseChoice::AsIs,
        Some(s) => {
            study.setups.push(s);
            PoseChoice::Setup(study.setups.len() - 1)
        }
    };
    let images = a
        .images
        .iter()
        .map(|s| image_index(&study, s))
        .collect::<Result<Vec<_>>>()?;
    let req = ProjectRequest {
        structures: a.structures.clone(),
        images,
        pose,
        label: a.set.clone().unwrap_or_default(),
    };
    let (set, lines) = planar_structs::project_many(&study, &req, p)?;
    let ds = core.session.dataset_mut(&a.dataset)?;
    let k = planar_structs::file_set(&mut ds.study, set);
    let mut out = set_json(&ds.study, k);
    out["empty"] = json!(lines
        .iter()
        .filter(|l| l.rings == 0)
        .map(|l| json!({"structure": clean_text(&l.structure), "image": l.image + 1}))
        .collect::<Vec<_>>());
    Ok(out)
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ListArgs {
    pub dataset: String,
}

pub fn list_planar_structures(core: &mut Core, a: ListArgs, _p: &Progress) -> Result<Value> {
    let ds = core.session.dataset(&a.dataset)?;
    let st = &ds.study;
    Ok(json!({
        "sets": (0..st.planar_structs.len()).map(|k| set_json(st, k)).collect::<Vec<_>>(),
        "registrations": st.planar_regs.iter().enumerate().map(|(k, r)| reg_json(st, k, r)).collect::<Vec<_>>(),
    }))
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DrawArgs {
    pub dataset: String,
    /// The planar image, by number or label.
    pub image: String,
    /// The set (number or label); a new set labelled after the image when
    /// omitted.
    #[serde(default)]
    pub set: Option<String>,
    /// The structure's name; added when the set has none of that name.
    pub name: String,
    /// `[r, g, b]` for a new structure.
    #[serde(default)]
    pub color: Option<[u8; 3]>,
    /// Polygons of `[column, row]` pixel positions, the first pixel's
    /// centre at `[0, 0]`.
    pub polygons: Vec<Vec<[f64; 2]>>,
    /// `new` (or `extend`, the default: added) or `subtract` (cut out).
    #[serde(default)]
    pub mode: Option<String>,
}

pub fn draw_planar_structure(core: &mut Core, a: DrawArgs, _p: &Progress) -> Result<Value> {
    let cut = match a.mode.as_deref().unwrap_or("extend") {
        "new" | "extend" | "add" => false,
        "subtract" | "cut" => true,
        other => bail!("mode '{other}': new, extend or subtract"),
    };
    if a.polygons.iter().all(|p| p.len() < 3) {
        bail!("polygons: at least one polygon of three points or more");
    }
    let ds = core.session.dataset_mut(&a.dataset)?;
    let st = &mut ds.study;
    let i = image_index(st, &a.image)?;
    let (uid, label, study_uid) = {
        let img = &st.planar_images[i];
        if img.sop_instance_uid.is_empty() {
            bail!("the image has no SOP Instance UID to attach a contour to");
        }
        (
            img.sop_instance_uid.clone(),
            img.label.clone(),
            img.study_uid.clone(),
        )
    };
    let k = match &a.set {
        Some(s) => set_index(st, s)?,
        None => {
            st.planar_structs
                .push(PlanarStructureSet::new(&label, &study_uid, "drawn"));
            st.planar_structs.len() - 1
        }
    };
    if st.planar_structs[k].locked {
        bail!("the set is locked");
    }
    let set = &mut st.planar_structs[k];
    let s = set.structure_entry(a.name.trim(), a.color);
    let c = s.contour_entry(&uid);
    for poly in a.polygons.iter().filter(|p| p.len() >= 3) {
        planar_structs::apply_ring(&mut c.region, &Poly::new(poly.clone()), cut);
    }
    c.source = ContourSource::Drawn;
    s.prune();
    set.refresh_images();
    Ok(set_json(st, k))
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RegisterPlanarArgs {
    pub dataset: String,
    /// The image the answer is shown on (today's radiograph).
    pub fixed: String,
    /// The image moved onto it (the reference, a DRR).
    pub moving: String,
    /// `intensity` (default), `points`, `points_then_intensity` or
    /// `deformable`.
    #[serde(default)]
    pub method: Option<String>,
    /// `ncc` (default), `gc`, `gd`, `pi`, `lncc` or `mi`.
    #[serde(default)]
    pub metric: Option<String>,
    /// `powell` (default) or `nelder-mead`.
    #[serde(default)]
    pub optimizer: Option<String>,
    /// `shift`, `shift_rotation` (default) or `similarity`.
    #[serde(default)]
    pub dof: Option<String>,
    #[serde(default)]
    pub levels: Option<usize>,
    /// The finest level's longest side (default 256).
    #[serde(default)]
    pub pixels: Option<usize>,
    #[serde(default)]
    pub range_mm: Option<f64>,
    #[serde(default)]
    pub range_deg: Option<f64>,
    #[serde(default)]
    pub refine: Option<bool>,
    /// Polygons on the fixed image (pixels) the measure is taken over.
    #[serde(default)]
    pub region: Option<Vec<Vec<[f64; 2]>>>,
    /// Pairs of points: `[[fixed col, row], [moving col, row]]`.
    #[serde(default)]
    pub pairs: Option<Vec<[[f64; 2]; 2]>>,
    /// Find markers of this size (mm) in both images and pair them.
    #[serde(default)]
    pub detect_markers_mm: Option<f64>,
    /// The B-spline's node spacing at the first and the last level, mm.
    #[serde(default)]
    pub grid_mm: Option<f64>,
    #[serde(default)]
    pub grid_min_mm: Option<f64>,
    /// The bending energy's weight (0.01 for msd, 0.3 for mi by default).
    #[serde(default)]
    pub lambda: Option<f64>,
    /// `msd` (default) or `mi`.
    #[serde(default)]
    pub deformable_metric: Option<String>,
    /// File the moving image moved onto the fixed one as a planar image.
    #[serde(default)]
    pub file_copy: Option<bool>,
}

fn key<T>(v: &Option<String>, what: &str, f: impl Fn(&str) -> Option<T>) -> Result<Option<T>> {
    match v.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        None => Ok(None),
        Some(s) => f(s)
            .map(Some)
            .ok_or_else(|| anyhow!("{what} '{s}' is not one of the choices")),
    }
}

fn reg_json(st: &LoadedStudy, k: usize, r: &PlanarReg) -> Value {
    let num = |uid: &str| st.planar_by_uid(uid).map(|i| i + 1);
    let mut j = r.result.json();
    j["registration"] = json!(k + 1);
    j["label"] = json!(clean_text(&r.label));
    j["fixed_image"] = json!(num(&r.fixed_uid));
    j["moving_image"] = json!(num(&r.moving_uid));
    j["moved_copy_image"] = json!(r.moved_copy_uid.as_deref().and_then(num));
    j
}

pub fn register_planar(core: &mut Core, a: RegisterPlanarArgs, p: &Progress) -> Result<Value> {
    let ds = core.session.dataset(&a.dataset)?;
    let st = &ds.study;
    let (fi, mi) = (image_index(st, &a.fixed)?, image_index(st, &a.moving)?);
    if fi == mi {
        bail!("the fixed and the moving image are the same");
    }
    let (f, m) = (&st.planar_images[fi], &st.planar_images[mi]);
    if f.sop_instance_uid.is_empty() || m.sop_instance_uid.is_empty() {
        bail!("an image has no SOP Instance UID");
    }
    let mut req = Request2d::new(Planar2d::of(f), Planar2d::of(m));
    if let Some(v) = key(&a.method, "method", Method2d::from_key)? {
        req.method = v;
    }
    if let Some(v) = key(&a.metric, "metric", Metric2d::from_key)? {
        req.metric = v;
    }
    if let Some(v) = key(&a.optimizer, "optimizer", Optimizer2d::from_key)? {
        req.optimizer = v;
    }
    if let Some(v) = key(&a.dof, "dof", Dof2d::from_key)? {
        req.dof = v;
    }
    if let Some(v) = key(
        &a.deformable_metric,
        "deformable_metric",
        DeformMetric::from_key,
    )? {
        req.deformable_metric = v;
        req.lambda = v.default_lambda();
    }
    if let Some(v) = a.levels {
        req.levels = v.clamp(1, 5);
    }
    if let Some(v) = a.pixels {
        req.max_pixels = v.clamp(32, 2048);
    }
    if let Some(v) = a.range_mm {
        req.range_mm = v.clamp(1.0, 150.0);
    }
    if let Some(v) = a.range_deg {
        req.range_deg = v.clamp(0.0, 45.0);
    }
    req.refine = a.refine.unwrap_or(false);
    if let Some(r) = &a.region {
        req.mask = Some(crate::setup::region_mask([f.cols, f.rows], r));
    }
    req.pairs = a.pairs.iter().flatten().map(|p| (p[0], p[1])).collect();
    req.detect_markers = a.detect_markers_mm.filter(|v| *v > 0.0);
    if let Some(v) = a.grid_mm {
        req.grid_mm = v.clamp(2.0, 256.0);
    }
    if let Some(v) = a.grid_min_mm {
        req.grid_min_mm = v.clamp(2.0, 256.0);
    }
    if let Some(v) = a.lambda {
        req.lambda = v.clamp(0.0, 100.0);
    }
    let (fixed_uid, moving_uid) = (f.sop_instance_uid.clone(), m.sop_instance_uid.clone());
    let label = format!("{} onto {}", m.label, f.label);
    p.set("Registering the planar images");
    let result = p2::register(&req, p)?;
    let ds = core.session.dataset_mut(&a.dataset)?;
    let st = &mut ds.study;
    let mut copy = None;
    if a.file_copy.unwrap_or(true) {
        let (f, m) = (&st.planar_images[fi], &st.planar_images[mi]);
        let img = p2::resample(&Planar2d::of(m), &result.transform, &Planar2d::of(f));
        let (lo, hi) = img
            .data
            .iter()
            .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
        let uid = crate::dicom_export::new_uid();
        let made = crate::extras::PlanarImage {
            label: label.clone(),
            modality: m.modality.clone(),
            rows: f.rows,
            cols: f.cols,
            spacing: f.spacing,
            data: img.data,
            min_value: lo,
            max_value: hi,
            window: m.window,
            sop_instance_uid: uid.clone(),
            series_instance_uid: crate::dicom_export::new_uid(),
            frame_of_reference_uid: f.frame_of_reference_uid.clone(),
            study_uid: f.study_uid.clone(),
            iec: f.iec,
            camera: f.camera,
            megavoltage: f.megavoltage,
            frame_count: 1,
            ..Default::default()
        };
        st.planar_images.push(made);
        copy = Some(uid);
    }
    st.planar_regs.push(PlanarReg {
        label,
        fixed_uid,
        moving_uid,
        result,
        moved_copy_uid: copy,
        when: crate::dicom_export::today().0,
    });
    let k = st.planar_regs.len() - 1;
    Ok(reg_json(st, k, &st.planar_regs[k]))
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CarryArgs {
    pub dataset: String,
    /// The 2D registration, by number or label.
    pub registration: String,
    /// Only the structures of this set (number or label).
    #[serde(default)]
    pub set: Option<String>,
}

pub fn carry_planar_structures(core: &mut Core, a: CarryArgs, _p: &Progress) -> Result<Value> {
    let ds = core.session.dataset_mut(&a.dataset)?;
    let st = &mut ds.study;
    let k = reg_index(st, &a.registration)?;
    let only = a.set.as_deref().map(|s| set_index(st, s)).transpose()?;
    let reg = st.planar_regs[k].clone();
    let (fi, mi) = (
        st.planar_by_uid(&reg.fixed_uid)
            .ok_or_else(|| anyhow!("the fixed image is no longer loaded"))?,
        st.planar_by_uid(&reg.moving_uid)
            .ok_or_else(|| anyhow!("the moving image is no longer loaded"))?,
    );
    let (from, to) = (&st.planar_images[mi], &st.planar_images[fi]);
    let mut set = PlanarStructureSet::new(
        &format!("Carried · {}", reg.label),
        &to.study_uid,
        "carried",
    );
    let mut refused = 0;
    for (si, s) in st.planar_structs.iter().enumerate() {
        if only.is_some_and(|o| o != si) {
            continue;
        }
        for x in &s.structures {
            let Some(c) = x.contour(&reg.moving_uid) else {
                continue;
            };
            match planar_structs::carry(&c.region, &reg.result.transform, from, to) {
                Some(region) if !region.is_empty() => {
                    let d = set.structure_entry(&x.name, Some(x.color));
                    let e = d.contour_entry(&reg.fixed_uid);
                    e.region = region;
                    e.source = ContourSource::Carried {
                        from_image: reg.moving_uid.clone(),
                        registration: reg.label.clone(),
                    };
                }
                Some(_) => {}
                None => refused += 1,
            }
        }
    }
    set.refresh_images();
    if set.structures.is_empty() {
        bail!(if refused > 0 {
            "the warp folds where the contours are: nothing was carried"
        } else {
            "the moving image has no planar structure to carry"
        });
    }
    let k = planar_structs::file_set(st, set);
    let mut out = set_json(st, k);
    out["refused_in_folds"] = json!(refused);
    Ok(out)
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SetupFromPlanarArgs {
    pub dataset: String,
    /// The CT series (number); the displayed one when omitted.
    #[serde(default)]
    pub series: Option<u32>,
    /// The 2D registrations to combine (number or label), one per view.
    pub registrations: Vec<String>,
    /// x, y, z in patient coordinates, mm; the fixed images' isocentre
    /// when omitted.
    #[serde(default)]
    pub isocenter: Option<[f64; 3]>,
    /// A Patient Position code (HFS, FFS, ...); the CT's when omitted.
    #[serde(default)]
    pub patient: Option<String>,
}

pub fn setup_from_planar(core: &mut Core, a: SetupFromPlanarArgs, p: &Progress) -> Result<Value> {
    let ds = core.session.dataset(&a.dataset)?;
    let idx = core.session.series_index(ds, a.series).ok();
    let volume = match idx {
        Some(i) if !ds.study.series.is_empty() => Some(core.session.volume(&a.dataset, i, p)?),
        _ => None,
    };
    let ds = core.session.dataset(&a.dataset)?;
    let mut study = planar_structs::projection_view(&ds.study);
    // The pixels are needed here: the views keep the fixed images.
    study.planar_images = ds.study.planar_images.clone();
    study.planar_regs = ds.study.planar_regs.clone();
    if let (Some(v), Some(i)) = (volume, idx) {
        study.volume = v;
        study.active_series = i;
    }
    let ks = a
        .registrations
        .iter()
        .map(|s| reg_index(&study, s))
        .collect::<Result<Vec<_>>>()?;
    let regs: Vec<&PlanarReg> = ks.iter().map(|&k| &study.planar_regs[k]).collect();
    let mut opts = crate::setup::SetupOptions {
        isocenter: a.isocenter.map(|v| Vec3::new(v[0], v[1], v[2])),
        ..Default::default()
    };
    if let Some(s) = &a.patient {
        opts.patient = Some(crate::iec::PatientFrame::Position(
            crate::iec::PatientPosition::from_code(s)
                .ok_or_else(|| anyhow!("patient '{s}' is not a Patient Position code"))?,
        ));
    }
    let result = crate::setup::from_planar(&study, &regs, &opts, p)?;
    let id = core.session.mint("setup");
    let mut j = result.json();
    j["label"] = json!(clean_text(&result.label));
    let out = json!({
        "setup": id,
        "dataset": a.dataset,
        "result": j,
        "note": "a setup read from 2D/2D matches: the correction maps the room onto the CT, \
                 as register_2d3d's does; describe_setup and export_setup take its handle",
    });
    core.session.setups.push(Setup {
        id,
        dataset: a.dataset.clone(),
        series: idx.unwrap_or(0),
        result,
    });
    Ok(out)
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SaveArgs {
    pub dataset: String,
    /// The set, by number or label.
    pub set: String,
    /// Name of the folder under the session's output folder.
    #[serde(default)]
    pub folder: Option<String>,
}

pub fn save_planar_structures(core: &mut Core, a: SaveArgs, _p: &Progress) -> Result<Value> {
    let k = {
        let ds = core.session.dataset(&a.dataset)?;
        set_index(&ds.study, &a.set)?
    };
    let dir = core
        .session
        .fresh_out_subdir(a.folder.as_deref().unwrap_or("planar-structures"))?;
    let ds = core.session.dataset(&a.dataset)?;
    let (files, _) = crate::gsps::save_set(&ds.study, k, &dir, &|_| None)?;
    Ok(json!({
        "set": k + 1,
        "folder": dir.to_string_lossy(),
        "files": files
            .iter()
            .filter_map(|f| f.file_name())
            .map(|f| f.to_string_lossy().into_owned())
            .collect::<Vec<_>>(),
    }))
}

/// The planar structure sets and 2D registrations, in describe_dataset.
pub fn summary(st: &LoadedStudy) -> (Vec<Value>, Vec<Value>) {
    (
        (0..st.planar_structs.len())
            .map(|k| set_json(st, k))
            .collect(),
        st.planar_regs
            .iter()
            .enumerate()
            .map(|(k, r)| reg_json(st, k, r))
            .collect(),
    )
}
