//! Patient setup from radiographs: 2D/3D registration of a CT to planar
//! images of its dataset, and what can be done with the answer.

use anyhow::{anyhow, bail, Context, Result};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{json, Value};

use super::super::phi::clean_text;
use super::super::session::Setup;
use super::super::Core;
use crate::drr::{Geometry, HuMode};
use crate::geometry::Vec3;
use crate::iec::{PatientFrame, PatientPosition};
use crate::progress::Progress;
use crate::registration::metric2d::Metric2d;
use crate::registration::projective::{Dof2d3d, Init2d3d, Optimizer2d};
use crate::setup::{self, GeometrySource, MarkerSource, SetupMethod, SetupOptions, ViewChoice};

/// One radiograph of a run.
#[derive(Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct ImageArg {
    /// A planar image of the dataset: its number as describe_dataset lists
    /// it, or its label.
    pub image: String,
    /// Where its geometry comes from: `image` (its own RT Image attributes,
    /// the default), `beam` (a beam of the plan: `plan` and `beam` numbers),
    /// `room` (a view of imaging-rooms.json: `room` and `view`), or `typed`
    /// (the IEC numbers below).
    #[serde(default)]
    pub geometry: Option<String>,
    #[serde(default)]
    pub plan: Option<usize>,
    #[serde(default)]
    pub beam: Option<usize>,
    #[serde(default)]
    pub room: Option<String>,
    #[serde(default)]
    pub view: Option<String>,
    /// For `typed` (and the distances of `beam`): gantry and couch angles,
    /// degrees; source-axis and source-imager distances, mm.
    #[serde(default)]
    pub gantry_deg: Option<f64>,
    #[serde(default)]
    pub couch_deg: Option<f64>,
    #[serde(default)]
    pub sad_mm: Option<f64>,
    #[serde(default)]
    pub sid_mm: Option<f64>,
    /// How much this view counts (default 1).
    #[serde(default)]
    pub weight: Option<f64>,
    /// The pixels are detector signal (bright where little attenuates).
    #[serde(default)]
    pub transmission: Option<bool>,
    /// The part of the image to compare: polygons of `[column, row]`
    /// pixel positions (the first pixel's centre at `[0, 0]`); the whole
    /// image when omitted.
    #[serde(default)]
    pub region: Option<Vec<Vec<[f64; 2]>>>,
    /// Markers placed on this image by hand: a marker's name and its
    /// `[column, row]`, standing for what detection would find there.
    #[serde(default)]
    pub picks: Option<Vec<PickArg>>,
}

/// A marker placed by hand.
#[derive(Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct PickArg {
    pub marker: String,
    pub at: [f64; 2],
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Register2d3dArgs {
    pub dataset: String,
    /// The CT series (number); the displayed one when omitted.
    #[serde(default)]
    pub series: Option<u32>,
    pub images: Vec<ImageArg>,
    /// `intensity` (default), `markers` or `hybrid` (markers, then
    /// intensity from their pose).
    #[serde(default)]
    pub method: Option<String>,
    /// `ncc`, `gc` (default), `gd`, `pi`, `lncc` or `mi`.
    #[serde(default)]
    pub metric: Option<String>,
    /// `powell` (default) or `nelder-mead`.
    #[serde(default)]
    pub optimizer: Option<String>,
    /// `plan`, `in-plane` (default) or `grid`.
    #[serde(default)]
    pub init: Option<String>,
    #[serde(default)]
    pub levels: Option<usize>,
    /// `six` (default) or `translations`.
    #[serde(default)]
    pub dof: Option<String>,
    /// Half the size of a clip box about the isocentre, mm: rays integrate
    /// inside it only. None by default.
    #[serde(default)]
    pub clip_mm: Option<f64>,
    /// The clip box about this structure instead, `clip_margin_mm`
    /// (default 10) larger on every side.
    #[serde(default)]
    pub clip_structure: Option<String>,
    #[serde(default)]
    pub clip_margin_mm: Option<f64>,
    /// Rays integrate inside the clip box only; by default the DRRs see
    /// the whole CT and the box chooses the pixels compared.
    #[serde(default)]
    pub clip_rays: Option<bool>,
    /// Voxels below this value are left out of the DRR (default -800;
    /// 150 makes a bone DRR).
    #[serde(default)]
    pub threshold_hu: Option<f32>,
    /// `water` (default), `mv` (megavoltage) or `raw`.
    #[serde(default)]
    pub hu: Option<String>,
    /// The finest level's longest side, pixels (default 256).
    #[serde(default)]
    pub pixels: Option<usize>,
    /// Ray step at the finest level, mm (default 1).
    #[serde(default)]
    pub step_mm: Option<f64>,
    /// Bounds about the start: translation in mm, rotation in degrees
    /// (default 30 and 15).
    #[serde(default)]
    pub range_mm: Option<f64>,
    #[serde(default)]
    pub range_deg: Option<f64>,
    /// Finite-difference refinement after the search.
    #[serde(default)]
    pub refine: Option<bool>,
    /// x, y, z in patient coordinates, mm; the images' or the plan's
    /// isocentre when omitted.
    #[serde(default)]
    pub isocenter: Option<[f64; 3]>,
    /// A Patient Position code, or `upright:DEG`; the CT's when omitted.
    #[serde(default)]
    pub patient: Option<String>,
    /// Markers: the points of interest of these names; the MARKER ones when
    /// empty or omitted.
    #[serde(default)]
    pub markers: Option<Vec<String>>,
    /// Find the markers in the CT instead.
    #[serde(default)]
    pub detect_markers: Option<bool>,
    /// Around each marker's projection, mm (default 15).
    #[serde(default)]
    pub search_mm: Option<f64>,
    /// The markers' diameter, mm (default 1.5).
    #[serde(default)]
    pub marker_mm: Option<f64>,
    /// File the DRRs at the answer as planar images of the dataset.
    #[serde(default)]
    pub file_drrs: Option<bool>,
}

fn opt_key<T>(v: &Option<String>, what: &str, f: impl Fn(&str) -> Option<T>) -> Result<Option<T>> {
    match v.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        None => Ok(None),
        Some(s) => f(s)
            .map(Some)
            .ok_or_else(|| anyhow!("{what} '{s}' is not one of the choices")),
    }
}

/// A planar image of the study by number or label.
fn image_index(study: &crate::loader::LoadedStudy, s: &str) -> Result<usize> {
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

fn patient_of(s: &str) -> Result<PatientFrame> {
    let l = s.trim().to_ascii_lowercase();
    if let Some(rest) = l.strip_prefix("upright") {
        let facing_deg = rest.trim_start_matches(':').trim().parse().unwrap_or(0.0);
        return Ok(PatientFrame::Upright { facing_deg });
    }
    PatientPosition::from_code(s)
        .map(PatientFrame::Position)
        .ok_or_else(|| anyhow!("patient '{s}' is neither a Patient Position code nor upright:DEG"))
}

pub fn register_2d3d(core: &mut Core, a: Register2d3dArgs, p: &Progress) -> Result<Value> {
    if a.images.is_empty() {
        bail!("images is empty: name at least one planar image");
    }
    let ds = core.session.dataset(&a.dataset)?;
    let idx = core.session.series_index(ds, a.series)?;
    let volume = core.session.volume(&a.dataset, idx, p)?;
    let ds = core.session.dataset(&a.dataset)?;
    // The study as the run sees it: the chosen series as the volume.
    let mut study = ds.study.clone();
    study.volume = volume;
    study.active_series = idx;

    let mut views = Vec::with_capacity(a.images.len());
    for im in &a.images {
        let i = image_index(&study, &im.image)?;
        let typed = |base: Geometry| Geometry {
            gantry_deg: im.gantry_deg.unwrap_or(base.gantry_deg),
            couch_deg: im.couch_deg.unwrap_or(base.couch_deg),
            sad: im.sad_mm.unwrap_or(base.sad),
            sid: im.sid_mm.unwrap_or(base.sid),
            ..base
        };
        let source = match im.geometry.as_deref().unwrap_or("image") {
            "image" => GeometrySource::Image,
            "beam" => GeometrySource::Beam {
                plan: im.plan.unwrap_or(1).saturating_sub(1),
                beam: im
                    .beam
                    .ok_or_else(|| anyhow!("geometry beam needs a beam number"))?
                    .saturating_sub(1),
                base: typed(Geometry::default()),
            },
            "room" => GeometrySource::Room {
                room: im
                    .room
                    .clone()
                    .ok_or_else(|| anyhow!("geometry room needs a room"))?,
                view: im
                    .view
                    .clone()
                    .ok_or_else(|| anyhow!("geometry room needs a view"))?,
            },
            "typed" => GeometrySource::Typed(typed(Geometry::default())),
            other => bail!("geometry '{other}': image, beam, room or typed"),
        };
        views.push(ViewChoice {
            image: i,
            source,
            weight: im.weight.unwrap_or(1.0).max(0.0),
            transmission: im.transmission.unwrap_or(false),
            region: im.region.clone().unwrap_or_default(),
            picks: im
                .picks
                .iter()
                .flatten()
                .map(|p| (p.marker.clone(), p.at))
                .collect(),
        });
    }

    let mut opts = SetupOptions::default();
    if let Some(m) = opt_key(&a.method, "method", SetupMethod::from_key)? {
        opts.method = m;
    }
    if let Some(m) = opt_key(&a.metric, "metric", Metric2d::from_key)? {
        opts.metric = m;
    }
    if let Some(o) = opt_key(&a.optimizer, "optimizer", Optimizer2d::from_key)? {
        opts.optimizer = o;
    }
    if let Some(i) = opt_key(&a.init, "init", Init2d3d::from_key)? {
        opts.init = i;
    }
    if let Some(d) = opt_key(&a.dof, "dof", |s| match s {
        "six" | "6" => Some(Dof2d3d::Six),
        "translations" | "3" => Some(Dof2d3d::Translations),
        _ => None,
    })? {
        opts.dof = d;
    }
    if let Some(h) = opt_key(&a.hu, "hu", HuMode::from_key)? {
        opts.hu = h;
    }
    if let Some(v) = a.levels {
        opts.levels = v.clamp(1, 5);
    }
    opts.clip_half_mm = a.clip_mm.filter(|v| *v > 0.0);
    opts.clip_structure = a
        .clip_structure
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from);
    if let Some(m) = a.clip_margin_mm {
        opts.clip_margin_mm = m.clamp(0.0, 100.0);
    }
    opts.clip_rays = a.clip_rays.unwrap_or(false);
    if let Some(v) = a.threshold_hu {
        opts.threshold_hu = v;
    }
    if let Some(v) = a.pixels {
        opts.max_pixels = v.clamp(32, 1024);
    }
    if let Some(v) = a.step_mm {
        opts.step_mm = v.clamp(0.25, 10.0);
    }
    if let Some(v) = a.range_mm {
        opts.range_mm = v.clamp(1.0, 100.0);
    }
    if let Some(v) = a.range_deg {
        opts.range_deg = v.clamp(0.5, 45.0);
    }
    opts.refine = a.refine.unwrap_or(false);
    opts.isocenter = a.isocenter.map(|v| Vec3::new(v[0], v[1], v[2]));
    if let Some(s) = &a.patient {
        opts.patient = Some(patient_of(s)?);
    }
    opts.markers = if a.detect_markers.unwrap_or(false) {
        MarkerSource::Detect
    } else {
        match &a.markers {
            Some(n) if !n.is_empty() => MarkerSource::Named(n.clone()),
            _ => MarkerSource::Pois,
        }
    };
    if let Some(v) = a.search_mm {
        opts.search_mm = v.clamp(2.0, 60.0);
    }
    if let Some(v) = a.marker_mm {
        opts.marker_mm = v.clamp(0.3, 10.0);
    }
    opts.device = core.session.config.device_pref();

    let rooms = crate::rooms::load_default().0;
    p.set("Registering the radiographs");
    let result = setup::run(&study, &views, &opts, Some(&rooms), p)?;
    let id = core.session.mint("setup");
    let mut filed = Vec::new();
    if a.file_drrs.unwrap_or(false) {
        let ds = core.session.dataset_mut(&a.dataset)?;
        for k in 0..result.result.views.len() {
            let radiograph = views
                .get(k)
                .and_then(|v| ds.study.planar_images.get(v.image))
                .cloned();
            if let Some(img) = result.drr_planar(k, radiograph.as_ref()) {
                ds.study.planar_images.push(img);
                filed.push(ds.study.planar_images.len());
            }
        }
    }
    let mut answer = report_json(&id, &a.dataset, idx, &result);
    if !filed.is_empty() {
        answer["drrs_filed_as_planar_images"] = json!(filed);
    }
    core.session.setups.push(Setup {
        id,
        dataset: a.dataset.clone(),
        series: idx,
        result,
    });
    Ok(answer)
}

fn report_json(id: &str, dataset: &str, series: usize, s: &setup::SetupResult) -> Value {
    let mut j = s.json();
    // Labels come from the files: capped and cleaned, and the redactor
    // sees the whole answer after this.
    j["label"] = json!(clean_text(&s.label));
    if let Some(views) = j["views"].as_array_mut() {
        for v in views {
            if let Some(l) = v["label"].as_str() {
                v["label"] = json!(clean_text(l));
            }
        }
    }
    if let Some(res) = j["markers"]["residuals"].as_array_mut() {
        for r in res {
            for k in ["marker", "view"] {
                if let Some(l) = r[k].as_str() {
                    r[k] = json!(clean_text(l));
                }
            }
        }
    }
    json!({
        "setup": id,
        "dataset": dataset,
        "series": series + 1,
        "result": j,
        "note": "the correction maps the room (the radiographs' planned frame) onto the CT: \
                 applied to the patient it puts him where he was planned; the displacement \
                 is how he was found",
    })
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SetupArgs {
    /// A `setup` handle from register_2d3d.
    pub setup: String,
}

pub fn describe_setup(core: &mut Core, a: SetupArgs, _p: &Progress) -> Result<Value> {
    let s = core.session.setup(&a.setup)?;
    Ok(report_json(&s.id, &s.dataset, s.series, &s.result))
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ExportSetupArgs {
    pub setup: String,
    /// Name of the folder under the session's output folder.
    #[serde(default)]
    pub folder: Option<String>,
}

/// The rigid Spatial Registration, the DRRs at the answer as RT Images and
/// the report as JSON, into one folder.
pub fn export_setup(core: &mut Core, a: ExportSetupArgs, _p: &Progress) -> Result<Value> {
    let (dataset, result) = {
        let s = core.session.setup(&a.setup)?;
        (s.dataset.clone(), s.result.clone())
    };
    let dir = core
        .session
        .fresh_out_subdir(a.folder.as_deref().unwrap_or(&format!("{}-setup", a.setup)))?;
    std::fs::create_dir_all(&dir).context("create the output folder")?;
    let ds = core.session.dataset(&dataset)?;
    let files = setup::write_files(&ds.study, &result, &dir)?;
    Ok(json!({
        "setup": a.setup,
        "folder": dir.to_string_lossy(),
        "files": files.iter().filter_map(|f| f.file_name()).map(|f| f.to_string_lossy().into_owned()).collect::<Vec<_>>(),
    }))
}
