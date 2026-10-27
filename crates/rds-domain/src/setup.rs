//! Patient setup from radiographs: a 2D/3D registration as the program asks
//! for it and files it.
//!
//! The engine ([`crate::registration::projective`],
//! [`crate::registration::markers`]) takes images and cameras; this module
//! makes them out of a study - its planar images, the geometry each one
//! states or is given (its own RT Image attributes, a beam of the plan, a
//! calibrated room view, IEC parameters typed in), the plan's isocentre, the
//! CT's patient position, the markers among its points of interest - and
//! keeps what comes back as a [`SetupResult`] on the study, where the data
//! tree, the report, the positioned copy and the DICOM writers find it. The
//! viewer, the MCP server and the workflow step all go through [`run`], so
//! the three cannot ask the same question differently.

use anyhow::{anyhow, bail, Result};

use crate::drr::{Geometry, HuMode};
use crate::extras::PlanarImage;
use crate::geometry::Vec3;
use crate::iec::PatientFrame;
use crate::loader::LoadedStudy;
use crate::nn::device::DevicePref;
use crate::progress::ProgressSink;
use crate::projection::ProjectionGeometry;
use crate::registration::markers::{self, Detect3d, Marker3d, MarkerRequest};
use crate::registration::metric2d::{Image2d, Mask2d, Metric2d};
use crate::registration::projective::{
    self, ClipBox, Dof2d3d, Init2d3d, Optimizer2d, Request2d3d, Result2d3d, View2d,
};
use crate::registration::RigidTransform;
use crate::rooms::RoomsFile;

/// Where a view's camera comes from.
#[derive(Clone, Debug, PartialEq)]
pub enum GeometrySource {
    /// The image's own RT Image attributes.
    Image,
    /// A beam of a plan of the study: its gantry and couch angles and its
    /// isocentre, with the distances of `base` and a panel the size of the
    /// image at its pixel spacing.
    Beam {
        plan: usize,
        beam: usize,
        base: Geometry,
    },
    /// A view of a calibrated room, about the isocentre.
    Room { room: String, view: String },
    /// IEC parameters typed in; the panel and the pixels follow the image.
    Typed(Geometry),
}

impl GeometrySource {
    /// `RT image`, `beam 2`, `room: kV A`, `typed`.
    pub fn describe(&self, study: &LoadedStudy) -> String {
        match self {
            GeometrySource::Image => "RT image".into(),
            GeometrySource::Beam { plan, beam, .. } => study
                .plans
                .get(*plan)
                .and_then(|p| p.beams.get(*beam))
                .map(|b| format!("beam {}", b.name))
                .unwrap_or_else(|| "beam".into()),
            GeometrySource::Room { view, .. } => format!("room: {view}"),
            GeometrySource::Typed(_) => "typed".into(),
        }
    }
}

/// One radiograph of a run.
#[derive(Clone, Debug, PartialEq)]
pub struct ViewChoice {
    /// Index into the study's planar images.
    pub image: usize,
    pub source: GeometrySource,
    pub weight: f64,
    /// The pixels are detector signal, not attenuation.
    pub transmission: bool,
    /// The part of the image the measure compares: polygons in its pixels
    /// (the centre of the first at `(0, 0)`); empty for the whole image.
    pub region: Vec<Vec<[f64; 2]>>,
    /// Markers placed by hand on the image: a name of the run's markers
    /// and the pixel it is at, standing for what detection would find.
    pub picks: Vec<(String, [f64; 2])>,
}

impl ViewChoice {
    pub fn new(image: usize, source: GeometrySource) -> Self {
        ViewChoice {
            image,
            source,
            weight: 1.0,
            transmission: false,
            region: Vec::new(),
            picks: Vec::new(),
        }
    }
}

/// The IEC parameters `g` set for an image's raster: its pixel count and,
/// at the imager, its pixel spacing.
fn for_raster(g: &Geometry, img: &PlanarImage) -> Geometry {
    Geometry {
        dims: [img.cols, img.rows],
        panel_mm: [
            img.cols as f64 * img.spacing[0],
            img.rows as f64 * img.spacing[1],
        ],
        image_center: [0.0, 0.0],
        ..*g
    }
}

/// The camera of planar image `idx` of `study`, in the CT's patient
/// coordinates, from `source`; `iso` and `frame` complete a geometry that
/// does not state them.
pub fn camera_of(
    study: &LoadedStudy,
    idx: usize,
    source: &GeometrySource,
    iso: Vec3,
    frame: PatientFrame,
    rooms: Option<&RoomsFile>,
) -> Result<ProjectionGeometry> {
    let img = study
        .planar_images
        .get(idx)
        .ok_or_else(|| anyhow!("no planar image {}", idx + 1))?;
    if img.kind != crate::extras::PlanarKind::Projection {
        bail!("{}: an {}, not a radiograph", img.label, img.kind.label());
    }
    let cam = match source {
        GeometrySource::Image => img.camera.ok_or_else(|| {
            anyhow!(
                "{}: no geometry of its own ({} missing)",
                img.label,
                img.geometry_missing
                    .as_deref()
                    .unwrap_or("RT Image attributes")
            )
        })?,
        GeometrySource::Beam { plan, beam, base } => {
            let b = study
                .plans
                .get(*plan)
                .and_then(|p| p.beams.get(*beam))
                .ok_or_else(|| anyhow!("no beam {} of plan {}", beam + 1, plan + 1))?;
            let g = Geometry {
                patient: frame,
                isocenter: iso,
                ..for_raster(base, img)
            }
            .from_beam(b);
            g.camera()
        }
        GeometrySource::Room { room, view } => {
            let rooms =
                rooms.ok_or_else(|| anyhow!("no imaging-rooms.json to take {view} from"))?;
            let r = rooms
                .rooms
                .iter()
                .find(|r| &r.name == room)
                .ok_or_else(|| anyhow!("no room {room:?}"))?;
            let v = r
                .views
                .iter()
                .find(|v| &v.name == view)
                .ok_or_else(|| anyhow!("no view {view:?} in {room:?}"))?;
            v.camera_in_patient(&r.patient_frame(), iso, 0.0, 0.0, 0.0)
                .ok_or_else(|| anyhow!("{view:?} has no usable calibration"))?
        }
        GeometrySource::Typed(g) => Geometry {
            patient: frame,
            isocenter: iso,
            ..for_raster(g, img)
        }
        .camera(),
    };
    if cam.dims != [img.cols, img.rows] {
        bail!(
            "{}: the geometry is {}×{} pixels, the image {}×{}",
            img.label,
            cam.dims[0],
            cam.dims[1],
            img.cols,
            img.rows
        );
    }
    Ok(cam)
}

/// What a run does.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum SetupMethod {
    /// The radiographs against the DRRs.
    #[default]
    Intensity,
    /// The markers alone.
    Markers,
    /// The markers' pose, then the intensity run from it.
    Hybrid,
}

impl SetupMethod {
    pub const ALL: [SetupMethod; 3] = [
        SetupMethod::Intensity,
        SetupMethod::Markers,
        SetupMethod::Hybrid,
    ];

    pub fn label(self) -> &'static str {
        match self {
            SetupMethod::Intensity => "Intensity",
            SetupMethod::Markers => "Markers",
            SetupMethod::Hybrid => "Markers, then intensity",
        }
    }

    pub fn key(self) -> &'static str {
        match self {
            SetupMethod::Intensity => "intensity",
            SetupMethod::Markers => "markers",
            SetupMethod::Hybrid => "hybrid",
        }
    }

    pub fn from_key(s: &str) -> Option<SetupMethod> {
        let s = s.trim().to_ascii_lowercase();
        SetupMethod::ALL.into_iter().find(|m| m.key() == s)
    }
}

/// Where the markers come from.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum MarkerSource {
    /// The study's points of interest of type MARKER, or those named.
    #[default]
    Pois,
    Named(Vec<String>),
    /// Found in the CT inside the clip box.
    Detect,
}

/// Everything about a run but the views.
#[derive(Clone, Debug)]
pub struct SetupOptions {
    pub method: SetupMethod,
    pub metric: Metric2d,
    pub optimizer: Optimizer2d,
    pub init: Init2d3d,
    pub levels: usize,
    pub dof: Dof2d3d,
    /// Half the size of the clip box, mm, about `clip_center` (the
    /// isocentre when `None`); no box when `None`.
    pub clip_half_mm: Option<f64>,
    pub clip_center: Option<Vec3>,
    /// The clip box about a structure instead, `clip_margin_mm` larger on
    /// every side than its extent.
    pub clip_structure: Option<String>,
    pub clip_margin_mm: f64,
    /// The rays integrate inside the box only, rather than the box saying
    /// only which pixels are compared.
    pub clip_rays: bool,
    pub threshold_hu: f32,
    pub hu: HuMode,
    pub max_pixels: usize,
    pub step_mm: f64,
    pub range_mm: f64,
    pub range_deg: f64,
    pub refine: bool,
    pub conditioning: bool,
    /// The plan's (or the images') isocentre when `None`.
    pub isocenter: Option<Vec3>,
    /// The CT's Patient Position when `None`.
    pub patient: Option<PatientFrame>,
    pub markers: MarkerSource,
    pub search_mm: f64,
    pub marker_mm: f64,
    /// Start from this pose rather than the plan.
    pub start: Option<RigidTransform>,
    /// Where the DRRs of the intensity search are rendered.
    pub device: DevicePref,
}

impl Default for SetupOptions {
    fn default() -> Self {
        SetupOptions {
            method: SetupMethod::Intensity,
            metric: Metric2d::GradientCorrelation,
            optimizer: Optimizer2d::Powell,
            init: Init2d3d::InPlane,
            levels: 3,
            dof: Dof2d3d::Six,
            clip_half_mm: None,
            clip_center: None,
            clip_structure: None,
            clip_margin_mm: 10.0,
            clip_rays: false,
            threshold_hu: -800.0,
            hu: HuMode::Water,
            max_pixels: 256,
            step_mm: 1.0,
            range_mm: 30.0,
            range_deg: 15.0,
            refine: false,
            conditioning: true,
            isocenter: None,
            patient: None,
            markers: MarkerSource::Pois,
            search_mm: 15.0,
            marker_mm: 1.5,
            start: None,
            device: DevicePref::Auto,
        }
    }
}

/// The isocentre a run turns about when none is given: the first chosen
/// image's own, else the first beam's of the study's plans, else the CT's
/// centre.
pub fn default_isocenter(study: &LoadedStudy, views: &[ViewChoice]) -> Vec3 {
    views
        .iter()
        .find_map(|v| {
            study
                .planar_images
                .get(v.image)
                .and_then(|p| p.iec.as_ref())
                .filter(|_| study.planar_images[v.image].camera.is_some())
                .map(|g| g.isocenter)
        })
        .or_else(|| {
            study
                .plans
                .iter()
                .flat_map(|p| p.beams.iter())
                .find_map(|b| b.isocenter)
        })
        .unwrap_or_else(|| Geometry::centered_on(&study.volume).isocenter)
}

/// How the patient lay on the CT, as its series says.
pub fn default_patient(study: &LoadedStudy) -> PatientFrame {
    study
        .series
        .get(study.active_series)
        .and_then(crate::loader::series_patient_position)
        .map(PatientFrame::Position)
        .unwrap_or_default()
}

/// The box about structure `name` on the study's CT - the extent of its
/// voxels in patient coordinates - grown by `margin_mm` on every side.
pub fn structure_box(study: &LoadedStudy, name: &str, margin_mm: f64) -> Result<ClipBox> {
    let s = crate::workflow::select::find(study, name, None)
        .ok_or_else(|| anyhow!("no structure is named '{name}'"))?;
    let vol = &study.volume;
    let mask = s.mask_on(&vol.grid())?;
    let [nx, ny, _] = vol.dims;
    let mut lo = [usize::MAX; 3];
    let mut hi = [0usize; 3];
    for (o, _) in mask.iter().enumerate().filter(|(_, &m)| m != 0) {
        let c = [o % nx, (o / nx) % ny, o / (nx * ny)];
        for a in 0..3 {
            lo[a] = lo[a].min(c[a]);
            hi[a] = hi[a].max(c[a]);
        }
    }
    if lo[0] == usize::MAX {
        bail!("'{name}' has no voxel on this CT");
    }
    // The corners of the voxels' outer faces, in patient coordinates.
    let mut a = Vec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
    let mut b = Vec3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for c in 0..8 {
        let at = |bit: usize, ax: usize| {
            if c & bit == 0 {
                lo[ax] as f64 - 0.5
            } else {
                hi[ax] as f64 + 0.5
            }
        };
        let p = vol.voxel_to_patient(at(1, 0), at(2, 1), at(4, 2));
        a = Vec3::new(a.x.min(p.x), a.y.min(p.y), a.z.min(p.z));
        b = Vec3::new(b.x.max(p.x), b.y.max(p.y), b.z.max(p.z));
    }
    let m = margin_mm.max(0.0);
    Ok(ClipBox {
        center: (a + b) * 0.5,
        half: (b - a) * 0.5 + Vec3::new(m, m, m),
    })
}

/// The pixels of an image `dims` inside any of `polygons` (pixel
/// coordinates, even-odd rule per polygon).
pub fn region_mask(dims: [usize; 2], polygons: &[Vec<[f64; 2]>]) -> Mask2d {
    let [w, h] = dims;
    let mut m = vec![0u8; w * h];
    for poly in polygons.iter().filter(|p| p.len() >= 3) {
        let (y0, y1) = poly
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), p| {
                (a.min(p[1]), b.max(p[1]))
            });
        let j0 = y0.ceil().max(0.0) as usize;
        let j1 = (y1.floor().max(-1.0) + 1.0).min(h as f64) as usize;
        for j in j0..j1 {
            let y = j as f64;
            // Where the edges cross this row, left to right.
            let mut xs: Vec<f64> = Vec::new();
            for k in 0..poly.len() {
                let (p, q) = (poly[k], poly[(k + 1) % poly.len()]);
                if (p[1] <= y) != (q[1] <= y) {
                    xs.push(p[0] + (y - p[1]) / (q[1] - p[1]) * (q[0] - p[0]));
                }
            }
            xs.sort_by(f64::total_cmp);
            for pair in xs.as_chunks::<2>().0 {
                let i0 = pair[0].ceil().max(0.0) as usize;
                let i1 = (pair[1].floor() + 1.0).clamp(0.0, w as f64) as usize;
                for i in i0..i1.max(i0) {
                    m[j * w + i] = 1;
                }
            }
        }
    }
    m
}

/// The markers of the study a run would use.
pub fn study_markers(
    study: &LoadedStudy,
    opts: &SetupOptions,
    clip: Option<&ClipBox>,
) -> Vec<Marker3d> {
    let pois = || {
        study
            .structure_sets
            .iter()
            .flat_map(|s| s.rois.iter())
            .filter_map(|r| r.point().map(|p| (r, p)))
    };
    match &opts.markers {
        MarkerSource::Pois => pois()
            .filter(|(r, _)| r.roi_type.eq_ignore_ascii_case("MARKER"))
            .map(|(r, p)| Marker3d {
                name: r.name.clone(),
                position: p,
            })
            .collect(),
        MarkerSource::Named(names) => names
            .iter()
            .filter_map(|n| {
                pois()
                    .find(|(r, _)| r.name.eq_ignore_ascii_case(n))
                    .map(|(r, p)| Marker3d {
                        name: r.name.clone(),
                        position: p,
                    })
            })
            .collect(),
        MarkerSource::Detect => markers::detect_3d(
            &study.volume,
            &Detect3d {
                clip: clip.map(|c| c.corners()),
                ..Detect3d::default()
            },
        ),
    }
}

/// A finished run, as the study keeps it.
#[derive(Clone, Debug)]
pub struct SetupResult {
    pub label: String,
    pub method: SetupMethod,
    pub result: Result2d3d,
    /// The planar images it registered, by index and by label (the label
    /// says which they were after one is removed).
    pub images: Vec<usize>,
    pub image_labels: Vec<String>,
    /// The CT series it registered and its frame of reference.
    pub series_uid: String,
    pub for_uid: String,
    pub isocenter: Vec3,
    pub patient: PatientFrame,
    /// How the DRRs were made, to make them again at another pose.
    pub hu: HuMode,
    pub threshold_hu: f32,
    /// `YYYY-MM-DD HH:MM` (UTC): when the run finished.
    pub when: String,
    /// Moved by hand after the run.
    pub adjusted: bool,
    /// The portal images' fields: where the edge is against the plan.
    pub fields: Vec<FieldReport>,
}

/// One portal image's field against its beam's planned aperture.
#[derive(Clone, Debug)]
pub struct FieldReport {
    /// Which view of the result.
    pub view: usize,
    pub beam: String,
    /// The planned aperture on the image, `0..1` across and down.
    pub outline: Vec<([f32; 2], [f32; 2])>,
    /// The field edge found on the image; `None` when too few rays saw it.
    pub edge: Option<crate::portal::FieldEdge>,
    /// How the anatomy at the isocentre was found displaced, along the
    /// beam limiting device's `Xb` and `Yb`, mm.
    pub anatomy_mm: [f64; 2],
}

impl FieldReport {
    /// The field's offset from the anatomy: where the field is against
    /// the patient, which is what the portal check is for.
    pub fn relative_mm(&self) -> Option<[f64; 2]> {
        self.edge.map(|e| {
            [
                e.shift_mm[0] - self.anatomy_mm[0],
                e.shift_mm[1] - self.anatomy_mm[1],
            ]
        })
    }

    pub fn line(&self) -> String {
        match (self.edge, self.relative_mm()) {
            (Some(e), Some(r)) => format!(
                "{}: field {:+.1}, {:+.1} mm · anatomy {:+.1}, {:+.1} mm · field to anatomy {:+.1}, {:+.1} mm",
                self.beam, e.shift_mm[0], e.shift_mm[1], self.anatomy_mm[0], self.anatomy_mm[1], r[0], r[1]
            ),
            _ => format!("{}: no field edge found", self.beam),
        }
    }
}

/// The geometry and planned aperture of a portal image: an MV image whose
/// referenced beam (or the beam its geometry comes from) has jaws or an
/// MLC; and the beam's name.
fn portal_aperture(
    study: &LoadedStudy,
    v: &ViewChoice,
    iso: Vec3,
    frame: PatientFrame,
) -> Option<(Geometry, crate::rtplan::Aperture, String)> {
    let img = study.planar_images.get(v.image)?;
    let beams = || study.plans.iter().flat_map(|p| p.beams.iter());
    let beam = match &v.source {
        GeometrySource::Beam { plan, beam, .. } => study.plans.get(*plan)?.beams.get(*beam),
        _ => {
            if !img.megavoltage {
                return None;
            }
            beams().find(|b| Some(b.number) == img.referenced_beam && b.aperture.is_some())
        }
    }?;
    let ap = beam.aperture.clone()?;
    let mut g = match &v.source {
        GeometrySource::Image => img.iec?,
        GeometrySource::Beam { base, .. } => Geometry {
            patient: frame,
            isocenter: iso,
            ..base.from_beam(beam)
        },
        GeometrySource::Typed(t) => Geometry {
            patient: frame,
            isocenter: iso,
            ..*t
        },
        GeometrySource::Room { .. } => return None,
    };
    if !matches!(v.source, GeometrySource::Image) || g.collimator_deg == 0.0 {
        g.collimator_deg = beam.collimator_angle.unwrap_or(g.collimator_deg);
    }
    Some((g, ap, beam.name.clone()))
}

impl SetupResult {
    /// The whole report as text.
    pub fn report(&self) -> String {
        let mut s = format!(
            "{}\nMethod: {}\nIsocentre: ({:.1}, {:.1}, {:.1}) mm, {}\nImages: {}\n{}",
            self.label,
            self.method.label(),
            self.isocenter.x,
            self.isocenter.y,
            self.isocenter.z,
            self.patient.describe(),
            self.image_labels.join(", "),
            self.result.report()
        );
        for f in &self.fields {
            s.push_str(&f.line());
            s.push('\n');
        }
        s.push_str("Matrix, room to CT (LPS, mm):\n");
        for row in self.matrix() {
            s.push_str(&format!(
                "{:>11.6} {:>11.6} {:>11.6} {:>11.4}\n",
                row[0], row[1], row[2], row[3]
            ));
        }
        s
    }

    /// The correction as a 4 × 4 matrix, room to CT, LPS millimetres: what
    /// the registration object carries.
    pub fn matrix(&self) -> [[f64; 4]; 4] {
        crate::registration::Mat4::from_rigid(&self.result.transform).0
    }

    /// The numbers as tab-separated rows (quantity, then values), for a
    /// spreadsheet.
    pub fn table(&self) -> String {
        let r = &self.result;
        let c = r.correction();
        let d = r.displacement;
        let f2 = |v: f64| format!("{v:.2}");
        let mut rows: Vec<Vec<String>> = vec![
            vec!["Setup".into(), self.label.clone()],
            vec!["Method".into(), self.method.label().into()],
            vec![
                "Correction translation (mm)".into(),
                f2(c[3]),
                f2(c[4]),
                f2(c[5]),
            ],
            vec![
                "Correction rotation (deg)".into(),
                f2(c[0]),
                f2(c[1]),
                f2(c[2]),
            ],
            vec![
                "Couch lat lng vrt (mm)".into(),
                f2(r.couch.lateral),
                f2(r.couch.longitudinal),
                f2(r.couch.vertical),
            ],
            vec![
                "Couch yaw pitch roll (deg)".into(),
                f2(r.couch.yaw),
                f2(r.couch.pitch),
                f2(r.couch.roll),
            ],
            vec![
                "Displacement translation (mm)".into(),
                f2(d[3]),
                f2(d[4]),
                f2(d[5]),
            ],
            vec![
                "Displacement rotation (deg)".into(),
                f2(d[0]),
                f2(d[1]),
                f2(d[2]),
            ],
        ];
        for v in &r.views {
            rows.push(vec![
                format!("{} {}", v.label, r.metric.tag()),
                format!("{:.3}", v.before),
                format!("{:.3}", v.after),
            ]);
        }
        if let Some(m) = &r.markers {
            rows.push(vec!["Marker RMS (mm)".into(), f2(m.rms_mm)]);
        }
        for f in &self.fields {
            if let (Some(e), Some(rel)) = (f.edge, f.relative_mm()) {
                rows.push(vec![
                    format!("{} field, anatomy, field to anatomy (mm)", f.beam),
                    f2(e.shift_mm[0]),
                    f2(e.shift_mm[1]),
                    f2(f.anatomy_mm[0]),
                    f2(f.anatomy_mm[1]),
                    f2(rel[0]),
                    f2(rel[1]),
                ]);
            }
        }
        for (k, row) in self.matrix().iter().enumerate() {
            let mut line = vec![format!("Matrix row {}", k + 1)];
            line.extend(row.iter().map(|v| format!("{v:.6}")));
            rows.push(line);
        }
        rows.iter()
            .map(|r| r.join("\t"))
            .collect::<Vec<_>>()
            .join("\n")
            + "\n"
    }

    /// The report as JSON, for a file or an MCP answer.
    pub fn json(&self) -> serde_json::Value {
        let r = &self.result;
        let round = |v: f64, k: f64| (v * k).round() / k;
        let six = |p: [f64; 6]| {
            serde_json::json!({
                "translation_mm": [round(p[3], 100.0), round(p[4], 100.0), round(p[5], 100.0)],
                "rotation_deg": [round(p[0], 100.0), round(p[1], 100.0), round(p[2], 100.0)],
            })
        };
        serde_json::json!({
            "label": self.label,
            "method": self.method.key(),
            "metric": r.metric.key(),
            "isocenter_mm": [round(self.isocenter.x, 10.0), round(self.isocenter.y, 10.0), round(self.isocenter.z, 10.0)],
            "patient": self.patient.describe(),
            "correction_lps": six(r.correction()),
            "displacement_lps": six(r.displacement),
            "couch": {
                "lateral_mm": round(r.couch.lateral, 100.0),
                "longitudinal_mm": round(r.couch.longitudinal, 100.0),
                "vertical_mm": round(r.couch.vertical, 100.0),
                "yaw_deg": round(r.couch.yaw, 100.0),
                "pitch_deg": round(r.couch.pitch, 100.0),
                "roll_deg": round(r.couch.roll, 100.0),
            },
            "views": r.views.iter().map(|v| serde_json::json!({
                "label": v.label,
                "before": round(v.before, 1000.0),
                "after": round(v.after, 1000.0),
                "inverted": v.inverted,
            })).collect::<Vec<_>>(),
            "markers": r.markers.as_ref().map(|m| serde_json::json!({
                "rms_mm": round(m.rms_mm, 100.0),
                "dof": match m.dof { Dof2d3d::Six => "six", Dof2d3d::Translations => "translations" },
                "residuals": m.residuals.iter().map(|x| serde_json::json!({
                    "marker": x.marker, "view": x.view, "residual_mm": round(x.residual_mm, 100.0), "picked": x.picked,
                })).collect::<Vec<_>>(),
                "triangulated": m.triangulated.iter().map(|t| serde_json::json!({
                    "marker": t.marker, "offset_mm": round(t.offset_mm, 100.0), "moved": t.migrated,
                })).collect::<Vec<_>>(),
                "not_found": m.missing,
            })),
            "fields": self.fields.iter().map(|f| serde_json::json!({
                "view": f.view,
                "beam": f.beam,
                "field_shift_mm": f.edge.map(|e| [round(e.shift_mm[0], 100.0), round(e.shift_mm[1], 100.0)]),
                "field_rms_mm": f.edge.map(|e| round(e.rms_mm, 100.0)),
                "anatomy_mm": [round(f.anatomy_mm[0], 100.0), round(f.anatomy_mm[1], 100.0)],
                "field_to_anatomy_mm": f.relative_mm().map(|r| [round(r[0], 100.0), round(r[1], 100.0)]),
            })).collect::<Vec<_>>(),
            "start": r.start,
            "evaluations": r.evaluations,
            "seconds": round(r.elapsed_secs, 10.0),
            "projector": (!r.device.is_empty()).then_some(&r.device),
            "matrix": self.matrix(),
            "warning": r.warning,
        })
    }

    /// The DRR of view `k` at the answer as a planar image, read the way the
    /// radiograph it is laid over reads (inverted with it), and its camera
    /// in the CT's coordinates.
    pub fn drr_planar(&self, k: usize, radiograph: Option<&PlanarImage>) -> Option<PlanarImage> {
        let v = self.result.views.get(k)?;
        let (lo, hi) = v
            .drr
            .data
            .iter()
            .fold((f32::MAX, f32::MIN), |(a, b), &x| (a.min(x), b.max(x)));
        // The working raster is the radiograph halved some times: the same
        // field of view, coarser pixels.
        let f = radiograph
            .map(|r| r.cols as f64 / v.drr.dims[0].max(1) as f64)
            .unwrap_or(1.0);
        let spacing = radiograph
            .map(|r| [r.spacing[0] * f, r.spacing[1] * f])
            .unwrap_or_else(|| v.camera.spacing());
        let data: Vec<f32> = if v.inverted {
            v.drr.data.iter().map(|x| lo + hi - x).collect()
        } else {
            v.drr.data.clone()
        };
        Some(PlanarImage {
            label: format!("DRR at setup · {}", v.label),
            modality: "RTIMAGE".into(),
            rows: v.drr.dims[1],
            cols: v.drr.dims[0],
            spacing,
            data,
            min_value: lo,
            max_value: hi,
            window: ((lo + hi) * 0.5, (hi - lo).max(1e-6)),
            info: vec![
                (
                    "Source".into(),
                    format!("DRR at the pose of {}", self.label),
                ),
                ("Correction".into(), self.result.correction_line()),
            ],
            frame_of_reference_uid: self.for_uid.clone(),
            camera: Some(v.camera.transformed(&self.result.transform)),
            frame_count: 1,
            ..Default::default()
        })
    }
}

/// Make the request and run it.
pub fn run(
    study: &LoadedStudy,
    views: &[ViewChoice],
    opts: &SetupOptions,
    rooms: Option<&RoomsFile>,
    sink: &dyn ProgressSink,
) -> Result<SetupResult> {
    if !study.has_volume() {
        bail!("the workspace has no CT");
    }
    if views.is_empty() {
        bail!("no radiograph chosen");
    }
    let iso = opts
        .isocenter
        .unwrap_or_else(|| default_isocenter(study, views));
    let frame = opts.patient.unwrap_or_else(|| default_patient(study));
    let clip = match &opts.clip_structure {
        Some(name) => Some(structure_box(study, name, opts.clip_margin_mm)?),
        None => opts
            .clip_half_mm
            .map(|h| ClipBox::around(opts.clip_center.unwrap_or(iso), h)),
    };
    let mut v2d = Vec::with_capacity(views.len());
    let mut apertures = Vec::with_capacity(views.len());
    for v in views {
        let cam = camera_of(study, v.image, &v.source, iso, frame, rooms)?;
        let img = &study.planar_images[v.image];
        // A portal image of a beam with an aperture compares the inside of
        // the field only, clear of its edge: the DRR has no field edge.
        let aperture = portal_aperture(study, v, iso, frame);
        let field = aperture
            .as_ref()
            .map(|(g, ap, _)| crate::portal::aperture_mask(g, ap, &cam, 5.0))
            .filter(|m| m.iter().filter(|&&x| x != 0).count() > 64);
        // The region drawn on the image, inside the field when there is one.
        let mask = if v.region.is_empty() {
            field
        } else {
            let mut m = region_mask([img.cols, img.rows], &v.region);
            if let Some(f) = &field {
                m.iter_mut().zip(f).for_each(|(a, b)| *a &= *b);
            }
            if m.iter().filter(|&&x| x != 0).count() <= 64 {
                bail!("{}: the region drawn covers too few pixels", img.label);
            }
            Some(m)
        };
        apertures.push(aperture);
        v2d.push(View2d {
            label: img.label.clone(),
            image: Image2d::new(img.data.clone(), [img.cols, img.rows]),
            camera: cam,
            mask,
            weight: v.weight,
            transmission: v.transmission,
        });
    }
    // Portal images alone are megavoltage images: their attenuation is the
    // electron density's.
    let all_mv = views.iter().all(|v| {
        study
            .planar_images
            .get(v.image)
            .is_some_and(|p| p.megavoltage)
    });
    let hu = if all_mv && opts.hu == HuMode::Water {
        HuMode::MegaVoltage
    } else {
        opts.hu
    };
    let mut req = Request2d3d::new(v2d, iso);
    req.patient = frame;
    req.clip = clip;
    req.clip_rays = opts.clip_rays;
    req.threshold_hu = opts.threshold_hu;
    req.hu = hu;
    req.metric = opts.metric;
    req.optimizer = opts.optimizer;
    req.levels = opts.levels;
    req.init = opts.init;
    req.dof = opts.dof;
    req.refine_gradient = opts.refine;
    req.conditioning = opts.conditioning;
    req.start = opts.start.clone();
    req.max_pixels = opts.max_pixels;
    req.step_mm = opts.step_mm;
    req.range_mm = opts.range_mm;
    req.range_deg = opts.range_deg;
    req.device = opts.device;
    let vol = &study.volume;
    let result = match opts.method {
        SetupMethod::Intensity => projective::register_2d3d(vol, &req, sink)?,
        SetupMethod::Markers | SetupMethod::Hybrid => {
            let found = study_markers(study, opts, clip.as_ref());
            if found.is_empty() {
                bail!(match opts.markers {
                    MarkerSource::Detect => "no marker found in the CT",
                    _ => "the study has no marker points of interest",
                });
            }
            // Markers placed by hand, by name, per view.
            let picks = views
                .iter()
                .map(|v| {
                    found
                        .iter()
                        .map(|m| {
                            v.picks
                                .iter()
                                .find(|(n, _)| n.eq_ignore_ascii_case(&m.name))
                                .map(|(_, p)| *p)
                        })
                        .collect()
                })
                .collect();
            let mut mreq = MarkerRequest::new(found, req.views.clone(), iso);
            mreq.picks = picks;
            mreq.patient = frame;
            mreq.start = opts.start.clone();
            mreq.dof = opts.dof;
            mreq.search_mm = opts.search_mm;
            mreq.marker_mm = opts.marker_mm;
            mreq.hu = hu;
            mreq.threshold_hu = opts.threshold_hu;
            mreq.max_pixels = opts.max_pixels.max(256);
            if let Some(c) = &clip {
                mreq.scale_mm = c.half_diagonal();
            }
            if opts.method == SetupMethod::Markers {
                markers::register_markers(vol, &mreq, sink)?
            } else {
                markers::register_hybrid(vol, &mreq, &req, sink)?
            }
        }
    };
    // The field edge of every portal image against its planned aperture,
    // and the anatomy's displacement in the same axes.
    let mut fields = Vec::new();
    for (k, a) in apertures.iter().enumerate() {
        let (Some((g, ap, beam)), Some(view)) = (a, req.views.get(k)) else {
            continue;
        };
        let axes = g.bld_axes();
        let moved = result.transform.unmap(iso) - iso;
        // The image as it was taken: the edge is a step of the signal.
        let image = &view.image;
        fields.push(FieldReport {
            view: k,
            beam: beam.clone(),
            outline: crate::portal::aperture_outline(g, ap, &view.camera),
            edge: crate::portal::field_edge(image, g, ap, &view.camera, 10.0),
            anatomy_mm: [moved.dot(axes[0]), moved.dot(axes[1])],
        });
    }
    let n = study.planar_images.len();
    let label = format!(
        "Setup · {}",
        views
            .iter()
            .filter(|v| v.image < n)
            .map(|v| study.planar_images[v.image].label.clone())
            .collect::<Vec<_>>()
            .join(" + ")
    );
    Ok(SetupResult {
        label,
        method: opts.method,
        result,
        images: views.iter().map(|v| v.image).collect(),
        image_labels: views
            .iter()
            .filter(|v| v.image < n)
            .map(|v| study.planar_images[v.image].label.clone())
            .collect(),
        series_uid: study
            .series
            .get(study.active_series)
            .map(|s| s.uid.clone())
            .unwrap_or_default(),
        for_uid: study.volume.frame_of_reference_uid.clone(),
        isocenter: iso,
        patient: frame,
        hu,
        threshold_hu: opts.threshold_hu,
        when: now_text(),
        adjusted: false,
        fields,
    })
}

/// The DRR of every view of `s` at `pose`, at the resolution the run
/// compared them at.
pub fn render_at(
    study: &LoadedStudy,
    s: &SetupResult,
    pose: &RigidTransform,
    sink: &dyn ProgressSink,
) -> Result<Vec<Image2d>> {
    let mu = crate::projection::MuVolume::prepare(&study.volume, s.hu, s.threshold_hu, None);
    let step = mu.finest_spacing().clamp(0.5, 2.0);
    s.result
        .views
        .iter()
        .map(|v| {
            let px = crate::projection::render_posed(&mu, &v.camera.transformed(pose), step, sink)?;
            Ok(Image2d::new(px, v.camera.dims))
        })
        .collect()
}

/// Move `s` to `pose` by hand: the readouts made again, and each view's DRR
/// (from [`render_at`]) and similarity at the new pose.
pub fn adjust(s: &mut SetupResult, pose: RigidTransform, drrs: Vec<Image2d>) {
    let (couch, displacement) = projective::readout(&pose, &s.patient);
    s.result.transform = pose;
    s.result.couch = couch;
    s.result.displacement = displacement;
    let radius = 7;
    for (v, d) in s.result.views.iter_mut().zip(drrs) {
        if d.dims == v.image.dims {
            v.after =
                s.result
                    .metric
                    .similarity(&v.image.data, &d.data, d.dims, Some(&v.mask), radius);
            v.drr = d;
        }
    }
    s.adjusted = true;
}

/// The DRRs of the chosen views at the start pose (the plan, or
/// `opts.start`), each read the way its radiograph reads and with the
/// radiograph's field: the check that the geometry is right before
/// anything is registered.
pub fn preview(
    study: &LoadedStudy,
    views: &[ViewChoice],
    opts: &SetupOptions,
    rooms: Option<&RoomsFile>,
    sink: &dyn ProgressSink,
) -> Result<Vec<PlanarImage>> {
    if !study.has_volume() {
        bail!("the workspace has no CT");
    }
    let iso = opts
        .isocenter
        .unwrap_or_else(|| default_isocenter(study, views));
    let frame = opts.patient.unwrap_or_else(|| default_patient(study));
    let pose = opts
        .start
        .as_ref()
        .map(|s| s.recentered(iso))
        .unwrap_or_else(|| RigidTransform::identity(iso));
    let mu = crate::projection::MuVolume::prepare(&study.volume, opts.hu, opts.threshold_hu, None);
    let step = mu.finest_spacing().clamp(0.5, 2.0);
    let mut out = Vec::with_capacity(views.len());
    for v in views {
        let cam = camera_of(study, v.image, &v.source, iso, frame, rooms)?;
        let img = &study.planar_images[v.image];
        let view = View2d {
            label: img.label.clone(),
            image: Image2d::new(img.data.clone(), [img.cols, img.rows]),
            camera: cam,
            mask: None,
            weight: 1.0,
            transmission: v.transmission,
        };
        let (work, mask, wcam) = projective::working_view(&view, opts.max_pixels, None)?;
        let inverted = projective::inverted(&mu, &work, &mask, &wcam, &pose, sink)?;
        let px = crate::projection::render_posed(&mu, &wcam.transformed(&pose), step, sink)?;
        let (lo, hi) = px
            .iter()
            .fold((f32::MAX, f32::MIN), |(a, b), &x| (a.min(x), b.max(x)));
        let data: Vec<f32> = if inverted {
            px.iter().map(|x| lo + hi - x).collect()
        } else {
            px
        };
        let f = img.cols as f64 / wcam.dims[0].max(1) as f64;
        out.push(PlanarImage {
            label: format!("DRR at plan · {}", img.label),
            modality: "RTIMAGE".into(),
            rows: wcam.dims[1],
            cols: wcam.dims[0],
            spacing: [img.spacing[0] * f, img.spacing[1] * f],
            data,
            min_value: lo,
            max_value: hi,
            window: ((lo + hi) * 0.5, (hi - lo).max(1e-6)),
            info: vec![("Source".into(), "DRR at the plan's pose".into())],
            frame_of_reference_uid: study.volume.frame_of_reference_uid.clone(),
            camera: Some(wcam.transformed(&pose)),
            frame_count: 1,
            ..Default::default()
        });
    }
    Ok(out)
}

/// The outline of a structure (`mask` on the CT's grid) as the camera sees
/// it at `pose`: where any ray crosses it, contoured, as line segments in
/// the camera's raster scaled to `0..1` across and down.
pub fn silhouette(
    vol: &crate::volume::Volume,
    mask: &[u8],
    cam: &ProjectionGeometry,
    pose: &RigidTransform,
) -> Vec<([f32; 2], [f32; 2])> {
    let Some(mu) = crate::projection::MuVolume::from_mask(vol, mask) else {
        return Vec::new();
    };
    let step = mu.finest_spacing().clamp(0.5, 3.0);
    let Ok(px) =
        crate::projection::render_posed(&mu, &cam.transformed(pose), step, &crate::progress::Quiet)
    else {
        return Vec::new();
    };
    let [w, h] = cam.dims;
    let field: Vec<f32> = px.iter().map(|v| (*v > 0.0) as u8 as f32).collect();
    crate::render::marching_squares(&field, w, h, 0.5)
        .into_iter()
        .map(|(a, b)| {
            let n = |p: [f32; 2]| [(p[0] + 0.5) / w as f32, (p[1] + 0.5) / h as f32];
            (n(a), n(b))
        })
        .collect()
}

/// `YYYY-MM-DD HH:MM` (UTC).
fn now_text() -> String {
    let (d, t) = crate::dicom_export::today();
    if d.len() == 8 && t.len() >= 4 {
        format!(
            "{}-{}-{} {}:{}",
            &d[..4],
            &d[4..6],
            &d[6..8],
            &t[..2],
            &t[2..4]
        )
    } else {
        format!("{d} {t}")
    }
}

/// The study as the patient was found: the CT moved by the displacement
/// `T⁻¹` (structures, doses and isocentres with it), labelled as a
/// positioned copy. Paired with the original through `T` - the copy is the
/// fixed image, the planning CT the moving one - it fuses, propagates and
/// compares like any registered pair.
pub fn positioned_copy(
    study: &LoadedStudy,
    setup: &SetupResult,
    progress: &crate::progress::Progress,
) -> LoadedStudy {
    let mut copy = moved_copy(study, &setup.result.transform.inverse(), progress);
    let name = format!("Positioned · {}", setup.label);
    copy.meta.study_description = name.clone();
    for s in &mut copy.series {
        s.description = name.clone();
        s.study_description = "Positioned".into();
    }
    copy
}

/// `study` moved by `displacement` (planned point → where it is found),
/// through the simulator: the CT resampled, structures, doses and plan
/// isocentres carried along.
pub fn moved_copy(
    study: &LoadedStudy,
    displacement: &RigidTransform,
    progress: &crate::progress::Progress,
) -> LoadedStudy {
    let vol = &study.volume;
    let centre = vol.voxel_to_patient(
        (vol.dims[0] as f64 - 1.0) * 0.5,
        (vol.dims[1] as f64 - 1.0) * 0.5,
        (vol.dims[2] as f64 - 1.0) * 0.5,
    );
    // The simulator turns about the volume's centre.
    let p = displacement.recentered(centre).params();
    let params = crate::simulate::SimParams {
        translation: [p[3], p[4], p[5]],
        rotation_deg: [p[0].to_degrees(), p[1].to_degrees(), p[2].to_degrees()],
        bump_amp: [0.0; 3],
        bump_center: [0.0; 3],
        bump_sigma: 25.0,
    };
    crate::simulate::generate_transformed_study(study, &params, progress)
}

/// Write a setup out: the correction as a rigid Spatial Registration (the
/// room's frame - the radiographs' own when they name one, else a new one
/// the description names - onto the planning CT's), the DRRs at the answer
/// as RT Images, and the report as JSON. Returns the files written.
pub fn write_files(
    study: &LoadedStudy,
    s: &SetupResult,
    dir: &std::path::Path,
) -> Result<Vec<std::path::PathBuf>> {
    use crate::dicom_export::{self, ObjectMeta};
    std::fs::create_dir_all(dir).map_err(|e| anyhow!("create {}: {e}", dir.display()))?;
    let meta = |label: &str, desc: &str| ObjectMeta::of_study(study, label, desc);
    let mut files = Vec::new();
    let source_for = s
        .images
        .iter()
        .filter_map(|i| study.planar_images.get(*i))
        .map(|p| p.frame_of_reference_uid.clone())
        .find(|u| !u.is_empty() && *u != s.for_uid)
        .unwrap_or_else(dicom_export::new_uid);
    let m = s.matrix();
    let flat: [f64; 16] = std::array::from_fn(|k| m[k / 4][k % 4]);
    let reg = dir.join("setup_reg.dcm");
    dicom_export::write_rigid_registration(
        &reg,
        &flat,
        &source_for,
        &s.for_uid,
        &meta("SETUP", &format!("{}: room to planning CT", s.label)),
    )?;
    files.push(reg);
    for k in 0..s.result.views.len() {
        let radiograph = s.images.get(k).and_then(|i| study.planar_images.get(*i));
        if let Some(img) = s.drr_planar(k, radiograph) {
            let path = dir.join(format!("setup_drr_{}.dcm", k + 1));
            dicom_export::write_rt_image(&path, &img, &meta("DRR", "DRR at the setup pose"))?;
            files.push(path);
        }
    }
    let json_path = dir.join("setup_report.json");
    let text = serde_json::to_string_pretty(&s.json())?;
    std::fs::write(&json_path, text).map_err(|e| anyhow!("write {}: {e}", json_path.display()))?;
    files.push(json_path);
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn method_keys_read_back() {
        for m in SetupMethod::ALL {
            assert_eq!(SetupMethod::from_key(m.key()), Some(m));
        }
    }

    #[test]
    fn a_drawn_region_is_the_pixels_inside_it() {
        // A rectangle from (1, 1) to (3, 2): the pixels whose centres are
        // on it or inside.
        let m = region_mask(
            [5, 4],
            &[vec![[1.0, 1.0], [3.0, 1.0], [3.0, 2.0], [1.0, 2.0]]],
        );
        let on: Vec<(usize, usize)> = (0..20)
            .filter(|&o| m[o] != 0)
            .map(|o| (o % 5, o / 5))
            .collect();
        assert_eq!(on, vec![(1, 1), (2, 1), (3, 1)]);
        // A triangle, and two polygons together.
        let tri = vec![[-0.5, -0.5], [4.5, -0.5], [-0.5, 3.5]];
        let m = region_mask([5, 4], std::slice::from_ref(&tri));
        assert_eq!(m[0], 1);
        assert_eq!(m[4], 0, "the far corner is outside the hypotenuse");
        let both = region_mask([5, 4], &[tri, vec![[4.0, 3.0], [4.5, 3.0], [4.5, 3.5]]]);
        assert_eq!(
            both.iter().filter(|&&v| v != 0).count(),
            m.iter().filter(|&&v| v != 0).count() + 1
        );
        // Too few corners, or nothing on the image: nothing.
        assert!(region_mask([5, 4], &[vec![[0.0, 0.0], [3.0, 3.0]]])
            .iter()
            .all(|&v| v == 0));
        assert!(
            region_mask([5, 4], &[vec![[9.0, 9.0], [12.0, 9.0], [12.0, 12.0]]])
                .iter()
                .all(|&v| v == 0)
        );
    }
}
