//! RT Plan (RTPLAN / RT Ion Plan) parsing: prescription, fractionation and
//! per-beam summary, supporting both photon (BeamSequence) and ion / proton
//! (IonBeamSequence) plans.

use std::path::Path;

use anyhow::{Context, Result};
use dicom_dictionary_std::tags;

use crate::geometry::Vec3;
use crate::loader::{f64_of, f64s_of, i32_of, items_of, str_of};

#[derive(Debug, Clone)]
pub struct BeamInfo {
    pub number: i32,
    pub name: String,
    pub radiation_type: String,
    pub delivery_type: String,
    pub scan_mode: String,
    pub gantry_angle: Option<f64>,
    pub couch_angle: Option<f64>,
    pub isocenter: Option<Vec3>,
    pub energy_min: Option<f64>,
    pub energy_max: Option<f64>,
    pub n_control_points: usize,
    pub meterset: Option<f64>,
    pub beam_dose: Option<f64>,
    /// Beam Limiting Device Angle at the first control point, degrees.
    pub collimator_angle: Option<f64>,
    /// The jaws and the MLC at the first control point (photon beams).
    pub aperture: Option<Aperture>,
}

/// A photon beam's aperture at its first control point: jaws and MLC, in
/// the beam limiting device's own axes, millimetres at the isocentre plane
/// (Leaf/Jaw Positions, PS3.3 C.8.8.14).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Aperture {
    /// `[x1, x2]` and `[y1, y2]`; `None` for a beam without that pair.
    pub jaw_x: Option<[f64; 2]>,
    pub jaw_y: Option<[f64; 2]>,
    pub mlc: Option<Mlc>,
}

/// A multileaf collimator's leaves at one control point.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Mlc {
    /// The leaves travel along `Xb` (MLCX) or `Yb` (MLCY).
    pub along_x: bool,
    /// `n + 1` leaf boundaries across the travel, mm.
    pub boundaries: Vec<f64>,
    /// Per pair, the A bank's (negative side) and the B bank's positions.
    pub pairs: Vec<[f64; 2]>,
}

impl Aperture {
    /// The open field as rectangles `[x0, x1, y0, y1]`: each leaf pair's
    /// opening within the jaws, or the jaws alone.
    pub fn rectangles(&self) -> Vec<[f64; 4]> {
        const OPEN: [f64; 2] = [-1e4, 1e4];
        let jx = self.jaw_x.unwrap_or(OPEN);
        let jy = self.jaw_y.unwrap_or(OPEN);
        let clip = |r: [f64; 4]| {
            let c = [
                r[0].max(jx[0]),
                r[1].min(jx[1]),
                r[2].max(jy[0]),
                r[3].min(jy[1]),
            ];
            (c[1] > c[0] && c[3] > c[2]).then_some(c)
        };
        match &self.mlc {
            None if self.jaw_x.is_some() || self.jaw_y.is_some() => {
                clip([jx[0], jx[1], jy[0], jy[1]]).into_iter().collect()
            }
            None => Vec::new(),
            Some(m) => m
                .pairs
                .iter()
                .enumerate()
                .filter_map(|(i, p)| {
                    let (lo, hi) = (*m.boundaries.get(i)?, *m.boundaries.get(i + 1)?);
                    clip(if m.along_x {
                        [p[0], p[1], lo, hi]
                    } else {
                        [lo, hi, p[0], p[1]]
                    })
                })
                .collect(),
        }
    }

    /// Whether the point `(x, y)` of the isocentre plane is in the field.
    pub fn contains(&self, x: f64, y: f64) -> bool {
        self.rectangles()
            .iter()
            .any(|r| x >= r[0] && x <= r[1] && y >= r[2] && y <= r[3])
    }
}

/// The aperture of a beam from its Beam Limiting Device Sequence (the leaf
/// boundaries) and its first control point's positions.
fn parse_aperture(
    beam: &dicom_object::InMemDicomObject,
    cp0: &dicom_object::InMemDicomObject,
) -> Option<Aperture> {
    let mut boundaries: Vec<(String, Vec<f64>)> = Vec::new();
    if let Some(devs) = items_of(beam, tags::BEAM_LIMITING_DEVICE_SEQUENCE) {
        for d in devs {
            let ty = str_of(d, tags::RT_BEAM_LIMITING_DEVICE_TYPE).unwrap_or_default();
            if let Some(b) = f64s_of(d, tags::LEAF_POSITION_BOUNDARIES) {
                boundaries.push((ty, b));
            }
        }
    }
    let positions = items_of(cp0, tags::BEAM_LIMITING_DEVICE_POSITION_SEQUENCE)?;
    let mut ap = Aperture::default();
    for p in positions {
        let ty = str_of(p, tags::RT_BEAM_LIMITING_DEVICE_TYPE).unwrap_or_default();
        let Some(v) = f64s_of(p, tags::LEAF_JAW_POSITIONS) else {
            continue;
        };
        match ty.as_str() {
            "X" | "ASYMX" if v.len() >= 2 => ap.jaw_x = Some([v[0], v[1]]),
            "Y" | "ASYMY" if v.len() >= 2 => ap.jaw_y = Some([v[0], v[1]]),
            "MLCX" | "MLCY" => {
                let n = v.len() / 2;
                let Some((_, b)) = boundaries.iter().find(|(t, _)| *t == ty) else {
                    continue;
                };
                if n == 0 || b.len() != n + 1 {
                    continue;
                }
                ap.mlc = Some(Mlc {
                    along_x: ty == "MLCX",
                    boundaries: b.clone(),
                    pairs: (0..n).map(|i| [v[i], v[n + i]]).collect(),
                });
            }
            _ => {}
        }
    }
    (ap.jaw_x.is_some() || ap.jaw_y.is_some() || ap.mlc.is_some()).then_some(ap)
}

#[derive(Debug, Clone, Default)]
pub struct PlanInfo {
    pub label: String,
    pub name: String,
    pub date: String,
    pub plan_kind: String, // "Photon/Electron" or "Ion"
    pub n_fractions: Option<i32>,
    pub target_prescription_dose: Option<f64>,
    /// SOP Instance UID of this plan (referenced by RTDOSE objects).
    pub sop_instance_uid: String,
    /// Series Instance UID of this object's own series. Kept so that an
    /// export asked to preserve the identifiers really preserves them: an
    /// archive files an instance under its series, and the same instance
    /// arriving under a different one is a different object to it.
    pub series_instance_uid: String,
    /// Study this plan belongs to.
    pub study_uid: String,
    /// SOP Instance UID of the structure set the plan was made on.
    pub referenced_structset_uid: String,
    pub beams: Vec<BeamInfo>,
}

pub fn load(path: &Path) -> Result<PlanInfo> {
    let obj = crate::dicomfile::open_full(path)
        .with_context(|| format!("open RTPLAN {}", path.display()))?;

    let mut plan = PlanInfo {
        label: str_of(&obj, tags::RT_PLAN_LABEL).unwrap_or_default(),
        name: str_of(&obj, tags::RT_PLAN_NAME).unwrap_or_default(),
        date: str_of(&obj, tags::RT_PLAN_DATE).unwrap_or_default(),
        sop_instance_uid: str_of(&obj, tags::SOP_INSTANCE_UID).unwrap_or_default(),
        series_instance_uid: str_of(&obj, tags::SERIES_INSTANCE_UID).unwrap_or_default(),
        study_uid: str_of(&obj, tags::STUDY_INSTANCE_UID).unwrap_or_default(),
        referenced_structset_uid: items_of(&obj, tags::REFERENCED_STRUCTURE_SET_SEQUENCE)
            .and_then(|items| items.first())
            .and_then(|it| str_of(it, tags::REFERENCED_SOP_INSTANCE_UID))
            .unwrap_or_default(),
        ..Default::default()
    };
    if plan.label.is_empty() {
        plan.label = path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
    }

    // Prescription.
    if let Some(items) = items_of(&obj, tags::DOSE_REFERENCE_SEQUENCE) {
        for it in items {
            if let Some(d) = f64_of(it, tags::TARGET_PRESCRIPTION_DOSE) {
                plan.target_prescription_dose = Some(d);
                break;
            }
        }
    }

    // Fractionation + per-beam meterset/dose references.
    let mut beam_msets: Vec<(i32, Option<f64>, Option<f64>)> = Vec::new();
    if let Some(items) = items_of(&obj, tags::FRACTION_GROUP_SEQUENCE) {
        if let Some(first) = items.first() {
            plan.n_fractions = i32_of(first, tags::NUMBER_OF_FRACTIONS_PLANNED);
            if let Some(rbs) = items_of(first, tags::REFERENCED_BEAM_SEQUENCE) {
                for rb in rbs {
                    let num = i32_of(rb, tags::REFERENCED_BEAM_NUMBER).unwrap_or(-1);
                    let mset = f64_of(rb, tags::BEAM_METERSET);
                    let bdose = f64_of(rb, tags::BEAM_DOSE);
                    beam_msets.push((num, mset, bdose));
                }
            }
        }
    }

    // Beams: photon/electron plans use BeamSequence + ControlPointSequence,
    // ion (proton/carbon) plans use IonBeamSequence + IonControlPointSequence.
    let (beam_items, cp_tag, kind) = if let Some(items) = items_of(&obj, tags::ION_BEAM_SEQUENCE) {
        (Some(items), tags::ION_CONTROL_POINT_SEQUENCE, "Ion")
    } else if let Some(items) = items_of(&obj, tags::BEAM_SEQUENCE) {
        (Some(items), tags::CONTROL_POINT_SEQUENCE, "Photon/Electron")
    } else {
        (None, tags::CONTROL_POINT_SEQUENCE, "")
    };
    plan.plan_kind = kind.to_string();

    if let Some(items) = beam_items {
        for b in items {
            let number = i32_of(b, tags::BEAM_NUMBER).unwrap_or(-1);
            let mut info = BeamInfo {
                number,
                name: str_of(b, tags::BEAM_NAME)
                    .or_else(|| str_of(b, tags::BEAM_DESCRIPTION))
                    .unwrap_or_else(|| format!("Beam {number}")),
                radiation_type: str_of(b, tags::RADIATION_TYPE).unwrap_or_default(),
                delivery_type: str_of(b, tags::TREATMENT_DELIVERY_TYPE).unwrap_or_default(),
                scan_mode: str_of(b, tags::SCAN_MODE).unwrap_or_default(),
                gantry_angle: None,
                couch_angle: None,
                isocenter: None,
                energy_min: None,
                energy_max: None,
                n_control_points: 0,
                meterset: None,
                beam_dose: None,
                collimator_angle: None,
                aperture: None,
            };

            if let Some(cps) = items_of(b, cp_tag) {
                info.n_control_points = cps.len();
                for (ci, cp) in cps.iter().enumerate() {
                    if ci == 0 {
                        info.collimator_angle = f64_of(cp, tags::BEAM_LIMITING_DEVICE_ANGLE);
                        if kind != "Ion" {
                            info.aperture = parse_aperture(b, cp);
                        }
                        info.gantry_angle = f64_of(cp, tags::GANTRY_ANGLE);
                        info.couch_angle = f64_of(cp, tags::PATIENT_SUPPORT_ANGLE);
                        info.isocenter = f64s_of(cp, tags::ISOCENTER_POSITION)
                            .filter(|v| v.len() >= 3)
                            .map(|v| Vec3::from_slice(&v));
                    }
                    if let Some(e) = f64_of(cp, tags::NOMINAL_BEAM_ENERGY) {
                        info.energy_min = Some(info.energy_min.map_or(e, |m: f64| m.min(e)));
                        info.energy_max = Some(info.energy_max.map_or(e, |m: f64| m.max(e)));
                    }
                }
            }

            if let Some((_, mset, bdose)) = beam_msets.iter().find(|(n, _, _)| *n == number) {
                info.meterset = *mset;
                info.beam_dose = *bdose;
            }

            plan.beams.push(info);
        }
    }

    Ok(plan)
}
