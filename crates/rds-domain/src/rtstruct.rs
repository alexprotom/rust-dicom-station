//! RT Structure Set (RTSTRUCT) parsing: ROI names, colors and contours.
//!
//! Two things a structure set needs are read with fallbacks, because some
//! planning systems write them where they do not belong (RayStation's
//! "RS: Unapproved" exports, for one):
//!
//! * the Frame of Reference UID, first non-empty of: the first item of
//!   `ReferencedFrameOfReferenceSequence` (3006,0010), where the standard
//!   puts it; any `ReferencedFrameOfReferenceUID` of the
//!   `StructureSetROISequence`; a top-level `FrameOfReferenceUID`
//!   (0020,0052), which the IOD does not have; and last the UID found as
//!   `ReferencedSOPInstanceUID` (0008,1155) inside that first item, which
//!   is taken only when it is the frame of reference of the series the set
//!   references ([`Loaded::settle`]), never on its own;
//! * the ROI names: from `StructureSetROISequence`, which is Type 1 but
//!   sometimes absent, and else from the `ROIObservationLabel` of the
//!   `RTROIObservationsSequence`, before "ROI n".
//!
//! Whatever had to be patched is reported in [`Loaded::notes`], one load
//! warning per file; the export writes a conforming set either way.

use std::path::Path;

use anyhow::{Context, Result};
use dicom_dictionary_std::tags;
use dicom_object::InMemDicomObject;

use crate::geometry::Vec3;
use crate::loader::{f64s_of, i32_of, items_of, str_of};

#[derive(Clone)]
pub struct Contour {
    /// Points in patient coordinates (mm). Closed planar contours are not
    /// explicitly closed (last point != first); rendering closes them.
    pub points: Vec<Vec3>,
    /// CLOSED_PLANAR, OPEN_PLANAR, POINT, …
    pub geometric_type: String,
}

#[derive(Clone)]
pub struct Roi {
    pub number: i32,
    pub name: String,
    pub color: [u8; 3],
    /// RT ROI Interpreted Type: PTV, CTV, GTV, ORGAN, EXTERNAL, AVOIDANCE, …
    pub roi_type: String,
    /// ROI Description (3006,0028). Free text in the standard, and the one
    /// place a *derived* structure can keep its recipe so that it survives
    /// export and re-import (see [`crate::derived`]).
    pub description: String,
    pub contours: Vec<Contour>,
}

impl Roi {
    /// Whether this is a *point of interest* rather than a volume: an ROI
    /// whose geometry is one `POINT` contour.
    ///
    /// RTSTRUCT has no separate object for a marker, an isocentre or a
    /// reference point - they are ROIs with a point where the contours
    /// would be, which is why this is a question about the geometry rather
    /// than a flag.
    pub fn is_poi(&self) -> bool {
        !self.contours.is_empty()
            && self
                .contours
                .iter()
                .all(|c| c.geometric_type == "POINT" || c.points.len() == 1)
    }

    /// Where the point is, in patient coordinates.
    pub fn point(&self) -> Option<Vec3> {
        self.is_poi()
            .then(|| self.contours.first()?.points.first().copied())
            .flatten()
    }

    /// Put the point there, making this ROI a point of interest if it was
    /// empty. An ROI that already carries contours is left alone: a
    /// structure is not turned into a marker by accident.
    pub fn set_point(&mut self, p: Vec3) -> bool {
        if !self.contours.is_empty() && !self.is_poi() {
            return false;
        }
        self.contours = vec![Contour {
            points: vec![p],
            geometric_type: "POINT".into(),
        }];
        true
    }

    /// The localization point of its structure set: the one POI a planning
    /// system uses to line the patient up.
    pub fn is_localization(&self) -> bool {
        self.is_poi() && self.roi_type == "ISOCENTER"
    }
}

#[derive(Clone)]
pub struct StructureSet {
    pub label: String,
    pub frame_of_reference_uid: String,
    /// SOP Instance UID of this structure set (referenced by RTPLANs).
    pub sop_instance_uid: String,
    /// Series Instance UID of this object's own series. Kept so that an
    /// export asked to preserve the identifiers really preserves them: an
    /// archive files an instance under its series, and the same instance
    /// arriving under a different one is a different object to it.
    pub series_instance_uid: String,
    /// Study this structure set belongs to.
    pub study_uid: String,
    /// Image series the contours were drawn on
    /// (ReferencedFrameOfReference ▶ RTReferencedStudy ▶ RTReferencedSeries).
    pub referenced_series_uid: String,
    /// Source file name (disambiguates multiple sets with equal labels).
    pub file_name: String,
    /// Read-only: the geometry of this set may not be edited.
    ///
    /// This is Approval Status (300E,0002) by another name - a set that
    /// arrives `APPROVED` opens locked, and a set locked here is written
    /// back as approved. It is a guard against a slip of the hand, not an
    /// electronic signature: there is no user, no password and no audit
    /// trail behind it, and the interface says so.
    pub locked: bool,
    pub rois: Vec<Roi>,
}

/// ROI Observation Label (3006,0085). Retired from the standard, which is
/// why the dictionary marks it deprecated, but still written: some systems
/// keep the only copy of an ROI's name there.
const ROI_OBSERVATION_LABEL: dicom_core::Tag = dicom_core::Tag(0x3006, 0x0085);

/// Fallback palette for ROIs without a display color.
const PALETTE: &[[u8; 3]] = &[
    [230, 25, 75],
    [60, 180, 75],
    [255, 225, 25],
    [0, 130, 200],
    [245, 130, 48],
    [145, 30, 180],
    [70, 240, 240],
    [240, 50, 230],
    [210, 245, 60],
    [250, 190, 190],
    [0, 128, 128],
    [170, 110, 40],
];

/// A structure set as read from a file, with what had to be made good on
/// the way.
pub struct Loaded {
    pub set: StructureSet,
    /// What the file lacked and where it was found instead, as phrases for
    /// the load's warnings ("no Structure Set ROI Sequence - ROI names taken
    /// from RT ROI Observations"). Empty for a conforming file.
    pub notes: Vec<String>,
    /// A Frame of Reference UID found only as `ReferencedSOPInstanceUID`
    /// inside `ReferencedFrameOfReferenceSequence`, waiting for
    /// [`Loaded::settle`] to compare it with the referenced series.
    pub misplaced_for_uid: Option<String>,
}

impl Loaded {
    /// Decide on a frame of reference the file gave only in the wrong tag:
    /// taken when it is `series_for`, the frame of reference of the series
    /// the set references, and dropped otherwise (a UID in that tag may
    /// name anything). Without a series to compare it with it is dropped
    /// too. Does nothing when the set already has its frame of reference.
    pub fn settle(&mut self, series_for: Option<&str>) {
        let Some(uid) = self.misplaced_for_uid.take() else {
            return;
        };
        if !self.set.frame_of_reference_uid.is_empty() {
            return;
        }
        if series_for.is_some_and(|f| !f.is_empty() && f == uid) {
            self.notes.push(
                "names its frame of reference only as the Referenced SOP Instance UID of its \
                 Referenced Frame of Reference Sequence - used, as it is the referenced series' own"
                    .into(),
            );
            self.set.frame_of_reference_uid = uid;
        } else {
            self.notes.push(
                "has no Frame of Reference UID (the Referenced SOP Instance UID of its \
                 Referenced Frame of Reference Sequence is not the referenced series' frame \
                 of reference and is not used)"
                    .into(),
            );
        }
    }
}

pub fn load(path: &Path) -> Result<StructureSet> {
    Ok(load_noted(path)?.set)
}

/// [`load`], keeping the notes on what was patched and a misplaced frame of
/// reference for [`Loaded::settle`].
pub fn load_noted(path: &Path) -> Result<Loaded> {
    let obj = crate::dicomfile::open_full(path)
        .with_context(|| format!("open RTSTRUCT {}", path.display()))?;
    let file_name = path
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    Ok(read(&obj, file_name))
}

/// The frame of reference of a structure set, by the fallbacks of the
/// module docs: the UID, a note when it did not come from where the
/// standard puts it, and the misplaced candidate when nothing else had one.
fn frame_of_reference(obj: &InMemDicomObject) -> (String, Option<&'static str>, Option<String>) {
    let first_ref =
        items_of(obj, tags::REFERENCED_FRAME_OF_REFERENCE_SEQUENCE).and_then(|items| items.first());
    if let Some(uid) = first_ref.and_then(|it| str_of(it, tags::FRAME_OF_REFERENCE_UID)) {
        return (uid, None, None);
    }
    if let Some(uid) = items_of(obj, tags::STRUCTURE_SET_ROI_SEQUENCE).and_then(|items| {
        items
            .iter()
            .find_map(|it| str_of(it, tags::REFERENCED_FRAME_OF_REFERENCE_UID))
    }) {
        return (
            uid,
            Some("names its frame of reference only in the Structure Set ROI Sequence"),
            None,
        );
    }
    if let Some(uid) = str_of(obj, tags::FRAME_OF_REFERENCE_UID) {
        return (
            uid,
            Some(
                "names no frame of reference in its Referenced Frame of Reference Sequence - \
                 the top-level Frame of Reference UID is used",
            ),
            None,
        );
    }
    let misplaced = first_ref.and_then(|it| str_of(it, tags::REFERENCED_SOP_INSTANCE_UID));
    (String::new(), None, misplaced)
}

/// Read a structure set out of a DICOM object already in memory.
pub fn read(obj: &InMemDicomObject, file_name: String) -> Loaded {
    let mut notes: Vec<String> = Vec::new();

    let label = str_of(obj, tags::STRUCTURE_SET_LABEL)
        .or_else(|| str_of(obj, tags::STRUCTURE_SET_NAME))
        .unwrap_or_else(|| "Structure Set".into());

    let (frame_of_reference_uid, for_note, misplaced_for_uid) = frame_of_reference(obj);
    notes.extend(for_note.map(String::from));

    // Which image series the contours reference (for study-tree links).
    let referenced_series_uid = items_of(obj, tags::REFERENCED_FRAME_OF_REFERENCE_SEQUENCE)
        .and_then(|items| items.first())
        .and_then(|it| items_of(it, tags::RT_REFERENCED_STUDY_SEQUENCE))
        .and_then(|items| items.first())
        .and_then(|it| items_of(it, tags::RT_REFERENCED_SERIES_SEQUENCE))
        .and_then(|items| items.first())
        .and_then(|it| str_of(it, tags::SERIES_INSTANCE_UID))
        .unwrap_or_default();

    // ROI number -> (name, description)
    let mut names: Vec<(i32, String)> = Vec::new();
    let mut descriptions: Vec<(i32, String)> = Vec::new();
    let roi_sequence = items_of(obj, tags::STRUCTURE_SET_ROI_SEQUENCE);
    if let Some(items) = roi_sequence {
        for it in items {
            let number = i32_of(it, tags::ROI_NUMBER).unwrap_or(-1);
            let name = str_of(it, tags::ROI_NAME).unwrap_or_else(|| format!("ROI {number}"));
            names.push((number, name));
            if let Some(d) = str_of(it, tags::ROI_DESCRIPTION) {
                descriptions.push((number, d));
            }
        }
    }

    // ROI number -> interpreted type, and the observation label: the name
    // of last resort for a set that lists its ROIs nowhere else.
    let mut types: Vec<(i32, String)> = Vec::new();
    let mut labels: Vec<(i32, String)> = Vec::new();
    if let Some(items) = items_of(obj, tags::RTROI_OBSERVATIONS_SEQUENCE) {
        for it in items {
            let number = i32_of(it, tags::REFERENCED_ROI_NUMBER).unwrap_or(-1);
            let t = str_of(it, tags::RTROI_INTERPRETED_TYPE).unwrap_or_default();
            types.push((number, t));
            if let Some(l) = str_of(it, ROI_OBSERVATION_LABEL) {
                labels.push((number, l));
            }
        }
    }

    let mut rois = Vec::new();
    let mut from_labels = 0usize;
    if let Some(items) = items_of(obj, tags::ROI_CONTOUR_SEQUENCE) {
        for (idx, it) in items.iter().enumerate() {
            let number = i32_of(it, tags::REFERENCED_ROI_NUMBER).unwrap_or(-1);
            let name = names
                .iter()
                .find(|(n, _)| *n == number)
                .map(|(_, s)| s.clone())
                .or_else(|| {
                    let l = labels.iter().find(|(n, _)| *n == number)?;
                    from_labels += 1;
                    Some(l.1.clone())
                })
                .unwrap_or_else(|| format!("ROI {number}"));
            let roi_type = types
                .iter()
                .find(|(n, _)| *n == number)
                .map(|(_, s)| s.clone())
                .unwrap_or_default();
            let color = f64s_of(it, tags::ROI_DISPLAY_COLOR)
                .filter(|v| v.len() >= 3)
                .map(|v| {
                    [
                        v[0].clamp(0.0, 255.0) as u8,
                        v[1].clamp(0.0, 255.0) as u8,
                        v[2].clamp(0.0, 255.0) as u8,
                    ]
                })
                .unwrap_or(PALETTE[idx % PALETTE.len()]);

            let mut contours = Vec::new();
            if let Some(citems) = items_of(it, tags::CONTOUR_SEQUENCE) {
                for c in citems {
                    let geometric_type = str_of(c, tags::CONTOUR_GEOMETRIC_TYPE)
                        .unwrap_or_else(|| "CLOSED_PLANAR".into());
                    let Some(data) = f64s_of(c, tags::CONTOUR_DATA) else {
                        continue;
                    };
                    if data.len() < 3 {
                        continue;
                    }
                    let points: Vec<Vec3> = data
                        .as_chunks::<3>()
                        .0
                        .iter()
                        .map(|&[x, y, z]| Vec3::new(x, y, z))
                        .collect();
                    contours.push(Contour {
                        points,
                        geometric_type,
                    });
                }
            }

            let description = descriptions
                .iter()
                .find(|(n, _)| *n == number)
                .map(|(_, s)| s.clone())
                .unwrap_or_default();
            rois.push(Roi {
                number,
                name,
                color,
                roi_type,
                description,
                contours,
            });
        }
    }

    if roi_sequence.is_none() && !rois.is_empty() {
        notes.push(if from_labels > 0 {
            "has no Structure Set ROI Sequence - ROI names taken from RT ROI Observations".into()
        } else {
            "has no Structure Set ROI Sequence - ROIs named by number".into()
        });
    } else if from_labels > 0 {
        notes.push(format!(
            "lists {from_labels} ROI(s) only in RT ROI Observations - their names are taken \
             from there"
        ));
    }

    // Stable, user-friendly ordering: external/body first, then targets, then rest alphabetically.
    rois.sort_by(|a, b| {
        fn rank(r: &Roi) -> u8 {
            match r.roi_type.as_str() {
                "EXTERNAL" => 0,
                "PTV" => 1,
                "CTV" => 2,
                "GTV" => 3,
                _ => 4,
            }
        }
        rank(a)
            .cmp(&rank(b))
            .then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase()))
    });

    let set = StructureSet {
        label,
        frame_of_reference_uid,
        sop_instance_uid: str_of(obj, tags::SOP_INSTANCE_UID).unwrap_or_default(),
        series_instance_uid: str_of(obj, tags::SERIES_INSTANCE_UID).unwrap_or_default(),
        study_uid: str_of(obj, tags::STUDY_INSTANCE_UID).unwrap_or_default(),
        referenced_series_uid,
        file_name,
        // A structure set someone has approved opens read-only, which is
        // what approval is for.
        locked: str_of(obj, tags::APPROVAL_STATUS).as_deref() == Some("APPROVED"),
        rois,
    };
    Loaded {
        set,
        notes,
        misplaced_for_uid,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dicom_export::{put_ds, put_is, put_seq, put_str};
    use dicom_core::VR;

    /// A structure set in memory: one ROI (number 1) with one contour, its
    /// name and frame of reference only where the arguments put them.
    struct Spec<'a> {
        /// `ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID`.
        ref_for: Option<&'a str>,
        /// ... and its misplaced `ReferencedSOPInstanceUID`.
        ref_sop: Option<&'a str>,
        /// `StructureSetROISequence[0]`: (name, referenced FoR); `None`
        /// leaves the whole sequence out.
        roi_item: Option<(&'a str, Option<&'a str>)>,
        top_level_for: Option<&'a str>,
        observation_label: Option<&'a str>,
    }

    fn object(s: &Spec) -> InMemDicomObject {
        let mut o = InMemDicomObject::new_empty();
        let mut rfr = InMemDicomObject::new_empty();
        if let Some(f) = s.ref_for {
            put_str(&mut rfr, tags::FRAME_OF_REFERENCE_UID, VR::UI, f);
        }
        if let Some(f) = s.ref_sop {
            put_str(&mut rfr, tags::REFERENCED_SOP_INSTANCE_UID, VR::UI, f);
        }
        put_seq(
            &mut o,
            tags::REFERENCED_FRAME_OF_REFERENCE_SEQUENCE,
            vec![rfr],
        );
        if let Some((name, f)) = s.roi_item {
            let mut it = InMemDicomObject::new_empty();
            put_is(&mut it, tags::ROI_NUMBER, 1);
            put_str(&mut it, tags::ROI_NAME, VR::LO, name);
            if let Some(f) = f {
                put_str(&mut it, tags::REFERENCED_FRAME_OF_REFERENCE_UID, VR::UI, f);
            }
            put_seq(&mut o, tags::STRUCTURE_SET_ROI_SEQUENCE, vec![it]);
        }
        if let Some(f) = s.top_level_for {
            put_str(&mut o, tags::FRAME_OF_REFERENCE_UID, VR::UI, f);
        }
        let mut ob = InMemDicomObject::new_empty();
        put_is(&mut ob, tags::REFERENCED_ROI_NUMBER, 1);
        put_str(&mut ob, tags::RTROI_INTERPRETED_TYPE, VR::CS, "ORGAN");
        if let Some(l) = s.observation_label {
            put_str(&mut ob, ROI_OBSERVATION_LABEL, VR::SH, l);
        }
        put_seq(&mut o, tags::RTROI_OBSERVATIONS_SEQUENCE, vec![ob]);
        let mut co = InMemDicomObject::new_empty();
        put_str(
            &mut co,
            tags::CONTOUR_GEOMETRIC_TYPE,
            VR::CS,
            "CLOSED_PLANAR",
        );
        put_ds(
            &mut co,
            tags::CONTOUR_DATA,
            &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        );
        let mut rc = InMemDicomObject::new_empty();
        put_is(&mut rc, tags::REFERENCED_ROI_NUMBER, 1);
        put_seq(&mut rc, tags::CONTOUR_SEQUENCE, vec![co]);
        put_seq(&mut o, tags::ROI_CONTOUR_SEQUENCE, vec![rc]);
        o
    }

    const NONE: Spec = Spec {
        ref_for: None,
        ref_sop: None,
        roi_item: None,
        top_level_for: None,
        observation_label: None,
    };

    #[test]
    fn a_conforming_set_needs_no_notes() {
        let l = read(
            &object(&Spec {
                ref_for: Some("1.1"),
                roi_item: Some(("Liver", Some("1.1"))),
                top_level_for: Some("9.9"),
                ..NONE
            }),
            "RS.dcm".into(),
        );
        assert_eq!(
            l.set.frame_of_reference_uid, "1.1",
            "the standard's place wins"
        );
        assert_eq!(l.set.rois[0].name, "Liver");
        assert!(l.notes.is_empty(), "{:?}", l.notes);
    }

    #[test]
    fn the_frame_of_reference_falls_back_rung_by_rung() {
        // 2: the ROI sequence's reference.
        let l = read(
            &object(&Spec {
                roi_item: Some(("Liver", Some("2.2"))),
                top_level_for: Some("3.3"),
                ref_sop: Some("4.4"),
                ..NONE
            }),
            "RS.dcm".into(),
        );
        assert_eq!(l.set.frame_of_reference_uid, "2.2");
        assert_eq!(l.notes.len(), 1);
        // 3: the top-level tag.
        let l = read(
            &object(&Spec {
                roi_item: Some(("Liver", None)),
                top_level_for: Some("3.3"),
                ref_sop: Some("4.4"),
                ..NONE
            }),
            "RS.dcm".into(),
        );
        assert_eq!(l.set.frame_of_reference_uid, "3.3");
        assert!(l.notes[0].contains("top-level"), "{:?}", l.notes);
        // 4: the misplaced UID, once it is that of the referenced series.
        let spec = Spec {
            roi_item: Some(("Liver", None)),
            ref_sop: Some("4.4"),
            ..NONE
        };
        let mut l = read(&object(&spec), "RS.dcm".into());
        assert_eq!(l.set.frame_of_reference_uid, "", "not on its own");
        l.settle(Some("4.4"));
        assert_eq!(l.set.frame_of_reference_uid, "4.4");
        assert_eq!(l.notes.len(), 1, "{:?}", l.notes);
    }

    #[test]
    fn a_misplaced_uid_that_matches_nothing_is_not_used() {
        let spec = Spec {
            roi_item: Some(("Liver", None)),
            ref_sop: Some("4.4"),
            ..NONE
        };
        for series_for in [Some("5.5"), Some(""), None] {
            let mut l = read(&object(&spec), "RS.dcm".into());
            l.settle(series_for);
            assert_eq!(l.set.frame_of_reference_uid, "", "{series_for:?}");
            assert!(l.notes[0].contains("not used"), "{:?}", l.notes);
        }
        // A set that has its frame of reference keeps it, whatever the
        // misplaced UID says.
        let mut l = read(
            &object(&Spec {
                top_level_for: Some("3.3"),
                ref_sop: Some("4.4"),
                roi_item: Some(("Liver", None)),
                ..NONE
            }),
            "RS.dcm".into(),
        );
        l.settle(Some("4.4"));
        assert_eq!(l.set.frame_of_reference_uid, "3.3");
    }

    #[test]
    fn names_come_from_the_observations_when_the_roi_sequence_is_missing() {
        let l = read(
            &object(&Spec {
                ref_for: Some("1.1"),
                observation_label: Some("Spinal cord"),
                ..NONE
            }),
            "RS.dcm".into(),
        );
        assert_eq!(l.set.rois[0].name, "Spinal cord");
        assert_eq!(l.notes.len(), 1);
        assert!(l.notes[0].contains("no Structure Set ROI Sequence"));
        assert!(l.notes[0].contains("RT ROI Observations"));
        // Without a label either, the number is all there is.
        let l = read(
            &object(&Spec {
                ref_for: Some("1.1"),
                ..NONE
            }),
            "RS.dcm".into(),
        );
        assert_eq!(l.set.rois[0].name, "ROI 1");
        assert!(l.notes[0].contains("named by number"));
    }
}
