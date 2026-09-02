//! What one export run is: which patients, studies and series go out, under
//! which names and UIDs, in which format, and how the references between the
//! objects survive the trip.
//!
//! The writer lives in [`crate::dicom_export`]; this module decides what it
//! is asked to write. The two are separate because the hard part of an export
//! is not producing bytes, it is keeping a study *a study*: a structure set
//! that no longer names the series it was drawn on is not a smaller export,
//! it is a broken one.
//!
//! ## The plan
//!
//! [`ExportPlan`] is the tree the dialog shows and the runner walks:
//!
//! ```text
//! dataset A
//!   patient  STAR_Rambam_2            PatientName / PatientID
//!     study  CCT  20250728            StudyInstanceUID / description / ID / date
//!       4D group  4DCT (10 phases)    ticking it takes every phase
//!         series  CT 0%               SeriesInstanceUID / FrameOfReferenceUID / …
//!         series  CT 10%
//!       series  CT16                  318 files
//!       RTSTRUCT  CCT RTSTRUCT        RTSTRUCT | SEG
//!       RTDOSE, RTPLAN
//! ```
//!
//! Every identifier in it is editable and every one of them is shown, because
//! an export whose UIDs you cannot see is an export you cannot file.
//!
//! ## Keeping the links
//!
//! The run is two passes. The image pass writes (or copies) the slices and
//! records, per source series, the UID each slice ended up with and where it
//! sits along the slice axis. The object pass then writes the RT objects
//! through the same [`UidMap`], so:
//!
//! * an RTSTRUCT names its study, its image series and every slice, and each
//!   contour names the image it lies on;
//! * a SEG names the image series and the slices its frames belong to;
//! * a plan names its structure set and a dose names its plan;
//! * everything shares one Frame of Reference UID per frame of reference.
//!
//! When a structure set is exported without its images, that is not silently
//! degraded: the object still goes out, and the run reports it.
//!
//! ## Images are copied, not re-encoded
//!
//! A series that still has its source files is copied file by file with only
//! the identifying attributes patched. Nothing else is touched - not the
//! private tags, not the padding value, not the transfer syntax, not one bit
//! of pixel data - which is what keeps 4D acquisitions, dual-energy series
//! and vendor extensions intact through a round trip. Only a series the
//! application invented (a simulation, a resampled volume) is rendered from
//! its voxels.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use dicom_core::Tag;
use dicom_dictionary_std::tags;

use crate::dicom_export::{self, ExportParams, ImageRef};
use crate::dicomseg::{self, SegSeries};
use crate::fourd::Role;
use crate::loader::{LoadedStudy, SeriesInfo};
use crate::progress::Progress;
use crate::rtstruct::StructureSet;
use crate::segmentation::{self, Segmentation};
use crate::volume::Volume;

// ---------------------------------------------------------------------------
// Run-wide options
// ---------------------------------------------------------------------------

/// Where the identifiers of an export come from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum UidMode {
    /// Write the UIDs the data already has. The export *is* the same study;
    /// re-importing it into the archive it came from updates it rather than
    /// duplicating it, and references from objects outside the export still
    /// resolve.
    Keep,
    /// Mint a new UID for every study, series, frame of reference and
    /// instance. The export is a new study that happens to look like the old
    /// one - what you want when the edited copy has to live beside its
    /// source.
    New,
}

/// How the output folder is arranged.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Layout {
    /// `PatientID / Study / Series / files`.
    Tree,
    /// One folder per study, its series and objects flat inside it.
    StudyFolders,
    /// Everything in the output folder.
    Flat,
}

/// The format a set of structures is written in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StructFormat {
    /// RT Structure Set: contours, the native form of an RTSTRUCT.
    RtStruct,
    /// Segmentation Storage: one binary mask per structure.
    Seg,
}

impl StructFormat {
    pub fn label(self) -> &'static str {
        match self {
            StructFormat::RtStruct => "RTSTRUCT",
            StructFormat::Seg => "SEG",
        }
    }
}

// ---------------------------------------------------------------------------
// Editable fields
// ---------------------------------------------------------------------------

/// One editable cell of the plan: what will be written, what the data said,
/// and - for identifiers - the freshly minted alternative, kept so that
/// flipping [`UidMode`] back and forth is stable rather than minting a new
/// UID every time the radio is clicked.
#[derive(Clone)]
pub struct Field {
    pub value: String,
    pub original: String,
    pub fresh: Option<String>,
}

impl Field {
    pub fn text(v: impl Into<String>) -> Self {
        let v = v.into();
        Field {
            original: v.clone(),
            value: v,
            fresh: None,
        }
    }

    /// A UID field: an original (possibly empty) and a replacement.
    pub fn uid(v: impl Into<String>) -> Self {
        let v = v.into();
        let fresh = dicom_export::new_uid();
        Field {
            // A missing UID has no "keep" to offer, so the fresh one stands
            // in from the start.
            value: if v.is_empty() {
                fresh.clone()
            } else {
                v.clone()
            },
            original: v,
            fresh: Some(fresh),
        }
    }

    /// Re-fill from the mode, unless the user typed something of their own.
    pub fn apply_mode(&mut self, mode: UidMode) {
        let Some(fresh) = &self.fresh else { return };
        if self.value != self.original && self.value != *fresh {
            return; // hand-written, left alone
        }
        self.value = match mode {
            UidMode::Keep if !self.original.is_empty() => self.original.clone(),
            _ => fresh.clone(),
        };
    }

    pub fn is_new(&self) -> bool {
        self.fresh.as_deref() == Some(self.value.as_str()) && self.value != self.original
    }

    pub fn trimmed(&self) -> &str {
        self.value.trim()
    }
}

// ---------------------------------------------------------------------------
// The tree
// ---------------------------------------------------------------------------

/// What an object node is.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ObjKind {
    /// An RT Structure Set as loaded (contours).
    Structures,
    /// A segmentation series as loaded or drawn (masks).
    Segmentation,
    Dose,
    Plan,
}

impl ObjKind {
    pub fn glyph(self) -> &'static str {
        match self {
            // The same square the data tree marks a structure set with, a
            // shaded one for a mask, a grid for a dose grid and the radiation
            // sign for a plan. All four are in `app::glyphs::ALLOWED`.
            ObjKind::Structures => "▣",
            ObjKind::Segmentation => "◍",
            ObjKind::Dose => "⊞",
            ObjKind::Plan => "☢",
        }
    }
    /// Whether the RTSTRUCT / SEG radio applies.
    pub fn is_structures(self) -> bool {
        matches!(self, ObjKind::Structures | ObjKind::Segmentation)
    }

    /// The format this kind of object was read in.
    pub fn native_format(self) -> StructFormat {
        match self {
            ObjKind::Segmentation => StructFormat::Seg,
            _ => StructFormat::RtStruct,
        }
    }
}

/// One image series.
#[derive(Clone)]
pub struct SeriesNode {
    pub slot: usize,
    /// Index into `LoadedStudy::series`.
    pub index: usize,
    pub selected: bool,
    pub modality: String,
    pub n_files: usize,
    /// Series Instance UID as loaded - the key every reference uses.
    pub source_uid: String,
    pub uid: Field,
    pub for_uid: Field,
    pub description: Field,
    pub number: Field,
    /// Index of the 4D group in [`StudyNode::groups`] and the member label.
    pub fourd: Option<(usize, String)>,
}

/// One RT object.
#[derive(Clone)]
pub struct ObjNode {
    pub slot: usize,
    pub kind: ObjKind,
    /// Index into the study's `structure_sets` / `seg_series` / `doses` /
    /// `plans`.
    pub index: usize,
    pub selected: bool,
    /// What the tree calls it.
    pub label: String,
    /// Extra line under the label (ROI count, grid size, beams).
    pub detail: String,
    /// SOP Instance UID as loaded.
    pub source_uid: String,
    pub sop_uid: Field,
    pub series_uid: Field,
    pub description: Field,
    pub number: Field,
    /// The format this one is written in (structures only).
    pub format: StructFormat,
    /// Image series it belongs to, as loaded.
    pub referenced_series_uid: String,
}

/// A 4D acquisition inside a study. Its members are listed under it so that
/// one tick takes all the phases, which is the only way an exported 4D study
/// is still a 4D study.
#[derive(Clone)]
pub struct GroupNode {
    pub name: String,
    /// Indices into [`StudyNode::series`], in member order.
    pub members: Vec<usize>,
    /// How many of them are phases (the rest are AVG / MIP reconstructions).
    pub phases: usize,
}

#[derive(Clone)]
pub struct StudyNode {
    pub slot: usize,
    pub source_uid: String,
    pub uid: Field,
    pub description: Field,
    pub id: Field,
    pub date: Field,
    pub series: Vec<SeriesNode>,
    pub objects: Vec<ObjNode>,
    pub groups: Vec<GroupNode>,
}

#[derive(Clone)]
pub struct PatientNode {
    pub slot: usize,
    pub key: String,
    pub name: Field,
    pub id: Field,
    pub studies: Vec<StudyNode>,
}

#[derive(Clone)]
pub struct DatasetNode {
    pub slot: usize,
    /// "A" / "B".
    pub label: &'static str,
    pub patients: Vec<PatientNode>,
}

/// Everything one export run is told to do.
#[derive(Clone)]
pub struct ExportPlan {
    pub datasets: Vec<DatasetNode>,
    pub uid_mode: UidMode,
    pub layout: Layout,
    /// Patient / equipment attributes written into every object. The tree's
    /// own fields win over these for the tags they cover.
    pub params: ExportParams,
    /// Write the images from the loaded voxels instead of copying the source
    /// files. Off by default: copying is lossless and much faster.
    pub rerender_images: bool,
}

// ---------------------------------------------------------------------------
// Building the plan from what is loaded
// ---------------------------------------------------------------------------

/// The tags the tree owns. The common-tag table hides them, and
/// [`ExportParams`] values for them are overridden per node.
pub const PER_NODE_TAGS: [Tag; 7] = [
    tags::PATIENT_NAME,
    tags::PATIENT_ID,
    tags::STUDY_ID,
    tags::STUDY_DESCRIPTION,
    tags::STUDY_DATE,
    tags::SERIES_DESCRIPTION,
    tags::SERIES_NUMBER,
];

impl ExportPlan {
    /// Build the plan from the loaded datasets. Everything starts selected -
    /// the common case is "write out what I have", and unticking is easier
    /// than hunting.
    pub fn build(studies: [Option<&LoadedStudy>; 2], params: ExportParams) -> Self {
        let mut datasets = Vec::new();
        for (slot, study) in studies.into_iter().enumerate() {
            let Some(study) = study else { continue };
            let patients = build_patients(slot, study);
            if patients.is_empty() {
                continue;
            }
            datasets.push(DatasetNode {
                slot,
                label: if slot == 0 { "A" } else { "B" },
                patients,
            });
        }
        ExportPlan {
            datasets,
            uid_mode: UidMode::Keep,
            layout: Layout::Tree,
            params,
            rerender_images: false,
        }
    }

    /// Re-fill every identifier from the mode.
    pub fn set_uid_mode(&mut self, mode: UidMode) {
        self.uid_mode = mode;
        for d in &mut self.datasets {
            for p in &mut d.patients {
                for st in &mut p.studies {
                    st.uid.apply_mode(mode);
                    for se in &mut st.series {
                        se.uid.apply_mode(mode);
                        se.for_uid.apply_mode(mode);
                    }
                    for ob in &mut st.objects {
                        ob.sop_uid.apply_mode(mode);
                        ob.series_uid.apply_mode(mode);
                    }
                }
            }
        }
        self.sync_format_uids();
    }

    /// Set every structure node's format at once.
    pub fn set_all_formats(&mut self, format: StructFormat) {
        for st in self.studies_mut() {
            for ob in st.objects.iter_mut().filter(|o| o.kind.is_structures()) {
                ob.format = format;
            }
        }
        self.sync_format_uids();
    }

    /// Give every converted object a new SOP Instance UID.
    ///
    /// Contours written as a segmentation are a different SOP class, so what
    /// is written is a new instance and not the one that was read - whatever
    /// [`UidMode`] says. Two objects of different SOP classes sharing one
    /// instance UID is the one thing an archive cannot forgive, so this is
    /// not left to the user to notice. Converting back restores the original,
    /// and a UID typed by hand is never touched.
    pub fn sync_format_uids(&mut self) {
        let mode = self.uid_mode;
        for st in self.studies_mut() {
            for ob in st.objects.iter_mut().filter(|o| o.kind.is_structures()) {
                let Some(fresh) = ob.sop_uid.fresh.clone() else {
                    continue;
                };
                if ob.sop_uid.value != ob.sop_uid.original && ob.sop_uid.value != fresh {
                    continue;
                }
                let converted = ob.format != ob.kind.native_format();
                ob.sop_uid.value =
                    if converted || mode == UidMode::New || ob.sop_uid.original.is_empty() {
                        fresh
                    } else {
                        ob.sop_uid.original.clone()
                    };
            }
        }
    }

    pub fn studies_mut(&mut self) -> impl Iterator<Item = &mut StudyNode> {
        self.datasets
            .iter_mut()
            .flat_map(|d| d.patients.iter_mut())
            .flat_map(|p| p.studies.iter_mut())
    }

    pub fn studies(&self) -> impl Iterator<Item = &StudyNode> {
        self.datasets
            .iter()
            .flat_map(|d| d.patients.iter())
            .flat_map(|p| p.studies.iter())
    }

    /// (image series, RT objects) currently ticked.
    pub fn counts(&self) -> (usize, usize) {
        let mut series = 0;
        let mut objects = 0;
        for st in self.studies() {
            series += st.series.iter().filter(|s| s.selected).count();
            objects += st.objects.iter().filter(|o| o.selected).count();
        }
        (series, objects)
    }

    pub fn is_empty(&self) -> bool {
        self.counts() == (0, 0)
    }
}

impl StudyNode {
    /// Tick or untick everything in this study.
    pub fn set_all(&mut self, on: bool) {
        for s in &mut self.series {
            s.selected = on;
        }
        for o in &mut self.objects {
            o.selected = on;
        }
    }
    /// None when the study is partly selected.
    pub fn all_selected(&self) -> Option<bool> {
        let mut any = false;
        let mut all = true;
        for on in self
            .series
            .iter()
            .map(|s| s.selected)
            .chain(self.objects.iter().map(|o| o.selected))
        {
            any |= on;
            all &= on;
        }
        match (any, all) {
            (false, _) => Some(false),
            (true, true) => Some(true),
            _ => None,
        }
    }
}

impl PatientNode {
    pub fn set_all(&mut self, on: bool) {
        for s in &mut self.studies {
            s.set_all(on);
        }
    }
    pub fn all_selected(&self) -> Option<bool> {
        merge_tri(self.studies.iter().map(|s| s.all_selected()))
    }
}

impl DatasetNode {
    pub fn set_all(&mut self, on: bool) {
        for p in &mut self.patients {
            p.set_all(on);
        }
    }
    pub fn all_selected(&self) -> Option<bool> {
        merge_tri(self.patients.iter().map(|p| p.all_selected()))
    }
}

fn merge_tri(it: impl Iterator<Item = Option<bool>>) -> Option<bool> {
    let mut any = false;
    let mut all = true;
    let mut empty = true;
    for t in it {
        empty = false;
        match t {
            Some(true) => any = true,
            Some(false) => all = false,
            None => {
                any = true;
                all = false;
            }
        }
    }
    if empty {
        return Some(false);
    }
    match (any, all) {
        (false, _) => Some(false),
        (true, true) => Some(true),
        _ => None,
    }
}

/// The study a loose object belongs to: what it says itself, else the study
/// of the image series it references.
fn study_of(study: &LoadedStudy, own: &str, referenced: &str) -> String {
    if !own.is_empty() {
        return own.to_string();
    }
    study
        .series
        .iter()
        .find(|s| s.uid == referenced)
        .map(|s| s.study_uid.clone())
        .unwrap_or_default()
}

fn build_patients(slot: usize, study: &LoadedStudy) -> Vec<PatientNode> {
    // patient key -> study uids, in first-seen order.
    let mut order: Vec<(String, Vec<String>)> = Vec::new();
    let mut patient_of_study: HashMap<String, String> = HashMap::new();
    for s in &study.series {
        let pk = s.patient_key().to_string();
        patient_of_study
            .entry(s.study_uid.clone())
            .or_insert_with(|| pk.clone());
        let slot_i = match order.iter().position(|(k, _)| *k == pk) {
            Some(i) => i,
            None => {
                order.push((pk.clone(), Vec::new()));
                order.len() - 1
            }
        };
        if !order[slot_i].1.contains(&s.study_uid) {
            order[slot_i].1.push(s.study_uid.clone());
        }
    }

    // Objects whose study has no image series in this dataset still have to
    // land somewhere: they get their own study under the dataset's patient.
    let mut loose: Vec<String> = Vec::new();
    let note = |uid: &str, loose: &mut Vec<String>| {
        if !uid.is_empty()
            && !patient_of_study.contains_key(uid)
            && !loose.contains(&uid.to_string())
        {
            loose.push(uid.to_string());
        }
    };
    for ss in &study.structure_sets {
        note(
            &study_of(study, &ss.study_uid, &ss.referenced_series_uid),
            &mut loose,
        );
    }
    for sg in &study.seg_series {
        note(
            &study_of(study, &sg.study_uid, &sg.referenced_series_uid),
            &mut loose,
        );
    }
    for p in &study.plans {
        note(&p.study_uid, &mut loose);
    }
    if !loose.is_empty() || order.is_empty() {
        let pk = if study.meta.patient_id.is_empty() {
            study.meta.patient_name.clone()
        } else {
            study.meta.patient_id.clone()
        };
        let pk = if pk.is_empty() { "?".to_string() } else { pk };
        let i = match order.iter().position(|(k, _)| *k == pk) {
            Some(i) => i,
            None => {
                order.push((pk.clone(), Vec::new()));
                order.len() - 1
            }
        };
        for uid in loose {
            patient_of_study.insert(uid.clone(), pk.clone());
            order[i].1.push(uid);
        }
        if order[i].1.is_empty() {
            order[i].1.push(String::new());
        }
    }

    order
        .into_iter()
        .map(|(key, study_uids)| {
            let first = study.series.iter().find(|s| s.patient_key() == key);
            let (pname, pid) = match first {
                Some(s) => (s.patient_name.clone(), s.patient_id.clone()),
                None => (
                    study.meta.patient_name.clone(),
                    study.meta.patient_id.clone(),
                ),
            };
            PatientNode {
                slot,
                key: key.clone(),
                name: Field::text(pname),
                id: Field::text(pid),
                studies: study_uids
                    .into_iter()
                    .map(|uid| build_study(slot, study, &uid))
                    .collect(),
            }
        })
        .collect()
}

fn build_study(slot: usize, study: &LoadedStudy, study_uid: &str) -> StudyNode {
    let members: Vec<(usize, &SeriesInfo)> = study
        .series
        .iter()
        .enumerate()
        .filter(|(_, s)| s.study_uid == study_uid)
        .collect();
    let first = members.first().map(|(_, s)| *s);

    // 4D groups of this study, and the member each series belongs to.
    let mut groups: Vec<GroupNode> = Vec::new();
    let mut group_of: HashMap<String, (usize, String)> = HashMap::new();
    for g in study
        .fourd_groups
        .iter()
        .filter(|g| g.study_uid == study_uid && !g.dissolved)
    {
        let gi = groups.len();
        let mut phases = 0usize;
        for m in &g.members {
            if m.role == Role::Phase {
                phases += 1;
            }
            let tag = m.role.tag();
            let label = if tag.is_empty() {
                m.label.clone()
            } else {
                format!("{} {tag}", m.label)
            };
            group_of.insert(m.series_uid.clone(), (gi, label));
        }
        groups.push(GroupNode {
            name: g.name.clone(),
            members: Vec::new(),
            phases,
        });
    }

    let mut series = Vec::new();
    for (index, s) in &members {
        let fourd = group_of.get(&s.uid).cloned();
        if let Some((gi, _)) = &fourd {
            groups[*gi].members.push(series.len());
        }
        series.push(SeriesNode {
            slot,
            index: *index,
            selected: true,
            modality: s.modality.clone(),
            n_files: s.files.len(),
            source_uid: s.uid.clone(),
            uid: Field::uid(s.uid.clone()),
            for_uid: Field::uid(frame_of_reference_of(study, s)),
            description: Field::text(s.description.clone()),
            number: Field::text(
                s.series_number
                    .map(|n| n.to_string())
                    .unwrap_or_else(|| (series.len() + 1).to_string()),
            ),
            fourd,
        });
    }
    // Members of a group come out in member order, not file order.
    for g in &mut groups {
        g.members
            .sort_by_key(|i| series[*i].fourd.as_ref().map(|f| f.1.clone()));
    }

    let mut objects = Vec::new();
    for (i, ss) in study.structure_sets.iter().enumerate() {
        if study_of(study, &ss.study_uid, &ss.referenced_series_uid) != study_uid {
            continue;
        }
        objects.push(ObjNode {
            slot,
            kind: ObjKind::Structures,
            index: i,
            selected: true,
            label: ss.label.clone(),
            detail: format!("{} ROI(s)", ss.rois.len()),
            source_uid: ss.sop_instance_uid.clone(),
            sop_uid: Field::uid(ss.sop_instance_uid.clone()),
            series_uid: Field::uid(ss.series_instance_uid.clone()),
            description: Field::text(ss.label.clone()),
            number: Field::text((300 + i).to_string()),
            format: StructFormat::RtStruct,
            referenced_series_uid: ss.referenced_series_uid.clone(),
        });
    }
    for (i, sg) in study.seg_series.iter().enumerate() {
        if study_of(study, &sg.study_uid, &sg.referenced_series_uid) != study_uid {
            continue;
        }
        objects.push(ObjNode {
            slot,
            kind: ObjKind::Segmentation,
            index: i,
            selected: true,
            label: sg.label.clone(),
            detail: format!("{} segment(s)", sg.segs.len()),
            source_uid: sg.sop_instance_uid.clone(),
            sop_uid: Field::uid(sg.sop_instance_uid.clone()),
            series_uid: Field::uid(sg.series_instance_uid.clone()),
            description: Field::text(sg.label.clone()),
            number: Field::text((400 + i).to_string()),
            format: StructFormat::Seg,
            referenced_series_uid: sg.referenced_series_uid.clone(),
        });
    }
    for (i, d) in study.doses.iter().enumerate() {
        // A dose belongs to the study of the plan it was computed for; with
        // no plan it follows the only study there is.
        if members.is_empty() && study.doses.len() > 1 {
            continue;
        }
        objects.push(ObjNode {
            slot,
            kind: ObjKind::Dose,
            index: i,
            selected: true,
            label: if d.summation_type.is_empty() {
                "Dose".into()
            } else {
                d.summation_type.clone()
            },
            detail: format!("{:?}, max {:.2} {}", d.dims, d.max_dose, d.units),
            source_uid: d.sop_instance_uid.clone(),
            sop_uid: Field::uid(d.sop_instance_uid.clone()),
            series_uid: Field::uid(d.series_instance_uid.clone()),
            description: Field::text("Dose".to_string()),
            number: Field::text((500 + i).to_string()),
            format: StructFormat::RtStruct,
            referenced_series_uid: String::new(),
        });
    }
    for (i, p) in study.plans.iter().enumerate() {
        if !p.study_uid.is_empty() && p.study_uid != study_uid {
            continue;
        }
        objects.push(ObjNode {
            slot,
            kind: ObjKind::Plan,
            index: i,
            selected: true,
            label: if p.label.is_empty() {
                "Plan".into()
            } else {
                p.label.clone()
            },
            detail: format!("{} beam(s)", p.beams.len()),
            source_uid: p.sop_instance_uid.clone(),
            sop_uid: Field::uid(p.sop_instance_uid.clone()),
            series_uid: Field::uid(p.series_instance_uid.clone()),
            description: Field::text(p.name.clone()),
            number: Field::text((600 + i).to_string()),
            format: StructFormat::RtStruct,
            referenced_series_uid: String::new(),
        });
    }

    StudyNode {
        slot,
        source_uid: study_uid.to_string(),
        uid: Field::uid(study_uid.to_string()),
        description: Field::text(
            first
                .map(|s| s.study_description.clone())
                .unwrap_or_else(|| study.meta.study_description.clone()),
        ),
        id: Field::text("1"),
        date: Field::text(
            first
                .map(|s| s.study_date.clone())
                .unwrap_or_else(|| study.meta.study_date.clone()),
        ),
        series,
        objects,
        groups,
    }
}

/// Frame of Reference of a series. The scan itself is the authority; the
/// loaded volume only speaks for the series it was reconstructed from.
fn frame_of_reference_of(study: &LoadedStudy, s: &SeriesInfo) -> String {
    if study
        .series
        .get(study.active_series)
        .is_some_and(|a| a.uid == s.uid)
    {
        return study.volume.frame_of_reference_uid.clone();
    }
    // Otherwise take it from the structure set or segmentation that names
    // this series, and fall back to the volume's.
    study
        .structure_sets
        .iter()
        .find(|ss| ss.referenced_series_uid == s.uid && !ss.frame_of_reference_uid.is_empty())
        .map(|ss| ss.frame_of_reference_uid.clone())
        .or_else(|| {
            study
                .seg_series
                .iter()
                .find(|sg| {
                    sg.referenced_series_uid == s.uid && !sg.grid.frame_of_reference_uid.is_empty()
                })
                .map(|sg| sg.grid.frame_of_reference_uid.clone())
        })
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Running the plan
// ---------------------------------------------------------------------------

/// What one exported image series turned into: the identifiers it was written
/// with, and where each of its slices ended up. This is the memory that makes
/// the RT objects resolvable - see the module docs.
#[derive(Clone, Default)]
struct Written {
    series_uid: String,
    study_uid: String,
    for_uid: String,
    sop_class: String,
    /// (SOP Instance UID, position along `normal`) per slice.
    slices: Vec<(String, f64)>,
    normal: crate::geometry::Vec3,
    spacing: f64,
}

impl Written {
    fn image_ref(&self) -> ImageRef {
        ImageRef {
            series_uid: self.series_uid.clone(),
            study_uid: self.study_uid.clone(),
            sop_class: self.sop_class.clone(),
            slices: self.slices.clone(),
            normal: self.normal,
            spacing: self.spacing,
        }
    }
}

/// The outcome of a run.
pub struct ExportSummary {
    pub files: usize,
    pub root: PathBuf,
    /// Everything the run could not do exactly as asked. An export that
    /// quietly drops a reference is worse than one that says it did.
    pub warnings: Vec<String>,
}

impl ExportSummary {
    pub fn message(&self) -> String {
        let n = self.warnings.len();
        let head = format!(
            "✔ {} DICOM file(s) written to {}",
            self.files,
            self.root.display()
        );
        match n {
            0 => head,
            1 => format!("{head}  (1 note)"),
            _ => format!("{head}  ({n} notes)"),
        }
    }
}

/// File- and folder-name-safe form of a label.
fn safe(s: &str, fallback: &str) -> String {
    let out: String = s
        .trim()
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || matches!(c, '-' | '_' | '.' | '+') {
                c
            } else {
                '_'
            }
        })
        .collect();
    let out = out.trim_matches('_').to_string();
    if out.is_empty() {
        fallback.to_string()
    } else {
        out.chars().take(64).collect()
    }
}

/// Run the plan. Nothing is written until the whole tree has been walked for
/// the folders it needs, so a failure part way through leaves a partial
/// export rather than a confusing one.
pub fn run(
    plan: &ExportPlan,
    studies: [Option<&LoadedStudy>; 2],
    root: &Path,
    progress: &Progress,
) -> Result<ExportSummary> {
    if plan.is_empty() {
        bail!("Nothing is selected for export");
    }
    std::fs::create_dir_all(root)
        .with_context(|| format!("create directory {}", root.display()))?;

    // The run-wide tags, minus the ones the tree owns.
    let params = plan.params.without(&PER_NODE_TAGS);
    let (today_date, today_time) = dicom_export::today();
    let mut files = 0usize;
    let mut warnings: Vec<String> = Vec::new();

    for dataset in &plan.datasets {
        let Some(study) = studies[dataset.slot] else {
            continue;
        };
        for patient in &dataset.patients {
            let pdir = match plan.layout {
                Layout::Tree => root.join(safe(
                    if patient.id.trimmed().is_empty() {
                        patient.name.trimmed()
                    } else {
                        patient.id.trimmed()
                    },
                    "patient",
                )),
                _ => root.to_path_buf(),
            };
            for st in &patient.studies {
                if st.all_selected() == Some(false) {
                    continue;
                }
                let sdir = match plan.layout {
                    Layout::Flat => root.to_path_buf(),
                    _ => pdir.join(safe(
                        &format!("{}_{}", st.date.trimmed(), st.description.trimmed()),
                        "study",
                    )),
                };
                export_one_study(
                    plan,
                    study,
                    patient,
                    st,
                    &params,
                    (&today_date, &today_time),
                    &sdir,
                    progress,
                    &mut files,
                    &mut warnings,
                )?;
            }
        }
    }

    progress.set("done");
    Ok(ExportSummary {
        files,
        root: root.to_path_buf(),
        warnings,
    })
}

/// The identity patches every object of one study carries.
fn identity_patches(patient: &PatientNode, st: &StudyNode) -> Vec<(Tag, dicom_core::VR, String)> {
    use dicom_core::VR;
    vec![
        (
            tags::PATIENT_NAME,
            VR::PN,
            patient.name.trimmed().to_string(),
        ),
        (tags::PATIENT_ID, VR::LO, patient.id.trimmed().to_string()),
        (
            tags::STUDY_INSTANCE_UID,
            VR::UI,
            st.uid.trimmed().to_string(),
        ),
        (tags::STUDY_ID, VR::SH, st.id.trimmed().to_string()),
        (
            tags::STUDY_DESCRIPTION,
            VR::LO,
            st.description.trimmed().to_string(),
        ),
        (tags::STUDY_DATE, VR::DA, st.date.trimmed().to_string()),
    ]
}

#[allow(clippy::too_many_arguments)]
fn export_one_study(
    plan: &ExportPlan,
    study: &LoadedStudy,
    patient: &PatientNode,
    st: &StudyNode,
    params: &ExportParams,
    now: (&str, &str),
    dir: &Path,
    progress: &Progress,
    files: &mut usize,
    warnings: &mut Vec<String>,
) -> Result<()> {
    // Source series UID -> what it became.
    let mut written: HashMap<String, Written> = HashMap::new();

    // ---- image series -----------------------------------------------------
    let selected: Vec<&SeriesNode> = st.series.iter().filter(|s| s.selected).collect();
    for (n, node) in selected.iter().enumerate() {
        let Some(src) = study.series.get(node.index) else {
            continue;
        };
        progress.set(format!(
            "Series {}/{}: {} {}",
            n + 1,
            selected.len(),
            src.modality,
            node.description.trimmed()
        ));
        let sdir = match plan.layout {
            Layout::Tree => dir.join(safe(
                &format!(
                    "{}_{}_{}",
                    node.number.trimmed(),
                    src.modality,
                    node.description.trimmed()
                ),
                "series",
            )),
            _ => dir.to_path_buf(),
        };
        std::fs::create_dir_all(&sdir)
            .with_context(|| format!("create directory {}", sdir.display()))?;

        let w = if plan.rerender_images || src.files.is_empty() {
            render_series(study, patient, st, node, src, params, now, &sdir, files)?
        } else {
            copy_series(plan, patient, st, node, src, params, &sdir, progress, files)?
        };
        match w {
            Some(w) => {
                written.insert(node.source_uid.clone(), w);
            }
            None => warnings.push(format!(
                "series “{}” has neither source files nor loaded voxels and was skipped",
                node.description.trimmed()
            )),
        }
    }

    // ---- warn about a 4D group that is going out in pieces ----------------
    for g in &st.groups {
        let picked = g.members.iter().filter(|i| st.series[**i].selected).count();
        if picked > 0 && picked < g.members.len() {
            warnings.push(format!(
                "4D group “{}”: {} of {} series selected - the export is not a complete \
                 4D acquisition and will not regroup as one",
                g.name,
                picked,
                g.members.len()
            ));
        }
    }

    // ---- RT objects -------------------------------------------------------
    // The study folder exists once a series has been written into it; a study
    // whose objects go out on their own still needs one.
    if st.objects.iter().any(|o| o.selected) {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("create directory {}", dir.display()))?;
    }
    // Every object's written SOP Instance UID, by the UID it had when loaded,
    // so plan ▶ structure set and dose ▶ plan still resolve.
    let sop_map: HashMap<&str, &str> = st
        .objects
        .iter()
        .filter(|o| !o.source_uid.is_empty())
        .map(|o| (o.source_uid.as_str(), o.sop_uid.trimmed()))
        .collect();

    for (n, node) in st.objects.iter().filter(|o| o.selected).enumerate() {
        progress.set(format!("Writing {} {}", node.kind.glyph(), node.label));
        let for_uid = object_frame_of_reference(study, node, st, &written);
        let ctx = dicom_export::Ctx {
            study_uid: st.uid.trimmed().to_string(),
            for_uid,
            date: if st.date.trimmed().is_empty() {
                now.0.to_string()
            } else {
                st.date.trimmed().to_string()
            },
            time: now.1.to_string(),
            params,
        };
        let ident = identity_patches(patient, st);
        let series_number: i64 = node.number.trimmed().parse().unwrap_or(300 + n as i64);

        match node.kind {
            ObjKind::Structures | ObjKind::Segmentation => write_structures(
                study,
                node,
                &ctx,
                &ident,
                series_number,
                &written,
                dir,
                files,
                warnings,
            )?,
            ObjKind::Dose => {
                let Some(d) = study.doses.get(node.index) else {
                    continue;
                };
                let plan_ref = study
                    .plans
                    .iter()
                    .find(|p| p.sop_instance_uid == d.referenced_plan_uid)
                    .map(|p| {
                        (
                            dicom_export::plan_sop_class(p),
                            sop_map
                                .get(p.sop_instance_uid.as_str())
                                .copied()
                                .unwrap_or(p.sop_instance_uid.as_str()),
                        )
                    });
                let mut o = dicom_export::build_dose(
                    d,
                    &ctx,
                    node.sop_uid.trimmed(),
                    node.series_uid.trimmed(),
                    series_number,
                    plan_ref,
                );
                dicom_export::apply(&mut o, &ident);
                let path = dir.join(format!("RD_{}.dcm", safe(&node.label, &format!("{n}"))));
                dicom_export::write_object(o, dicom_export::SOP_RTDOSE, &path)?;
                *files += 1;
            }
            ObjKind::Plan => {
                let Some(p) = study.plans.get(node.index) else {
                    continue;
                };
                let struct_ref = sop_map
                    .get(p.referenced_structset_uid.as_str())
                    .copied()
                    .or(if p.referenced_structset_uid.is_empty() {
                        None
                    } else {
                        Some(p.referenced_structset_uid.as_str())
                    });
                let mut o = dicom_export::build_plan(
                    p,
                    &ctx,
                    node.sop_uid.trimmed(),
                    node.series_uid.trimmed(),
                    series_number,
                    struct_ref,
                );
                dicom_export::apply(&mut o, &ident);
                let path = dir.join(format!("RP_{}.dcm", safe(&node.label, &format!("{n}"))));
                dicom_export::write_object(o, dicom_export::plan_sop_class(p), &path)?;
                *files += 1;
            }
        }
    }
    Ok(())
}

/// Frame of Reference an RT object is written with: the one its image series
/// went out under, else what the object itself carried.
fn object_frame_of_reference(
    study: &LoadedStudy,
    node: &ObjNode,
    st: &StudyNode,
    written: &HashMap<String, Written>,
) -> String {
    if let Some(w) = written.get(&node.referenced_series_uid) {
        if !w.for_uid.is_empty() {
            return w.for_uid.clone();
        }
    }
    let own = match node.kind {
        ObjKind::Structures => study
            .structure_sets
            .get(node.index)
            .map(|s| s.frame_of_reference_uid.clone()),
        ObjKind::Segmentation => study
            .seg_series
            .get(node.index)
            .map(|s| s.grid.frame_of_reference_uid.clone()),
        _ => None,
    }
    .unwrap_or_default();
    if !own.is_empty() {
        // Keep it only if no exported series renamed that frame.
        for se in &st.series {
            if se.for_uid.original == own && se.selected {
                return se.for_uid.trimmed().to_string();
            }
        }
        return own;
    }
    st.series
        .iter()
        .find(|s| s.selected)
        .map(|s| s.for_uid.trimmed().to_string())
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Image series
// ---------------------------------------------------------------------------

/// Copy a series file by file, patching only what the plan changed.
#[allow(clippy::too_many_arguments)]
fn copy_series(
    plan: &ExportPlan,
    patient: &PatientNode,
    st: &StudyNode,
    node: &SeriesNode,
    src: &SeriesInfo,
    params: &ExportParams,
    dir: &Path,
    progress: &Progress,
    files: &mut usize,
) -> Result<Option<Written>> {
    use dicom_core::VR;
    let mut base = identity_patches(patient, st);
    base.extend([
        (
            tags::SERIES_INSTANCE_UID,
            VR::UI,
            node.uid.trimmed().to_string(),
        ),
        (
            tags::SERIES_NUMBER,
            VR::IS,
            node.number.trimmed().to_string(),
        ),
        (
            tags::SERIES_DESCRIPTION,
            VR::LO,
            node.description.trimmed().to_string(),
        ),
        (
            tags::FRAME_OF_REFERENCE_UID,
            VR::UI,
            node.for_uid.trimmed().to_string(),
        ),
    ]);
    // On a copy only the rows the user actually edited are applied: the
    // defaults of the common table name *this application* as the equipment,
    // and stamping that over a scanner's own tags would be a lie about where
    // the images came from.
    for f in params
        .fields
        .iter()
        .filter(|f| f.enabled && f.value != f.suggested)
    {
        base.push((f.tag, f.vr, f.value.trim().to_string()));
    }

    let mut out = Written {
        series_uid: node.uid.trimmed().to_string(),
        study_uid: st.uid.trimmed().to_string(),
        for_uid: node.for_uid.trimmed().to_string(),
        spacing: 1.0,
        ..Written::default()
    };
    for (i, file) in src.files.iter().enumerate() {
        if i % 25 == 0 {
            progress.set(format!(
                "{} {}: copying {}/{}",
                src.modality,
                node.description.trimmed(),
                i + 1,
                src.files.len()
            ));
        }
        let mut set = base.clone();
        if plan.uid_mode == UidMode::New {
            set.push((tags::SOP_INSTANCE_UID, VR::UI, dicom_export::new_uid()));
        }
        let dst = dir.join(format!("{}_{i:04}.dcm", safe(&src.modality, "IM")));
        let done = dicom_export::copy_patched(file, &dst, &set)?;
        if out.sop_class.is_empty() {
            out.sop_class = done.sop_class.clone();
            out.normal = done.normal;
        }
        out.slices.push((done.sop_uid, done.axis));
        *files += 1;
    }
    finish(&mut out);
    Ok(Some(out))
}

/// Render a series from its voxels - the path for a volume the application
/// made rather than read.
#[allow(clippy::too_many_arguments)]
fn render_series(
    study: &LoadedStudy,
    patient: &PatientNode,
    st: &StudyNode,
    node: &SeriesNode,
    src: &SeriesInfo,
    params: &ExportParams,
    now: (&str, &str),
    dir: &Path,
    files: &mut usize,
) -> Result<Option<Written>> {
    let Some(vol) = series_volume(study, node.index, src) else {
        return Ok(None);
    };
    let ctx = dicom_export::Ctx {
        study_uid: st.uid.trimmed().to_string(),
        for_uid: node.for_uid.trimmed().to_string(),
        date: if st.date.trimmed().is_empty() {
            now.0.to_string()
        } else {
            st.date.trimmed().to_string()
        },
        time: now.1.to_string(),
        params,
    };
    let ident = identity_patches(patient, st);
    let modality = if src.modality.is_empty() {
        "CT"
    } else {
        &src.modality
    };
    let series_number: i64 = node.number.trimmed().parse().unwrap_or(1);
    let desc = node.description.trimmed().to_string();
    let mut out = Written {
        series_uid: node.uid.trimmed().to_string(),
        study_uid: st.uid.trimmed().to_string(),
        for_uid: node.for_uid.trimmed().to_string(),
        sop_class: dicom_export::SOP_CT.to_string(),
        normal: vol.normal,
        spacing: vol.spacing[2],
        slices: Vec::with_capacity(vol.dims[2]),
    };
    // The plan's own UIDs are per series; the slices need one each. In Keep
    // mode the volume no longer knows which file each slice came from, so
    // they are minted either way and the slice-level identity is the one
    // thing a rendered series cannot preserve.
    for k in 0..vol.dims[2] {
        let sop_uid = dicom_export::new_uid();
        let mut o = dicom_export::build_image_slice(
            &vol,
            k,
            &ctx,
            modality,
            &sop_uid,
            node.uid.trimmed(),
            series_number,
            if desc.is_empty() { None } else { Some(&desc) },
            study.default_window,
        );
        dicom_export::apply(&mut o, &ident);
        // 4D: the phase a series belongs to is an attribute of its images.
        if let Some(t) = src.temporal_id {
            dicom_export::put_is(&mut o, tags::TEMPORAL_POSITION_IDENTIFIER, t);
        }
        let path = dir.join(format!("{}_{k:04}.dcm", safe(modality, "IM")));
        dicom_export::write_object(o, dicom_export::SOP_CT, &path)?;
        let pos = vol.voxel_to_patient(0.0, 0.0, k as f64).dot(vol.normal);
        out.slices.push((sop_uid, pos));
        *files += 1;
    }
    finish(&mut out);
    Ok(Some(out))
}

/// Order the slices along the axis and take the spacing from them, so a
/// contour can be matched to the image it lies on.
fn finish(w: &mut Written) {
    w.slices.sort_by(|a, b| a.1.total_cmp(&b.1));
    if w.slices.len() > 1 {
        let steps: Vec<f64> = w
            .slices
            .windows(2)
            .map(|p| (p[1].1 - p[0].1).abs())
            .filter(|d| *d > 1e-6)
            .collect();
        if !steps.is_empty() {
            w.spacing = steps.iter().sum::<f64>() / steps.len() as f64;
        }
    }
    if w.spacing <= 0.0 {
        w.spacing = 1.0;
    }
}

/// The voxels of a series: the displayed volume when it is the active one,
/// otherwise a fresh read of its files.
fn series_volume(study: &LoadedStudy, index: usize, src: &SeriesInfo) -> Option<Volume> {
    if index == study.active_series && !study.volume.is_empty() {
        return Some((*study.volume).clone());
    }
    if src.files.is_empty() {
        return None;
    }
    crate::loader::load_series_volume(src, &Progress::default())
        .ok()
        .map(|(v, _, _)| v)
}

// ---------------------------------------------------------------------------
// Structures, in either format
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn write_structures(
    study: &LoadedStudy,
    node: &ObjNode,
    ctx: &dicom_export::Ctx,
    ident: &[(Tag, dicom_core::VR, String)],
    series_number: i64,
    written: &HashMap<String, Written>,
    dir: &Path,
    files: &mut usize,
    warnings: &mut Vec<String>,
) -> Result<()> {
    // The image series this belongs to, as it was written.
    let image = match written.get(&node.referenced_series_uid) {
        Some(w) => w.image_ref(),
        None => {
            // Fall back to any exported series in the same frame of
            // reference: contours that sit on a scan still belong to it even
            // when the object named a series that is not in this export.
            let same_for = written
                .values()
                .find(|w| !w.for_uid.is_empty() && w.for_uid == ctx.for_uid);
            match same_for {
                Some(w) => w.image_ref(),
                None => {
                    warnings.push(format!(
                        "“{}” was exported without its image series - it carries a frame of \
                         reference but no reference to the images, so a planning system will \
                         not show it on a scan until they are exported too",
                        node.label
                    ));
                    ImageRef::default()
                }
            }
        }
    };

    let name = safe(&node.label, "structures");
    match (node.kind, node.format) {
        // Native: contours as RTSTRUCT.
        (ObjKind::Structures, StructFormat::RtStruct) => {
            let Some(ss) = study.structure_sets.get(node.index) else {
                return Ok(());
            };
            let mut relabelled = ss.clone();
            relabelled.label = node.description.trimmed().to_string();
            let mut o = dicom_export::build_rtstruct(
                &relabelled,
                ctx,
                series_number,
                node.sop_uid.trimmed(),
                &image,
            );
            dicom_export::apply(&mut o, ident);
            dicom_export::put_str(
                &mut o,
                tags::SERIES_INSTANCE_UID,
                dicom_core::VR::UI,
                node.series_uid.trimmed(),
            );
            dicom_export::write_object(
                o,
                dicom_export::SOP_RTSTRUCT,
                &dir.join(format!("RS_{name}.dcm")),
            )?;
            *files += 1;
        }
        // Contours rasterised onto the image lattice and written as SEG.
        (ObjKind::Structures, StructFormat::Seg) => {
            let Some(ss) = study.structure_sets.get(node.index) else {
                return Ok(());
            };
            let Some(grid) = structure_grid(study, ss) else {
                warnings.push(format!(
                    "“{}” could not be written as SEG: a segmentation needs the voxel lattice \
                     of its image series, and that series is not loaded. It was written as \
                     RTSTRUCT instead",
                    node.label
                ));
                let mut o = dicom_export::build_rtstruct(
                    ss,
                    ctx,
                    series_number,
                    node.sop_uid.trimmed(),
                    &image,
                );
                dicom_export::apply(&mut o, ident);
                dicom_export::write_object(
                    o,
                    dicom_export::SOP_RTSTRUCT,
                    &dir.join(format!("RS_{name}.dcm")),
                )?;
                *files += 1;
                return Ok(());
            };
            let mut ser = SegSeries::new(
                node.description.trimmed().to_string(),
                grid.clone(),
                image.series_uid.clone(),
                ctx.study_uid.clone(),
            );
            let mut empty = Vec::new();
            for roi in &ss.rois {
                match segmentation::rasterize_roi(&grid, roi) {
                    Some(mask) => ser.segs.push(Segmentation::from_mask(
                        roi.name.clone(),
                        roi.color,
                        grid.dims,
                        mask,
                    )),
                    None => empty.push(roi.name.clone()),
                }
            }
            if !empty.is_empty() {
                warnings.push(format!(
                    "“{}” as SEG: {} ROI(s) had no contour inside the image volume and were \
                     left out ({})",
                    node.label,
                    empty.len(),
                    empty.join(", ")
                ));
            }
            write_seg(
                &ser,
                ctx,
                ident,
                series_number,
                node,
                &image,
                dir,
                files,
                &name,
            )?;
        }
        // Native: masks as SEG.
        (ObjKind::Segmentation, StructFormat::Seg) => {
            let Some(sg) = study.seg_series.get(node.index) else {
                return Ok(());
            };
            let mut ser = sg.clone();
            ser.label = node.description.trimmed().to_string();
            write_seg(
                &ser,
                ctx,
                ident,
                series_number,
                node,
                &image,
                dir,
                files,
                &name,
            )?;
        }
        // Masks contoured and written as RTSTRUCT.
        (ObjKind::Segmentation, StructFormat::RtStruct) => {
            let Some(sg) = study.seg_series.get(node.index) else {
                return Ok(());
            };
            let mut ss = StructureSet {
                label: node.description.trimmed().to_string(),
                frame_of_reference_uid: ctx.for_uid.clone(),
                sop_instance_uid: node.sop_uid.trimmed().to_string(),
                series_instance_uid: node.series_uid.trimmed().to_string(),
                study_uid: ctx.study_uid.clone(),
                referenced_series_uid: image.series_uid.clone(),
                file_name: String::new(),
                rois: Vec::new(),
            };
            for (i, seg) in sg.segs.iter().enumerate() {
                if seg.count == 0 {
                    continue;
                }
                ss.rois
                    .push(segmentation::mask_to_roi(seg, &sg.grid, i as i32 + 1));
            }
            if ss.rois.is_empty() {
                warnings.push(format!("“{}” is empty and was not written", node.label));
                return Ok(());
            }
            let mut o = dicom_export::build_rtstruct(
                &ss,
                ctx,
                series_number,
                node.sop_uid.trimmed(),
                &image,
            );
            dicom_export::apply(&mut o, ident);
            dicom_export::put_str(
                &mut o,
                tags::SERIES_INSTANCE_UID,
                dicom_core::VR::UI,
                node.series_uid.trimmed(),
            );
            dicom_export::write_object(
                o,
                dicom_export::SOP_RTSTRUCT,
                &dir.join(format!("RS_{name}.dcm")),
            )?;
            *files += 1;
        }
        _ => {}
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_seg(
    ser: &SegSeries,
    ctx: &dicom_export::Ctx,
    ident: &[(Tag, dicom_core::VR, String)],
    series_number: i64,
    node: &ObjNode,
    image: &ImageRef,
    dir: &Path,
    files: &mut usize,
    name: &str,
) -> Result<()> {
    if ser.segs.iter().all(|s| s.count == 0) {
        return Ok(());
    }
    let sop_uids: Vec<String> = image.slices.iter().map(|(u, _)| u.clone()).collect();
    let seg_ctx = dicomseg::SegWriteCtx {
        study_uid: &ctx.study_uid,
        for_uid: &ctx.for_uid,
        date: &ctx.date,
        time: &ctx.time,
        series_number,
        image_series_uid: &image.series_uid,
        image_sop_uids: &sop_uids,
        params: ctx.params,
    };
    let mut o = dicomseg::build(ser, &seg_ctx);
    dicom_export::apply(&mut o, ident);
    dicom_export::put_str(
        &mut o,
        tags::SOP_INSTANCE_UID,
        dicom_core::VR::UI,
        node.sop_uid.trimmed(),
    );
    dicom_export::put_str(
        &mut o,
        tags::SERIES_INSTANCE_UID,
        dicom_core::VR::UI,
        node.series_uid.trimmed(),
    );
    dicom_export::write_object(o, dicomseg::SOP_SEG, &dir.join(format!("SEG_{name}.dcm")))?;
    *files += 1;
    Ok(())
}

/// The lattice a structure set's contours are rasterised onto.
fn structure_grid(study: &LoadedStudy, ss: &StructureSet) -> Option<crate::volume::Grid> {
    if let Some(i) = study
        .series
        .iter()
        .position(|s| s.uid == ss.referenced_series_uid)
    {
        if let Some(v) = study.series.get(i).and_then(|s| series_volume(study, i, s)) {
            return Some(v.grid());
        }
    }
    if !study.volume.is_empty() {
        return Some(study.volume.grid());
    }
    None
}

impl ExportPlan {
    /// Take the patient and study identity from [`ExportParams`] instead of
    /// from the data. Used by the whole-study convenience export, whose
    /// caller has only the params to say it with.
    pub fn adopt_params_identity(&mut self) {
        let get = |tag| self.params.value(tag).map(|s| s.to_string());
        let (pname, pid) = (get(tags::PATIENT_NAME), get(tags::PATIENT_ID));
        let (sid, sdesc, sdate) = (
            get(tags::STUDY_ID),
            get(tags::STUDY_DESCRIPTION),
            get(tags::STUDY_DATE),
        );
        let sdesc_off = self.params.value(tags::STUDY_DESCRIPTION).is_none();
        let series_desc = get(tags::SERIES_DESCRIPTION);
        for d in &mut self.datasets {
            for p in &mut d.patients {
                if let Some(v) = &pname {
                    p.name.value = v.clone();
                }
                if let Some(v) = &pid {
                    p.id.value = v.clone();
                }
                for st in &mut p.studies {
                    if let Some(v) = &sid {
                        st.id.value = v.clone();
                    }
                    if let Some(v) = &sdesc {
                        st.description.value = v.clone();
                    }
                    if sdesc_off {
                        st.description.value = String::new();
                    }
                    if let Some(v) = &sdate {
                        st.date.value = v.clone();
                    }
                    for se in &mut st.series {
                        if let Some(v) = &series_desc {
                            se.description.value = v.clone();
                        }
                    }
                }
            }
        }
    }
}
