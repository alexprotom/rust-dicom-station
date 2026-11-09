//! The tools, as plain functions: `fn(&mut Core, Args, &Progress) -> Result<Value>`.
//!
//! Each argument type derives `Deserialize` and `JsonSchema`, so the schema
//! the client sees is the struct the tool reads: they cannot drift apart.
//! Results are `serde_json::Value`s built freely here; the redaction happens
//! once, in [`Core::call_public`](super::Core::call_public), not in the
//! tools. Free text that comes from DICOM files goes through
//! [`phi::clean_text`](super::phi::clean_text) so it is capped and carries
//! no control characters.

pub mod analysis;
pub mod archive;
pub mod curation;
pub mod fourd;
pub mod output;
pub mod planar;
pub mod register;
pub mod segment;
pub mod session;
pub mod setup;
pub mod synthesis;
pub mod workflows;

use anyhow::{bail, Context, Result};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;

use super::{Core, ToolSpec};
use crate::progress::Progress;

/// Arguments of a tool that takes none.
#[derive(Deserialize, JsonSchema, Default)]
#[serde(default, deny_unknown_fields)]
pub struct NoArgs {}

fn schema_of<T: JsonSchema>() -> Value {
    serde_json::to_value(schemars::schema_for!(T)).expect("a schema serializes")
}

/// The registry: name, argument type, function, whether it is long, and
/// the description the client reads.
macro_rules! registry {
    ($( $name:literal => ($args:ty, $f:path, $long:expr, $desc:literal), )*) => {
        /// Every tool, in this order.
        pub fn specs() -> Vec<ToolSpec> {
            vec![$( ToolSpec {
                name: $name,
                description: $desc,
                schema: schema_of::<$args>(),
                long: $long,
            }, )*]
        }

        /// Parse the arguments and run the tool.
        pub fn dispatch(core: &mut Core, name: &str, args: Value, p: &Progress) -> Result<Value> {
            let args = if args.is_null() { Value::Object(Default::default()) } else { args };
            match name {
                $( $name => {
                    let a: $args = serde_json::from_value(args)
                        .with_context(|| format!("arguments of {}", $name))?;
                    $f(core, a, p)
                } )*
                other => bail!("unknown tool '{other}'"),
            }
        }
    };
}

registry! {
    // ---- session ------------------------------------------------------
    "open_dataset" => (session::OpenArgs, session::open_dataset, true,
        "Open a DICOM folder (or explicit files) under one of the configured roots as a dataset. \
         Returns the dataset handle and its contents: image series, 4D groups, structure sets, \
         segmentations, doses, plans, planar images, reports (a CT dose SR and a scanner's \
         protocol page as tables: CTDIvol, DLP, kV, mAs per acquisition). A dataset that still \
         carries identifying tags is refused \
         under the default policy; anonymize it first (see the anonymize tool). No patient \
         identifiers are ever returned."),
    "describe_dataset" => (session::DatasetArgs, session::describe_dataset, false,
        "Describe an open dataset again: series, groups, structures, doses, plans, planar \
         images, reports with their tables."),
    "list_structures" => (session::DatasetArgs, session::list_structures, false,
        "List every structure of a dataset (RTSTRUCT ROIs and segmentation segments) with its \
         set, kind and colour. Structures are referred to by name in every other tool."),
    "close_dataset" => (session::DatasetArgs, session::close_dataset, false,
        "Close a dataset and drop the registrations that involve it."),
    "describe_session" => (NoArgs, session::describe_session, false,
        "What is open: datasets, registrations, group registrations, motion runs; the PHI policy; \
         the output folder; memory in use."),

    // ---- segment ------------------------------------------------------
    "list_models" => (NoArgs, segment::list_models, false,
        "Every automatic segmentation model segment_organs can run: key, what it segments \
         (modality, class count, sub-models), its licence, and whether its weights are on \
         this machine or how much there is to download."),
    "segment_organs" => (segment::OrgansArgs, segment::segment_organs, true,
        "Run an automatic segmentation model on one image series and file what it finds as \
         segments. model: a key from list_models (TotalSegmentator CT, MR and task models, \
         MRSegmentator, lungmask, MONAI whole body, CT-FM, VISTA-3D); without it, variant picks a TotalSegmentator CT model: \
         fast (3 mm), high (1.5 mm), preview (6 mm). parts chooses the sub-models of a model \
         that has them. tg263 names the structures by AAPM TG-263. Weights must already be \
         present unless downloads are allowed in the configuration."),
    "segment_body" => (segment::BodyArgs, segment::segment_body, true,
        "Contour the patient outline (EXTERNAL) of one image series. method: classical or \
         model_assisted."),
    "combine_structures" => (segment::CombineArgs, segment::combine_structures, true,
        "Boolean algebra on structures of one dataset: union, intersect or subtract (first minus \
         the rest), each operand optionally expanded or shrunk by a margin in mm (uniform or per \
         patient direction), an optional margin and cleanup on the result. Files the result as a \
         new segment."),

    // ---- synthetic CT -------------------------------------------------
    "synthesize_ct" => (synthesis::SynthArgs, synthesis::synthesize_ct, true,
        "Synthesize a CT from an MR or CBCT series with TotalSynth (a 2.5-D UNet++, trained on \
         SynthRAD 2023 and 2025, brain to pelvis) and add it to the dataset as a new CT series \
         on the same frame of reference, written under the output folder. app: sct_mr, \
         sct_cbct, sct_mr_cbct, sct_mr_dixon (default by modality); checkpoints / \
         all_checkpoints and tta (0 to 2) choose between the quick look (CV_0, no mirrors) and \
         the published recipe (all five, two mirrors); mask sets the air outside the patient. \
         The weights are CC BY-NC 4.0 (non-commercial use). Returns the new series number."),
    "evaluate_sct" => (synthesis::EvalArgs, synthesis::evaluate_sct, true,
        "Compare a synthetic CT with a reference CT of the same patient: MAE, mean error and RMSE \
         in HU inside a mask (overall and for air, soft tissue and bone by the reference), PSNR \
         and SSIM over -1024 to 3071 HU, optionally the water-equivalent thickness difference \
         along the three axes (wet, hlut) and the Dice of TotalSegmentator's structures \
         (structures). The reference is resampled onto the synthetic CT through reg, or \
         directly when the two share a frame of reference. Returns the numbers and a CSV."),

    // ---- register and propagate -------------------------------------
    "register" => (register::RegisterArgs, register::register, true,
        "Register a moving series onto a fixed series. method: elastix_rigid, elastix_bspline, \
         plastimatch_bspline. region restricts the run to a structure (plus margin) of the \
         fixed dataset, which makes it local; start refines an earlier registration instead \
         of replacing it; init says where the search starts (auto: the identity when the \
         images overlap, else their centres of gravity; or a structure contoured on both). \
         Returns a reg handle and the quality analysis. The transform maps fixed patient \
         coordinates to moving ones."),
    "register_structures" => (register::ShapeRegisterArgs, register::register_structures, true,
        "Register a moving series onto a fixed series by structures contoured on both, not by \
         their intensities: each structure's surface is laid onto its partner's signed distance \
         map, both ways (rigid or translation only, Gauss-Newton), then optionally a local \
         B-spline refinement per structure on the distance maps (refine). structures pairs them \
         by name (moving names a structure that is named differently on the moving dataset; \
         weight says how much one counts). For two images whose contours agree but whose \
         intensities never will: contrast against none, CT against MR or CBCT. Returns a reg \
         handle like register (propagate, describe_registration and export_registration take \
         it), the analysis, and per structure the mean surface distance and Dice before and \
         after."),
    "describe_registration" => (register::RegArgs, register::describe_registration, false,
        "The numbers of an earlier registration again."),
    "propagate" => (register::PropagateArgs, register::propagate, true,
        "Carry structures across a registration. to: fixed or moving, the side they land on \
         (the side they come from is the other). Each arrives as a new segment with its volume \
         before and after."),

    // ---- patient setup from radiographs --------------------------------
    "register_2d3d" => (setup::Register2d3dArgs, setup::register_2d3d, true,
        "2D/3D registration: the rigid pose of a CT series whose DRRs match planar images of the \
         dataset (kV radiographs, MV portal images; see planar_images in describe_dataset, kind \
         projection; an additional object, a protocol page say, is refused). Each \
         image's geometry is its own RT Image attributes, a plan beam, a calibrated room view or \
         IEC numbers typed in. method: intensity (the images against DRRs; metric gc, ncc, gd, \
         pi, lncc, mi), markers (points of interest of type MARKER, named ones, or markers \
         detected in the CT, paired with spots on the images) or hybrid (markers, then \
         intensity). Returns a setup handle, the correction in patient coordinates and as couch \
         moves, the displacement, per-view similarity before and after, marker residuals and \
         any warning (a weakly constrained direction, a marker that moved). The correction maps \
         the room onto the CT."),
    // ---- planar images: 2D contours, presentation states, 2D/2D -------
    "project_structures" => (planar::ProjectArgs, planar::project_structures, true,
        "Project structures of the CT onto planar images as 2D contours: each structure's          silhouette through the image's camera (a DRR as rendered, a radiograph at the plan), or          at a setup's pose (pose: a setup handle). Files them as a planar structure set and          returns per structure and image the rings, the area in mm2 and the box in pixels."),
    "list_planar_structures" => (planar::ListArgs, planar::list_planar_structures, false,
        "Every planar structure set of a dataset (structures, their contours per image, where          each came from, images not loaded) and every 2D registration."),
    "draw_planar_structure" => (planar::DrawArgs, planar::draw_planar_structure, false,
        "Contour on a planar image: polygons in pixels (the first pixel's centre at [0, 0])          added to, or cut out of, a named structure of a planar structure set."),
    "register_planar" => (planar::RegisterPlanarArgs, planar::register_planar, true,
        "Register one planar image onto another (2D/2D): by intensity (shift, rotation,          optionally scale), by pairs of points or markers found, the points then the          intensity, or deformably (a B-spline). The transform maps the fixed image's frame          (mm about its centre) onto the moving image's. Returns the shift (at the detector and,          with a camera, at the isocentre), the rotation, the scale, the measure before and          after, the pairs' residuals, the warp's statistics and the moved copy's image number."),
    "carry_planar_structures" => (planar::CarryArgs, planar::carry_planar_structures, false,
        "Carry the planar structures of a 2D registration's moving image onto its fixed image,          into a set named after the registration."),
    "setup_from_planar" => (planar::SetupFromPlanarArgs, planar::setup_from_planar, true,
        "Read two (or more) 2D registrations of two views as one 3D setup correction, through          the fixed images' cameras: the translation and the rotation, the couch readout, the          weakly seen axes named. Returns a setup handle as register_2d3d does."),
    "save_planar_structures" => (planar::SaveArgs, planar::save_planar_structures, false,
        "Save a planar structure set as a DICOM Grayscale Softcopy Presentation State into the          output folder, with every image it references that has no file of its own as an RT          Image. Returns the file names."),
    "describe_setup" => (setup::SetupArgs, setup::describe_setup, false,
        "The numbers of an earlier register_2d3d again."),
    "export_setup" => (setup::ExportSetupArgs, setup::export_setup, false,
        "Write a setup into the output folder: the correction as a rigid DICOM Spatial \
         Registration (room to planning CT), the DRRs at the answer as RT Images, and the \
         report as JSON."),

    // ---- 4D -----------------------------------------------------------
    "list_4d_groups" => (session::DatasetArgs, fourd::list_4d_groups, false,
        "The 4D groups of a dataset (phases in temporal order, reconstructions such as AVG/MIP, \
         the ungated untagged series). A group the header rules found carries group, the id \
         the archive's catalog lists the same set under; a call may name a group by it."),
    "propagate_to_group" => (fourd::GroupArgs, fourd::propagate_to_group, true,
        "Register one series onto every phase of a 4D group (deformable, one registration per \
         phase) and carry structures across onto each phase. The per-phase transforms are kept \
         as a greg handle and reused by a later call with the same series and group. Empty \
         structures means register only. anchor: a structure contoured on the source and on \
         every phase (the heart, say) makes the run anchored on it - centroids matched, a rigid \
         fit on the structure plus a margin, then a local deformable refinement - and reports \
         the anchor's Dice against each phase's own contour. Use it for a cardiac CT onto a \
         4DCT, where the two images share no frame of reference."),
    "analyse_motion" => (fourd::MotionArgs, fourd::analyse_motion, true,
        "The 4D motion pipeline: register the reference phase to every other phase (rigid, then \
         deformable on top), propagate the targets, and measure centroid tracks, peak-to-peak \
         amplitudes, correlation with a reference structure (typically the heart), per-phase \
         registration QA, and motion-encompassing ITVs (with optional margin) filed on the \
         reference phase. Returns a run handle and the report."),

    // ---- analysis -----------------------------------------------------
    "compare_structures" => (analysis::CompareArgs, analysis::compare_structures, true,
        "Compare two structures (possibly of different datasets on the same frame of reference): \
         volumes, centroid offset, Dice, HD95, mean surface distance."),
    "compute_dvh" => (analysis::DvhArgs, analysis::compute_dvh, true,
        "Dose-volume histograms of structures against a dose grid of the dataset, with metrics \
         (D95, V20Gy, Dmean and the like), an optional protocol of constraints, and CSV text."),
    "motion_report" => (analysis::RunArgs, analysis::motion_report, false,
        "The report of an earlier analyse_motion run, as JSON and as CSV."),

    // ---- output -------------------------------------------------------
    "export" => (output::ExportArgs, output::export, true,
        "Write a dataset's structures (as SEG or RTSTRUCT), doses, plans and optionally its \
         images as DICOM into the session's output folder. uid_mode keep writes the study's own \
         identifiers; new mints fresh ones."),
    "export_registration" => (output::ExportRegArgs, output::export_registration, true,
        "Write a registration as a DICOM Deformable Spatial Registration object into the output \
         folder."),
    "import_to_archive" => (output::ImportArgs, output::import_to_archive, true,
        "File an exported folder into the station's local archive, where the viewer's start \
         screen offers it."),
    // ---- the archive --------------------------------------------------
    "list_archive" => (archive::ListArchiveArgs, archive::list_archive, true,
        "What the station's archive (its PACS) holds: every patient as an alias (never a name \
         or an ID; their code under the pseudonymized policy), their studies by Study Instance \
         UID with the year, the modalities, the description and the series (modality, number, \
         description, files) and their 4D sets found from the headers (fourd: group id, name, \
         scan, phases, reconstructions, evidence, confirmed; fourd_scans counts scans, a \
         standard and an Extended-FOV set of one scan being one; a series' fourd is its phase \
         or role; fourd_orphans: a scout or untagged series whose phases are not in the \
         archive); labels under the pseudonymized policy. summary gives counts per cohort only \
         (start there for a large archive), and with filter one line per study that passes; \
         filter {\"has_4d\": true | \"probable\" | false} keeps the studies with a 4DCT, with a set \
         only the geometry says is one (to confirm), or without any; cohort and patient narrow \
         the listing; offset and limit page through it (25 patients a page, 100 study lines, \
         next_offset says where the next starts). The catalog is brought up to date first."),
    "list_archive_4d_groups" => (archive::ArchiveStudyArgs, archive::list_archive_4d_groups, true,
        "The 4D groups of one study of the archive, from the catalog alone (no file opened): \
         each group's id, name, scan, evidence (phase-labels, header-4d, same-acquisition, \
         geometry-only = probable, description, custom), whether it is confirmed, and its \
         members (Series Instance UID, number, role phase / untagged / AVG / MIP / MinIP / \
         trace, phase such as 30% or t3, inferred when the phase was read from the series' \
         place, files), with the structure sets drawn on its phases."),
    "describe_archive_study" => (archive::ArchiveStudyArgs, archive::describe_archive_study, true,
        "One study of the archive series by series, from the headers alone (no pixels, no \
         identifiers): what each series becomes when loaded (image volume, planar image, \
         additional object, report, structure set, dose, plan, ...), its SOP class, image type, pixels, \
         frames, samples per pixel and whether its files carry a position. The way to see why \
         a series is shown as it is, a scanner's protocol page say, without opening it. One \
         header per series is read, whatever the study's size; a course study also lists its \
         sessions by day (series, files)."),
    "open_archive_study" => (archive::ArchiveStudyArgs, archive::open_archive_study, true,
        "Open a study of the archive as a dataset, through the same identity gate as \
         open_dataset: refused under the default policy when its headers still name the \
         patient, opened with the values redacted in memory under redact. series opens only \
         those series (their UIDs from describe_archive_study): one session of a course study \
         of thousands of files; fourd_group opens one 4D group (its id from list_archive) with \
         the structure sets drawn on its phases, and the dataset's list_4d_groups lists it \
         under the same id. Returns the dataset handle and its contents, as open_dataset \
         does."),
    // ---- curating the archive ([pacs] in mcp.toml) ---------------------
    "audit_folder" => (curation::AuditArgs, curation::audit_folder, true,
        "Check a folder under the [pacs] roots from the headers and element lengths alone (no \
         pixel decoded), findings grouped by study with a severity (broken, dangling-reference, \
         warning): files that do not parse or parse with a problem (a length past its \
         container, a transfer syntax that does not match the bytes), pixel data shorter than \
         the header asks for, slice gaps, doubled positions, mixed matrices or frames of \
         reference in a series, missing instance numbers, RTSTRUCTs with fewer \
         StructureSetROI than ROIContour items or their frame of reference in (0008,1155), \
         structure sets and registrations referencing series, images or frames that are \
         neither in the folder nor in the archive, contour planes on no slice, structured \
         reports whose Modality is DX. Read-only."),
    "find_duplicates" => (curation::DuplicatesArgs, curation::find_duplicates, true,
        "Which files of a source folder (an import) a [pacs] folder holds already, per source \
         series: identical (same SOP Instance UID and the same raw elements but for \
         ignore_tags, default PatientName and PatientID), pixel-twin (new UIDs, but the same \
         SHA-256 of the pixel data at the same ImagePositionPatient: a re-anonymized copy), \
         hu-twin (the same values, HU for a CT, at the same position, in other bytes: a copy \
         another program re-encoded), partial, ambiguous (twins in several archive series and \
         no one series holding them all: never merge on that), new; with \
         the archive series each matches. A pixel twin counts only within one patient folder \
         (same_patient, default true; folder names through pacs-rules.toml). Inside the \
         source: repeated SOP Instance UIDs (byte-identical, or versions: the largest kept) \
         and series that re-export another (the copy kept by prefer: full, case, plan). \
         Secondary captures and placeholder images are left out of the pixel comparison. \
         Reports splits (a source study or frame with copies in two archive ones) and saves \
         the UID map the copies give, for remap_references. Pixel hashes are kept in the \
         records folder, so only new or changed files are read again. quarantine_copies also \
         plans moving the identical, pixel-twin and hu-twin series and the copies inside the source to \
         the quarantine folder (a manifest for apply_manifest). Otherwise read-only."),
    "check_references" => (curation::AuditArgs, curation::check_references, true,
        "Before an apply: every reference the objects of a folder make (structure sets, plans, \
         doses, RT images, registrations, records, segmentations, reports) to series, \
         instances and frames of reference, checked against the folder and the archive; \
         lists what does not resolve. Read-only."),
    "quarantine" => (curation::QuarantineArgs, curation::quarantine, false,
        "Plan moving files (or the files of some series or studies) under the [pacs] roots to \
         the quarantine folder, keeping their path below the root (cohort first). Writes only \
         a manifest into the records folder and returns its id and the first operations; \
         apply_manifest moves them, undo_journal moves them back."),
    "set_patient_identity" => (curation::IdentityArgs, curation::set_patient_identity, true,
        "Plan setting PatientName and PatientID of files under the [pacs] roots (or of some \
         of their series or studies) to the values given, for the files that differ. The old \
         values are never returned. Writes only a manifest; apply_manifest rewrites the two \
         elements of each file in place wherever they are (in sequences too, where a \
         scanner repeats the identity), every other element left byte for byte as it was, \
         verified by reading the result back before it replaces the file, a copy of each \
         file kept for undo_journal."),
    "apply_manifest" => (curation::ManifestArgs, curation::apply_manifest, true,
        "Do what a planning tool (quarantine, set_patient_identity, find_duplicates with \
         quarantine_copies) planned, by its manifest id. Each operation is journaled in the \
         records folder as it completes; a run that stops (a failure, a cancel) is resumed by \
         applying again. Every path is checked again: only under the [pacs] roots and \
         quarantine, below a root's top level, under 260 characters."),
    "undo_journal" => (curation::ManifestArgs, curation::undo_journal, true,
        "Take back what apply_manifest did for a manifest, newest first: moves are moved \
         back, rewritten files restored from the copy kept, each only if nobody changed it \
         since (its SHA-256 is in the journal). Entries that cannot be undone are skipped \
         and listed."),
    "relink_structure_set" => (curation::RelinkArgs, curation::relink_structure_set, true,
        "Plan re-pointing a structure set at a CT series already in the archive: every image \
         it names is mapped to a target image (mapping pixel: through pixel twins of its own \
         CT; z: by slice position, after checking orientation, in-plane origin and spacing, \
         or by its contours' z where its CT is missing, a slice no contour lies on by its \
         place in the series-level list), and its series, study and frame references follow; \
         ROIs with an empty frame get the target's. companions (the export's plans and doses) \
         are rewritten with it: a plan's structure set reference pointed at it, a dose's plan \
         reference at the one plan among them. merge_series moves series of the export that shared its frame of \
         reference (4D phases) into the target study and frame - only with the pixel mapping. \
         Refuses when an image has no counterpart, an old UID would stay or a reference would \
         lead nowhere; otherwise saves a manifest for apply_manifest."),
    "remap_references" => (curation::RemapArgs, curation::remap_references, true,
        "Plan rewriting the UIDs objects (structure sets, plans, doses, RT images, \
         registrations, records) refer to, through three typed maps - objects, studies, \
         frames of reference - chosen by where each UID sits, never by its value (a Philips 4D \
         series shares its UID with its frame). The maps come from the pixel twins of a source \
         folder in the archive (source, against; a source study or frame with copies in two \
         archive ones is refused as a split), from a saved map, or by hand. The objects' own \
         SOP and series UIDs and private elements are never touched. Refuses if an old UID \
         would stay; lists references that would not resolve. Saves the map and a manifest."),
    "merge_studies" => (curation::MergeArgs, curation::merge_studies, true,
        "Plan merging studies of the archive into one (an earlier split: a course study cut \
         in two, a registration's two-file study): every file under the paths that holds one \
         of the from Study Instance UIDs, as its own or in a reference, gets the into UID. A \
         manifest for apply_manifest; plan_reorganize then sorts the merged files."),
    "merge_frames" => (curation::MergeArgs, curation::merge_frames, true,
        "Plan merging frames of reference into one (one patient position given two FoR UIDs \
         by two systems): every Frame of Reference UID under the paths, own or referenced, \
         that is one of from becomes into. A manifest for apply_manifest."),
    "dedupe_rt_objects" => (curation::DedupeArgs, curation::dedupe_rt_objects, true,
        "Which RT objects of an import (structure sets, plans, doses, RT images, \
         registrations, records, reports) the archive holds already, when re-anonymization \
         gave them new UIDs: by a content fingerprint with every referenced UID put through \
         the UID map (from the image twins, use_twins, and a saved map), structure sets first, \
         then plans, then the rest; else by content without UIDs told apart by the creation \
         stamp, by a unique stamp, or by a plan's beams or a structure set's contours. Copies \
         inside the import keep the largest. Saves the extended map for remap_references; \
         quarantine_copies plans moving the copies."),
    "phi_report" => (curation::PhiArgs, curation::phi_report, true,
        "Where the patients' identifiers occur in a folder (an identified import), per \
         patient folder: the tag places (sequence>element) whose text holds a token of any \
         name or ID the patient's files carry, with file counts; the removed-list tags present; \
         full birth dates; private elements. No value is returned. Read-only: what to know \
         before plan_pseudonymize."),
    "plan_pseudonymize" => (curation::PseudonymizeArgs, curation::plan_pseudonymize, true,
        "Plan pseudonymizing a folder as the archive keeps its trial data: PatientName and \
         PatientID become the patient's code (patient, or the patient folder through \
         pacs-rules.toml), the birth date YYYY0101, the other names and IDs, addresses, \
         phones, record locator, issuer and physicians removed; dates, UIDs, staff names and \
         private elements (keep_private, default true) kept. In place, or moved to move_to as \
         rewritten. Each file is rewritten in memory first and scanned: a token of any of the \
         patient's names or IDs left in any text element (nested, private) refuses the plan, \
         by tag. Writes only a manifest."),
    "patient_demographics" => (curation::PhiArgs, curation::patient_demographics, true,
        "Patients whose files disagree on PatientBirthDate or PatientSex across exports: how \
         many distinct values and files per variant, never the values. Read-only."),
    "set_patient_demographics" => (curation::DemographicsArgs, curation::set_patient_demographics, true,
        "Plan setting PatientBirthDate (YYYY0101 or empty) and PatientSex (M, F, O or empty) \
         in a patient's files, verified and journaled like set_patient_identity; old values \
         are never returned. Writes only a manifest."),
    "plan_reorganize" => (curation::ReorganizeArgs, curation::plan_reorganize, true,
        "Plan sorting a folder (an import, or a cohort of the archive) into the archive's \
         layout cohort/patient/study/series/file by the rules in records/pacs-rules.toml (or \
         the defaults): study folders <date>_<Upright|Supine>-<note>_StudyUID-<last 12>, \
         series folders Series-<n>_<Modality>_<note>_SeriesUID-<last 12>, files \
         <CT|RS|RP|RD|RI|...>.<SOP UID>.dcm; dates and notes from the earliest session, \
         Upright by the imaging scanner, patient folders mapped to codes. A study the archive \
         has is merged into its folder; a byte-identical copy goes to quarantine; \
         PatientName/PatientID become the code unless the cohort keeps them; non-DICOM \
         attachments are reported or moved (attachments). Writes only a manifest."),
    "apply_reorganize" => (curation::ManifestArgs, curation::apply_reorganize, true,
        "Apply a plan_reorganize manifest: the identity rewrites first (verified, kept for \
         undo), then the moves; journaled, resumable, undone by undo_journal."),
    "catalog_relink" => (curation::CohortArgs, curation::catalog_relink, false,
        "In the station's PACS catalog, file every study of a cohort under one patient entry \
         per patient folder (the entry whose ID is the folder's code, else the one with most \
         files), carrying the other entries' labels and cohort memberships over and dropping \
         entries no study uses. A copy of the catalog is kept in records/catalog-backups."),
    "set_4d_group" => (curation::Set4dArgs, curation::set_4d_group, false,
        "Make a 4D group of a study of the station's PACS catalog by hand: members are Series \
         Instance UIDs, phases in order (or {series, role, phase}); it replaces every group \
         the rules found that shares a series with it, keeps that group's id, and survives \
         rescans. A copy of the catalog is kept in records/catalog-backups first."),
    "confirm_4d" => (curation::Confirm4dArgs, curation::confirm_4d, false,
        "Decide about a 4D group of the station's PACS catalog: yes confirms it (a probable, \
         geometry-only set becomes a 4DCT), no dissolves it (it stays away through rescans), \
         reset forgets what was decided. A copy of the catalog is kept first."),
    "catalog_ignore" => (curation::IgnoreArgs, curation::catalog_ignore, false,
        "List, add or remove the folder name patterns a scan of the station's PACS never looks \
         into ('*' any run of characters, case ignored), beside the built-in ones (to_be_sorted, \
         *-duplicates, *_duplicates): staging and quarantine folders whose copies would relink \
         sorted studies. Their files leave the catalog with the next scan; nothing on disk is \
         touched. A copy of the catalog is kept first."),
    "check_patient_identity" => (NoArgs, curation::check_patient_identity, true,
        "Find Patient IDs of the station's PACS catalog filed in several cohort folders whose \
         files share no birth date and sex (two people under one code, listed as one \
         patient): reads one header per study, returns the cohorts, folders (for codes), \
         study counts and how many variants, never the values. Split by recoding one \
         cohort's folder with set_patient_identity."),
    "set_label" => (curation::LabelArgs, curation::set_label, false,
        "Set the label of a patient (by key) or a study (by UID) in the station's PACS \
         catalog; empty clears it. A copy of the catalog is kept first."),
    "fill_study_labels" => (curation::CohortArgs, curation::fill_study_labels, true,
        "Label every unlabelled study of a cohort in the PACS catalog as '<YYYY-MM-DD|undated> \
         (<Upright|Supine>: <note>) \u{b7} <modalities>': the study's date or its folder's, the \
         position and note of its folder's name (or the scanner and the description), the \
         sorted modalities. A label someone typed is never overwritten; a copy of the catalog \
         is kept first."),
    "open_in_viewer" => (output::OpenViewerArgs, output::open_in_viewer, false,
        "Launch the Rust DICOM Station viewer on one or two folders (an exported one, or a root \
         subfolder), so a person can look at the result."),
    // ---- workflows ----------------------------------------------------
    "list_workflows" => (NoArgs, workflows::list_workflows, false,
        "The saved workflows the server can run (the workflows folder and the program's \
         examples): name, description, steps and the inputs each needs, by title. A workflow \
         is the viewer's own steps wired into a file - load, segment, register, propagate, \
         measure the motion, export."),
    "run_workflow" => (workflows::RunArgs, workflows::run_workflow, true,
        "Run a saved workflow headless on folders under the roots, one per input (by title; \
         see list_workflows). The inputs must pass the PHI gate (anonymized, unless the \
         policy is allow). Everything it writes goes into a run folder under the output \
         folder. Returns every step's result, the reports' tables, and with open_results the \
         studies it worked on as datasets. A workflow over a folder of cases runs once per \
         subfolder and reports a table across them."),
    "anonymize" => (output::AnonymizeArgs, output::anonymize, true,
        "Write an anonymized copy of a folder under a root into the output folder: patient \
         identifiers replaced by a deterministic alias, dates fixed, physicians and institution \
         cleared, private tags removed, UIDs remapped consistently. Never in place. The copy can \
         then be opened under the default policy."),
}
