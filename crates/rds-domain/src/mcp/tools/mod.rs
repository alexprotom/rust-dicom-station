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
pub mod fourd;
pub mod output;
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
    "describe_setup" => (setup::SetupArgs, setup::describe_setup, false,
        "The numbers of an earlier register_2d3d again."),
    "export_setup" => (setup::ExportSetupArgs, setup::export_setup, false,
        "Write a setup into the output folder: the correction as a rigid DICOM Spatial \
         Registration (room to planning CT), the DRRs at the answer as RT Images, and the \
         report as JSON."),

    // ---- 4D -----------------------------------------------------------
    "list_4d_groups" => (session::DatasetArgs, fourd::list_4d_groups, false,
        "The 4D groups of a dataset (phases in temporal order, reconstructions such as AVG/MIP)."),
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
         or an ID), their studies by Study Instance UID with the year, the modalities, the \
         description and the series (modality, number, description, files). patient narrows \
         the listing to one alias. The catalog is brought up to date first."),
    "describe_archive_study" => (archive::ArchiveStudyArgs, archive::describe_archive_study, true,
        "One study of the archive series by series, from the headers alone (no pixels, no \
         identifiers): what each series becomes when loaded (image volume, planar image, \
         additional object, report, structure set, dose, plan, ...), its SOP class, image type, pixels, \
         frames, samples per pixel and whether its files carry a position. The way to see why \
         a series is shown as it is, a scanner's protocol page say, without opening it."),
    "open_archive_study" => (archive::ArchiveStudyArgs, archive::open_archive_study, true,
        "Open a study of the archive as a dataset, through the same identity gate as \
         open_dataset: refused under the default policy when its headers still name the \
         patient, opened with the values redacted in memory under redact. Returns the dataset \
         handle and its contents, as open_dataset does."),
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
