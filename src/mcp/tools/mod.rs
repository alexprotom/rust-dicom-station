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
pub mod fourd;
pub mod output;
pub mod register;
pub mod segment;
pub mod session;

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
         segmentations, doses, plans. A dataset that still carries identifying tags is refused \
         under the default policy; anonymize it first (see the anonymize tool). No patient \
         identifiers are ever returned."),
    "describe_dataset" => (session::DatasetArgs, session::describe_dataset, false,
        "Describe an open dataset again: series, groups, structures, doses, plans."),
    "list_structures" => (session::DatasetArgs, session::list_structures, false,
        "List every structure of a dataset (RTSTRUCT ROIs and segmentation segments) with its \
         set, kind and colour. Structures are referred to by name in every other tool."),
    "close_dataset" => (session::DatasetArgs, session::close_dataset, false,
        "Close a dataset and drop the registrations that involve it."),
    "describe_session" => (NoArgs, session::describe_session, false,
        "What is open: datasets, registrations, group registrations, motion runs; the PHI policy; \
         the output folder; memory in use."),

    // ---- segment ------------------------------------------------------
    "segment_organs" => (segment::OrgansArgs, segment::segment_organs, true,
        "Run the TotalSegmentator re-implementation on one image series and file the organs as \
         segments. variant: fast (3 mm), high (1.5 mm), preview (6 mm). For the high variant, \
         parts chooses the sub-models: organs, vertebrae, cardiac, muscles, ribs. Weights must \
         already be present unless downloads are allowed in the configuration."),
    "segment_body" => (segment::BodyArgs, segment::segment_body, true,
        "Contour the patient outline (EXTERNAL) of one image series. method: classical or \
         model_assisted."),
    "combine_structures" => (segment::CombineArgs, segment::combine_structures, true,
        "Boolean algebra on structures of one dataset: union, intersect or subtract (first minus \
         the rest), each operand optionally expanded or shrunk by a margin in mm (uniform or per \
         patient direction), an optional margin and cleanup on the result. Files the result as a \
         new segment."),

    // ---- register and propagate -------------------------------------
    "register" => (register::RegisterArgs, register::register, true,
        "Register a moving series onto a fixed series. method: elastix_rigid, elastix_bspline, \
         plastimatch_bspline. region restricts a deformable run to a structure (plus margin) of \
         the fixed dataset, which makes it local; start refines an earlier registration instead \
         of replacing it. Returns a reg handle and the quality analysis. The transform maps fixed \
         patient coordinates to moving ones."),
    "describe_registration" => (register::RegArgs, register::describe_registration, false,
        "The numbers of an earlier registration again."),
    "propagate" => (register::PropagateArgs, register::propagate, true,
        "Carry structures across a registration. to: fixed or moving, the side they land on \
         (the side they come from is the other). Each arrives as a new segment with its volume \
         before and after."),

    // ---- 4D -----------------------------------------------------------
    "list_4d_groups" => (session::DatasetArgs, fourd::list_4d_groups, false,
        "The 4D groups of a dataset (phases in temporal order, reconstructions such as AVG/MIP)."),
    "propagate_to_group" => (fourd::GroupArgs, fourd::propagate_to_group, true,
        "Register one series onto every phase of a 4D group (deformable, one registration per \
         phase) and carry structures across onto each phase. The per-phase transforms are kept \
         as a greg handle and reused by a later call with the same series and group. Empty \
         structures means register only."),
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
    "open_in_viewer" => (output::OpenViewerArgs, output::open_in_viewer, false,
        "Launch the Rust DICOM Station viewer on one or two folders (an exported one, or a root \
         subfolder), so a person can look at the result."),
    "anonymize" => (output::AnonymizeArgs, output::anonymize, true,
        "Write an anonymized copy of a folder under a root into the output folder: patient \
         identifiers replaced by a deterministic alias, dates fixed, physicians and institution \
         cleared, private tags removed, UIDs remapped consistently. Never in place. The copy can \
         then be opened under the default policy."),
}
