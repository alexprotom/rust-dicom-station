//! Every kind of node a workflow can hold: its ports, its parameters and the
//! few lines it shows about itself on the canvas.
//!
//! A node's parameters are a plain struct per kind, gathered in the enum
//! [`Op`]; in the file the kind is `kind` and the struct is `params`
//! (serde's adjacent tagging). Every struct reads with `#[serde(default)]`,
//! so a parameter added later gets its default in an older file, and the
//! choices inside are string enums (`"structure_set"`, `"fast"`), so the
//! file reads as what it means.
//!
//! **Ports are typed.** What flows along a wire is one of six things
//! ([`PortType`]); an input lists what it accepts. Several inputs accept
//! either a 4D group or a set of structures: structures made on every phase
//! of a group carry that group with them, so a wire from the step that made
//! them both names the group and makes the next step wait for them.
//!
//! What each node *does* is in `nodes.rs`; this file only describes.

use serde::{Deserialize, Serialize};

use crate::autoseg::classes::TOTAL_CLASS_NAMES;

// The parameters every caller of an engine shares - the workflow file, the
// MCP tools, the viewer's dialogs - are defined once, in `workflow::params`;
// the workflow file names them as it always did.
pub use crate::workflow::params::{
    AnchorBy, AutosegVariant, BodyMethod, DeformMethod, Device, Effort, FinishParams, Landing,
    NameClash, OutputKind, RegInit, RegMethodChoice, SetChoice,
};

/// What travels along a wire.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PortType {
    /// Everything read from one folder: patients, studies, series, RT
    /// objects, 4D groups.
    Study,
    /// One image series of a study.
    Image,
    /// One 4D group of a study: its phases, in order.
    Group,
    /// Structures by name, on one image or on every phase of a group.
    Structures,
    /// A registration, or one per phase of a group.
    Registration,
    /// What a step measured: tables, notes, a motion report.
    Report,
}

impl PortType {
    pub fn label(self) -> &'static str {
        match self {
            PortType::Study => "a study",
            PortType::Image => "an image series",
            PortType::Group => "a 4D group",
            PortType::Structures => "structures",
            PortType::Registration => "a registration",
            PortType::Report => "a report",
        }
    }

    /// The pin colour on the canvas, one per type, so a wire's colour says
    /// what it carries.
    pub fn color(self) -> [u8; 3] {
        match self {
            PortType::Study => [165, 165, 175],
            PortType::Image => [170, 110, 235],
            PortType::Group => [235, 165, 55],
            PortType::Structures => [70, 200, 120],
            PortType::Registration => [80, 150, 245],
            PortType::Report => [235, 90, 90],
        }
    }
}

/// One input of a node.
#[derive(Clone, Copy, Debug)]
pub struct PortSpec {
    pub name: &'static str,
    pub accepts: &'static [PortType],
    /// The node runs without it.
    pub optional: bool,
    /// More than one wire may arrive.
    pub many: bool,
    pub hint: &'static str,
}

impl PortSpec {
    const fn one(name: &'static str, accepts: &'static [PortType], hint: &'static str) -> Self {
        PortSpec {
            name,
            accepts,
            optional: false,
            many: false,
            hint,
        }
    }

    const fn optional(self) -> Self {
        PortSpec {
            optional: true,
            ..self
        }
    }

    const fn many(self) -> Self {
        PortSpec { many: true, ..self }
    }

    /// `a study or an image series`.
    pub fn accepts_text(&self) -> String {
        let labels: Vec<&str> = self.accepts.iter().map(|t| t.label()).collect();
        match labels.len() {
            0 => String::new(),
            1 => labels[0].to_string(),
            n => format!("{} or {}", labels[..n - 1].join(", "), labels[n - 1]),
        }
    }
}

/// One output of a node.
#[derive(Clone, Copy, Debug)]
pub struct OutSpec {
    pub name: &'static str,
    pub ty: PortType,
    pub hint: &'static str,
}

const fn out(name: &'static str, ty: PortType, hint: &'static str) -> OutSpec {
    OutSpec { name, ty, hint }
}

use PortType as T;

/// The groups the node palette is arranged in, and the header colour of
/// their nodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Category {
    Input,
    Select,
    Segment,
    Edit,
    Register,
    FourD,
    Measure,
    Output,
}

impl Category {
    pub const ALL: [Category; 8] = [
        Category::Input,
        Category::Select,
        Category::Segment,
        Category::Edit,
        Category::Register,
        Category::FourD,
        Category::Measure,
        Category::Output,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Category::Input => "Input",
            Category::Select => "Find in the data",
            Category::Segment => "Segment",
            Category::Edit => "Edit structures",
            Category::Register => "Register and propagate",
            Category::FourD => "4D",
            Category::Measure => "Measure the dose",
            Category::Output => "Output",
        }
    }

    /// Header fill of a node of this category.
    pub fn color(self) -> [u8; 3] {
        match self {
            Category::Input => [70, 85, 110],
            Category::Select => [40, 110, 115],
            Category::Segment => [55, 115, 60],
            Category::Edit => [95, 110, 45],
            Category::Register => [50, 80, 140],
            Category::FourD => [140, 95, 35],
            Category::Measure => [120, 70, 130],
            Category::Output => [125, 45, 70],
        }
    }
}

/// What the palette and the canvas say about a kind.
pub struct KindInfo {
    pub name: &'static str,
    pub glyph: &'static str,
    pub category: Category,
    pub blurb: &'static str,
    /// The node is the end of a branch by nature: nothing is expected to
    /// read what it makes.
    pub ends_a_branch: bool,
    /// The node writes files.
    pub writes: bool,
}

/// Every kind of node, for the palette.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    LoadFolder,
    LoadFolders,
    LoadFromArchive,
    Anonymize,
    SelectImage,
    SelectGroup,
    SelectStructures,
    AutoSegment,
    BodyContour,
    SegVolText,
    Combine,
    Rename,
    Register,
    Propagate,
    Transfer,
    PropagateToGroup,
    CopyToPhases,
    Motion,
    Dvh,
    DoseMetrics,
    ExportDicom,
    SaveReport,
    ArchiveImport,
    Drr,
}

impl Kind {
    pub const ALL: [Kind; 24] = [
        Kind::LoadFolder,
        Kind::LoadFolders,
        Kind::LoadFromArchive,
        Kind::Anonymize,
        Kind::SelectImage,
        Kind::SelectGroup,
        Kind::SelectStructures,
        Kind::AutoSegment,
        Kind::BodyContour,
        Kind::SegVolText,
        Kind::Combine,
        Kind::Rename,
        Kind::Register,
        Kind::Propagate,
        Kind::Transfer,
        Kind::PropagateToGroup,
        Kind::CopyToPhases,
        Kind::Motion,
        Kind::Dvh,
        Kind::DoseMetrics,
        Kind::ExportDicom,
        Kind::SaveReport,
        Kind::ArchiveImport,
        Kind::Drr,
    ];

    pub fn info(self) -> &'static KindInfo {
        match self {
            Kind::LoadFolder => &KindInfo {
                name: "DICOM folder",
                glyph: "📂",
                category: Category::Input,
                blurb: "Read every DICOM file in a folder and its subfolders: patients, studies, \
                        series, RT structures, doses, plans, and the 4D groups among the series. \
                        The folder can be changed in the run dialog, so the same workflow runs \
                        on the next patient.",
                ends_a_branch: false,
                writes: false,
            },
            Kind::LoadFolders => &KindInfo {
                name: "DICOM folders",
                glyph: "📁",
                category: Category::Input,
                blurb: "A batch: the workflow runs once for every subfolder of a folder (one \
                        patient each), every case in a folder of its own, and ends with one \
                        table across the cases for every table its reports make.",
                ends_a_branch: false,
                writes: false,
            },
            Kind::LoadFromArchive => &KindInfo {
                name: "From the archive",
                glyph: "🏥",
                category: Category::Input,
                blurb: "Take a study out of the station's local archive (Tools > PACS): the \
                        patient by ID or name, the study by date or description, or the \
                        newest.",
                ends_a_branch: false,
                writes: false,
            },
            Kind::Anonymize => &KindInfo {
                name: "Anonymize",
                glyph: "🔏",
                category: Category::Input,
                blurb: "Write an anonymized copy of the study's folder into the run folder - \
                        identifiers replaced by an alias, dates fixed, private tags removed, \
                        UIDs remapped - and go on with the copy.",
                ends_a_branch: false,
                writes: true,
            },
            Kind::SelectImage => &KindInfo {
                name: "Image series",
                glyph: "🔍",
                category: Category::Select,
                blurb: "Pick one image series of a study by modality, by words in its \
                        description and by size - the planning CT, the cardiac CT.",
                ends_a_branch: false,
                writes: false,
            },
            Kind::SelectGroup => &KindInfo {
                name: "4D group",
                glyph: "🎞",
                category: Category::Select,
                blurb: "Pick the 4D group of a study - its phases in temporal order - by \
                        words in its name. When the loader recognised none, the image series \
                        of one modality can be grouped instead.",
                ends_a_branch: false,
                writes: false,
            },
            Kind::SelectStructures => &KindInfo {
                name: "Find structures",
                glyph: "🎯",
                category: Category::Select,
                blurb: "Find structures by name in RT structure sets and segmentations: \
                        exact names or patterns with * (target*, GTV*), case does not \
                        matter. On an image series it looks at the structures drawn on \
                        that series; on a 4D group, at every phase.",
                ends_a_branch: false,
                writes: false,
            },
            Kind::AutoSegment => &KindInfo {
                name: "Auto-segmentation",
                glyph: "🔬",
                category: Category::Segment,
                blurb: "TotalSegmentator (117 classes, pure Rust) on one image series or on \
                        every phase of a 4D group. Keeps the organs you list, under the names \
                        you give them, as RT structures in the image's own structure set, as \
                        segments, or both.",
                ends_a_branch: false,
                writes: false,
            },
            Kind::BodyContour => &KindInfo {
                name: "Body contour",
                glyph: "👤",
                category: Category::Segment,
                blurb: "The patient outline (EXTERNAL) of one image series or of every phase \
                        of a 4D group: classical thresholding and morphology, or guided by \
                        TotalSegmentator's body network.",
                ends_a_branch: false,
                writes: false,
            },
            Kind::SegVolText => &KindInfo {
                name: "Prompt by name",
                glyph: "💬",
                category: Category::Segment,
                blurb: "SegVol, prompted with structure names in plain text (liver, \
                        pancreas, aorta), on one image series or on every phase of a 4D \
                        group. For what TotalSegmentator's fixed classes do not name.",
                ends_a_branch: false,
                writes: false,
            },
            Kind::Combine => &KindInfo {
                name: "Combine structures",
                glyph: "⊕",
                category: Category::Edit,
                blurb: "Boolean algebra with margins: A alone with a margin (a PTV from an \
                        ITV), A ∪ B, A ∩ B or A minus B, each side grown or shrunk first, the \
                        result cleaned. On an image, or on every phase of a 4D group.",
                ends_a_branch: false,
                writes: false,
            },
            Kind::Rename => &KindInfo {
                name: "Rename or delete",
                glyph: "✏",
                category: Category::Edit,
                blurb: "Rename structures by name or pattern (GTV* to GTV), or delete them, \
                        where they are: on an image, on every phase of a 4D group, or in \
                        the whole study.",
                ends_a_branch: false,
                writes: false,
            },
            Kind::Register => &KindInfo {
                name: "Register",
                glyph: "⇄",
                category: Category::Register,
                blurb: "Register a moving image series onto a fixed one: rigid, or rigid \
                        plus B-spline (elastix or plastimatch style).",
                ends_a_branch: false,
                writes: false,
            },
            Kind::Propagate => &KindInfo {
                name: "Propagate",
                glyph: "➕",
                category: Category::Register,
                blurb: "Carry structures across a registration onto the other image series.",
                ends_a_branch: false,
                writes: false,
            },
            Kind::Transfer => &KindInfo {
                name: "Transfer by relationship",
                glyph: "📌",
                category: Category::Register,
                blurb: "Place a structure onto another image at the same offset from a \
                        reference structure both images have - a target placed by the heart \
                        when the two share no registration. Onto one image, or onto every \
                        phase of a 4D group.",
                ends_a_branch: false,
                writes: false,
            },
            Kind::PropagateToGroup => &KindInfo {
                name: "Propagate to 4D group",
                glyph: "⏩",
                category: Category::FourD,
                blurb: "Carry structures from one image series onto every phase of a 4D \
                        group, one registration per phase. With an anchor - a structure \
                        contoured on the source and on every phase, the heart for a cardiac \
                        CT onto a 4DCT - the run is anchored on it and its Dice on every \
                        phase is the check.",
                ends_a_branch: false,
                writes: false,
            },
            Kind::CopyToPhases => &KindInfo {
                name: "Copy to each phase",
                glyph: "🔁",
                category: Category::FourD,
                blurb: "Copy structures onto every phase of a 4D group as they are, in \
                        patient coordinates - no registration: the structure stays where it \
                        is while the anatomy under it moves (a fixed margin, a couch, an ITV).",
                ends_a_branch: false,
                writes: false,
            },
            Kind::Motion => &KindInfo {
                name: "Motion and ITV",
                glyph: "📈",
                category: Category::FourD,
                blurb: "Measure the motion of targets through the phases of a 4D group - \
                        as contoured on every phase, rigidly and deformably - their \
                        correlation with a reference structure, and build the \
                        motion-encompassing ITV.",
                ends_a_branch: false,
                writes: false,
            },
            Kind::Dvh => &KindInfo {
                name: "DVH",
                glyph: "📊",
                category: Category::Measure,
                blurb: "Dose-volume histograms of structures against a dose of the study, \
                        the metrics you list (D95%, V20Gy, Dmean) and, with a protocol, \
                        which constraints hold.",
                ends_a_branch: false,
                writes: false,
            },
            Kind::DoseMetrics => &KindInfo {
                name: "Dose estimation",
                glyph: "📐",
                category: Category::Measure,
                blurb: "One table of dose metrics per structure against the physical or the \
                        RBE-weighted dose - what the Dose estimation module shows.",
                ends_a_branch: false,
                writes: false,
            },
            Kind::ExportDicom => &KindInfo {
                name: "Export DICOM",
                glyph: "💾",
                category: Category::Output,
                blurb: "Write the structures of the studies it is given - as RTSTRUCT or \
                        SEG, only what the run made or changed, or all of them - and \
                        optionally the images, into a folder of the run.",
                ends_a_branch: true,
                writes: true,
            },
            Kind::SaveReport => &KindInfo {
                name: "Save report",
                glyph: "📋",
                category: Category::Output,
                blurb: "Write what the steps measured as CSV tables and one readable \
                        summary into a folder of the run.",
                ends_a_branch: true,
                writes: true,
            },
            Kind::ArchiveImport => &KindInfo {
                name: "File in the archive",
                glyph: "📥",
                category: Category::Output,
                blurb: "File the studies into the station's local archive: what the run \
                        exported of them, or the folders they were read from (an anonymized \
                        copy, say).",
                ends_a_branch: true,
                writes: true,
            },
            Kind::Drr => &KindInfo {
                name: "DRR",
                glyph: "☢",
                category: Category::Output,
                blurb: "Digitally reconstructed radiographs of an image series at the angles \
                        you list, or at the plan's beams: PNG files in the run folder, and \
                        planar images in the study.",
                ends_a_branch: false,
                writes: true,
            },
        }
    }

    pub fn inputs(self) -> &'static [PortSpec] {
        const STUDY: &[PortType] = &[T::Study];
        const IMAGE_OR_GROUP: &[PortType] = &[T::Image, T::Group];
        const ANYWHERE: &[PortType] = &[T::Study, T::Image, T::Group];
        const DATA: &[PortType] = &[T::Study, T::Image, T::Group, T::Structures];
        const GROUPISH: &[PortType] = &[T::Group, T::Structures];
        match self {
            Kind::LoadFolder | Kind::LoadFolders | Kind::LoadFromArchive => &[],
            Kind::Anonymize => {
                const P: &[PortSpec] = &[PortSpec::one(
                    "Study",
                    STUDY,
                    "the study whose folder is copied anonymized",
                )];
                P
            }
            Kind::SegVolText => {
                const P: &[PortSpec] = &[PortSpec::one(
                    "Images",
                    IMAGE_OR_GROUP,
                    "one image series, or a 4D group to run on every phase",
                )];
                P
            }
            Kind::Combine => {
                const P: &[PortSpec] = &[
                    PortSpec::one(
                        "A",
                        &[T::Structures],
                        "the first operand (several names are joined first)",
                    ),
                    PortSpec::one(
                        "B",
                        &[T::Structures],
                        "the second operand; leave it free to grow or shrink A alone",
                    )
                    .optional()
                    .many(),
                ];
                P
            }
            Kind::Rename => {
                const P: &[PortSpec] = &[PortSpec::one(
                    "Structures",
                    &[T::Structures],
                    "where to rename: the image, the phases or the study they were found on",
                )];
                P
            }
            Kind::Transfer => {
                const P: &[PortSpec] = &[
                    PortSpec::one(
                        "Target",
                        &[T::Structures],
                        "what to place, on the source image",
                    ),
                    PortSpec::one(
                        "Reference",
                        &[T::Structures],
                        "the reference structure on the source image (the heart)",
                    ),
                    PortSpec::one(
                        "Onto",
                        &[T::Structures],
                        "the same reference structure on the destination: one image, or \
                         every phase of a 4D group",
                    ),
                ];
                P
            }
            Kind::CopyToPhases => {
                const P: &[PortSpec] = &[
                    PortSpec::one("Structures", &[T::Structures], "what to copy"),
                    PortSpec::one("Onto", GROUPISH, "the 4D group, or structures made on it"),
                ];
                P
            }
            Kind::Dvh | Kind::DoseMetrics => {
                const P: &[PortSpec] = &[PortSpec::one(
                    "Structures",
                    &[T::Structures],
                    "the structures, on the image the dose was planned on",
                )
                .many()];
                P
            }
            Kind::ArchiveImport => {
                const P: &[PortSpec] = &[PortSpec::one(
                    "Data",
                    DATA,
                    "the studies to file: a study, or anything made on one",
                )
                .many()];
                P
            }
            Kind::Drr => {
                const P: &[PortSpec] = &[PortSpec::one(
                    "Image",
                    &[T::Image],
                    "the image series to project",
                )];
                P
            }
            Kind::SelectImage => {
                const P: &[PortSpec] = &[PortSpec::one("Study", STUDY, "the study to look in")];
                P
            }
            Kind::SelectGroup => {
                const P: &[PortSpec] = &[PortSpec::one("Study", STUDY, "the study to look in")];
                P
            }
            Kind::SelectStructures => {
                const P: &[PortSpec] = &[PortSpec::one(
                    "In",
                    ANYWHERE,
                    "a whole study, one image series, or every phase of a 4D group",
                )];
                P
            }
            Kind::AutoSegment | Kind::BodyContour => {
                const P: &[PortSpec] = &[PortSpec::one(
                    "Images",
                    IMAGE_OR_GROUP,
                    "one image series, or a 4D group to run on every phase",
                )];
                P
            }
            Kind::Register => {
                const P: &[PortSpec] = &[
                    PortSpec::one("Fixed", &[T::Image], "the image that stays put"),
                    PortSpec::one("Moving", &[T::Image], "the image that is moved onto it"),
                ];
                P
            }
            Kind::Propagate => {
                const P: &[PortSpec] = &[
                    PortSpec::one(
                        "Registration",
                        &[T::Registration],
                        "a registration of two images",
                    ),
                    PortSpec::one(
                        "Structures",
                        &[T::Structures],
                        "structures on one of its two images; they land on the other",
                    ),
                ];
                P
            }
            Kind::PropagateToGroup => {
                const P: &[PortSpec] = &[
                    PortSpec::one(
                        "Structures",
                        &[T::Structures],
                        "what to carry, on the source image series",
                    ),
                    PortSpec::one(
                        "Anchor",
                        &[T::Structures],
                        "a structure on the source that every phase also has (the heart); \
                     leave it free for a plain deformable run",
                    )
                    .optional(),
                    PortSpec::one(
                        "Onto",
                        GROUPISH,
                        "the 4D group, or structures made on it (which then run first)",
                    ),
                ];
                P
            }
            Kind::Motion => {
                const P: &[PortSpec] = &[
                    PortSpec::one(
                        "Targets",
                        &[T::Structures],
                        "the targets, on the 4D group (or on its reference phase)",
                    ),
                    PortSpec::one(
                        "Reference",
                        &[T::Structures],
                        "a structure to compare the targets' motion with (the heart)",
                    )
                    .optional(),
                    PortSpec::one(
                        "Group",
                        GROUPISH,
                        "the 4D group; taken from the targets when left free",
                    )
                    .optional(),
                ];
                P
            }
            Kind::ExportDicom => {
                const P: &[PortSpec] = &[PortSpec::one(
                    "Data",
                    DATA,
                    "the studies to write: a study, or anything made on one",
                )
                .many()];
                P
            }
            Kind::SaveReport => {
                const P: &[PortSpec] =
                    &[PortSpec::one("Reports", &[T::Report], "what to write").many()];
                P
            }
        }
    }

    pub fn outputs(self) -> &'static [OutSpec] {
        match self {
            Kind::LoadFolder => {
                const P: &[OutSpec] = &[out("Study", T::Study, "everything in the folder")];
                P
            }
            Kind::LoadFolders => {
                const P: &[OutSpec] = &[out("Study", T::Study, "one subfolder per run")];
                P
            }
            Kind::LoadFromArchive => {
                const P: &[OutSpec] = &[out("Study", T::Study, "the study taken out")];
                P
            }
            Kind::Anonymize => {
                const P: &[OutSpec] = &[out("Study", T::Study, "the anonymized copy")];
                P
            }
            Kind::SegVolText => {
                const P: &[OutSpec] = &[
                    out("Structures", T::Structures, "what the prompts found"),
                    out("Report", T::Report, "volumes per image"),
                ];
                P
            }
            Kind::Combine => {
                const P: &[OutSpec] = &[
                    out("Result", T::Structures, "the combined structure"),
                    out("Report", T::Report, "its volume per image"),
                ];
                P
            }
            Kind::Rename => {
                const P: &[OutSpec] = &[out(
                    "Structures",
                    T::Structures,
                    "the structures under their new names (none after a delete)",
                )];
                P
            }
            Kind::Transfer => {
                const P: &[OutSpec] = &[
                    out("Placed", T::Structures, "the structure where it landed"),
                    out("Report", T::Report, "offsets and volumes"),
                ];
                P
            }
            Kind::CopyToPhases => {
                const P: &[OutSpec] = &[out(
                    "Copied",
                    T::Structures,
                    "the structures on every phase",
                )];
                P
            }
            Kind::Dvh => {
                const P: &[OutSpec] = &[out("Report", T::Report, "metrics and constraints")];
                P
            }
            Kind::DoseMetrics => {
                const P: &[OutSpec] = &[out("Report", T::Report, "one row per structure")];
                P
            }
            Kind::Drr => {
                const P: &[OutSpec] = &[out("Report", T::Report, "the images written")];
                P
            }
            Kind::ArchiveImport => &[],
            Kind::SelectImage => {
                const P: &[OutSpec] = &[out("Image", T::Image, "the image series found")];
                P
            }
            Kind::SelectGroup => {
                const P: &[OutSpec] = &[out("Group", T::Group, "the 4D group found")];
                P
            }
            Kind::SelectStructures => {
                const P: &[OutSpec] = &[out("Structures", T::Structures, "the structures found")];
                P
            }
            Kind::AutoSegment => {
                const P: &[OutSpec] = &[
                    out(
                        "Organs",
                        T::Structures,
                        "the organs kept, under their new names",
                    ),
                    out("Report", T::Report, "organ volumes per image"),
                ];
                P
            }
            Kind::BodyContour => {
                const P: &[OutSpec] = &[
                    out("Body", T::Structures, "the body outline"),
                    out("Report", T::Report, "the outline's volume per image"),
                ];
                P
            }
            Kind::Register => {
                const P: &[OutSpec] = &[
                    out("Registration", T::Registration, "fixed onto moving"),
                    out("Report", T::Report, "what the registration did"),
                ];
                P
            }
            Kind::Propagate => {
                const P: &[OutSpec] = &[
                    out("Structures", T::Structures, "the structures as they landed"),
                    out("Report", T::Report, "volumes before and after"),
                ];
                P
            }
            Kind::PropagateToGroup => {
                const P: &[OutSpec] = &[
                    out("Landed", T::Structures, "the structures on every phase"),
                    out("Registrations", T::Registration, "one per phase"),
                    out(
                        "Report",
                        T::Report,
                        "registration, anchor check and volumes per phase",
                    ),
                ];
                P
            }
            Kind::Motion => {
                const P: &[OutSpec] = &[
                    out("ITV", T::Structures, "the ITVs, on the reference phase"),
                    out("Report", T::Report, "tracks, amplitudes, correlations, QA"),
                ];
                P
            }
            Kind::ExportDicom | Kind::SaveReport => &[],
        }
    }

    /// A node of this kind with its default parameters.
    pub fn default_op(self) -> Op {
        match self {
            Kind::LoadFolder => Op::LoadFolder(Default::default()),
            Kind::LoadFolders => Op::LoadFolders(Default::default()),
            Kind::LoadFromArchive => Op::LoadFromArchive(Default::default()),
            Kind::Anonymize => Op::Anonymize(Default::default()),
            Kind::SegVolText => Op::SegVolText(Default::default()),
            Kind::Combine => Op::Combine(Default::default()),
            Kind::Rename => Op::Rename(Default::default()),
            Kind::Transfer => Op::Transfer(Default::default()),
            Kind::CopyToPhases => Op::CopyToPhases(Default::default()),
            Kind::Dvh => Op::Dvh(Default::default()),
            Kind::DoseMetrics => Op::DoseMetrics(Default::default()),
            Kind::ArchiveImport => Op::ArchiveImport(Default::default()),
            Kind::Drr => Op::Drr(Default::default()),
            Kind::SelectImage => Op::SelectImage(Default::default()),
            Kind::SelectGroup => Op::SelectGroup(Default::default()),
            Kind::SelectStructures => Op::SelectStructures(Default::default()),
            Kind::AutoSegment => Op::AutoSegment(Default::default()),
            Kind::BodyContour => Op::BodyContour(Default::default()),
            Kind::Register => Op::Register(Default::default()),
            Kind::Propagate => Op::Propagate(Default::default()),
            Kind::PropagateToGroup => Op::PropagateToGroup(Default::default()),
            Kind::Motion => Op::Motion(Default::default()),
            Kind::ExportDicom => Op::ExportDicom(Default::default()),
            Kind::SaveReport => Op::SaveReport(Default::default()),
        }
    }

    /// The kinds of one palette group, in palette order.
    pub fn of(category: Category) -> impl Iterator<Item = Kind> {
        Kind::ALL
            .into_iter()
            .filter(move |k| k.info().category == category)
    }
}

/// A node's kind and parameters: `kind` + `params` in the file.
// A node's parameters are read and written, never moved in a loop: the size
// of the largest block does not matter, and a box around it would only make
// every `Op::Combine(p)` in the program read worse.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "params", rename_all = "snake_case")]
pub enum Op {
    LoadFolder(LoadFolder),
    LoadFolders(LoadFolders),
    LoadFromArchive(LoadFromArchive),
    Anonymize(Anonymize),
    SegVolText(SegVolText),
    Combine(Combine),
    Rename(Rename),
    Transfer(Transfer),
    CopyToPhases(CopyToPhases),
    Dvh(Dvh),
    DoseMetrics(DoseMetrics),
    ArchiveImport(ArchiveImport),
    Drr(Drr),
    SelectImage(SelectImage),
    SelectGroup(SelectGroup),
    SelectStructures(SelectStructures),
    AutoSegment(AutoSegment),
    BodyContour(BodyContour),
    Register(Register),
    Propagate(Propagate),
    PropagateToGroup(PropagateToGroup),
    Motion(Motion),
    ExportDicom(ExportDicom),
    SaveReport(SaveReport),
}

impl Op {
    pub fn kind(&self) -> Kind {
        match self {
            Op::LoadFolder(_) => Kind::LoadFolder,
            Op::LoadFolders(_) => Kind::LoadFolders,
            Op::LoadFromArchive(_) => Kind::LoadFromArchive,
            Op::Anonymize(_) => Kind::Anonymize,
            Op::SegVolText(_) => Kind::SegVolText,
            Op::Combine(_) => Kind::Combine,
            Op::Rename(_) => Kind::Rename,
            Op::Transfer(_) => Kind::Transfer,
            Op::CopyToPhases(_) => Kind::CopyToPhases,
            Op::Dvh(_) => Kind::Dvh,
            Op::DoseMetrics(_) => Kind::DoseMetrics,
            Op::ArchiveImport(_) => Kind::ArchiveImport,
            Op::Drr(_) => Kind::Drr,
            Op::SelectImage(_) => Kind::SelectImage,
            Op::SelectGroup(_) => Kind::SelectGroup,
            Op::SelectStructures(_) => Kind::SelectStructures,
            Op::AutoSegment(_) => Kind::AutoSegment,
            Op::BodyContour(_) => Kind::BodyContour,
            Op::Register(_) => Kind::Register,
            Op::Propagate(_) => Kind::Propagate,
            Op::PropagateToGroup(_) => Kind::PropagateToGroup,
            Op::Motion(_) => Kind::Motion,
            Op::ExportDicom(_) => Kind::ExportDicom,
            Op::SaveReport(_) => Kind::SaveReport,
        }
    }

    pub fn inputs(&self) -> &'static [PortSpec] {
        self.kind().inputs()
    }

    pub fn outputs(&self) -> &'static [OutSpec] {
        self.kind().outputs()
    }

    /// Up to three short lines for the node's body on the canvas: what it
    /// is set to do, so a workflow reads without opening every node.
    pub fn summary(&self) -> Vec<String> {
        let quoted = |s: &str| format!("'{}'", s.trim());
        match self {
            Op::LoadFolder(p) => {
                let path = p.path.trim().replace('\\', "/");
                let tail: Vec<&str> = path.rsplit('/').filter(|s| !s.is_empty()).take(2).collect();
                let shown = if path.is_empty() {
                    "no folder yet".to_string()
                } else {
                    tail.into_iter().rev().collect::<Vec<_>>().join("/")
                };
                let mut v = vec![shown];
                if p.workspace != Workspace::Auto {
                    v.push(format!("shown in workspace {}", p.workspace.label()));
                }
                v
            }
            Op::SelectImage(p) => {
                let modality = if p.modality.trim().is_empty() {
                    "any modality".to_string()
                } else {
                    p.modality.trim().to_uppercase()
                };
                let mut v = vec![format!("{modality}, {}", p.pick.label())];
                if !p.description.trim().is_empty() {
                    v.push(format!("description has {}", quoted(&p.description)));
                }
                if p.outside_4d {
                    v.push("not a 4D phase".into());
                }
                v
            }
            Op::SelectGroup(p) => {
                let mut v = vec![if p.name.trim().is_empty() {
                    "the first 4D group".to_string()
                } else {
                    format!("name has {}", quoted(&p.name))
                }];
                if p.if_none == IfNoGroup::GroupAll {
                    v.push(format!("else group every {} series", p.modality.trim()));
                }
                v
            }
            Op::SelectStructures(p) => vec![
                if p.names.trim().is_empty() {
                    "no names yet".into()
                } else {
                    p.names.trim().to_string()
                },
                if p.first_only {
                    "the first match".into()
                } else {
                    "every match".into()
                },
            ],
            Op::AutoSegment(p) => {
                let organs = if p.organs.is_empty() {
                    "every organ found".to_string()
                } else {
                    p.organs
                        .iter()
                        .map(|o| {
                            if o.name.trim().is_empty() || o.name.trim() == o.organ.trim() {
                                o.organ.trim().to_string()
                            } else {
                                format!("{} as {}", o.organ.trim(), quoted(&o.name))
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(", ")
                };
                vec![
                    organs,
                    format!("{}, {}", p.variant.label(), p.output.label()),
                ]
            }
            Op::BodyContour(p) => vec![
                quoted(&p.name),
                format!("{}, {}", p.method.label(), p.output.label()),
            ],
            Op::Register(p) => vec![p.method.label().to_string(), p.init.label().to_string()],
            Op::Propagate(p) => vec![format!("into the {}", p.landing.label())],
            Op::PropagateToGroup(p) => vec![
                format!(
                    "{}{}",
                    if p.rigid_only {
                        "rigid"
                    } else {
                        "rigid + deformable"
                    },
                    format_args!(", anchored by {}", p.anchor_by.label())
                ),
                format!("into each phase's {}", p.landing.label()),
            ],
            Op::Motion(p) => {
                let mut models = Vec::new();
                if p.contoured {
                    models.push("as contoured");
                }
                if p.rigid {
                    models.push("rigid");
                }
                if p.deformable {
                    models.push("deformable");
                }
                let mut v = vec![models.join(", ")];
                if p.build_itv {
                    v.push(if p.itv_margin_mm > 0.0 {
                        format!("ITV + {:.1} mm", p.itv_margin_mm)
                    } else {
                        "ITV".into()
                    });
                }
                v
            }
            Op::ExportDicom(p) => vec![
                format!("to {}", quoted(&p.folder)),
                format!(
                    "{}, {}{}",
                    p.format.label(),
                    p.sets.label(),
                    if p.images { ", with images" } else { "" }
                ),
            ],
            Op::SaveReport(p) => vec![format!("to {}", quoted(&p.folder))],
            Op::LoadFolders(p) => {
                let path = p.path.trim().replace('\\', "/");
                let tail = path.rsplit('/').find(|s| !s.is_empty()).unwrap_or("");
                vec![
                    if path.is_empty() {
                        "no folder yet".to_string()
                    } else {
                        format!("every subfolder of {tail}")
                    },
                    if p.pattern.trim().is_empty() || p.pattern.trim() == "*" {
                        "one run per subfolder".to_string()
                    } else {
                        format!("named like {}", quoted(&p.pattern))
                    },
                ]
            }
            Op::LoadFromArchive(p) => vec![
                if p.patient.trim().is_empty() {
                    "no patient yet".to_string()
                } else {
                    format!("patient {}", quoted(&p.patient))
                },
                if p.study.trim().is_empty() {
                    "the newest study".to_string()
                } else {
                    format!("study like {}", quoted(&p.study))
                },
            ],
            Op::Anonymize(p) => vec![
                format!("to {}", quoted(&p.folder)),
                format!(
                    "{}{}",
                    if p.remap_uids {
                        "new UIDs"
                    } else {
                        "UIDs kept"
                    },
                    if p.remove_private {
                        ", private tags removed"
                    } else {
                        ""
                    }
                ),
            ],
            Op::SegVolText(p) => vec![
                if p.prompts.is_empty() {
                    "no prompt yet".to_string()
                } else {
                    p.prompts
                        .iter()
                        .map(|r| r.landed_name())
                        .collect::<Vec<_>>()
                        .join(", ")
                },
                p.output.label().to_string(),
            ],
            Op::Combine(p) => {
                let a = p.margin_a.describe();
                let r = p.margin.describe();
                let expr = match p.op {
                    CombineOp::Union => "A ∪ B",
                    CombineOp::Intersect => "A ∩ B",
                    CombineOp::Subtract => "A − B",
                };
                let mut v = vec![format!("{} = {expr}", quoted(&p.name))];
                if !a.is_empty() || !r.is_empty() {
                    v.push(
                        [
                            (!a.is_empty()).then(|| format!("A {a}")),
                            (!r.is_empty()).then(|| format!("result {r}")),
                        ]
                        .into_iter()
                        .flatten()
                        .collect::<Vec<_>>()
                        .join(", "),
                    );
                }
                v
            }
            Op::Rename(p) => {
                let rules: Vec<String> = p
                    .rules
                    .iter()
                    .map(|r| match p.action {
                        RenameAction::Rename => {
                            format!("{} to {}", r.from.trim(), quoted(&r.to))
                        }
                        RenameAction::Delete => r.from.trim().to_string(),
                    })
                    .collect();
                vec![
                    match p.action {
                        RenameAction::Rename => "rename".to_string(),
                        RenameAction::Delete => "delete".to_string(),
                    },
                    if rules.is_empty() {
                        "no names yet".into()
                    } else {
                        rules.join(", ")
                    },
                ]
            }
            Op::Transfer(p) => vec![
                "same offset from the reference".to_string(),
                format!("{}, {}", p.output.label(), p.names.label()),
            ],
            Op::CopyToPhases(p) => vec![format!("into each phase's {}", p.landing.label())],
            Op::Dvh(p) => {
                let mut v = vec![if p.metrics.trim().is_empty() {
                    "the default metrics".to_string()
                } else {
                    p.metrics.trim().to_string()
                }];
                let n = crate::dvh::parse_protocol(&p.protocol).len();
                if n > 0 {
                    v.push(format!("{n} constraints"));
                } else if !p.protocol_file.trim().is_empty() {
                    v.push("protocol from a file".into());
                }
                v
            }
            Op::DoseMetrics(p) => vec![
                p.dose_kind.label().to_string(),
                p.metrics.trim().to_string(),
            ],
            Op::ArchiveImport(p) => vec![
                p.source.label().to_string(),
                if p.archive.trim().is_empty() {
                    "the station's archive".to_string()
                } else {
                    quoted(&p.archive)
                },
            ],
            Op::Drr(p) => vec![
                if p.plan_beams {
                    "at the plan's beams".to_string()
                } else {
                    format!("gantry {}", p.angles.trim())
                },
                format!("to {}", quoted(&p.folder)),
            ],
        }
    }

    /// What is wrong with the parameters on their own, before anything
    /// runs. The input folder is not checked for existence here: the run
    /// dialog may point it elsewhere.
    pub fn param_problems(&self) -> Vec<String> {
        let mut v = Vec::new();
        match self {
            Op::LoadFolder(p) if p.path.trim().is_empty() => {
                v.push("no folder is given (set one here or in the run dialog)".into());
            }
            Op::SelectStructures(p) if split_names(&p.names).is_empty() => {
                v.push("no structure names are given".into());
            }
            Op::AutoSegment(p) => {
                for o in &p.organs {
                    if organ_label(&o.organ).is_none() {
                        v.push(format!(
                            "'{}' is not a TotalSegmentator class (heart, liver or aorta, say)",
                            o.organ.trim()
                        ));
                    }
                }
            }
            Op::BodyContour(p) if p.name.trim().is_empty() => {
                v.push("the outline needs a name".into());
            }
            Op::ExportDicom(p) if p.folder.trim().is_empty() => {
                v.push("no folder is given".into());
            }
            Op::SaveReport(p) if p.folder.trim().is_empty() => {
                v.push("no folder is given".into());
            }
            Op::LoadFolders(p) if p.path.trim().is_empty() => {
                v.push("no folder is given (set one here or in the run dialog)".into());
            }
            Op::LoadFromArchive(p) if p.patient.trim().is_empty() => {
                v.push("no patient is given".into());
            }
            Op::Anonymize(p) if p.folder.trim().is_empty() => {
                v.push("no folder is given".into());
            }
            Op::SegVolText(p) if p.prompts.iter().all(|r| r.structure.trim().is_empty()) => {
                v.push("no structure name to prompt with".into());
            }
            Op::Combine(p) if p.name.trim().is_empty() => {
                v.push("the result needs a name".into());
            }
            Op::Rename(p) => {
                if p.rules.iter().all(|r| r.from.trim().is_empty()) {
                    v.push("no names are given".into());
                }
                if p.action == RenameAction::Rename
                    && p.rules
                        .iter()
                        .any(|r| !r.from.trim().is_empty() && r.to.trim().is_empty())
                {
                    v.push("a rename needs the new name".into());
                }
            }
            Op::Dvh(p) => {
                for m in split_names(&p.metrics) {
                    if crate::dvh::Metric::parse(&m).is_none() {
                        v.push(format!("'{m}' is not a metric (D95%, D2cc, V20Gy, Dmean)"));
                    }
                }
            }
            Op::DoseMetrics(p) => {
                for m in split_names(&p.metrics) {
                    if crate::dvh::Metric::parse(&m).is_none() {
                        v.push(format!("'{m}' is not a metric (D95%, D2cc, V20Gy, Dmean)"));
                    }
                }
            }
            Op::Drr(p) => {
                if !p.plan_beams && parse_angles(&p.angles).is_none() {
                    v.push("the angles are numbers separated by commas: 0, 90".into());
                }
                if p.folder.trim().is_empty() {
                    v.push("no folder is given".into());
                }
            }
            _ => {}
        }
        v
    }
}

/// The TotalSegmentator class a name means (1-based global label), case
/// and spaces forgiven: `Heart`, `lung upper lobe left`.
pub fn organ_label(name: &str) -> Option<u8> {
    let want = name.trim().to_lowercase().replace([' ', '-'], "_");
    TOTAL_CLASS_NAMES
        .iter()
        .position(|n| *n == want)
        .map(|i| (i + 1) as u8)
}

/// Names split on commas, semicolons and new lines, trimmed, empties dropped.
pub fn split_names(s: &str) -> Vec<String> {
    s.split([',', ';', '\n'])
        .map(str::trim)
        .filter(|x| !x.is_empty())
        .map(str::to_string)
        .collect()
}

/// Gantry angles written as `0, 90, 180`; `None` when one is not a number.
pub fn parse_angles(s: &str) -> Option<Vec<f64>> {
    let v: Option<Vec<f64>> = split_names(s)
        .iter()
        .map(|a| a.trim_end_matches('°').trim().parse::<f64>().ok())
        .collect();
    v.filter(|v| !v.is_empty())
}

/// Does `name` match `pattern`? Case-insensitive; `*` stands for any run
/// of characters and `?` for one.
pub fn name_matches(pattern: &str, name: &str) -> bool {
    let p: Vec<char> = pattern.trim().to_lowercase().chars().collect();
    let s: Vec<char> = name.trim().to_lowercase().chars().collect();
    // Iterative wildcard match with one backtrack point.
    let (mut i, mut j) = (0usize, 0usize);
    let (mut star, mut mark) = (None::<usize>, 0usize);
    while j < s.len() {
        if i < p.len() && (p[i] == '?' || p[i] == s[j]) {
            i += 1;
            j += 1;
        } else if i < p.len() && p[i] == '*' {
            star = Some(i);
            mark = j;
            i += 1;
        } else if let Some(st) = star {
            i = st + 1;
            mark += 1;
            j = mark;
        } else {
            return false;
        }
    }
    while i < p.len() && p[i] == '*' {
        i += 1;
    }
    i == p.len()
}

// ---- the parameter blocks ----------------------------------------------

/// Which workspace a study is shown in when a run shows its steps.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Workspace {
    /// The next free letter, in the order the folders are read.
    #[default]
    Auto,
    A,
    B,
    C,
    D,
}

impl Workspace {
    pub const ALL: [Workspace; 5] = [
        Workspace::Auto,
        Workspace::A,
        Workspace::B,
        Workspace::C,
        Workspace::D,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Workspace::Auto => "next free",
            Workspace::A => "A",
            Workspace::B => "B",
            Workspace::C => "C",
            Workspace::D => "D",
        }
    }

    /// The slot index, or `None` for automatic.
    pub fn slot(self) -> Option<usize> {
        match self {
            Workspace::Auto => None,
            Workspace::A => Some(0),
            Workspace::B => Some(1),
            Workspace::C => Some(2),
            Workspace::D => Some(3),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LoadFolder {
    /// The folder, read with its subfolders.
    pub path: String,
    /// Where a run that shows its steps puts this study.
    pub workspace: Workspace,
}

/// Which of the matching series an [`SelectImage`] takes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImagePick {
    /// The one with the most slices.
    #[default]
    Largest,
    First,
    Last,
}

impl ImagePick {
    pub const ALL: [ImagePick; 3] = [ImagePick::Largest, ImagePick::First, ImagePick::Last];

    pub fn label(self) -> &'static str {
        match self {
            ImagePick::Largest => "the largest",
            ImagePick::First => "the first",
            ImagePick::Last => "the last",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SelectImage {
    /// CT, MR, PT; empty takes any.
    pub modality: String,
    /// Words the series description must contain (case-insensitive); empty
    /// takes any.
    pub description: String,
    pub pick: ImagePick,
    /// Leave out series that are phases (or AVG / MIP) of a 4D group - the
    /// planning CT beside a 4DCT, not one of its phases.
    pub outside_4d: bool,
}

impl Default for SelectImage {
    fn default() -> Self {
        SelectImage {
            modality: "CT".into(),
            description: String::new(),
            pick: ImagePick::Largest,
            outside_4d: false,
        }
    }
}

/// What [`SelectGroup`] does when the study has no 4D group.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IfNoGroup {
    /// Stop the run with a message.
    #[default]
    Fail,
    /// Group every image series of the modality, ordered by phase percent,
    /// temporal position, series number.
    GroupAll,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SelectGroup {
    /// Words the group's name must contain; empty takes the first group.
    pub name: String,
    pub if_none: IfNoGroup,
    /// The modality grouped by `group_all`.
    pub modality: String,
}

impl Default for SelectGroup {
    fn default() -> Self {
        SelectGroup {
            name: String::new(),
            if_none: IfNoGroup::GroupAll,
            modality: "CT".into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SelectStructures {
    /// Names or patterns, separated by commas: `target*, GTV*`.
    pub names: String,
    /// Take only the first structure found (patterns tried in order).
    pub first_only: bool,
    /// Stop the run when nothing matches.
    pub required: bool,
}

impl Default for SelectStructures {
    fn default() -> Self {
        SelectStructures {
            names: String::new(),
            first_only: false,
            required: true,
        }
    }
}

/// One organ to keep, and the name it is filed under.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct OrganRule {
    /// The TotalSegmentator class: `heart`, `aorta`, `lung_upper_lobe_left`.
    pub organ: String,
    /// What it is called in the study; empty keeps the class name.
    pub name: String,
}

impl OrganRule {
    /// The name it lands under.
    pub fn landed_name(&self) -> String {
        if self.name.trim().is_empty() {
            self.organ.trim().to_string()
        } else {
            self.name.trim().to_string()
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AutoSegment {
    pub variant: AutosegVariant,
    /// The organs kept. Empty keeps every organ found.
    pub organs: Vec<OrganRule>,
    pub output: OutputKind,
    pub set: SetChoice,
    /// The label of a new structure set or segmentation series.
    pub set_label: String,
    /// What happens when an organ's name is taken in the set it goes into.
    pub names: NameClash,
    pub device: Device,
}

impl Default for AutoSegment {
    fn default() -> Self {
        AutoSegment {
            variant: AutosegVariant::Fast,
            organs: vec![OrganRule {
                organ: "heart".into(),
                name: String::new(),
            }],
            output: OutputKind::Structures,
            set: SetChoice::Own,
            set_label: "Auto-segmentation".into(),
            names: NameClash::Counter,
            device: Device::Auto,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BodyContour {
    pub method: BodyMethod,
    pub name: String,
    pub output: OutputKind,
    pub set: SetChoice,
    pub set_label: String,
    pub names: NameClash,
    pub device: Device,
}

impl Default for BodyContour {
    fn default() -> Self {
        BodyContour {
            method: BodyMethod::Classical,
            name: "BODY".into(),
            output: OutputKind::Structures,
            set: SetChoice::Own,
            set_label: "Body contour".into(),
            names: NameClash::Counter,
            device: Device::Auto,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Register {
    pub method: RegMethodChoice,
    pub init: RegInit,
    pub effort: Effort,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Propagate {
    pub landing: Landing,
    /// Appended to each landed structure's name; empty keeps the names.
    pub suffix: String,
    /// What happens when a landed name is taken on the destination.
    pub names: NameClash,
    pub finish: FinishParams,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PropagateToGroup {
    /// The deformable stage (the whole registration of an unanchored run).
    pub method: DeformMethod,
    pub anchor_by: AnchorBy,
    /// Dilation of the anchor that bounds the registration, mm.
    pub anchor_margin_mm: f64,
    /// Stop an anchored run after its rigid stage.
    pub rigid_only: bool,
    /// The name the carried anchor lands under beside each phase's own
    /// contour; empty is `<anchor>_prop`.
    pub anchor_landed_as: String,
    pub landing: Landing,
    /// What happens when a landed name is taken on a phase - a target the
    /// phases were contoured with.
    pub names: NameClash,
    pub finish: FinishParams,
    pub effort: Effort,
}

impl Default for PropagateToGroup {
    fn default() -> Self {
        PropagateToGroup {
            method: DeformMethod::ElastixBspline,
            anchor_by: AnchorBy::Contours,
            anchor_margin_mm: 10.0,
            rigid_only: false,
            anchor_landed_as: String::new(),
            landing: Landing::StructureSet,
            names: NameClash::Counter,
            finish: FinishParams::default(),
            effort: Effort::default(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Motion {
    /// The reference phase's label (`0%`); empty is the group's default.
    pub reference_phase: String,
    /// The phases taking part, by label, separated by commas; empty is all.
    pub phases: String,
    /// Read each phase's own contour of the targets (no registration).
    pub contoured: bool,
    pub rigid: bool,
    /// The rigid model fitted around each structure, mm; 0 is one global
    /// rigid body per phase.
    pub local_rigid_margin_mm: f64,
    pub deformable: bool,
    pub build_itv: bool,
    pub itv_margin_mm: f64,
    /// Where the ITVs are filed on the reference phase.
    pub itv_landing: Landing,
    /// What happens when an ITV's name is taken there.
    pub names: NameClash,
    /// Also file every propagated per-phase mask.
    pub keep_phase_segs: bool,
    pub effort: Effort,
}

impl Default for Motion {
    fn default() -> Self {
        Motion {
            reference_phase: String::new(),
            phases: String::new(),
            contoured: true,
            rigid: true,
            local_rigid_margin_mm: 15.0,
            deformable: true,
            build_itv: true,
            itv_margin_mm: 0.0,
            itv_landing: Landing::StructureSet,
            names: NameClash::Counter,
            keep_phase_segs: false,
            effort: Effort::default(),
        }
    }
}

/// How structures are written.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructFormatChoice {
    #[default]
    Rtstruct,
    Seg,
}

impl StructFormatChoice {
    pub const ALL: [StructFormatChoice; 2] =
        [StructFormatChoice::Rtstruct, StructFormatChoice::Seg];

    pub fn label(self) -> &'static str {
        match self {
            StructFormatChoice::Rtstruct => "RTSTRUCT",
            StructFormatChoice::Seg => "SEG",
        }
    }
}

/// Which structure sets and segmentation series are written.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WhichSets {
    /// Only those this run made or added to.
    #[default]
    Changed,
    All,
}

impl WhichSets {
    pub const ALL: [WhichSets; 2] = [WhichSets::Changed, WhichSets::All];

    pub fn label(self) -> &'static str {
        match self {
            WhichSets::Changed => "what the run changed",
            WhichSets::All => "every set",
        }
    }
}

/// The identifiers the written objects carry.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UidChoice {
    /// The study's own UIDs: the export is the same study.
    #[default]
    Keep,
    /// Fresh UIDs throughout.
    New,
}

impl UidChoice {
    pub const ALL: [UidChoice; 2] = [UidChoice::Keep, UidChoice::New];

    pub fn label(self) -> &'static str {
        match self {
            UidChoice::Keep => "keep the UIDs",
            UidChoice::New => "new UIDs",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ExportDicom {
    /// Relative to the run folder, or absolute. `{input}` is the title of
    /// the folder node the study was read by.
    pub folder: String,
    pub format: StructFormatChoice,
    pub sets: WhichSets,
    /// Write the image series too.
    pub images: bool,
    /// Write doses and plans too.
    pub doses_and_plans: bool,
    pub uids: UidChoice,
}

impl Default for ExportDicom {
    fn default() -> Self {
        ExportDicom {
            folder: "{input}".into(),
            format: StructFormatChoice::Rtstruct,
            sets: WhichSets::Changed,
            images: false,
            doses_and_plans: false,
            uids: UidChoice::Keep,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SaveReport {
    /// Relative to the run folder, or absolute.
    pub folder: String,
    /// One CSV per table.
    pub csv: bool,
    /// One readable summary (Markdown).
    pub text: bool,
}

impl Default for SaveReport {
    fn default() -> Self {
        SaveReport {
            folder: "reports".into(),
            csv: true,
            text: true,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LoadFolders {
    /// The folder whose subfolders are the cases.
    pub path: String,
    /// Which subfolders: `*` is all, `P*` those starting with P.
    pub pattern: String,
    pub workspace: Workspace,
}

impl Default for LoadFolders {
    fn default() -> Self {
        LoadFolders {
            path: String::new(),
            pattern: "*".into(),
            workspace: Workspace::Auto,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LoadFromArchive {
    /// The archive's folder; empty is the station's (Tools > PACS).
    pub archive: String,
    /// The patient's ID or name, or a pattern with `*`.
    pub patient: String,
    /// Words of the study's description, or its date (YYYYMMDD); empty
    /// takes the newest.
    pub study: String,
    pub workspace: Workspace,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Anonymize {
    /// Inside the run folder, or absolute; `{input}` is the study's title.
    pub folder: String,
    pub remove_private: bool,
    pub remap_uids: bool,
    /// Clear the study and series descriptions too.
    pub clear_descriptions: bool,
}

impl Default for Anonymize {
    fn default() -> Self {
        Anonymize {
            folder: "anonymized/{input}".into(),
            remove_private: true,
            remap_uids: true,
            clear_descriptions: false,
        }
    }
}

/// One text prompt, and the name its result is filed under.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PromptRule {
    /// What the prompt says: a structure name (liver, pancreas).
    pub structure: String,
    /// What it is called in the study; empty keeps the prompt.
    pub name: String,
}

impl PromptRule {
    pub fn landed_name(&self) -> String {
        if self.name.trim().is_empty() {
            self.structure.trim().to_string()
        } else {
            self.name.trim().to_string()
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SegVolText {
    pub prompts: Vec<PromptRule>,
    /// The sliding-window refinement pass (slower, sharper).
    pub refine: bool,
    /// Probability threshold of the mask.
    pub threshold: f32,
    pub output: OutputKind,
    pub set: SetChoice,
    pub set_label: String,
    pub names: NameClash,
    pub device: Device,
}

impl Default for SegVolText {
    fn default() -> Self {
        SegVolText {
            prompts: vec![PromptRule {
                structure: "liver".into(),
                name: String::new(),
            }],
            refine: true,
            threshold: 0.5,
            output: OutputKind::Structures,
            set: SetChoice::Own,
            set_label: "SegVol".into(),
            names: NameClash::Counter,
            device: Device::Auto,
        }
    }
}

/// How the two operands of *Combine structures* are joined.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CombineOp {
    /// A ∪ B, or A alone when B is free.
    #[default]
    Union,
    Intersect,
    /// A minus B.
    Subtract,
}

impl CombineOp {
    pub const ALL: [CombineOp; 3] = [CombineOp::Union, CombineOp::Intersect, CombineOp::Subtract];

    pub fn label(self) -> &'static str {
        match self {
            CombineOp::Union => "A ∪ B (or A alone)",
            CombineOp::Intersect => "A ∩ B",
            CombineOp::Subtract => "A − B",
        }
    }

    pub fn bool_op(self) -> crate::structops::BoolOp {
        match self {
            CombineOp::Union => crate::structops::BoolOp::Union,
            CombineOp::Intersect => crate::structops::BoolOp::Intersect,
            CombineOp::Subtract => crate::structops::BoolOp::Subtract,
        }
    }
}

/// A margin in millimetres: one number for every direction, or one per
/// patient direction overriding it. Negative shrinks.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MarginMm {
    pub uniform_mm: f64,
    pub right_mm: Option<f64>,
    pub left_mm: Option<f64>,
    pub anterior_mm: Option<f64>,
    pub posterior_mm: Option<f64>,
    pub superior_mm: Option<f64>,
    pub inferior_mm: Option<f64>,
}

impl MarginMm {
    pub fn margin(&self) -> crate::structops::Margin {
        let u = self.uniform_mm;
        crate::structops::Margin {
            right: self.right_mm.unwrap_or(u),
            left: self.left_mm.unwrap_or(u),
            anterior: self.anterior_mm.unwrap_or(u),
            posterior: self.posterior_mm.unwrap_or(u),
            superior: self.superior_mm.unwrap_or(u),
            inferior: self.inferior_mm.unwrap_or(u),
        }
    }

    /// `+5 mm`, or the structops description of a directional margin;
    /// empty for none.
    pub fn describe(&self) -> String {
        let m = self.margin();
        if m.is_none() {
            String::new()
        } else if m.is_uniform() {
            format!("{:+} mm", m.right)
        } else {
            m.describe()
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Combine {
    pub op: CombineOp,
    /// Applied to A (and to B) before they are joined.
    pub margin_a: MarginMm,
    pub margin_b: MarginMm,
    /// Applied to the result.
    pub margin: MarginMm,
    pub fill_holes: bool,
    pub close_mm: f64,
    pub keep_largest: bool,
    pub min_volume_cm3: f64,
    pub name: String,
    pub output: OutputKind,
    pub set: SetChoice,
    pub set_label: String,
    pub names: NameClash,
}

impl Default for Combine {
    fn default() -> Self {
        Combine {
            op: CombineOp::Union,
            margin_a: MarginMm::default(),
            margin_b: MarginMm::default(),
            margin: MarginMm {
                uniform_mm: 5.0,
                ..MarginMm::default()
            },
            fill_holes: false,
            close_mm: 0.0,
            keep_largest: false,
            min_volume_cm3: 0.0,
            name: "PTV".into(),
            output: OutputKind::Structures,
            set: SetChoice::Own,
            set_label: "Combined".into(),
            names: NameClash::Counter,
        }
    }
}

/// What *Rename or delete* does to the structures its rules name.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RenameAction {
    #[default]
    Rename,
    Delete,
}

impl RenameAction {
    pub const ALL: [RenameAction; 2] = [RenameAction::Rename, RenameAction::Delete];

    pub fn label(self) -> &'static str {
        match self {
            RenameAction::Rename => "rename",
            RenameAction::Delete => "delete",
        }
    }
}

/// One rule: the structures named like `from` get the name `to`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RenameRule {
    /// A name, or a pattern with `*` and `?`.
    pub from: String,
    pub to: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Rename {
    pub action: RenameAction,
    pub rules: Vec<RenameRule>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Transfer {
    /// What it lands as; empty keeps the target's name.
    pub name: String,
    pub output: OutputKind,
    pub set: SetChoice,
    pub set_label: String,
    pub names: NameClash,
}

impl Default for Transfer {
    fn default() -> Self {
        Transfer {
            name: String::new(),
            output: OutputKind::Structures,
            set: SetChoice::Own,
            set_label: "Transferred".into(),
            names: NameClash::Counter,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CopyToPhases {
    pub landing: Landing,
    pub names: NameClash,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Dvh {
    /// Words of the dose's label; empty takes the study's first dose.
    pub dose: String,
    /// Metrics separated by commas: D95%, D2cc, V20Gy, V20Gy[cc], Dmean.
    pub metrics: String,
    /// Constraints, one per line: `Heart Dmean < 5`, `PTV D95% >= 25`.
    pub protocol: String,
    /// A protocol file (the DVH window's format), read when the run starts.
    pub protocol_file: String,
    /// Bin width in dose units; 0 derives it from the dose maximum.
    pub bin_width: f64,
    /// Put the cumulative curves in the report too.
    pub curves: bool,
}

impl Default for Dvh {
    fn default() -> Self {
        Dvh {
            dose: String::new(),
            metrics: "D95%, D2%, Dmean, Dmax".into(),
            protocol: String::new(),
            protocol_file: String::new(),
            bin_width: 0.0,
            curves: false,
        }
    }
}

/// Which dose *Dose estimation* measures against.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DoseKindChoice {
    /// The first dose, whatever its type.
    #[default]
    Any,
    Physical,
    /// RBE-weighted.
    Effective,
}

impl DoseKindChoice {
    pub const ALL: [DoseKindChoice; 3] = [
        DoseKindChoice::Any,
        DoseKindChoice::Physical,
        DoseKindChoice::Effective,
    ];

    pub fn label(self) -> &'static str {
        match self {
            DoseKindChoice::Any => "the study's dose",
            DoseKindChoice::Physical => "the physical dose",
            DoseKindChoice::Effective => "the RBE-weighted dose",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DoseMetrics {
    pub dose_kind: DoseKindChoice,
    /// Words of the dose's label; empty takes the first of that type.
    pub dose: String,
    /// The columns, separated by commas.
    pub metrics: String,
}

impl Default for DoseMetrics {
    fn default() -> Self {
        DoseMetrics {
            dose_kind: DoseKindChoice::Any,
            dose: String::new(),
            metrics: "Volume, Dmean, Dmin, Dmax, D95%, D2%".into(),
        }
    }
}

/// What *File in the archive* files.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImportSource {
    /// What an Export DICOM step of this run wrote of the study; the folder
    /// it was read from when none did.
    #[default]
    Exported,
    /// The folder the study was read from.
    Read,
}

impl ImportSource {
    pub const ALL: [ImportSource; 2] = [ImportSource::Exported, ImportSource::Read];

    pub fn label(self) -> &'static str {
        match self {
            ImportSource::Exported => "what the run exported",
            ImportSource::Read => "the folder it was read from",
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ArchiveImport {
    /// The archive's folder; empty is the station's.
    pub archive: String,
    pub source: ImportSource,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Drr {
    /// Gantry angles, degrees, separated by commas.
    pub angles: String,
    pub couch_deg: f64,
    /// At the beams of the study's plan instead (angles and isocentre).
    pub plan_beams: bool,
    /// Dark bone on a light background, as a radiograph looks.
    pub invert: bool,
    /// Image size, pixels (square).
    pub size_px: usize,
    /// Inside the run folder, or absolute; `{input}` is the study's title.
    pub folder: String,
    /// Also file each image as a planar image of the study.
    pub file_into_study: bool,
}

impl Default for Drr {
    fn default() -> Self {
        Drr {
            angles: "0, 90".into(),
            couch_deg: 0.0,
            plan_beams: false,
            invert: true,
            size_px: 512,
            folder: "drr/{input}".into(),
            file_into_study: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_kind_is_described_and_round_trips_its_defaults() {
        for k in Kind::ALL {
            let op = k.default_op();
            assert_eq!(op.kind(), k);
            assert!(!k.info().name.is_empty());
            let text = serde_json::to_string(&op).unwrap();
            let back: Op = serde_json::from_str(&text).unwrap();
            assert_eq!(back, op, "{text}");
            for spec in k.inputs() {
                assert!(!spec.accepts.is_empty(), "{:?} {}", k, spec.name);
            }
        }
        let n: usize = Category::ALL.iter().map(|c| Kind::of(*c).count()).sum();
        assert_eq!(n, Kind::ALL.len(), "every kind is in one palette group");
    }

    #[test]
    fn patterns_match_like_a_person_expects() {
        assert!(name_matches("target*", "target_volume"));
        assert!(name_matches("TARGET*", "target"));
        assert!(name_matches("*heart*", "Heart total"));
        assert!(name_matches("gtv?", "GTV1"));
        assert!(!name_matches("gtv?", "GTV12"));
        assert!(name_matches("heart total", "Heart Total"));
        assert!(!name_matches("heart", "heart total"));
        assert!(name_matches("*", "anything"));
        assert_eq!(
            split_names("target*, GTV*;\nPTV"),
            ["target*", "GTV*", "PTV"]
        );
    }

    #[test]
    fn organ_names_are_forgiving_and_checked() {
        assert_eq!(organ_label("heart"), Some(51));
        assert_eq!(
            organ_label("Lung upper lobe left"),
            organ_label("lung_upper_lobe_left")
        );
        assert!(organ_label("hearts").is_none());
        let bad = Op::AutoSegment(AutoSegment {
            organs: vec![OrganRule {
                organ: "hearts".into(),
                name: String::new(),
            }],
            ..AutoSegment::default()
        });
        assert_eq!(bad.param_problems().len(), 1);
    }
}
