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
    Register,
    FourD,
    Output,
}

impl Category {
    pub const ALL: [Category; 6] = [
        Category::Input,
        Category::Select,
        Category::Segment,
        Category::Register,
        Category::FourD,
        Category::Output,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Category::Input => "Input",
            Category::Select => "Find in the data",
            Category::Segment => "Segment",
            Category::Register => "Register and propagate",
            Category::FourD => "4D",
            Category::Output => "Output",
        }
    }

    /// Header fill of a node of this category.
    pub fn color(self) -> [u8; 3] {
        match self {
            Category::Input => [70, 85, 110],
            Category::Select => [40, 110, 115],
            Category::Segment => [55, 115, 60],
            Category::Register => [50, 80, 140],
            Category::FourD => [140, 95, 35],
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
    SelectImage,
    SelectGroup,
    SelectStructures,
    AutoSegment,
    BodyContour,
    Register,
    Propagate,
    PropagateToGroup,
    Motion,
    ExportDicom,
    SaveReport,
}

impl Kind {
    pub const ALL: [Kind; 12] = [
        Kind::LoadFolder,
        Kind::SelectImage,
        Kind::SelectGroup,
        Kind::SelectStructures,
        Kind::AutoSegment,
        Kind::BodyContour,
        Kind::Register,
        Kind::Propagate,
        Kind::PropagateToGroup,
        Kind::Motion,
        Kind::ExportDicom,
        Kind::SaveReport,
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
        }
    }

    pub fn inputs(self) -> &'static [PortSpec] {
        const STUDY: &[PortType] = &[T::Study];
        const IMAGE_OR_GROUP: &[PortType] = &[T::Image, T::Group];
        const ANYWHERE: &[PortType] = &[T::Study, T::Image, T::Group];
        const DATA: &[PortType] = &[T::Study, T::Image, T::Group, T::Structures];
        const GROUPISH: &[PortType] = &[T::Group, T::Structures];
        match self {
            Kind::LoadFolder => &[],
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
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "params", rename_all = "snake_case")]
pub enum Op {
    LoadFolder(LoadFolder),
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

/// TotalSegmentator's three models.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AutosegVariant {
    /// One 3 mm model, all 117 classes.
    #[default]
    Fast,
    /// The 1.5 mm sub-models; only those holding the organs asked for run.
    High,
    /// One 6 mm model.
    Preview,
}

impl AutosegVariant {
    pub const ALL: [AutosegVariant; 3] = [
        AutosegVariant::Fast,
        AutosegVariant::High,
        AutosegVariant::Preview,
    ];

    pub fn label(self) -> &'static str {
        match self {
            AutosegVariant::Fast => "3 mm",
            AutosegVariant::High => "1.5 mm",
            AutosegVariant::Preview => "6 mm",
        }
    }
}

/// Where segmentation results are filed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputKind {
    /// Contours in an RT structure set.
    #[default]
    Structures,
    /// Segments of a segmentation series.
    Segments,
    Both,
}

impl OutputKind {
    pub const ALL: [OutputKind; 3] = [
        OutputKind::Structures,
        OutputKind::Segments,
        OutputKind::Both,
    ];

    pub fn label(self) -> &'static str {
        match self {
            OutputKind::Structures => "RT structures",
            OutputKind::Segments => "segments",
            OutputKind::Both => "RT structures and segments",
        }
    }

    pub fn structures(self) -> bool {
        matches!(self, OutputKind::Structures | OutputKind::Both)
    }

    pub fn segments(self) -> bool {
        matches!(self, OutputKind::Segments | OutputKind::Both)
    }
}

/// Which structure set RT structures go into.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SetChoice {
    /// The image's own structure set - the one that references the series
    /// (each phase's own on a 4D group) - or a new one when it has none.
    #[default]
    Own,
    /// Always a new structure set.
    New,
}

impl SetChoice {
    pub const ALL: [SetChoice; 2] = [SetChoice::Own, SetChoice::New];

    pub fn label(self) -> &'static str {
        match self {
            SetChoice::Own => "the image's own set",
            SetChoice::New => "a new set",
        }
    }
}

/// Where the engines run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Device {
    #[default]
    Auto,
    Gpu,
    Cpu,
}

impl Device {
    pub const ALL: [Device; 3] = [Device::Auto, Device::Gpu, Device::Cpu];

    pub fn label(self) -> &'static str {
        match self {
            Device::Auto => "automatic",
            Device::Gpu => "GPU",
            Device::Cpu => "CPU",
        }
    }

    pub fn pref(self) -> crate::nn::device::DevicePref {
        use crate::nn::device::DevicePref;
        match self {
            Device::Auto => DevicePref::Auto,
            Device::Gpu => DevicePref::Gpu,
            Device::Cpu => DevicePref::Cpu,
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
            device: Device::Auto,
        }
    }
}

/// How the body outline is found.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BodyMethod {
    #[default]
    Classical,
    ModelAssisted,
}

impl BodyMethod {
    pub const ALL: [BodyMethod; 2] = [BodyMethod::Classical, BodyMethod::ModelAssisted];

    pub fn label(self) -> &'static str {
        match self {
            BodyMethod::Classical => "classical",
            BodyMethod::ModelAssisted => "model-assisted",
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
            device: Device::Auto,
        }
    }
}

/// The registration engines a workflow offers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegMethodChoice {
    #[default]
    ElastixRigid,
    ElastixBspline,
    PlastimatchBspline,
}

impl RegMethodChoice {
    pub const ALL: [RegMethodChoice; 3] = [
        RegMethodChoice::ElastixRigid,
        RegMethodChoice::ElastixBspline,
        RegMethodChoice::PlastimatchBspline,
    ];

    pub fn label(self) -> &'static str {
        match self {
            RegMethodChoice::ElastixRigid => "rigid (elastix)",
            RegMethodChoice::ElastixBspline => "rigid + B-spline (elastix)",
            RegMethodChoice::PlastimatchBspline => "B-spline (plastimatch)",
        }
    }

    pub fn method(self) -> crate::registration::RegMethod {
        use crate::registration::RegMethod;
        match self {
            RegMethodChoice::ElastixRigid => RegMethod::ElastixRigid,
            RegMethodChoice::ElastixBspline => RegMethod::ElastixBSpline,
            RegMethodChoice::PlastimatchBspline => RegMethod::PlastimatchBSpline,
        }
    }
}

/// Where a registration starts its search.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegInit {
    /// The identity when the images overlap, else their centres of gravity.
    #[default]
    Auto,
    Identity,
    CentresOfGravity,
}

impl RegInit {
    pub const ALL: [RegInit; 3] = [RegInit::Auto, RegInit::Identity, RegInit::CentresOfGravity];

    pub fn label(self) -> &'static str {
        match self {
            RegInit::Auto => "start: automatic",
            RegInit::Identity => "start: identity",
            RegInit::CentresOfGravity => "start: centres of gravity",
        }
    }

    pub fn init(self) -> crate::registration::Init {
        use crate::registration::Init;
        match self {
            RegInit::Auto => Init::Auto,
            RegInit::Identity => Init::Identity,
            RegInit::CentresOfGravity => Init::CenterOfGravity,
        }
    }
}

/// The optimiser's effort, shared by every node that registers.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Effort {
    /// Resolution levels of the pyramid.
    pub levels: usize,
    /// Iterations per level.
    pub iterations: usize,
    /// Random samples per iteration.
    pub samples: usize,
    /// B-spline control point spacing, mm.
    pub grid_spacing_mm: f64,
    /// Sample only fixed-image voxels above this value (HU).
    pub fixed_threshold: f32,
}

impl Default for Effort {
    fn default() -> Self {
        let d = crate::registration::RegParams::default();
        Effort {
            levels: d.levels,
            iterations: d.iterations,
            samples: d.samples,
            grid_spacing_mm: d.grid_spacing_mm,
            fixed_threshold: d.fixed_threshold,
        }
    }
}

impl Effort {
    /// Registration parameters with this effort and `method`.
    pub fn params(&self, method: crate::registration::RegMethod) -> crate::registration::RegParams {
        crate::registration::RegParams {
            method,
            levels: self.levels.clamp(1, 6),
            iterations: self.iterations.clamp(1, 5000),
            samples: self.samples.clamp(100, 200_000),
            grid_spacing_mm: self.grid_spacing_mm.clamp(4.0, 200.0),
            fixed_threshold: self.fixed_threshold,
            ..crate::registration::RegParams::default()
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

/// Where carried structures are filed on their destination image.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Landing {
    /// Contours in the destination image's own RT structure set.
    #[default]
    StructureSet,
    /// Segments of a segmentation series bound to the image.
    Segmentation,
}

impl Landing {
    pub const ALL: [Landing; 2] = [Landing::StructureSet, Landing::Segmentation];

    pub fn label(self) -> &'static str {
        match self {
            Landing::StructureSet => "structure set",
            Landing::Segmentation => "segmentation series",
        }
    }

    pub fn group_landing(self) -> crate::workflow::group::Landing {
        match self {
            Landing::StructureSet => crate::workflow::group::Landing::StructureSet,
            Landing::Segmentation => crate::workflow::group::Landing::Segmentation,
        }
    }
}

/// What is done to carried structures once they have landed.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FinishParams {
    /// Morphological closing radius, mm (0: none).
    pub close_mm: f64,
    /// Fill the interior slice by slice.
    pub fill: bool,
    /// Carry each structure as a rigid body.
    pub keep_shape: bool,
}

impl FinishParams {
    pub fn finish(&self) -> crate::propagate::Finish {
        crate::propagate::Finish {
            close_mm: self.close_mm.clamp(0.0, 50.0),
            fill: self.fill,
            keep_shape: self.keep_shape,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Propagate {
    pub landing: Landing,
    /// Appended to each landed structure's name; empty keeps the names.
    pub suffix: String,
    pub finish: FinishParams,
}

/// What an anchored run compares.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnchorBy {
    /// The anchor's surfaces (signed distance maps).
    #[default]
    Contours,
    /// The images inside the anchor's region.
    Intensity,
}

impl AnchorBy {
    pub const ALL: [AnchorBy; 2] = [AnchorBy::Contours, AnchorBy::Intensity];

    pub fn label(self) -> &'static str {
        match self {
            AnchorBy::Contours => "contours",
            AnchorBy::Intensity => "intensity",
        }
    }
}

/// The deformable engines a run onto a 4D group may use.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeformMethod {
    #[default]
    ElastixBspline,
    PlastimatchBspline,
}

impl DeformMethod {
    pub const ALL: [DeformMethod; 2] = [
        DeformMethod::ElastixBspline,
        DeformMethod::PlastimatchBspline,
    ];

    pub fn label(self) -> &'static str {
        match self {
            DeformMethod::ElastixBspline => "B-spline (elastix)",
            DeformMethod::PlastimatchBspline => "B-spline (plastimatch)",
        }
    }

    pub fn method(self) -> crate::registration::RegMethod {
        use crate::registration::RegMethod;
        match self {
            DeformMethod::ElastixBspline => RegMethod::ElastixBSpline,
            DeformMethod::PlastimatchBspline => RegMethod::PlastimatchBSpline,
        }
    }
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
