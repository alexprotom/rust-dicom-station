//! The parameters of the engines, defined once for every caller.
//!
//! A registration's method and effort, an auto-segmentation's model, where a
//! result is filed and what happens when its name is taken: the workflow
//! file stores these ([`crate::workflow::graph::catalog`] re-exports them),
//! the MCP tools read them from their arguments by the same names
//! (`from_name`), and each turns into the engine's own settings in one
//! place (`method`, `init`, `params`, `variant`, `pref`). A default or a
//! name changed here changes for all of them.

use serde::{Deserialize, Serialize};

/// Parse one of `all` by its file name (`snake_case`, the spelling the
/// workflow file and the MCP arguments use), naming the choices when it is
/// none of them.
fn by_name<T: Copy + Serialize>(all: &[T], name: &str, what: &str) -> anyhow::Result<T> {
    let want = name.trim().to_ascii_lowercase().replace([' ', '-'], "_");
    let names: Vec<String> = all
        .iter()
        .map(|v| {
            serde_json::to_value(v)
                .ok()
                .and_then(|j| j.as_str().map(str::to_string))
                .unwrap_or_default()
        })
        .collect();
    match names.iter().position(|n| *n == want) {
        Some(i) => Ok(all[i]),
        None => anyhow::bail!("{what} must be one of {} (got '{name}')", names.join(", ")),
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

// ---- names, for the callers that pass them as text (the MCP tools) --------

impl AutosegVariant {
    /// `fast`, `high` or `preview`.
    pub fn from_name(name: &str) -> anyhow::Result<AutosegVariant> {
        by_name(&AutosegVariant::ALL, name, "variant")
    }

    /// The engine's own variant.
    pub fn variant(self) -> crate::autoseg::Variant {
        match self {
            AutosegVariant::Fast => crate::autoseg::Variant::Fast3mm,
            AutosegVariant::High => crate::autoseg::Variant::HighRes15mm,
            AutosegVariant::Preview => crate::autoseg::Variant::Preview6mm,
        }
    }
}

impl BodyMethod {
    /// `classical` or `model_assisted`.
    pub fn from_name(name: &str) -> anyhow::Result<BodyMethod> {
        by_name(&BodyMethod::ALL, name, "method")
    }

    pub fn method(self) -> crate::bodymask::Method {
        match self {
            BodyMethod::Classical => crate::bodymask::Method::Classical,
            BodyMethod::ModelAssisted => crate::bodymask::Method::ModelAssisted,
        }
    }
}

impl RegMethodChoice {
    /// `elastix_rigid`, `elastix_bspline` or `plastimatch_bspline`.
    pub fn from_name(name: &str) -> anyhow::Result<RegMethodChoice> {
        by_name(&RegMethodChoice::ALL, name, "method")
    }
}

impl Landing {
    /// `structure_set` (or `rtstruct`) or `segmentation`.
    pub fn from_name(name: &str) -> anyhow::Result<Landing> {
        if name.trim().eq_ignore_ascii_case("rtstruct") {
            return Ok(Landing::StructureSet);
        }
        by_name(&Landing::ALL, name, "land")
    }
}

impl Device {
    /// `auto`, `gpu` or `cpu`.
    pub fn from_name(name: &str) -> anyhow::Result<Device> {
        by_name(&Device::ALL, name, "device")
    }
}

/// What happens when a structure is filed under a name the structure set
/// (or segmentation series) it goes into already holds - the target a phase
/// was contoured with, the heart an earlier run filed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NameClash {
    /// Take the next free `name (2)`. The same name on every phase of a 4D
    /// group, and the name every later step looks it up by.
    #[default]
    Counter,
    /// File it as `name_prop` (then `name_prop (2)`, and so on), beside the
    /// one that is there.
    Suffix,
    /// Replace the structure of that name.
    Replace,
}

impl NameClash {
    pub const ALL: [NameClash; 3] = [NameClash::Counter, NameClash::Suffix, NameClash::Replace];

    pub fn label(self) -> &'static str {
        match self {
            NameClash::Counter => "add a counter: name (2)",
            NameClash::Suffix => "file it as name_prop",
            NameClash::Replace => "replace the one there",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn choices_are_read_by_their_file_names() {
        assert_eq!(
            AutosegVariant::from_name("high").unwrap(),
            AutosegVariant::High
        );
        assert_eq!(
            BodyMethod::from_name("model_assisted").unwrap(),
            BodyMethod::ModelAssisted
        );
        assert_eq!(
            RegMethodChoice::from_name("plastimatch_bspline").unwrap(),
            RegMethodChoice::PlastimatchBspline
        );
        assert_eq!(
            Landing::from_name("rtstruct").unwrap(),
            Landing::StructureSet
        );
        assert_eq!(Device::from_name(" GPU ").unwrap(), Device::Gpu);
        let e = format!("{:#}", AutosegVariant::from_name("slow").unwrap_err());
        assert!(e.contains("fast, high, preview"), "{e}");
    }
}
