//! The automatic models - a fixed class list in, labels out - across every
//! family, behind one enum.

use anyhow::Result;
use std::path::Path;

use super::{Family, Licence, Modality};
use crate::autoseg::{self, AutosegResult, NnOptions, NnTask, Variant};
use crate::nn::device::DevicePref;
use crate::progress::Progress;
use crate::segresnet::{self, SegResModel};
use crate::unet2d::{self, Lungmask};
use crate::vista3d::{self, Vista3d};
use crate::volume::Volume;

/// One automatic segmentation model, whichever family runs it.
#[derive(Clone, Copy, Debug)]
pub enum AutoModel {
    /// An nnU-Net task: TotalSegmentator (CT, MR, the catalogue),
    /// MRSegmentator, the body outline.
    Nn(&'static NnTask),
    /// One of lungmask's 2-D U-Nets.
    Lungmask(Lungmask),
    /// MONAI's whole-body SegResNet or CT-FM's.
    SegRes(SegResModel),
    /// VISTA-3D in its automatic mode.
    Vista(Vista3d),
}

impl PartialEq for AutoModel {
    fn eq(&self, other: &Self) -> bool {
        self.key() == other.key()
    }
}

impl Eq for AutoModel {}

/// How to run a model.
#[derive(Clone, Debug, Default)]
pub struct RunOptions {
    pub device: DevicePref,
    /// For a model of several sub-models, which ones to run (`None`: all).
    pub parts: Option<Vec<bool>>,
}

impl AutoModel {
    /// Every model, in the order the interface lists them.
    pub fn all() -> Vec<AutoModel> {
        let mut v: Vec<AutoModel> = autoseg::all_tasks()
            .into_iter()
            .map(AutoModel::Nn)
            .collect();
        v.extend(Lungmask::ALL.map(AutoModel::Lungmask));
        v.extend(SegResModel::ALL.map(AutoModel::SegRes));
        v.extend(Vista3d::ALL.map(AutoModel::Vista));
        v
    }

    /// The model a key names: a registry key, or one of the older names
    /// the command line and saved workflows used for the `total` variants
    /// (`fast3`, `highres-v3`, ...).
    pub fn from_key(key: &str) -> Option<AutoModel> {
        let k = key.trim();
        if let Some(m) = Self::all()
            .into_iter()
            .find(|m| m.key().eq_ignore_ascii_case(k))
        {
            return Some(m);
        }
        Variant::from_key(k).map(|v| AutoModel::Nn(v.task()))
    }

    /// What a run of nothing in particular uses on a series of this
    /// modality: `total` at 3 mm on CT, `total_mr` at 3 mm on MR.
    pub fn default_for(modality: &str) -> AutoModel {
        if modality.trim().eq_ignore_ascii_case("MR") {
            AutoModel::Nn(&autoseg::total::TOTAL_MR_FAST)
        } else {
            AutoModel::Nn(&autoseg::total::TOTAL_FAST)
        }
    }

    /// The `total` CT variant this model is, if it is one.
    pub fn total_variant(&self) -> Option<Variant> {
        match self {
            AutoModel::Nn(t) => Variant::of_task(t),
            _ => None,
        }
    }

    /// Stable identity: what workflows, the command line and the MCP
    /// server call it.
    pub fn key(&self) -> &'static str {
        match self {
            AutoModel::Nn(t) => t.key,
            AutoModel::Lungmask(m) => m.key(),
            AutoModel::SegRes(m) => m.key(),
            AutoModel::Vista(m) => m.key(),
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            AutoModel::Nn(t) => t.label,
            AutoModel::Lungmask(m) => m.label(),
            AutoModel::SegRes(m) => m.label(),
            AutoModel::Vista(m) => m.label(),
        }
    }

    /// The heading the interface lists it under.
    pub fn group(&self) -> &'static str {
        match self {
            AutoModel::Nn(t) => t.group,
            AutoModel::Lungmask(_) => "Lungs (lungmask)",
            AutoModel::SegRes(_) => "Whole body (SegResNet)",
            AutoModel::Vista(m) if m.weights() == vista3d::Weights::CtMr => {
                "NV-Segment-CTMR (VISTA-3D, CT and MR)"
            }
            AutoModel::Vista(_) => "VISTA-3D",
        }
    }

    pub fn detail(&self) -> &'static str {
        match self {
            AutoModel::Nn(t) => t.detail,
            AutoModel::Lungmask(m) => m.detail(),
            AutoModel::SegRes(m) => m.detail(),
            AutoModel::Vista(m) => m.detail(),
        }
    }

    pub fn family(&self) -> Family {
        match self {
            AutoModel::Nn(_) => Family::NnUnet,
            AutoModel::Lungmask(_) => Family::Unet2d,
            AutoModel::SegRes(_) => Family::SegResNet,
            AutoModel::Vista(_) => Family::Vista3d,
        }
    }

    pub fn modality(&self) -> Modality {
        match self {
            AutoModel::Nn(t) => t.modality,
            AutoModel::Lungmask(_) => Modality::Ct,
            AutoModel::Vista(m) if m.is_mr() => Modality::Mr,
            AutoModel::SegRes(_) | AutoModel::Vista(_) => Modality::Ct,
        }
    }

    pub fn licence(&self) -> Licence {
        match self {
            AutoModel::Nn(t) => t.licence,
            AutoModel::Lungmask(_) => Licence::Apache2,
            AutoModel::SegRes(_) => Licence::Apache2,
            AutoModel::Vista(m) => match m.weights() {
                vista3d::Weights::Ct => Licence::NvidiaOpenModel,
                vista3d::Weights::CtMr => Licence::NvidiaNonCommercial,
            },
        }
    }

    /// Where the weights come from, as the interface says it after the
    /// licence ("downloaded once from ...").
    pub fn weights_origin(&self) -> &'static str {
        use crate::autoseg::weights::Source;
        match self {
            AutoModel::Nn(t) => match t.parts.first().map(|p| &p.spec.source) {
                Some(Source::Licensed { .. }) => {
                    "downloaded once from the TotalSegmentator licence server for your \
                     licence number"
                }
                Some(Source::NnUnetV1 { .. }) => {
                    "one network of the published archive, downloaded once"
                }
                Some(Source::Local) => "converted once from the model folder you added",
                _ => "downloaded once from the official release",
            },
            AutoModel::Lungmask(_) => "downloaded once from the official release",
            AutoModel::SegRes(_) | AutoModel::Vista(_) => {
                "downloaded once from the official Hugging Face repository"
            }
        }
    }

    /// Label `l` of the output is `classes()[l - 1]`.
    pub fn classes(&self) -> &'static [&'static str] {
        match self {
            AutoModel::Nn(t) => t.classes,
            AutoModel::Lungmask(m) => m.classes(),
            AutoModel::SegRes(m) => m.classes(),
            AutoModel::Vista(m) => m.classes(),
        }
    }

    /// The sub-models a run may leave out, by name; empty for a model that
    /// is one network.
    pub fn part_names(&self) -> Vec<String> {
        match self {
            AutoModel::Nn(t) if t.parts.len() > 1 => t
                .parts
                .iter()
                .enumerate()
                .map(|(i, p)| part_name(p.spec.key, i, t.parts.len()))
                .collect(),
            _ => Vec::new(),
        }
    }

    /// For a model in parts, the parts whose networks produce one of
    /// `labels` - so a run asked for a few classes skips the rest. `None`
    /// (run everything) when no label is named or the model is one network.
    pub fn parts_holding(&self, labels: &[u8]) -> Option<Vec<bool>> {
        match self {
            AutoModel::Nn(t) if t.parts.len() > 1 && !labels.is_empty() => Some(
                t.parts
                    .iter()
                    .map(|p| p.lut.iter().any(|l| *l != 0 && labels.contains(l)))
                    .collect(),
            ),
            _ => None,
        }
    }

    /// Bytes still to download before the model runs offline (sub-models
    /// left out by `parts` not counted).
    pub fn download_needed(&self, parts: Option<&[bool]>, root: &Path) -> u64 {
        match self {
            AutoModel::Nn(t) => autoseg::download_needed(t, parts, root),
            AutoModel::Lungmask(m) => unet2d::download_needed(*m, root),
            AutoModel::SegRes(m) => segresnet::download_needed(*m, root),
            AutoModel::Vista(m) => vista3d::download_needed_for(m.weights(), root),
        }
    }

    /// Run the model. Blocking; observe and cancel through `progress`.
    /// `root` is the model folder.
    pub fn run(
        &self,
        volume: &Volume,
        opts: &RunOptions,
        root: &Path,
        progress: &Progress,
    ) -> Result<AutosegResult> {
        match self {
            AutoModel::Nn(t) => autoseg::run(
                volume,
                t,
                &NnOptions {
                    device: opts.device,
                    parts: opts.parts.clone(),
                },
                root,
                progress,
            ),
            AutoModel::Lungmask(m) => unet2d::run(volume, *m, opts.device, root, progress),
            AutoModel::SegRes(m) => segresnet::run(volume, *m, opts.device, root, progress),
            AutoModel::Vista(m) => vista3d::run(volume, *m, opts.device, root, progress),
        }
    }
}

/// A sub-model's short name, from its spec key: `total_part3_cardiac` is
/// "cardiac"; one with no name after its number is "part 2".
fn part_name(spec_key: &str, i: usize, n: usize) -> String {
    let tag = format!("part{}", i + 1);
    match spec_key.split_once(&tag) {
        Some((_, rest)) if rest.starts_with('_') && rest.len() > 1 => rest[1..].replace('_', " "),
        _ => format!("part {} of {n}", i + 1),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_are_unique_and_old_names_still_resolve() {
        let all = AutoModel::all();
        let mut keys: Vec<&str> = all.iter().map(|m| m.key()).collect();
        keys.sort();
        let n = keys.len();
        keys.dedup();
        assert_eq!(keys.len(), n);
        for m in &all {
            assert_eq!(AutoModel::from_key(m.key()), Some(*m));
            assert!(!m.classes().is_empty() && !m.label().is_empty() && !m.group().is_empty());
        }
        assert_eq!(
            AutoModel::from_key("fast3").map(|m| m.key()),
            Some("total_fast")
        );
        assert_eq!(
            AutoModel::from_key("small15-v3").map(|m| m.key()),
            Some("total_v3_small")
        );
        assert_eq!(
            AutoModel::from_key(" Lung_Vessels ").map(|m| m.key()),
            Some("lung_vessels")
        );
        assert!(AutoModel::from_key("nope").is_none());
        assert_eq!(AutoModel::default_for("MR").key(), "total_mr_fast");
        // The rows of this round: licensed TotalSegmentator, nnU-Net v1,
        // NV-Segment-CTMR.
        let brain = AutoModel::from_key("nv_ctmr_brain").unwrap();
        assert_eq!(brain.modality(), Modality::Mr);
        assert_eq!(brain.licence(), Licence::NvidiaNonCommercial);
        assert_eq!(brain.classes().len(), 132);
        let lung = AutoModel::from_key("msd_lung").unwrap();
        assert_eq!(lung.licence(), Licence::CcByNc4);
        assert_eq!(lung.classes(), ["lung_tumor"]);
        let heart = AutoModel::from_key("heartchambers_highres").unwrap();
        assert_eq!(heart.licence(), Licence::TsLicensed);
        assert_eq!(heart.classes().len(), 7);
        assert_eq!(
            AutoModel::from_key("thigh_shoulder_muscles_mr")
                .unwrap()
                .modality(),
            Modality::Mr
        );
        assert_eq!(AutoModel::default_for("CT").key(), "total_fast");
        // Where the weights come from follows the source, not the family.
        assert!(heart.weights_origin().contains("licence server"));
        assert!(lung.weights_origin().contains("archive"));
        assert!(brain.weights_origin().contains("Hugging Face"));
        assert!(AutoModel::from_key("total")
            .unwrap()
            .weights_origin()
            .contains("official release"));
    }

    #[test]
    fn sub_models_are_named_after_their_part() {
        let total = AutoModel::from_key("total").unwrap();
        assert_eq!(
            total.part_names(),
            ["organs", "vertebrae", "cardiac", "muscles", "ribs"]
        );
        let mr = AutoModel::from_key("total_mr").unwrap();
        assert_eq!(mr.part_names(), ["organs", "muscles"]);
        let hn = AutoModel::from_key("headneck_muscles").unwrap();
        assert_eq!(hn.part_names(), ["part 1 of 2", "part 2 of 2"]);
        assert!(AutoModel::from_key("total_fast")
            .unwrap()
            .part_names()
            .is_empty());
        // Heart (51) is in the cardiac part, liver (5) in the organs part.
        assert_eq!(
            total.parts_holding(&[51, 5]),
            Some(vec![true, false, true, false, false])
        );
        assert_eq!(total.parts_holding(&[]), None);
    }
}
