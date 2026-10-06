//! The model registry: every automatic segmentation model the engines can
//! run, as data, behind one way of naming, sizing, licensing and running
//! them.
//!
//! The engines keep their own folders and their own networks - the nnU-Net
//! family in [`crate::autoseg`], the 2-D U-Net of lungmask in
//! [`crate::unet2d`], MONAI's SegResNet in [`crate::segresnet`], VISTA-3D in
//! [`crate::vista3d`] - and this module is the one list in front of them.
//! What the viewer, the command line, the workflows and the MCP server
//! need to know about a model (its key, what it segments, on which
//! modality, under which licence, what it costs to download, how to run
//! it) is answered here and nowhere else; a new model is a new row.
//!
//! The prompt engines (SegVol, MedSAM2, and the point modes of VISTA-3D
//! and nnInteractive) are not rows here: they answer a question the user
//! draws, not a fixed class list. [`Family`] names them so the model
//! manager can group every download the same way.

mod auto;
mod tg263;

pub use auto::*;
pub use tg263::tg263_name;

/// The licence a model's weights are published under, as far as it
/// decides what the application may do with them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Licence {
    /// Apache License 2.0: use, modify, redistribute, also commercially.
    Apache2,
    /// MIT.
    Mit,
    /// Creative Commons Attribution-NonCommercial 4.0: research use only.
    CcByNc4,
    /// Creative Commons Attribution-NonCommercial-ShareAlike 4.0: research
    /// use only.
    CcByNcSa4,
    /// Creative Commons Attribution-ShareAlike 4.0.
    CcBySa4,
    /// The NVIDIA Open Model License: commercial use allowed, with
    /// attribution and conditions on modification and guardrails.
    NvidiaOpenModel,
    /// The NVIDIA OneWay Noncommercial License: academic research only.
    NvidiaNonCommercial,
    /// TotalSegmentator's licensed models: weights served for a licence
    /// number from its licence server (free for non-commercial use,
    /// a paid licence otherwise).
    TsLicensed,
    /// The model repository declares no licence.
    Undeclared,
}

impl Licence {
    /// The licence's usual short name.
    pub fn name(self) -> &'static str {
        match self {
            Licence::Apache2 => "Apache-2.0",
            Licence::Mit => "MIT",
            Licence::CcByNc4 => "CC BY-NC 4.0",
            Licence::CcByNcSa4 => "CC BY-NC-SA 4.0",
            Licence::CcBySa4 => "CC BY-SA 4.0",
            Licence::NvidiaOpenModel => "NVIDIA Open Model License",
            Licence::NvidiaNonCommercial => "NVIDIA non-commercial",
            Licence::TsLicensed => "TotalSegmentator licence",
            Licence::Undeclared => "no licence declared",
        }
    }

    /// Weights that may be used for anything, commercial work included,
    /// with no condition beyond attribution: the ones an installer may
    /// fetch in advance and a workflow may download unasked.
    pub fn open(self) -> bool {
        matches!(self, Licence::Apache2 | Licence::Mit)
    }

    /// Weights whose licence forbids commercial use (without a paid
    /// licence, for TotalSegmentator's).
    pub fn research_only(self) -> bool {
        matches!(
            self,
            Licence::CcByNc4
                | Licence::CcByNcSa4
                | Licence::NvidiaNonCommercial
                | Licence::TsLicensed
        )
    }

    /// Weights that need the user's licence number to download.
    pub fn needs_licence_number(self) -> bool {
        self == Licence::TsLicensed
    }
}

/// The imaging modality a model was trained for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Modality {
    Ct,
    Mr,
    /// Trained on both (MRSegmentator).
    CtMr,
}

impl Modality {
    pub fn label(self) -> &'static str {
        match self {
            Modality::Ct => "CT",
            Modality::Mr => "MR",
            Modality::CtMr => "CT, MR",
        }
    }

    /// Whether a series of DICOM modality `m` (`CT`, `MR`, ...) is what
    /// the model expects. Anything that is not MR counts as CT (CBCT and
    /// synthetic CT carry `CT`; PET is refused by both).
    pub fn accepts(self, m: &str) -> bool {
        let mr = m.trim().eq_ignore_ascii_case("MR");
        let pt = m.trim().eq_ignore_ascii_case("PT");
        match self {
            Modality::Ct => !mr && !pt,
            Modality::Mr => mr,
            Modality::CtMr => !pt,
        }
    }
}

/// Which runner a model needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Family {
    /// nnU-Net v2, plain or residual encoder ([`crate::autoseg`]).
    NnUnet,
    /// The 2-D U-Net of lungmask ([`crate::unet2d`]).
    Unet2d,
    /// MONAI's SegResNet ([`crate::segresnet`]).
    SegResNet,
    /// VISTA-3D ([`crate::vista3d`]).
    Vista3d,
    /// SegVol, text / box / point prompts ([`crate::segvol`]).
    SegVol,
    /// MedSAM2, slice propagation ([`crate::medsam2`]).
    MedSam2,
}

impl Family {
    pub fn label(self) -> &'static str {
        match self {
            Family::NnUnet => "nnU-Net",
            Family::Unet2d => "2-D U-Net",
            Family::SegResNet => "SegResNet",
            Family::Vista3d => "VISTA-3D",
            Family::SegVol => "SegVol",
            Family::MedSam2 => "MedSAM2",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modality_matching_follows_the_dicom_tag() {
        assert!(Modality::Ct.accepts("CT"));
        assert!(!Modality::Ct.accepts("mr"));
        assert!(Modality::Mr.accepts(" MR "));
        assert!(!Modality::Mr.accepts("CT"));
        assert!(Modality::CtMr.accepts("MR") && Modality::CtMr.accepts("CT"));
        assert!(!Modality::CtMr.accepts("PT") && !Modality::Ct.accepts("PT"));
    }

    #[test]
    fn licences_sort_into_open_and_research_only() {
        assert!(Licence::Apache2.open() && !Licence::Apache2.research_only());
        assert!(Licence::CcByNc4.research_only() && !Licence::CcByNc4.open());
        assert!(!Licence::NvidiaOpenModel.open() && !Licence::NvidiaOpenModel.research_only());
        assert!(Licence::NvidiaNonCommercial.research_only());
        assert!(Licence::TsLicensed.research_only() && Licence::TsLicensed.needs_licence_number());
    }
}
