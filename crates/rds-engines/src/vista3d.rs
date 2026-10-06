//! VISTA-3D (NVIDIA NV-Segment-CT: He et al., *VISTA3D: A Unified
//! Segmentation Foundation Model For 3D Medical Imaging*, CVPR 2025) -
//! automatic segmentation of up to 127 CT classes, tumours among them,
//! re-implemented natively.
//!
//! The network is MONAI's `vista3d132`: a `SegResNetDS2` image encoder
//! (48 base filters, InstanceNorm, five levels) whose class branch feeds
//! a class head - two residual blocks over the features, then, for every
//! class asked for, a learned 48-wide class embedding (through a small
//! MLP) dotted with the feature at every voxel. A class is present where
//! its logit is positive; where several are, the largest wins. The point
//! branch and its SAM-style head answer clicked prompts, which this
//! automatic mode does not use.
//!
//! The pipeline is the bundle's (`configs/inference.json`, version
//! 0.5.12): resampled to 1.5 mm in the scan's own axis order, cropped to
//! the voxels above 0 HU with a 10-voxel margin, HU -963.8..1053.7 scaled
//! to [0, 1], reoriented to RAS; 128-cubed windows with uniform weights,
//! overlap 0.25, edge-replicated padding (`SlidingWindowInfererAdapt`, as
//! the bundle's inferer calls it for class prompts); then back.
//!
//! Weights: `nvidia/NV-Segment-CT` on Hugging Face, NVIDIA Open Model
//! License (commercial use allowed, with attribution and conditions).
//!
//! NV-Segment-CTMR is the same network trained on CT and MR, with 345
//! classes (the brain parcellation among them) and NVIDIA's non-commercial
//! licence. Its pipeline (`vista3d_pipeline.py` of the repository) differs
//! in three places: intensities are scaled between the image's own 1st
//! and 99th percentiles instead of a fixed HU window, the windows are
//! 192 x 192 x 128, and which classes are asked for depends on the
//! modality (`CT_BODY`, `MRI_BODY`, `MRI_BRAIN`, the last for
//! skull-stripped T1 only). Its labels are numbered in the order of the
//! classes asked for, so the 345 ids fit an 8-bit label map.

use anyhow::{bail, Result};
use burn::tensor::activation::{gelu, leaky_relu};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use std::path::Path;

use crate::autoseg::infer::{self, Fill, InferHooks, WindowPlan};
use crate::autoseg::{organ_hits, AutosegResult};
use crate::medsam2::ops;
use crate::nn::cache::{self, RemoteFile};
use crate::nn::device::DevicePref;
use crate::nn::fastconv::{self, Activation};
use crate::nn::params::Params;
use crate::progress::{Progress, ProgressSink, CANCELLED};
use crate::segresnet::net::{Conv, NormKind, SegResNetDs};
use crate::segresnet::pre;
use crate::volume::{AxisOrder, Volume};

pub mod points;
pub mod session;

/// The engine folder under the model root.
pub const DIR: &str = "vista3d";

/// VISTA-3D's 132 classes; label `l` is `CLASSES[l - 1]`
/// (`configs/metadata.json`).
pub const CLASSES: [&str; 132] = [
    "liver",
    "kidney",
    "spleen",
    "pancreas",
    "right kidney",
    "aorta",
    "inferior vena cava",
    "right adrenal gland",
    "left adrenal gland",
    "gallbladder",
    "esophagus",
    "stomach",
    "duodenum",
    "left kidney",
    "bladder",
    "prostate or uterus",
    "portal vein and splenic vein",
    "rectum",
    "small bowel",
    "lung",
    "bone",
    "brain",
    "lung tumor",
    "pancreatic tumor",
    "hepatic vessel",
    "hepatic tumor",
    "colon cancer primaries",
    "left lung upper lobe",
    "left lung lower lobe",
    "right lung upper lobe",
    "right lung middle lobe",
    "right lung lower lobe",
    "vertebrae L5",
    "vertebrae L4",
    "vertebrae L3",
    "vertebrae L2",
    "vertebrae L1",
    "vertebrae T12",
    "vertebrae T11",
    "vertebrae T10",
    "vertebrae T9",
    "vertebrae T8",
    "vertebrae T7",
    "vertebrae T6",
    "vertebrae T5",
    "vertebrae T4",
    "vertebrae T3",
    "vertebrae T2",
    "vertebrae T1",
    "vertebrae C7",
    "vertebrae C6",
    "vertebrae C5",
    "vertebrae C4",
    "vertebrae C3",
    "vertebrae C2",
    "vertebrae C1",
    "trachea",
    "left iliac artery",
    "right iliac artery",
    "left iliac vena",
    "right iliac vena",
    "colon",
    "left rib 1",
    "left rib 2",
    "left rib 3",
    "left rib 4",
    "left rib 5",
    "left rib 6",
    "left rib 7",
    "left rib 8",
    "left rib 9",
    "left rib 10",
    "left rib 11",
    "left rib 12",
    "right rib 1",
    "right rib 2",
    "right rib 3",
    "right rib 4",
    "right rib 5",
    "right rib 6",
    "right rib 7",
    "right rib 8",
    "right rib 9",
    "right rib 10",
    "right rib 11",
    "right rib 12",
    "left humerus",
    "right humerus",
    "left scapula",
    "right scapula",
    "left clavicula",
    "right clavicula",
    "left femur",
    "right femur",
    "left hip",
    "right hip",
    "sacrum",
    "left gluteus maximus",
    "right gluteus maximus",
    "left gluteus medius",
    "right gluteus medius",
    "left gluteus minimus",
    "right gluteus minimus",
    "left autochthon",
    "right autochthon",
    "left iliopsoas",
    "right iliopsoas",
    "left atrial appendage",
    "brachiocephalic trunk",
    "left brachiocephalic vein",
    "right brachiocephalic vein",
    "left common carotid artery",
    "right common carotid artery",
    "costal cartilages",
    "heart",
    "left kidney cyst",
    "right kidney cyst",
    "prostate",
    "pulmonary vein",
    "skull",
    "spinal cord",
    "sternum",
    "left subclavian artery",
    "right subclavian artery",
    "superior vena cava",
    "thyroid gland",
    "vertebrae S1",
    "bone lesion",
    "kidney mass",
    "liver tumor",
    "vertebrae L6",
    "airway",
];

/// NV-Segment-CTMR's 345 classes (`metadata.json` of the repository);
/// class `c` is `CTMR_CLASSES[c - 1]`.
pub const CTMR_CLASSES: [&str; 345] = [
    "liver",
    "kidney",
    "spleen",
    "pancreas",
    "right kidney",
    "aorta",
    "inferior vena cava",
    "right adrenal gland",
    "left adrenal gland",
    "gallbladder",
    "esophagus",
    "stomach",
    "duodenum",
    "left kidney",
    "bladder",
    "prostate or uterus (deprecated)",
    "portal vein and splenic vein",
    "rectum",
    "small bowel",
    "lung",
    "bone",
    "brain",
    "lung tumor",
    "pancreatic tumor",
    "hepatic vessel",
    "hepatic tumor",
    "colon cancer primaries",
    "left lung upper lobe",
    "left lung lower lobe",
    "right lung upper lobe",
    "right lung middle lobe",
    "right lung lower lobe",
    "vertebrae L5",
    "vertebrae L4",
    "vertebrae L3",
    "vertebrae L2",
    "vertebrae L1",
    "vertebrae T12",
    "vertebrae T11",
    "vertebrae T10",
    "vertebrae T9",
    "vertebrae T8",
    "vertebrae T7",
    "vertebrae T6",
    "vertebrae T5",
    "vertebrae T4",
    "vertebrae T3",
    "vertebrae T2",
    "vertebrae T1",
    "vertebrae C7",
    "vertebrae C6",
    "vertebrae C5",
    "vertebrae C4",
    "vertebrae C3",
    "vertebrae C2",
    "vertebrae C1",
    "trachea",
    "left iliac artery",
    "right iliac artery",
    "left iliac vena",
    "right iliac vena",
    "colon",
    "left rib 1",
    "left rib 2",
    "left rib 3",
    "left rib 4",
    "left rib 5",
    "left rib 6",
    "left rib 7",
    "left rib 8",
    "left rib 9",
    "left rib 10",
    "left rib 11",
    "left rib 12",
    "right rib 1",
    "right rib 2",
    "right rib 3",
    "right rib 4",
    "right rib 5",
    "right rib 6",
    "right rib 7",
    "right rib 8",
    "right rib 9",
    "right rib 10",
    "right rib 11",
    "right rib 12",
    "left humerus",
    "right humerus",
    "left scapula",
    "right scapula",
    "left clavicula",
    "right clavicula",
    "left femur",
    "right femur",
    "left hip",
    "right hip",
    "sacrum",
    "left gluteus maximus",
    "right gluteus maximus",
    "left gluteus medius",
    "right gluteus medius",
    "left gluteus minimus",
    "right gluteus minimus",
    "left autochthon",
    "right autochthon",
    "left iliopsoas",
    "right iliopsoas",
    "left atrial appendage",
    "brachiocephalic trunk",
    "left brachiocephalic vein",
    "right brachiocephalic vein",
    "left common carotid artery",
    "right common carotid artery",
    "costal cartilages",
    "heart",
    "left kidney cyst",
    "right kidney cyst",
    "prostate",
    "pulmonary vein",
    "skull",
    "spinal cord",
    "sternum",
    "left subclavian artery",
    "right subclavian artery",
    "superior vena cava",
    "thyroid gland",
    "vertebrae S1",
    "bone lesion",
    "kidney mass (deprecated)",
    "liver tumor (deprecated)",
    "vertebrae L6 (deprecated)",
    "airway",
    "fibula (deprecated)",
    "intervertebral discs",
    "left lung",
    "right lung",
    "left quadriceps femoris (deprecated)",
    "right quadriceps femoris (deprecated)",
    "left sartorius (deprecated)",
    "right sartorius (deprecated)",
    "left thigh medial compartment (deprecated)",
    "right thigh medial compartment (deprecated)",
    "left thigh posterior compartment (deprecated)",
    "right thigh posterior compartment (deprecated)",
    "tibia (deprecated)",
    "vertebrae",
    "prostate transitional zone",
    "prostate peripheral zone",
    "left atrium",
    "white matter hyperintensity",
    "left ventricle",
    "right ventricle",
    "right atrium",
    "left ventricle myocardium",
    "ascending aorta (deprecated)",
    "muscles",
    "fat",
    "abdominal tissue",
    "mediastinal tissue",
    "gonads",
    "uterocervix",
    "uterus (deprecated)",
    "breast left",
    "breast right",
    "thyroid left",
    "thyroid right",
    "thymus",
    "skin",
    "heart tissue",
    "celiac trunk",
    "pulmonary artery",
    "cheek left",
    "cheek right",
    "eyeball left",
    "eyeball right",
    "brain tumor",
    "chiasm",
    "left temporal lobe",
    "right temporal lobe",
    "left eye",
    "right eye",
    "left lens",
    "right lens",
    "left optic nerve",
    "right optic nerve",
    "left middle ear",
    "right middle ear",
    "left internal auditory canal",
    "right internal auditory canal",
    "left tympanic cavity",
    "right tympanic cavity",
    "left vestibular semicircular canals",
    "right vestibular semicircular canals",
    "left cochlea",
    "right cochlea",
    "left ethmoid bone",
    "right ethmoid bone",
    "pituitary",
    "oral cavity",
    "left mandible",
    "right mandible",
    "left submandibular",
    "right submandibular",
    "left parotid",
    "right parotid",
    "left mastoid",
    "right mastoid",
    "left temporomandibular joint",
    "right temporomandibular joint",
    "larynx",
    "larynx glottic",
    "larynx supraglot",
    "pharynxConst",
    "3rd-Ventricle",
    "4th-Ventricle",
    "Right-Accumbens-Area",
    "Left-Accumbens-Area",
    "Right-Amygdala",
    "Left-Amygdala",
    "Brain-Stem",
    "Right-Caudate",
    "Left-Caudate",
    "Right-Cerebellum-Exterior",
    "Left-Cerebellum-Exterior",
    "Right-Cerebellum-White-Matter",
    "Left-Cerebellum-White-Matter",
    "Right-Cerebral-White-Matter",
    "Left-Cerebral-White-Matter",
    "Right-Hippocampus",
    "Left-Hippocampus",
    "Right-Inf-Lat-Vent",
    "Left-Inf-Lat-Vent",
    "Right-Lateral-Ventricle",
    "Left-Lateral-Ventricle",
    "Right-Pallidum",
    "Left-Pallidum",
    "Right-Putamen",
    "Left-Putamen",
    "Right-Thalamus-Proper",
    "Left-Thalamus-Proper",
    "Right-Ventral-DC",
    "Left-Ventral-DC",
    "Cerebellar-Vermal-Lobules-I-V",
    "Cerebellar-Vermal-Lobules-VI-VII",
    "Cerebellar-Vermal-Lobules-VIII-X",
    "Left-Basal-Forebrain",
    "Right-Basal-Forebrain",
    "Right-ACgG--anterior-cingulate-gyrus",
    "Left-ACgG--anterior-cingulate-gyrus",
    "Right-AIns--anterior-insula",
    "Left-AIns--anterior-insula",
    "Right-AOrG--anterior-orbital-gyrus",
    "Left-AOrG--anterior-orbital-gyrus",
    "Right-AnG---angular-gyrus",
    "Left-AnG---angular-gyrus",
    "Right-Calc--calcarine-cortex",
    "Left-Calc--calcarine-cortex",
    "Right-CO----central-operculum",
    "Left-CO----central-operculum",
    "Right-Cun---cuneus",
    "Left-Cun---cuneus",
    "Right-Ent---entorhinal-area",
    "Left-Ent---entorhinal-area",
    "Right-FO----frontal-operculum",
    "Left-FO----frontal-operculum",
    "Right-FRP---frontal-pole",
    "Left-FRP---frontal-pole",
    "Right-FuG---fusiform-gyrus",
    "Left-FuG---fusiform-gyrus",
    "Right-GRe---gyrus-rectus",
    "Left-GRe---gyrus-rectus",
    "Right-IOG---inferior-occipital-gyrus",
    "Left-IOG---inferior-occipital-gyrus",
    "Right-ITG---inferior-temporal-gyrus",
    "Left-ITG---inferior-temporal-gyrus",
    "Right-LiG---lingual-gyrus",
    "Left-LiG---lingual-gyrus",
    "Right-LOrG--lateral-orbital-gyrus",
    "Left-LOrG--lateral-orbital-gyrus",
    "Right-MCgG--middle-cingulate-gyrus",
    "Left-MCgG--middle-cingulate-gyrus",
    "Right-MFC---medial-frontal-cortex",
    "Left-MFC---medial-frontal-cortex",
    "Right-MFG---middle-frontal-gyrus",
    "Left-MFG---middle-frontal-gyrus",
    "Right-MOG---middle-occipital-gyrus",
    "Left-MOG---middle-occipital-gyrus",
    "Right-MOrG--medial-orbital-gyrus",
    "Left-MOrG--medial-orbital-gyrus",
    "Right-MPoG--postcentral-gyrus",
    "Left-MPoG--postcentral-gyrus",
    "Right-MPrG--precentral-gyrus",
    "Left-MPrG--precentral-gyrus",
    "Right-MSFG--superior-frontal-gyrus",
    "Left-MSFG--superior-frontal-gyrus",
    "Right-MTG---middle-temporal-gyrus",
    "Left-MTG---middle-temporal-gyrus",
    "Right-OCP---occipital-pole",
    "Left-OCP---occipital-pole",
    "Right-OFuG--occipital-fusiform-gyrus",
    "Left-OFuG--occipital-fusiform-gyrus",
    "Right-OpIFG-opercular-part-of-the-IFG",
    "Left-OpIFG-opercular-part-of-the-IFG",
    "Right-OrIFG-orbital-part-of-the-IFG",
    "Left-OrIFG-orbital-part-of-the-IFG",
    "Right-PCgG--posterior-cingulate-gyrus",
    "Left-PCgG--posterior-cingulate-gyrus",
    "Right-PCu---precuneus",
    "Left-PCu---precuneus",
    "Right-PHG---parahippocampal-gyrus",
    "Left-PHG---parahippocampal-gyrus",
    "Right-PIns--posterior-insula",
    "Left-PIns--posterior-insula",
    "Right-PO----parietal-operculum",
    "Left-PO----parietal-operculum",
    "Right-PoG---postcentral-gyrus",
    "Left-PoG---postcentral-gyrus",
    "Right-POrG--posterior-orbital-gyrus",
    "Left-POrG--posterior-orbital-gyrus",
    "Right-PP----planum-polare",
    "Left-PP----planum-polare",
    "Right-PrG---precentral-gyrus",
    "Left-PrG---precentral-gyrus",
    "Right-PT----planum-temporale",
    "Left-PT----planum-temporale",
    "Right-SCA---subcallosal-area",
    "Left-SCA---subcallosal-area",
    "Right-SFG---superior-frontal-gyrus",
    "Left-SFG---superior-frontal-gyrus",
    "Right-SMC---supplementary-motor-cortex",
    "Left-SMC---supplementary-motor-cortex",
    "Right-SMG---supramarginal-gyrus",
    "Left-SMG---supramarginal-gyrus",
    "Right-SOG---superior-occipital-gyrus",
    "Left-SOG---superior-occipital-gyrus",
    "Right-SPL---superior-parietal-lobule",
    "Left-SPL---superior-parietal-lobule",
    "Right-STG---superior-temporal-gyrus",
    "Left-STG---superior-temporal-gyrus",
    "Right-TMP---temporal-pole",
    "Left-TMP---temporal-pole",
    "Right-TrIFG-triangular-part-of-the-IFG",
    "Left-TrIFG-triangular-part-of-the-IFG",
    "Right-TTG---transverse-temporal-gyrus",
    "Left-TTG---transverse-temporal-gyrus",
];

/// The weights as published.
pub const WEIGHTS: RemoteFile = RemoteFile {
    name: "model.safetensors",
    url: "https://huggingface.co/nvidia/NV-Segment-CT/resolve/main/vista3d_pretrained_model/model.safetensors",
    bytes: 871_894_112,
};

/// NV-Segment-CTMR's weights, kept in the [`DIR_CTMR`] sub-folder.
pub const WEIGHTS_CTMR: RemoteFile = RemoteFile {
    name: "model.safetensors",
    url: "https://huggingface.co/nvidia/NV-Segment-CTMR/resolve/main/vista3d_pretrained_model/model.safetensors",
    bytes: 871_892_040,
};

/// The CTMR weights' folder inside the engine folder.
pub const DIR_CTMR: &str = "ctmr";

/// Which training of the network.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Weights {
    /// NV-Segment-CT (VISTA-3D as published with MONAI).
    Ct,
    /// NV-Segment-CTMR.
    CtMr,
}

impl Weights {
    pub fn file(self) -> RemoteFile {
        match self {
            Weights::Ct => WEIGHTS,
            Weights::CtMr => WEIGHTS_CTMR,
        }
    }

    /// Where the file lives, given the model root.
    pub fn dir(self, root: &Path) -> std::path::PathBuf {
        match self {
            Weights::Ct => root.join(DIR),
            Weights::CtMr => root.join(DIR).join(DIR_CTMR),
        }
    }
}

/// What VISTA-3D is asked to find.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Vista3d {
    /// The bundle's `everything_labels`: every class but the groups
    /// (kidney, lung, bone), the duplicates, the lesions and the airway -
    /// 117 classes.
    Everything,
    /// The lesion classes: lung tumour, pancreatic tumour, hepatic tumour,
    /// colon cancer primaries, bone lesion, kidney mass, liver tumour.
    Lesions,
    /// NV-Segment-CTMR on CT: `CT_BODY`, the same 117 classes.
    CtmrCt,
    /// NV-Segment-CTMR on MR: `MRI_BODY`, 50 classes.
    CtmrMr,
    /// NV-Segment-CTMR on skull-stripped T1 MR: `MRI_BRAIN`, the 132
    /// classes of the brain parcellation.
    CtmrBrain,
}

/// NV-Segment-CTMR's `MRI_BODY` classes.
const MRI_BODY: [u16; 50] = [
    1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 22, 58, 59, 60, 61, 62, 87, 88, 89, 90,
    91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 115, 118, 121, 134,
    135, 136, 146,
];

impl Vista3d {
    pub const ALL: [Vista3d; 5] = [
        Vista3d::Everything,
        Vista3d::Lesions,
        Vista3d::CtmrCt,
        Vista3d::CtmrMr,
        Vista3d::CtmrBrain,
    ];

    pub fn key(self) -> &'static str {
        match self {
            Vista3d::Everything => "vista3d",
            Vista3d::Lesions => "vista3d_lesions",
            Vista3d::CtmrCt => "nv_ctmr_ct",
            Vista3d::CtmrMr => "nv_ctmr_mr",
            Vista3d::CtmrBrain => "nv_ctmr_brain",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Vista3d::Everything => "VISTA-3D",
            Vista3d::Lesions => "VISTA-3D lesions",
            Vista3d::CtmrCt => "NV-Segment-CTMR, CT",
            Vista3d::CtmrMr => "NV-Segment-CTMR, MR body",
            Vista3d::CtmrBrain => "NV-Segment-CTMR, MR brain",
        }
    }

    pub fn detail(self) -> &'static str {
        match self {
            Vista3d::Everything => {
                "NVIDIA VISTA-3D (NV-Segment-CT), automatic mode: 117 organs, vessels, bones \
                 and muscles at 1.5 mm. GPU recommended."
            }
            Vista3d::Lesions => {
                "VISTA-3D asked for its lesion classes only: lung, pancreatic and hepatic \
                 tumours, colon cancer primaries, bone lesions, kidney masses, liver tumours. \
                 GPU recommended."
            }
            Vista3d::CtmrCt => {
                "NVIDIA NV-Segment-CTMR (VISTA-3D trained on CT and MR) on CT: the same 117 \
                 classes, intensities scaled per image. Non-commercial. GPU recommended."
            }
            Vista3d::CtmrMr => {
                "NV-Segment-CTMR on MR: 50 abdominal and pelvic organs, vessels, bones and \
                 muscles. Non-commercial. GPU recommended."
            }
            Vista3d::CtmrBrain => {
                "NV-Segment-CTMR's brain parcellation, 132 structures, on a skull-stripped T1 \
                 MR. Non-commercial. GPU recommended."
            }
        }
    }

    pub fn weights(self) -> Weights {
        match self {
            Vista3d::Everything | Vista3d::Lesions => Weights::Ct,
            _ => Weights::CtMr,
        }
    }

    pub fn is_mr(self) -> bool {
        matches!(self, Vista3d::CtmrMr | Vista3d::CtmrBrain)
    }

    /// The class ids prompted, in the order their logits are stacked.
    pub fn prompts(self) -> Vec<u16> {
        let ct_body = || {
            (1..=132u16)
                .filter(|l| {
                    ![
                        2, 16, 18, 20, 21, 23, 24, 25, 26, 27, 128, 129, 130, 131, 132,
                    ]
                    .contains(l)
                })
                .collect()
        };
        match self {
            Vista3d::Everything | Vista3d::CtmrCt => ct_body(),
            Vista3d::Lesions => vec![23, 24, 26, 27, 128, 129, 130],
            Vista3d::CtmrMr => MRI_BODY.to_vec(),
            Vista3d::CtmrBrain => (214..=345).collect(),
        }
    }

    /// The label each prompted class gets in the output: its class id for
    /// NV-Segment-CT (which [`CLASSES`] names), its position in
    /// [`Self::prompts`] for NV-Segment-CTMR (ids go up to 345).
    pub fn output_labels(self) -> Vec<u8> {
        let p = self.prompts();
        match self.weights() {
            Weights::Ct => p.iter().map(|&c| c as u8).collect(),
            Weights::CtMr => (1..=p.len() as u8).collect(),
        }
    }

    /// Label `l` of the output is `classes()[l - 1]`.
    pub fn classes(self) -> &'static [&'static str] {
        static TABLES: [std::sync::OnceLock<Vec<&'static str>>; 3] =
            [const { std::sync::OnceLock::new() }; 3];
        let slot = match self {
            Vista3d::Everything | Vista3d::Lesions => return &CLASSES,
            Vista3d::CtmrCt => 0,
            Vista3d::CtmrMr => 1,
            Vista3d::CtmrBrain => 2,
        };
        TABLES[slot].get_or_init(|| {
            self.prompts()
                .iter()
                .map(|&c| CTMR_CLASSES[c as usize - 1])
                .collect()
        })
    }

    /// The sliding-window patch, in RAS axis order.
    pub fn patch(self) -> [usize; 3] {
        match self.weights() {
            Weights::Ct => [128; 3],
            Weights::CtMr => [192, 192, 128],
        }
    }
}

/// Bytes still to download before VISTA-3D runs offline.
pub fn download_needed(root: &Path) -> u64 {
    download_needed_for(Weights::Ct, root)
}

/// Bytes still to download before a training runs offline.
pub fn download_needed_for(w: Weights, root: &Path) -> u64 {
    if w.file().is_cached(&w.dir(root)) {
        0
    } else {
        w.file().bytes
    }
}

/// The weights, downloaded on first use.
pub fn load(root: &Path, sink: &dyn ProgressSink) -> Result<Params> {
    load_weights(Weights::Ct, root, sink)
}

/// One training's weights, downloaded on first use.
pub fn load_weights(w: Weights, root: &Path, sink: &dyn ProgressSink) -> Result<Params> {
    let path = w.file().ensure(&w.dir(root), sink)?;
    sink.report(0.0, "Loading weights (VISTA-3D)");
    Ok(strip_network_map(cache::load_safetensors(&path)?))
}

/// The Hugging Face export wraps the MONAI network as `network`; that
/// prefix is dropped.
fn strip_network_map(t: std::collections::HashMap<String, crate::nn::cache::WTensor>) -> Params {
    Params::new(
        t.into_iter()
            .map(|(k, v)| (k.strip_prefix("network.").unwrap_or(&k).to_string(), v))
            .collect(),
    )
}

/// MONAI's `UnetResBlock` with InstanceNorm (no affine) and LeakyReLU 0.01,
/// equal widths.
struct ResBlock<B: Backend> {
    conv1: Conv<B>,
    conv2: Conv<B>,
}

impl<B: Backend> ResBlock<B> {
    fn apply(&self, x: Tensor<B, 5>) -> Tensor<B, 5> {
        let c = x.dims()[1];
        let y = fastconv::group_norm_act(
            self.conv1.apply(x.clone()),
            c,
            None,
            None,
            1e-5,
            Activation::LeakyRelu(0.01),
        );
        let y =
            fastconv::group_norm_act(self.conv2.apply(y), c, None, None, 1e-5, Activation::None);
        leaky_relu(y + x, 0.01)
    }
}

/// The automatic half of `vista3d132`.
pub struct Vista3dNet<B: Backend> {
    encoder: SegResNetDs<B>,
    post: [ResBlock<B>; 2],
    /// The prompted classes' embeddings after the MLP, `[k, width]`.
    embed: Tensor<B, 2>,
    width: usize,
    device: B::Device,
}

impl<B: Backend> Vista3dNet<B> {
    /// `vista3d132`'s automatic half: 48 features, encoder levels of
    /// 1, 2, 2, 4, 4 blocks.
    pub fn load(p: &Params, prompts: &[u16], dev: &B::Device) -> Result<Self> {
        Self::load_with(p, prompts, &[1, 2, 2, 4, 4], dev)
    }

    /// Any width (read from the class embeddings) and encoder depth.
    pub fn load_with(
        p: &Params,
        prompts: &[u16],
        blocks_down: &[usize],
        dev: &B::Device,
    ) -> Result<Self> {
        let width = p
            .shape("class_head.class_embeddings.weight")
            .and_then(|s| s.get(1).copied())
            .ok_or_else(|| anyhow::anyhow!("checkpoint has no class embeddings"))?;
        #[allow(non_snake_case)]
        let F = width;
        let encoder = SegResNetDs::load(
            p,
            "image_encoder.",
            F,
            blocks_down,
            F,
            NormKind::Instance,
            None,
            Some("up_layers_auto"),
            true,
            dev,
        )?;
        let block = |i: usize| -> Result<ResBlock<B>> {
            Ok(ResBlock {
                conv1: Conv::load(
                    p,
                    &format!("class_head.image_post_mapping.{i}.layer.conv1.conv"),
                    F,
                    F,
                    3,
                    1,
                    false,
                    dev,
                )?,
                conv2: Conv::load(
                    p,
                    &format!("class_head.image_post_mapping.{i}.layer.conv2.conv"),
                    F,
                    F,
                    3,
                    1,
                    false,
                    dev,
                )?,
            })
        };
        // The class embeddings go through the MLP once, here: Linear,
        // InstanceNorm1d over the 48 features (no affine), exact GELU,
        // Linear.
        let table = p.get("class_head.class_embeddings.weight", &[512, F])?;
        let k = prompts.len();
        let mut e = Vec::with_capacity(k * F);
        for &c in prompts {
            if c as usize >= 512 {
                bail!("class id {c} is past the 512 class embeddings");
            }
            e.extend_from_slice(&table[c as usize * F..(c as usize + 1) * F]);
        }
        let x = ops::from_slice::<B, 2>(&e, [k, F], dev);
        let lin = |x: Tensor<B, 2>, name: &str| -> Result<Tensor<B, 2>> {
            let (w, b) = p.linear(name, F, F)?;
            let w = ops::from_slice::<B, 2>(w, [F, F], dev);
            let b = ops::from_slice::<B, 1>(b, [F], dev);
            Ok(x.matmul(w.transpose()) + b.unsqueeze_dim(0))
        };
        let h = lin(x, "class_head.mlp.0")?;
        let mean = h.clone().mean_dim(1);
        let c = h - mean;
        let var = c.clone().powf_scalar(2.0).mean_dim(1);
        let h = gelu(c / (var + 1e-5).sqrt());
        let embed = lin(h, "class_head.mlp.3")?;
        Ok(Vista3dNet {
            encoder,
            post: [block(0)?, block(1)?],
            embed,
            width: F,
            device: dev.clone(),
        })
    }

    /// One window `[p0, p1, p2]` → `[1 + k, p0, p1, p2]`: a zero
    /// background logit, then each prompted class's.
    pub fn window(&self, patch: &[f32], p: [usize; 3]) -> Vec<f32> {
        let x = ops::from_slice::<B, 5>(patch, [1, 1, p[0], p[1], p[2]], &self.device);
        let mut f = self.encoder.forward_auto(x);
        for b in &self.post {
            f = b.apply(f);
        }
        let n = p[0] * p[1] * p[2];
        let logits = self.embed.clone().matmul(f.reshape([self.width, n]));
        let zero = Tensor::<B, 2>::zeros([1, n], &self.device);
        ops::to_vec(Tensor::cat(vec![zero, logits], 0))
    }
}

struct Hooks<'a, B: Backend> {
    net: &'a Vista3dNet<B>,
    patch: [usize; 3],
    progress: &'a Progress,
}

impl<B: Backend> InferHooks for Hooks<'_, B> {
    fn forward(&self, patch: &[f32]) -> Result<Vec<f32>> {
        Ok(self.net.window(patch, self.patch))
    }
    fn tile_done(&self, done: usize, total: usize) -> bool {
        self.progress.report(
            done as f32 / total as f32,
            &format!("Segmenting (VISTA-3D): window {done}/{total}"),
        );
        !self.progress.cancelled()
    }
}

/// The bundle's automatic sliding window: 128-cubed, uniform weights,
/// overlap 0.25, edge-replicated padding.
pub fn plan(dims: [usize; 3]) -> WindowPlan {
    plan_for(dims, [128; 3])
}

/// The same with another patch (NV-Segment-CTMR's 192 x 192 x 128): the
/// overlap passed to the class-prompt inferer is its default, 0.25.
pub fn plan_for(dims: [usize; 3], patch: [usize; 3]) -> WindowPlan {
    let mut plan = infer::monai_plan(dims, patch, 0.25, Fill::Replicate);
    plan.profiles = std::array::from_fn(|a| vec![1.0; patch[a]]);
    plan.floor = 0.0;
    plan
}

fn windows<B: Backend>(
    params: &Params,
    prompts: &[u16],
    patch: [usize; 3],
    dev: &B::Device,
    data: &[f32],
    dims: [usize; 3],
    progress: &Progress,
) -> Result<Vec<u8>> {
    let net = Vista3dNet::<B>::load(params, prompts, dev)?;
    let hooks = Hooks {
        net: &net,
        patch,
        progress,
    };
    infer::predict_plan(
        data,
        dims,
        prompts.len() + 1,
        &plan_for(dims, patch),
        infer::ACC_BUDGET,
        &hooks,
    )
}

/// Permute and flip a C-order volume: output axis `a` is input axis
/// `perm[a]`, reversed where `flip[a]`.
pub fn permute<T: Copy + Default + Send + Sync>(
    data: &[T],
    dims: [usize; 3],
    perm: [usize; 3],
    flip: [bool; 3],
) -> (Vec<T>, [usize; 3]) {
    let out: [usize; 3] = std::array::from_fn(|a| dims[perm[a]]);
    let mut v = vec![T::default(); data.len()];
    for i in 0..out[0] {
        for j in 0..out[1] {
            for k in 0..out[2] {
                let o = [i, j, k];
                let mut src = [0usize; 3];
                for a in 0..3 {
                    src[perm[a]] = if flip[a] { out[a] - 1 - o[a] } else { o[a] };
                }
                v[(i * out[1] + j) * out[2] + k] =
                    data[(src[0] * dims[1] + src[1]) * dims[2] + src[2]];
            }
        }
    }
    (v, out)
}

/// The inverse of [`permute`].
pub fn unpermute<T: Copy + Default + Send + Sync>(
    data: &[T],
    dims_out: [usize; 3],
    perm: [usize; 3],
    flip: [bool; 3],
) -> Vec<T> {
    let pdims: [usize; 3] = std::array::from_fn(|a| dims_out[perm[a]]);
    let mut v = vec![T::default(); data.len()];
    for i in 0..pdims[0] {
        for j in 0..pdims[1] {
            for k in 0..pdims[2] {
                let o = [i, j, k];
                let mut src = [0usize; 3];
                for a in 0..3 {
                    src[perm[a]] = if flip[a] { pdims[a] - 1 - o[a] } else { o[a] };
                }
                v[(src[0] * dims_out[1] + src[1]) * dims_out[2] + src[2]] =
                    data[(i * pdims[1] + j) * pdims[2] + k];
            }
        }
    }
    v
}

/// A volume prepared the bundle's way, with what it takes to go back.
pub struct Prepared {
    /// The network's input: RAS axes, C-order.
    pub data: Vec<f32>,
    pub dims: [usize; 3],
    /// The scan in its own axis order (`[k, j, i]` of `Volume::data`).
    native: pre::Oriented,
    /// The 1.5 mm grid, and the foreground crop on it.
    rdims: [usize; 3],
    lo: [usize; 3],
    hi: [usize; 3],
    cdims: [usize; 3],
    /// Native crop axes → RAS axes.
    perm: [usize; 3],
    flip: [bool; 3],
}

/// How a prepared volume's intensities are scaled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scaling {
    /// NV-Segment-CT: HU -963.8..1053.7 onto [0, 1], clipped.
    HuWindow,
    /// NV-Segment-CTMR: the image's own 1st..99th percentiles onto [0, 1],
    /// clipped (`ScaleIntensityRangePercentiles`).
    Percentiles,
}

/// numpy's `percentile` (linear interpolation) of `data` at `q` percent.
pub fn percentile(data: &[f32], q: f64) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let mut v: Vec<f32> = data.to_vec();
    let pos = q / 100.0 * (v.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(v.len() - 1);
    let (_, &mut a, rest) = v.select_nth_unstable_by(lo, |x, y| x.total_cmp(y));
    let b = if hi == lo {
        a
    } else {
        rest.iter().copied().fold(f32::INFINITY, f32::min)
    };
    let t = pos - lo as f64;
    (a as f64 + (b as f64 - a as f64) * t) as f32
}

impl Prepared {
    /// The bundle's preprocessing: resampled to 1.5 mm in the scan's own
    /// axis order, cropped to HU > 0 with a 10-voxel margin, scaled, RAS.
    pub fn new(volume: &Volume) -> Prepared {
        Self::with_scaling(volume, Scaling::HuWindow)
    }

    /// [`Self::new`] with either training's intensity scaling.
    pub fn with_scaling(volume: &Volume, scaling: Scaling) -> Prepared {
        let native = pre::Oriented {
            perm: [2, 1, 0],
            flip: [false; 3],
            dims: [volume.dims[2], volume.dims[1], volume.dims[0]],
            spacing: [volume.spacing[2], volume.spacing[1], volume.spacing[0]],
        };
        let raw = native.read(volume);
        let (res, rdims) = pre::resample_linear(&raw, native.dims, native.spacing, [1.5; 3]);
        drop(raw);
        // CropForeground(margin=10, allow_smaller=True) on HU > 0.
        let (lo, hi) = {
            let (lo, hi) = pre::foreground_box(&res, rdims);
            if hi == rdims && lo == [0; 3] {
                (lo, hi)
            } else {
                (
                    std::array::from_fn(|a| lo[a].saturating_sub(10)),
                    std::array::from_fn(|a| (hi[a] + 10).min(rdims[a])),
                )
            }
        };
        let mut cropped = pre::crop(&res, rdims, lo, hi);
        let cdims: [usize; 3] = std::array::from_fn(|a| hi[a] - lo[a]);
        drop(res);
        match scaling {
            Scaling::HuWindow => {
                pre::scale_range_clip(&mut cropped, -963.824_77, 1_053.678_5, 0.0, 1.0)
            }
            Scaling::Percentiles => {
                let lo = percentile(&cropped, 1.0);
                let hi = percentile(&cropped, 99.0);
                if hi > lo {
                    pre::scale_range_clip(&mut cropped, lo, hi, 0.0, 1.0);
                } else {
                    // MONAI: a flat image is only shifted.
                    cropped.iter_mut().for_each(|v| *v -= lo);
                }
            }
        }
        // Orientation(RAS): the permutation of the native C-order axes.
        let (vperm, vflip) = volume.axes_toward(AxisOrder::Ras);
        let perm: [usize; 3] = std::array::from_fn(|a| 2 - vperm[a]);
        let (data, dims) = permute(&cropped, cdims, perm, vflip);
        Prepared {
            data,
            dims,
            native,
            rdims,
            lo,
            hi,
            cdims,
            perm,
            flip: vflip,
        }
    }

    /// A voxel position of the volume (`[x, y, z]`, fractional) in the
    /// prepared array's coordinates: the same point after the resampling,
    /// the crop and the reorientation (the bundle's evaluator maps clicks
    /// through the same affines).
    pub fn point(&self, v: [f64; 3]) -> [f32; 3] {
        let nat = [v[2], v[1], v[0]];
        let c: [f64; 3] =
            std::array::from_fn(|a| nat[a] * self.native.spacing[a] / 1.5 - self.lo[a] as f64);
        std::array::from_fn(|a| {
            let x = c[self.perm[a]];
            (if self.flip[a] {
                (self.cdims[self.perm[a]] - 1) as f64 - x
            } else {
                x
            }) as f32
        })
    }

    /// Labels on the prepared array back onto the volume's own grid.
    pub fn to_volume(&self, ids: &[u8], volume: &Volume) -> Vec<u8> {
        let native_order = unpermute(ids, self.cdims, self.perm, self.flip);
        let full = pre::uncrop(&native_order, self.rdims, self.lo, self.hi);
        let labels_native = pre::resample_nearest(
            &full,
            self.rdims,
            [1.5; 3],
            self.native.dims,
            self.native.spacing,
        );
        self.native.write_back(&labels_native, volume)
    }
}

/// Run VISTA-3D's automatic mode on a CT volume. Blocking; observe and
/// cancel through `progress`. `root` is the model folder.
pub fn run(
    volume: &Volume,
    which: Vista3d,
    device: DevicePref,
    root: &Path,
    progress: &Progress,
) -> Result<AutosegResult> {
    let t0 = std::time::Instant::now();
    progress.set_phase(0.0, 0.1);
    let params = load_weights(which.weights(), root, progress)?;
    if progress.cancelled() {
        bail!(CANCELLED);
    }

    // ---- the bundle's preprocessing ------------------------------------------
    progress.set_phase(0.1, 0.05);
    progress.report(0.0, "Preparing the volume");
    let scaling = match which.weights() {
        Weights::Ct => Scaling::HuWindow,
        Weights::CtMr => Scaling::Percentiles,
    };
    let prep = Prepared::with_scaling(volume, scaling);
    if progress.cancelled() {
        bail!(CANCELLED);
    }

    // ---- the network ----------------------------------------------------------
    progress.set_phase(0.15, 0.8);
    progress.set("Choosing the compute device");
    let prompts = which.prompts();
    let patch = which.patch();
    let gpu = device.resolve()?;
    let (data, dims) = (&prep.data, prep.dims);
    let (idx, device_desc) = match gpu {
        #[cfg(feature = "gpu")]
        Some(ctx) => {
            let desc = ctx.describe();
            progress.set_device(&desc);
            let l = crate::nn::device::guarded(|| {
                windows::<crate::medsam2::engine::Gpu>(
                    &params,
                    &prompts,
                    patch,
                    ctx.device(),
                    data,
                    dims,
                    progress,
                )
            })?;
            (l, desc)
        }
        #[cfg(not(feature = "gpu"))]
        Some(ctx) => ctx.unreachable(),
        None => {
            let desc = crate::nn::device::describe_cpu();
            progress.set_device(&desc);
            let dev = Default::default();
            let l = windows::<crate::medsam2::engine::Cpu>(
                &params, &prompts, patch, &dev, data, dims, progress,
            )?;
            (l, desc)
        }
    };

    // ---- back onto the scan ----------------------------------------------------
    progress.set_phase(0.95, 0.05);
    progress.report(0.0, "Mapping labels back to the scan grid");
    // Channel index → output label.
    let out = which.output_labels();
    let ids: Vec<u8> = idx
        .iter()
        .map(|&i| if i == 0 { 0 } else { out[i as usize - 1] })
        .collect();
    let labels = prep.to_volume(&ids, volume);
    let classes = which.classes();
    let organs = organ_hits(&labels, volume.spacing, classes);
    progress.report(1.0, "Segmentation finished");
    Ok(AutosegResult {
        labels,
        dims: volume.dims,
        organs,
        model: which.key().to_string(),
        model_label: which.label().to_string(),
        classes,
        device: device_desc,
        elapsed_secs: t0.elapsed().as_secs_f64(),
        frame_of_reference_uid: volume.frame_of_reference_uid.clone(),
        volume_dims: volume.dims,
        notes: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompts_follow_the_bundle() {
        assert_eq!(Vista3d::Everything.prompts().len(), 117);
        assert!(!Vista3d::Everything.prompts().contains(&2));
        assert_eq!(CLASSES[0], "liver");
        assert_eq!(CLASSES[131], "airway");
        for l in Vista3d::Lesions.prompts() {
            let n = CLASSES[l as usize - 1];
            assert!(
                n.contains("tumor")
                    || n.contains("lesion")
                    || n.contains("mass")
                    || n.contains("cancer"),
                "{n}"
            );
        }
    }

    #[test]
    fn ctmr_rows_number_their_classes_in_order() {
        assert_eq!(Vista3d::CtmrCt.prompts(), Vista3d::Everything.prompts());
        assert_eq!(Vista3d::CtmrMr.prompts().len(), 50);
        assert_eq!(Vista3d::CtmrBrain.prompts().len(), 132);
        for v in [Vista3d::CtmrCt, Vista3d::CtmrMr, Vista3d::CtmrBrain] {
            let out = v.output_labels();
            assert_eq!(out.len(), v.prompts().len());
            assert_eq!(out[0], 1);
            assert_eq!(v.classes().len(), out.len());
            assert_eq!(v.weights(), Weights::CtMr);
        }
        assert_eq!(Vista3d::CtmrBrain.classes()[0], "3rd-Ventricle");
        assert_eq!(Vista3d::CtmrMr.classes()[0], "liver");
        assert_eq!(Vista3d::Everything.output_labels()[0], 1);
        assert_eq!(Vista3d::Everything.classes().len(), 132);
        assert_eq!(CTMR_CLASSES[344], "Left-TTG---transverse-temporal-gyrus");
    }

    #[test]
    fn percentiles_interpolate_like_numpy() {
        // np.percentile(np.arange(11.0), [1, 99, 50]) == [0.1, 9.9, 5.0]
        let v: Vec<f32> = (0..=10).rev().map(|i| i as f32).collect();
        assert!((percentile(&v, 1.0) - 0.1).abs() < 1e-6);
        assert!((percentile(&v, 99.0) - 9.9).abs() < 1e-5);
        assert_eq!(percentile(&v, 50.0), 5.0);
        assert_eq!(percentile(&[3.0], 99.0), 3.0);
    }

    #[test]
    fn the_automatic_mode_matches_monai() {
        type B = burn::backend::NdArray;
        let io = crate::unet2d::net::tests::fixture("vista");
        // The fixture keeps the Hugging Face export's `network.` prefix.
        let weights = crate::unet2d::net::tests::fixture("vista.network");
        let dev = Default::default();
        let prompts: Vec<u16> = io
            .get("__prompts", &[2])
            .unwrap()
            .iter()
            .map(|v| *v as u16)
            .collect();
        let net = Vista3dNet::<B>::load_with(&weights, &prompts, &[1, 1, 1, 1, 1], &dev).unwrap();
        let x = io.get("__input", &[1, 1, 32, 32, 16]).unwrap();
        let want = io.get("__output", &[1, 2, 32, 32, 16]).unwrap();
        let got = net.window(x, [32, 32, 16]);
        // The engine prepends the zero background logit.
        let n = 32 * 32 * 16;
        assert!(got[..n].iter().all(|v| *v == 0.0));
        let w = crate::unet2d::net::tests::worst(&got[n..], want);
        assert!(w < 1e-4, "relative error {w:e}");
    }

    #[test]
    fn permutations_round_trip() {
        let dims = [2, 3, 4];
        let data: Vec<u16> = (0..24).collect();
        for (perm, flip) in [
            ([2, 1, 0], [false, true, true]),
            ([0, 2, 1], [true, false, false]),
        ] {
            let (p, pd) = permute(&data, dims, perm, flip);
            assert_eq!(pd, [dims[perm[0]], dims[perm[1]], dims[perm[2]]]);
            assert_eq!(unpermute(&p, dims, perm, flip), data);
        }
    }
}
