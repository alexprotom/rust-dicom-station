//! The open TotalSegmentator tasks beyond `total`, as data.
//!
//! **Generated, not hand-written.** Every number and name below comes from
//! TotalSegmentator's own tables at upstream commit `2af6a1e2f17c`
//! (`totalsegmentator/map_tasks_config.py`: dataset ids, trainer folders,
//! configurations, crop rules; `totalsegmentator/map_to_binary.py`: the
//! class maps), plus the size of each release zip as GitHub served it on
//! 2026-10-06. Only the display names, groups and one-line descriptions
//! are this project's. Regenerating means re-reading those two files; the
//! generator is not part of the repository (it is a reading of Python data,
//! and the repository holds no Python).
//!
//! The licensed tasks (`LICENSED_TASKS`) come from the same tables; their
//! weights are served by the TotalSegmentator licence server for the user's
//! licence number, and their download sizes are estimates (the server
//! publishes none).
//!
//! Crop rule, as upstream applies it (`python_api.totalsegmentator`): a task
//! with crop classes first runs a coarse model on the whole scan - the 6 mm
//! `total` model, the 3 mm one where the task asks for a robust crop, the
//! 3 mm `total_mr` model for MR tasks, the 6 mm body model when it crops to
//! `body_trunc`, or the task it names - takes the bounding box of those
//! classes, and widens it by 20 mm on every side. Upstream's configured
//! `crop_addon` is only honoured for a task with its own crop model
//! (teeth); for every other task the 20 mm default replaces it, which is
//! what TotalSegmentator does and therefore what this table records (the
//! configured value is kept in a comment).

use super::task::{Crop, CropBy, FoldUse, Licence, Modality, NnTask, Part, Post};
use super::weights::{Home, ModelSpec, Source};
use crate::volume::AxisOrder;

// ---- lung_vessels ----------------------------------------------------------
pub const CLASSES_LUNG_VESSELS: [&str; 4] = [
    "lung_airways",
    "lung_airways_wall",
    "lung_arteries",
    "lung_veins",
];

pub const SPEC_LUNG_VESSELS: ModelSpec = ModelSpec {
    key: "lung_vessels",
    label: "Lung airways and vessels",
    detail: "Airways, airway walls, pulmonary arteries and veins inside the lungs.",
    url: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.5.0-weights/Dataset117_lung_airways_arteries_veins_282subj.zip",
    zip_bytes: 229_853_539,
    plans: "nnUNetPlans",
    configuration: "3d_fullres",
    folds: 1,
    axes: AxisOrder::Sar,
    home: Home::TotalSegmentator,
    source: Source::Release,
};

const LUT_LUNG_VESSELS: [u8; 5] = [0, 1, 2, 3, 4];

// ---- lung_nodules ----------------------------------------------------------
pub const CLASSES_LUNG_NODULES: [&str; 2] = ["lung", "lung_nodules"];

pub const SPEC_LUNG_NODULES: ModelSpec = ModelSpec {
    key: "lung_nodules",
    label: "Lung nodules",
    detail: "The lungs and the nodules in them (BLUEMIND AI, partly LIDC-IDRI); residual-encoder network.",
    url: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.5.0-weights/Dataset913_lung_nodules.zip",
    zip_bytes: 765_102_050,
    plans: "nnUNetPlans",
    configuration: "3d_fullres",
    folds: 1,
    axes: AxisOrder::Sar,
    home: Home::TotalSegmentator,
    source: Source::Release,
};

const LUT_LUNG_NODULES: [u8; 3] = [0, 1, 2];

// ---- pleural_pericard_effusion ---------------------------------------------
pub const CLASSES_PLEURAL_PERICARD_EFFUSION: [&str; 3] =
    ["lung_pleural", "pleural_effusion", "pericardial_effusion"];

pub const SPEC_PLEURAL_PERICARD_EFFUSION: ModelSpec = ModelSpec {
    key: "pleural_pericard_effusion",
    label: "Pleural and pericardial effusion",
    detail: "Pleura, pleural effusion and pericardial effusion.",
    url: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.0.0-weights/Dataset315_thoraxCT.zip",
    zip_bytes: 233_744_954,
    plans: "nnUNetPlans",
    configuration: "3d_fullres",
    folds: 1,
    axes: AxisOrder::Sar,
    home: Home::TotalSegmentator,
    source: Source::Release,
};

const LUT_PLEURAL_PERICARD_EFFUSION: [u8; 4] = [0, 1, 2, 3];

// ---- trunk_cavities --------------------------------------------------------
pub const CLASSES_TRUNK_CAVITIES: [&str; 4] = [
    "abdominal_cavity",
    "thoracic_cavity",
    "pericardium",
    "mediastinum",
];

pub const SPEC_TRUNK_CAVITIES: ModelSpec = ModelSpec {
    key: "trunk_cavities",
    label: "Trunk cavities",
    detail: "Abdominal and thoracic cavity, pericardium, mediastinum.",
    url: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.5.0-weights/Dataset343_mediastinum_1786subj.zip",
    zip_bytes: 233_215_432,
    plans: "nnUNetPlans",
    configuration: "3d_fullres",
    folds: 1,
    axes: AxisOrder::Sar,
    home: Home::TotalSegmentator,
    source: Source::Release,
};

const LUT_TRUNK_CAVITIES: [u8; 5] = [0, 1, 2, 3, 4];

// ---- breasts ---------------------------------------------------------------
pub const CLASSES_BREASTS: [&str; 1] = ["breast"];

pub const SPEC_BREASTS: ModelSpec = ModelSpec {
    key: "breasts",
    label: "Breasts",
    detail: "Breast tissue.",
    url: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.5.0-weights/Dataset527_breasts_1559subj.zip",
    zip_bytes: 233_756_114,
    plans: "nnUNetPlans",
    configuration: "3d_fullres",
    folds: 1,
    axes: AxisOrder::Sar,
    home: Home::TotalSegmentator,
    source: Source::Release,
};

const LUT_BREASTS: [u8; 2] = [0, 1];

// ---- liver_vessels ---------------------------------------------------------
pub const CLASSES_LIVER_VESSELS: [&str; 2] = ["liver_vessels", "liver_tumor"];

pub const SPEC_LIVER_VESSELS: ModelSpec = ModelSpec {
    key: "liver_vessels",
    label: "Liver vessels",
    detail: "Hepatic vessels and liver tumour (MSD Task 8).",
    url: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.4.0-weights/Dataset008_HepaticVessel.zip",
    zip_bytes: 230_191_278,
    plans: "nnUNetPlans",
    configuration: "3d_fullres",
    folds: 1,
    axes: AxisOrder::Sar,
    home: Home::TotalSegmentator,
    source: Source::Release,
};

const LUT_LIVER_VESSELS: [u8; 3] = [0, 1, 2];

// ---- liver_segments --------------------------------------------------------
pub const CLASSES_LIVER_SEGMENTS: [&str; 8] = [
    "liver_segment_1",
    "liver_segment_2",
    "liver_segment_3",
    "liver_segment_4",
    "liver_segment_5",
    "liver_segment_6",
    "liver_segment_7",
    "liver_segment_8",
];

pub const SPEC_LIVER_SEGMENTS: ModelSpec = ModelSpec {
    key: "liver_segments",
    label: "Liver segments (Couinaud)",
    detail: "The eight Couinaud segments of the liver.",
    url: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.5.0-weights/Dataset570_ct_liver_segments.zip",
    zip_bytes: 230_049_561,
    plans: "nnUNetPlans",
    configuration: "3d_fullres",
    folds: 1,
    axes: AxisOrder::Sar,
    home: Home::TotalSegmentator,
    source: Source::Release,
};

const LUT_LIVER_SEGMENTS: [u8; 9] = [0, 1, 2, 3, 4, 5, 6, 7, 8];

// ---- liver_lesions ---------------------------------------------------------
pub const CLASSES_LIVER_LESIONS: [&str; 1] = ["liver_lesions"];

pub const SPEC_LIVER_LESIONS: ModelSpec = ModelSpec {
    key: "liver_lesions",
    label: "Liver lesions",
    detail: "Focal liver lesions.",
    url: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.5.0-weights/Dataset591_ct_liver_lesions_842subj.zip",
    zip_bytes: 230_940_114,
    plans: "nnUNetPlans",
    configuration: "3d_fullres_high",
    folds: 1,
    axes: AxisOrder::Sar,
    home: Home::TotalSegmentator,
    source: Source::Release,
};

const LUT_LIVER_LESIONS: [u8; 2] = [0, 1];

// ---- kidney_cysts ----------------------------------------------------------
pub const CLASSES_KIDNEY_CYSTS: [&str; 2] = ["kidney_cyst_left", "kidney_cyst_right"];

pub const SPEC_KIDNEY_CYSTS: ModelSpec = ModelSpec {
    key: "kidney_cysts",
    label: "Kidney cysts",
    detail: "Cysts of the left and right kidney.",
    url: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.5.0-weights/Dataset789_kidney_cyst_501subj.zip",
    zip_bytes: 230_592_397,
    plans: "nnUNetPlans",
    configuration: "3d_fullres",
    folds: 1,
    axes: AxisOrder::Sar,
    home: Home::TotalSegmentator,
    source: Source::Release,
};

const LUT_KIDNEY_CYSTS: [u8; 5] = [0, 1, 2, 0, 0];

// ---- abdominal_muscles -----------------------------------------------------
pub const CLASSES_ABDOMINAL_MUSCLES: [&str; 22] = [
    "pectoralis_major_right",
    "pectoralis_major_left",
    "rectus_abdominis_right",
    "rectus_abdominis_left",
    "serratus_anterior_right",
    "serratus_anterior_left",
    "latissimus_dorsi_right",
    "latissimus_dorsi_left",
    "trapezius_right",
    "trapezius_left",
    "external_oblique_right",
    "external_oblique_left",
    "internal_oblique_right",
    "internal_oblique_left",
    "erector_spinae_right",
    "erector_spinae_left",
    "transversospinalis_right",
    "transversospinalis_left",
    "psoas_major_right",
    "psoas_major_left",
    "quadratus_lumborum_right",
    "quadratus_lumborum_left",
];

pub const SPEC_ABDOMINAL_MUSCLES: ModelSpec = ModelSpec {
    key: "abdominal_muscles",
    label: "Abdominal muscles",
    detail: "Abdominal wall and back muscles between T4 and L4.",
    url: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.5.0-weights/Dataset952_abdominal_muscles_167subj.zip",
    zip_bytes: 124_276_110,
    plans: "nnUNetPlans",
    configuration: "3d_fullres_high",
    folds: 1,
    axes: AxisOrder::Sar,
    home: Home::TotalSegmentator,
    source: Source::Release,
};

const LUT_ABDOMINAL_MUSCLES: [u8; 23] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
];

// ---- head_glands_cavities --------------------------------------------------
pub const CLASSES_HEAD_GLANDS_CAVITIES: [&str; 19] = [
    "eye_left",
    "eye_right",
    "eye_lens_left",
    "eye_lens_right",
    "optic_nerve_left",
    "optic_nerve_right",
    "parotid_gland_left",
    "parotid_gland_right",
    "submandibular_gland_right",
    "submandibular_gland_left",
    "nasopharynx",
    "oropharynx",
    "hypopharynx",
    "nasal_cavity_right",
    "nasal_cavity_left",
    "auditory_canal_right",
    "auditory_canal_left",
    "soft_palate",
    "hard_palate",
];

pub const SPEC_HEAD_GLANDS_CAVITIES: ModelSpec = ModelSpec {
    key: "head_glands_cavities",
    label: "Head glands and cavities",
    detail: "Eyes, lenses, optic nerves, salivary glands, sinuses, auditory canals.",
    url: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.3.0-weights/Dataset775_head_glands_cavities_492subj.zip",
    zip_bytes: 231_091_065,
    plans: "nnUNetPlans",
    configuration: "3d_fullres_high",
    folds: 1,
    axes: AxisOrder::Sar,
    home: Home::TotalSegmentator,
    source: Source::Release,
};

const LUT_HEAD_GLANDS_CAVITIES: [u8; 20] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
];

// ---- headneck_bones_vessels ------------------------------------------------
pub const CLASSES_HEADNECK_BONES_VESSELS: [&str; 12] = [
    "larynx_air",
    "thyroid_cartilage",
    "hyoid",
    "cricoid_cartilage",
    "zygomatic_arch_right",
    "zygomatic_arch_left",
    "styloid_process_right",
    "styloid_process_left",
    "internal_carotid_artery_right",
    "internal_carotid_artery_left",
    "internal_jugular_vein_right",
    "internal_jugular_vein_left",
];

pub const SPEC_HEADNECK_BONES_VESSELS: ModelSpec = ModelSpec {
    key: "headneck_bones_vessels",
    label: "Head and neck bones and vessels",
    detail: "Larynx cartilages, hyoid, carotid and vertebral arteries, jugular veins.",
    url: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.3.0-weights/Dataset776_headneck_bones_vessels_492subj.zip",
    zip_bytes: 230_805_136,
    plans: "nnUNetPlans",
    configuration: "3d_fullres_high",
    folds: 1,
    axes: AxisOrder::Sar,
    home: Home::TotalSegmentator,
    source: Source::Release,
};

const LUT_HEADNECK_BONES_VESSELS: [u8; 13] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

// ---- head_muscles ----------------------------------------------------------
pub const CLASSES_HEAD_MUSCLES: [&str; 11] = [
    "masseter_right",
    "masseter_left",
    "temporalis_right",
    "temporalis_left",
    "lateral_pterygoid_right",
    "lateral_pterygoid_left",
    "medial_pterygoid_right",
    "medial_pterygoid_left",
    "tongue",
    "digastric_right",
    "digastric_left",
];

pub const SPEC_HEAD_MUSCLES: ModelSpec = ModelSpec {
    key: "head_muscles",
    label: "Head muscles",
    detail: "Masticatory and facial muscles, tongue.",
    url: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.3.0-weights/Dataset777_head_muscles_492subj.zip",
    zip_bytes: 230_370_685,
    plans: "nnUNetPlans",
    configuration: "3d_fullres_high",
    folds: 1,
    axes: AxisOrder::Sar,
    home: Home::TotalSegmentator,
    source: Source::Release,
};

const LUT_HEAD_MUSCLES: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

// ---- headneck_muscles ------------------------------------------------------
pub const CLASSES_HEADNECK_MUSCLES: [&str; 23] = [
    "sternocleidomastoid_right",
    "sternocleidomastoid_left",
    "superior_pharyngeal_constrictor",
    "middle_pharyngeal_constrictor",
    "inferior_pharyngeal_constrictor",
    "trapezius_right",
    "trapezius_left",
    "platysma_right",
    "platysma_left",
    "levator_scapulae_right",
    "levator_scapulae_left",
    "anterior_scalene_right",
    "anterior_scalene_left",
    "middle_scalene_right",
    "middle_scalene_left",
    "posterior_scalene_right",
    "posterior_scalene_left",
    "sterno_thyroid_right",
    "sterno_thyroid_left",
    "thyrohyoid_right",
    "thyrohyoid_left",
    "prevertebral_right",
    "prevertebral_left",
];

pub const SPEC_HEADNECK_MUSCLES_PART1: ModelSpec = ModelSpec {
    key: "headneck_muscles_part1",
    label: "Head and neck muscles (1/2)",
    detail: "Neck muscles and pharyngeal constrictors (two sub-models).",
    url: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.3.0-weights/Dataset778_headneck_muscles_part1_492subj.zip",
    zip_bytes: 230_197_164,
    plans: "nnUNetPlans",
    configuration: "3d_fullres_high",
    folds: 1,
    axes: AxisOrder::Sar,
    home: Home::TotalSegmentator,
    source: Source::Release,
};

const LUT_HEADNECK_MUSCLES_PART1: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

pub const SPEC_HEADNECK_MUSCLES_PART2: ModelSpec = ModelSpec {
    key: "headneck_muscles_part2",
    label: "Head and neck muscles (2/2)",
    detail: "Neck muscles and pharyngeal constrictors (two sub-models).",
    url: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.3.0-weights/Dataset779_headneck_muscles_part2_492subj.zip",
    zip_bytes: 230_397_368,
    plans: "nnUNetPlans",
    configuration: "3d_fullres_high",
    folds: 1,
    axes: AxisOrder::Sar,
    home: Home::TotalSegmentator,
    source: Source::Release,
};

const LUT_HEADNECK_MUSCLES_PART2: [u8; 13] = [0, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23];

// ---- oculomotor_muscles ----------------------------------------------------
pub const CLASSES_OCULOMOTOR_MUSCLES: [&str; 19] = [
    "skull",
    "eyeball_right",
    "lateral_rectus_muscle_right",
    "superior_oblique_muscle_right",
    "levator_palpebrae_superioris_right",
    "superior_rectus_muscle_right",
    "medial_rectus_muscle_left",
    "inferior_oblique_muscle_right",
    "inferior_rectus_muscle_right",
    "optic_nerve_left",
    "eyeball_left",
    "lateral_rectus_muscle_left",
    "superior_oblique_muscle_left",
    "levator_palpebrae_superioris_left",
    "superior_rectus_muscle_left",
    "medial_rectus_muscle_right",
    "inferior_oblique_muscle_left",
    "inferior_rectus_muscle_left",
    "optic_nerve_right",
];

pub const SPEC_OCULOMOTOR_MUSCLES: ModelSpec = ModelSpec {
    key: "oculomotor_muscles",
    label: "Oculomotor muscles",
    detail: "Extra-ocular muscles, eyeballs, optic nerves (18 training cases).",
    url: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.4.0-weights/Dataset351_oculomotor_muscles_18subj.zip",
    zip_bytes: 230_606_852,
    plans: "nnUNetPlans",
    configuration: "3d_fullres",
    folds: 1,
    axes: AxisOrder::Sar,
    home: Home::TotalSegmentator,
    source: Source::Release,
};

const LUT_OCULOMOTOR_MUSCLES: [u8; 20] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
];

// ---- craniofacial_structures -----------------------------------------------
pub const CLASSES_CRANIOFACIAL_STRUCTURES: [&str; 7] = [
    "mandible",
    "teeth_lower",
    "skull",
    "head",
    "sinus_maxillary",
    "sinus_frontal",
    "teeth_upper",
];

pub const SPEC_CRANIOFACIAL_STRUCTURES: ModelSpec = ModelSpec {
    key: "craniofacial_structures",
    label: "Craniofacial structures",
    detail: "Mandible, teeth (upper and lower), skull, head, sinuses.",
    url: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.5.0-weights/Dataset115_mandible.zip",
    zip_bytes: 230_321_497,
    plans: "nnUNetPlans",
    configuration: "3d_fullres",
    folds: 1,
    axes: AxisOrder::Sar,
    home: Home::TotalSegmentator,
    source: Source::Release,
};

const LUT_CRANIOFACIAL_STRUCTURES: [u8; 8] = [0, 1, 2, 3, 4, 5, 6, 7];

// ---- teeth -----------------------------------------------------------------
pub const CLASSES_TEETH: [&str; 77] = [
    "lower_jawbone",
    "upper_jawbone",
    "left_inferior_alveolar_canal",
    "right_inferior_alveolar_canal",
    "left_maxillary_sinus",
    "right_maxillary_sinus",
    "pharynx",
    "bridge",
    "crown",
    "implant",
    "upper_right_central_incisor_fdi11",
    "upper_right_lateral_incisor_fdi12",
    "upper_right_canine_fdi13",
    "upper_right_first_premolar_fdi14",
    "upper_right_second_premolar_fdi15",
    "upper_right_first_molar_fdi16",
    "upper_right_second_molar_fdi17",
    "upper_right_third_molar_fdi18",
    "upper_left_central_incisor_fdi21",
    "upper_left_lateral_incisor_fdi22",
    "upper_left_canine_fdi23",
    "upper_left_first_premolar_fdi24",
    "upper_left_second_premolar_fdi25",
    "upper_left_first_molar_fdi26",
    "upper_left_second_molar_fdi27",
    "upper_left_third_molar_fdi28",
    "lower_left_central_incisor_fdi31",
    "lower_left_lateral_incisor_fdi32",
    "lower_left_canine_fdi33",
    "lower_left_first_premolar_fdi34",
    "lower_left_second_premolar_fdi35",
    "lower_left_first_molar_fdi36",
    "lower_left_second_molar_fdi37",
    "lower_left_third_molar_fdi38",
    "lower_right_central_incisor_fdi41",
    "lower_right_lateral_incisor_fdi42",
    "lower_right_canine_fdi43",
    "lower_right_first_premolar_fdi44",
    "lower_right_second_premolar_fdi45",
    "lower_right_first_molar_fdi46",
    "lower_right_second_molar_fdi47",
    "lower_right_third_molar_fdi48",
    "left_mandibular_incisive_canal_fdi103",
    "right_mandibular_incisive_canal_fdi104",
    "lingual_canal",
    "upper_right_central_incisor_pulp_fdi111",
    "upper_right_lateral_incisor_pulp_fdi112",
    "upper_right_canine_pulp_fdi113",
    "upper_right_first_premolar_pulp_fdi114",
    "upper_right_second_premolar_pulp_fdi115",
    "upper_right_first_molar_pulp_fdi116",
    "upper_right_second_molar_pulp_fdi117",
    "upper_right_third_molar_pulp_fdi118",
    "upper_left_central_incisor_pulp_fdi121",
    "upper_left_lateral_incisor_pulp_fdi122",
    "upper_left_canine_pulp_fdi123",
    "upper_left_first_premolar_pulp_fdi124",
    "upper_left_second_premolar_pulp_fdi125",
    "upper_left_first_molar_pulp_fdi126",
    "upper_left_second_molar_pulp_fdi127",
    "upper_left_third_molar_pulp_fdi128",
    "lower_left_central_incisor_pulp_fdi131",
    "lower_left_lateral_incisor_pulp_fdi132",
    "lower_left_canine_pulp_fdi133",
    "lower_left_first_premolar_pulp_fdi134",
    "lower_left_second_premolar_pulp_fdi135",
    "lower_left_first_molar_pulp_fdi136",
    "lower_left_second_molar_pulp_fdi137",
    "lower_left_third_molar_pulp_fdi138",
    "lower_right_central_incisor_pulp_fdi141",
    "lower_right_lateral_incisor_pulp_fdi142",
    "lower_right_canine_pulp_fdi143",
    "lower_right_first_premolar_pulp_fdi144",
    "lower_right_second_premolar_pulp_fdi145",
    "lower_right_first_molar_pulp_fdi146",
    "lower_right_second_molar_pulp_fdi147",
    "lower_right_third_molar_pulp_fdi148",
];

pub const SPEC_TEETH: ModelSpec = ModelSpec {
    key: "teeth",
    label: "Teeth (FDI) and jaw",
    detail: "Every tooth by FDI number, jaws, canals, pulp (ToothFairy3); cropped to the teeth found by Craniofacial structures.",
    url: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.5.0-weights/Dataset113_ToothFairy3.zip",
    zip_bytes: 232_066_830,
    plans: "nnUNetPlans",
    configuration: "3d_lowres_high",
    folds: 1,
    axes: AxisOrder::Sar,
    home: Home::TotalSegmentator,
    source: Source::Release,
};

const LUT_TEETH: [u8; 78] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
    50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73,
    74, 75, 76, 77,
];

// ---- cerebral_bleed --------------------------------------------------------
pub const CLASSES_CEREBRAL_BLEED: [&str; 1] = ["intracerebral_hemorrhage"];

pub const SPEC_CEREBRAL_BLEED: ModelSpec = ModelSpec {
    key: "cerebral_bleed",
    label: "Intracerebral haemorrhage",
    detail: "Intracerebral haemorrhage on non-contrast head CT.",
    url: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.0.0-weights/Dataset150_icb_v0.zip",
    zip_bytes: 325_569_459,
    plans: "nnUNetPlans",
    configuration: "3d_fullres",
    folds: 1,
    axes: AxisOrder::Sar,
    home: Home::TotalSegmentator,
    source: Source::Release,
};

const LUT_CEREBRAL_BLEED: [u8; 2] = [0, 1];

// ---- ventricle_parts -------------------------------------------------------
pub const CLASSES_VENTRICLE_PARTS: [&str; 12] = [
    "ventricle_frontal_horn_left",
    "ventricle_occipital_horn_left",
    "ventricle_body_left",
    "ventricle_temporal_horn_left",
    "ventricle_trigone_left",
    "ventricle_frontal_horn_right",
    "ventricle_occipital_horn_right",
    "ventricle_body_right",
    "ventricle_temporal_horn_right",
    "ventricle_trigone_right",
    "third_ventricle",
    "fourth_ventricle",
];

pub const SPEC_VENTRICLE_PARTS: ModelSpec = ModelSpec {
    key: "ventricle_parts",
    label: "Brain ventricle parts",
    detail: "The parts of the ventricular system (38 training cases).",
    url: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.5.0-weights/Dataset552_ventricle_parts_38subj.zip",
    zip_bytes: 232_836_551,
    plans: "nnUNetPlans",
    configuration: "3d_fullres",
    folds: 1,
    axes: AxisOrder::Sar,
    home: Home::TotalSegmentator,
    source: Source::Release,
};

const LUT_VENTRICLE_PARTS: [u8; 13] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

// ---- hip_implant -----------------------------------------------------------
pub const CLASSES_HIP_IMPLANT: [&str; 1] = ["hip_implant"];

pub const SPEC_HIP_IMPLANT: ModelSpec = ModelSpec {
    key: "hip_implant",
    label: "Hip implant",
    detail: "Metal hip prostheses.",
    url: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.0.0-weights/Dataset260_hip_implant_71subj.zip",
    zip_bytes: 232_729_513,
    plans: "nnUNetPlans",
    configuration: "3d_fullres",
    folds: 1,
    axes: AxisOrder::Sar,
    home: Home::TotalSegmentator,
    source: Source::Release,
};

const LUT_HIP_IMPLANT: [u8; 2] = [0, 1];

// ---- vertebrae_body --------------------------------------------------------
pub const CLASSES_VERTEBRAE_BODY: [&str; 2] = ["vertebrae_body", "intervertebral_discs"];

pub const SPEC_VERTEBRAE_BODY: ModelSpec = ModelSpec {
    key: "vertebrae_body",
    label: "Vertebral bodies and discs",
    detail: "Vertebral bodies (without the posterior elements) and intervertebral discs.",
    url: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.5.0-weights/Dataset305_vertebrae_discs_1559subj.zip",
    zip_bytes: 233_569_643,
    plans: "nnUNetPlans",
    configuration: "3d_fullres",
    folds: 1,
    axes: AxisOrder::Sar,
    home: Home::TotalSegmentator,
    source: Source::Release,
};

const LUT_VERTEBRAE_BODY: [u8; 3] = [0, 1, 2];

// ---- vertebrae_pp ----------------------------------------------------------
pub const CLASSES_VERTEBRAE_PP: [&str; 24] = [
    "vertebrae_C1",
    "vertebrae_C2",
    "vertebrae_C3",
    "vertebrae_C4",
    "vertebrae_C5",
    "vertebrae_C6",
    "vertebrae_C7",
    "vertebrae_T1",
    "vertebrae_T2",
    "vertebrae_T3",
    "vertebrae_T4",
    "vertebrae_T5",
    "vertebrae_T6",
    "vertebrae_T7",
    "vertebrae_T8",
    "vertebrae_T9",
    "vertebrae_T10",
    "vertebrae_T11",
    "vertebrae_T12",
    "vertebrae_L1",
    "vertebrae_L2",
    "vertebrae_L3",
    "vertebrae_L4",
    "vertebrae_L5",
];

pub const SPEC_VERTEBRAE_PP: ModelSpec = ModelSpec {
    key: "vertebrae_pp",
    label: "Vertebral bodies, numbered",
    detail: "Each vertebral body by level, with the upstream relabelling of touching vertebrae.",
    url: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.5.0-weights/Dataset803_TotalSegmentator_vertebrae_inner_1559subj.zip",
    zip_bytes: 233_669_247,
    plans: "nnUNetPlans",
    configuration: "3d_fullres",
    folds: 1,
    axes: AxisOrder::Sar,
    home: Home::TotalSegmentator,
    source: Source::Release,
};

const LUT_VERTEBRAE_PP: [u8; 25] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
];

// ---- vertebrae_mr ----------------------------------------------------------
pub const CLASSES_VERTEBRAE_MR: [&str; 25] = [
    "sacrum",
    "vertebrae_L5",
    "vertebrae_L4",
    "vertebrae_L3",
    "vertebrae_L2",
    "vertebrae_L1",
    "vertebrae_T12",
    "vertebrae_T11",
    "vertebrae_T10",
    "vertebrae_T9",
    "vertebrae_T8",
    "vertebrae_T7",
    "vertebrae_T6",
    "vertebrae_T5",
    "vertebrae_T4",
    "vertebrae_T3",
    "vertebrae_T2",
    "vertebrae_T1",
    "vertebrae_C7",
    "vertebrae_C6",
    "vertebrae_C5",
    "vertebrae_C4",
    "vertebrae_C3",
    "vertebrae_C2",
    "vertebrae_C1",
];

pub const SPEC_VERTEBRAE_MR: ModelSpec = ModelSpec {
    key: "vertebrae_mr",
    label: "Vertebrae and discs (MR)",
    detail: "Every vertebra and the intervertebral discs and spinal canal on MR.",
    url: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.5.0-weights/Dataset756_mri_vertebrae_1076subj.zip",
    zip_bytes: 230_791_939,
    plans: "nnUNetPlans",
    configuration: "3d_fullres",
    folds: 1,
    axes: AxisOrder::Sar,
    home: Home::TotalSegmentator,
    source: Source::Release,
};

const LUT_VERTEBRAE_MR: [u8; 26] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
];

// ---- liver_segments_mr -----------------------------------------------------
pub const CLASSES_LIVER_SEGMENTS_MR: [&str; 8] = [
    "liver_segment_1",
    "liver_segment_2",
    "liver_segment_3",
    "liver_segment_4",
    "liver_segment_5",
    "liver_segment_6",
    "liver_segment_7",
    "liver_segment_8",
];

pub const SPEC_LIVER_SEGMENTS_MR: ModelSpec = ModelSpec {
    key: "liver_segments_mr",
    label: "Liver segments (MR)",
    detail: "The eight Couinaud segments on MR.",
    url: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.5.0-weights/Dataset576_mri_liver_segments_120subj.zip",
    zip_bytes: 229_689_699,
    plans: "nnUNetPlans",
    configuration: "3d_fullres",
    folds: 1,
    axes: AxisOrder::Sar,
    home: Home::TotalSegmentator,
    source: Source::Release,
};

const LUT_LIVER_SEGMENTS_MR: [u8; 9] = [0, 1, 2, 3, 4, 5, 6, 7, 8];

// ---- liver_lesions_mr ------------------------------------------------------
pub const CLASSES_LIVER_LESIONS_MR: [&str; 1] = ["liver_lesions"];

pub const SPEC_LIVER_LESIONS_MR: ModelSpec = ModelSpec {
    key: "liver_lesions_mr",
    label: "Liver lesions (MR)",
    detail: "Focal liver lesions on MR (trained on CT and MR).",
    url: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.5.0-weights/Dataset589_ct_mri_liver_lesions_750subj.zip",
    zip_bytes: 231_339_706,
    plans: "nnUNetPlans",
    configuration: "3d_fullres",
    folds: 1,
    axes: AxisOrder::Sar,
    home: Home::TotalSegmentator,
    source: Source::Release,
};

const LUT_LIVER_LESIONS_MR: [u8; 2] = [0, 1];

// ---- brain_aneurysm --------------------------------------------------------
pub const CLASSES_BRAIN_ANEURYSM: [&str; 1] = ["brain_aneurysm"];

pub const SPEC_BRAIN_ANEURYSM: ModelSpec = ModelSpec {
    key: "brain_aneurysm",
    label: "Brain aneurysm (TOF MRA)",
    detail: "Intracranial aneurysms on time-of-flight MR angiography (MAXIMUS, five folds).",
    url: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.5.0-weights/Dataset615_MAXIMUS.zip",
    zip_bytes: 1_168_954_978,
    plans: "nnUNetPlans",
    configuration: "3d_fullres",
    folds: 5,
    axes: AxisOrder::Sar,
    home: Home::TotalSegmentator,
    source: Source::Release,
};

const LUT_BRAIN_ANEURYSM: [u8; 2] = [0, 1];

// ---- total_mr ------------------------------------------------------------
pub const CLASSES_TOTAL_MR: [&str; 50] = [
    "spleen",
    "kidney_right",
    "kidney_left",
    "gallbladder",
    "liver",
    "stomach",
    "pancreas",
    "adrenal_gland_right",
    "adrenal_gland_left",
    "lung_left",
    "lung_right",
    "esophagus",
    "small_bowel",
    "duodenum",
    "colon",
    "urinary_bladder",
    "prostate",
    "sacrum",
    "vertebrae",
    "intervertebral_discs",
    "spinal_cord",
    "heart",
    "aorta",
    "inferior_vena_cava",
    "portal_vein_and_splenic_vein",
    "iliac_artery_left",
    "iliac_artery_right",
    "iliac_vena_left",
    "iliac_vena_right",
    "humerus_left",
    "humerus_right",
    "scapula_left",
    "scapula_right",
    "clavicula_left",
    "clavicula_right",
    "femur_left",
    "femur_right",
    "hip_left",
    "hip_right",
    "gluteus_maximus_left",
    "gluteus_maximus_right",
    "gluteus_medius_left",
    "gluteus_medius_right",
    "gluteus_minimus_left",
    "gluteus_minimus_right",
    "autochthon_left",
    "autochthon_right",
    "iliopsoas_left",
    "iliopsoas_right",
    "brain",
];

pub const SPEC_TOTAL_MR_3MM: ModelSpec = ModelSpec {
    key: "total_mr_3mm",
    label: "total MR 3 mm",
    detail: "MR: 50 organs, vessels, bones and muscles at 3 mm.",
    url: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.5.0-weights/Dataset852_TotalSegMRI_total_3mm_1088subj.zip",
    zip_bytes: 126_399_352,
    plans: "nnUNetPlans",
    configuration: "3d_fullres",
    folds: 1,
    axes: AxisOrder::Sar,
    home: Home::TotalSegmentator,
    source: Source::Release,
};

pub const SPEC_TOTAL_MR_6MM: ModelSpec = ModelSpec {
    key: "total_mr_6mm",
    label: "total MR 6 mm",
    detail: "MR: coarse preview, the quickest look.",
    url: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.5.0-weights/Dataset853_TotalSegMRI_total_6mm_1088subj.zip",
    zip_bytes: 44_657_261,
    plans: "nnUNetPlans",
    configuration: "3d_fullres",
    folds: 1,
    axes: AxisOrder::Sar,
    home: Home::TotalSegmentator,
    source: Source::Release,
};

pub const SPEC_TOTAL_MR_PART1_ORGANS: ModelSpec = ModelSpec {
    key: "total_mr_part1_organs",
    label: "total MR 1.5 mm organs (1/2)",
    detail: "MR full-resolution sub-model; the two together are the reference quality.",
    url: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.5.0-weights/Dataset850_TotalSegMRI_part1_organs_1088subj.zip",
    zip_bytes: 231_681_650,
    plans: "nnUNetPlans",
    configuration: "3d_fullres",
    folds: 1,
    axes: AxisOrder::Sar,
    home: Home::TotalSegmentator,
    source: Source::Release,
};

pub const LUT_TOTAL_MR_PART1_ORGANS: [u8; 30] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29,
];

pub const SPEC_TOTAL_MR_PART2_MUSCLES: ModelSpec = ModelSpec {
    key: "total_mr_part2_muscles",
    label: "total MR 1.5 mm muscles (2/2)",
    detail: "MR full-resolution sub-model; the two together are the reference quality.",
    url: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.5.0-weights/Dataset851_TotalSegMRI_part2_muscles_1088subj.zip",
    zip_bytes: 232_156_399,
    plans: "nnUNetPlans",
    configuration: "3d_fullres",
    folds: 1,
    axes: AxisOrder::Sar,
    home: Home::TotalSegmentator,
    source: Source::Release,
};

pub const LUT_TOTAL_MR_PART2_MUSCLES: [u8; 22] = [
    0, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
];

// ---- body_mr, 6 mm -------------------------------------------------------
pub const SPEC_BODY_MR_6MM: ModelSpec = ModelSpec {
    key: "body_mr_6mm",
    label: "body MR 6 mm",
    detail: "Patient outline on MR at 6 mm - quick, for the same decision as the 1.5 mm model.",
    url: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.5.0-weights/Dataset598_mri_body_6mm_139subj.zip",
    zip_bytes: 42_741_891,
    plans: "nnUNetPlans",
    configuration: "3d_fullres",
    folds: 1,
    axes: AxisOrder::Sar,
    home: Home::TotalSegmentator,
    source: Source::Release,
};

// ==== the licensed tasks ======================================================
// ---- heartchambers_highres -------------------------------------------------
pub const CLASSES_HEARTCHAMBERS_HIGHRES: [&str; 7] = [
    "heart_myocardium",
    "heart_atrium_left",
    "heart_ventricle_left",
    "heart_atrium_right",
    "heart_ventricle_right",
    "aorta",
    "pulmonary_artery",
];

pub const SPEC_HEARTCHAMBERS_HIGHRES: ModelSpec = ModelSpec {
    key: "heartchambers_highres",
    label: "Heart chambers, 1.5 mm",
    detail: "Myocardium, atria, ventricles, aorta and pulmonary artery at full resolution (cropped to the heart).",
    url: "",
    zip_bytes: 235_000_000,
    plans: "nnUNetPlans",
    configuration: "3d_fullres",
    folds: 1,
    axes: AxisOrder::Sar,
    home: Home::TotalSegmentator,
    source: Source::Licensed { task: "heartchambers_highres" },
};

const LUT_HEARTCHAMBERS_HIGHRES: [u8; 8] = [0, 1, 2, 3, 4, 5, 6, 7];

// ---- coronary_arteries -----------------------------------------------------
pub const CLASSES_CORONARY_ARTERIES: [&str; 1] = ["coronary_arteries"];

pub const SPEC_CORONARY_ARTERIES: ModelSpec = ModelSpec {
    key: "coronary_arteries",
    label: "Coronary arteries",
    detail: "The coronary arteries at 0.7 mm, cropped to the heart.",
    url: "",
    zip_bytes: 235_000_000,
    plans: "nnUNetPlans",
    configuration: "3d_fullres_high",
    folds: 1,
    axes: AxisOrder::Sar,
    home: Home::TotalSegmentator,
    source: Source::Licensed {
        task: "coronary_arteries",
    },
};

const LUT_CORONARY_ARTERIES: [u8; 2] = [0, 1];

// ---- aortic_sinuses --------------------------------------------------------
pub const CLASSES_AORTIC_SINUSES: [&str; 4] = [
    "left_ventricular_outflow_tract",
    "right_coronary_cusp",
    "left_coronary_cusp",
    "non_coronary_cusp",
];

pub const SPEC_AORTIC_SINUSES: ModelSpec = ModelSpec {
    key: "aortic_sinuses",
    label: "Aortic sinuses",
    detail: "Left ventricular outflow tract and the three coronary cusps at 0.7 mm.",
    url: "",
    zip_bytes: 235_000_000,
    plans: "nnUNetPlans",
    configuration: "3d_fullres_high",
    folds: 1,
    axes: AxisOrder::Sar,
    home: Home::TotalSegmentator,
    source: Source::Licensed {
        task: "aortic_sinuses",
    },
};

const LUT_AORTIC_SINUSES: [u8; 5] = [0, 1, 2, 3, 4];

// ---- aorta_annulus ---------------------------------------------------------
pub const CLASSES_AORTA_ANNULUS: [&str; 2] = ["annulus_proper", "sinotubular_junction"];

pub const SPEC_AORTA_ANNULUS: ModelSpec = ModelSpec {
    key: "aorta_annulus",
    label: "Aortic annulus",
    detail: "Aortic annulus and sinotubular junction at 0.8 mm (five folds).",
    url: "",
    zip_bytes: 1_175_000_000,
    plans: "nnUNetPlans",
    configuration: "3d_fullres_high",
    folds: 5,
    axes: AxisOrder::Sar,
    home: Home::TotalSegmentator,
    source: Source::Licensed {
        task: "aorta_annulus",
    },
};

const LUT_AORTA_ANNULUS: [u8; 3] = [0, 1, 2];

// ---- aortic_dissection -----------------------------------------------------
pub const CLASSES_AORTIC_DISSECTION: [&str; 2] = ["aorta_true_lumen", "aorta_false_lumen"];

pub const SPEC_AORTIC_DISSECTION: ModelSpec = ModelSpec {
    key: "aortic_dissection",
    label: "Aortic dissection",
    detail: "True and false lumen of a dissected aorta at 0.8 mm (five folds).",
    url: "",
    zip_bytes: 1_175_000_000,
    plans: "nnUNetPlans",
    configuration: "3d_fullres_high",
    folds: 5,
    axes: AxisOrder::Sar,
    home: Home::TotalSegmentator,
    source: Source::Licensed {
        task: "aortic_dissection",
    },
};

const LUT_AORTIC_DISSECTION: [u8; 3] = [0, 1, 2];

// ---- pulmonary_artery_landmarks --------------------------------------------
pub const CLASSES_PULMONARY_ARTERY_LANDMARKS: [&str; 7] = [
    "annulus",
    "sinotubular_junction",
    "pul_annulus",
    "pul_sinotubular_junction",
    "pul_bifurcation",
    "pul_left_start",
    "pul_right_start",
];

pub const SPEC_PULMONARY_ARTERY_LANDMARKS: ModelSpec = ModelSpec {
    key: "pulmonary_artery_landmarks",
    label: "Pulmonary artery landmarks",
    detail: "Aortic and pulmonary annulus, sinotubular junctions, pulmonary bifurcation and branches (five folds).",
    url: "",
    zip_bytes: 1_175_000_000,
    plans: "nnUNetPlans",
    configuration: "3d_fullres",
    folds: 5,
    axes: AxisOrder::Sar,
    home: Home::TotalSegmentator,
    source: Source::Licensed { task: "pulmonary_artery_landmarks" },
};

const LUT_PULMONARY_ARTERY_LANDMARKS: [u8; 8] = [0, 1, 2, 3, 4, 5, 6, 7];

// ---- renal_arteries --------------------------------------------------------
pub const CLASSES_RENAL_ARTERIES: [&str; 3] = [
    "celiac_trunk",
    "superior_mesenteric_artery",
    "renal_arteries",
];

pub const SPEC_RENAL_ARTERIES: ModelSpec = ModelSpec {
    key: "renal_arteries",
    label: "Renal arteries",
    detail: "Coeliac trunk, superior mesenteric and renal arteries.",
    url: "",
    zip_bytes: 235_000_000,
    plans: "nnUNetPlans",
    configuration: "3d_fullres",
    folds: 1,
    axes: AxisOrder::Sar,
    home: Home::TotalSegmentator,
    source: Source::Licensed {
        task: "renal_arteries",
    },
};

const LUT_RENAL_ARTERIES: [u8; 5] = [0, 1, 2, 3, 0];

// ---- tissue_types ----------------------------------------------------------
pub const CLASSES_TISSUE_TYPES: [&str; 3] = ["subcutaneous_fat", "torso_fat", "skeletal_muscle"];

pub const SPEC_TISSUE_TYPES: ModelSpec = ModelSpec {
    key: "tissue_types",
    label: "Tissue types",
    detail: "Subcutaneous fat, torso fat and skeletal muscle.",
    url: "",
    zip_bytes: 235_000_000,
    plans: "nnUNetPlans",
    configuration: "3d_fullres",
    folds: 1,
    axes: AxisOrder::Sar,
    home: Home::TotalSegmentator,
    source: Source::Licensed {
        task: "tissue_types",
    },
};

const LUT_TISSUE_TYPES: [u8; 4] = [0, 1, 2, 3];

// ---- tissue_4_types --------------------------------------------------------
pub const CLASSES_TISSUE_4_TYPES: [&str; 4] = [
    "subcutaneous_fat",
    "torso_fat",
    "skeletal_muscle",
    "intermuscular_fat",
];

pub const SPEC_TISSUE_4_TYPES: ModelSpec = ModelSpec {
    key: "tissue_4_types",
    label: "Tissue types, four",
    detail: "Subcutaneous fat, torso fat, skeletal muscle and intermuscular fat.",
    url: "",
    zip_bytes: 235_000_000,
    plans: "nnUNetPlans",
    configuration: "3d_fullres",
    folds: 1,
    axes: AxisOrder::Sar,
    home: Home::TotalSegmentator,
    source: Source::Licensed {
        task: "tissue_4_types",
    },
};

const LUT_TISSUE_4_TYPES: [u8; 5] = [0, 1, 2, 3, 4];

// ---- appendicular_bones ----------------------------------------------------
pub const CLASSES_APPENDICULAR_BONES: [&str; 11] = [
    "patella",
    "tibia",
    "fibula",
    "tarsal",
    "metatarsal",
    "phalanges_feet",
    "ulna",
    "radius",
    "carpal",
    "metacarpal",
    "phalanges_hand",
];

pub const SPEC_APPENDICULAR_BONES: ModelSpec = ModelSpec {
    key: "appendicular_bones",
    label: "Appendicular bones",
    detail: "Bones of the hands, forearms, knees and feet.",
    url: "",
    zip_bytes: 235_000_000,
    plans: "nnUNetPlans",
    configuration: "3d_fullres",
    folds: 1,
    axes: AxisOrder::Sar,
    home: Home::TotalSegmentator,
    source: Source::Licensed {
        task: "appendicular_bones",
    },
};

const LUT_APPENDICULAR_BONES: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 0, 0, 0];

// ---- thigh_shoulder_muscles ------------------------------------------------
pub const CLASSES_THIGH_SHOULDER_MUSCLES: [&str; 18] = [
    "quadriceps_femoris_left",
    "quadriceps_femoris_right",
    "thigh_medial_compartment_left",
    "thigh_medial_compartment_right",
    "thigh_posterior_compartment_left",
    "thigh_posterior_compartment_right",
    "sartorius_left",
    "sartorius_right",
    "deltoid",
    "supraspinatus",
    "infraspinatus",
    "subscapularis",
    "coracobrachial",
    "trapezius",
    "pectoralis_minor",
    "serratus_anterior",
    "teres_major",
    "triceps_brachii",
];

pub const SPEC_THIGH_SHOULDER_MUSCLES: ModelSpec = ModelSpec {
    key: "thigh_shoulder_muscles",
    label: "Thigh and shoulder muscles",
    detail: "Thigh compartments and shoulder-girdle muscles (the MR-trained model, also run on CT upstream).",
    url: "",
    zip_bytes: 235_000_000,
    plans: "nnUNetPlans",
    configuration: "3d_fullres",
    folds: 1,
    axes: AxisOrder::Sar,
    home: Home::TotalSegmentator,
    source: Source::Licensed { task: "thigh_shoulder_muscles_mr" },
};

const LUT_THIGH_SHOULDER_MUSCLES: [u8; 19] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
];

// ---- face ------------------------------------------------------------------
pub const CLASSES_FACE: [&str; 1] = ["face"];

pub const SPEC_FACE: ModelSpec = ModelSpec {
    key: "face",
    label: "Face",
    detail: "The face surface region, as used for defacing.",
    url: "",
    zip_bytes: 235_000_000,
    plans: "nnUNetPlans",
    configuration: "3d_fullres",
    folds: 1,
    axes: AxisOrder::Sar,
    home: Home::TotalSegmentator,
    source: Source::Licensed { task: "face" },
};

const LUT_FACE: [u8; 2] = [0, 1];

// ---- brain_structures ------------------------------------------------------
pub const CLASSES_BRAIN_STRUCTURES: [&str; 16] = [
    "brainstem",
    "subarachnoid_space",
    "venous_sinuses",
    "septum_pellucidum",
    "cerebellum",
    "caudate_nucleus",
    "lentiform_nucleus",
    "insular_cortex",
    "internal_capsule",
    "ventricle",
    "central_sulcus",
    "frontal_lobe",
    "parietal_lobe",
    "occipital_lobe",
    "temporal_lobe",
    "thalamus",
];

pub const SPEC_BRAIN_STRUCTURES: ModelSpec = ModelSpec {
    key: "brain_structures",
    label: "Brain structures",
    detail: "Sixteen brain structures on head CT, cropped to the brain.",
    url: "",
    zip_bytes: 235_000_000,
    plans: "nnUNetPlans",
    configuration: "3d_fullres_high",
    folds: 1,
    axes: AxisOrder::Sar,
    home: Home::TotalSegmentator,
    source: Source::Licensed {
        task: "brain_structures",
    },
};

const LUT_BRAIN_STRUCTURES: [u8; 17] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

// ---- tissue_types_mr -------------------------------------------------------
pub const CLASSES_TISSUE_TYPES_MR: [&str; 3] = ["subcutaneous_fat", "torso_fat", "skeletal_muscle"];

pub const SPEC_TISSUE_TYPES_MR: ModelSpec = ModelSpec {
    key: "tissue_types_mr",
    label: "Tissue types",
    detail: "Subcutaneous fat, torso fat and skeletal muscle on MR.",
    url: "",
    zip_bytes: 235_000_000,
    plans: "nnUNetPlans",
    configuration: "3d_fullres",
    folds: 1,
    axes: AxisOrder::Sar,
    home: Home::TotalSegmentator,
    source: Source::Licensed {
        task: "tissue_types_mr",
    },
};

const LUT_TISSUE_TYPES_MR: [u8; 4] = [0, 1, 2, 3];

// ---- appendicular_bones_mr -------------------------------------------------
pub const CLASSES_APPENDICULAR_BONES_MR: [&str; 8] = [
    "patella",
    "tibia",
    "fibula",
    "tarsal",
    "metatarsal",
    "phalanges_feet",
    "ulna",
    "radius",
];

pub const SPEC_APPENDICULAR_BONES_MR: ModelSpec = ModelSpec {
    key: "appendicular_bones_mr",
    label: "Appendicular bones",
    detail: "Bones of the forearms, knees and feet on MR.",
    url: "",
    zip_bytes: 235_000_000,
    plans: "nnUNetPlans",
    configuration: "3d_fullres",
    folds: 1,
    axes: AxisOrder::Sar,
    home: Home::TotalSegmentator,
    source: Source::Licensed {
        task: "appendicular_bones_mr",
    },
};

const LUT_APPENDICULAR_BONES_MR: [u8; 9] = [0, 1, 2, 3, 4, 5, 6, 7, 8];

// ---- thigh_shoulder_muscles_mr ---------------------------------------------
pub const CLASSES_THIGH_SHOULDER_MUSCLES_MR: [&str; 18] = [
    "quadriceps_femoris_left",
    "quadriceps_femoris_right",
    "thigh_medial_compartment_left",
    "thigh_medial_compartment_right",
    "thigh_posterior_compartment_left",
    "thigh_posterior_compartment_right",
    "sartorius_left",
    "sartorius_right",
    "deltoid",
    "supraspinatus",
    "infraspinatus",
    "subscapularis",
    "coracobrachial",
    "trapezius",
    "pectoralis_minor",
    "serratus_anterior",
    "teres_major",
    "triceps_brachii",
];

const LUT_THIGH_SHOULDER_MUSCLES_MR: [u8; 19] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
];

// ---- face_mr ---------------------------------------------------------------
pub const CLASSES_FACE_MR: [&str; 1] = ["face"];

pub const SPEC_FACE_MR: ModelSpec = ModelSpec {
    key: "face_mr",
    label: "Face",
    detail: "The face surface region on MR.",
    url: "",
    zip_bytes: 235_000_000,
    plans: "nnUNetPlans",
    configuration: "3d_fullres",
    folds: 1,
    axes: AxisOrder::Sar,
    home: Home::TotalSegmentator,
    source: Source::Licensed { task: "face_mr" },
};

const LUT_FACE_MR: [u8; 4] = [0, 1, 0, 0];

/// Every open task, in the order the interface lists them.
pub static TASKS: [NnTask; 26] = [
    NnTask {
        key: "lung_vessels",
        label: "Lung airways and vessels",
        group: "Thorax",
        detail: "Airways, airway walls, pulmonary arteries and veins inside the lungs.",
        modality: Modality::Ct,
        licence: Licence::Apache2,
        classes: &CLASSES_LUNG_VESSELS,
        parts: &[
            Part { spec: SPEC_LUNG_VESSELS, lut: &LUT_LUNG_VESSELS },
        ],
        folds: FoldUse::All,
        step: 0.5,
        crop: Some(Crop { by: CropBy::Total3mm, classes: &["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"], addon_mm: 20.0 }), // configured [3, 3, 3]
        post: Post::None,
    },
    NnTask {
        key: "lung_nodules",
        label: "Lung nodules",
        group: "Thorax",
        detail: "The lungs and the nodules in them (BLUEMIND AI, partly LIDC-IDRI); residual-encoder network.",
        modality: Modality::Ct,
        licence: Licence::Apache2,
        classes: &CLASSES_LUNG_NODULES,
        parts: &[
            Part { spec: SPEC_LUNG_NODULES, lut: &LUT_LUNG_NODULES },
        ],
        folds: FoldUse::All,
        step: 0.5,
        crop: Some(Crop { by: CropBy::Total6mm, classes: &["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"], addon_mm: 20.0 }), // configured [10, 10, 10]
        post: Post::None,
    },
    NnTask {
        key: "pleural_pericard_effusion",
        label: "Pleural and pericardial effusion",
        group: "Thorax",
        detail: "Pleura, pleural effusion and pericardial effusion.",
        modality: Modality::Ct,
        licence: Licence::Apache2,
        classes: &CLASSES_PLEURAL_PERICARD_EFFUSION,
        parts: &[
            Part { spec: SPEC_PLEURAL_PERICARD_EFFUSION, lut: &LUT_PLEURAL_PERICARD_EFFUSION },
        ],
        folds: FoldUse::All,
        step: 0.5,
        crop: Some(Crop { by: CropBy::Total6mm, classes: &["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"], addon_mm: 20.0 }), // configured [50, 50, 50]
        post: Post::None,
    },
    NnTask {
        key: "trunk_cavities",
        label: "Trunk cavities",
        group: "Thorax",
        detail: "Abdominal and thoracic cavity, pericardium, mediastinum.",
        modality: Modality::Ct,
        licence: Licence::Apache2,
        classes: &CLASSES_TRUNK_CAVITIES,
        parts: &[
            Part { spec: SPEC_TRUNK_CAVITIES, lut: &LUT_TRUNK_CAVITIES },
        ],
        folds: FoldUse::All,
        step: 0.5,
        crop: None,
        post: Post::None,
    },
    NnTask {
        key: "breasts",
        label: "Breasts",
        group: "Thorax",
        detail: "Breast tissue.",
        modality: Modality::Ct,
        licence: Licence::Apache2,
        classes: &CLASSES_BREASTS,
        parts: &[
            Part { spec: SPEC_BREASTS, lut: &LUT_BREASTS },
        ],
        folds: FoldUse::All,
        step: 0.5,
        crop: None,
        post: Post::None,
    },
    NnTask {
        key: "liver_vessels",
        label: "Liver vessels",
        group: "Abdomen",
        detail: "Hepatic vessels and liver tumour (MSD Task 8).",
        modality: Modality::Ct,
        licence: Licence::Apache2,
        classes: &CLASSES_LIVER_VESSELS,
        parts: &[
            Part { spec: SPEC_LIVER_VESSELS, lut: &LUT_LIVER_VESSELS },
        ],
        folds: FoldUse::All,
        step: 0.5,
        crop: Some(Crop { by: CropBy::Total6mm, classes: &["liver"], addon_mm: 20.0 }),
        post: Post::None,
    },
    NnTask {
        key: "liver_segments",
        label: "Liver segments (Couinaud)",
        group: "Abdomen",
        detail: "The eight Couinaud segments of the liver.",
        modality: Modality::Ct,
        licence: Licence::Apache2,
        classes: &CLASSES_LIVER_SEGMENTS,
        parts: &[
            Part { spec: SPEC_LIVER_SEGMENTS, lut: &LUT_LIVER_SEGMENTS },
        ],
        folds: FoldUse::All,
        step: 0.5,
        crop: Some(Crop { by: CropBy::Total6mm, classes: &["liver"], addon_mm: 20.0 }), // configured [10, 10, 10]
        post: Post::None,
    },
    NnTask {
        key: "liver_lesions",
        label: "Liver lesions",
        group: "Abdomen",
        detail: "Focal liver lesions.",
        modality: Modality::Ct,
        licence: Licence::Apache2,
        classes: &CLASSES_LIVER_LESIONS,
        parts: &[
            Part { spec: SPEC_LIVER_LESIONS, lut: &LUT_LIVER_LESIONS },
        ],
        folds: FoldUse::All,
        step: 0.5,
        crop: Some(Crop { by: CropBy::Total3mm, classes: &["liver"], addon_mm: 20.0 }), // configured [10, 10, 10]
        post: Post::None,
    },
    NnTask {
        key: "kidney_cysts",
        label: "Kidney cysts",
        group: "Abdomen",
        detail: "Cysts of the left and right kidney.",
        modality: Modality::Ct,
        licence: Licence::Apache2,
        classes: &CLASSES_KIDNEY_CYSTS,
        parts: &[
            Part { spec: SPEC_KIDNEY_CYSTS, lut: &LUT_KIDNEY_CYSTS },
        ],
        folds: FoldUse::All,
        step: 0.5,
        crop: Some(Crop { by: CropBy::Total6mm, classes: &["kidney_left", "kidney_right", "liver", "spleen", "colon"], addon_mm: 20.0 }), // configured [10, 10, 10]
        post: Post::None,
    },
    NnTask {
        key: "abdominal_muscles",
        label: "Abdominal muscles",
        group: "Abdomen",
        detail: "Abdominal wall and back muscles between T4 and L4.",
        modality: Modality::Ct,
        licence: Licence::Apache2,
        classes: &CLASSES_ABDOMINAL_MUSCLES,
        parts: &[
            Part { spec: SPEC_ABDOMINAL_MUSCLES, lut: &LUT_ABDOMINAL_MUSCLES },
        ],
        folds: FoldUse::All,
        step: 0.5,
        crop: Some(Crop { by: CropBy::Body6mm, classes: &["body_trunc"], addon_mm: 20.0 }),
        post: Post::None,
    },
    NnTask {
        key: "head_glands_cavities",
        label: "Head glands and cavities",
        group: "Head and neck",
        detail: "Eyes, lenses, optic nerves, salivary glands, sinuses, auditory canals.",
        modality: Modality::Ct,
        licence: Licence::Apache2,
        classes: &CLASSES_HEAD_GLANDS_CAVITIES,
        parts: &[
            Part { spec: SPEC_HEAD_GLANDS_CAVITIES, lut: &LUT_HEAD_GLANDS_CAVITIES },
        ],
        folds: FoldUse::All,
        step: 0.5,
        crop: Some(Crop { by: CropBy::Total6mm, classes: &["skull"], addon_mm: 20.0 }), // configured [10, 10, 10]
        post: Post::None,
    },
    NnTask {
        key: "headneck_bones_vessels",
        label: "Head and neck bones and vessels",
        group: "Head and neck",
        detail: "Larynx cartilages, hyoid, carotid and vertebral arteries, jugular veins.",
        modality: Modality::Ct,
        licence: Licence::Apache2,
        classes: &CLASSES_HEADNECK_BONES_VESSELS,
        parts: &[
            Part { spec: SPEC_HEADNECK_BONES_VESSELS, lut: &LUT_HEADNECK_BONES_VESSELS },
        ],
        folds: FoldUse::All,
        step: 0.5,
        crop: Some(Crop { by: CropBy::Total6mm, classes: &["clavicula_left", "clavicula_right", "vertebrae_C1", "vertebrae_C5", "vertebrae_T1", "vertebrae_T4"], addon_mm: 20.0 }), // configured [40, 40, 40]
        post: Post::None,
    },
    NnTask {
        key: "head_muscles",
        label: "Head muscles",
        group: "Head and neck",
        detail: "Masticatory and facial muscles, tongue.",
        modality: Modality::Ct,
        licence: Licence::Apache2,
        classes: &CLASSES_HEAD_MUSCLES,
        parts: &[
            Part { spec: SPEC_HEAD_MUSCLES, lut: &LUT_HEAD_MUSCLES },
        ],
        folds: FoldUse::All,
        step: 0.5,
        crop: Some(Crop { by: CropBy::Total6mm, classes: &["skull"], addon_mm: 20.0 }), // configured [10, 10, 10]
        post: Post::None,
    },
    NnTask {
        key: "headneck_muscles",
        label: "Head and neck muscles",
        group: "Head and neck",
        detail: "Neck muscles and pharyngeal constrictors (two sub-models).",
        modality: Modality::Ct,
        licence: Licence::Apache2,
        classes: &CLASSES_HEADNECK_MUSCLES,
        parts: &[
            Part { spec: SPEC_HEADNECK_MUSCLES_PART1, lut: &LUT_HEADNECK_MUSCLES_PART1 },
            Part { spec: SPEC_HEADNECK_MUSCLES_PART2, lut: &LUT_HEADNECK_MUSCLES_PART2 },
        ],
        folds: FoldUse::All,
        step: 0.5,
        crop: Some(Crop { by: CropBy::Total6mm, classes: &["clavicula_left", "clavicula_right", "vertebrae_C1", "vertebrae_C5", "vertebrae_T1", "vertebrae_T4"], addon_mm: 20.0 }), // configured [40, 40, 40]
        post: Post::None,
    },
    NnTask {
        key: "oculomotor_muscles",
        label: "Oculomotor muscles",
        group: "Head and neck",
        detail: "Extra-ocular muscles, eyeballs, optic nerves (18 training cases).",
        modality: Modality::Ct,
        licence: Licence::Apache2,
        classes: &CLASSES_OCULOMOTOR_MUSCLES,
        parts: &[
            Part { spec: SPEC_OCULOMOTOR_MUSCLES, lut: &LUT_OCULOMOTOR_MUSCLES },
        ],
        folds: FoldUse::All,
        step: 0.5,
        crop: Some(Crop { by: CropBy::Total6mm, classes: &["skull"], addon_mm: 20.0 }),
        post: Post::None,
    },
    NnTask {
        key: "craniofacial_structures",
        label: "Craniofacial structures",
        group: "Head and neck",
        detail: "Mandible, teeth (upper and lower), skull, head, sinuses.",
        modality: Modality::Ct,
        licence: Licence::Apache2,
        classes: &CLASSES_CRANIOFACIAL_STRUCTURES,
        parts: &[
            Part { spec: SPEC_CRANIOFACIAL_STRUCTURES, lut: &LUT_CRANIOFACIAL_STRUCTURES },
        ],
        folds: FoldUse::All,
        step: 0.5,
        crop: Some(Crop { by: CropBy::Total6mm, classes: &["skull"], addon_mm: 20.0 }),
        post: Post::None,
    },
    NnTask {
        key: "teeth",
        label: "Teeth (FDI) and jaw",
        group: "Head and neck",
        detail: "Every tooth by FDI number, jaws, canals, pulp (ToothFairy3); cropped to the teeth found by Craniofacial structures.",
        modality: Modality::Ct,
        licence: Licence::Apache2,
        classes: &CLASSES_TEETH,
        parts: &[
            Part { spec: SPEC_TEETH, lut: &LUT_TEETH },
        ],
        folds: FoldUse::All,
        step: 0.5,
        crop: Some(Crop { by: CropBy::Task("craniofacial_structures"), classes: &["teeth_lower", "teeth_upper"], addon_mm: 10.0 }),
        post: Post::None,
    },
    NnTask {
        key: "cerebral_bleed",
        label: "Intracerebral haemorrhage",
        group: "Brain",
        detail: "Intracerebral haemorrhage on non-contrast head CT.",
        modality: Modality::Ct,
        licence: Licence::Apache2,
        classes: &CLASSES_CEREBRAL_BLEED,
        parts: &[
            Part { spec: SPEC_CEREBRAL_BLEED, lut: &LUT_CEREBRAL_BLEED },
        ],
        folds: FoldUse::All,
        step: 0.5,
        crop: Some(Crop { by: CropBy::Total6mm, classes: &["brain"], addon_mm: 20.0 }), // configured [3, 3, 3]
        post: Post::None,
    },
    NnTask {
        key: "ventricle_parts",
        label: "Brain ventricle parts",
        group: "Brain",
        detail: "The parts of the ventricular system (38 training cases).",
        modality: Modality::Ct,
        licence: Licence::Apache2,
        classes: &CLASSES_VENTRICLE_PARTS,
        parts: &[
            Part { spec: SPEC_VENTRICLE_PARTS, lut: &LUT_VENTRICLE_PARTS },
        ],
        folds: FoldUse::All,
        step: 0.5,
        crop: Some(Crop { by: CropBy::Total6mm, classes: &["brain"], addon_mm: 20.0 }), // configured [0, 0, 0]
        post: Post::None,
    },
    NnTask {
        key: "hip_implant",
        label: "Hip implant",
        group: "Bones",
        detail: "Metal hip prostheses.",
        modality: Modality::Ct,
        licence: Licence::Apache2,
        classes: &CLASSES_HIP_IMPLANT,
        parts: &[
            Part { spec: SPEC_HIP_IMPLANT, lut: &LUT_HIP_IMPLANT },
        ],
        folds: FoldUse::All,
        step: 0.5,
        crop: Some(Crop { by: CropBy::Total6mm, classes: &["femur_left", "femur_right", "hip_left", "hip_right"], addon_mm: 20.0 }), // configured [3, 3, 3]
        post: Post::None,
    },
    NnTask {
        key: "vertebrae_body",
        label: "Vertebral bodies and discs",
        group: "Bones",
        detail: "Vertebral bodies (without the posterior elements) and intervertebral discs.",
        modality: Modality::Ct,
        licence: Licence::Apache2,
        classes: &CLASSES_VERTEBRAE_BODY,
        parts: &[
            Part { spec: SPEC_VERTEBRAE_BODY, lut: &LUT_VERTEBRAE_BODY },
        ],
        folds: FoldUse::All,
        step: 0.5,
        crop: None,
        post: Post::None,
    },
    NnTask {
        key: "vertebrae_pp",
        label: "Vertebral bodies, numbered",
        group: "Bones",
        detail: "Each vertebral body by level, with the upstream relabelling of touching vertebrae.",
        modality: Modality::Ct,
        licence: Licence::Apache2,
        classes: &CLASSES_VERTEBRAE_PP,
        parts: &[
            Part { spec: SPEC_VERTEBRAE_PP, lut: &LUT_VERTEBRAE_PP },
        ],
        folds: FoldUse::All,
        step: 0.5,
        crop: None,
        post: Post::VertebraePp,
    },
    NnTask {
        key: "vertebrae_mr",
        label: "Vertebrae and discs (MR)",
        group: "MR",
        detail: "Every vertebra and the intervertebral discs and spinal canal on MR.",
        modality: Modality::Mr,
        licence: Licence::Apache2,
        classes: &CLASSES_VERTEBRAE_MR,
        parts: &[
            Part { spec: SPEC_VERTEBRAE_MR, lut: &LUT_VERTEBRAE_MR },
        ],
        folds: FoldUse::All,
        step: 0.5,
        crop: None,
        post: Post::None,
    },
    NnTask {
        key: "liver_segments_mr",
        label: "Liver segments (MR)",
        group: "MR",
        detail: "The eight Couinaud segments on MR.",
        modality: Modality::Mr,
        licence: Licence::Apache2,
        classes: &CLASSES_LIVER_SEGMENTS_MR,
        parts: &[
            Part { spec: SPEC_LIVER_SEGMENTS_MR, lut: &LUT_LIVER_SEGMENTS_MR },
        ],
        folds: FoldUse::All,
        step: 0.5,
        crop: Some(Crop { by: CropBy::TotalMr3mm, classes: &["liver"], addon_mm: 20.0 }), // configured [10, 10, 10]
        post: Post::None,
    },
    NnTask {
        key: "liver_lesions_mr",
        label: "Liver lesions (MR)",
        group: "MR",
        detail: "Focal liver lesions on MR (trained on CT and MR).",
        modality: Modality::Mr,
        licence: Licence::Apache2,
        classes: &CLASSES_LIVER_LESIONS_MR,
        parts: &[
            Part { spec: SPEC_LIVER_LESIONS_MR, lut: &LUT_LIVER_LESIONS_MR },
        ],
        folds: FoldUse::All,
        step: 0.5,
        crop: Some(Crop { by: CropBy::TotalMr3mm, classes: &["liver"], addon_mm: 20.0 }), // configured [3, 3, 3]
        post: Post::None,
    },
    NnTask {
        key: "brain_aneurysm",
        label: "Brain aneurysm (TOF MRA)",
        group: "MR",
        detail: "Intracranial aneurysms on time-of-flight MR angiography (MAXIMUS, five folds).",
        modality: Modality::Mr,
        licence: Licence::CcByNc4,
        classes: &CLASSES_BRAIN_ANEURYSM,
        parts: &[
            Part { spec: SPEC_BRAIN_ANEURYSM, lut: &LUT_BRAIN_ANEURYSM },
        ],
        folds: FoldUse::All,
        step: 0.5,
        crop: None,
        post: Post::None,
    },
];

/// The licensed tasks: weights from the TotalSegmentator licence server,
/// for the user's licence number.
pub static LICENSED_TASKS: [NnTask; 17] = [
    NnTask {
        key: "heartchambers_highres",
        label: "Heart chambers, 1.5 mm",
        group: "Licensed (TotalSegmentator)",
        detail: "Myocardium, atria, ventricles, aorta and pulmonary artery at full resolution (cropped to the heart).",
        modality: Modality::Ct,
        licence: Licence::TsLicensed,
        classes: &CLASSES_HEARTCHAMBERS_HIGHRES,
        parts: &[
            Part { spec: SPEC_HEARTCHAMBERS_HIGHRES, lut: &LUT_HEARTCHAMBERS_HIGHRES },
        ],
        folds: FoldUse::All,
        step: 0.5,
        crop: Some(Crop { by: CropBy::Total3mm, classes: &["heart"], addon_mm: 20.0 }), // configured [5, 5, 5]
        post: Post::RemoveOutside { classes: &["heart", "aorta", "inferior_vena_cava"], dilation_mm: 10.0 },
    },
    NnTask {
        key: "coronary_arteries",
        label: "Coronary arteries",
        group: "Licensed (TotalSegmentator)",
        detail: "The coronary arteries at 0.7 mm, cropped to the heart.",
        modality: Modality::Ct,
        licence: Licence::TsLicensed,
        classes: &CLASSES_CORONARY_ARTERIES,
        parts: &[
            Part { spec: SPEC_CORONARY_ARTERIES, lut: &LUT_CORONARY_ARTERIES },
        ],
        folds: FoldUse::All,
        step: 0.5,
        crop: Some(Crop { by: CropBy::Total6mm, classes: &["heart"], addon_mm: 20.0 }),
        post: Post::None,
    },
    NnTask {
        key: "aortic_sinuses",
        label: "Aortic sinuses",
        group: "Licensed (TotalSegmentator)",
        detail: "Left ventricular outflow tract and the three coronary cusps at 0.7 mm.",
        modality: Modality::Ct,
        licence: Licence::TsLicensed,
        classes: &CLASSES_AORTIC_SINUSES,
        parts: &[
            Part { spec: SPEC_AORTIC_SINUSES, lut: &LUT_AORTIC_SINUSES },
        ],
        folds: FoldUse::All,
        step: 0.5,
        crop: Some(Crop { by: CropBy::Total6mm, classes: &["heart"], addon_mm: 20.0 }), // configured [0, 0, 0]
        post: Post::None,
    },
    NnTask {
        key: "aorta_annulus",
        label: "Aortic annulus",
        group: "Licensed (TotalSegmentator)",
        detail: "Aortic annulus and sinotubular junction at 0.8 mm (five folds).",
        modality: Modality::Ct,
        licence: Licence::TsLicensed,
        classes: &CLASSES_AORTA_ANNULUS,
        parts: &[
            Part { spec: SPEC_AORTA_ANNULUS, lut: &LUT_AORTA_ANNULUS },
        ],
        folds: FoldUse::All,
        step: 0.5,
        crop: None,
        post: Post::None,
    },
    NnTask {
        key: "aortic_dissection",
        label: "Aortic dissection",
        group: "Licensed (TotalSegmentator)",
        detail: "True and false lumen of a dissected aorta at 0.8 mm (five folds).",
        modality: Modality::Ct,
        licence: Licence::TsLicensed,
        classes: &CLASSES_AORTIC_DISSECTION,
        parts: &[
            Part { spec: SPEC_AORTIC_DISSECTION, lut: &LUT_AORTIC_DISSECTION },
        ],
        folds: FoldUse::All,
        step: 0.5,
        crop: None,
        post: Post::None,
    },
    NnTask {
        key: "pulmonary_artery_landmarks",
        label: "Pulmonary artery landmarks",
        group: "Licensed (TotalSegmentator)",
        detail: "Aortic and pulmonary annulus, sinotubular junctions, pulmonary bifurcation and branches (five folds).",
        modality: Modality::Ct,
        licence: Licence::TsLicensed,
        classes: &CLASSES_PULMONARY_ARTERY_LANDMARKS,
        parts: &[
            Part { spec: SPEC_PULMONARY_ARTERY_LANDMARKS, lut: &LUT_PULMONARY_ARTERY_LANDMARKS },
        ],
        folds: FoldUse::All,
        step: 0.5,
        crop: None,
        post: Post::None,
    },
    NnTask {
        key: "renal_arteries",
        label: "Renal arteries",
        group: "Licensed (TotalSegmentator)",
        detail: "Coeliac trunk, superior mesenteric and renal arteries.",
        modality: Modality::Ct,
        licence: Licence::TsLicensed,
        classes: &CLASSES_RENAL_ARTERIES,
        parts: &[
            Part { spec: SPEC_RENAL_ARTERIES, lut: &LUT_RENAL_ARTERIES },
        ],
        folds: FoldUse::All,
        step: 0.5,
        crop: None,
        post: Post::None,
    },
    NnTask {
        key: "tissue_types",
        label: "Tissue types",
        group: "Licensed (TotalSegmentator)",
        detail: "Subcutaneous fat, torso fat and skeletal muscle.",
        modality: Modality::Ct,
        licence: Licence::TsLicensed,
        classes: &CLASSES_TISSUE_TYPES,
        parts: &[
            Part { spec: SPEC_TISSUE_TYPES, lut: &LUT_TISSUE_TYPES },
        ],
        folds: FoldUse::All,
        step: 0.5,
        crop: None,
        post: Post::None,
    },
    NnTask {
        key: "tissue_4_types",
        label: "Tissue types, four",
        group: "Licensed (TotalSegmentator)",
        detail: "Subcutaneous fat, torso fat, skeletal muscle and intermuscular fat.",
        modality: Modality::Ct,
        licence: Licence::TsLicensed,
        classes: &CLASSES_TISSUE_4_TYPES,
        parts: &[
            Part { spec: SPEC_TISSUE_4_TYPES, lut: &LUT_TISSUE_4_TYPES },
        ],
        folds: FoldUse::All,
        step: 0.5,
        crop: None,
        post: Post::None,
    },
    NnTask {
        key: "appendicular_bones",
        label: "Appendicular bones",
        group: "Licensed (TotalSegmentator)",
        detail: "Bones of the hands, forearms, knees and feet.",
        modality: Modality::Ct,
        licence: Licence::TsLicensed,
        classes: &CLASSES_APPENDICULAR_BONES,
        parts: &[
            Part { spec: SPEC_APPENDICULAR_BONES, lut: &LUT_APPENDICULAR_BONES },
        ],
        folds: FoldUse::All,
        step: 0.5,
        crop: None,
        post: Post::None,
    },
    NnTask {
        key: "thigh_shoulder_muscles",
        label: "Thigh and shoulder muscles",
        group: "Licensed (TotalSegmentator)",
        detail: "Thigh compartments and shoulder-girdle muscles (the MR-trained model, also run on CT upstream).",
        modality: Modality::Ct,
        licence: Licence::TsLicensed,
        classes: &CLASSES_THIGH_SHOULDER_MUSCLES,
        parts: &[
            Part { spec: SPEC_THIGH_SHOULDER_MUSCLES, lut: &LUT_THIGH_SHOULDER_MUSCLES },
        ],
        folds: FoldUse::All,
        step: 0.5,
        crop: None,
        post: Post::None,
    },
    NnTask {
        key: "face",
        label: "Face",
        group: "Licensed (TotalSegmentator)",
        detail: "The face surface region, as used for defacing.",
        modality: Modality::Ct,
        licence: Licence::TsLicensed,
        classes: &CLASSES_FACE,
        parts: &[
            Part { spec: SPEC_FACE, lut: &LUT_FACE },
        ],
        folds: FoldUse::All,
        step: 0.5,
        crop: None,
        post: Post::None,
    },
    NnTask {
        key: "brain_structures",
        label: "Brain structures",
        group: "Licensed (TotalSegmentator)",
        detail: "Sixteen brain structures on head CT, cropped to the brain.",
        modality: Modality::Ct,
        licence: Licence::TsLicensed,
        classes: &CLASSES_BRAIN_STRUCTURES,
        parts: &[
            Part { spec: SPEC_BRAIN_STRUCTURES, lut: &LUT_BRAIN_STRUCTURES },
        ],
        folds: FoldUse::All,
        step: 0.5,
        crop: Some(Crop { by: CropBy::Total6mm, classes: &["brain"], addon_mm: 20.0 }), // configured [10, 10, 10]
        post: Post::None,
    },
    NnTask {
        key: "tissue_types_mr",
        label: "Tissue types (MR)",
        group: "Licensed (TotalSegmentator)",
        detail: "Subcutaneous fat, torso fat and skeletal muscle on MR.",
        modality: Modality::Mr,
        licence: Licence::TsLicensed,
        classes: &CLASSES_TISSUE_TYPES_MR,
        parts: &[
            Part { spec: SPEC_TISSUE_TYPES_MR, lut: &LUT_TISSUE_TYPES_MR },
        ],
        folds: FoldUse::All,
        step: 0.5,
        crop: None,
        post: Post::None,
    },
    NnTask {
        key: "appendicular_bones_mr",
        label: "Appendicular bones (MR)",
        group: "Licensed (TotalSegmentator)",
        detail: "Bones of the forearms, knees and feet on MR.",
        modality: Modality::Mr,
        licence: Licence::TsLicensed,
        classes: &CLASSES_APPENDICULAR_BONES_MR,
        parts: &[
            Part { spec: SPEC_APPENDICULAR_BONES_MR, lut: &LUT_APPENDICULAR_BONES_MR },
        ],
        folds: FoldUse::All,
        step: 0.5,
        crop: None,
        post: Post::None,
    },
    NnTask {
        key: "thigh_shoulder_muscles_mr",
        label: "Thigh and shoulder muscles (MR)",
        group: "Licensed (TotalSegmentator)",
        detail: "Thigh compartments and shoulder-girdle muscles on MR.",
        modality: Modality::Mr,
        licence: Licence::TsLicensed,
        classes: &CLASSES_THIGH_SHOULDER_MUSCLES_MR,
        parts: &[
            Part { spec: SPEC_THIGH_SHOULDER_MUSCLES, lut: &LUT_THIGH_SHOULDER_MUSCLES_MR },
        ],
        folds: FoldUse::All,
        step: 0.5,
        crop: None,
        post: Post::None,
    },
    NnTask {
        key: "face_mr",
        label: "Face (MR)",
        group: "Licensed (TotalSegmentator)",
        detail: "The face surface region on MR.",
        modality: Modality::Mr,
        licence: Licence::TsLicensed,
        classes: &CLASSES_FACE_MR,
        parts: &[
            Part { spec: SPEC_FACE_MR, lut: &LUT_FACE_MR },
        ],
        folds: FoldUse::All,
        step: 0.5,
        crop: None,
        post: Post::None,
    },
];
