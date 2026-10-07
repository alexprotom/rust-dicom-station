//! The `total` family as tasks: TotalSegmentator's 117-class CT task in both
//! generations and every resolution, its MR counterpart, the body outline,
//! and MRSegmentator.
//!
//! These are the tasks upstream runs through `--fast` / `--fastest` and the
//! sub-model sets; each resolution is its own row here, because each is its
//! own download and its own runtime. The catalogue of every other open
//! task is generated data in [`super::tasks`].

use super::classes::{TOTAL_CLASS_NAMES, TOTAL_V3_CLASS_NAMES};
use super::task::{shifted, FoldUse, Licence, Modality, NnTask, Part, Post, IDENTITY_LUT};
use super::tasks::{
    CLASSES_TOTAL_MR, LUT_TOTAL_MR_PART1_ORGANS, LUT_TOTAL_MR_PART2_MUSCLES, SPEC_BODY_MR_6MM,
    SPEC_TOTAL_MR_3MM, SPEC_TOTAL_MR_6MM, SPEC_TOTAL_MR_PART1_ORGANS, SPEC_TOTAL_MR_PART2_MUSCLES,
};
use super::weights::{
    SPECS_15MM, SPECS_V3_15MM, SPECS_V3_SMALL_15MM, SPEC_3MM, SPEC_6MM, SPEC_BODY_15MM,
    SPEC_BODY_6MM, SPEC_BODY_MR, SPEC_MRSEGMENTATOR, SPEC_V3_3MM, SPEC_V3_6MM, SPEC_V3_SMALL_3MM,
};

const GROUP_CT: &str = "TotalSegmentator CT";
const GROUP_MR: &str = "MR whole body";
const GROUP_BODY: &str = "Body outline";

// The five 1.5 mm sub-models of `total` predict contiguous slices of the
// 117-class table: organs 1-24, vertebrae 25-50, cardiac 51-68, muscles
// 69-91, ribs 92-117 (`classes::PART_OFFSET`).
const LUT_P1: [u8; 25] = shifted(0);
const LUT_P2: [u8; 27] = shifted(24);
const LUT_P3: [u8; 19] = shifted(50);
const LUT_P4: [u8; 24] = shifted(68);
const LUT_P5: [u8; 27] = shifted(91);

pub static TOTAL_FAST: NnTask = NnTask {
    key: "total_fast",
    label: "total, 3 mm",
    group: GROUP_CT,
    detail: "All 117 structures at 3 mm, one network (v2 weights, TotalSegmentator's default). \
             Good quality, practical on any CPU.",
    modality: Modality::Ct,
    licence: Licence::Apache2,
    classes: &TOTAL_CLASS_NAMES,
    parts: &[Part {
        spec: SPEC_3MM,
        lut: &IDENTITY_LUT,
    }],
    folds: FoldUse::All,
    step: 0.8,
    crop: None,
    post: Post::None,
};

pub static TOTAL: NnTask = NnTask {
    key: "total",
    label: "total, 1.5 mm",
    group: GROUP_CT,
    detail: "All 117 structures at 1.5 mm, five sub-models (organs, vertebrae, cardiac, \
             muscles, ribs) - the reference quality. Slow without a GPU.",
    modality: Modality::Ct,
    licence: Licence::Apache2,
    classes: &TOTAL_CLASS_NAMES,
    parts: &[
        Part {
            spec: SPECS_15MM[0],
            lut: &LUT_P1,
        },
        Part {
            spec: SPECS_15MM[1],
            lut: &LUT_P2,
        },
        Part {
            spec: SPECS_15MM[2],
            lut: &LUT_P3,
        },
        Part {
            spec: SPECS_15MM[3],
            lut: &LUT_P4,
        },
        Part {
            spec: SPECS_15MM[4],
            lut: &LUT_P5,
        },
    ],
    folds: FoldUse::All,
    step: 0.8,
    crop: None,
    post: Post::None,
};

pub static TOTAL_FASTEST: NnTask = NnTask {
    key: "total_fastest",
    label: "total, 6 mm",
    group: GROUP_CT,
    detail: "All 117 structures at 6 mm - a quick look, and the model the task catalogue \
             crops with.",
    modality: Modality::Ct,
    licence: Licence::Apache2,
    classes: &TOTAL_CLASS_NAMES,
    parts: &[Part {
        spec: SPEC_6MM,
        lut: &IDENTITY_LUT,
    }],
    folds: FoldUse::All,
    step: 0.8,
    crop: None,
    post: Post::None,
};

pub static TOTAL_V3_FAST: NnTask = NnTask {
    key: "total_v3_fast",
    label: "total v3, 3 mm",
    group: GROUP_CT,
    detail: "v3 weights (organs, cardiac and muscles retrained on 1830 subjects), all 117 \
             structures at 3 mm. Label 26 is vertebrae_L6.",
    modality: Modality::Ct,
    licence: Licence::Apache2,
    classes: &TOTAL_V3_CLASS_NAMES,
    parts: &[Part {
        spec: SPEC_V3_3MM,
        lut: &IDENTITY_LUT,
    }],
    folds: FoldUse::All,
    step: 0.8,
    crop: None,
    post: Post::None,
};

pub static TOTAL_V3: NnTask = NnTask {
    key: "total_v3",
    label: "total v3, 1.5 mm",
    group: GROUP_CT,
    detail: "v3 weights, five 1.5 mm sub-models - the reference quality of the newer \
             training. Slow without a GPU.",
    modality: Modality::Ct,
    licence: Licence::Apache2,
    classes: &TOTAL_V3_CLASS_NAMES,
    parts: &[
        Part {
            spec: SPECS_V3_15MM[0],
            lut: &LUT_P1,
        },
        Part {
            spec: SPECS_V3_15MM[1],
            lut: &LUT_P2,
        },
        Part {
            spec: SPECS_V3_15MM[2],
            lut: &LUT_P3,
        },
        Part {
            spec: SPECS_V3_15MM[3],
            lut: &LUT_P4,
        },
        Part {
            spec: SPECS_V3_15MM[4],
            lut: &LUT_P5,
        },
    ],
    folds: FoldUse::All,
    step: 0.8,
    crop: None,
    post: Post::None,
};

pub static TOTAL_V3_FASTEST: NnTask = NnTask {
    key: "total_v3_fastest",
    label: "total v3, 6 mm",
    group: GROUP_CT,
    detail: "v3 weights at 6 mm - a quick look.",
    modality: Modality::Ct,
    licence: Licence::Apache2,
    classes: &TOTAL_V3_CLASS_NAMES,
    parts: &[Part {
        spec: SPEC_V3_6MM,
        lut: &IDENTITY_LUT,
    }],
    folds: FoldUse::All,
    step: 0.8,
    crop: None,
    post: Post::None,
};

pub static TOTAL_V3_SMALL_FAST: NnTask = NnTask {
    key: "total_v3_small_fast",
    label: "total v3 small, 3 mm",
    group: GROUP_CT,
    detail: "v3 residual-encoder network (TotalSegmentator's 'small' size, 8 base \
             features), all 117 structures at 3 mm; a 128-cubed patch.",
    modality: Modality::Ct,
    licence: Licence::Apache2,
    classes: &TOTAL_V3_CLASS_NAMES,
    parts: &[Part {
        spec: SPEC_V3_SMALL_3MM,
        lut: &IDENTITY_LUT,
    }],
    folds: FoldUse::All,
    step: 0.8,
    crop: None,
    post: Post::None,
};

pub static TOTAL_V3_SMALL: NnTask = NnTask {
    key: "total_v3_small",
    label: "total v3 small, 1.5 mm",
    group: GROUP_CT,
    detail: "v3 residual-encoder sub-models at 1.5 mm; 192-cubed patches, GPU recommended.",
    modality: Modality::Ct,
    licence: Licence::Apache2,
    classes: &TOTAL_V3_CLASS_NAMES,
    parts: &[
        Part {
            spec: SPECS_V3_SMALL_15MM[0],
            lut: &LUT_P1,
        },
        Part {
            spec: SPECS_V3_SMALL_15MM[1],
            lut: &LUT_P2,
        },
        Part {
            spec: SPECS_V3_SMALL_15MM[2],
            lut: &LUT_P3,
        },
        Part {
            spec: SPECS_V3_SMALL_15MM[3],
            lut: &LUT_P4,
        },
        Part {
            spec: SPECS_V3_SMALL_15MM[4],
            lut: &LUT_P5,
        },
    ],
    folds: FoldUse::All,
    step: 0.8,
    crop: None,
    post: Post::None,
};

// ---- MR ---------------------------------------------------------------------

pub static TOTAL_MR_FAST: NnTask = NnTask {
    key: "total_mr_fast",
    label: "total MR, 3 mm",
    group: GROUP_MR,
    detail: "TotalSegmentator MRI: 50 organs, vessels, bones and muscles on any MR sequence, \
             one network at 3 mm.",
    modality: Modality::Mr,
    licence: Licence::Apache2,
    classes: &CLASSES_TOTAL_MR,
    parts: &[Part {
        spec: SPEC_TOTAL_MR_3MM,
        lut: &IDENTITY_LUT,
    }],
    folds: FoldUse::All,
    step: 0.8,
    crop: None,
    post: Post::None,
};

pub static TOTAL_MR: NnTask = NnTask {
    key: "total_mr",
    label: "total MR, 1.5 mm",
    group: GROUP_MR,
    detail: "TotalSegmentator MRI at 1.5 mm, two sub-models (organs, muscles) - the \
             reference quality.",
    modality: Modality::Mr,
    licence: Licence::Apache2,
    classes: &CLASSES_TOTAL_MR,
    parts: &[
        Part {
            spec: SPEC_TOTAL_MR_PART1_ORGANS,
            lut: &LUT_TOTAL_MR_PART1_ORGANS,
        },
        Part {
            spec: SPEC_TOTAL_MR_PART2_MUSCLES,
            lut: &LUT_TOTAL_MR_PART2_MUSCLES,
        },
    ],
    folds: FoldUse::All,
    step: 0.8,
    crop: None,
    post: Post::None,
};

pub static TOTAL_MR_FASTEST: NnTask = NnTask {
    key: "total_mr_fastest",
    label: "total MR, 6 mm",
    group: GROUP_MR,
    detail: "TotalSegmentator MRI at 6 mm - a quick look.",
    modality: Modality::Mr,
    licence: Licence::Apache2,
    classes: &CLASSES_TOTAL_MR,
    parts: &[Part {
        spec: SPEC_TOTAL_MR_6MM,
        lut: &IDENTITY_LUT,
    }],
    folds: FoldUse::All,
    step: 0.8,
    crop: None,
    post: Post::None,
};

/// MRSegmentator's 40 classes (its `dataset.json`, weights 1.2).
pub const CLASSES_MRSEGMENTATOR: [&str; 40] = [
    "spleen",
    "right_kidney",
    "left_kidney",
    "gallbladder",
    "liver",
    "stomach",
    "pancreas",
    "right_adrenal_gland",
    "left_adrenal_gland",
    "left_lung",
    "right_lung",
    "heart",
    "aorta",
    "inferior_vena_cava",
    "portal_vein_and_splenic_vein",
    "left_iliac_artery",
    "right_iliac_artery",
    "left_iliac_vena",
    "right_iliac_vena",
    "esophagus",
    "small_bowel",
    "duodenum",
    "colon",
    "urinary_bladder",
    "spine",
    "sacrum",
    "left_hip",
    "right_hip",
    "left_femur",
    "right_femur",
    "left_autochthonous_muscle",
    "right_autochthonous_muscle",
    "left_iliopsoas_muscle",
    "right_iliopsoas_muscle",
    "left_gluteus_maximus",
    "right_gluteus_maximus",
    "left_gluteus_medius",
    "right_gluteus_medius",
    "left_gluteus_minimus",
    "right_gluteus_minimus",
];

pub static MRSEGMENTATOR: NnTask = NnTask {
    key: "mrsegmentator",
    label: "MRSegmentator",
    group: GROUP_MR,
    detail: "MRSegmentator 1.2 (Haentze et al., Radiology 2025): 40 structures on MR (T1, \
             T2, Dixon) and on CT, the five cross-validation folds ensembled as upstream \
             does by default.",
    modality: Modality::CtMr,
    licence: Licence::Apache2,
    classes: &CLASSES_MRSEGMENTATOR,
    parts: &[Part {
        spec: SPEC_MRSEGMENTATOR,
        lut: &IDENTITY_LUT,
    }],
    folds: FoldUse::All,
    step: 0.5,
    crop: None,
    post: Post::None,
};

pub static MRSEGMENTATOR_FOLD0: NnTask = NnTask {
    key: "mrsegmentator_fold0",
    label: "MRSegmentator, one fold",
    group: GROUP_MR,
    detail: "MRSegmentator with fold 0 only: a fifth of the work of the ensemble, a little \
             less accurate.",
    modality: Modality::CtMr,
    licence: Licence::Apache2,
    classes: &CLASSES_MRSEGMENTATOR,
    parts: &[Part {
        spec: SPEC_MRSEGMENTATOR,
        lut: &IDENTITY_LUT,
    }],
    folds: FoldUse::First,
    step: 0.5,
    crop: None,
    post: Post::None,
};

// ---- body outline -----------------------------------------------------------

/// The body tasks' two classes, CT and MR alike.
pub const CLASSES_BODY: [&str; 2] = ["body_trunc", "body_extremities"];

pub static BODY_FAST: NnTask = NnTask {
    key: "body_fast",
    label: "body, 6 mm",
    group: GROUP_BODY,
    detail: "Patient outline, trunk and extremities, at 6 mm - what the body-contour tool's \
             model-assisted method uses on CT.",
    modality: Modality::Ct,
    licence: Licence::Apache2,
    classes: &CLASSES_BODY,
    parts: &[Part {
        spec: SPEC_BODY_6MM,
        lut: &IDENTITY_LUT,
    }],
    folds: FoldUse::All,
    step: 0.5,
    crop: None,
    post: Post::Body,
};

pub static BODY: NnTask = NnTask {
    key: "body",
    label: "body, 1.5 mm",
    group: GROUP_BODY,
    detail: "Patient outline at 1.5 mm.",
    modality: Modality::Ct,
    licence: Licence::Apache2,
    classes: &CLASSES_BODY,
    parts: &[Part {
        spec: SPEC_BODY_15MM,
        lut: &IDENTITY_LUT,
    }],
    folds: FoldUse::All,
    step: 0.5,
    crop: None,
    post: Post::Body,
};

pub static BODY_MR_FAST: NnTask = NnTask {
    key: "body_mr_fast",
    label: "body MR, 6 mm",
    group: GROUP_BODY,
    detail: "Patient outline on MR at 6 mm.",
    modality: Modality::Mr,
    licence: Licence::Apache2,
    classes: &CLASSES_BODY,
    parts: &[Part {
        spec: SPEC_BODY_MR_6MM,
        lut: &IDENTITY_LUT,
    }],
    folds: FoldUse::All,
    step: 0.5,
    crop: None,
    post: Post::None,
};

pub static BODY_MR: NnTask = NnTask {
    key: "body_mr",
    label: "body MR, 1.5 mm",
    group: GROUP_BODY,
    detail: "Patient outline on MR at 1.5 mm.",
    modality: Modality::Mr,
    licence: Licence::Apache2,
    classes: &CLASSES_BODY,
    parts: &[Part {
        spec: SPEC_BODY_MR,
        lut: &IDENTITY_LUT,
    }],
    folds: FoldUse::All,
    step: 0.5,
    crop: None,
    post: Post::None,
};

/// The family, in the order the interface lists it.
pub static TASKS: [&NnTask; 17] = [
    &TOTAL_FAST,
    &TOTAL,
    &TOTAL_FASTEST,
    &TOTAL_V3_FAST,
    &TOTAL_V3,
    &TOTAL_V3_FASTEST,
    &TOTAL_V3_SMALL_FAST,
    &TOTAL_V3_SMALL,
    &TOTAL_MR_FAST,
    &TOTAL_MR,
    &TOTAL_MR_FASTEST,
    &MRSEGMENTATOR,
    &MRSEGMENTATOR_FOLD0,
    &BODY_FAST,
    &BODY,
    &BODY_MR_FAST,
    &BODY_MR,
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autoseg::classes::{PART_CLASSES, PART_OFFSET};

    #[test]
    fn the_five_part_tables_tile_the_117_classes() {
        let luts: [&[u8]; 5] = [&LUT_P1, &LUT_P2, &LUT_P3, &LUT_P4, &LUT_P5];
        let mut seen = [false; 118];
        for (p, lut) in luts.iter().enumerate() {
            assert_eq!(lut.len(), PART_CLASSES[p] + 1);
            assert_eq!(lut[1], PART_OFFSET[p] + 1);
            for &l in &lut[1..] {
                assert!(!seen[l as usize]);
                seen[l as usize] = true;
            }
        }
        assert!(seen[1..].iter().all(|s| *s));
    }

    #[test]
    fn the_mr_parts_cover_the_mr_table() {
        let mut seen = [false; 51];
        for &l in LUT_TOTAL_MR_PART1_ORGANS[1..]
            .iter()
            .chain(&LUT_TOTAL_MR_PART2_MUSCLES[1..])
        {
            seen[l as usize] = true;
        }
        assert_eq!(CLASSES_TOTAL_MR.len(), 50);
        assert!(seen[1..].iter().all(|s| *s));
    }
}
