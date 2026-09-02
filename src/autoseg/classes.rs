//! TotalSegmentator v2 class tables.
//!
//! Global label ids 1..=117 follow `class_map["total"]` of
//! `totalsegmentator/map_to_binary.py` (v2.x). The five 1.5 mm sub-models
//! (nnU-Net datasets 291-295) each predict a *local* contiguous label range
//! that maps onto a contiguous slice of the global table, in this order:
//! organs (1-24), vertebrae (25-50), cardiac (51-68), muscles (69-91),
//! ribs (92-117). The 3 mm / 6 mm single models predict the global ids
//! directly.

/// Global class names, index 0 = label 1 ("spleen") … index 116 = label 117.
pub const TOTAL_CLASS_NAMES: [&str; 117] = [
    "spleen",
    "kidney_right",
    "kidney_left",
    "gallbladder",
    "liver",
    "stomach",
    "pancreas",
    "adrenal_gland_right",
    "adrenal_gland_left",
    "lung_upper_lobe_left",
    "lung_lower_lobe_left",
    "lung_upper_lobe_right",
    "lung_middle_lobe_right",
    "lung_lower_lobe_right",
    "esophagus",
    "trachea",
    "thyroid_gland",
    "small_bowel",
    "duodenum",
    "colon",
    "urinary_bladder",
    "prostate",
    "kidney_cyst_left",
    "kidney_cyst_right",
    "sacrum",
    "vertebrae_S1",
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
    "heart",
    "aorta",
    "pulmonary_vein",
    "brachiocephalic_trunk",
    "subclavian_artery_right",
    "subclavian_artery_left",
    "common_carotid_artery_right",
    "common_carotid_artery_left",
    "brachiocephalic_vein_left",
    "brachiocephalic_vein_right",
    "atrial_appendage_left",
    "superior_vena_cava",
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
    "spinal_cord",
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
    "skull",
    "rib_left_1",
    "rib_left_2",
    "rib_left_3",
    "rib_left_4",
    "rib_left_5",
    "rib_left_6",
    "rib_left_7",
    "rib_left_8",
    "rib_left_9",
    "rib_left_10",
    "rib_left_11",
    "rib_left_12",
    "rib_right_1",
    "rib_right_2",
    "rib_right_3",
    "rib_right_4",
    "rib_right_5",
    "rib_right_6",
    "rib_right_7",
    "rib_right_8",
    "rib_right_9",
    "rib_right_10",
    "rib_right_11",
    "rib_right_12",
    "sternum",
    "costal_cartilages",
];

/// Name of a global label (1-based); empty string for 0 / out of range.
pub fn class_name(label: u8) -> &'static str {
    if label == 0 || label as usize > TOTAL_CLASS_NAMES.len() {
        ""
    } else {
        TOTAL_CLASS_NAMES[label as usize - 1]
    }
}

/// Global-label offset for each of the five 1.5 mm sub-models:
/// local label `l` (1-based) of part `p` maps to global `PART_OFFSET[p] + l`.
/// Part order = nnU-Net datasets 291, 292, 293, 294, 295.
pub const PART_OFFSET: [u8; 5] = [0, 24, 50, 68, 91];

/// Number of foreground classes per 1.5 mm sub-model (291..=295).
pub const PART_CLASSES: [usize; 5] = [24, 26, 18, 23, 26];

pub const PART_NAMES: [&str; 5] = ["organs", "vertebrae", "cardiac", "muscles", "ribs"];

/// Display color for a global label: curated for the common organs, a
/// golden-angle palette for the rest (stable across runs).
pub fn class_color(label: u8) -> [u8; 3] {
    match class_name(label) {
        "spleen" => [157, 108, 162],
        "kidney_right" | "kidney_left" => [185, 102, 83],
        "gallbladder" => [139, 150, 98],
        "liver" => [221, 130, 101],
        "stomach" => [216, 132, 105],
        "pancreas" => [249, 180, 111],
        "esophagus" => [211, 171, 143],
        "trachea" => [182, 228, 255],
        "thyroid_gland" => [220, 160, 30],
        "small_bowel" => [205, 167, 142],
        "duodenum" => [255, 253, 229],
        "colon" => [204, 168, 143],
        "urinary_bladder" => [222, 154, 132],
        "prostate" => [230, 158, 140],
        "heart" => [206, 110, 84],
        "aorta" => [224, 97, 76],
        "spinal_cord" => [244, 214, 49],
        "brain" => [250, 250, 225],
        "skull" => [241, 213, 144],
        "sternum" => [244, 217, 154],
        "costal_cartilages" => [200, 200, 235],
        n if n.starts_with("lung_") => [197, 165, 145],
        n if n.starts_with("vertebrae_") || n == "sacrum" => [226, 202, 134],
        n if n.starts_with("rib_") => [253, 232, 158],
        n if n.contains("artery") || n.contains("trunk") => [216, 101, 79],
        n if n.contains("vein") || n.contains("vena") || n.contains("atrial") => [0, 151, 206],
        n if n.contains("gluteus") || n.contains("autochthon") || n.contains("iliopsoas") => {
            [192, 104, 88]
        }
        n if n.contains("femur")
            || n.contains("hip")
            || n.contains("humerus")
            || n.contains("scapula")
            || n.contains("clavicula") =>
        {
            [212, 188, 102]
        }
        _ => {
            // golden-angle hue rotation, fixed saturation/value
            let h = (label as f32 * 137.508) % 360.0;
            hsv(h, 0.65, 0.85)
        }
    }
}

fn hsv(h: f32, s: f32, v: f32) -> [u8; 3] {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;
    let (r, g, b) = match (h / 60.0) as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    [
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_shape() {
        assert_eq!(TOTAL_CLASS_NAMES.len(), 117);
        assert_eq!(class_name(1), "spleen");
        assert_eq!(class_name(117), "costal_cartilages");
        assert_eq!(class_name(0), "");
        // part offsets + sizes tile the whole table contiguously
        let mut expect = 0u8;
        for p in 0..5 {
            assert_eq!(PART_OFFSET[p], expect);
            expect += PART_CLASSES[p] as u8;
        }
        assert_eq!(expect, 117);
        // global ids of part boundaries
        assert_eq!(class_name(PART_OFFSET[1] + 1), "sacrum");
        assert_eq!(class_name(PART_OFFSET[2] + 1), "heart");
        assert_eq!(class_name(PART_OFFSET[3] + 1), "humerus_left");
        assert_eq!(class_name(PART_OFFSET[4] + 1), "rib_left_1");
    }
}
