//! AAPM TG-263 names for the classes the models produce.
//!
//! Ten models spell the same organ ten ways (`kidney_left`, `left_kidney`,
//! `left kidney`); a structure set built from two of them should read as
//! one set, in the nomenclature a treatment planning system and a dose
//! audit expect. This is the look-up from a model's class name to the
//! TG-263 primary name (Mayo et al., *American Association of Physicists in
//! Medicine Task Group 263: Standardizing Nomenclatures in Radiation
//! Oncology*, IJROBP 100(4), 2018), applied when the results window is set
//! to TG-263 names.
//!
//! Only classes with an exact TG-263 counterpart are mapped. A class that is
//! a union TG-263 splits (`portal_vein_and_splenic_vein`), a part TG-263
//! does not name (`gluteus_maximus_left`, the Couinaud segments, `sternum`),
//! or one whose extent differs between the model and the TG-263 definition
//! (`iliac_artery_left` is not certain to be the common iliac only) keeps
//! the model's own name. Every target below is a primary name of the TG-263
//! table as published; that check was made against the table outside the
//! repository, as the table itself is not part of it.

/// Model class name (any of the models' spellings: lower case, words
/// joined by `_` or by spaces) → TG-263 primary name.
const TABLE: &[(&str, &str)] = &[
    // abdomen
    ("spleen", "Spleen"),
    ("kidney_right", "Kidney_R"),
    ("kidney_left", "Kidney_L"),
    ("right_kidney", "Kidney_R"),
    ("left_kidney", "Kidney_L"),
    ("gallbladder", "Gallbladder"),
    ("liver", "Liver"),
    ("stomach", "Stomach"),
    ("pancreas", "Pancreas"),
    ("adrenal_gland_right", "Glnd_Adrenal_R"),
    ("adrenal_gland_left", "Glnd_Adrenal_L"),
    ("right_adrenal_gland", "Glnd_Adrenal_R"),
    ("left_adrenal_gland", "Glnd_Adrenal_L"),
    ("esophagus", "Esophagus"),
    ("small_bowel", "Bowel_Small"),
    ("duodenum", "Duodenum"),
    ("colon", "Colon"),
    ("urinary_bladder", "Bladder"),
    ("bladder", "Bladder"),
    ("prostate", "Prostate"),
    ("rectum", "Rectum"),
    // thorax
    ("lung_upper_lobe_left", "Lung_LUL"),
    ("lung_lower_lobe_left", "Lung_LLL"),
    ("lung_upper_lobe_right", "Lung_RUL"),
    ("lung_middle_lobe_right", "Lung_RML"),
    ("lung_lower_lobe_right", "Lung_RLL"),
    ("left_lung_upper_lobe", "Lung_LUL"),
    ("left_lung_lower_lobe", "Lung_LLL"),
    ("right_lung_upper_lobe", "Lung_RUL"),
    ("right_lung_middle_lobe", "Lung_RML"),
    ("right_lung_lower_lobe", "Lung_RLL"),
    ("lung_left", "Lung_L"),
    ("lung_right", "Lung_R"),
    ("left_lung", "Lung_L"),
    ("right_lung", "Lung_R"),
    ("lung", "Lungs"),
    ("lungs", "Lungs"),
    ("trachea", "Trachea"),
    ("thyroid_gland", "Glnd_Thyroid"),
    ("thyroid", "Glnd_Thyroid"),
    ("heart", "Heart"),
    ("heart_atrium_left", "Atrium_L"),
    ("heart_atrium_right", "Atrium_R"),
    ("heart_ventricle_left", "Ventricle_L"),
    ("heart_ventricle_right", "Ventricle_R"),
    ("left_atrium", "Atrium_L"),
    ("right_atrium", "Atrium_R"),
    ("left_ventricle", "Ventricle_L"),
    ("right_ventricle", "Ventricle_R"),
    ("pericardium", "Pericardium"),
    ("mediastinum", "Mediastinum"),
    ("breast", "Breasts"),
    ("spinal_cord", "SpinalCord"),
    ("spinal_canal", "SpinalCanal"),
    // vessels
    ("aorta", "A_Aorta"),
    ("pulmonary_artery", "A_Pulmonary"),
    ("pulmonary_vein", "V_Pulmonary"),
    ("brachiocephalic_trunk", "A_Brachiocephls"),
    ("subclavian_artery_right", "A_Subclavian_R"),
    ("subclavian_artery_left", "A_Subclavian_L"),
    ("common_carotid_artery_right", "A_Carotid_R"),
    ("common_carotid_artery_left", "A_Carotid_L"),
    ("brachiocephalic_vein_left", "V_Brachioceph_L"),
    ("brachiocephalic_vein_right", "V_Brachioceph_R"),
    ("superior_vena_cava", "V_Venacava_S"),
    ("inferior_vena_cava", "V_Venacava_I"),
    ("portal_vein", "V_Portal"),
    ("internal_jugular_vein_right", "V_Jugular_Int_R"),
    ("internal_jugular_vein_left", "V_Jugular_Int_L"),
    // bones
    ("sacrum", "Sacrum"),
    ("humerus_left", "Humerus_L"),
    ("humerus_right", "Humerus_R"),
    ("scapula_left", "Scapula_L"),
    ("scapula_right", "Scapula_R"),
    ("clavicula_left", "Clavicle_L"),
    ("clavicula_right", "Clavicle_R"),
    ("femur_left", "Femur_L"),
    ("femur_right", "Femur_R"),
    ("left_femur", "Femur_L"),
    ("right_femur", "Femur_R"),
    ("hip_left", "Bone_Pelvic_L"),
    ("hip_right", "Bone_Pelvic_R"),
    ("left_hip", "Bone_Pelvic_L"),
    ("right_hip", "Bone_Pelvic_R"),
    ("skull", "Skull"),
    ("mandible", "Bone_Mandible"),
    ("hyoid", "Bone_Hyoid"),
    ("thyroid_cartilage", "Cartlg_Thyroid"),
    ("cricoid_cartilage", "Cricoid"),
    ("zygomatic_arch_right", "Bone_Zygomatic_R"),
    ("zygomatic_arch_left", "Bone_Zygomatic_L"),
    // head and neck
    ("brain", "Brain"),
    ("eye_left", "Eye_L"),
    ("eye_right", "Eye_R"),
    ("eyeball_left", "Eye_L"),
    ("eyeball_right", "Eye_R"),
    ("eye_lens_left", "Lens_L"),
    ("eye_lens_right", "Lens_R"),
    ("optic_nerve_left", "OpticNrv_L"),
    ("optic_nerve_right", "OpticNrv_R"),
    ("parotid_gland_left", "Parotid_L"),
    ("parotid_gland_right", "Parotid_R"),
    ("submandibular_gland_left", "Glnd_Submand_L"),
    ("submandibular_gland_right", "Glnd_Submand_R"),
    ("nasopharynx", "Nasopharynx"),
    ("oropharynx", "Oropharynx"),
    ("hypopharynx", "Laryngl_Pharynx"),
    ("soft_palate", "Palate_Soft"),
    ("hard_palate", "Hardpalate"),
    ("sinus_maxillary", "Sinus_Maxilry"),
    ("sinus_frontal", "Sinus_Frontal"),
    ("masseter_left", "Musc_Masseter_L"),
    ("masseter_right", "Musc_Masseter_R"),
    ("temporalis_left", "Musc_Temporal_L"),
    ("temporalis_right", "Musc_Temporal_R"),
    ("lateral_pterygoid_left", "Pterygoid_Lat_L"),
    ("lateral_pterygoid_right", "Pterygoid_Lat_R"),
    ("medial_pterygoid_left", "Pterygoid_Med_L"),
    ("medial_pterygoid_right", "Pterygoid_Med_R"),
    ("digastric_left", "Musc_Digastric_L"),
    ("digastric_right", "Musc_Digastric_R"),
    ("tongue", "Tongue"),
    ("sternocleidomastoid_left", "Musc_Sclmast_L"),
    ("sternocleidomastoid_right", "Musc_Sclmast_R"),
    ("platysma_left", "Musc_Platysma_L"),
    ("platysma_right", "Musc_Platysma_R"),
    ("superior_pharyngeal_constrictor", "Musc_Constrict_S"),
    ("middle_pharyngeal_constrictor", "Musc_Constrict_M"),
    ("inferior_pharyngeal_constrictor", "Musc_Constrict_I"),
    // body
    ("body", "Body"),
];

/// The TG-263 primary name of a model class, if it has one.
pub fn tg263_name(class: &str) -> Option<String> {
    let key: String = class
        .trim()
        .to_ascii_lowercase()
        .chars()
        .map(|c| if c == ' ' || c == '-' { '_' } else { c })
        .collect();
    if let Some((_, t)) = TABLE.iter().find(|(k, _)| *k == key) {
        return Some((*t).to_string());
    }
    // Vertebrae: `vertebrae_T7` → `VB_T07` (TG-263 pads the thoracic
    // levels), `vertebrae_C1` → `VB_C1`, `vertebrae_S1` → `VB_S1`.
    if let Some(level) = key.strip_prefix("vertebrae_") {
        let (region, n) = level.split_at(1);
        let n: u32 = n.parse().ok()?;
        return match region {
            "c" if (1..=7).contains(&n) => Some(format!("VB_C{n}")),
            "t" if (1..=12).contains(&n) => Some(format!("VB_T{n:02}")),
            "l" if (1..=5).contains(&n) => Some(format!("VB_L{n}")),
            "s" if (1..=5).contains(&n) => Some(format!("VB_S{n}")),
            _ => None,
        };
    }
    // Ribs: `rib_left_3` → `Rib03_L`.
    for (side, tag) in [("left", "L"), ("right", "R")] {
        if let Some(n) = key.strip_prefix(&format!("rib_{side}_")) {
            let n: u32 = n.parse().ok()?;
            if (1..=12).contains(&n) {
                return Some(format!("Rib{n:02}_{tag}"));
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_models_spellings_meet_in_one_name() {
        assert_eq!(tg263_name("kidney_left").as_deref(), Some("Kidney_L"));
        assert_eq!(tg263_name("left_kidney").as_deref(), Some("Kidney_L"));
        assert_eq!(tg263_name("Left Kidney").as_deref(), Some("Kidney_L"));
        assert_eq!(tg263_name("vertebrae_T7").as_deref(), Some("VB_T07"));
        assert_eq!(tg263_name("vertebrae_C1").as_deref(), Some("VB_C1"));
        assert_eq!(tg263_name("vertebrae_S1").as_deref(), Some("VB_S1"));
        assert_eq!(tg263_name("vertebrae_L6"), None);
        assert_eq!(tg263_name("rib_right_12").as_deref(), Some("Rib12_R"));
        assert_eq!(tg263_name("rib_left_13"), None);
        assert_eq!(tg263_name("gluteus_maximus_left"), None);
        assert_eq!(tg263_name("portal_vein_and_splenic_vein"), None);
    }

    #[test]
    fn every_total_class_either_maps_or_is_known_not_to() {
        let mapped = crate::autoseg::classes::TOTAL_CLASS_NAMES
            .iter()
            .filter(|n| tg263_name(n).is_some())
            .count();
        // 117 classes; the muscles, the cysts, the sternum, the costal
        // cartilages and the unions TG-263 has no single name for stay as
        // they are.
        assert!(mapped >= 90, "{mapped}");
        let keys: Vec<&str> = TABLE.iter().map(|(k, _)| *k).collect();
        let mut dedup = keys.clone();
        dedup.sort();
        dedup.dedup();
        assert_eq!(dedup.len(), keys.len(), "duplicate key");
    }
}
