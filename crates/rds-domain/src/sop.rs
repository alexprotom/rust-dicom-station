//! SOP Class UIDs: the one table of the storage classes this station reads,
//! writes or has to recognise, and the short names a listing shows them by.
//!
//! The loader's classification ([`crate::loader::classify`]), the readers
//! (`extras`, `dicomseg`, `report`), the writers (`dicom_export`,
//! `gen_test_data`) and the MCP archive listing all take their UIDs from
//! here, so a class is spelt once. The values are those of DICOM PS3.4
//! Annex B (Storage Service Class), which names every one of them.

/// CT Image Storage.
pub const SOP_CT: &str = "1.2.840.10008.5.1.4.1.1.2";
/// Digital X-Ray Image Storage, for presentation.
pub const SOP_DX_PRESENTATION: &str = "1.2.840.10008.5.1.4.1.1.1.1";
/// Digital X-Ray Image Storage, for processing.
pub const SOP_DX_PROCESSING: &str = "1.2.840.10008.5.1.4.1.1.1.1.1";
/// Secondary Capture Image Storage; its multi-frame variants append
/// `.1` to `.4` ([`is_secondary_capture`]).
pub const SOP_SECONDARY_CAPTURE: &str = "1.2.840.10008.5.1.4.1.1.7";
/// RT Image Storage.
pub const SOP_RTIMAGE: &str = "1.2.840.10008.5.1.4.1.1.481.1";
/// RT Dose Storage.
pub const SOP_RTDOSE: &str = "1.2.840.10008.5.1.4.1.1.481.2";
/// RT Structure Set Storage.
pub const SOP_RTSTRUCT: &str = "1.2.840.10008.5.1.4.1.1.481.3";
/// RT Beams Treatment Record Storage.
pub const SOP_RT_BEAMS_RECORD: &str = "1.2.840.10008.5.1.4.1.1.481.4";
/// RT Plan Storage.
pub const SOP_RTPLAN: &str = "1.2.840.10008.5.1.4.1.1.481.5";
/// RT Ion Plan Storage.
pub const SOP_RTIONPLAN: &str = "1.2.840.10008.5.1.4.1.1.481.8";
/// RT Ion Beams Treatment Record Storage.
pub const SOP_RT_ION_BEAMS_RECORD: &str = "1.2.840.10008.5.1.4.1.1.481.9";
/// Raw Data Storage: bytes only their maker reads.
pub const SOP_RAW_DATA: &str = "1.2.840.10008.5.1.4.1.1.66";
/// Spatial Registration Storage (rigid matrices).
pub const SOP_SPATIAL_REG: &str = "1.2.840.10008.5.1.4.1.1.66.1";
/// Deformable Spatial Registration Storage.
pub const SOP_DEFORMABLE_REG: &str = "1.2.840.10008.5.1.4.1.1.66.3";
/// Segmentation Storage (DICOM SEG).
pub const SOP_SEG: &str = "1.2.840.10008.5.1.4.1.1.66.4";
/// Key Object Selection Document: a list of references, encoded as an SR
/// but describing nothing of its own.
pub const SOP_KEY_OBJECT_SELECTION: &str = "1.2.840.10008.5.1.4.1.1.88.59";
/// Encapsulated PDF Storage.
pub const SOP_ENCAPSULATED_PDF: &str = "1.2.840.10008.5.1.4.1.1.104.1";
/// Detached Study Management, the SOP class an RTSTRUCT names a study with.
pub const SOP_DETACHED_STUDY: &str = "1.2.840.10008.3.1.2.3.1";

/// The prefixes of whole families of storage classes, each followed by a
/// sub-number: `…1.11.n` the presentation states, `…1.9.n.n` the
/// waveforms, `…1.104.n` the encapsulated documents (PDF, CDA, STL, OBJ,
/// MTL), `…1.88.n` the Structured Reports.
const PRESENTATION_STATE_PREFIX: &str = "1.2.840.10008.5.1.4.1.1.11.";
const WAVEFORM_PREFIX: &str = "1.2.840.10008.5.1.4.1.1.9.";
const ENCAPSULATED_PREFIX: &str = "1.2.840.10008.5.1.4.1.1.104.";
const SR_PREFIX: &str = "1.2.840.10008.5.1.4.1.1.88.";

/// Is this a (multi-frame) Secondary Capture SOP class?
pub fn is_secondary_capture(sop_class: &str) -> bool {
    sop_class == SOP_SECONDARY_CAPTURE
        || sop_class
            .strip_prefix(SOP_SECONDARY_CAPTURE)
            .is_some_and(|rest| matches!(rest, ".1" | ".2" | ".3" | ".4"))
}

/// Is this a Spatial Registration class, rigid or deformable?
pub fn is_registration(sop_class: &str) -> bool {
    sop_class == SOP_SPATIAL_REG || sop_class == SOP_DEFORMABLE_REG
}

/// Is this a beams treatment record this station reads (photon or ion)?
pub fn is_beams_record(sop_class: &str) -> bool {
    sop_class == SOP_RT_BEAMS_RECORD || sop_class == SOP_RT_ION_BEAMS_RECORD
}

/// Is this a Structured Report class whose content tree is worth reading
/// as a report? Every class under the SR prefix but the Key Object
/// Selection, which holds references and no content.
pub fn is_structured_report(sop_class: &str) -> bool {
    sop_class.starts_with(SR_PREFIX) && sop_class != SOP_KEY_OBJECT_SELECTION
}

/// Does this SOP class carry no image at all, so that nothing here can show
/// it: an encapsulated document (a PDF, a CDA, a surface mesh), a
/// presentation state, a key object selection, raw data, a waveform.
///
/// Their Modality is no help (a PDF of a report says `OT` or `DOC`, a
/// presentation state `PR`, raw data anything), and they carry no Image
/// Position either, so without this they would look like single images
/// and fail when their pixels are decoded.
pub fn is_non_image(sop_class: &str) -> bool {
    sop_class.starts_with(ENCAPSULATED_PREFIX)
        || sop_class.starts_with(PRESENTATION_STATE_PREFIX)
        || sop_class.starts_with(WAVEFORM_PREFIX)
        || sop_class == SOP_KEY_OBJECT_SELECTION
        || sop_class == SOP_RAW_DATA
}

/// A short name for a storage class, for listings and warnings: `None`
/// for one this table does not know, which the caller shows as the UID.
pub fn name(uid: &str) -> Option<&'static str> {
    let n = match uid {
        SOP_CT => "CT Image",
        "1.2.840.10008.5.1.4.1.1.2.1" => "Enhanced CT Image",
        "1.2.840.10008.5.1.4.1.1.4" => "MR Image",
        "1.2.840.10008.5.1.4.1.1.4.1" => "Enhanced MR Image",
        "1.2.840.10008.5.1.4.1.1.128" => "PET Image",
        "1.2.840.10008.5.1.4.1.1.130" => "Enhanced PET Image",
        "1.2.840.10008.5.1.4.1.1.20" => "NM Image",
        "1.2.840.10008.5.1.4.1.1.6.1" => "Ultrasound Image",
        "1.2.840.10008.5.1.4.1.1.3.1" => "Ultrasound Multi-frame Image",
        SOP_SECONDARY_CAPTURE => "Secondary Capture Image",
        "1.2.840.10008.5.1.4.1.1.7.1" => "Multi-frame Single Bit SC Image",
        "1.2.840.10008.5.1.4.1.1.7.2" => "Multi-frame Grayscale Byte SC Image",
        "1.2.840.10008.5.1.4.1.1.7.3" => "Multi-frame Grayscale Word SC Image",
        "1.2.840.10008.5.1.4.1.1.7.4" => "Multi-frame True Color SC Image",
        "1.2.840.10008.5.1.4.1.1.1" => "CR Image",
        SOP_DX_PRESENTATION => "DX Image (presentation)",
        SOP_DX_PROCESSING => "DX Image (processing)",
        "1.2.840.10008.5.1.4.1.1.12.1" => "X-Ray Angiographic Image",
        "1.2.840.10008.5.1.4.1.1.12.2" => "X-Ray Radiofluoroscopic Image",
        SOP_RTIMAGE => "RT Image",
        SOP_RTDOSE => "RT Dose",
        SOP_RTSTRUCT => "RT Structure Set",
        SOP_RT_BEAMS_RECORD => "RT Beams Treatment Record",
        SOP_RTPLAN => "RT Plan",
        "1.2.840.10008.5.1.4.1.1.481.6" => "RT Brachy Treatment Record",
        "1.2.840.10008.5.1.4.1.1.481.7" => "RT Treatment Summary Record",
        SOP_RTIONPLAN => "RT Ion Plan",
        SOP_RT_ION_BEAMS_RECORD => "RT Ion Beams Treatment Record",
        SOP_RAW_DATA => "Raw Data",
        SOP_SPATIAL_REG => "Spatial Registration",
        SOP_DEFORMABLE_REG => "Deformable Spatial Registration",
        SOP_SEG => "Segmentation",
        "1.2.840.10008.5.1.4.1.1.88.11" => "Basic Text SR",
        "1.2.840.10008.5.1.4.1.1.88.22" => "Enhanced SR",
        "1.2.840.10008.5.1.4.1.1.88.33" => "Comprehensive SR",
        "1.2.840.10008.5.1.4.1.1.88.34" => "Comprehensive 3D SR",
        SOP_KEY_OBJECT_SELECTION => "Key Object Selection",
        "1.2.840.10008.5.1.4.1.1.88.67" => "X-Ray Radiation Dose SR",
        SOP_ENCAPSULATED_PDF => "Encapsulated PDF",
        "1.2.840.10008.5.1.4.1.1.104.2" => "Encapsulated CDA",
        "1.2.840.10008.5.1.4.1.1.104.3" => "Encapsulated STL",
        "1.2.840.10008.5.1.4.1.1.104.4" => "Encapsulated OBJ",
        "1.2.840.10008.5.1.4.1.1.104.5" => "Encapsulated MTL",
        "1.2.840.10008.5.1.4.1.1.11.1" => "Grayscale Softcopy Presentation State",
        "1.2.840.10008.5.1.4.1.1.11.2" => "Color Softcopy Presentation State",
        "1.2.840.10008.5.1.4.1.1.11.3" => "Pseudo-Color Softcopy Presentation State",
        "1.2.840.10008.5.1.4.1.1.11.4" => "Blending Softcopy Presentation State",
        _ => return None,
    };
    Some(n)
}

/// [`name`], or a family name for a class of a known family, or the UID
/// itself: what a listing prints, never empty for a non-empty UID.
pub fn name_or_uid(uid: &str) -> String {
    if let Some(n) = name(uid) {
        return n.to_string();
    }
    let family = if uid.starts_with(ENCAPSULATED_PREFIX) {
        "Encapsulated document"
    } else if uid.starts_with(PRESENTATION_STATE_PREFIX) {
        "Presentation State"
    } else if uid.starts_with(WAVEFORM_PREFIX) {
        "Waveform"
    } else if uid.starts_with(SR_PREFIX) {
        "Structured Report"
    } else {
        return uid.to_string();
    };
    format!("{family} ({uid})")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn objects_without_images_are_told_apart_from_images() {
        for uid in [
            SOP_ENCAPSULATED_PDF,
            "1.2.840.10008.5.1.4.1.1.104.2",
            "1.2.840.10008.5.1.4.1.1.104.3",
            "1.2.840.10008.5.1.4.1.1.11.1",
            "1.2.840.10008.5.1.4.1.1.11.4",
            SOP_KEY_OBJECT_SELECTION,
            SOP_RAW_DATA,
            "1.2.840.10008.5.1.4.1.1.9.1.1",
        ] {
            assert!(is_non_image(uid), "{uid}");
        }
        // Prefixes are whole components: CR (`…1.1`) is not a presentation
        // state (`…1.11.`), the registrations under Raw Data's number are
        // not raw data, an SR is a report.
        for uid in [
            "1.2.840.10008.5.1.4.1.1.1",
            SOP_CT,
            SOP_SPATIAL_REG,
            SOP_SEG,
            SOP_RTIMAGE,
            "1.2.840.10008.5.1.4.1.1.88.67",
            SOP_SECONDARY_CAPTURE,
        ] {
            assert!(!is_non_image(uid), "{uid}");
        }
        assert!(is_structured_report("1.2.840.10008.5.1.4.1.1.88.67"));
        assert!(!is_structured_report(SOP_KEY_OBJECT_SELECTION));
    }

    #[test]
    fn every_class_a_listing_meets_has_a_name() {
        assert_eq!(
            name(SOP_RT_ION_BEAMS_RECORD),
            Some("RT Ion Beams Treatment Record")
        );
        assert_eq!(name(SOP_RT_BEAMS_RECORD), Some("RT Beams Treatment Record"));
        assert_eq!(name(SOP_KEY_OBJECT_SELECTION), Some("Key Object Selection"));
        assert_eq!(
            name("1.2.840.10008.5.1.4.1.1.104.4"),
            Some("Encapsulated OBJ")
        );
        assert_eq!(name("1.2.3"), None);
        assert_eq!(name_or_uid("1.2.3"), "1.2.3");
        assert_eq!(
            name_or_uid("1.2.840.10008.5.1.4.1.1.11.9"),
            "Presentation State (1.2.840.10008.5.1.4.1.1.11.9)"
        );
    }
}
