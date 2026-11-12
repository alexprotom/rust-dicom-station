//! A planar image with its 2D contours drawn into its pixels, written as an
//! RGB Secondary Capture.
//!
//! **Why there is such a thing.** The faithful form of 2D contours is a
//! Grayscale Softcopy Presentation State ([`crate::gsps`]): the image stays
//! as it was and the contours are graphics beside it, editable and read
//! back as structures. Many viewers never draw a presentation state's
//! graphics, though (most research tools among them), and an image whose
//! contours were written that way opens there as the bare image. A burned-in
//! copy is what any viewer shows as it is: the image through a window, as
//! 8-bit grey, with each structure's outline (and its points of interest)
//! drawn over it in the structure's colour. It is a picture, not data: the
//! contours cannot be read back out of it, which is why it goes *beside* the
//! image and its presentation state, never instead of them.
//!
//! **What is written** ([`write_secondary_capture`]): SOP class Secondary
//! Capture Image Storage (`1.2.840.10008.5.1.4.1.1.7`), modality `OT`,
//! Image Type `DERIVED\SECONDARY`, Conversion Type `WSD`, Burned In
//! Annotation `YES` (PS3.3 C.7.6.1.1.7: a reader is told the pixels carry
//! annotations), `RGB` at 8 bits a sample, interleaved (Planar Configuration
//! 0), Pixel Spacing when the image has one, and the image it was made from
//! in the Source Image Sequence with a Derivation Description.
//!
//! **How it is drawn** ([`render`]). The grey is the image's values through
//! the window `(centre, width)` exactly as the viewer's planar panes map
//! them (`(v - (c - w/2)) * 255 / w`, clamped); an RGB image (a secondary
//! capture already) keeps its own colours. A ring is a closed polyline in
//! the planar structures' pixel convention (the centre of the first pixel at
//! `(0, 0)`); each pixel takes the structure's colour by its coverage of a
//! stroke [`line_width`] pixels wide - one minus its distance to the nearest
//! segment, past half the width - so a line is antialiased and the same
//! thickness at any angle. Coverage is the maximum over a structure's
//! segments, not their sum, so a vertex where two segments meet is not
//! drawn twice as dark. A point of interest is an × of the same stroke.
//! Structures are drawn in the order given, each blended over what is there.

use std::path::Path;

use anyhow::{bail, Result};
use dicom_core::value::{PrimitiveValue, C};
use dicom_core::{DataElement, VR};
use dicom_dictionary_std::tags;
use dicom_object::InMemDicomObject;

use crate::dicom_export::{
    put_ds_exact, put_is, put_seq, put_str, put_strs, put_us, write_object, ObjectMeta,
};
use crate::extras::PlanarImage;
use crate::planar_structs::{PlanarStructure, PlanarStructureSet};
use crate::sop::{SOP_RTIMAGE, SOP_SECONDARY_CAPTURE};

/// What is drawn of one structure on one image.
#[derive(Clone, Debug, Default)]
pub struct Overlay {
    pub color: [u8; 3],
    /// Closed rings, in pixels (centre of the first pixel at `(0, 0)`).
    pub rings: Vec<Vec<[f64; 2]>>,
    /// Points of interest, in the same pixels.
    pub points: Vec<[f64; 2]>,
}

/// What `sets` have on the image `image_uid`: one [`Overlay`] per structure
/// with a contour or a point there that `keep` lets through, in the sets'
/// order. Empty when there is nothing to draw.
pub fn overlays<'a>(
    sets: impl IntoIterator<Item = &'a PlanarStructureSet>,
    image_uid: &str,
    keep: &dyn Fn(&PlanarStructureSet, &PlanarStructure) -> bool,
) -> Vec<Overlay> {
    let mut out = Vec::new();
    if image_uid.is_empty() {
        return out;
    }
    for set in sets {
        for s in &set.structures {
            if !keep(set, s) {
                continue;
            }
            let rings: Vec<Vec<[f64; 2]>> = s
                .contour(image_uid)
                .map(|c| {
                    c.region
                        .rings
                        .iter()
                        .filter(|r| r.pts.len() >= 2)
                        .map(|r| r.pts.clone())
                        .collect()
                })
                .unwrap_or_default();
            let points: Vec<[f64; 2]> = s
                .points
                .iter()
                .filter(|(u, _)| u == image_uid)
                .map(|(_, p)| *p)
                .collect();
            if rings.is_empty() && points.is_empty() {
                continue;
            }
            out.push(Overlay {
                color: s.color,
                rings,
                points,
            });
        }
    }
    out
}

/// The stroke a contour is burned in with, in pixels: about what a pane
/// shows of a 1.5-point line on an image fitted to it - a four-hundredth of
/// the image's longer side, never thinner than 1.5 nor thicker than 6.
pub fn line_width(img: &PlanarImage) -> f64 {
    (img.cols.max(img.rows) as f64 / 400.0).clamp(1.5, 6.0)
}

/// The image through `window` with `overlays` drawn over it: `rows · cols`
/// RGB triplets, row by row.
pub fn render(img: &PlanarImage, window: (f32, f32), overlays: &[Overlay]) -> Vec<u8> {
    let (w, h) = (img.cols, img.rows);
    let n = w * h;
    let mut rgb: Vec<u8> = match &img.rgb {
        Some(c) if c.len() >= n * 3 => c[..n * 3].to_vec(),
        _ => {
            let lo = window.0 - window.1 * 0.5;
            let scale = 255.0 / window.1.max(1e-12);
            let mut out = Vec::with_capacity(n * 3);
            for &v in img.data.iter().take(n) {
                let g = ((v - lo) * scale).clamp(0.0, 255.0) as u8;
                out.extend_from_slice(&[g, g, g]);
            }
            out.resize(n * 3, 0);
            out
        }
    };
    if n == 0 {
        return rgb;
    }
    let half = line_width(img) * 0.5;
    // How much of each pixel one structure's strokes cover, and which
    // pixels were touched, so the buffer is cleared in time with the work.
    let mut cover = vec![0f32; n];
    let mut touched: Vec<usize> = Vec::new();
    for ov in overlays {
        for ring in &ov.rings {
            for k in 0..ring.len() {
                let a = ring[k];
                let b = ring[(k + 1) % ring.len()];
                stamp(&mut cover, &mut touched, w, h, a, b, half);
            }
        }
        let r = half * 6.0 + 2.0;
        for p in &ov.points {
            for (dx, dy) in [(r, r), (r, -r)] {
                let a = [p[0] - dx, p[1] - dy];
                let b = [p[0] + dx, p[1] + dy];
                stamp(&mut cover, &mut touched, w, h, a, b, half);
            }
        }
        for &i in &touched {
            let c = cover[i];
            if c <= 0.0 {
                continue;
            }
            for (ch, col) in ov.color.iter().enumerate() {
                let v = &mut rgb[i * 3 + ch];
                *v = (*v as f32 * (1.0 - c) + *col as f32 * c).round() as u8;
            }
            cover[i] = 0.0;
        }
        touched.clear();
    }
    rgb
}

/// Cover the pixels within `half` of segment `a`-`b` (antialiased by one
/// pixel), keeping each pixel's largest coverage.
fn stamp(
    cover: &mut [f32],
    touched: &mut Vec<usize>,
    w: usize,
    h: usize,
    a: [f64; 2],
    b: [f64; 2],
    half: f64,
) {
    let reach = half + 1.0;
    let x0 = (a[0].min(b[0]) - reach).floor().max(0.0);
    let x1 = (a[0].max(b[0]) + reach).ceil().min(w as f64 - 1.0);
    let y0 = (a[1].min(b[1]) - reach).floor().max(0.0);
    let y1 = (a[1].max(b[1]) + reach).ceil().min(h as f64 - 1.0);
    if !(x0 <= x1 && y0 <= y1) {
        return;
    }
    let d = [b[0] - a[0], b[1] - a[1]];
    let len2 = d[0] * d[0] + d[1] * d[1];
    for y in y0 as usize..=y1 as usize {
        for x in x0 as usize..=x1 as usize {
            let p = [x as f64 - a[0], y as f64 - a[1]];
            let t = if len2 > 0.0 {
                ((p[0] * d[0] + p[1] * d[1]) / len2).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let q = [p[0] - t * d[0], p[1] - t * d[1]];
            let dist = (q[0] * q[0] + q[1] * q[1]).sqrt();
            let c = (half + 0.5 - dist).clamp(0.0, 1.0) as f32;
            if c <= 0.0 {
                continue;
            }
            let i = y * w + x;
            if cover[i] == 0.0 {
                touched.push(i);
            }
            if c > cover[i] {
                cover[i] = c;
            }
        }
    }
}

/// Write `rgb` (from [`render`] of `img`) at `path` as a Secondary Capture in
/// series `series_uid` (a fresh one when empty), naming `img` - its SOP
/// Instance UID as it goes out - as its source. `meta` gives the patient,
/// the study and the description. Returns the SOP Instance UID written.
pub fn write_secondary_capture(
    path: &Path,
    img: &PlanarImage,
    rgb: &[u8],
    meta: &ObjectMeta,
    series_uid: &str,
    series_number: i64,
) -> Result<String> {
    let n = img.rows * img.cols;
    if n == 0 || rgb.len() != n * 3 {
        bail!("the burned-in image has no pixels");
    }
    if img.rows > u16::MAX as usize || img.cols > u16::MAX as usize {
        bail!("the image is too large for DICOM rows and columns");
    }
    let mut o = InMemDicomObject::new_empty();
    let (sop_uid, _) = meta.write_as(&mut o, SOP_SECONDARY_CAPTURE, "OT", None, Some(series_uid));
    put_is(&mut o, tags::SERIES_NUMBER, series_number);
    put_strs(
        &mut o,
        tags::IMAGE_TYPE,
        VR::CS,
        &["DERIVED".into(), "SECONDARY".into()],
    );
    put_str(&mut o, tags::CONVERSION_TYPE, VR::CS, "WSD");
    put_str(&mut o, tags::BURNED_IN_ANNOTATION, VR::CS, "YES");
    put_str(&mut o, tags::PATIENT_ORIENTATION, VR::CS, "");
    put_str(&mut o, tags::MANUFACTURER, VR::LO, "Rust DICOM Station");
    if !img.sop_instance_uid.is_empty() {
        let class = if img.sop_class_uid.is_empty() {
            SOP_RTIMAGE
        } else {
            img.sop_class_uid.as_str()
        };
        let mut item = InMemDicomObject::new_empty();
        put_str(&mut item, tags::REFERENCED_SOP_CLASS_UID, VR::UI, class);
        put_str(
            &mut item,
            tags::REFERENCED_SOP_INSTANCE_UID,
            VR::UI,
            img.sop_instance_uid.as_str(),
        );
        put_seq(&mut o, tags::SOURCE_IMAGE_SEQUENCE, vec![item]);
    }
    put_str(
        &mut o,
        tags::DERIVATION_DESCRIPTION,
        VR::ST,
        format!(
            "2D contours burned in by Rust DICOM Station {}",
            env!("CARGO_PKG_VERSION")
        ),
    );
    // A spacing of exactly one is what a planar image says when it knows
    // none, and a picture claiming millimetres it does not have would
    // mislead a measurement more than saying nothing.
    if img.spacing != [1.0, 1.0] && img.spacing.iter().all(|s| *s > 0.0) {
        // Pixel Spacing is [between rows, between columns].
        put_ds_exact(
            &mut o,
            tags::PIXEL_SPACING,
            &[img.spacing[1], img.spacing[0]],
        );
    }
    put_us(&mut o, tags::ROWS, img.rows as u16);
    put_us(&mut o, tags::COLUMNS, img.cols as u16);
    put_us(&mut o, tags::SAMPLES_PER_PIXEL, 3);
    put_str(&mut o, tags::PHOTOMETRIC_INTERPRETATION, VR::CS, "RGB");
    put_us(&mut o, tags::PLANAR_CONFIGURATION, 0);
    put_us(&mut o, tags::BITS_ALLOCATED, 8);
    put_us(&mut o, tags::BITS_STORED, 8);
    put_us(&mut o, tags::HIGH_BIT, 7);
    put_us(&mut o, tags::PIXEL_REPRESENTATION, 0);
    let mut bytes = rgb.to_vec();
    // An element's value is an even number of bytes.
    if bytes.len() % 2 == 1 {
        bytes.push(0);
    }
    o.put(DataElement::new(
        tags::PIXEL_DATA,
        VR::OB,
        PrimitiveValue::U8(C::from_vec(bytes)),
    ));
    write_object(o, SOP_SECONDARY_CAPTURE, path)?;
    Ok(sop_uid)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(cols: usize, rows: usize) -> PlanarImage {
        PlanarImage {
            label: "img".into(),
            modality: "RTIMAGE".into(),
            rows,
            cols,
            spacing: [0.5, 0.5],
            data: vec![100.0; rows * cols],
            window: (100.0, 200.0),
            sop_instance_uid: "1.2.826.0.99.1".into(),
            ..Default::default()
        }
    }

    fn square(c: [f64; 2], r: f64) -> Vec<[f64; 2]> {
        vec![
            [c[0] - r, c[1] - r],
            [c[0] + r, c[1] - r],
            [c[0] + r, c[1] + r],
            [c[0] - r, c[1] + r],
        ]
    }

    #[test]
    fn the_outline_is_drawn_in_its_colour_and_nothing_else_changes() {
        let img = image(64, 48);
        let ov = Overlay {
            color: [255, 0, 0],
            rings: vec![square([32.0, 24.0], 10.0)],
            points: Vec::new(),
        };
        let rgb = render(&img, img.window, &[ov]);
        assert_eq!(rgb.len(), 64 * 48 * 3);
        let px = |x: usize, y: usize| {
            let i = (y * 64 + x) * 3;
            [rgb[i], rgb[i + 1], rgb[i + 2]]
        };
        // 100 through (100, 200) is the middle grey.
        assert_eq!(px(0, 0), [127, 127, 127]);
        assert_eq!(px(32, 24), [127, 127, 127], "the inside stays as it was");
        // On the top edge of the square, and on a vertex.
        assert_eq!(px(32, 14), [255, 0, 0]);
        assert_eq!(px(22, 14), [255, 0, 0]);
        // Antialiased: the next pixel out is blended, the one after left.
        let side = px(32, 13);
        assert!(side[0] > 127 && side[1] < 127 && side[1] > 0, "{side:?}");
        assert_eq!(px(32, 12), [127, 127, 127]);
    }

    #[test]
    fn only_what_is_on_the_image_and_let_through_is_drawn() {
        let mut set = PlanarStructureSet::new("S", "", "drawn");
        set.structure_entry("PTV", None)
            .contour_entry("1.2.3")
            .region = crate::contours::Region::from_ring(crate::contours::Poly::new(square(
            [10.0, 10.0],
            4.0,
        )));
        set.structure_entry("CORD", None)
            .points
            .push(("1.2.3".into(), [5.0, 5.0]));
        set.structure_entry("ELSEWHERE", None)
            .contour_entry("9.9.9")
            .region = crate::contours::Region::from_ring(crate::contours::Poly::new(square(
            [10.0, 10.0],
            4.0,
        )));
        let all = overlays([&set], "1.2.3", &|_, _| true);
        assert_eq!(all.len(), 2, "PTV's ring and CORD's point");
        let no_cord = overlays([&set], "1.2.3", &|_, s| s.name != "CORD");
        assert_eq!(no_cord.len(), 1);
        assert!(overlays([&set], "", &|_, _| true).is_empty());
    }

    #[test]
    fn a_secondary_capture_reads_back_as_a_colour_image() {
        let img = image(20, 10);
        let ov = Overlay {
            color: [0, 255, 0],
            rings: vec![square([10.0, 5.0], 3.0)],
            points: vec![[2.0, 2.0]],
        };
        let rgb = render(&img, img.window, &[ov]);
        let dir = std::env::temp_dir().join("rds_burn_in_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("sc.dcm");
        let meta = ObjectMeta {
            label: "CONTOURS".into(),
            description: "img with contours".into(),
            ..Default::default()
        };
        let uid = write_secondary_capture(&path, &img, &rgb, &meta, "", 900).unwrap();
        assert!(!uid.is_empty());
        let back = crate::extras::load_planar(&path).unwrap();
        assert_eq!((back.cols, back.rows), (20, 10));
        assert_eq!(back.sop_class_uid, SOP_SECONDARY_CAPTURE);
        let colours = back.rgb.expect("an RGB image");
        assert_eq!(colours, rgb, "every pixel as written");
        let obj = crate::dicomfile::open_full(&path).unwrap();
        assert_eq!(
            crate::loader::str_of(&obj, tags::BURNED_IN_ANNOTATION).as_deref(),
            Some("YES")
        );
        let _ = std::fs::remove_file(&path);
    }
}
