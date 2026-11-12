//! A planar structure set as a DICOM Grayscale Softcopy Presentation State
//! (PS3.3 A.33.1), and back.
//!
//! **What is written** ([`write`]): one GSPS (SOP class
//! `1.2.840.10008.5.1.4.1.1.11.1`, modality `PR`) per set, in the set's own
//! series:
//!
//! - *Presentation State Identification* (C.11.10): Content Label (the
//!   set's label as a CS), Content Description (the label in full),
//!   Presentation Creation Date / Time, Content Creator's Name.
//! - *Presentation State Relationship* (C.11.11): the Referenced Series
//!   Sequence, one item per series of the referenced images, each with its
//!   Referenced Image Sequence.
//! - *Graphic Layer* (C.10.7): one layer per structure - the name as a CS
//!   of up to 16 characters made unique ([`unique_layer_names`]), the full
//!   name in the Graphic Layer Description, the colour as the Recommended
//!   Display CIELab Value and, for a viewer that reads only that, the
//!   Recommended Display Grayscale Value (the colour's luminance).
//! - *Graphic Annotation* (C.10.5): one item per image and structure (an
//!   item names one layer), with a closed `POLYLINE` per ring (the first
//!   point repeated last), `PIXEL` units, not filled, and a `POINT` per
//!   point of interest.
//! - *Displayed Area* (C.10.4): the whole image, `SCALE TO FIT`, with the
//!   image's pixel spacing; *Softcopy VOI LUT* (C.11.8): the window each
//!   image was shown with; *Softcopy Presentation LUT*: `IDENTITY`.
//!
//! **Pixel coordinates.** PS3.3 C.10.5.1.2 puts the origin of `PIXEL` units
//! at the top left corner of the top left pixel (`0\0`), its bottom right
//! corner at `1\1` and the image's bottom right corner at `Columns\Rows`,
//! column first. The planar structures put the *centre* of the first pixel
//! at `(0, 0)`, so the writer adds one half and the reader subtracts it.
//!
//! **What is read** ([`read`]): what RDS writes and what others write.
//! `POLYLINE` and `INTERPOLATED` (taken as a polyline through its points)
//! become rings, `CIRCLE` (centre, a point on it) and `ELLIPSE` (the ends of
//! the major, then of the minor axis) are resampled to 64 points, `POINT`s
//! are points of interest, and a text object's anchor is a point of its
//! layer. `DISPLAY` units are mapped through the image's Displayed Area
//! Selection; without one the object is skipped with a warning. A graphic
//! in a layer with no Graphic Layer item gets a structure of its own. A
//! layer's colour is its CIELab value, else its grayscale value, else the
//! next of the palette. The referenced images are kept by SOP Instance UID
//! whether or not they are loaded: a set attaches to its images when they
//! arrive.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use dicom_core::value::{PrimitiveValue, C};
use dicom_core::{DataElement, VR};
use dicom_dictionary_std::tags;
use dicom_object::InMemDicomObject;

use crate::contours::{Poly, Region};
use crate::dicom_export::{
    put_ds_exact, put_is, put_seq, put_str, put_us, sanitize_cs, today, write_object, ObjectMeta,
};
use crate::extras::PlanarImage;
use crate::loader::{items_of, str_of, LoadedStudy};
use crate::planar_structs::{unique_layer_names, ContourSource, PlanarStructureSet, PALETTE};
use crate::sop::{SOP_GSPS, SOP_RTIMAGE};

/// What a reader sees of an image: the SOP Class UID it is referenced by.
fn class_of(img: &PlanarImage) -> &str {
    if img.sop_class_uid.is_empty() {
        SOP_RTIMAGE
    } else {
        &img.sop_class_uid
    }
}

fn image_ref(img: &PlanarImage) -> InMemDicomObject {
    let mut it = InMemDicomObject::new_empty();
    put_str(
        &mut it,
        tags::REFERENCED_SOP_CLASS_UID,
        VR::UI,
        class_of(img),
    );
    put_str(
        &mut it,
        tags::REFERENCED_SOP_INSTANCE_UID,
        VR::UI,
        img.sop_instance_uid.as_str(),
    );
    it
}

fn put_fl(o: &mut InMemDicomObject, tag: dicom_core::Tag, v: Vec<f32>) {
    o.put(DataElement::new(
        tag,
        VR::FL,
        PrimitiveValue::F32(C::from_vec(v)),
    ));
}

fn put_sl(o: &mut InMemDicomObject, tag: dicom_core::Tag, v: Vec<i32>) {
    o.put(DataElement::new(
        tag,
        VR::SL,
        PrimitiveValue::I32(C::from_vec(v)),
    ));
}

fn put_us3(o: &mut InMemDicomObject, tag: dicom_core::Tag, v: [u16; 3]) {
    o.put(DataElement::new(
        tag,
        VR::US,
        PrimitiveValue::U16(C::from_vec(v.to_vec())),
    ));
}

/// The grayscale value a viewer that reads no colour draws a layer in: the
/// colour's luminance on `0..=65535`.
fn grayscale_of(c: [u8; 3]) -> u16 {
    let y = 0.299 * c[0] as f64 + 0.587 * c[1] as f64 + 0.114 * c[2] as f64;
    (y / 255.0 * 65535.0).round().clamp(0.0, 65535.0) as u16
}

/// Write `set` as a GSPS at `path`. `images` are the referenced images with
/// the window each is shown with; every image a contour or point of the set
/// is on must be among them. `meta` gives the patient and study. Returns the
/// SOP Instance UID written.
pub fn write(
    path: &Path,
    set: &PlanarStructureSet,
    images: &[(&PlanarImage, (f32, f32))],
    meta: &ObjectMeta,
) -> Result<String> {
    write_as(path, set, images, meta, "")
}

/// [`write`] under a SOP Instance UID given (an export keeping or choosing
/// the state's identity); a fresh one when `sop_uid` is empty.
pub fn write_as(
    path: &Path,
    set: &PlanarStructureSet,
    images: &[(&PlanarImage, (f32, f32))],
    meta: &ObjectMeta,
    sop_uid: &str,
) -> Result<String> {
    // Every image a graphic is on, in the set's order.
    let mut used: Vec<&str> = Vec::new();
    for s in &set.structures {
        for c in s.contours.iter().filter(|c| !c.region.is_empty()) {
            if !used.contains(&c.image_uid.as_str()) {
                used.push(&c.image_uid);
            }
        }
        for (u, _) in &s.points {
            if !used.contains(&u.as_str()) {
                used.push(u);
            }
        }
    }
    if used.is_empty() {
        bail!("the set has no contour to write");
    }
    let find = |uid: &str| images.iter().find(|(i, _)| i.sop_instance_uid == uid);
    if let Some(missing) = used.iter().find(|u| find(u).is_none()) {
        bail!("the set has contours on an image that is not loaded ({missing})");
    }
    let refs: Vec<&(&PlanarImage, (f32, f32))> = used.iter().filter_map(|u| find(u)).collect();

    let mut o = InMemDicomObject::new_empty();
    let meta = ObjectMeta {
        label: set.label.clone(),
        description: set.label.clone(),
        study_uid: if set.study_uid.is_empty() {
            meta.study_uid.clone()
        } else {
            set.study_uid.clone()
        },
        ..meta.clone()
    };
    let (sop_uid, _) = meta.write_as(
        &mut o,
        SOP_GSPS,
        "PR",
        Some(sop_uid),
        Some(&set.series_instance_uid),
    );
    put_str(&mut o, tags::CONTENT_LABEL, VR::CS, sanitize_cs(&set.label));
    let (date, time) = today();
    put_str(&mut o, tags::PRESENTATION_CREATION_DATE, VR::DA, date);
    put_str(&mut o, tags::PRESENTATION_CREATION_TIME, VR::TM, time);
    put_str(&mut o, tags::CONTENT_CREATOR_NAME, VR::PN, "RDS");
    put_str(&mut o, tags::MANUFACTURER, VR::LO, "Rust DICOM Station");

    // Presentation State Relationship: one item per series.
    let mut series: Vec<(String, Vec<InMemDicomObject>)> = Vec::new();
    for (img, _) in &refs {
        let s_uid = img.series_instance_uid.clone();
        match series.iter_mut().find(|(u, _)| *u == s_uid) {
            Some((_, list)) => list.push(image_ref(img)),
            None => series.push((s_uid, vec![image_ref(img)])),
        }
    }
    let series_items = series
        .into_iter()
        .map(|(uid, list)| {
            let mut it = InMemDicomObject::new_empty();
            put_str(&mut it, tags::SERIES_INSTANCE_UID, VR::UI, uid);
            put_seq(&mut it, tags::REFERENCED_IMAGE_SEQUENCE, list);
            it
        })
        .collect();
    put_seq(&mut o, tags::REFERENCED_SERIES_SEQUENCE, series_items);

    // Graphic Layer: one per structure.
    let layers = unique_layer_names(set);
    let layer_items = set
        .structures
        .iter()
        .zip(&layers)
        .enumerate()
        .map(|(k, (s, name))| {
            let mut it = InMemDicomObject::new_empty();
            put_str(&mut it, tags::GRAPHIC_LAYER, VR::CS, name.as_str());
            put_is(&mut it, tags::GRAPHIC_LAYER_ORDER, k as i64 + 1);
            put_us(
                &mut it,
                tags::GRAPHIC_LAYER_RECOMMENDED_DISPLAY_GRAYSCALE_VALUE,
                grayscale_of(s.color),
            );
            put_us3(
                &mut it,
                tags::GRAPHIC_LAYER_RECOMMENDED_DISPLAY_CIE_LAB_VALUE,
                crate::dicomseg::rgb_to_cielab(s.color),
            );
            put_str(
                &mut it,
                tags::GRAPHIC_LAYER_DESCRIPTION,
                VR::LO,
                s.name.as_str(),
            );
            it
        })
        .collect();
    put_seq(&mut o, tags::GRAPHIC_LAYER_SEQUENCE, layer_items);

    // Graphic Annotation: one item per image and structure.
    let mut annotations = Vec::new();
    for (img, _) in &refs {
        let uid = img.sop_instance_uid.as_str();
        for (s, layer) in set.structures.iter().zip(&layers) {
            let mut objects = Vec::new();
            if let Some(c) = s.contour(uid) {
                for ring in &c.region.rings {
                    if ring.pts.len() < 3 {
                        continue;
                    }
                    let mut data: Vec<f32> = Vec::with_capacity(2 * ring.pts.len() + 2);
                    for p in ring.pts.iter().chain(ring.pts.first()) {
                        data.push((p[0] + 0.5) as f32);
                        data.push((p[1] + 0.5) as f32);
                    }
                    objects.push(graphic(&data, "POLYLINE"));
                }
            }
            for (_, p) in s.points.iter().filter(|(u, _)| u == uid) {
                objects.push(graphic(
                    &[(p[0] + 0.5) as f32, (p[1] + 0.5) as f32],
                    "POINT",
                ));
            }
            if objects.is_empty() {
                continue;
            }
            let mut it = InMemDicomObject::new_empty();
            put_seq(
                &mut it,
                tags::REFERENCED_IMAGE_SEQUENCE,
                vec![image_ref(img)],
            );
            put_str(&mut it, tags::GRAPHIC_LAYER, VR::CS, layer.as_str());
            put_seq(&mut it, tags::GRAPHIC_OBJECT_SEQUENCE, objects);
            annotations.push(it);
        }
    }
    put_seq(&mut o, tags::GRAPHIC_ANNOTATION_SEQUENCE, annotations);

    // Displayed Area and Softcopy VOI LUT, per image.
    let areas = refs
        .iter()
        .map(|(img, _)| {
            let mut it = InMemDicomObject::new_empty();
            put_seq(
                &mut it,
                tags::REFERENCED_IMAGE_SEQUENCE,
                vec![image_ref(img)],
            );
            put_sl(
                &mut it,
                tags::DISPLAYED_AREA_TOP_LEFT_HAND_CORNER,
                vec![1, 1],
            );
            put_sl(
                &mut it,
                tags::DISPLAYED_AREA_BOTTOM_RIGHT_HAND_CORNER,
                vec![img.cols as i32, img.rows as i32],
            );
            put_str(
                &mut it,
                tags::PRESENTATION_SIZE_MODE,
                VR::CS,
                "SCALE TO FIT",
            );
            // Row spacing first, as Pixel Spacing has it.
            put_ds_exact(
                &mut it,
                tags::PRESENTATION_PIXEL_SPACING,
                &[img.spacing[1], img.spacing[0]],
            );
            it
        })
        .collect();
    put_seq(&mut o, tags::DISPLAYED_AREA_SELECTION_SEQUENCE, areas);
    let vois = refs
        .iter()
        .map(|(img, wl)| {
            let mut it = InMemDicomObject::new_empty();
            put_seq(
                &mut it,
                tags::REFERENCED_IMAGE_SEQUENCE,
                vec![image_ref(img)],
            );
            put_ds_exact(&mut it, tags::WINDOW_CENTER, &[wl.0 as f64]);
            put_ds_exact(&mut it, tags::WINDOW_WIDTH, &[(wl.1 as f64).max(1e-6)]);
            it
        })
        .collect();
    put_seq(&mut o, tags::SOFTCOPY_VOILUT_SEQUENCE, vois);
    put_str(&mut o, tags::PRESENTATION_LUT_SHAPE, VR::CS, "IDENTITY");
    write_object(o, SOP_GSPS, path)?;
    Ok(sop_uid)
}

/// One graphic object in `PIXEL` units.
fn graphic(data: &[f32], kind: &str) -> InMemDicomObject {
    let mut g = InMemDicomObject::new_empty();
    put_str(&mut g, tags::GRAPHIC_ANNOTATION_UNITS, VR::CS, "PIXEL");
    put_us(&mut g, tags::GRAPHIC_DIMENSIONS, 2);
    put_us(
        &mut g,
        tags::NUMBER_OF_GRAPHIC_POINTS,
        (data.len() / 2) as u16,
    );
    put_fl(&mut g, tags::GRAPHIC_DATA, data.to_vec());
    put_str(&mut g, tags::GRAPHIC_TYPE, VR::CS, kind);
    put_str(&mut g, tags::GRAPHIC_FILLED, VR::CS, "N");
    g
}

/// Read a presentation state file ([`read`]).
pub fn load(path: &Path) -> Result<(PlanarStructureSet, Vec<String>)> {
    let obj = crate::dicomfile::open_full(path)
        .with_context(|| format!("open presentation state {}", path.display()))?;
    let (mut set, notes) = read(&obj)?;
    set.file_name = path
        .file_name()
        .map(|f| f.to_string_lossy().into_owned())
        .unwrap_or_default();
    set.source_path = Some(path.to_path_buf());
    Ok((set, notes))
}

fn floats(obj: &InMemDicomObject, tag: dicom_core::Tag) -> Option<Vec<f64>> {
    obj.element(tag).ok()?.to_multi_float64().ok()
}

fn ints(obj: &InMemDicomObject, tag: dicom_core::Tag) -> Option<Vec<i64>> {
    obj.element(tag).ok()?.to_multi_int::<i64>().ok()
}

/// The images an item references: its Referenced Image Sequence's SOP
/// Instance UIDs (empty: every image of the state).
fn referenced(it: &InMemDicomObject) -> Vec<String> {
    items_of(it, tags::REFERENCED_IMAGE_SEQUENCE)
        .into_iter()
        .flatten()
        .filter_map(|r| str_of(r, tags::REFERENCED_SOP_INSTANCE_UID))
        .collect()
}

/// A layer as read: its description, colour and order.
struct Layer {
    name: String,
    color: Option<[u8; 3]>,
}

/// A planar structure set out of a presentation state object, with a note
/// per graphic that could not be read.
pub fn read(obj: &InMemDicomObject) -> Result<(PlanarStructureSet, Vec<String>)> {
    let sop_class = str_of(obj, tags::SOP_CLASS_UID).unwrap_or_default();
    if !sop_class.starts_with("1.2.840.10008.5.1.4.1.1.11.") {
        bail!("not a presentation state ({sop_class})");
    }
    let mut notes = Vec::new();
    let label = str_of(obj, tags::CONTENT_DESCRIPTION)
        .filter(|s| !s.trim().is_empty())
        .or_else(|| str_of(obj, tags::CONTENT_LABEL))
        .unwrap_or_else(|| "Presentation state".into());
    let mut set = PlanarStructureSet {
        label,
        sop_instance_uid: str_of(obj, tags::SOP_INSTANCE_UID).unwrap_or_default(),
        series_instance_uid: str_of(obj, tags::SERIES_INSTANCE_UID).unwrap_or_default(),
        study_uid: str_of(obj, tags::STUDY_INSTANCE_UID).unwrap_or_default(),
        ..Default::default()
    };
    // The referenced images, loaded or not.
    for s in items_of(obj, tags::REFERENCED_SERIES_SEQUENCE)
        .into_iter()
        .flatten()
    {
        for u in referenced(s) {
            if !set.images.contains(&u) {
                set.images.push(u);
            }
        }
    }
    // The layers.
    let mut layers: Vec<(String, Layer)> = Vec::new();
    let mut ordered: Vec<(i64, String)> = Vec::new();
    for it in items_of(obj, tags::GRAPHIC_LAYER_SEQUENCE)
        .into_iter()
        .flatten()
    {
        let Some(id) = str_of(it, tags::GRAPHIC_LAYER) else {
            continue;
        };
        let name = str_of(it, tags::GRAPHIC_LAYER_DESCRIPTION)
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| id.clone());
        let lab = ints(it, tags::GRAPHIC_LAYER_RECOMMENDED_DISPLAY_CIE_LAB_VALUE)
            .filter(|v| v.len() >= 3)
            .map(|v| crate::dicomseg::cielab_to_rgb([v[0] as u16, v[1] as u16, v[2] as u16]));
        let grey = ints(it, tags::GRAPHIC_LAYER_RECOMMENDED_DISPLAY_GRAYSCALE_VALUE)
            .and_then(|v| v.first().copied())
            .map(|g| {
                let v = (g as f64 / 65535.0 * 255.0).round().clamp(0.0, 255.0) as u8;
                [v, v, v]
            });
        let order = ints(it, tags::GRAPHIC_LAYER_ORDER)
            .and_then(|v| v.first().copied())
            .unwrap_or(i64::MAX);
        ordered.push((order, id.clone()));
        layers.push((
            id,
            Layer {
                name,
                color: lab.or(grey),
            },
        ));
    }
    // Structures in layer order, so a set written and read keeps its order.
    ordered.sort_by_key(|(o, _)| *o);
    for (_, id) in &ordered {
        if let Some((_, l)) = layers.iter().find(|(i, _)| i == id) {
            set.structure_entry(&l.name, l.color);
        }
    }
    // Displayed areas, per image (`None` key: every image).
    let mut areas: Vec<(Vec<String>, [f64; 4])> = Vec::new();
    for it in items_of(obj, tags::DISPLAYED_AREA_SELECTION_SEQUENCE)
        .into_iter()
        .flatten()
    {
        let (Some(tl), Some(br)) = (
            floats(it, tags::DISPLAYED_AREA_TOP_LEFT_HAND_CORNER),
            floats(it, tags::DISPLAYED_AREA_BOTTOM_RIGHT_HAND_CORNER),
        ) else {
            continue;
        };
        if tl.len() >= 2 && br.len() >= 2 {
            areas.push((referenced(it), [tl[0], tl[1], br[0], br[1]]));
        }
    }
    let area_of = |uid: &str| -> Option<[f64; 4]> {
        areas
            .iter()
            .find(|(r, _)| r.iter().any(|u| u == uid))
            .or_else(|| areas.iter().find(|(r, _)| r.is_empty()))
            .map(|(_, a)| *a)
    };
    let mut skipped_display = 0usize;
    let mut skipped_kind: Vec<String> = Vec::new();
    for it in items_of(obj, tags::GRAPHIC_ANNOTATION_SEQUENCE)
        .into_iter()
        .flatten()
    {
        let mut on = referenced(it);
        if on.is_empty() {
            on = set.images.clone();
        }
        let layer_id = str_of(it, tags::GRAPHIC_LAYER).unwrap_or_default();
        let (name, color) = match layers.iter().find(|(i, _)| *i == layer_id) {
            Some((_, l)) => (l.name.clone(), l.color),
            // A layer the state does not describe: a structure of its own.
            None => (
                if layer_id.trim().is_empty() {
                    "Annotation".to_string()
                } else {
                    layer_id.clone()
                },
                None,
            ),
        };
        let color = color.or(Some(PALETTE[set.structures.len() % PALETTE.len()]));
        for uid in &on {
            if !set.images.contains(uid) {
                set.images.push(uid.clone());
            }
            // From the state's units to the structures' pixels.
            let to_px = |units: &str, x: f64, y: f64| -> Option<[f64; 2]> {
                match units {
                    "DISPLAY" => {
                        let a = area_of(uid)?;
                        // Corners in pixel-corner units: the left edge of
                        // pixel TLHC (one-based) to the right edge of BRHC.
                        let (x0, y0) = (a[0] - 1.0, a[1] - 1.0);
                        let (x1, y1) = (a[2], a[3]);
                        Some([x0 + x * (x1 - x0) - 0.5, y0 + y * (y1 - y0) - 0.5])
                    }
                    _ => Some([x - 0.5, y - 0.5]),
                }
            };
            let mut region = Region::new();
            let mut points: Vec<[f64; 2]> = Vec::new();
            for g in items_of(it, tags::GRAPHIC_OBJECT_SEQUENCE)
                .into_iter()
                .flatten()
            {
                let units =
                    str_of(g, tags::GRAPHIC_ANNOTATION_UNITS).unwrap_or_else(|| "PIXEL".into());
                let kind = str_of(g, tags::GRAPHIC_TYPE).unwrap_or_default();
                let data = floats(g, tags::GRAPHIC_DATA).unwrap_or_default();
                let raw: Vec<(f64, f64)> = data
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| (c[0], c[1]))
                    .collect();
                let pts: Option<Vec<[f64; 2]>> =
                    raw.iter().map(|&(x, y)| to_px(&units, x, y)).collect();
                let Some(pts) = pts else {
                    skipped_display += 1;
                    continue;
                };
                match kind.as_str() {
                    "POLYLINE" | "INTERPOLATED" => {
                        let mut p = pts;
                        // A closed polyline repeats its first point last.
                        if p.len() >= 2 {
                            let (a, b) = (p[0], p[p.len() - 1]);
                            if (a[0] - b[0]).abs() < 1e-6 && (a[1] - b[1]).abs() < 1e-6 {
                                p.pop();
                            }
                        }
                        if p.len() >= 3 {
                            region.rings.push(Poly::new(p));
                        }
                    }
                    "CIRCLE" if pts.len() >= 2 => {
                        let (c, e) = (pts[0], pts[1]);
                        let r = ((e[0] - c[0]).powi(2) + (e[1] - c[1]).powi(2)).sqrt();
                        region.rings.push(Poly::circle(c, r, 64));
                    }
                    "ELLIPSE" if pts.len() >= 4 => {
                        region.rings.push(ellipse(pts[0], pts[1], pts[2], pts[3]));
                    }
                    "POINT" if !pts.is_empty() => points.push(pts[0]),
                    other => {
                        if !skipped_kind.iter().any(|k| k == other) {
                            skipped_kind.push(other.to_string());
                        }
                    }
                }
            }
            // A text object's anchor is a point of its layer.
            for t in items_of(it, tags::TEXT_OBJECT_SEQUENCE)
                .into_iter()
                .flatten()
            {
                let Some(a) = floats(t, tags::ANCHOR_POINT).filter(|a| a.len() >= 2) else {
                    continue;
                };
                let units = str_of(t, tags::ANCHOR_POINT_ANNOTATION_UNITS)
                    .unwrap_or_else(|| "PIXEL".into());
                match to_px(&units, a[0], a[1]) {
                    Some(p) => points.push(p),
                    None => skipped_display += 1,
                }
            }
            if region.rings.is_empty() && points.is_empty() {
                continue;
            }
            region.normalize();
            let s = set.structure_entry(&name, color);
            if !region.rings.is_empty() {
                let c = s.contour_entry(uid);
                c.region.rings.extend(region.rings);
                c.region.normalize();
                c.source = ContourSource::File;
            }
            for p in points {
                s.points.push((uid.clone(), p));
            }
        }
    }
    if skipped_display > 0 {
        notes.push(format!(
            "{skipped_display} graphic(s) in DISPLAY units without a displayed area were skipped"
        ));
    }
    if !skipped_kind.is_empty() {
        notes.push(format!(
            "graphic type(s) not read: {}",
            skipped_kind.join(", ")
        ));
    }
    // Structures that came with no graphic at all are kept (a layer the
    // state named): they are what it said.
    Ok((set, notes))
}

/// An ellipse through the ends of its major axis `a0`-`a1` and of its minor
/// axis `b0`-`b1`, as 64 points.
fn ellipse(a0: [f64; 2], a1: [f64; 2], b0: [f64; 2], b1: [f64; 2]) -> Poly {
    let c = [(a0[0] + a1[0]) * 0.5, (a0[1] + a1[1]) * 0.5];
    let u = [(a1[0] - a0[0]) * 0.5, (a1[1] - a0[1]) * 0.5];
    let v = [(b1[0] - b0[0]) * 0.5, (b1[1] - b0[1]) * 0.5];
    let pts = (0..64)
        .map(|k| {
            let t = k as f64 / 64.0 * std::f64::consts::TAU;
            let (s, co) = t.sin_cos();
            [c[0] + u[0] * co + v[0] * s, c[1] + u[1] * co + v[1] * s]
        })
        .collect();
    let mut p = Poly::new(pts);
    p.orient(true);
    p
}

/// The images made here (no file of their own) written into one folder so
/// far, for [`write_made_images`] and [`write_state`]: each goes once, under
/// one series for all that had none.
#[derive(Clone, Debug)]
pub struct MadeImages {
    /// The SOP Instance UIDs written.
    pub written: HashSet<String>,
    /// The series an image made here without one is written in, so a
    /// state's series item names the series it went out in.
    pub series: String,
}

impl Default for MadeImages {
    fn default() -> Self {
        MadeImages {
            written: HashSet::new(),
            series: crate::dicom_export::new_uid(),
        }
    }
}

impl MadeImages {
    /// Image `img` as it goes out of this folder: an image made here as an
    /// RT Image, in [`MadeImages::series`] when it has no series of its own.
    fn as_written(&self, img: &PlanarImage) -> PlanarImage {
        let mut img = img.clone();
        if img.source_path.is_none() {
            if img.series_instance_uid.is_empty() {
                img.series_instance_uid = self.series.clone();
            }
            img.sop_class_uid = SOP_RTIMAGE.to_string();
        }
        img
    }
}

/// The file name a made-here image is written under: its SOP Instance UID
/// hashed (FNV-1a), so the same image saved twice into one folder - by two
/// sets, or one set saved twice - lands in the same file rather than two
/// files under one UID, and two images never take one name. File names
/// carry no label: a label may be anything, a patient's name included.
fn made_image_name(sop_uid: &str) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in sop_uid.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    format!("rtimage_{h:016x}.dcm")
}

/// Write into `dir` each image of `uids` (SOP Instance UIDs) that was made
/// here and has no file of its own, as an RT Image under the UIDs it already
/// has - skipping those `made` says were written into this folder already.
/// Returns the files written.
pub fn write_made_images(
    study: &LoadedStudy,
    uids: &[String],
    dir: &Path,
    made: &mut MadeImages,
) -> Result<Vec<PathBuf>> {
    std::fs::create_dir_all(dir).map_err(|e| anyhow!("create {}: {e}", dir.display()))?;
    let mut files = Vec::new();
    for uid in uids {
        let Some(i) = study.planar_by_uid(uid) else {
            continue;
        };
        let src = &study.planar_images[i];
        if src.source_path.is_some() || made.written.contains(uid) {
            continue;
        }
        let img = made.as_written(src);
        let path = dir.join(made_image_name(uid));
        crate::dicom_export::write_rt_image(
            &path,
            &img,
            &ObjectMeta::of_study(study, "IMAGE", &img.label),
        )?;
        made.written.insert(uid.clone());
        files.push(path);
    }
    Ok(files)
}

/// Write planar structure set `k` of `study` into `dir` as a presentation
/// state referencing `images` (SOP Instance UIDs) and nothing else: the
/// contours and points on other images are left out, and so is a structure
/// left with none. `windows` gives the window of an image by index, where a
/// pane shows it with one; otherwise its own. `made` is what
/// [`write_made_images`] wrote into the folder, for the series a made-here
/// image went out in. Returns the file and the state's SOP Instance UID.
pub fn write_state(
    study: &LoadedStudy,
    k: usize,
    dir: &Path,
    images: &[String],
    windows: &dyn Fn(usize) -> Option<(f32, f32)>,
    made: &MadeImages,
) -> Result<(PathBuf, String)> {
    let set = study
        .planar_structs
        .get(k)
        .ok_or_else(|| anyhow!("no planar structure set {}", k + 1))?;
    std::fs::create_dir_all(dir).map_err(|e| anyhow!("create {}: {e}", dir.display()))?;
    // The set narrowed to `images`. A structure whose graphics are all on
    // other images goes; one that has none anywhere (a layer the set was
    // read with, a structure not drawn yet) stays the empty layer it was.
    let mut narrow = set.clone();
    let drawn = |s: &crate::planar_structs::PlanarStructure| {
        s.contours.iter().any(|c| !c.region.is_empty()) || !s.points.is_empty()
    };
    narrow.structures.retain_mut(|s| {
        let had = drawn(s);
        s.contours.retain(|c| images.contains(&c.image_uid));
        s.points.retain(|(u, _)| images.contains(u));
        drawn(s) || !had
    });
    narrow.images.retain(|u| images.contains(u));
    let mut owned: Vec<(PlanarImage, (f32, f32))> = Vec::new();
    for uid in images {
        let Some(i) = study.planar_by_uid(uid) else {
            continue;
        };
        let img = made.as_written(&study.planar_images[i]);
        let wl = windows(i).unwrap_or(img.window);
        owned.push((img, wl));
    }
    let refs: Vec<(&PlanarImage, (f32, f32))> = owned.iter().map(|(i, w)| (i, *w)).collect();
    let meta = ObjectMeta::of_study(study, &set.label, &set.label);
    let path = dir.join(format!("gsps_{}.dcm", k + 1));
    let uid = write(&path, &narrow, &refs, &meta)?;
    Ok((path, uid))
}

/// Save planar structure set `k` of `study` into `dir`: every referenced
/// image made here (no file of its own) as an RT Image under the UIDs it
/// already has ([`write_made_images`]), then the presentation state
/// ([`write_state`]), so the folder is self-contained. `windows` gives the
/// window of an image by index, where a pane shows it with one; otherwise
/// its own. Returns the files written and the SOP Instance UID of the state.
pub fn save_set(
    study: &LoadedStudy,
    k: usize,
    dir: &Path,
    windows: &dyn Fn(usize) -> Option<(f32, f32)>,
) -> Result<(Vec<PathBuf>, String)> {
    let set = study
        .planar_structs
        .get(k)
        .ok_or_else(|| anyhow!("no planar structure set {}", k + 1))?;
    let mut made = MadeImages::default();
    let mut files = write_made_images(study, &set.images, dir, &mut made)?;
    let (path, uid) = write_state(study, k, dir, &set.images, windows, &made)?;
    files.push(path);
    Ok((files, uid))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obj_ref(uid: &str) -> InMemDicomObject {
        let mut it = InMemDicomObject::new_empty();
        put_str(&mut it, tags::REFERENCED_SOP_CLASS_UID, VR::UI, SOP_RTIMAGE);
        put_str(&mut it, tags::REFERENCED_SOP_INSTANCE_UID, VR::UI, uid);
        it
    }

    fn object(kind: &str, units: &str, data: &[f32]) -> InMemDicomObject {
        let mut g = graphic(data, kind);
        put_str(&mut g, tags::GRAPHIC_ANNOTATION_UNITS, VR::CS, units);
        g
    }

    /// A state as another system might write it: no Graphic Layer item,
    /// a circle, an interpolated curve, an ellipse, a graphic in DISPLAY
    /// units over a displayed area, and a text object.
    #[test]
    fn a_state_from_elsewhere_reads_with_the_documented_substitutions() {
        let mut o = InMemDicomObject::new_empty();
        put_str(&mut o, tags::SOP_CLASS_UID, VR::UI, SOP_GSPS);
        put_str(&mut o, tags::SOP_INSTANCE_UID, VR::UI, "1.2.3.9");
        put_str(&mut o, tags::CONTENT_LABEL, VR::CS, "MARKUP");
        let mut series = InMemDicomObject::new_empty();
        put_str(&mut series, tags::SERIES_INSTANCE_UID, VR::UI, "1.2.3.1");
        put_seq(
            &mut series,
            tags::REFERENCED_IMAGE_SEQUENCE,
            vec![obj_ref("1.2.3.1.1")],
        );
        put_seq(&mut o, tags::REFERENCED_SERIES_SEQUENCE, vec![series]);
        // The displayed area: pixels 11..=110 across, 21..=220 down.
        let mut area = InMemDicomObject::new_empty();
        put_sl(
            &mut area,
            tags::DISPLAYED_AREA_TOP_LEFT_HAND_CORNER,
            vec![11, 21],
        );
        put_sl(
            &mut area,
            tags::DISPLAYED_AREA_BOTTOM_RIGHT_HAND_CORNER,
            vec![110, 220],
        );
        put_seq(&mut o, tags::DISPLAYED_AREA_SELECTION_SEQUENCE, vec![area]);
        let mut ann = InMemDicomObject::new_empty();
        put_str(&mut ann, tags::GRAPHIC_LAYER, VR::CS, "ROI");
        put_seq(
            &mut ann,
            tags::GRAPHIC_OBJECT_SEQUENCE,
            vec![
                object("CIRCLE", "PIXEL", &[50.5, 60.5, 60.5, 60.5]),
                object(
                    "INTERPOLATED",
                    "PIXEL",
                    &[150.5, 150.5, 170.5, 150.5, 170.5, 170.5, 150.5, 170.5],
                ),
                object(
                    "ELLIPSE",
                    "PIXEL",
                    &[200.5, 50.5, 240.5, 50.5, 220.5, 40.5, 220.5, 60.5],
                ),
                // The middle of the displayed area.
                object("POINT", "DISPLAY", &[0.5, 0.5]),
            ],
        );
        let mut text = InMemDicomObject::new_empty();
        put_str(&mut text, tags::UNFORMATTED_TEXT_VALUE, VR::ST, "note");
        put_str(
            &mut text,
            tags::ANCHOR_POINT_ANNOTATION_UNITS,
            VR::CS,
            "PIXEL",
        );
        put_fl(&mut text, tags::ANCHOR_POINT, vec![5.5, 6.5]);
        put_seq(&mut ann, tags::TEXT_OBJECT_SEQUENCE, vec![text]);
        put_seq(&mut o, tags::GRAPHIC_ANNOTATION_SEQUENCE, vec![ann]);

        let (set, notes) = read(&o).unwrap();
        assert!(notes.is_empty(), "{notes:?}");
        assert_eq!(set.images, ["1.2.3.1.1"]);
        assert_eq!(
            set.structures.len(),
            1,
            "the layer with no item is a structure"
        );
        let s = &set.structures[0];
        assert_eq!(s.name, "ROI");
        let c = s.contour("1.2.3.1.1").unwrap();
        assert_eq!(c.region.rings.len(), 3);
        // The circle: centre (50, 60) in the structures' pixels, radius 10.
        let circle = c
            .region
            .rings
            .iter()
            .find(|r| r.pts.len() == 64 && (r.bbox()[2] - r.bbox()[0] - 20.0).abs() < 0.2)
            .expect("the circle");
        let cc = circle.centroid();
        assert!((cc[0] - 50.0).abs() < 1e-6 && (cc[1] - 60.0).abs() < 1e-6);
        // The ellipse: 40 across, 20 down, centred on (220, 50).
        let e = c
            .region
            .rings
            .iter()
            .map(|r| r.bbox())
            .find(|b| (b[2] - b[0] - 40.0).abs() < 1e-6)
            .expect("the ellipse");
        assert!((e[3] - e[1] - 20.0).abs() < 1e-6);
        // The DISPLAY point: the middle of pixels 11..=110 is between the
        // centres of 59 and 60 (zero-based), the text anchor is pixel (5, 6).
        assert_eq!(s.points.len(), 2);
        let p = s.points[0].1;
        assert!(
            (p[0] - 59.5).abs() < 1e-9 && (p[1] - 119.5).abs() < 1e-9,
            "{p:?}"
        );
        assert_eq!(s.points[1].1, [5.0, 6.0]);
    }

    #[test]
    fn display_units_without_a_displayed_area_are_skipped_and_said() {
        let mut o = InMemDicomObject::new_empty();
        put_str(&mut o, tags::SOP_CLASS_UID, VR::UI, SOP_GSPS);
        let mut ann = InMemDicomObject::new_empty();
        put_seq(
            &mut ann,
            tags::REFERENCED_IMAGE_SEQUENCE,
            vec![obj_ref("9.9")],
        );
        put_str(&mut ann, tags::GRAPHIC_LAYER, VR::CS, "A");
        put_seq(
            &mut ann,
            tags::GRAPHIC_OBJECT_SEQUENCE,
            vec![object(
                "POLYLINE",
                "DISPLAY",
                &[0.1, 0.1, 0.2, 0.1, 0.2, 0.2, 0.1, 0.1],
            )],
        );
        put_seq(&mut o, tags::GRAPHIC_ANNOTATION_SEQUENCE, vec![ann]);
        let (set, notes) = read(&o).unwrap();
        assert_eq!(notes.len(), 1);
        assert!(set.structures.iter().all(|s| s.contours.is_empty()));
    }
}
