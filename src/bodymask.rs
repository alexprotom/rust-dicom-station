//! Automatic **BODY / External** contouring — the outer patient surface,
//! with the couch, the chair and the immobilisation left outside it.
//!
//! Every downstream calculation starts here. A dose engine needs to know
//! where the patient begins, because that is where the range budget starts;
//! a DRR needs to know what is not patient, or it projects the couch through
//! the anatomy; a registration wants to sample inside the body and nowhere
//! else. So this is the one contour that has to be right, on the first scan
//! of the day, without anybody drawing it.
//!
//! Two methods, sharing everything after the first step:
//!
//! * [`Method::Classical`] — thresholding and morphology, deterministic,
//!   instantaneous, nothing to download. Equipment is separated from anatomy
//!   by two geometric facts and no semantics: a device shell is *thin*
//!   (a couch top is two carbon skins around foam that is already below the
//!   threshold; a thermoplastic mask is 2–3 mm), and it is *extruded* —
//!   the same footprint repeats slice after slice, which no part of a
//!   patient does. See [`morphology::axis_persistence`].
//!
//! * [`Method::ModelAssisted`] — TotalSegmentator's openly licensed
//!   body-outline nnU-Net (Apache-2.0, the same engine as
//!   [`crate::autoseg`]) decides *what* is patient; the threshold still
//!   decides *where* the skin is. The network's own output is far too
//!   coarse to be a skin surface — it is planned at 6 mm or 1.5 mm — so it
//!   is used dilated, as a mask on the thresholded image, never as the
//!   answer. This is what removes a device in gap-free contact with the
//!   skin, which no amount of geometry can.
//!
//! Both end with the same post-processing: keep the components big enough to
//! be a patient, give back the thin anatomy the opening took (ears, nose,
//! fingers), fill the interior slice by slice, and optionally close the
//! staircase off the surface.
//!
//! Known limitation, stated rather than hidden: where a shell touches the
//! skin with no air gap at all, the classical method keeps the shell's
//! thickness over the contact patch — 2–5 mm, over the patches only. It is
//! not detectable by geometry, because locally it *is* a slightly thicker
//! patient. The model-assisted method is the answer to that.

use anyhow::{bail, Result};
use rayon::prelude::*;
use std::path::Path;

use crate::autoseg::weights::{ModelSpec, SPEC_BODY_15MM, SPEC_BODY_6MM, SPEC_BODY_MR};
use crate::morphology as morph;
use crate::nn::device::DevicePref;
use crate::progress::{Progress, ProgressSink, CANCELLED};
use crate::volume::Volume;

/// How the patient is told apart from everything else in the image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    /// Threshold + morphology. No download, no network, a few seconds.
    Classical,
    /// A body-outline network decides what is patient; the threshold still
    /// places the skin.
    ModelAssisted,
}

impl Method {
    pub const ALL: [Method; 2] = [Method::Classical, Method::ModelAssisted];
    pub fn label(&self) -> &'static str {
        match self {
            Method::Classical => "Classical (threshold + morphology)",
            Method::ModelAssisted => "Model-assisted (TotalSegmentator body)",
        }
    }
}

/// Which body-outline network the model-assisted method runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BodyModel {
    /// Dataset300, 6 mm — 124 MB, seconds even on a CPU. Plenty, because
    /// the network only has to say *which side of the skin* a voxel is on.
    Ct6mm,
    /// Dataset299, 1.5 mm — 233 MB, minutes on a CPU.
    Ct15mm,
    /// Dataset597, the MR body model — 230 MB.
    Mr,
}

impl BodyModel {
    pub const ALL: [BodyModel; 3] = [BodyModel::Ct6mm, BodyModel::Ct15mm, BodyModel::Mr];
    pub fn label(&self) -> &'static str {
        match self {
            BodyModel::Ct6mm => "CT 6 mm (fast)",
            BodyModel::Ct15mm => "CT 1.5 mm",
            BodyModel::Mr => "MR",
        }
    }
    pub fn spec(&self) -> ModelSpec {
        match self {
            BodyModel::Ct6mm => SPEC_BODY_6MM,
            BodyModel::Ct15mm => SPEC_BODY_15MM,
            BodyModel::Mr => SPEC_BODY_MR,
        }
    }
    /// The model to reach for on a series of this modality.
    pub fn for_modality(modality: &str) -> BodyModel {
        if modality.eq_ignore_ascii_case("MR") {
            BodyModel::Mr
        } else {
            BodyModel::Ct6mm
        }
    }
}

/// How the foreground — "anything at all, patient or not" — is found.
///
/// CT has an absolute scale, so a fixed HU threshold is exactly right. MR
/// has none: the same tissue is a different number on the next sequence,
/// and the receive coils make it a different number on the other side of
/// the same slice. So the MR path divides out a smooth estimate of the coil
/// sensitivity first, then thresholds relative to what is left.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Foreground {
    /// Hounsfield units. −300 sits in the fat/air gap; the skin edge moves
    /// about half a millimetre per 100 HU through the partial-volume ramp.
    Hu(f32),
    /// Fraction of the bias-corrected 99th percentile, with the bias field
    /// estimated by a `sigma_mm` blur.
    MrRelative { fraction: f32, sigma_mm: f64 },
    /// Otsu's threshold on the bias-corrected image — no constant to pick,
    /// but it splits bright from dark rather than tissue from air, so it
    /// runs high on fat-suppressed series.
    MrOtsu { sigma_mm: f64 },
}

impl Foreground {
    /// The sensible default for a series of this modality.
    pub fn for_modality(modality: &str) -> Foreground {
        if modality.eq_ignore_ascii_case("MR") {
            Foreground::MrRelative {
                fraction: 0.12,
                sigma_mm: 40.0,
            }
        } else {
            Foreground::Hu(-300.0)
        }
    }
    pub fn is_mr(&self) -> bool {
        !matches!(self, Foreground::Hu(_))
    }
}

/// Everything one run of the body contour is told.
#[derive(Clone, Debug, PartialEq)]
pub struct BodyParams {
    pub method: Method,
    pub model: BodyModel,
    pub device: DevicePref,
    pub foreground: Foreground,
    /// Radius of the opening that decides what is big enough to *be* a
    /// body. Everything thinner than twice this is set aside for the
    /// thin-anatomy step to judge.
    pub open_mm: f64,
    /// A shell whose largest inscribed ball is smaller than this is a
    /// candidate for equipment — so shells up to about twice it.
    ///
    /// Deliberately far smaller than [`Self::open_mm`], and not a knob to
    /// turn up. A couch skin is one or two millimetres of carbon and a
    /// thermoplastic mask two or three; the thinnest tissue anyone would
    /// miss — the chest wall over a lung — is five or six. At 2 mm the two
    /// are cleanly separated. At 3 mm a six-millimetre chest wall is
    /// itself a candidate, and since it repeats slice after slice it is
    /// then indistinguishable from a couch: the whole ribcage goes.
    pub device_thin_mm: f64,
    /// Run the extruded-equipment test. Off by default in the
    /// model-assisted method, where the network has already answered.
    pub remove_devices: bool,
    /// How far a device footprint has to repeat to count as extruded.
    pub persist_window_mm: f64,
    /// And in what fraction of that window's slices.
    pub persist_frac: f64,
    /// Components smaller than this are noise, a cable or a pillow — not a
    /// patient. Kept as a *volume* so two legs both survive.
    pub min_volume_cm3: f64,
    /// Give back the thin pieces the opening removed.
    pub recover_thin: bool,
    /// How big a piece standing clear of the body's own surface may be and
    /// still count as anatomy — an ear, a nose, a fingertip. A pad, a
    /// blanket or a bolus is larger, and stays out.
    pub thin_max_extent_mm: f64,
    /// How far the network's coarse answer is grown before it is used as a
    /// mask, in the model-assisted method.
    pub guide_margin_mm: f64,
    /// Report the body as the solid object it is, rather than as the shell
    /// of tissue the threshold sees. Off gives the tissue mask instead,
    /// with the lungs and the bowel gas left open.
    pub fill_interior: bool,
    /// Closing radius applied last, to take the staircase off the surface.
    pub close_mm: f64,
    pub name: String,
    /// Also append an RTSTRUCT ROI of type EXTERNAL.
    pub make_external: bool,
}

impl Default for BodyParams {
    fn default() -> Self {
        BodyParams {
            method: Method::Classical,
            model: BodyModel::Ct6mm,
            device: DevicePref::Auto,
            foreground: Foreground::Hu(-300.0),
            open_mm: 8.0,
            device_thin_mm: 2.0,
            remove_devices: true,
            persist_window_mm: 150.0,
            persist_frac: 0.8,
            min_volume_cm3: 50.0,
            recover_thin: true,
            thin_max_extent_mm: 100.0,
            guide_margin_mm: 6.0,
            fill_interior: true,
            close_mm: 0.0,
            name: "BODY".to_string(),
            make_external: true,
        }
    }
}

impl BodyParams {
    /// The defaults that suit a series of this modality — the CT thresholds
    /// are meaningless on MR and vice versa, so the tool window re-seeds
    /// itself whenever the displayed series changes.
    pub fn for_modality(modality: &str) -> BodyParams {
        BodyParams {
            foreground: Foreground::for_modality(modality),
            model: BodyModel::for_modality(modality),
            ..BodyParams::default()
        }
    }
}

/// One piece of the finished contour, for the results line.
#[derive(Clone, Debug)]
pub struct Piece {
    pub voxels: u64,
    pub cm3: f64,
}

/// What a finished run hands back.
pub struct BodyResult {
    /// 0/1 per voxel, in [`Volume::data`] index order. Omitted from the
    /// `Debug` output, which is otherwise 35 MB of ones and zeros.
    pub mask: Vec<u8>,
    pub dims: [usize; 3],
    pub voxels: u64,
    pub cm3: f64,
    /// The separate bodies kept — two legs are two pieces, and saying so is
    /// more use than silently keeping the larger one.
    pub pieces: Vec<Piece>,
    /// Voxels above the threshold that were judged not to be patient — the
    /// equipment the extrusion test caught, plus every component too small
    /// or too detached to be a body.
    pub removed_voxels: u64,
    /// Thin voxels handed back to the body after the opening.
    pub recovered_voxels: u64,
    pub method: Method,
    /// Which device the network ran on; empty for the classical method.
    pub device: String,
    pub elapsed_secs: f64,
    pub name: String,
    pub make_external: bool,
    /// Identity of the volume this was computed on.
    pub frame_of_reference_uid: String,
    pub volume_dims: [usize; 3],
}

/// The mask is 35 MB of ones and zeros on a normal CT, so it is named
/// rather than printed — everything else is what one wants to see when a
/// test or a batch run reports a surprise.
impl std::fmt::Debug for BodyResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BodyResult")
            .field("mask", &format_args!("<{} voxels>", self.mask.len()))
            .field("dims", &self.dims)
            .field("voxels", &self.voxels)
            .field("cm3", &self.cm3)
            .field("pieces", &self.pieces)
            .field("removed_voxels", &self.removed_voxels)
            .field("recovered_voxels", &self.recovered_voxels)
            .field("method", &self.method)
            .field("device", &self.device)
            .field("elapsed_secs", &self.elapsed_secs)
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

/// Total download still needed for the model-assisted method, in bytes.
pub fn download_needed(model: BodyModel, models_dir: &Path) -> u64 {
    let spec = model.spec();
    if crate::autoseg::weights::is_cached(&spec, models_dir) {
        0
    } else {
        spec.zip_bytes
    }
}

// ---------------------------------------------------------------------------
// The pipeline
// ---------------------------------------------------------------------------

/// Contour the patient. Blocking — call from a worker thread and watch it
/// through `progress`.
pub fn contour_body(
    volume: &Volume,
    params: &BodyParams,
    models_dir: &Path,
    progress: &Progress,
) -> Result<BodyResult> {
    let t0 = std::time::Instant::now();
    let dims = volume.dims;
    let spacing = volume.spacing;
    let voxel_cm3 = spacing[0] * spacing[1] * spacing[2] / 1000.0;

    // ---- 1. foreground ---------------------------------------------------
    // Six equal steps follow, over whatever is left of the bar: all of it
    // for the classical method, the last 30 % when a network has had the
    // first 70 %.
    const STEPS: f32 = 6.0;
    let (fg_base, fg_span) = match params.method {
        Method::Classical => (0.0, 1.0 / STEPS),
        Method::ModelAssisted => (0.70, 0.30 / STEPS),
    };
    let mut device = String::new();
    let mut guide: Option<Vec<u8>> = None;
    if params.method == Method::ModelAssisted {
        progress.set_phase(0.0, 0.70);
        let spec = params.model.spec();
        let (labels, dev) = crate::autoseg::run_specs(
            volume,
            &[spec],
            "body outline",
            params.device,
            models_dir,
            progress,
        )?;
        device = dev;
        guide = Some(body_from_labels(&labels, dims, spacing, progress)?);
    }
    progress.set_phase(fg_base, fg_span);
    progress.report(0.0, "Separating tissue from air…");
    let mut fg = foreground(volume, params.foreground);
    if progress.cancelled() {
        bail!(CANCELLED);
    }

    // The network's answer is coarse by construction — it is planned at
    // 6 mm or 1.5 mm — so it is grown by a margin and used as a *mask* on
    // the thresholded image. What survives has the network's semantics and
    // the image's resolution.
    if let Some(g) = &guide {
        let grown = morph::dilate_mm(g, dims, spacing, params.guide_margin_mm);
        fg.par_iter_mut().zip(grown.par_iter()).for_each(|(m, &g)| {
            if g == 0 {
                *m = 0;
            }
        });
    }
    let mut mask = fg.clone();

    // ---- 2. extruded equipment ------------------------------------------
    // Judged on the *unfilled* foreground, while a shell is still a shell,
    // and at a radius small enough that only equipment qualifies.
    progress.set_phase(fg_base + fg_span, fg_span);
    if params.remove_devices {
        progress.report(0.0, "Looking for couch, chair and immobilisation…");
        let opened = morph::open_mm(&mask, dims, spacing, params.device_thin_mm);
        let mut thin: Vec<u8> = mask
            .par_iter()
            .zip(opened.par_iter())
            .map(|(&m, &o)| u8::from(m != 0 && o == 0))
            .collect();
        let mut device_mask = vec![0u8; mask.len()];
        for axis in 0..3 {
            if progress.cancelled() {
                bail!(CANCELLED);
            }
            let p = morph::axis_persistence(
                &thin,
                dims,
                spacing,
                axis,
                params.persist_window_mm,
                params.persist_frac,
            );
            device_mask
                .par_iter_mut()
                .zip(p.par_iter())
                .for_each(|(d, &v)| *d |= v);
            progress.report((axis + 1) as f32 / 3.0, "");
        }
        // A device is thin *and* extruded; whatever is only one of the two
        // stays for the component test to judge.
        thin.par_iter_mut()
            .zip(device_mask.par_iter())
            .for_each(|(t, &d)| *t &= d);
        mask.par_iter_mut()
            .zip(thin.par_iter())
            .for_each(|(m, &d)| {
                if d != 0 {
                    *m = 0;
                }
            });
    }
    if progress.cancelled() {
        bail!(CANCELLED);
    }

    // ---- 3. a body is a solid object -------------------------------------
    // Everything below reasons about the *body*, and a threshold does not
    // see one: it sees a shell of tissue wrapped round two lungs. Left that
    // way, the chest wall over a lung is a five-millimetre sheet that
    // repeats slice after slice — which is to say, indistinguishable from a
    // couch skin, and duly deleted. Filling the interior first is what
    // makes the wall part of a solid object again. It happens after the
    // equipment step, so that a couch top with a closed profile is not
    // turned into a solid slab before it can be recognised.
    //
    // Axial means "perpendicular to the patient's superior axis", which is
    // the volume axis the direction cosines put closest to it, not whichever
    // axis happens to be third in the array.
    let axial = volume.canonical_axes().0[0];
    progress.set_phase(fg_base + 2.0 * fg_span, fg_span);
    progress.report(0.0, "Closing the interior…");
    let mut solid = mask.clone();
    morph::fill_holes_2d(&mut solid, dims, axial);
    if progress.cancelled() {
        bail!(CANCELLED);
    }

    // ---- 4. which components are a patient -------------------------------
    progress.set_phase(fg_base + 3.0 * fg_span, fg_span);
    progress.report(0.0, "Finding the body…");
    let core = morph::open_mm(&solid, dims, spacing, params.open_mm);
    let comps = morph::components(&core, dims);
    if comps.is_empty() {
        bail!(
            "nothing above the threshold looks like a body — lower the threshold \
             or reduce the opening radius"
        );
    }
    let min_voxels = (params.min_volume_cm3 / voxel_cm3).max(1.0) as usize;
    // Everything big enough, not merely the largest: a leg scan is two
    // bodies, and an arm cut off by the field of view is a third.
    let mut kept: Vec<&morph::Component> = comps.iter().filter(|c| c.len() >= min_voxels).collect();
    if kept.is_empty() {
        kept.push(&comps[0]);
    }
    let mut body = vec![0u8; mask.len()];
    for c in &kept {
        for &v in &c.voxels {
            body[v as usize] = 1;
        }
    }
    if progress.cancelled() {
        bail!(CANCELLED);
    }

    // ---- 5. give back the thin anatomy -----------------------------------
    progress.set_phase(fg_base + 4.0 * fg_span, fg_span);
    let mut recovered = 0u64;
    if params.recover_thin {
        progress.report(0.0, "Recovering ears, nose and fingers…");
        // Two questions, because there are two kinds of thing here.
        //
        // What the opening shaved off the body's *own* surface lies, by
        // construction, within one opening radius of what is left — a skin
        // rim, the edge of a shoulder, the sharp flank of a cross-section.
        // It can run the whole length of the scan and still be nothing but
        // patient, so its size says nothing and is not asked.
        //
        // What stands clear of that is a separate object that happens to
        // touch: an ear, a nose and a fingertip, which are small, or a pad,
        // a blanket and a bolus, which are not. There, size is exactly the
        // question. Two rounds, because a fingertip hangs off a finger.
        let reach = params.open_mm + spacing.iter().cloned().fold(0.0, f64::max);
        for _ in 0..2 {
            let residue: Vec<u8> = solid
                .par_iter()
                .zip(body.par_iter())
                .map(|(&m, &b)| u8::from(m != 0 && b == 0))
                .collect();
            let near = morph::dilate_mm(&body, dims, spacing, reach);
            let mut grew = false;
            for c in morph::components(&residue, dims) {
                let shaved_off_the_body = c.voxels.iter().all(|&v| near[v as usize] != 0);
                if !shaved_off_the_body && c.extent_mm(spacing) > params.thin_max_extent_mm {
                    continue;
                }
                if !morph::touches(&c, &body, dims) {
                    continue;
                }
                for &v in &c.voxels {
                    body[v as usize] = 1;
                }
                recovered += c.len() as u64;
                grew = true;
            }
            if !grew {
                break;
            }
        }
    }
    if progress.cancelled() {
        bail!(CANCELLED);
    }

    // ---- 6. finish -------------------------------------------------------
    progress.set_phase(fg_base + 5.0 * fg_span, fg_span);
    progress.report(0.0, "Finishing the surface…");
    // The opening can leave a dent the fill has to close again, and a
    // cavity that is open on every slice can still be enclosed in space.
    morph::fill_holes_2d(&mut body, dims, axial);
    morph::fill_holes_3d(&mut body, dims);
    if params.close_mm > 0.0 {
        body = morph::close_mm(&body, dims, spacing, params.close_mm);
    }
    // Everything above the threshold that is not patient: the equipment the
    // extrusion test caught, plus every component too small or too detached
    // to be a body. Counted here, against the original foreground, because
    // by now the body also contains an interior that was never above the
    // threshold at all.
    let rejected: u64 = fg
        .par_iter()
        .zip(body.par_iter())
        .map(|(&f, &b)| u64::from(f != 0 && b == 0))
        .sum();
    if !params.fill_interior {
        body.par_iter_mut()
            .zip(fg.par_iter())
            .for_each(|(b, &f)| *b &= (f != 0) as u8);
    }

    // ---- 7. statistics ---------------------------------------------------
    let voxels: u64 = body.par_iter().map(|&v| u64::from(v != 0)).sum();
    let pieces: Vec<Piece> = morph::components(&body, dims)
        .into_iter()
        .filter(|c| c.len() >= min_voxels)
        .map(|c| Piece {
            voxels: c.len() as u64,
            cm3: c.cm3(spacing),
        })
        .collect();
    progress.report(1.0, "Body contour finished");
    Ok(BodyResult {
        mask: body,
        dims,
        voxels,
        cm3: voxels as f64 * voxel_cm3,
        pieces,
        removed_voxels: rejected,
        recovered_voxels: recovered,
        method: params.method,
        device,
        elapsed_secs: t0.elapsed().as_secs_f64(),
        name: params.name.clone(),
        make_external: params.make_external,
        frame_of_reference_uid: volume.frame_of_reference_uid.clone(),
        volume_dims: dims,
    })
}

// ---------------------------------------------------------------------------
// Steps
// ---------------------------------------------------------------------------

/// Everything that is not air, by the rule the modality allows.
pub fn foreground(volume: &Volume, how: Foreground) -> Vec<u8> {
    match how {
        Foreground::Hu(t) => {
            let t = t as i16;
            volume.data.par_iter().map(|&v| u8::from(v >= t)).collect()
        }
        Foreground::MrRelative { fraction, sigma_mm } => {
            let flat = flatten_bias(volume, sigma_mm);
            let hi = high_percentile(&flat, 0.99);
            let t = hi * fraction;
            flat.par_iter().map(|&v| u8::from(v >= t)).collect()
        }
        Foreground::MrOtsu { sigma_mm } => {
            let flat = flatten_bias(volume, sigma_mm);
            let t = otsu(&flat);
            flat.par_iter().map(|&v| u8::from(v >= t)).collect()
        }
    }
}

/// Divide out a smooth estimate of the receive-coil sensitivity, so that one
/// threshold holds across the whole image.
///
/// The estimate is a **normalized convolution**: the image blurred far
/// beyond any anatomy (40 mm by default), but weighted so that only voxels
/// plausibly inside *something* contribute, and divided by the same blur of
/// the weights. A plain blur would not do — near the skin, and anywhere the
/// body is thin, it is dominated by the surrounding air and reports a bias
/// that is really just "how much background is nearby", which flattens the
/// anatomy instead of the shading. Weighting fixes exactly that, for the
/// cost of one more pass.
///
/// It is a poor man's N4 — no iteration, no histogram model — which is all a
/// *body outline* needs, because the boundary it is looking for is the
/// largest step in the image. The output is rescaled to the mean signal
/// inside the object, so the numbers stay readable and a threshold given as
/// a fraction of the 99th percentile means the same thing before and after.
///
/// `sigma_mm` has to sit between two scales: **well above** any anatomy, or
/// the estimate follows the tissue and flattens the very contrast the
/// threshold needs, and **well below** the body, or it degenerates into a
/// global mean and corrects nothing. 40 mm on a clinical field of view is
/// comfortably inside that window.
pub fn flatten_bias(volume: &Volume, sigma_mm: f64) -> Vec<f32> {
    let raw: Vec<f32> = volume.data.par_iter().map(|&v| v as f32).collect();
    if sigma_mm <= 0.0 {
        return raw;
    }
    // "Plausibly inside something" — deliberately generous, since this only
    // has to keep the estimate off the air, not find the body.
    let floor = 0.05 * high_percentile(&raw, 0.99);
    let weight: Vec<f32> = raw.par_iter().map(|&v| f32::from(v >= floor)).collect();
    let signal: Vec<f32> = raw
        .par_iter()
        .zip(weight.par_iter())
        .map(|(&v, &w)| v * w)
        .collect();
    let num = morph::blur_mm(&signal, volume.dims, volume.spacing, sigma_mm);
    let den = morph::blur_mm(&weight, volume.dims, volume.spacing, sigma_mm);
    // The level everything is rescaled to, and the fallback wherever the
    // weight is too thin for a local estimate to mean anything.
    let (sum, count) = signal
        .par_iter()
        .zip(weight.par_iter())
        .map(|(&v, &w)| (v as f64, w as f64))
        .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
    if count == 0.0 {
        return raw;
    }
    let mean_in = (sum / count) as f32;
    let guard = (mean_in * 0.05).max(1e-3);
    raw.par_iter()
        .zip(num.par_iter())
        .zip(den.par_iter())
        .map(|((&v, &n), &d)| {
            // A weighted mean is meaningful wherever *any* weight reached
            // this voxel; only genuine 0/0, far from anything, falls back.
            let bias = if d > 1e-6 { n / d } else { mean_in };
            v * mean_in / bias.max(guard)
        })
        .collect()
}

/// The value below which `q` of the (positive) samples fall — a robust
/// stand-in for the maximum, immune to a single hot voxel.
fn high_percentile(v: &[f32], q: f64) -> f32 {
    let mut pos: Vec<f32> = v.par_iter().copied().filter(|x| *x > 0.0).collect();
    if pos.is_empty() {
        return 0.0;
    }
    let k = ((pos.len() as f64 - 1.0) * q).round() as usize;
    let (_, nth, _) = pos.select_nth_unstable_by(k, |a, b| a.total_cmp(b));
    *nth
}

/// Otsu's threshold over a 256-bin histogram of `[0, p99.9]`.
fn otsu(v: &[f32]) -> f32 {
    let hi = high_percentile(v, 0.999);
    if hi <= 0.0 {
        return 0.0;
    }
    let bins = 256usize;
    let scale = bins as f32 / hi;
    let mut hist = vec![0u64; bins];
    for &x in v {
        if x <= 0.0 {
            hist[0] += 1;
        } else {
            hist[((x * scale) as usize).min(bins - 1)] += 1;
        }
    }
    let total: u64 = hist.iter().sum();
    let sum: f64 = hist
        .iter()
        .enumerate()
        .map(|(i, &c)| i as f64 * c as f64)
        .sum();
    let (mut w0, mut s0, mut best, mut best_t) = (0u64, 0f64, -1f64, 0usize);
    for (t, &c) in hist.iter().enumerate() {
        w0 += c;
        if w0 == 0 {
            continue;
        }
        let w1 = total - w0;
        if w1 == 0 {
            break;
        }
        s0 += t as f64 * c as f64;
        let m0 = s0 / w0 as f64;
        let m1 = (sum - s0) / w1 as f64;
        let between = w0 as f64 * w1 as f64 * (m0 - m1) * (m0 - m1);
        if between > best {
            best = between;
            best_t = t;
        }
    }
    // The *upper* edge of the winning bin, not its centre: on a histogram
    // with two well-separated peaks every split between them scores the
    // same, argmax takes the first, and its centre would sit inside the
    // background peak rather than above it.
    (best_t as f32 + 1.0) / scale
}

/// TotalSegmentator's own post-processing of the body task, then the union.
///
/// The network answers in two classes — trunk and extremities — and the
/// reference implementation cleans each differently: the trunk is one
/// object, so only its largest blob is kept, while extremities are several
/// and are merely filtered for size (50 000 mm³, the same constant
/// upstream uses). The body is what is left of both.
fn body_from_labels(
    labels: &[u8],
    dims: [usize; 3],
    spacing: [f64; 3],
    progress: &Progress,
) -> Result<Vec<u8>> {
    progress.report(0.95, "Cleaning up the network's answer…");
    let voxel_mm3 = spacing[0] * spacing[1] * spacing[2];
    let trunk: Vec<u8> = labels.par_iter().map(|&l| u8::from(l == 1)).collect();
    let limbs: Vec<u8> = labels.par_iter().map(|&l| u8::from(l == 2)).collect();
    let mut out = vec![0u8; labels.len()];
    if let Some(c) = morph::components(&trunk, dims).first() {
        for &v in &c.voxels {
            out[v as usize] = 1;
        }
    }
    let min_voxels = (50_000.0 / voxel_mm3).max(1.0) as usize;
    for c in morph::components(&limbs, dims) {
        if c.len() < min_voxels {
            continue;
        }
        for &v in &c.voxels {
            out[v as usize] = 1;
        }
    }
    if out.iter().all(|&v| v == 0) {
        bail!("the body-outline network found no patient in this volume");
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Vec3;

    fn vol(dims: [usize; 3], spacing: [f64; 3], data: Vec<i16>) -> Volume {
        Volume {
            data,
            dims,
            spacing,
            origin: Vec3::new(0.0, 0.0, 0.0),
            row_dir: Vec3::new(1.0, 0.0, 0.0),
            col_dir: Vec3::new(0.0, 1.0, 0.0),
            normal: Vec3::new(0.0, 0.0, 1.0),
            frame_of_reference_uid: "1.2.3".into(),
            min_value: -1000,
            max_value: 1000,
        }
    }

    #[test]
    fn a_hounsfield_threshold_is_exactly_a_threshold() {
        let v = vol([4, 1, 1], [1.0; 3], vec![-1000, -301, -300, 40]);
        let f = foreground(&v, Foreground::Hu(-300.0));
        assert_eq!(f, vec![0, 0, 1, 1]);
    }

    #[test]
    fn flattening_rescues_a_threshold_that_the_coil_shading_had_broken() {
        // The case that matters on MR, and the one a fixed threshold cannot
        // survive: an exponential coil falloff steep enough that background
        // *near* the coil is brighter than body *far* from it. No single
        // threshold on the raw image can separate them — that is asserted
        // below, not assumed.
        let dims = [256, 64, 8];
        let mut data = vec![0i16; dims[0] * dims[1] * dims[2]];
        let at = |i: usize, j: usize, k: usize| k * dims[0] * dims[1] + j * dims[0] + i;
        for i in 0..dims[0] {
            let gain = (-4.0 * i as f32 / dims[0] as f32).exp();
            for j in 0..dims[1] {
                let inside = (16..240).contains(&i) && (8..56).contains(&j);
                let v = if inside { 1000.0 } else { 60.0 } * gain;
                for k in 0..dims[2] {
                    data[at(i, j, k)] = v.round() as i16;
                }
            }
        }
        let v = vol(dims, [1.0; 3], data);
        let (body_near, body_far) = (at(30, 32, 4), at(225, 32, 4));
        let (air_near, air_far) = (at(30, 2, 4), at(225, 2, 4));
        assert!(
            v.data[body_far] < v.data[air_near],
            "the phantom has to be unthresholdable to be worth the test: \
             body {} vs air {}",
            v.data[body_far],
            v.data[air_near]
        );

        let flat = flatten_bias(&v, 20.0);
        assert!(
            flat[body_far] > flat[air_near],
            "flattening did not reorder body and air: {} vs {}",
            flat[body_far],
            flat[air_near]
        );
        let raw_ratio = v.data[body_near] as f32 / v.data[body_far] as f32;
        let flat_ratio = flat[body_near] / flat[body_far];
        assert!(
            flat_ratio < raw_ratio / 4.0,
            "residual shading {flat_ratio:.1} was barely better than {raw_ratio:.1}"
        );

        // The rule the tool actually applies, on the same phantom.
        let mask = foreground(
            &v,
            Foreground::MrRelative {
                fraction: 0.12,
                sigma_mm: 20.0,
            },
        );
        assert_eq!(mask[body_near], 1, "the bright end of the body");
        assert_eq!(mask[body_far], 1, "the dim end of the body");
        assert_eq!(mask[air_near], 0, "bright air near the coil");
        assert_eq!(mask[air_far], 0, "dim air");
    }

    #[test]
    fn otsu_splits_a_two_peaked_histogram_between_the_peaks() {
        let mut v = vec![10.0f32; 500];
        v.extend(std::iter::repeat_n(900.0f32, 500));
        let t = otsu(&v);
        assert!((10.0..900.0).contains(&t), "threshold {t}");
    }

    #[test]
    fn the_defaults_follow_the_modality() {
        let ct = BodyParams::for_modality("CT");
        assert_eq!(ct.foreground, Foreground::Hu(-300.0));
        assert_eq!(ct.model, BodyModel::Ct6mm);
        let mr = BodyParams::for_modality("mr");
        assert!(mr.foreground.is_mr());
        assert_eq!(mr.model, BodyModel::Mr);
    }
}
