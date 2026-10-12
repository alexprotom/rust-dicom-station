//! Synthetic CT as a series of the study: the body mask, the engine
//! ([`crate::synth`]), and the result written as DICOM files that join the
//! source's study (*Modules ▶ Image synthesis*, the MCP `synthesize_ct`
//! tool, the `SynthesizeCt` workflow node and `sct_cli` all come here).
//!
//! An sCT is a CT image series, not a mask, so it is written as files
//! first and read back through the loader: a series held only in memory
//! cannot be shown again once another series is displayed, nor exported,
//! nor reloaded by a workflow. The files go to `<data folder>/derived/<study
//! UID>/<series UID>/` (never beside the source, which may be read-only),
//! carry the source's patient, study and Frame of Reference, and say what
//! made them (Image Type `DERIVED\SECONDARY`, a DerivationDescription, a
//! Source Image Sequence). [`eval`] compares one with a reference CT.

pub mod eval;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, Context, Result};

use crate::bodymask::{self, BodyModel, BodyParams, Method};
use crate::dicom_export::{self, DerivedImage, DerivedSeries};
use crate::loader::{self, LoadedStudy, SeriesInfo};
use crate::progress::{Progress, ProgressSink, CANCELLED};
use crate::synth::{self, App, Options, SctResult};
use crate::volume::Volume;

/// Where the air outside the patient comes from. The apps set everything
/// outside the body (dilated by five voxels of the working grid) to
/// -1024 HU; upstream runs its own body network at that step.
#[derive(Clone, Debug, PartialEq)]
pub enum BodySource {
    /// The body contour guided by TotalSegmentator's body network (`body`
    /// 6 mm on CT and CBCT, `body_mr` 6 mm on MR): upstream's choice.
    Network,
    /// The body contour by threshold and morphology: nothing to download.
    Classical,
    /// A structure already drawn (BODY, External), as a mask on the scan's
    /// grid.
    Structure { name: String, mask: Vec<u8> },
    /// No mask: nothing is set to air.
    None,
}

impl BodySource {
    /// How the interface and the derivation name it.
    pub fn label(&self) -> String {
        match self {
            BodySource::Network => "body contour (TotalSegmentator body network, 6 mm)".into(),
            BodySource::Classical => "body contour (threshold and morphology)".into(),
            BodySource::Structure { name, .. } => format!("structure {name}"),
            BodySource::None => "none".into(),
        }
    }

    /// The key the MCP tools and the workflow node take: `network`,
    /// `classical`, `none`. Anything else names a structure (BODY,
    /// External), so no structure name is taken as a keyword.
    pub fn from_key(key: &str) -> Option<BodySource> {
        match key.trim().to_ascii_lowercase().as_str() {
            "network" => Some(BodySource::Network),
            "classical" => Some(BodySource::Classical),
            "none" => Some(BodySource::None),
            _ => None,
        }
    }
}

/// One synthetic CT to make.
#[derive(Clone, Debug)]
pub struct SctRequest {
    pub app: App,
    pub opts: Options,
    pub body: BodySource,
    /// Series Number of the new series ([`next_series_number`]); the
    /// uncertainty series takes the next one.
    pub series_number: i64,
}

/// A synthetic CT, written and read back.
pub struct SctOutcome {
    /// The sCT on the source's grid.
    pub volume: Arc<Volume>,
    pub series: DerivedSeries,
    /// The members' standard deviation (HU) as a second series.
    pub uncertainty: Option<(Arc<Volume>, DerivedSeries)>,
    /// The written series, to merge into the source's workspace
    /// ([`loader::merge_study`]); no volume.
    pub study: LoadedStudy,
    pub description: String,
    pub app_label: &'static str,
    pub checkpoints: Vec<usize>,
    pub tta: usize,
    pub members: usize,
    pub device: String,
    /// What masked the air.
    pub body: String,
    /// The clip the normalisation used: the scan's minimum and its 99.5th
    /// percentile on the working grid.
    pub clip: (i16, f64),
    pub working_dims: [usize; 3],
    pub elapsed_secs: f64,
}

/// Where derived series are written: `<data folder>/derived`.
pub fn default_out_root() -> PathBuf {
    crate::settings::data_dir().join("derived")
}

/// One past the highest Series Number of the study (1 for a study with
/// none).
pub fn next_series_number(study: &LoadedStudy) -> i64 {
    study
        .series
        .iter()
        .filter_map(|s| s.series_number)
        .max()
        .map_or(1, |n| n + 1)
}

/// A string cut to `max` characters (DICOM LO is 64).
fn cut(s: &str, max: usize) -> String {
    s.chars().take(max).collect()
}

/// The new series' SeriesDescription: `sCT (MR, 5 CV, TTA 2) from <source>`.
pub fn description(app: App, checkpoints: usize, tta: usize, source: &str) -> String {
    let short = match app {
        App::Mr => "MR",
        App::Cbct => "CBCT",
        App::MrCbct => "MR or CBCT",
        App::Dixon => "Dixon",
        App::Custom(c) => c.key.trim_start_matches("sct_custom_"),
    };
    let src = source.trim();
    let base = format!("sCT ({short}, {checkpoints} CV, TTA {tta})");
    if src.is_empty() {
        base
    } else {
        cut(&format!("{base} from {src}"), 64)
    }
}

/// The description of a series derived from an sCT (`what`: `spread`,
/// `minus CT`): `sCT spread (MR, 5 CV, TTA 2) from <source>`, cut to 64.
pub fn derived_description(what: &str, sct_description: &str) -> String {
    let rest = sct_description.trim();
    match rest.strip_prefix("sCT") {
        Some(tail) => cut(&format!("sCT {what}{tail}"), 64),
        None => cut(&format!("sCT {what}: {rest}"), 64),
    }
}

/// The body mask a request asks for, on the scan's grid.
fn body_mask(
    volume: &Volume,
    modality: &str,
    body: &BodySource,
    root: &Path,
    progress: &Progress,
) -> Result<Option<Vec<u8>>> {
    let method = match body {
        BodySource::None => return Ok(None),
        BodySource::Structure { mask, name } => {
            if mask.len() != volume.data.len() {
                bail!("structure {name} is not on the scan's grid");
            }
            if !mask.iter().any(|&m| m != 0) {
                bail!("structure {name} is empty on this scan");
            }
            return Ok(Some(mask.clone()));
        }
        BodySource::Network => Method::ModelAssisted,
        BodySource::Classical => Method::Classical,
    };
    let mr = modality.trim().eq_ignore_ascii_case("MR");
    let params = BodyParams {
        method,
        model: if mr {
            BodyModel::Mr6mm
        } else {
            BodyModel::Ct6mm
        },
        make_external: false,
        ..BodyParams::for_modality(modality)
    };
    let r = bodymask::contour_body(volume, &params, root, progress).context("body contour")?;
    if r.voxels == 0 {
        bail!("the body contour found no patient on this scan");
    }
    Ok(Some(r.mask))
}

/// A volume on `like`'s grid holding `data`.
pub fn on_grid(like: &Volume, data: Vec<i16>) -> Volume {
    let (lo, hi) = data
        .iter()
        .fold((i16::MAX, i16::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    Volume {
        min_value: if data.is_empty() { 0 } else { lo },
        max_value: if data.is_empty() { 0 } else { hi },
        data,
        dims: like.dims,
        spacing: like.spacing,
        origin: like.origin,
        row_dir: like.row_dir,
        col_dir: like.col_dir,
        normal: like.normal,
        frame_of_reference_uid: like.frame_of_reference_uid.clone(),
    }
}

/// The folder a derived series of `study_uid` is written to.
pub fn series_dir(out_root: &Path, study_uid: &str, series_uid: &str) -> PathBuf {
    let safe = |s: &str| -> String {
        let t: String = s
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '.' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        if t.is_empty() {
            "unknown".into()
        } else {
            t
        }
    };
    out_root.join(safe(study_uid)).join(safe(series_uid))
}

/// Make a synthetic CT of `volume` (the displayed series `source`) and
/// write it as a series of the source's study under `out_root`. Blocking;
/// observe and cancel through `progress`. `root` is the model folder.
pub fn synthesize(
    volume: &Volume,
    source: &SeriesInfo,
    req: &SctRequest,
    root: &Path,
    out_root: &Path,
    progress: &Progress,
) -> Result<SctOutcome> {
    let start = std::time::Instant::now();
    if volume.data.is_empty() {
        bail!("no image to synthesize from");
    }
    let modality = source.modality.trim();
    if !req.app.modality().accepts(modality) {
        bail!(
            "{} turns {} into a CT; this series is {}",
            req.app.label(),
            req.app.modality().label(),
            if modality.is_empty() {
                "of no modality"
            } else {
                modality
            }
        );
    }

    // ---- body mask ---------------------------------------------------------
    let has_mask_step = !matches!(req.body, BodySource::None);
    let (mask_span, write_span): (f32, f32) = (if has_mask_step { 0.12 } else { 0.0 }, 0.06);
    progress.set_outer(0.0, mask_span.max(1e-3));
    progress.set("Finding the body");
    let mask = body_mask(volume, modality, &req.body, root, progress)?;
    if progress.cancelled() {
        bail!(CANCELLED);
    }

    // ---- the engine --------------------------------------------------------
    progress.set_outer(mask_span, 1.0 - mask_span - write_span);
    let r: SctResult = synth::run(volume, mask.as_deref(), req.app, &req.opts, root, progress)?;
    drop(mask);

    // ---- the series ----------------------------------------------------------
    progress.set_outer(1.0 - write_span, write_span);
    progress.report(0.0, "Writing the synthetic CT");
    let sct = Arc::new(on_grid(volume, r.hu));
    let description = description(req.app, r.checkpoints.len(), r.tta, &source.description);
    let derivation = format!(
        "Synthetic CT from {} series {}{} by {} (TotalSynth, Boussot et al. 2026; weights {}) in \
         rust-dicom-station {}: checkpoints {}, test-time mirrors {}, {} members averaged, \
         body mask: {}, working grid 1 x 1 x 3 mm. Research use; not for clinical decisions \
         without the department's own acceptance.",
        if modality.is_empty() { "the" } else { modality },
        source
            .series_number
            .map(|n| n.to_string())
            .unwrap_or_else(|| "(unnumbered)".into()),
        if source.description.trim().is_empty() {
            String::new()
        } else {
            format!(" '{}'", source.description.trim())
        },
        req.app.label(),
        req.app.licence().name(),
        env!("CARGO_PKG_VERSION"),
        r.checkpoints
            .iter()
            .map(|k| format!("CV_{k}"))
            .collect::<Vec<_>>()
            .join(", "),
        r.tta,
        r.members,
        req.body.label(),
    );
    let study_uid = if source.study_uid.is_empty() {
        "unknown".to_string()
    } else {
        source.study_uid.clone()
    };
    let series_uid = dicom_export::new_uid();
    let dir = series_dir(out_root, &study_uid, &series_uid);
    let written = dicom_export::write_derived_image_series(
        &sct,
        source,
        &dir,
        &DerivedImage {
            series_uid: &series_uid,
            description: &description,
            derivation: &derivation,
            series_number: req.series_number,
            window: (40.0, 400.0),
            made_by: "Synthetic CT",
        },
        progress,
    )?;
    let mut files = written.files.clone();
    let uncertainty = match r.sd {
        Some(sd) => {
            let data: Vec<i16> = sd
                .iter()
                .map(|&v| v.round().clamp(0.0, 32767.0) as i16)
                .collect();
            let vol = Arc::new(on_grid(volume, data));
            let hi = crate::loader::percentile_windows(&vol.data, &[(0.0, 99.0)])[0];
            let uid = dicom_export::new_uid();
            let desc = derived_description("spread", &description);
            let s = dicom_export::write_derived_image_series(
                &vol,
                source,
                &series_dir(out_root, &study_uid, &uid),
                &DerivedImage {
                    series_uid: &uid,
                    description: &desc,
                    derivation: &format!(
                        "Voxel-wise standard deviation (HU) of the {} members of the synthetic CT \
                         '{description}': {derivation}",
                        r.members
                    ),
                    series_number: req.series_number + 1,
                    window: hi,
                    made_by: "Synthetic CT",
                },
                progress,
            )?;
            files.extend(s.files.iter().cloned());
            Some((vol, s))
        }
        None => None,
    };
    let study = loader::load_files_to_merge(&files, &dir.display().to_string(), progress)
        .context("read the synthetic CT back")?;
    progress.report(1.0, "Synthetic CT written");
    progress.set_outer(0.0, 1.0);
    Ok(SctOutcome {
        volume: sct,
        series: written,
        uncertainty,
        study,
        description,
        app_label: req.app.label(),
        checkpoints: r.checkpoints,
        tta: r.tta,
        members: r.members,
        device: r.device,
        body: req.body.label(),
        clip: r.clip,
        working_dims: r.working_dims,
        elapsed_secs: start.elapsed().as_secs_f64(),
    })
}

/// A one-line account of an outcome, for status lines and the MCP reply.
pub fn summary(o: &SctOutcome) -> String {
    format!(
        "{}: {} member{} ({} checkpoint{}, {} mirror{}) on {} in {:.0} s",
        o.description,
        o.members,
        if o.members == 1 { "" } else { "s" },
        o.checkpoints.len(),
        if o.checkpoints.len() == 1 { "" } else { "s" },
        o.tta,
        if o.tta == 1 { "" } else { "s" },
        if o.device.is_empty() {
            "CPU"
        } else {
            &o.device
        },
        o.elapsed_secs
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptions_fit_a_long_string() {
        let d = description(App::Mr, 5, 2, "T2 TSE axial pelvis");
        assert_eq!(d, "sCT (MR, 5 CV, TTA 2) from T2 TSE axial pelvis");
        let long = description(App::Cbct, 1, 0, &"x".repeat(100));
        assert_eq!(long.chars().count(), 64);
        assert_eq!(
            description(App::MrCbct, 1, 0, "  "),
            "sCT (MR or CBCT, 1 CV, TTA 0)"
        );
    }

    #[test]
    fn derived_series_say_what_they_are() {
        assert_eq!(
            derived_description("spread", "sCT (MR, 5 CV, TTA 2) from T2 pelvis"),
            "sCT spread (MR, 5 CV, TTA 2) from T2 pelvis"
        );
        assert_eq!(
            derived_description("minus CT", "Body CT"),
            "sCT minus CT: Body CT"
        );
        assert_eq!(
            derived_description("minus CT", &"x".repeat(90))
                .chars()
                .count(),
            64
        );
    }

    #[test]
    fn body_sources_have_keys() {
        assert_eq!(BodySource::from_key("Network"), Some(BodySource::Network));
        assert_eq!(
            BodySource::from_key("classical"),
            Some(BodySource::Classical)
        );
        assert_eq!(BodySource::from_key("none"), Some(BodySource::None));
        assert_eq!(BodySource::from_key("BODY"), None, "a structure's name");
    }

    #[test]
    fn series_folders_are_safe_names() {
        let d = series_dir(Path::new("/d"), "1.2.3", "2.25.9");
        assert_eq!(d, Path::new("/d/1.2.3/2.25.9"));
        assert_eq!(
            series_dir(Path::new("/d"), "", "a/b"),
            Path::new("/d/unknown/a_b")
        );
    }
}
