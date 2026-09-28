//! What every headless caller of the pipelines holds while it works: the
//! image volumes it has read, and the rules for filing what the engines
//! make.
//!
//! Two callers run the program's steps without the viewer: a workflow run
//! ([`crate::workflow::graph::exec`]) and the MCP server (`mcp::session`).
//! Both open studies, read the volumes of series other than the displayed
//! one, and file masks as RT structures or segments on the image they
//! belong to. The parts of that which have to behave the same for both -
//! and the same as the viewer, where the viewer does the same thing - are
//! here:
//!
//! * [`Volumes`]: the volumes read so far, kept under a memory budget and
//!   shared by everything one run does, the 4D pipelines included - a
//!   workflow whose second step walks the phases the first step walked
//!   finds them in memory instead of on disk;
//! * [`file_items`]: filing masks into an image's structure set and / or
//!   segmentation series, with a rule for names that are taken
//!   ([`NameClash`]) that gives a structure the same name on every phase of
//!   a 4D group;
//! * [`new_uid`]: identifiers for what is made in memory, unique even when
//!   two are made in the same instant.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::{bail, Result};

use crate::dicomseg::{resample_mask, SegSeries};
use crate::loader::{self, LoadedStudy, SeriesInfo};
use crate::progress::Progress;
use crate::propagate::Propagated;
use crate::rtstruct::StructureSet;
use crate::segmentation::Segmentation;
use crate::volume::{Grid, Volume};

pub use crate::workflow::params::{NameClash, OutputKind, SetChoice};

/// A new UID under the `2.25.` root: the clock and a process-wide counter,
/// so two structure sets made in the same millisecond - the phases of a 4D
/// group filed one after another - still get identifiers of their own.
pub fn new_uid() -> String {
    crate::dicom_export::new_uid()
}

// ---- volumes ------------------------------------------------------------

/// Volumes kept whatever the budget says: the two images a registration
/// reads.
const MIN_KEEP: usize = 2;

struct Kept {
    /// The series UID and its first file: the same series read from two
    /// folders is the same data, a UID reused by another export is not.
    key: (String, PathBuf),
    vol: Arc<Volume>,
    bytes: usize,
}

/// The image volumes one run has read, newest last, under a memory budget.
struct Cache {
    kept: Vec<Kept>,
    budget: usize,
    /// Kept whatever the budget: [`MIN_KEEP`], or none for a cache that
    /// keeps nothing.
    min_keep: usize,
}

impl Cache {
    fn bytes(&self) -> usize {
        self.kept.iter().map(|k| k.bytes).sum()
    }

    fn trim(&mut self) {
        while self.kept.len() > self.min_keep && self.bytes() > self.budget {
            self.kept.remove(0);
        }
    }
}

fn key_of(series: &SeriesInfo) -> (String, PathBuf) {
    (
        series.uid.clone(),
        series.files.first().cloned().unwrap_or_default(),
    )
}

/// The volumes a run has read, shared by every step and every pipeline it
/// calls (cloning the handle shares the cache).
///
/// A 4D workflow walks the same phases several times: segmented on every
/// phase, then carried onto every phase, then measured through every
/// phase. A phase of a thoracic 4DCT is a few hundred megabytes that take
/// seconds to read. So what was read is kept, the most recently used last,
/// until the budget is spent; the two newest are kept whatever it says,
/// since a registration needs both its images at once.
#[derive(Clone)]
pub struct Volumes(Arc<Mutex<Cache>>);

impl Volumes {
    /// A cache of up to `budget_mb` megabytes.
    pub fn with_budget_mb(budget_mb: usize) -> Volumes {
        Volumes(Arc::new(Mutex::new(Cache {
            kept: Vec::new(),
            budget: budget_mb.saturating_mul(1024 * 1024),
            min_keep: MIN_KEEP,
        })))
    }

    /// A handle that keeps nothing: every volume is read from disk, as the
    /// viewer's own windows read them (they hold what they display).
    pub fn none() -> Volumes {
        Volumes(Arc::new(Mutex::new(Cache {
            kept: Vec::new(),
            budget: 0,
            min_keep: 0,
        })))
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Cache> {
        self.0.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// The series' volume, from memory when it is kept, else read (and
    /// kept). The read happens outside the lock, so two branches of a run
    /// reading two series do not wait for each other.
    pub fn load(&self, series: &SeriesInfo, p: &Progress) -> Result<Arc<Volume>> {
        if let Some(v) = self.get(series) {
            return Ok(v);
        }
        let (vol, _, _) = loader::load_series_volume(series, p)?;
        let vol = Arc::new(vol);
        self.put(series, vol.clone());
        Ok(vol)
    }

    /// The series' volume when it is kept; it becomes the newest.
    pub fn get(&self, series: &SeriesInfo) -> Option<Arc<Volume>> {
        let key = key_of(series);
        let mut c = self.lock();
        let i = c.kept.iter().position(|k| k.key == key)?;
        let k = c.kept.remove(i);
        let v = k.vol.clone();
        c.kept.push(k);
        Some(v)
    }

    /// Keep a volume read elsewhere.
    pub fn put(&self, series: &SeriesInfo, vol: Arc<Volume>) {
        let key = key_of(series);
        let mut c = self.lock();
        c.kept.retain(|k| k.key != key);
        let bytes = vol.data.len() * std::mem::size_of::<i16>();
        c.kept.push(Kept { key, vol, bytes });
        c.trim();
    }

    /// Megabytes held now.
    pub fn held_mb(&self) -> usize {
        self.lock().bytes() / (1024 * 1024)
    }

    /// How many volumes are held.
    pub fn len(&self) -> usize {
        self.lock().kept.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for Volumes {
    fn default() -> Self {
        Volumes::none()
    }
}

impl std::fmt::Debug for Volumes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Volumes({} held)", self.len())
    }
}

// ---- filing -------------------------------------------------------------

/// Where and how masks are filed on an image.
#[derive(Clone, Debug)]
pub struct Filing<'a> {
    /// RT structures, segments, or both.
    pub kind: OutputKind,
    /// The image's own structure set (segmentation series), or a new one.
    pub set: SetChoice,
    /// The label of a set or series that has to be made.
    pub set_label: &'a str,
    /// What happens to a name that is taken. `None` files the names as
    /// they are, duplicates and all (a segmentation series allows them).
    pub clash: Option<NameClash>,
    /// Names taken beyond the ones in the set filed into: on a 4D group,
    /// every phase's ([`taken_names`]), so that each phase picks the same
    /// name.
    pub taken: Option<&'a BTreeSet<String>>,
}

/// What [`file_items`] did.
#[derive(Clone, Debug, Default)]
pub struct Filed {
    /// Item by item, in the order given: the name it was filed under, or
    /// `None` when it was not filed (an empty mask, or no contour).
    pub names: Vec<Option<String>>,
    /// The structure set filed into, by index.
    pub set: Option<usize>,
    /// The segmentation series filed into, by index.
    pub seg_series: Option<usize>,
}

/// The structure set structures filed on `series_uid` go into: its own (the
/// last that references it), unless a new one is asked for.
fn own_set(study: &LoadedStudy, series_uid: &str, set: SetChoice) -> Option<usize> {
    match set {
        SetChoice::Own => study
            .structure_sets
            .iter()
            .rposition(|s| s.referenced_series_uid == series_uid),
        SetChoice::New => None,
    }
}

/// The segmentation series segments filed on `series_uid` (lattice `grid`)
/// go into, under the same rule.
fn own_seg_series(
    study: &LoadedStudy,
    series_uid: &str,
    grid: Option<&Grid>,
    set: SetChoice,
) -> Option<usize> {
    match set {
        SetChoice::Own => study.seg_series.iter().rposition(|s| {
            s.referenced_series_uid == series_uid && grid.is_none_or(|g| s.grid.matches(g))
        }),
        SetChoice::New => None,
    }
}

/// Every name a filing on these images would meet: the structures of the
/// set, and the segments of the series, each image's filing goes into.
pub fn taken_names<'a>(
    study: &LoadedStudy,
    series: impl IntoIterator<Item = &'a SeriesInfo>,
    kind: OutputKind,
    set: SetChoice,
) -> BTreeSet<String> {
    let mut taken = BTreeSet::new();
    for se in series {
        if kind.structures() {
            if let Some(i) = own_set(study, &se.uid, set) {
                taken.extend(study.structure_sets[i].rois.iter().map(|r| r.name.clone()));
            }
        }
        if kind.segments() {
            if let Some(i) = own_seg_series(study, &se.uid, None, set) {
                taken.extend(study.seg_series[i].segs.iter().map(|s| s.name.clone()));
            }
        }
    }
    taken
}

/// The name `base` is filed under when `taken` holds the names already
/// there: itself when free (or when it replaces), else `base (2)`,
/// `base (3)`; `base_prop` first under [`NameClash::Suffix`].
pub fn chosen_name(base: &str, clash: NameClash, taken: &BTreeSet<String>) -> String {
    let stem = match clash {
        NameClash::Replace => return base.to_string(),
        NameClash::Counter => base.to_string(),
        NameClash::Suffix if taken.contains(base) => format!("{base}_prop"),
        NameClash::Suffix => base.to_string(),
    };
    if !taken.contains(&stem) {
        return stem;
    }
    let mut n = 2;
    while taken.contains(&format!("{stem} ({n})")) {
        n += 1;
    }
    format!("{stem} ({n})")
}

/// A new, empty structure set bound to one image series; returns its index.
pub fn new_structure_set(
    study: &mut LoadedStudy,
    series: &SeriesInfo,
    grid: &Grid,
    label: &str,
    file_name: &str,
) -> usize {
    let sop = new_uid();
    study.structure_sets.push(StructureSet {
        label: label.to_string(),
        frame_of_reference_uid: grid.frame_of_reference_uid.clone(),
        series_instance_uid: new_uid(),
        sop_instance_uid: sop,
        study_uid: series.study_uid.clone(),
        referenced_series_uid: series.uid.clone(),
        file_name: file_name.to_string(),
        locked: false,
        rois: Vec::new(),
    });
    study.structure_sets.len() - 1
}

/// File masks made on the image `series` (lattice `grid`) of `study`: as
/// contours in its structure set, as segments of its segmentation series,
/// or both - the way the viewer's tools file them. `roi_types` is each
/// item's RT ROI Interpreted Type (empty when unknown).
pub fn file_items(
    study: &mut LoadedStudy,
    series: &SeriesInfo,
    grid: &Grid,
    items: &[Propagated],
    roi_types: &[String],
    f: &Filing,
) -> Result<Filed> {
    let mut filed = Filed {
        names: vec![None; items.len()],
        ..Filed::default()
    };
    let set_idx = f
        .kind
        .structures()
        .then(|| own_set(study, &series.uid, f.set));
    let seg_idx = f
        .kind
        .segments()
        .then(|| own_seg_series(study, &series.uid, Some(grid), f.set));
    if let Some(Some(i)) = set_idx {
        let ss = &study.structure_sets[i];
        if ss.locked {
            bail!(
                "the structure set '{}' is locked (approved); file into a new set instead",
                ss.label
            );
        }
    }

    // The names, chosen once for both kinds of filing and in item order, so
    // a structure is `heart (2)` as a contour and as a segment alike.
    let mut taken: BTreeSet<String> = f.taken.cloned().unwrap_or_default();
    if let Some(Some(i)) = set_idx {
        taken.extend(study.structure_sets[i].rois.iter().map(|r| r.name.clone()));
    }
    if let Some(Some(i)) = seg_idx {
        taken.extend(study.seg_series[i].segs.iter().map(|s| s.name.clone()));
    }
    let mut names: Vec<Option<String>> = Vec::with_capacity(items.len());
    for it in items {
        if it.voxels == 0 {
            names.push(None);
            continue;
        }
        let name = match f.clash {
            None => it.name.clone(),
            Some(c) => chosen_name(&it.name, c, &taken),
        };
        taken.insert(name.clone());
        names.push(Some(name));
    }
    let replace = f.clash == Some(NameClash::Replace);

    if f.kind.structures() {
        let idx = match set_idx.flatten() {
            Some(i) => i,
            None => new_structure_set(study, series, grid, f.set_label, "workflow"),
        };
        let ss = &mut study.structure_sets[idx];
        for (k, it) in items.iter().enumerate() {
            let Some(name) = &names[k] else {
                continue;
            };
            let seg = Segmentation::from_label_map(name.clone(), it.color, grid.dims, &it.mask, 1);
            let mut roi = crate::segmentation::mask_to_roi(&seg, grid, 0);
            if roi.contours.is_empty() {
                continue;
            }
            roi.roi_type = roi_types.get(k).cloned().unwrap_or_default();
            if replace {
                ss.rois.retain(|r| r.name != *name);
            }
            roi.number = ss.rois.iter().map(|r| r.number).max().unwrap_or(0) + 1;
            filed.names[k] = Some(name.clone());
            ss.rois.push(roi);
        }
        filed.set = Some(idx);
    }
    if f.kind.segments() {
        let idx = match seg_idx.flatten() {
            Some(i) => i,
            None => {
                study.seg_series.push(SegSeries::new(
                    f.set_label.to_string(),
                    grid.clone(),
                    series.uid.clone(),
                    series.study_uid.clone(),
                ));
                study.seg_series.len() - 1
            }
        };
        let sr = &mut study.seg_series[idx];
        for (k, it) in items.iter().enumerate() {
            let Some(name) = &names[k] else {
                continue;
            };
            let mask = if sr.grid.matches(grid) {
                it.mask.clone()
            } else {
                resample_mask(&it.mask, grid, &sr.grid)
            };
            if replace {
                sr.segs.retain(|s| s.name != *name);
            }
            sr.segs.push(Segmentation::from_mask(
                name.clone(),
                it.color,
                sr.grid.dims,
                mask,
            ));
            if filed.names[k].is_none() {
                filed.names[k] = Some(name.clone());
            }
        }
        filed.seg_series = Some(idx);
    }
    Ok(filed)
}

/// A mask as something [`file_items`] files.
pub fn item(name: impl Into<String>, color: [u8; 3], mask: Vec<u8>) -> Propagated {
    let voxels = crate::morphology::count_set(&mask);
    Propagated {
        name: name.into(),
        color,
        mask,
        voxels,
        source_cm3: 0.0,
        result_cm3: 0.0,
        mapped_cm3: 0.0,
        source_surface_cm3: None,
        rigid_residual_mm: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Vec3;

    fn grid() -> Grid {
        Grid {
            dims: [12, 10, 6],
            spacing: [2.0, 2.0, 3.0],
            origin: Vec3::ZERO,
            row_dir: Vec3::new(1.0, 0.0, 0.0),
            col_dir: Vec3::new(0.0, 1.0, 0.0),
            normal: Vec3::new(0.0, 0.0, 1.0),
            frame_of_reference_uid: "1.2.3".into(),
        }
    }

    fn block(g: &Grid, lo: [usize; 3], hi: [usize; 3]) -> Vec<u8> {
        let [nx, ny, nz] = g.dims;
        let mut m = vec![0u8; nx * ny * nz];
        for k in lo[2]..hi[2] {
            for j in lo[1]..hi[1] {
                for i in lo[0]..hi[0] {
                    m[(k * ny + j) * nx + i] = 1;
                }
            }
        }
        m
    }

    fn study_with_series(uids: &[&str]) -> (LoadedStudy, Vec<SeriesInfo>) {
        let series: Vec<SeriesInfo> = uids
            .iter()
            .map(|u| SeriesInfo {
                uid: u.to_string(),
                study_uid: "9.9".into(),
                modality: "CT".into(),
                ..SeriesInfo::default()
            })
            .collect();
        let study = LoadedStudy {
            series: series.clone(),
            ..LoadedStudy::default()
        };
        (study, series)
    }

    #[test]
    fn a_taken_name_gets_the_same_counter_on_every_phase() {
        let g = grid();
        let (mut st, series) = study_with_series(&["p0", "p1"]);
        let filing = |taken| Filing {
            kind: OutputKind::Structures,
            set: SetChoice::Own,
            set_label: "Phase set",
            clash: Some(NameClash::Counter),
            taken,
        };
        // Phase 0 already has a target; phase 1 does not.
        let target = item("target", [1, 2, 3], block(&g, [2, 2, 1], [6, 6, 4]));
        file_items(&mut st, &series[0], &g, &[target], &[], &filing(None)).unwrap();
        let taken = taken_names(&st, &series, OutputKind::Structures, SetChoice::Own);
        assert!(taken.contains("target"));
        let again = || item("target", [1, 2, 3], block(&g, [3, 3, 1], [7, 7, 4]));
        let a = file_items(
            &mut st,
            &series[0],
            &g,
            &[again()],
            &[],
            &filing(Some(&taken)),
        )
        .unwrap();
        let b = file_items(
            &mut st,
            &series[1],
            &g,
            &[again()],
            &[],
            &filing(Some(&taken)),
        )
        .unwrap();
        assert_eq!(a.names, vec![Some("target (2)".to_string())]);
        assert_eq!(b.names, a.names, "the same name on the phase that had none");
        assert_ne!(a.set, b.set, "each phase's own set");
    }

    #[test]
    fn a_suffix_or_a_replacement_instead_of_a_counter() {
        let g = grid();
        let (mut st, series) = study_with_series(&["s"]);
        let file = |st: &mut LoadedStudy, clash, kind| {
            let f = Filing {
                kind,
                set: SetChoice::Own,
                set_label: "Set",
                clash: Some(clash),
                taken: None,
            };
            let it = item("gtv", [9, 9, 9], block(&g, [1, 1, 1], [4, 4, 3]));
            file_items(st, &series[0], &g, &[it], &[], &f).unwrap()
        };
        file(&mut st, NameClash::Counter, OutputKind::Both);
        let s = file(&mut st, NameClash::Suffix, OutputKind::Both);
        assert_eq!(s.names, vec![Some("gtv_prop".to_string())]);
        let s2 = file(&mut st, NameClash::Suffix, OutputKind::Both);
        assert_eq!(s2.names, vec![Some("gtv_prop (2)".to_string())]);
        let r = file(&mut st, NameClash::Replace, OutputKind::Both);
        assert_eq!(r.names, vec![Some("gtv".to_string())]);
        let ss = &st.structure_sets[r.set.unwrap()];
        assert_eq!(ss.rois.iter().filter(|x| x.name == "gtv").count(), 1);
        let sr = &st.seg_series[r.seg_series.unwrap()];
        assert_eq!(sr.segs.iter().filter(|x| x.name == "gtv").count(), 1);
        assert_eq!(sr.segs.len(), 3, "gtv, gtv_prop, gtv_prop (2)");
    }

    #[test]
    fn identifiers_made_together_differ() {
        let a = new_uid();
        let b = new_uid();
        assert_ne!(a, b);
        assert!(a.starts_with("2.25."));
    }

    #[test]
    fn the_volume_cache_keeps_to_its_budget_but_never_below_two() {
        let series = |u: &str| SeriesInfo {
            uid: u.into(),
            ..SeriesInfo::default()
        };
        let vol = |n: usize| {
            Arc::new(Volume {
                data: vec![0; n],
                ..Volume::empty()
            })
        };
        // 1 MB budget; each volume 0.75 MB.
        let v = Volumes::with_budget_mb(1);
        let n = 3 * 1024 * 1024 / 8;
        v.put(&series("a"), vol(n));
        v.put(&series("b"), vol(n));
        v.put(&series("c"), vol(n));
        assert_eq!(v.len(), 2, "over budget, but two are kept");
        assert!(v.get(&series("a")).is_none(), "the oldest went");
        assert!(v.get(&series("b")).is_some());
        // b is now the newest: c goes next.
        v.put(&series("d"), vol(n));
        assert!(v.get(&series("c")).is_none());
        assert!(v.get(&series("b")).is_some());
        let none = Volumes::none();
        none.put(&series("a"), vol(10));
        assert!(none.is_empty(), "a cache that keeps nothing");
    }
}
