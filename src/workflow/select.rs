//! Structures by name: the one way both the viewer's recipes and the MCP
//! server refer to a contour or a segment, and how one becomes a mask on
//! whatever lattice a run needs.

use crate::dicomseg::{resample_mask, SegSeries};
use crate::loader::LoadedStudy;
use crate::propagate::Subject;
use crate::rtstruct::Roi;
use crate::segmentation;
use crate::volume::Grid;

use anyhow::{bail, Context, Result};

/// Where a structure's geometry comes from.
pub enum Source {
    /// RTSTRUCT contours; rasterized onto the lattice a run asks for.
    Contours(Roi),
    /// A voxel mask on the lattice it was made on; resampled when needed.
    Mask { mask: Vec<u8>, grid: Grid },
}

/// One structure frozen for a worker thread: its identity and geometry,
/// with no reference back into the study it came from.
pub struct Structure {
    pub name: String,
    pub color: [u8; 3],
    pub source: Source,
}

impl Structure {
    /// From an RTSTRUCT ROI.
    pub fn from_roi(roi: &Roi) -> Structure {
        Structure {
            name: roi.name.clone(),
            color: roi.color,
            source: Source::Contours(roi.clone()),
        }
    }

    /// From one segment of a segmentation series.
    pub fn from_segment(series: &SegSeries, idx: usize) -> Option<Structure> {
        let seg = series.segs.get(idx)?;
        Some(Structure {
            name: seg.name.clone(),
            color: seg.color,
            source: Source::Mask {
                mask: seg.mask.clone(),
                grid: series.grid.clone(),
            },
        })
    }

    /// The structure as a mask on `grid` (one byte per voxel, 1 inside).
    ///
    /// Contours are rasterized, masks resampled unless the lattice already
    /// matches. An empty result is an error: a run that continued with it
    /// would report a centroid of nothing.
    pub fn mask_on(&self, grid: &Grid) -> Result<Vec<u8>> {
        let mask = match &self.source {
            Source::Contours(roi) => segmentation::rasterize_roi(grid, roi).with_context(|| {
                format!("'{}' has no contour inside the reference phase", self.name)
            })?,
            Source::Mask { mask, grid: from } => {
                if from.matches(grid) {
                    mask.clone()
                } else {
                    resample_mask(mask, from, grid)
                }
            }
        };
        if mask.iter().all(|&v| v == 0) {
            bail!("'{}' is empty on the reference phase", self.name);
        }
        Ok(mask)
    }

    /// The structure as something `propagate` carries, on `grid`.
    pub fn subject_on(&self, grid: &Grid) -> Result<Subject> {
        Ok(Subject {
            name: self.name.clone(),
            color: self.color,
            mask: self.mask_on(grid)?,
        })
    }
}

/// What kind of object a named structure is.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    /// An ROI of an RT structure set.
    Roi,
    /// A segment of a segmentation series.
    Segment,
}

/// One entry of [`list`]: enough to show a structure and to find it again.
pub struct Entry {
    pub name: String,
    pub kind: Kind,
    /// Which structure set / segmentation series, and the position in it.
    pub set: usize,
    pub idx: usize,
    /// Label of the set the structure is in.
    pub set_label: String,
    pub color: [u8; 3],
}

/// Every structure of a study, structure sets first, then segmentation
/// series, each in file order.
pub fn list(study: &LoadedStudy) -> Vec<Entry> {
    let mut out = Vec::new();
    for (set, ss) in study.structure_sets.iter().enumerate() {
        for (idx, roi) in ss.rois.iter().enumerate() {
            out.push(Entry {
                name: roi.name.clone(),
                kind: Kind::Roi,
                set,
                idx,
                set_label: ss.label.clone(),
                color: roi.color,
            });
        }
    }
    for (set, ser) in study.seg_series.iter().enumerate() {
        for (idx, seg) in ser.segs.iter().enumerate() {
            out.push(Entry {
                name: seg.name.clone(),
                kind: Kind::Segment,
                set,
                idx,
                set_label: ser.label.clone(),
                color: seg.color,
            });
        }
    }
    out
}

/// Freeze the structure an [`Entry`] points at.
pub fn structure(study: &LoadedStudy, e: &Entry) -> Option<Structure> {
    match e.kind {
        Kind::Roi => Some(Structure::from_roi(
            study.structure_sets.get(e.set)?.rois.get(e.idx)?,
        )),
        Kind::Segment => Structure::from_segment(study.seg_series.get(e.set)?, e.idx),
    }
}

/// Find a structure by name.
///
/// An exact match wins; failing that, a case-insensitive one. Names repeat
/// across structure sets (every 4D phase may carry its own "Heart"), so
/// `set_label`, when given, narrows the search to one set. With several
/// matches left the *last* one wins: the most recently added set is the one
/// a run just produced.
pub fn find(study: &LoadedStudy, name: &str, set_label: Option<&str>) -> Option<Structure> {
    let all = list(study);
    let in_set = |e: &&Entry| set_label.is_none_or(|l| e.set_label == l);
    let exact = all.iter().filter(in_set).rfind(|e| e.name == name);
    let lax = || {
        let lower = name.to_lowercase();
        all.iter()
            .filter(in_set)
            .rfind(|e| e.name.to_lowercase() == lower)
    };
    exact.or_else(lax).and_then(|e| structure(study, e))
}
