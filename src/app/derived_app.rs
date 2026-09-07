//! Derived structures in the application: resolving a recipe's operands,
//! working out whether the result is still current, re-evaluating it, and the
//! two rules that keep the whole thing honest.
//!
//! The recipe itself, its storage and its statuses are [`crate::derived`];
//! this is everything that needs to know about slots, sets and jobs.
//!
//! The two rules:
//!
//! * **A hand edit overrides.** The moment a contour tool touches a derived
//!   structure, the recipe stops describing what is on the screen, and the
//!   structure says so (`Overridden`) instead of quietly claiming otherwise.
//! * **An operand that no longer resolves is an error, not a shrug.** A
//!   recipe naming a structure that has been renamed or deleted cannot be
//!   evaluated, and the status says *needs update* with the reason, rather
//!   than combining whatever is left.

use crate::derived::{self, Derived, Expr, Status};
use crate::structops::{self, Operand, Recipe};

use super::combine::ItemRef;
use super::*;

/// The statuses of one structure set, computed once per change rather than
/// once per frame.
#[derive(Default)]
pub(super) struct DerivedCache {
    /// `settings_gen` the cache was built at.
    pub gen: u64,
    /// The set it describes.
    pub set: usize,
    /// One entry per ROI of that set: `None` for a structure that is not
    /// derived.
    pub items: Vec<Option<Status>>,
}

/// What one finished re-evaluation hands back.
pub struct DerivedResult {
    pub set: usize,
    pub roi: usize,
    pub mask: Vec<u8>,
    pub hash: u64,
    pub name: String,
    pub volume_dims: [usize; 3],
    pub frame_of_reference_uid: String,
    pub cm3: f64,
}

impl ViewerApp {
    // -- reading -----------------------------------------------------------

    pub(super) fn derived_of(&self, slot: usize, set: usize, roi: usize) -> Option<Derived> {
        let r = self.slots[slot]
            .study
            .as_ref()?
            .structure_sets
            .get(set)?
            .rois
            .get(roi)?;
        derived::of_roi(r)
    }

    /// Where an operand's name points in this dataset: a structure of any set
    /// (the active one first), else a segment of any series.
    pub(super) fn resolve_dep(&self, slot: usize, name: &str) -> Option<ItemRef> {
        let study = self.slots[slot].study.as_ref()?;
        let active = self.slots[slot].active_structs;
        let sets = (0..study.structure_sets.len())
            .map(|i| if i == 0 { active } else { i })
            .chain(std::iter::once(active));
        for si in sets {
            let Some(set) = study.structure_sets.get(si) else {
                continue;
            };
            if let Some(ii) = set.rois.iter().position(|r| r.name == name) {
                return Some(ItemRef {
                    kind: SetKind::Structures,
                    set: si,
                    idx: ii,
                });
            }
        }
        for (si, ser) in study.seg_series.iter().enumerate() {
            if let Some(ii) = ser.segs.iter().position(|s| s.name == name) {
                return Some(ItemRef {
                    kind: SetKind::Segmentations,
                    set: si,
                    idx: ii,
                });
            }
        }
        None
    }

    /// Fingerprint of everything a recipe depends on, or the name of the
    /// first operand that no longer resolves.
    pub(super) fn dep_fingerprint(&self, slot: usize, expr: &Expr) -> Result<u64, String> {
        let study = self.slots[slot]
            .study
            .as_ref()
            .ok_or_else(|| "no study".to_string())?;
        let mut parts = Vec::with_capacity(expr.deps.len());
        for d in &expr.deps {
            let item = self
                .resolve_dep(slot, &d.name)
                .ok_or_else(|| format!("'{}' is not in this dataset any more", d.name))?;
            let h = match item.kind {
                SetKind::Structures => study
                    .structure_sets
                    .get(item.set)
                    .and_then(|s| s.rois.get(item.idx))
                    .map(derived::hash_roi),
                SetKind::Segmentations => study
                    .seg_series
                    .get(item.set)
                    .and_then(|s| s.segs.get(item.idx))
                    .map(|s| derived::hash_mask(&s.mask)),
            }
            .ok_or_else(|| format!("'{}' is not in this dataset any more", d.name))?;
            // The margin is part of what the result depends on.
            parts.push(h);
            for v in d.margin.all() {
                parts.push((v * 1000.0).round() as i64 as u64);
            }
        }
        for v in expr.margin.all() {
            parts.push((v * 1000.0).round() as i64 as u64);
        }
        Ok(derived::fingerprint(&parts))
    }

    /// Refresh the status cache of the slot's active structure set. Cheap
    /// when nothing changed, which is every frame but the ones that matter.
    pub(super) fn refresh_derived(&mut self, slot: usize) {
        let set = self.slots[slot].active_structs;
        let gen = self.settings_gen;
        if self.derived[slot].gen == gen && self.derived[slot].set == set {
            return;
        }
        let n = self.slots[slot]
            .active_structures()
            .map(|ss| ss.rois.len())
            .unwrap_or(0);
        let mut items = vec![None; n];
        for (i, item) in items.iter_mut().enumerate() {
            let Some(d) = self.derived_of(slot, set, i) else {
                continue;
            };
            let current = self.dep_fingerprint(slot, &d.expr).unwrap_or(0);
            *item = Some(d.status(current));
        }
        self.derived[slot] = DerivedCache { gen, set, items };
    }

    /// The status marker the structure list draws, if the ROI is derived.
    pub(super) fn derived_status(&self, slot: usize, roi: usize) -> Option<Status> {
        self.derived[slot].items.get(roi).copied().flatten()
    }

    // -- writing -----------------------------------------------------------

    fn write_derived(&mut self, slot: usize, set: usize, roi: usize, d: Option<&Derived>) {
        if let Some(r) = self.slots[slot].roi_mut(set, roi) {
            r.description = d.map(|d| d.encode()).unwrap_or_default();
        }
        self.settings_gen += 1;
    }

    /// Attach a recipe to a structure (the Combine window's *Derived* tick).
    pub(super) fn set_derived(&mut self, slot: usize, set: usize, roi: usize, expr: Expr) {
        let hash = self.dep_fingerprint(slot, &expr).unwrap_or(0);
        let d = Derived {
            expr,
            hash,
            overridden: false,
        };
        self.write_derived(slot, set, roi, Some(&d));
    }

    /// Discard the recipe and leave an ordinary structure behind.
    pub(super) fn underive(&mut self, slot: usize, set: usize, roi: usize) {
        self.write_derived(slot, set, roi, None);
    }

    /// A hand edit means the geometry is no longer what the recipe produced.
    /// Called from the contour tools' write-back, once per structure.
    pub(super) fn mark_overridden(&mut self, slot: usize, set: usize, roi: usize) {
        let Some(mut d) = self.derived_of(slot, set, roi) else {
            return;
        };
        if d.overridden {
            return;
        }
        d.overridden = true;
        self.write_derived(slot, set, roi, Some(&d));
    }

    // -- re-evaluating -----------------------------------------------------

    /// Re-run the recipe of one derived structure on a worker thread.
    ///
    /// Same engine as the Combine tool, because it is the same computation:
    /// the operands are rasterized onto the displayed lattice here, where the
    /// sets are known, and `structops` does the rest.
    pub(super) fn start_derived_update(&mut self, slot: usize, set: usize, roi: usize) {
        if self.derived_job.is_some() {
            return;
        }
        let Some(d) = self.derived_of(slot, set, roi) else {
            return;
        };
        let Some(study) = self.slots[slot].study.as_ref() else {
            return;
        };
        let grid = study.volume.grid();
        let name = study
            .structure_sets
            .get(set)
            .and_then(|s| s.rois.get(roi))
            .map(|r| r.name.clone())
            .unwrap_or_default();
        let hash = match self.dep_fingerprint(slot, &d.expr) {
            Ok(h) => h,
            Err(why) => {
                self.error = Some(format!("'{name}' cannot be updated: {why}."));
                return;
            }
        };
        let mut operands = Vec::with_capacity(d.expr.deps.len());
        for dep in &d.expr.deps {
            let Some(item) = self.resolve_dep(slot, &dep.name) else {
                self.error = Some(format!(
                    "'{name}' cannot be updated: '{}' is not in this dataset any more.",
                    dep.name
                ));
                return;
            };
            match self.operand_mask(slot, item, &grid) {
                Some(mask) => operands.push(Operand {
                    name: dep.name.clone(),
                    mask,
                    margin: dep.margin,
                }),
                None => {
                    self.error = Some(format!(
                        "'{name}' cannot be updated: '{}' has nothing on this image \
                         series.",
                        dep.name
                    ));
                    return;
                }
            }
        }
        let recipe = Recipe {
            op: d.expr.op,
            operands,
            margin: d.expr.margin,
            cleanup: d.expr.cleanup,
        };
        let progress = Arc::new(Progress::default());
        progress.set(format!("Updating {name}"));
        self.derived_slot = slot;
        self.derived_job = Some(Job::spawn(progress, move |p| {
            let out = structops::combine(&recipe, &grid, p).map(|c| DerivedResult {
                set,
                roi,
                mask: c.mask,
                hash,
                name,
                volume_dims: grid.dims,
                frame_of_reference_uid: grid.frame_of_reference_uid.clone(),
                cm3: c.cm3,
            });
            (slot, out)
        }));
    }

    /// The evaluation finished: replace the structure's geometry with it and
    /// record the fingerprint it was computed from.
    pub(super) fn on_derived_done(&mut self, slot: usize, result: DerivedResult) {
        if !self.slot_still_shows(slot, result.volume_dims, &result.frame_of_reference_uid) {
            self.error = Some(
                "The dataset changed while the structure was being updated; nothing was \
                 written."
                    .into(),
            );
            return;
        }
        let Some(study) = self.slots[slot].study.as_ref() else {
            return;
        };
        let grid = study.volume.grid();
        let number = study
            .structure_sets
            .get(result.set)
            .and_then(|s| s.rois.get(result.roi))
            .map(|r| r.number)
            .unwrap_or(1);
        let seg = crate::segmentation::Segmentation::from_mask(
            result.name.clone(),
            [200, 200, 200],
            grid.dims,
            result.mask,
        );
        let fresh = crate::segmentation::mask_to_roi(&seg, &grid, number);
        let mut d = match self.derived_of(slot, result.set, result.roi) {
            Some(d) => d,
            None => return,
        };
        d.hash = result.hash;
        d.overridden = false;
        if let Some(r) = self.slots[slot].roi_mut(result.set, result.roi) {
            r.contours = fresh.contours;
        }
        self.write_derived(slot, result.set, result.roi, Some(&d));
        // The working stack of the contour tools may be this very structure.
        self.edit = None;
        self.interp = None;
        self.notice = Some(format!("{} updated: {:.1} cm³", result.name, result.cm3));
    }

    /// Update every derived structure of the active set that needs it, one
    /// after another (the job runs one at a time; the queue is drained as
    /// each finishes).
    pub(super) fn update_all_derived(&mut self, slot: usize) {
        let set = self.slots[slot].active_structs;
        self.refresh_derived(slot);
        let todo: Vec<usize> = self.derived[slot]
            .items
            .iter()
            .enumerate()
            .filter(|(_, s)| matches!(s, Some(Status::NeedsUpdate)))
            .map(|(i, _)| i)
            .collect();
        if todo.is_empty() {
            self.notice = Some("Every derived structure here is up to date.".into());
            return;
        }
        self.derived_queue = todo.iter().skip(1).map(|&r| (set, r)).collect();
        self.start_derived_update(slot, set, todo[0]);
    }

    /// Open the Combine window on an existing recipe, so it can be edited
    /// and re-applied to the same structure.
    pub(super) fn edit_derived(&mut self, slot: usize, set: usize, roi: usize) {
        let Some(d) = self.derived_of(slot, set, roi) else {
            return;
        };
        let name = self.slots[slot]
            .roi(set, roi)
            .map(|r| r.name.clone())
            .unwrap_or_default();
        let roi_type = self.slots[slot]
            .roi(set, roi)
            .map(|r| r.roi_type.clone())
            .unwrap_or_else(|| "ORGAN".into());
        let mut rows = Vec::new();
        for dep in &d.expr.deps {
            let Some(item) = self.resolve_dep(slot, &dep.name) else {
                self.error = Some(format!(
                    "'{}' is not in this dataset any more, so the recipe cannot be \
                     opened as it stands.",
                    dep.name
                ));
                return;
            };
            rows.push(super::combine::Row {
                item,
                margin: dep.margin,
                per_direction: !dep.margin.is_uniform(),
            });
        }
        self.open_combine_dialog(slot, Vec::new());
        if let Some(dlg) = &mut self.combine_dialog {
            dlg.slot = slot;
            dlg.op = d.expr.op;
            dlg.rows = rows;
            dlg.margin = d.expr.margin;
            dlg.margin_per_direction = !d.expr.margin.is_uniform();
            dlg.cleanup = d.expr.cleanup;
            dlg.name = name;
            dlg.output = super::combine::Output::Structure;
            dlg.roi_type = roi_type;
            dlg.derived = true;
            dlg.status = Some(
                "Editing an existing recipe: running it appends a new structure with the \
                 same name; delete the old one when you are happy with it."
                    .into(),
            );
        }
    }

    /// Called when a derived update finishes: start the next queued one.
    pub(super) fn next_derived_in_queue(&mut self, slot: usize) {
        if self.derived_job.is_some() || self.derived_queue.is_empty() {
            return;
        }
        let (set, roi) = self.derived_queue.remove(0);
        self.start_derived_update(slot, set, roi);
    }
}
