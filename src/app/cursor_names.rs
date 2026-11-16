//! The names at the crosshair: the RT structures and segments that hold the
//! point the crosshair stands on, written in the status bar after its
//! position and value and before the dose.
//!
//! **What counts is what is drawn.** A structure counts when the views draw
//! it: the workspace's active structure set with its series row ticked, the
//! structure's own tick box on, and *View ▸ Contours* on. A segment counts
//! when its series is the active one, laid on the displayed volume's lattice,
//! its row ticked, and its own tick box on. Hidden ones are left out: a name
//! the readout gives for something not on screen would be a name for nothing
//! the person can see.
//!
//! **Inside.** A structure holds the point when the contours of the slice
//! nearest to it enclose it, by the even-odd rule
//! ([`crate::render::PreparedRoi::contains`]): the same contours, mapped
//! into the volume through its direction vectors from their LPS millimetres,
//! that the axial view draws on that slice. The mapping is the views' own,
//! taken from [`super::slot::RoiPreps`], so a structure set is mapped once
//! for both. A segment holds the point when its mask is set at the voxel
//! the crosshair rounds to, the voxel whose value the readout gives.
//!
//! **Cost.** Looked up when the key moves: the crosshair, the displayed
//! volume, the structures' generation, ticks and set
//! ([`ViewerApp::contour_settings_hash`]), and the segments' generations and
//! ticks ([`ViewerApp::seg_overlay_hash`]). Between those a frame costs the
//! key, as the views' caches do.

use super::*;

/// How many names the readout writes before it counts the rest.
pub(super) const NAMES_SHOWN: usize = 3;

/// The names at one workspace's crosshair, and what they were found for.
#[derive(Default)]
pub(super) struct CursorNames {
    /// 0 until a lookup ran.
    key: u64,
    /// Structures first, in the set's order, then segments in the series'.
    pub(super) names: Vec<String>,
}

/// The readout's text for `names`: the first [`NAMES_SHOWN`], comma
/// separated, then `+N` for the rest; with the whole list, one a line, for a
/// tooltip when some were left out. `None` when there are none.
pub(super) fn names_text(names: &[String]) -> Option<(String, Option<String>)> {
    if names.is_empty() {
        return None;
    }
    let shown = names.len().min(NAMES_SHOWN);
    let mut text = names[..shown].join(", ");
    if names.len() > shown {
        text.push_str(&format!(", +{}", names.len() - shown));
        return Some((text, Some(names.join("\n"))));
    }
    Some((text, None))
}

impl ViewerApp {
    /// What the names at workspace `slot`'s crosshair depend on.
    fn cursor_names_key(&self, slot: usize) -> u64 {
        let s = &self.slots[slot];
        let Some(study) = s.study.as_ref() else {
            return 0;
        };
        let mut h = mix(0xC0_5A_4E_77, slot as u64 + 1);
        for c in s.cursor {
            h = mix(h, c.to_bits());
        }
        h = mix(h, Arc::as_ptr(&study.volume) as u64);
        if self.show_contours && s.structs_shown {
            h = mix(h, self.contour_settings_hash(slot));
        }
        if s.segs_shown {
            h = mix(h, s.seg_series_idx().map_or(0, |i| i as u64 + 1));
            h = mix(h, self.seg_overlay_hash(slot));
        }
        h | 1
    }

    /// Look up the names at workspace `slot`'s crosshair again if anything
    /// they depend on moved since the last time.
    pub(super) fn refresh_cursor_names(&mut self, slot: usize) {
        if !self.slots[slot].has_volume() {
            self.slots[slot].cursor_names = CursorNames::default();
            return;
        }
        let key = self.cursor_names_key(slot);
        if self.slots[slot].cursor_names.key == key {
            return;
        }
        let contours_on = self.show_contours && self.slots[slot].structs_shown;
        let seg_idx = self.slots[slot]
            .segs_shown
            .then(|| self.slots[slot].seg_series_idx())
            .flatten();
        let settings_gen = self.settings_gen;
        let StudySlot {
            study,
            cursor,
            roi_visible,
            active_structs,
            roi_prep,
            roi_edits,
            cursor_names,
            ..
        } = &mut self.slots[slot];
        let Some(study) = study.as_ref() else {
            return;
        };
        let vol = &study.volume;
        let mut names: Vec<String> = Vec::new();
        if let Some(ss) = study
            .structure_sets
            .get(*active_structs)
            .filter(|_| contours_on)
        {
            let visible: Vec<usize> = (0..ss.rois.len())
                .filter(|&ri| roi_visible.get(ri).copied().unwrap_or(false))
                .collect();
            let prep =
                roi_prep.prepared(*active_structs, settings_gen, vol, ss, roi_edits, &visible);
            for ri in visible {
                if prep.rois[ri]
                    .as_ref()
                    .is_some_and(|(_, p)| p.contains(*cursor))
                {
                    names.push(ss.rois[ri].name.clone());
                }
            }
        }
        if let Some(ser) = seg_idx
            .and_then(|i| study.seg_series.get(i))
            .filter(|sr| sr.grid.dims == vol.dims)
        {
            if let Some(at) = voxel_index(*cursor, vol.dims) {
                for seg in ser.segs.iter().filter(|g| g.visible) {
                    if seg.mask.get(at).is_some_and(|&m| m != 0) {
                        names.push(seg.name.clone());
                    }
                }
            }
        }
        // One name once: a structure and a segment of the same organ say
        // the same thing.
        let mut seen = std::collections::HashSet::new();
        names.retain(|n| seen.insert(n.clone()));
        *cursor_names = CursorNames { key, names };
    }
}

/// The index into a volume's (or a mask's) data of the voxel `c` rounds
/// to, when it is inside `dims`.
fn voxel_index(c: [f64; 3], dims: [usize; 3]) -> Option<usize> {
    let mut ijk = [0usize; 3];
    for a in 0..3 {
        let r = c[a].round();
        if !(r >= 0.0 && r < dims[a] as f64) {
            return None;
        }
        ijk[a] = r as usize;
    }
    Some(ijk[2] * dims[0] * dims[1] + ijk[1] * dims[0] + ijk[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn a_few_names_are_written_and_the_rest_counted() {
        assert_eq!(names_text(&[]), None);
        assert_eq!(names_text(&s(&["Liver"])), Some(("Liver".into(), None)));
        assert_eq!(
            names_text(&s(&["PTV", "CTV", "GTV"])),
            Some(("PTV, CTV, GTV".into(), None))
        );
        assert_eq!(
            names_text(&s(&["Body", "PTV", "CTV", "GTV", "Liver"])),
            Some((
                "Body, PTV, CTV, +2".into(),
                Some("Body\nPTV\nCTV\nGTV\nLiver".into())
            ))
        );
    }

    #[test]
    fn the_crosshair_voxel_is_the_one_it_rounds_to() {
        let dims = [4, 3, 2];
        assert_eq!(voxel_index([0.0, 0.0, 0.0], dims), Some(0));
        assert_eq!(voxel_index([1.4, 0.6, 0.0], dims), Some(4 + 1));
        assert_eq!(voxel_index([3.4, 2.4, 1.4], dims), Some(12 + 8 + 3));
        assert_eq!(voxel_index([3.6, 0.0, 0.0], dims), None);
        assert_eq!(voxel_index([-0.6, 0.0, 0.0], dims), None);
        assert_eq!(voxel_index([f64::NAN, 0.0, 0.0], dims), None);
    }
}
