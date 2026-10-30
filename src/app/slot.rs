//! A workspace's state: its study, its three views and what they
//! keep between frames, and the structures prepared for drawing.

use super::*;

// Per-viewport state and caches

pub(super) struct ViewState {
    pub(super) plane: ViewPlane,
    pub(super) slice: usize,
    /// Screen pixels per mm; 0 means auto-fit.
    pub(super) zoom: f32,
    /// Pan offset of the image center relative to the viewport center, mm.
    pub(super) pan: Vec2,
    /// Fractional mouse-wheel line accumulator for slice stepping.
    pub(super) scroll_accum: f32,
    pub(super) tex: Option<TextureHandle>,
    pub(super) dose_tex: Option<TextureHandle>,
    pub(super) img_key: Option<(usize, u32, u32)>,
    pub(super) dose_key: Option<u64>,
    pub(super) contour_key: Option<u64>,
    pub(super) slice_buf: Vec<i16>,
    /// What `slice_buf` holds: (volume, plane, slice). A window/level drag
    /// changes only the mapping to grey, so the slice itself is kept.
    pub(super) slice_key: Option<(usize, ViewPlane, usize)>,
    pub(super) dose_plane: Vec<f32>,
    pub(super) iso_segs: Vec<(usize, render::Segment)>,
    pub(super) contours: Vec<(usize, render::RoiPlaneGraphics)>,
    /// The sagittal / coronal crossings of `contours` as egui tessellates
    /// them, and what they were tessellated for (the contours, their
    /// colours and widths, where the image sits on screen, the pixel
    /// density). A dense set crosses a slice in tens of thousands of short
    /// strokes, and tessellating them all again on every repaint - every
    /// pointer move over the window - was the cost of drawing it. Behind a
    /// lock only because the paint pass reads the view through `&self`.
    pub(super) contour_mesh: std::sync::Mutex<Option<(u64, Arc<egui::epaint::Mesh>)>>,
    pub(super) fusion_tex: Option<TextureHandle>,
    pub(super) fusion_key: Option<u64>,
    /// The other image sampled at every pixel of the fused slice (`None`
    /// outside it), and what it was sampled for: everything but the window
    /// and the blend weight, which only recolour these values.
    pub(super) fusion_samples: Vec<Option<f32>>,
    pub(super) fusion_geom: Option<u64>,
    pub(super) seg_tex: Option<TextureHandle>,
    pub(super) seg_key: Option<u64>,
    /// Identity of the vector-field geometry cached below.
    pub(super) field_key: Option<u64>,
    /// Arrows of the deformation field on this slice, display-pixel space.
    pub(super) field_arrows: Vec<registration::dvf::Glyph>,
    /// The deformed lattice of this slice, display-pixel space.
    pub(super) field_lines: Vec<Vec<[f32; 2]>>,
}

impl ViewState {
    pub(super) fn new(plane: ViewPlane) -> Self {
        ViewState {
            plane,
            slice: 0,
            zoom: 0.0,
            pan: Vec2::ZERO,
            scroll_accum: 0.0,
            tex: None,
            dose_tex: None,
            img_key: None,
            dose_key: None,
            contour_key: None,
            slice_buf: Vec::new(),
            slice_key: None,
            dose_plane: Vec::new(),
            iso_segs: Vec::new(),
            contours: Vec::new(),
            contour_mesh: std::sync::Mutex::new(None),
            fusion_tex: None,
            fusion_key: None,
            fusion_samples: Vec::new(),
            fusion_geom: None,
            seg_tex: None,
            seg_key: None,
            field_key: None,
            field_arrows: Vec::new(),
            field_lines: Vec::new(),
        }
    }

    pub(super) fn invalidate(&mut self) {
        self.img_key = None;
        self.slice_key = None;
        self.dose_key = None;
        self.contour_key = None;
        // The crossings' mesh is keyed on `contour_key`, which a rebuild on
        // another phase can bring back with the same value.
        *self
            .contour_mesh
            .get_mut()
            .unwrap_or_else(|e| e.into_inner()) = None;
        self.fusion_key = None;
        self.fusion_geom = None;
        self.seg_key = None;
        self.field_key = None;
    }
}

/// Which of the three view slots shows a plane - the order `fresh_views`
/// builds them in.
pub(super) fn plane_index(plane: ViewPlane) -> usize {
    match plane {
        ViewPlane::Axial => 0,
        ViewPlane::Sagittal => 1,
        ViewPlane::Coronal => 2,
    }
}

fn fresh_views() -> [ViewState; 3] {
    [
        ViewState::new(ViewPlane::Axial),
        ViewState::new(ViewPlane::Sagittal),
        ViewState::new(ViewPlane::Coronal),
    ]
}

// A loaded study with its own display state ("A" = primary, "B" = comparison)

pub(super) struct StudySlot {
    pub(super) study: Option<LoadedStudy>,
    pub(super) views: [ViewState; 3],
    /// Fractional voxel coords of the linked crosshair (in this slot's volume).
    pub(super) cursor: [f64; 3],
    pub(super) roi_visible: Vec<bool>,
    /// Index of the active plan within `study.plans`: the one whose details
    /// are listed and whose isocenters the views draw. One at a time, like
    /// the structure sets and the segmentation series.
    pub(super) active_plan: usize,
    /// The row's own switch: whether the active plan's isocenters are drawn
    /// at all. Clicking the active row again turns them off without losing
    /// which plan is selected.
    pub(super) plans_shown: bool,
    /// The dose row's own switch, the same for the active dose grid: whether
    /// this workspace draws it. *How* dose is drawn (colorwash, isodose
    /// lines) is `ViewerApp::dose_mode`, one choice for every workspace.
    pub(super) dose_shown: bool,
    /// Index of the active structure set within `study.structure_sets`.
    pub(super) active_structs: usize,
    /// The tick box on the RT structures series row: whether the active set
    /// is drawn at all. Unticking it clears the contours from the image views
    /// while the set stays selected, so the ROI list, the drawing tools and a
    /// 3D scene opened on it all keep working.
    pub(super) structs_shown: bool,
    /// The same tick box on the segmentation series row.
    pub(super) segs_shown: bool,
    pub(super) active_dose: usize,
    pub(super) dose_reference: f32,
    /// Index of the active segmentation series within `study.seg_series`.
    pub(super) active_seg_series: usize,
    /// Index of the segment the tools edit, within that series.
    pub(super) active_seg: usize,
    /// Index of the ROI the contour tools edit, within the active structure
    /// set. Out of range simply means "none yet".
    pub(super) active_roi: usize,
    /// The active structure set's ROIs mapped into the displayed volume's
    /// voxel space, for the image views' contours - for the last few
    /// (set, volume) pairs, so a 4D cine that follows the structures maps
    /// each phase's set once rather than on every pass.
    pub(super) roi_prep: RoiPreps,
    /// Per (structure set, ROI): the workspace's contour counter
    /// (`ViewerApp::struct_gen`) when a contour edit last wrote that ROI.
    /// What a per-ROI cache compares, so a brush sample costs the edited
    /// structure and not the set (see [`StudySlot::roi_stamp`]).
    pub(super) roi_edits: std::collections::HashMap<(usize, usize), u64>,
    /// The row's two planar panes, and the additional pane after them
    /// (`planar::ADDITIONAL_PANE`), which works the same way over the
    /// workspace's additional objects.
    pub(super) planar: [PlanarView; settings::PLANAR_PANES + 1],
    /// The planar pane that was given an image last: Ctrl-click in the
    /// tree adds to it, and a plain click with both panes on screen goes to
    /// the other one.
    pub(super) planar_last: usize,
}

/// The ROIs of one structure set prepared for drawing on one volume
/// ([`render::PreparedRoi`]): scrolling through a view, dragging the
/// crosshair or playing a 4D cine then costs the contours that cross each
/// slice, instead of mapping every point of every visible structure per view
/// per step. Filled per ROI the first time it is drawn, and again for one
/// ROI when a contour edit has written it since ([`StudySlot::roi_stamp`]).
pub(super) struct RoiPrep {
    /// (structure set index, `ViewerApp::settings_gen`) the ROIs are for.
    /// Contour edits do not move `settings_gen`; they are told apart per
    /// ROI by the stamp kept with each entry.
    pub(super) key: (usize, u64),
    /// The volume they were mapped into. Weak: a cache must not keep a
    /// replaced volume's voxels alive, and unlike its address a `Weak` cannot
    /// be taken over by the next volume allocated in the same place.
    pub(super) volume: std::sync::Weak<Volume>,
    /// Per ROI: the edit stamp it was prepared at, and the preparation.
    pub(super) rois: Vec<Option<(u64, render::PreparedRoi)>>,
}

impl RoiPrep {
    /// Memory the prepared ROIs hold.
    pub(super) fn bytes(&self) -> usize {
        self.rois.iter().flatten().map(|(_, p)| p.bytes()).sum()
    }
}

/// The [`RoiPrep`]s of the last few (structure set, volume) pairs of one
/// workspace, the one in use last.
///
/// Stepping a 4D cine with *follow structures* on changes both the set and
/// the volume at every phase; with one cache, each step mapped the whole
/// set of the new phase again, on every pass through the cycle. A phase's
/// volume is kept alive by the phase cache, so its entry stays usable for
/// as long as the cine runs.
#[derive(Default)]
pub(super) struct RoiPreps {
    pub(super) sets: Vec<RoiPrep>,
}

impl RoiPreps {
    /// How many pairs are kept, and how much memory they may hold together
    /// (the one in use is never dropped for it).
    const KEEP: usize = 12;
    const BYTES: usize = 256 << 20;

    /// The cache for `(set, gen)` on `vol`, made empty when there is none.
    /// Entries of an older `gen`, or of a volume that is gone, are dropped:
    /// nothing can ask for them again.
    pub(super) fn for_set(
        &mut self,
        set: usize,
        gen: u64,
        vol: &Arc<Volume>,
        n_rois: usize,
    ) -> &mut RoiPrep {
        let found = self.sets.iter().position(|p| {
            p.key == (set, gen)
                && std::ptr::eq(p.volume.as_ptr(), Arc::as_ptr(vol))
                && p.rois.len() == n_rois
        });
        match found {
            Some(i) => {
                let p = self.sets.remove(i);
                self.sets.push(p);
            }
            None => {
                self.sets
                    .retain(|p| p.key.1 == gen && p.volume.strong_count() > 0);
                let mut rois = Vec::new();
                rois.resize_with(n_rois, || None);
                self.sets.push(RoiPrep {
                    key: (set, gen),
                    volume: Arc::downgrade(vol),
                    rois,
                });
            }
        }
        // Over either cap: the least recently used go first.
        while self.sets.len() > Self::KEEP
            || (self.sets.len() > 1
                && self.sets.iter().map(RoiPrep::bytes).sum::<usize>() > Self::BYTES)
        {
            self.sets.remove(0);
        }
        self.sets.last_mut().expect("just pushed or moved there")
    }
}

impl StudySlot {
    /// The currently selected structure set of this slot, if any.
    pub(super) fn active_structures(&self) -> Option<&crate::rtstruct::StructureSet> {
        self.study
            .as_ref()
            .and_then(|s| s.structure_sets.get(self.active_structs))
    }

    /// Whether this slot shows an image volume.
    ///
    /// A slot can hold a perfectly good workspace with none - RT images, a
    /// structure set, a plan. Every feature that needs voxels (the MPR views,
    /// the brush, registration, the segmentation engines, the DRR) asks this
    /// rather than `study.is_some()`.
    pub(super) fn has_volume(&self) -> bool {
        self.study.as_ref().is_some_and(|st| st.has_volume())
    }

    /// Voxel spacing of the displayed volume, or unit spacing without one -
    /// what every conversion from voxels to millimetres falls back on.
    pub(super) fn spacing_or_unit(&self) -> [f64; 3] {
        self.study
            .as_ref()
            .map(|st| st.volume.spacing)
            .unwrap_or([1.0; 3])
    }

    /// `voxels` of the displayed volume, in cm³.
    pub(super) fn voxels_cm3(&self, voxels: u64) -> f64 {
        let sp = self.spacing_or_unit();
        voxels as f64 * crate::volume::voxel_cm3(sp)
    }

    /// The edit stamp of one ROI: the contour counter of the workspace when
    /// a contour edit last wrote it, 0 when none has. Only meaningful next to
    /// `settings_gen`, which everything else that touches a structure moves.
    pub(super) fn roi_stamp(&self, set: usize, roi: usize) -> u64 {
        self.roi_edits.get(&(set, roi)).copied().unwrap_or(0)
    }

    /// One ROI of one structure set, by index.
    pub(super) fn roi(&self, set: usize, roi: usize) -> Option<&crate::rtstruct::Roi> {
        self.study
            .as_ref()
            .and_then(|st| st.structure_sets.get(set))
            .and_then(|ss| ss.rois.get(roi))
    }

    /// [`Self::roi`] for editing.
    pub(super) fn roi_mut(&mut self, set: usize, roi: usize) -> Option<&mut crate::rtstruct::Roi> {
        self.study
            .as_mut()
            .and_then(|st| st.structure_sets.get_mut(set))
            .and_then(|ss| ss.rois.get_mut(roi))
    }

    /// Series UID of the displayed volume, when there is one.
    pub(super) fn displayed_uid(&self) -> Option<&str> {
        let st = self.study.as_ref()?;
        if !st.has_volume() {
            return None;
        }
        st.series.get(st.active_series).map(|se| se.uid.as_str())
    }

    /// Index of the segmentation series the tools edit, clamped to what the
    /// study actually holds.
    pub(super) fn seg_series_idx(&self) -> Option<usize> {
        let st = self.study.as_ref()?;
        (!st.seg_series.is_empty()).then(|| self.active_seg_series.min(st.seg_series.len() - 1))
    }

    /// Segments of the active segmentation series - empty unless they live
    /// on the displayed volume's lattice, because every overlay, brush
    /// stroke and mesh indexes them with that volume's dimensions. A series
    /// drawn on another image series simply has nothing to show here.
    pub(super) fn segs(&self) -> &[Segmentation] {
        match (self.study.as_ref(), self.seg_series_idx()) {
            (Some(st), Some(i))
                if st.has_volume() && st.seg_series[i].grid.dims == st.volume.dims =>
            {
                &st.seg_series[i].segs
            }
            _ => &[],
        }
    }

    /// [`Self::segs`] for editing; `None` when there is nothing editable.
    pub(super) fn segs_mut(&mut self) -> Option<&mut Vec<Segmentation>> {
        let i = self.seg_series_idx()?;
        let st = self.study.as_mut()?;
        if !st.has_volume() {
            return None;
        }
        let dims = st.volume.dims;
        let ser = &mut st.seg_series[i];
        (ser.grid.dims == dims).then_some(&mut ser.segs)
    }
}

impl StudySlot {
    pub(super) fn empty() -> Self {
        StudySlot {
            study: None,
            views: fresh_views(),
            cursor: [0.0; 3],
            roi_visible: Vec::new(),
            active_plan: 0,
            plans_shown: true,
            dose_shown: true,
            active_structs: 0,
            structs_shown: true,
            segs_shown: true,
            active_dose: 0,
            dose_reference: 1.0,
            active_seg_series: 0,
            active_seg: 0,
            active_roi: usize::MAX,
            roi_prep: RoiPreps::default(),
            roi_edits: std::collections::HashMap::new(),
            planar: Default::default(),
            planar_last: 1,
        }
    }
}

#[cfg(test)]
mod roi_prep_tests {
    use super::*;
    use crate::geometry::Vec3;

    fn volume() -> Arc<Volume> {
        let dims = [4, 4, 3];
        Arc::new(Volume {
            data: vec![0i16; dims[0] * dims[1] * dims[2]],
            dims,
            spacing: [1.0, 1.0, 1.0],
            origin: Vec3::ZERO,
            row_dir: Vec3 {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
            col_dir: Vec3 {
                x: 0.0,
                y: 1.0,
                z: 0.0,
            },
            normal: Vec3 {
                x: 0.0,
                y: 0.0,
                z: 1.0,
            },
            frame_of_reference_uid: String::new(),
            min_value: 0,
            max_value: 0,
        })
    }

    fn prepared(vol: &Volume) -> render::PreparedRoi {
        let roi = crate::rtstruct::Roi {
            number: 1,
            name: "r".into(),
            color: [1, 2, 3],
            roi_type: "ORGAN".into(),
            description: String::new(),
            contours: Vec::new(),
        };
        render::PreparedRoi::new(vol, &roi)
    }

    /// A cine that follows the structures steps through (set, volume)
    /// pairs: each pair keeps what it prepared until the cycle comes back
    /// to it, while anything else that touches structures (a new
    /// `settings_gen`) starts every pair afresh.
    #[test]
    fn the_phases_of_a_cine_keep_their_prepared_structures() {
        let (a, b) = (volume(), volume());
        let mut preps = RoiPreps::default();
        preps.for_set(0, 7, &a, 2).rois[1] = Some((0, prepared(&a)));
        preps.for_set(1, 7, &b, 3).rois[0] = Some((0, prepared(&b)));
        // Back on the first phase: its ROI is still there.
        assert!(preps.for_set(0, 7, &a, 2).rois[1].is_some());
        assert!(preps.for_set(1, 7, &b, 3).rois[0].is_some());
        assert_eq!(preps.sets.len(), 2);
        // Another generation: nothing of the old one survives.
        assert!(preps.for_set(0, 8, &a, 2).rois.iter().all(Option::is_none));
        assert_eq!(preps.sets.len(), 1);
        // A volume that is gone takes its entry with it.
        let c = volume();
        preps.for_set(2, 8, &c, 1);
        drop(c);
        preps.for_set(0, 8, &a, 2);
        preps.for_set(3, 8, &b, 1);
        assert_eq!(preps.sets.len(), 2);
        // No more than `KEEP` pairs at a time.
        let vols: Vec<_> = (0..RoiPreps::KEEP + 3).map(|_| volume()).collect();
        for (i, v) in vols.iter().enumerate() {
            preps.for_set(i, 8, v, 1);
        }
        assert_eq!(preps.sets.len(), RoiPreps::KEEP);
    }
}
