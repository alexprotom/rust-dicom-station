//! The live 3D structure scene: mesh cache, camera, and the painter-order
//! software renderer that draws RTSTRUCT and segmentation surfaces.
//!
//! The scene is a pane of a workspace's row ([`PaneKind::Scene3d`]), put in
//! and taken out by the toolbar's *3D* button or *Settings ▸ View layout*.
//! It used to have a window of its own as well; the pane did everything the
//! window did, with its own bar, so the window went. The state is still
//! called [`D3Window`], one per workspace, made when the pane is first drawn
//! and dropped when it leaves the row.

use super::*;

/// What *Prepare phases* answers with: the structure meshes of each phase,
/// each under the [`ViewerApp::d3_key_of`] the window will look it up by.
pub(super) type PhaseMeshes = Vec<(u64, Vec<RoiMesh>)>;

/// A segmentation meshing result: the meshes, the generation each was built
/// from, and the geometry they were built on (see `D3Window::seg_mesh_gens`).
pub(super) type SegMeshes = (Vec<RoiMesh>, Vec<u64>, u64);

pub(super) struct D3Window {
    pub(super) slot: usize,
    pub(super) yaw: f32,
    pub(super) pitch: f32,
    /// Zoom multiplier on the auto-fit scale.
    pub(super) zoom: f32,
    pub(super) pan: Vec2,
    /// The hand: while it is on, dragging with the left button moves the
    /// scene instead of turning it. Middle-drag pans either way, but a
    /// wheel mouse is not something every desk has.
    pub(super) pan_mode: bool,
    pub(super) opacity: f32,
    pub(super) meshes: Option<Arc<Vec<RoiMesh>>>,
    /// Scene bounding-sphere (patient mm) for auto-fit.
    pub(super) center: [f32; 3],
    pub(super) radius: f32,
    /// Identity of the structure set the meshes were built from.
    pub(super) key: u64,
    /// Identity of the image the scene stands on ([`ViewerApp::d3_scene_id`]).
    /// When it changes the camera starts over - turned, zoomed and centred
    /// for the new anatomy - since a view kept from another image orbits a
    /// point that is not in this one.
    pub(super) scene: u64,
    /// Fit the camera to the meshes when they land: the first time, and
    /// after the image changed; a rebuild after an edit keeps the view where
    /// it was.
    pub(super) refit: bool,
    /// Counts every set of meshes that landed; the frame cache is keyed on
    /// it. Only [`super::d3::set_meshes`] may change `meshes`, because a
    /// set that arrives without bumping this is drawn through the previous
    /// set's triangle order.
    pub(super) mesh_gen: u64,
    /// Geometry hash per ROI the meshes were built from, and the ROIs a
    /// running partial rebuild replaces (`None`: a full build).
    pub(super) roi_hashes: Vec<u64>,
    /// The edit stamp per ROI ([`StudySlot::roi_stamp`]) at the same time,
    /// and the (set, `settings_gen`) they belong to: while that pair holds,
    /// the stamps alone say which structures an edit changed.
    pub(super) roi_stamps: Vec<u64>,
    pub(super) roi_base: Option<(usize, u64)>,
    pub(super) rebuilding: Option<Vec<usize>>,
    /// Isodose surfaces of the workspace's active dose at the isodose lines'
    /// levels: the meshes (`roi_index` = level index), the build in flight,
    /// what they were built from, and their opacity.
    pub(super) show_iso: bool,
    pub(super) iso_meshes: Option<Arc<Vec<RoiMesh>>>,
    pub(super) iso_job: Option<Job<Vec<RoiMesh>>>,
    pub(super) iso_built: u64,
    pub(super) iso_opacity: f32,
    /// The *Structures* panel: opacity per structure, times the window's
    /// own, keyed by ROI index; absent means 1.
    pub(super) show_list: bool,
    pub(super) roi_alpha: std::collections::HashMap<usize, f32>,
    pub(super) job: Option<Job<Vec<RoiMesh>>>,
    /// Live meshes of the painted segmentations (`roi_index` = seg index).
    pub(super) seg_meshes: Option<Arc<Vec<RoiMesh>>>,
    /// The [`Segmentation::gen`] each of `seg_meshes` was built from, in the
    /// same order, and the volume geometry they were built on: a rebuild
    /// re-meshes only the segments whose generation it has no mesh for.
    pub(super) seg_mesh_gens: Vec<u64>,
    pub(super) seg_geom: u64,
    pub(super) seg_job: Option<Job<SegMeshes>>,
    /// Hash of the segmentation state `seg_meshes` was built from.
    pub(super) seg_built: u64,
    /// Also draw the *other* workspace's structures, mapped through the active
    /// registration - the two anatomies in one scene is what makes a
    /// deformable result readable at all.
    pub(super) show_other: bool,
    /// Opacity of that second workspace, independent of this one's.
    pub(super) other_opacity: f32,
    pub(super) other_meshes: Option<Arc<Vec<RoiMesh>>>,
    pub(super) other_job: Option<Job<Vec<RoiMesh>>>,
    /// Identity of the (structure set, registration) `other_meshes` were
    /// built from.
    pub(super) other_key: u64,
    /// Draw the deformation field as arrows in the scene.
    pub(super) show_field: bool,
    /// Paint the surfaces with the dose that falls on them instead of the
    /// structure's own colour: what a hot spot does to an organ, on the
    /// organ, rather than on a slice through it.
    pub(super) show_dose: bool,
    /// Structure meshes already built, by the key they were built for.
    /// Each phase of a 4D group has its own structure set, so stepping
    /// through them would otherwise re-mesh the same anatomy on every
    /// pass; *Prepare phases* fills this up front so the cine runs.
    pub(super) mesh_cache: std::collections::HashMap<u64, Arc<Vec<RoiMesh>>>,
    /// The `structures_gen` the cache was filled at. An edit changes every
    /// key at once, so the meshes are dropped rather than kept as meshes of
    /// something that no longer exists.
    pub(super) mesh_cache_gen: u64,
    /// The isodose shells, filled as the phases are seen, by the dose and
    /// lines they were built for: no structure edit touches them.
    pub(super) iso_cache: std::collections::HashMap<u64, Arc<Vec<RoiMesh>>>,
    /// Meshing every phase of the group up front.
    pub(super) prep_job: Option<Job<PhaseMeshes>>,
    /// Cached projected geometry for the current camera.
    pub(super) frame: D3Frame,
}

/// Cached triangle soup of a 3D window.
///
/// egui repaints on every pointer move, and projecting + depth-sorting a few
/// hundred thousand triangles takes several milliseconds, so the soup is
/// rebuilt only when something it actually depends on changes. The draw
/// order depends on orientation and visibility alone, so panning and zooming
/// reuse the existing sort; the turned and shaded vertices (the lighting, the
/// dose painted on the surface) do not depend on pan or zoom either, so those
/// only place the kept vertices again.
#[derive(Default)]
pub(super) struct D3Frame {
    /// Identity of the current depth sort (orientation + visibility + meshes).
    pub(super) order_key: Option<u64>,
    /// Identity of the turned, shaded vertices: the sort's, plus colours,
    /// opacities, the dose painted on them and the centre turned about.
    pub(super) shade_key: Option<u64>,
    /// Identity of the current projected vertices (also zoom / pan / size).
    pub(super) vertex_key: Option<u64>,
    pub(super) mesh: Arc<egui::epaint::Mesh>,
    /// Per vertex, the turned position across and up the screen before
    /// scaling and centring: what a pan or a zoom places again.
    pub(super) base: Vec<[f32; 2]>,
    /// Triangles in scene order, indexed by the sort below.
    pub(super) tris: Vec<[u32; 3]>,
    /// View-space depth per vertex.
    pub(super) depth: Vec<f32>,
    /// `(monotone depth key) << 32 | triangle slot`, sorted far-to-near.
    pub(super) order: Vec<u64>,
}

/// Map an `f32` to a `u32` that sorts in the same order, so the painter's
/// algorithm can use a primitive sort instead of a float comparator.
#[inline]
fn depth_key(d: f32) -> u32 {
    let b = d.to_bits();
    if b & 0x8000_0000 != 0 {
        !b
    } else {
        b ^ 0x8000_0000
    }
}

/// The identity of one structure's geometry, for the 3D window's partial
/// rebuilds.
///
/// Geometry only. The colour is not part of it: a mesh carries the colour
/// it was built with, but the scene paints every structure in the colour
/// the structure set gives it *now* ([`ViewerApp::d3_scene`]), so a
/// recolour never has to re-mesh anything - and never fails to show, in a
/// window or in a row, whether or not anything else made the meshes
/// rebuild.
pub(super) fn roi_hash(roi: &crate::rtstruct::Roi) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    roi.contours.len().hash(&mut h);
    for c in &roi.contours {
        c.geometric_type.hash(&mut h);
        for p in &c.points {
            (p.x.to_bits(), p.y.to_bits(), p.z.to_bits()).hash(&mut h);
        }
    }
    h.finish()
}

/// Identity of a volume geometry, for the segmentation meshes kept from one
/// build to the next: a mesh is in patient millimetres, so it only stands for
/// its mask on the lattice it was built on.
fn grid_geom_key(g: &GridGeom) -> u64 {
    let mut h = 0x6A09_E667_F3BC_C908u64;
    for v in [g.origin, g.row_dir, g.col_dir, g.normal] {
        for c in [v.x, v.y, v.z] {
            h = mix(h, c.to_bits());
        }
    }
    for s in g.spacing {
        h = mix(h, s.to_bits());
    }
    h
}

/// One segment of a segmentation meshing run.
enum SegMeshPart {
    /// (segment index, generation, position in the last build's meshes):
    /// unchanged since then, so its mesh is taken over.
    Keep(usize, u64, usize),
    /// (segment index, generation, colour, snapshot): meshed afresh.
    Build(usize, u64, [u8; 3], crate::segmentation::MeshGrid),
}

impl SegMeshPart {
    /// The segment's mesh and the generation it stands for.
    fn mesh(self, old: Option<&[RoiMesh]>, geom: &GridGeom) -> Option<(RoiMesh, u64)> {
        match self {
            SegMeshPart::Keep(i, gen, pos) => {
                let mut m = old?.get(pos)?.clone();
                m.roi_index = i;
                Some((m, gen))
            }
            SegMeshPart::Build(i, gen, color, (grid, gdims, lo, stride)) => mesh3d::mesh_from_mask(
                &grid, gdims, lo, stride, geom,
            )
            .map(|(verts, normals, tris)| {
                let m = RoiMesh {
                    roi_index: i,
                    color,
                    external: false,
                    verts,
                    normals,
                    tris,
                };
                (m, gen)
            }),
        }
    }
}

/// What the pane's control strip asks the app to do afterwards, once the
/// borrow of the window it drew has ended.
#[derive(Default)]
struct D3PaneActions {
    prepare: bool,
    play: bool,
    maximize: bool,
    /// The fold was clicked; the app flips the switch every pane shares.
    fold: bool,
    /// Out of the row into a window of its own, or back.
    detach: bool,
}

/// Where a 3D pane's furniture sits.
///
/// A 2D pane names itself in its top-left corner and chains its buttons
/// leftwards from its top-right one, 22 px down, 24 x 20 each with 4 px
/// between them; a 3D pane is the same pane with a different picture in
/// it, so it uses the same numbers. The rectangles are worked out before
/// the scene is drawn, because the scene takes the whole pane and has to
/// be told which parts of it belong to the buttons.
struct D3PaneBar {
    maximize: egui::Rect,
    /// The arrow that folds the rest of the bar away.
    fold: egui::Rect,
    /// Is it folded right now?
    folded: bool,
    /// Out of the row into a window, and back; only while unfolded.
    detach: egui::Rect,
    /// The *Structures* panel over the right-hand edge, or
    /// [`egui::Rect::NOTHING`] while it is off.
    list: egui::Rect,
    reset: egui::Rect,
    hand: egui::Rect,
    zoom_out: egui::Rect,
    zoom_in: egui::Rect,
    play: egui::Rect,
    prepare: egui::Rect,
    /// What is left of the row for the controls only a scene has.
    left: egui::Rect,
    phases: bool,
    /// Is the pointer on any of it?
    over: bool,
}

/// Put a set of structure meshes on a window.
///
/// Always through here. The projected-geometry cache in [`D3Frame`] is
/// keyed on `mesh_gen`, not on the meshes themselves, so a set that arrives
/// without bumping it is drawn with the previous set's triangle order: the
/// scene freezes, and the next thing that does move the camera or the
/// opacity draws the new vertices through the old indices and tears the
/// surface apart.
fn set_meshes(w: &mut D3Window, meshes: Arc<Vec<RoiMesh>>) {
    w.meshes = Some(meshes);
    w.mesh_gen += 1;
}

/// Fit a window's camera sphere to `meshes`: centre on their bounding box,
/// radius to its half-diagonal. False, and the camera untouched, when there
/// is no vertex to fit to.
fn fit_to_meshes(w: &mut D3Window, meshes: &[RoiMesh]) -> bool {
    let (mut mn, mut mx) = ([f32::MAX; 3], [f32::MIN; 3]);
    for m in meshes {
        for v in &m.verts {
            for a in 0..3 {
                mn[a] = mn[a].min(v[a]);
                mx[a] = mx[a].max(v[a]);
            }
        }
    }
    if mn[0] >= mx[0] {
        return false;
    }
    w.center = [
        (mn[0] + mx[0]) * 0.5,
        (mn[1] + mx[1]) * 0.5,
        (mn[2] + mx[2]) * 0.5,
    ];
    w.radius = (0..3)
        .map(|a| (mx[a] - mn[a]) * 0.5)
        .fold(0.0f32, |acc, v| (acc * acc + v * v).sqrt())
        .max(10.0);
    true
}

/// The most sets of meshes a window keeps by key, and the most bytes they
/// may hold together. The count alone let 64 phases of a TotalSegmentator
/// set, or 64 sets of isodose shells, take hundreds of megabytes.
const MESH_CACHE_SETS: usize = 64;
const MESH_CACHE_BYTES: usize = 384 << 20;

/// What a set of meshes holds in memory: vertices, normals and triangles.
fn meshes_bytes(meshes: &[RoiMesh]) -> usize {
    meshes
        .iter()
        .map(|m| {
            m.verts.len() * std::mem::size_of::<[f32; 3]>()
                + m.normals.len() * std::mem::size_of::<[f32; 3]>()
                + m.tris.len() * std::mem::size_of::<[u32; 3]>()
        })
        .sum()
}

/// File `meshes` under `key` in a window's cache while the cache is under
/// both of its caps; a full cache simply stops growing, as it always has.
/// A key already there is replaced whatever the size.
fn cache_meshes(
    cache: &mut std::collections::HashMap<u64, Arc<Vec<RoiMesh>>>,
    key: u64,
    meshes: Arc<Vec<RoiMesh>>,
) {
    if let Some(slot) = cache.get_mut(&key) {
        *slot = meshes;
        return;
    }
    let used: usize = cache.values().map(|m| meshes_bytes(m)).sum();
    if cache.len() < MESH_CACHE_SETS && used + meshes_bytes(&meshes) <= MESH_CACHE_BYTES {
        cache.insert(key, meshes);
    }
}

const RESET_TIP: &str =
    "Reset the camera: default angle, centred on the structures, fit zoom and no offset";

/// What a scene reads from the workspaces besides its own state, gathered
/// once per frame by [`ViewerApp::d3_inputs`].
struct D3Inputs<'a> {
    /// The ROI ticks of the scene's own workspace.
    visible: Vec<bool>,
    /// Name and colour of every ROI of the active set, by index.
    names: Vec<(String, [u8; 3])>,
    /// Visibility and live colour of every segment.
    seg_disp: Vec<(bool, [u8; 3])>,
    /// The ROI ticks of the workspace whose structures can be shown
    /// through the registration ([`ViewerApp::d3_partner`]).
    other_visible: Vec<bool>,
    /// The registration's field when the scene shows its fixed image.
    reg_here: Option<(Arc<crate::registration::dvf::VectorField>, &'static str)>,
    /// The dose the surfaces can be painted with, and its reference.
    dose_here: Option<(&'a crate::rtdose::DoseGrid, f32)>,
}

/// The camera a scene starts with: the default turn, no zoom, no pan,
/// looking at `sphere` (centre and radius, patient mm) until meshes land to
/// fit to.
fn reset_camera(w: &mut D3Window, sphere: ([f32; 3], f32)) {
    w.yaw = 0.7;
    w.pitch = -0.5;
    w.zoom = 1.0;
    w.pan = Vec2::ZERO;
    (w.center, w.radius) = sphere;
    w.refit = true;
}

/// Read back what the opacity sliders write: "80 %", or just "80".
fn percent_parser(text: &str) -> Option<f64> {
    text.trim()
        .trim_end_matches('%')
        .trim()
        .parse::<f64>()
        .ok()
        .map(|v| v / 100.0)
}

/// The *Structures* panel: one opacity slider per structure, multiplying
/// the scene's own.
///
/// The pane lays it over the scene's right-hand edge.
fn structure_list(
    w: &mut D3Window,
    ui: &mut egui::Ui,
    names: &[(String, [u8; 3])],
    visible: &[bool],
) {
    ui.horizontal(|ui| {
        ui.strong("Opacity per structure");
        if ui
            .small_button("All 100 %")
            .on_hover_text("Every structure back to the scene's opacity")
            .clicked()
        {
            w.roi_alpha.clear();
        }
    });
    ui.weak("Times the scene's opacity");
    egui::ScrollArea::vertical().show(ui, |ui| {
        for m in w.meshes.iter().flat_map(|a| a.iter()) {
            let Some((name, color)) = names.get(m.roi_index) else {
                continue;
            };
            let on = visible.get(m.roi_index).copied().unwrap_or(true);
            ui.horizontal(|ui| {
                let (r, _) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                ui.painter().rect_filled(r, 2.0, theme::rgb(*color));
                let mut a = w.roi_alpha.get(&m.roi_index).copied().unwrap_or(1.0);
                let slider = ui.add_enabled(
                    on,
                    egui::Slider::new(&mut a, 0.0..=1.0)
                        .show_value(false)
                        .text(name.as_str()),
                );
                if slider.changed() {
                    if (a - 1.0).abs() < 1e-6 {
                        w.roi_alpha.remove(&m.roi_index);
                    } else {
                        w.roi_alpha.insert(m.roi_index, a);
                    }
                }
                if !on {
                    slider.on_hover_text("Unticked in the list");
                }
            });
        }
    });
}

impl ViewerApp {
    // -- 3D structure windows ----------------------------------------------
    /// The workspace a scene of `slot` shows beside its own: the other image
    /// of the active registration when `slot` holds one of its two, since
    /// that is the transform the other structures are mapped through, and
    /// otherwise the next open workspace. Any of the four workspaces may
    /// be either.
    fn d3_partner(&self, slot: usize) -> Option<usize> {
        if let Some(r) = &self.registration {
            if r.fixed_slot == slot && r.moving_slot != slot {
                return Some(r.moving_slot);
            }
            if r.moving_slot == slot && r.fixed_slot != slot {
                return Some(r.fixed_slot);
            }
        }
        self.other_open(slot)
    }

    /// Everything a scene of `slot` reads from the workspaces this frame.
    fn d3_inputs(&self, slot: usize) -> D3Inputs<'_> {
        let s = &self.slots[slot];
        let other = self.d3_partner(slot);
        D3Inputs {
            visible: s.roi_visible.clone(),
            names: s
                .active_structures()
                .map(|ss| ss.rois.iter().map(|r| (r.name.clone(), r.color)).collect())
                .unwrap_or_default(),
            seg_disp: s.segs().iter().map(|g| (g.visible, g.color)).collect(),
            other_visible: other
                .map(|o| self.slots[o].roi_visible.clone())
                .unwrap_or_default(),
            reg_here: self
                .registration
                .as_ref()
                .filter(|r| r.shows_fixed(slot, &self.slots))
                .map(|r| (r.field.clone(), r.result.method.short())),
            dose_here: s
                .study
                .as_ref()
                .and_then(|st| st.doses.get(s.active_dose))
                .map(|d| (d, s.dose_reference.max(1e-6))),
        }
    }

    /// Identity of the structure set a 3D window would be built from.
    pub(super) fn d3_key(&self, slot: usize) -> u64 {
        self.d3_key_of(slot, self.slots[slot].active_structs)
    }

    /// [`Self::d3_key`] for a structure set that is not the active one:
    /// what the window's key *will* be once the workspace steps to the phase
    /// that set belongs to, which is how the phase meshes are filed.
    pub(super) fn d3_key_of(&self, slot: usize, structs: usize) -> u64 {
        let mut h: u64 = 0x9E3779B97F4A7C15 ^ (slot as u64);
        if let Some(ss) = self.slots[slot]
            .study
            .as_ref()
            .and_then(|st| st.structure_sets.get(structs))
        {
            for b in ss.sop_instance_uid.bytes().chain(ss.file_name.bytes()) {
                h = h.wrapping_mul(31).wrapping_add(b as u64);
            }
            h ^= (structs as u64) << 40;
            h ^= ss.rois.len() as u64;
        }
        // Every contour edit changes the generation, so a moved structure
        // is re-meshed while its window is open.
        h ^= self
            .structures_gen(slot)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15);
        h
    }

    /// Identity of the image a workspace's 3D scene stands on: the series
    /// on display, or - while it is a phase of a 4D group - the group, so
    /// stepping through the phases holds the view still while loading a
    /// different image starts it over.
    pub(super) fn d3_scene_id(&self, slot: usize) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut eat = |s: &str| {
            for b in s.bytes().chain(std::iter::once(0)) {
                h ^= b as u64;
                h = h.wrapping_mul(0x0100_0000_01b3);
            }
        };
        match self.fourd_phases(slot) {
            Some((name, idxs, _)) => {
                eat("4D");
                eat(&name);
                if let Some(st) = self.slots[slot].study.as_ref() {
                    for i in idxs {
                        if let Some(se) = st.series.get(i) {
                            eat(&se.uid);
                        }
                    }
                }
            }
            None => {
                if let Some(uid) = self.slots[slot].displayed_uid() {
                    eat(uid);
                }
            }
        }
        h
    }

    /// Centre and radius of a workspace's image volume, patient mm: where a
    /// scene looks before it has meshes of its own to fit to.
    fn volume_sphere(&self, slot: usize) -> ([f32; 3], f32) {
        self.slots[slot]
            .study
            .as_ref()
            .filter(|st| st.has_volume())
            .map(|st| {
                let v = &st.volume;
                let d = v.dims;
                let a = v.voxel_to_patient(0.0, 0.0, 0.0);
                let b = v.voxel_to_patient(d[0] as f64 - 1.0, d[1] as f64 - 1.0, d[2] as f64 - 1.0);
                let c = (a + b) * 0.5;
                let r = ((b - a).length() * 0.5).max(10.0);
                ([c.x as f32, c.y as f32, c.z as f32], r as f32)
            })
            .unwrap_or(([0.0; 3], 100.0))
    }

    /// Mesh every phase of the workspace's 4D group up front, so that
    /// playing them is a pointer swap per frame rather than a surface-nets
    /// run per frame.
    ///
    /// Each phase brings its own structure set, and the key a set will be
    /// filed under is known before the workspace steps onto it
    /// ([`Self::d3_key_of`]), so the whole group can be built in one pass
    /// and dropped straight into the window's cache.
    fn start_phase_meshes(&mut self, w: &mut D3Window) {
        if w.prep_job.is_some() {
            return;
        }
        let slot = w.slot;
        let Some((_, idxs, _)) = self.fourd_phases(slot) else {
            return;
        };
        let Some(study) = self.slots[slot].study.as_ref() else {
            return;
        };
        // (key, the ROIs of that phase's structure set), for the phases
        // whose meshes are not in the cache already.
        let mut work: Vec<(u64, Vec<(usize, crate::rtstruct::Roi)>)> = Vec::new();
        for idx in &idxs {
            let Some(uid) = study.series.get(*idx).map(|se| se.uid.clone()) else {
                continue;
            };
            let Some(si) = study
                .structure_sets
                .iter()
                .position(|ss| ss.referenced_series_uid == uid)
            else {
                continue;
            };
            let key = self.d3_key_of(slot, si);
            if w.mesh_cache.contains_key(&key) || work.iter().any(|(k, _)| *k == key) {
                continue;
            }
            let rois = study.structure_sets[si]
                .rois
                .iter()
                .cloned()
                .enumerate()
                .collect();
            work.push((key, rois));
        }
        if work.is_empty() {
            return;
        }
        let progress = Arc::new(Progress::default());
        w.mesh_cache_gen = self.structures_gen(w.slot);
        w.prep_job = Some(Job::spawn(progress, move |p| {
            let n = work.len();
            let mut out = Vec::with_capacity(n);
            for (i, (key, rois)) in work.into_iter().enumerate() {
                if p.cancelled() {
                    break;
                }
                p.set_phase(i as f32 / n as f32, 1.0 / n as f32);
                p.set(format!("Meshing phase {}/{n}", i + 1));
                let meshes: Vec<RoiMesh> = rois
                    .par_iter()
                    .filter_map(|(i, roi)| mesh3d::build_roi_mesh(*i, roi))
                    .collect();
                out.push((key, meshes));
            }
            out
        }));
    }

    /// How many of the group's phases already have their meshes.
    fn phase_meshes_ready(&self, w: &D3Window) -> (usize, usize) {
        let Some((_, idxs, _)) = self.fourd_phases(w.slot) else {
            return (0, 0);
        };
        let Some(study) = self.slots[w.slot].study.as_ref() else {
            return (0, 0);
        };
        let mut have = 0;
        for idx in &idxs {
            let Some(uid) = study.series.get(*idx).map(|se| se.uid.as_str()) else {
                continue;
            };
            let Some(si) = study
                .structure_sets
                .iter()
                .position(|ss| ss.referenced_series_uid == uid)
            else {
                continue;
            };
            if w.mesh_cache.contains_key(&self.d3_key_of(w.slot, si)) {
                have += 1;
            }
        }
        (have, idxs.len())
    }

    /// The structure meshes of an open window rebuilt in the background
    /// when its structures changed - one build in flight, the camera kept.
    fn refresh_d3_meshes(&mut self, w: &mut D3Window) {
        let key = self.d3_key(w.slot);
        if w.job.is_some() || w.key == key {
            return;
        }
        // An edit invalidates every key at once, so the phase meshes built
        // before it are not meshes of anything any more. The isodose shells
        // are keyed by the dose and its lines alone, which no structure edit
        // touches, so they stay.
        if w.mesh_cache_gen != self.structures_gen(w.slot) {
            w.mesh_cache.clear();
            w.mesh_cache_gen = self.structures_gen(w.slot);
        }
        w.key = key;
        // Already meshed, most likely by *Prepare phases* or by an earlier
        // pass through the cycle: a phase step is then a pointer swap.
        let slot_state = &self.slots[w.slot];
        let set = slot_state.active_structs;
        let base = Some((set, self.settings_gen));
        let stamps =
            |n: usize| -> Vec<u64> { (0..n).map(|i| slot_state.roi_stamp(set, i)).collect() };
        if let Some(cached) = w.mesh_cache.get(&key).cloned() {
            set_meshes(w, cached);
            w.roi_hashes = slot_state
                .active_structures()
                .map(|ss| ss.rois.iter().map(roi_hash).collect())
                .unwrap_or_default();
            w.roi_stamps = stamps(w.roi_hashes.len());
            w.roi_base = base;
            w.rebuilding = None;
            return;
        }
        let Some(ss) = slot_state.active_structures() else {
            set_meshes(w, Arc::new(Vec::new()));
            w.roi_hashes.clear();
            w.roi_stamps.clear();
            w.roi_base = None;
            return;
        };
        // Only the structures whose contours changed are meshed again: a
        // drag of a chamber volume must not re-mesh the heart every sample.
        let n = ss.rois.len();
        let now = stamps(n);
        let partial = w.meshes.is_some() && w.roi_hashes.len() == n;
        let changed: Vec<usize> = if partial && w.roi_base == base && w.roi_stamps.len() == n {
            // Nothing but contour edits since the last look, and an edit
            // stamps the one structure it wrote: walking the points of
            // every structure to find it cost a hash of the whole set per
            // brush sample. Only the stamped ones are hashed again.
            let changed: Vec<usize> = (0..n).filter(|&i| now[i] != w.roi_stamps[i]).collect();
            for &i in &changed {
                w.roi_hashes[i] = roi_hash(&ss.rois[i]);
            }
            changed
        } else {
            let hashes: Vec<u64> = ss.rois.iter().map(roi_hash).collect();
            let changed = if partial {
                (0..n).filter(|&i| hashes[i] != w.roi_hashes[i]).collect()
            } else {
                (0..n).collect()
            };
            w.roi_hashes = hashes;
            changed
        };
        w.roi_stamps = now;
        w.roi_base = base;
        if changed.is_empty() {
            return;
        }
        let rois: Vec<(usize, crate::rtstruct::Roi)> =
            changed.iter().map(|&i| (i, ss.rois[i].clone())).collect();
        let progress = Arc::new(Progress::default());
        progress.set("starting");
        let p2 = progress.clone();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let n = rois.len();
            let meshes: Vec<RoiMesh> = rois
                .par_iter()
                .enumerate()
                .filter_map(|(k, (i, roi))| {
                    p2.set(format!("Meshing structures {}/{n}", k + 1));
                    mesh3d::build_roi_mesh(*i, roi)
                })
                .collect();
            let _ = tx.send(meshes);
        });
        w.job = Some(Job { progress, rx });
        w.rebuilding = if partial { Some(changed) } else { None };
    }

    /// Make sure a slot has a scene's worth of state - the camera, the
    /// meshes and the jobs that build them - for its pane to draw.
    fn ensure_d3_window(&mut self, slot: usize) {
        if self.d3_windows.iter().any(|w| w.slot == slot) {
            return;
        }
        let key = self.d3_key(slot);
        let ss = self.slots[slot].active_structures().cloned();
        if ss.is_none() && self.slots[slot].segs().is_empty() {
            return;
        }
        // Initial auto-fit from the volume extents; replaced by the meshes'
        // own bounding sphere once structure meshes arrive. Keeps the camera
        // stable for segmentation-only scenes that rebuild while painting.
        let (center, radius) = self.volume_sphere(slot);
        let scene = self.d3_scene_id(slot);
        self.d3_windows.retain(|w| w.slot != slot);
        let job = ss.map(|ss| {
            let progress = Arc::new(Progress::default());
            progress.set("starting");
            let p2 = progress.clone();
            let (tx, rx) = mpsc::channel();
            std::thread::spawn(move || {
                let meshes = mesh3d::build_meshes(&ss, &p2);
                let _ = tx.send(meshes);
            });
            Job { progress, rx }
        });
        let no_structs = job.is_none();
        self.d3_windows.push(D3Window {
            slot,
            yaw: 0.7,
            pitch: -0.5,
            zoom: 1.0,
            pan: Vec2::ZERO,
            pan_mode: false,
            opacity: 1.0,
            meshes: no_structs.then(|| Arc::new(Vec::new())),
            seg_meshes: None,
            seg_mesh_gens: Vec::new(),
            seg_geom: 0,
            seg_job: None,
            seg_built: 0,
            show_other: false,
            other_opacity: 0.55,
            other_meshes: None,
            other_job: None,
            other_key: 0,
            show_field: false,
            show_dose: false,
            mesh_cache: std::collections::HashMap::new(),
            mesh_cache_gen: self.structures_gen(slot),
            iso_cache: std::collections::HashMap::new(),
            prep_job: None,
            frame: D3Frame::default(),
            center,
            radius,
            key,
            scene,
            job,
            refit: true,
            mesh_gen: 0,
            roi_hashes: Vec::new(),
            roi_stamps: Vec::new(),
            roi_base: None,
            rebuilding: None,
            show_iso: false,
            iso_meshes: None,
            iso_job: None,
            iso_built: 0,
            iso_opacity: 0.35,
            show_list: false,
            roi_alpha: std::collections::HashMap::new(),
        });
    }

    // -- 3D scenes (upkeep) --------------------------------------------------
    pub(super) fn d3_windows_ui(&mut self, ctx: &egui::Context) {
        let mut windows = std::mem::take(&mut self.d3_windows);
        for w in &mut windows {
            // The scene is drawn by its pane; this is the upkeep behind it:
            // its meshes built and its jobs polled.
            if !self.row_shows_scene(w.slot) {
                continue;
            }
            // Poll the phase pre-meshing, then the ordinary rebuild: the
            // phases land in the cache first, so a step that happens on the
            // same frame finds them there.
            {
                let mut err = None;
                if let Some(built) = poll_job(&mut w.prep_job, ctx, "Phase meshing", &mut err) {
                    for (key, meshes) in built {
                        cache_meshes(&mut w.mesh_cache, key, Arc::new(meshes));
                    }
                    // The phase on display may be one of the ones just
                    // built: take it from the cache now.
                    w.key = 0;
                }
                self.error = self.error.take().or(err);
            }
            // A different image under the scene: start the camera over, the
            // way opening the window afresh does. A view kept from the last
            // image orbits a point of *its* anatomy, at *its* scale.
            let scene = self.d3_scene_id(w.slot);
            if w.scene != scene {
                w.scene = scene;
                reset_camera(w, self.volume_sphere(w.slot));
            }
            // Poll mesh building, and start one when the structures changed.
            self.refresh_d3_meshes(w);
            {
                let mut err = None;
                if let Some(meshes) = poll_job(&mut w.job, ctx, "Meshing", &mut err) {
                    let meshes = match (w.rebuilding.take(), w.meshes.take()) {
                        (Some(changed), Some(old)) => {
                            let mut all: Vec<RoiMesh> = old
                                .iter()
                                .filter(|m| !changed.contains(&m.roi_index))
                                .cloned()
                                .collect();
                            all.extend(meshes);
                            all.sort_by_key(|m| std::cmp::Reverse(m.tris.len()));
                            all
                        }
                        _ => meshes,
                    };
                    // Fit to the whole scene, not only to the structures a
                    // partial rebuild just replaced.
                    if w.refit && fit_to_meshes(w, &meshes) {
                        w.refit = false;
                    }
                    let meshes = Arc::new(meshes);
                    // File it under the key it was built for, so stepping
                    // back onto this phase later costs nothing.
                    cache_meshes(&mut w.mesh_cache, w.key, meshes.clone());
                    set_meshes(w, meshes);
                }
                self.error = self.error.take().or(err);
            }
            // Nothing to wait for - the meshes came from the cache, or the
            // structures are the same ones on a new image: fit to what is
            // there now.
            if w.refit && w.job.is_none() {
                if let Some(meshes) = w.meshes.clone() {
                    if fit_to_meshes(w, &meshes) {
                        w.refit = false;
                    }
                }
            }

            // Live segmentation meshes: rebuilt in the background whenever a
            // mask changes (one build in flight; a newer state simply spawns
            // the next build once the current one lands), so painting shows
            // up in 3D essentially in real time. Only the segments that
            // changed are meshed again - painting one of forty organs from
            // an auto-segmentation re-meshes that organ - and the rest are
            // taken over from the last build by their generation, which
            // names one state of one mask (`Segmentation::gen`).
            {
                let mut err = None;
                if let Some((m, gens, geom)) =
                    poll_job(&mut w.seg_job, ctx, "Segmentation meshing", &mut err)
                {
                    w.seg_meshes = Some(Arc::new(m));
                    w.seg_mesh_gens = gens;
                    w.seg_geom = geom;
                }
                self.error = self.error.take().or(err);
                let hash = self.seg_mesh_hash(w.slot);
                if w.seg_job.is_none() && w.seg_built != hash {
                    w.seg_built = hash;
                    if let Some(study) = &self.slots[w.slot].study {
                        let geom = GridGeom::of(&study.volume);
                        let geom_key = grid_geom_key(&geom);
                        // The last build's meshes, by generation, when they
                        // stand on the same geometry.
                        let old = w
                            .seg_meshes
                            .clone()
                            .filter(|m| w.seg_geom == geom_key && m.len() == w.seg_mesh_gens.len());
                        let by_gen: std::collections::HashMap<u64, usize> = old
                            .as_ref()
                            .map(|_| {
                                w.seg_mesh_gens
                                    .iter()
                                    .enumerate()
                                    .map(|(pos, g)| (*g, pos))
                                    .collect()
                            })
                            .unwrap_or_default();
                        let parts: Vec<SegMeshPart> = self.slots[w.slot]
                            .segs()
                            .iter()
                            .enumerate()
                            .filter(|(_, s)| s.count > 0)
                            .filter_map(|(i, s)| match by_gen.get(&s.gen) {
                                Some(&pos) => Some(SegMeshPart::Keep(i, s.gen, pos)),
                                None => s
                                    .mesh_grid()
                                    .map(|g| SegMeshPart::Build(i, s.gen, s.color, g)),
                            })
                            .collect();
                        if parts.is_empty() {
                            w.seg_meshes = Some(Arc::new(Vec::new()));
                            w.seg_mesh_gens.clear();
                            w.seg_geom = geom_key;
                        } else {
                            let progress = Arc::new(Progress::default());
                            let (tx, rx) = mpsc::channel();
                            std::thread::spawn(move || {
                                let old: Option<&[RoiMesh]> = old.as_ref().map(|v| v.as_slice());
                                let built: Vec<(RoiMesh, u64)> = parts
                                    .into_par_iter()
                                    .filter_map(|part| part.mesh(old, &geom))
                                    .collect();
                                let (meshes, gens) = built.into_iter().unzip();
                                let _ = tx.send((meshes, gens, geom_key));
                            });
                            w.seg_job = Some(Job { progress, rx });
                        }
                    }
                }
            }

            // The other workspace's structures, mapped through the active
            // registration. Meshing and mapping both happen once, on a
            // worker: a deformable inverse is a fixed-point iteration per
            // vertex, which is not something a paint loop can afford.
            {
                let mut err = None;
                if let Some(m) = poll_job(&mut w.other_job, ctx, "Registered meshing", &mut err) {
                    w.other_meshes = Some(Arc::new(m));
                }
                self.error = self.error.take().or(err);
                let other = self.d3_partner(w.slot);
                let key = match (&self.registration, other) {
                    (Some(_), Some(other)) => {
                        let mut h = self.d3_key(other);
                        h = mix(h, self.reg_gen);
                        h = mix(h, other as u64);
                        mix(h, w.show_other as u64)
                    }
                    _ => 0,
                };
                if w.show_other && w.other_job.is_none() && w.other_key != key {
                    w.other_key = key;
                    w.other_meshes = None;
                    let reg = self.registration.as_ref();
                    let ss = other.and_then(|o| self.slots[o].active_structures().cloned());
                    if let (Some(reg), Some(ss)) = (reg, ss) {
                        // The transform maps fixed → moving. Whichever of the
                        // two this window shows, the *other* workspace has to
                        // come the other way round.
                        let inverse = reg.shows_fixed(w.slot, &self.slots);
                        let t = reg.result.transform.clone();
                        let progress = Arc::new(Progress::default());
                        progress.set("starting");
                        let p2 = progress.clone();
                        let (tx, rx) = mpsc::channel();
                        std::thread::spawn(move || {
                            let mut meshes = mesh3d::build_meshes(&ss, &p2);
                            p2.set("Mapping the surfaces through the registration");
                            let r = t.rigid.matrix();
                            meshes.par_iter_mut().for_each(|m| {
                                for v in &mut m.verts {
                                    let p = Vec3::new(v[0] as f64, v[1] as f64, v[2] as f64);
                                    let q = if inverse { t.unmap(p) } else { t.map(p) };
                                    *v = [q.x as f32, q.y as f32, q.z as f32];
                                }
                                // Normals follow the rigid part only: the
                                // deformable part varies from vertex to
                                // vertex and its effect on shading is far
                                // below what a surface at this scale shows.
                                for n in &mut m.normals {
                                    let (x, y, z) = (n[0] as f64, n[1] as f64, n[2] as f64);
                                    let q = if inverse {
                                        [
                                            r[0] * x + r[3] * y + r[6] * z,
                                            r[1] * x + r[4] * y + r[7] * z,
                                            r[2] * x + r[5] * y + r[8] * z,
                                        ]
                                    } else {
                                        [
                                            r[0] * x + r[1] * y + r[2] * z,
                                            r[3] * x + r[4] * y + r[5] * z,
                                            r[6] * x + r[7] * y + r[8] * z,
                                        ]
                                    };
                                    *n = [q[0] as f32, q[1] as f32, q[2] as f32];
                                }
                            });
                            let _ = tx.send(meshes);
                        });
                        w.other_job = Some(Job { progress, rx });
                    }
                }
                if !w.show_other && w.other_meshes.is_some() {
                    w.other_meshes = None;
                    w.other_key = 0;
                }
            }

            // Isodose surfaces: the active dose thresholded at every isodose
            // line that is on, meshed like a segmentation; rebuilt when the
            // dose, its reference or the lines change.
            {
                let mut err = None;
                if let Some(m) = poll_job(&mut w.iso_job, ctx, "Isodose meshing", &mut err) {
                    let m = Arc::new(m);
                    cache_meshes(&mut w.iso_cache, w.iso_built, m.clone());
                    w.iso_meshes = Some(m);
                }
                self.error = self.error.take().or(err);
                let dose = self.slots[w.slot]
                    .study
                    .as_ref()
                    .and_then(|st| st.doses.get(self.slots[w.slot].active_dose));
                let hash = match (w.show_iso, dose) {
                    (true, Some(d)) => {
                        let mut h = mix(0x1503_D0E5u64, self.slots[w.slot].active_dose as u64);
                        for b in d.sop_instance_uid.bytes() {
                            h = mix(h, b as u64);
                        }
                        // The UID can be empty and the index can shift when a
                        // dose is removed: the grid itself says which it is.
                        h = mix(h, d.data.as_ptr() as u64);
                        h = mix(h, d.data.len() as u64);
                        h = mix(h, d.max_dose.to_bits() as u64);
                        h = mix(h, self.slots[w.slot].dose_reference.to_bits() as u64);
                        for l in self.iso_levels.iter().filter(|l| l.on) {
                            h = mix(h, l.pct.to_bits() as u64);
                            let c = l.color;
                            h = mix(h, (c.r() as u64) << 16 | (c.g() as u64) << 8 | c.b() as u64);
                        }
                        h
                    }
                    _ => 0,
                };
                if w.iso_job.is_none() && w.iso_built != hash {
                    w.iso_built = hash;
                    // The shells of a phase already seen come straight back,
                    // which is what makes the second pass through a cycle
                    // smooth even with the dose following the phase.
                    if let Some(cached) = w.iso_cache.get(&hash) {
                        w.iso_meshes = Some(cached.clone());
                    } else {
                        match (w.show_iso, dose) {
                            (true, Some(d)) => {
                                let dose = d.clone();
                                let reference = self.slots[w.slot].dose_reference.max(1e-6);
                                let levels: Vec<(usize, f32, [u8; 3])> = self
                                    .iso_levels
                                    .iter()
                                    .enumerate()
                                    .filter(|(_, l)| l.on)
                                    .map(|(i, l)| {
                                        (i, l.pct, [l.color.r(), l.color.g(), l.color.b()])
                                    })
                                    .collect();
                                let progress = Arc::new(Progress::default());
                                let (tx, rx) = mpsc::channel();
                                std::thread::spawn(move || {
                                    let geom = dose.mesh_geom();
                                    let meshes: Vec<RoiMesh> = levels
                                        .into_par_iter()
                                        .filter_map(|(i, pct, color)| {
                                            let (grid, gdims, lo, stride) =
                                                dose.iso_mesh_grid(pct / 100.0 * reference)?;
                                            mesh3d::mesh_from_mask(&grid, gdims, lo, stride, &geom)
                                                .map(|(verts, normals, tris)| RoiMesh {
                                                    roi_index: i,
                                                    color,
                                                    external: false,
                                                    verts,
                                                    normals,
                                                    tris,
                                                })
                                        })
                                        .collect();
                                    let _ = tx.send(meshes);
                                });
                                w.iso_job = Some(Job { progress, rx });
                            }
                            _ => w.iso_meshes = None,
                        }
                    }
                }
            }
        }
        // A scene leaves with its pane: put back into the row, it starts
        // over from the structures as they then are.
        windows.retain(|w| self.row_shows_scene(w.slot));
        self.d3_windows = windows;
    }

    /// The surface scene as a pane of a row.
    ///
    /// The scene with the handful of controls that belong next to it on one
    /// strip along the top.
    pub(super) fn scene_pane(&mut self, ui: &mut egui::Ui, slot: usize, rect: egui::Rect) {
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, Color32::BLACK);
        // A row can ask for a scene before there is anything to mesh.
        if !self.slot_has_surfaces(slot) {
            painter.text(
                rect.center(),
                Align2::CENTER_CENTER,
                "No structures or segmentations to render",
                FontId::proportional(13.0),
                ui.visuals().weak_text_color(),
            );
            return;
        }
        self.ensure_d3_window(slot);
        let mut windows = std::mem::take(&mut self.d3_windows);
        let Some(w) = windows.iter_mut().find(|w| w.slot == slot) else {
            self.d3_windows = windows;
            return;
        };
        // What the scene reads from the workspaces this frame.
        let D3Inputs {
            visible,
            names,
            seg_disp,
            other_visible,
            reg_here,
            dose_here,
            ..
        } = self.d3_inputs(slot);
        // Worked out first, drawn last: the scene fills the whole pane, so
        // it has to know which parts of it the buttons own before it reads
        // a drag, and the buttons have to be registered after it so they
        // sit on top and get the hover.
        let bar = Self::d3_pane_bar(
            ui,
            rect,
            self.fourd_phases(slot).is_some(),
            dose_here.is_some(),
            w.show_list && !self.pane_buttons_hidden,
            self.pane_buttons_hidden,
        );
        self.d3_scene(
            w,
            ui,
            rect,
            &visible,
            &names,
            &seg_disp,
            &other_visible,
            &reg_here,
            &dose_here,
            bar.over,
        );
        let acts = self.d3_pane_chrome(w, ui, rect, &bar, dose_here.is_some(), &names, &visible);
        self.d3_windows = windows;
        if acts.prepare {
            self.with_d3_window(slot, |app, w| app.start_phase_meshes(w));
        }
        if acts.play {
            let now = ui.input(|i| i.time);
            self.play.from_pane = Some((slot, PaneKind::Scene3d));
            self.toggle_play(play::PlayTarget::Phases { slot }, now);
        }
        if acts.fold {
            self.pane_buttons_hidden = !self.pane_buttons_hidden;
            self.persist_settings();
        }
        if acts.maximize {
            self.toggle_pane_max(ui, slot, PaneKind::Scene3d);
        }
        if acts.detach {
            self.toggle_detached(slot, PaneKind::Scene3d);
        }
    }

    /// Lay a 3D pane's furniture out, to the millimetre a 2D pane uses.
    fn d3_pane_bar(
        ui: &egui::Ui,
        rect: egui::Rect,
        phases: bool,
        has_dose: bool,
        show_list: bool,
        folded: bool,
    ) -> D3PaneBar {
        let bsize = egui::vec2(24.0, 20.0);
        let psize = egui::vec2(38.0, 20.0);
        let prep_w = 58.0;
        let by = rect.top() + 22.0;
        let cell = |right: f32, size: egui::Vec2| {
            egui::Rect::from_min_size(egui::Pos2::new(right - size.x, by), size)
        };
        let maximize = cell(rect.right() - 4.0, bsize);
        let fold = cell(maximize.left() - 4.0, bsize);
        let detach = cell(fold.left() - 4.0, bsize);
        let reset = cell(detach.left() - 4.0, bsize);
        let hand = cell(reset.left() - 4.0, bsize);
        let zoom_out = cell(hand.left() - 4.0, bsize);
        let zoom_in = cell(zoom_out.left() - 4.0, bsize);
        let play = if phases {
            cell(zoom_in.left() - 4.0, psize)
        } else {
            zoom_in
        };
        let prepare = if phases {
            cell(play.left() - 4.0, egui::vec2(prep_w, 20.0))
        } else {
            play
        };
        // What a 2D pane has no counterpart for goes below the chain, out of
        // its way: three panes to a row leave a bar too narrow to hold both,
        // and the top row is the one that has to match. Two lines of its
        // own, opacity over the toggles, because at three panes to a row
        // neither line would hold the other.
        //
        // The width is a generous estimate of what those controls measure
        // rather than the measurement itself: they are drawn after the
        // scene, and the scene needs to know first which part of itself
        // belongs to them.
        let left_w = if has_dose { 230.0 } else { 170.0 };
        let left = egui::Rect::from_min_max(
            egui::Pos2::new(rect.left() + 6.0, by + 24.0),
            egui::Pos2::new(
                (rect.left() + 6.0 + left_w).min(rect.right() - 6.0),
                by + 68.0,
            ),
        );
        // The structure list, when it is out: down the right-hand edge,
        // below everything else the pane draws, so it covers none of it and
        // takes no width away from it either.
        let list = if show_list {
            let pw = 220.0f32.min(rect.width() * 0.8);
            egui::Rect::from_min_max(
                egui::Pos2::new(rect.right() - pw, left.bottom() + 4.0),
                egui::Pos2::new(rect.right(), rect.bottom()),
            )
        } else {
            egui::Rect::NOTHING
        };
        let over = ui
            .input(|i| i.pointer.interact_pos())
            .map(|p| {
                maximize.contains(p)
                    || fold.contains(p)
                    || (!folded
                        && (detach.contains(p)
                            || reset.contains(p)
                            || hand.contains(p)
                            || zoom_out.contains(p)
                            || zoom_in.contains(p)
                            || (phases && (play.contains(p) || prepare.contains(p)))
                            || left.contains(p)
                            || list.contains(p)))
            })
            .unwrap_or(false);
        D3PaneBar {
            maximize,
            fold,
            folded,
            detach,
            list,
            reset,
            hand,
            zoom_out,
            zoom_in,
            play,
            prepare,
            left,
            phases,
            over,
        }
    }

    /// A 3D pane's furniture: its name in one corner and its buttons in the
    /// other, laid out like every other pane's.
    ///
    /// The right-hand chain is the 2D pane's, glyph for glyph, with the
    /// camera's own three in the middle: zoom in, zoom out and the hand
    /// that turns a left-drag from a rotation into a move. What has no 2D
    /// counterpart - how solid the surfaces are, the structure list, the
    /// dose colouring - sits on the left of the same row, and is clipped
    /// rather than allowed to collide with the chain when a pane is narrow.
    #[allow(clippy::too_many_arguments)]
    fn d3_pane_chrome(
        &self,
        w: &mut D3Window,
        ui: &mut egui::Ui,
        rect: egui::Rect,
        bar: &D3PaneBar,
        has_dose: bool,
        names: &[(String, [u8; 3])],
        visible: &[bool],
    ) -> D3PaneActions {
        let mut acts = D3PaneActions::default();
        let slot = w.slot;
        if self.show_labels {
            let title = if self.comparing() {
                format!("3D · {}", SLOT_NAMES[slot])
            } else {
                "3D".to_string()
            };
            ui.painter_at(rect).text(
                rect.left_top() + Vec2::new(6.0, 4.0),
                Align2::LEFT_TOP,
                title,
                FontId::proportional(14.0),
                Color32::WHITE,
            );
        }

        // Folded, the bar is an arrow and a maximize and nothing else.
        let folded = bar.folded;
        // The scene's own controls, under the button row: opacity on one
        // line, the toggles on the next. Two short lines rather than one
        // long one, because a pane sharing a row with two others has no
        // room for the long one and would simply cut it off.
        if !folded && bar.left.width() > 40.0 {
            let mut row = |top: f32, add: &mut dyn FnMut(&mut egui::Ui)| {
                let strip = egui::Rect::from_min_max(
                    egui::Pos2::new(bar.left.left(), top),
                    egui::Pos2::new(bar.left.right(), top + 20.0),
                );
                let mut child = ui.new_child(
                    egui::UiBuilder::new()
                        .max_rect(strip)
                        .layout(egui::Layout::left_to_right(egui::Align::Center)),
                );
                child.set_clip_rect(strip.intersect(ui.clip_rect()));
                child.spacing_mut().item_spacing.x = 4.0;
                child.spacing_mut().slider_width = 60.0;
                add(&mut child);
            };
            row(bar.left.top(), &mut |ui: &mut egui::Ui| {
                ui.add(
                    egui::Slider::new(&mut w.opacity, 0.2..=1.0)
                        .text("Opacity")
                        .custom_formatter(|v, _| format!("{:.0} %", v * 100.0))
                        .custom_parser(percent_parser)
                        .handle_shape(egui::style::HandleShape::Rect { aspect_ratio: 0.5 }),
                )
                .on_hover_text("How solid the surfaces are");
            });
            let busy = w.job.is_some() || w.prep_job.is_some();
            row(bar.left.top() + 24.0, &mut |ui: &mut egui::Ui| {
                ui.toggle_value(&mut w.show_list, "Structures")
                    .on_hover_text("A panel with the opacity of every structure");
                if has_dose {
                    ui.toggle_value(&mut w.show_dose, "Dose")
                        .on_hover_text("Colour every surface by the dose that lands on it");
                    ui.toggle_value(&mut w.show_iso, "Isodose").on_hover_text(
                        "The active dose as translucent shells at the isodose lines",
                    );
                }
                ui.add_visible(busy, egui::Spinner::new().size(12.0))
                    .on_hover_text("Meshing the structures that changed");
            });
        }

        let (pointer_pos, any_click) =
            ui.input(|i| (i.pointer.interact_pos(), i.pointer.any_click()));
        let hit = |r: egui::Rect, resp: &egui::Response| {
            resp.clicked() || (any_click && pointer_pos.map(|p| r.contains(p)).unwrap_or(false))
        };

        if bar.phases && !folded {
            let playing = self.is_playing(play::PlayTarget::Phases { slot });
            let resp = ui
                .put(
                    bar.play,
                    egui::Button::new(if playing { "⏸4D" } else { "▶4D" }).small(),
                )
                .on_hover_text(if playing {
                    "Stop running through the phases"
                } else {
                    "Play 4D: run this workspace through the phases of its 4D group. The \
                     structures and the isodose shells follow, because each phase brings \
                     its own."
                });
            if hit(bar.play, &resp) {
                acts.play = true;
            }
            let (meshed, n) = self.phase_meshes_ready(w);
            let todo = meshed < n;
            let resp = ui
                .add_enabled_ui(todo, |ui| {
                    ui.put(bar.prepare, egui::Button::new("Prepare").small().truncate())
                })
                .inner
                .on_hover_text("Mesh the structures of every phase now, so playing them is smooth");
            if todo && hit(bar.prepare, &resp) {
                acts.prepare = true;
            }
        }
        if !folded {
            let resp = ui
                .put(bar.zoom_in, egui::Button::new("➕").small())
                .on_hover_text("Zoom in");
            if hit(bar.zoom_in, &resp) {
                w.zoom = (w.zoom * 1.25).clamp(0.1, 40.0);
            }
            let resp = ui
                .put(bar.zoom_out, egui::Button::new("➖").small())
                .on_hover_text("Zoom out");
            if hit(bar.zoom_out, &resp) {
                w.zoom = (w.zoom / 1.25).clamp(0.1, 40.0);
            }
            let resp = ui
                .put(
                    bar.hand,
                    egui::Button::selectable(w.pan_mode, "✋")
                        .frame_when_inactive(true)
                        .small(),
                )
                .on_hover_text(
                    "Move the scene: while this is on, dragging with the left button \
                     slides the scene instead of turning it. Off, a left drag turns it \
                     and a middle drag moves it.",
                );
            if hit(bar.hand, &resp) {
                w.pan_mode = !w.pan_mode;
            }
            let resp = ui
                .put(bar.reset, egui::Button::new("⟲").small())
                .on_hover_text(RESET_TIP);
            if hit(bar.reset, &resp) {
                reset_camera(w, (w.center, w.radius));
            }
            acts.detach = self.detach_button(ui, bar.detach, slot, PaneKind::Scene3d)
                || (any_click && pointer_pos.is_some_and(|p| bar.detach.contains(p)));
        }
        let resp = ui
            .put(
                bar.fold,
                egui::Button::new(if folded { "◀" } else { "▶" }).small(),
            )
            .on_hover_text(if folded {
                "Show the buttons of this bar. One switch for every pane, and it is \
                 remembered between runs."
            } else {
                "Fold these buttons away, leaving the scene. One switch for every pane, \
                 and it is remembered between runs."
            });
        if hit(bar.fold, &resp) {
            acts.fold = true;
        }
        let is_max = self.pane_is_max(ui, slot, PaneKind::Scene3d);
        let resp = ui
            .put(
                bar.maximize,
                egui::Button::new(if is_max { "⊞" } else { "⛶" }).small(),
            )
            .on_hover_text(if is_max {
                "Restore the multi-view layout"
            } else {
                "Maximize this view to the whole window"
            });
        if hit(bar.maximize, &resp) {
            acts.maximize = true;
        }

        // The structure list, over the scene's right-hand edge. Its own
        // backdrop, because it is text and sliders on top of a black scene.
        if !folded && w.show_list && bar.list.is_positive() {
            ui.painter_at(bar.list)
                .rect_filled(bar.list, 0.0, Color32::from_black_alpha(220));
            let mut child = ui.new_child(
                egui::UiBuilder::new()
                    .max_rect(bar.list.shrink(6.0))
                    .layout(egui::Layout::top_down(egui::Align::Min)),
            );
            child.set_clip_rect(bar.list.intersect(ui.clip_rect()));
            structure_list(w, &mut child, names, visible);
        }
        acts
    }

    /// Run `f` with one slot's 3D window taken out of `self`, so it can have
    /// both. The window goes back whatever `f` did.
    fn with_d3_window(&mut self, slot: usize, f: impl FnOnce(&mut Self, &mut D3Window)) {
        let mut windows = std::mem::take(&mut self.d3_windows);
        if let Some(w) = windows.iter_mut().find(|w| w.slot == slot) {
            f(self, w);
        }
        self.d3_windows = windows;
    }

    /// Has this workspace anything the 3D scene could draw?
    pub(super) fn slot_has_surfaces(&self, slot: usize) -> bool {
        self.slots[slot]
            .study
            .as_ref()
            .map(|s| !s.structure_sets.is_empty())
            .unwrap_or(false)
            || !self.slots[slot].segs().is_empty()
    }

    /// Draw one 3D scene into `rect`, and let the pointer turn it.
    ///
    /// Everything here is relative to the rect it is handed, and the only
    /// state it needs beyond the scene's own is what the caller has already
    /// taken out of `self`.
    ///
    /// `names` is the active structure set's (name, colour) per ROI: the
    /// colour a structure is drawn in comes from here, live, not from the
    /// mesh - which is what makes a recolour show at once, with no rebuild
    /// in between.
    #[allow(clippy::too_many_arguments)]
    fn d3_scene(
        &self,
        w: &mut D3Window,
        ui: &mut egui::Ui,
        rect: egui::Rect,
        visible: &[bool],
        names: &[(String, [u8; 3])],
        seg_disp: &[(bool, [u8; 3])],
        other_visible: &[bool],
        reg_here: &Option<(
            std::sync::Arc<crate::registration::dvf::VectorField>,
            &'static str,
        )>,
        dose_here: &Option<(&crate::rtdose::DoseGrid, f32)>,
        over_buttons: bool,
    ) {
        // The scene fills whatever rect it is given; everything below is
        // relative to that rect.
        let resp = ui.interact(
            rect,
            egui::Id::new(("d3_scene", w.slot)),
            Sense::click_and_drag(),
        );
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, Color32::BLACK);

        // Interaction. A pane draws its buttons over the scene, so a drag
        // that starts on one of them is the button's, not the camera's.
        if resp.dragged_by(egui::PointerButton::Primary) && !over_buttons {
            let d = resp.drag_delta();
            if w.pan_mode {
                w.pan += d;
            } else {
                w.yaw += d.x * 0.01;
                w.pitch = (w.pitch + d.y * 0.01).clamp(-1.55, 1.55);
            }
        }
        if resp.dragged_by(egui::PointerButton::Middle) && !over_buttons {
            w.pan += resp.drag_delta();
        }
        if resp.hovered() {
            let (lines, zd) = ui.input(|i| {
                let mut l = 0.0f32;
                for e in &i.events {
                    if let egui::Event::MouseWheel { unit, delta, .. } = e {
                        l += match unit {
                            egui::MouseWheelUnit::Line => delta.y,
                            egui::MouseWheelUnit::Point => delta.y / 40.0,
                            egui::MouseWheelUnit::Page => delta.y * 10.0,
                        };
                    }
                }
                (l, i.zoom_delta())
            });
            w.zoom = (w.zoom * (lines * 0.12).exp() * zd).clamp(0.1, 40.0);
        }

        // Render.
        let Some(meshes) = &w.meshes else { return };
        let seg_meshes = w.seg_meshes.clone();
        let n_seg = seg_meshes.as_ref().map(|m| m.len()).unwrap_or(0);
        let other_meshes = w.other_meshes.clone();
        let n_other = other_meshes.as_ref().map(|m| m.len()).unwrap_or(0);
        let iso_meshes = w.show_iso.then(|| w.iso_meshes.clone()).flatten();
        let n_iso = iso_meshes.as_ref().map(|m| m.len()).unwrap_or(0);
        let iso_alpha = (w.iso_opacity * 255.0) as u8;
        if meshes.is_empty() && n_seg == 0 && n_other == 0 && n_iso == 0 {
            painter.text(
                rect.center(),
                Align2::CENTER_CENTER,
                if w.seg_job.is_some() {
                    "Meshing segmentation"
                } else {
                    "No meshable structures"
                },
                FontId::proportional(14.0),
                Color32::GRAY,
            );
            return;
        }
        let scale = 0.45 * rect.width().min(rect.height()) / w.radius * w.zoom;
        let cx = rect.center().x + w.pan.x;
        let cyc = rect.center().y + w.pan.y;
        let alpha = (w.opacity * 255.0) as u8;
        let other_alpha = (w.other_opacity * 255.0) as u8;

        // What the cached geometry depends on. Orientation and
        // visibility fix the draw order; the rest only moves the
        // already-ordered triangles around on screen.
        // The generation, not the pointer: a rebuilt set can
        // land at the address the previous one had.
        let mut order_key = mix(0x243F6A8885A308D3, w.mesh_gen);
        order_key = mix(order_key, w.yaw.to_bits() as u64);
        order_key = mix(order_key, w.pitch.to_bits() as u64);
        for m in meshes.iter() {
            // A structure faded to nothing leaves the draw
            // order too; the sort has to know.
            let on = visible.get(m.roi_index).copied().unwrap_or(true)
                && w.roi_alpha.get(&m.roi_index).copied().unwrap_or(1.0) > 0.0;
            order_key = mix(order_key, on as u64);
        }
        if let Some(sm) = &seg_meshes {
            // The generation as well as the pointer, for the same
            // reason the structure meshes use one: a rebuilt set
            // can land at the address the previous one had, and
            // stepping through the phases of a group rebuilds
            // these over and over.
            order_key = mix(order_key, Arc::as_ptr(sm) as u64);
            order_key = mix(order_key, w.seg_built);
            for m in sm.iter() {
                let on = seg_disp.get(m.roi_index).map(|d| d.0).unwrap_or(false);
                order_key = mix(order_key, on as u64);
            }
        }
        if let Some(om) = &other_meshes {
            order_key = mix(order_key, Arc::as_ptr(om) as u64);
            order_key = mix(order_key, w.other_key);
            for m in om.iter() {
                let on = other_visible.get(m.roi_index).copied().unwrap_or(true);
                order_key = mix(order_key, on as u64);
            }
        }
        let paint_dose = w.show_dose && dose_here.is_some();
        // What the shaded, turned vertices depend on: everything but where
        // the scene sits on screen and how large. A pan or a zoom keeps
        // them and only places them again (below).
        let mut shade_key = mix(order_key, paint_dose as u64);
        if paint_dose {
            let (_, reference) = dose_here.as_ref().expect("checked");
            shade_key = mix(shade_key, reference.to_bits() as u64);
        }
        // The point the scene turns about: a camera started over for a new
        // image can land on the same scale with a different centre.
        for c in w.center {
            shade_key = mix(shade_key, c.to_bits() as u64);
        }
        shade_key = mix(shade_key, alpha as u64);
        shade_key = mix(shade_key, other_alpha as u64);
        if let Some(im) = &iso_meshes {
            order_key = mix(order_key, Arc::as_ptr(im) as u64);
            order_key = mix(order_key, w.iso_built);
            shade_key = mix(shade_key, iso_alpha as u64);
        }
        // The shells are part of the soup too: the full order key, so new
        // shells are shaded even when nothing else moved.
        shade_key = mix(shade_key, order_key);
        for (i, a) in &w.roi_alpha {
            shade_key = mix(shade_key, (*i as u64) << 32 | a.to_bits() as u64);
        }
        // Structure and segmentation colours are applied live at draw
        // time, so they are part of what the shaded vertices depend on.
        for c in names
            .iter()
            .map(|(_, c)| c)
            .chain(seg_disp.iter().map(|(_, c)| c))
        {
            shade_key = mix(
                shade_key,
                (c[0] as u64) | ((c[1] as u64) << 8) | ((c[2] as u64) << 16),
            );
        }
        let mut vertex_key = mix(shade_key, scale.to_bits() as u64);
        vertex_key = mix(vertex_key, cx.to_bits() as u64);
        vertex_key = mix(vertex_key, cyc.to_bits() as u64);

        if w.frame.shade_key == Some(shade_key) && w.frame.vertex_key != Some(vertex_key) {
            // Only moved or zoomed: the same turned and shaded vertices,
            // placed again by the same expression as when they were made,
            // so the picture is exactly what a full rebuild would draw.
            let f = &mut w.frame;
            let mesh = Arc::make_mut(&mut f.mesh);
            for (v, b) in mesh.vertices.iter_mut().zip(&f.base) {
                v.pos = Pos2::new(cx + b[0] * scale, cyc - b[1] * scale);
            }
            f.vertex_key = Some(vertex_key);
        }
        if w.frame.vertex_key != Some(vertex_key) {
            let (sy, cy) = w.yaw.sin_cos();
            let (sp, cp) = w.pitch.sin_cos();
            let c = w.center;
            // Yaw about patient z, then pitch about the screen x.
            let rot = |p: [f32; 3], centered: bool| -> [f32; 3] {
                let (x, y, z) = if centered {
                    (p[0] - c[0], p[1] - c[1], p[2] - c[2])
                } else {
                    (p[0], p[1], p[2])
                };
                let x1 = cy * x - sy * y;
                let y1 = sy * x + cy * y;
                let y2 = cp * y1 - sp * z;
                let z2 = sp * y1 + cp * z;
                [x1, y2, z2]
            };
            let reorder = w.frame.order_key != Some(order_key);
            let f = &mut w.frame;
            // Buffers are reused across frames; `make_mut` hands
            // back the previous allocation because the painter has
            // already dropped last frame's reference.
            let mesh = Arc::make_mut(&mut f.mesh);
            mesh.vertices.clear();
            f.depth.clear();
            f.base.clear();
            if reorder {
                f.tris.clear();
            }
            // One iterator over everything the scene draws: this
            // workspace's structures, its live segmentations, and
            // the other workspace's structures already mapped
            // through the registration - each with its own
            // opacity, which is the whole point of showing two
            // workspaces at once.
            let entries = meshes
                .iter()
                .map(|m| {
                    let own = w.roi_alpha.get(&m.roi_index).copied().unwrap_or(1.0);
                    (
                        m,
                        visible.get(m.roi_index).copied().unwrap_or(true) && own > 0.0,
                        // The set's colour of today, not the mesh's of
                        // when it was built.
                        names.get(m.roi_index).map(|(_, c)| *c).unwrap_or(m.color),
                        m.external,
                        (alpha as f32 * own).round() as u8,
                    )
                })
                .chain(seg_meshes.iter().flat_map(|a| a.iter()).map(|m| {
                    let (on, c) = seg_disp
                        .get(m.roi_index)
                        .copied()
                        .unwrap_or((false, m.color));
                    (m, on, c, false, alpha)
                }))
                .chain(other_meshes.iter().flat_map(|a| a.iter()).map(|m| {
                    let on = other_visible.get(m.roi_index).copied().unwrap_or(true);
                    (m, on, m.color, m.external, other_alpha)
                }))
                .chain(
                    iso_meshes
                        .iter()
                        .flat_map(|a| a.iter())
                        .map(|m| (m, true, m.color, false, iso_alpha)),
                );
            for (m, on, color, external, entry_alpha) in entries {
                if !on {
                    continue;
                }
                let base = mesh.vertices.len() as u32;
                // External/body contours render translucent so the
                // interior structures remain visible.
                let roi_alpha = if external {
                    (entry_alpha as f32 * 0.22) as u8
                } else {
                    entry_alpha
                };
                for (v, n) in m.verts.iter().zip(m.normals.iter()) {
                    let t = rot(*v, true);
                    let nn = rot(*n, false);
                    // The vertex's own colour: the structure's,
                    // or the dose that lands on it, sampled the
                    // same way the isodose lines are.
                    let color = match &dose_here {
                        Some((dose, reference)) if paint_dose => {
                            let p = Vec3::new(v[0] as f64, v[1] as f64, v[2] as f64);
                            match dose.sample(p) {
                                Some(d) => render::dose_colormap((d / reference).clamp(0.0, 1.0)),
                                // Outside the dose grid: grey, so
                                // "no dose here" cannot be read as
                                // "zero dose here".
                                None => [90, 90, 90],
                            }
                        }
                        _ => color,
                    };
                    // Headlight along the view axis, two-sided.
                    let inten = 0.30 + 0.70 * nn[1].abs();
                    let col = Color32::from_rgba_unmultiplied(
                        (color[0] as f32 * inten) as u8,
                        (color[1] as f32 * inten) as u8,
                        (color[2] as f32 * inten) as u8,
                        roi_alpha,
                    );
                    mesh.vertices.push(egui::epaint::Vertex {
                        pos: Pos2::new(cx + t[0] * scale, cyc - t[2] * scale),
                        uv: egui::epaint::WHITE_UV,
                        color: col,
                    });
                    f.depth.push(t[1]);
                    f.base.push([t[0], t[2]]);
                }
                if reorder {
                    f.tris.extend(
                        m.tris
                            .iter()
                            .map(|t| [base + t[0], base + t[1], base + t[2]]),
                    );
                }
            }

            if reorder {
                // Painter's algorithm: far triangles first (viewer
                // at -y). Packing the depth into the high half of a
                // u64 lets this be a primitive sort rather than a
                // float comparator over a tuple.
                f.order.clear();
                f.order.extend(f.tris.iter().enumerate().map(|(i, t)| {
                    let d =
                        (f.depth[t[0] as usize] + f.depth[t[1] as usize] + f.depth[t[2] as usize])
                            / 3.0;
                    ((!depth_key(d) as u64) << 32) | i as u64
                }));
                f.order.par_sort_unstable();
                mesh.indices.clear();
                mesh.indices.reserve(f.order.len() * 3);
                for &o in &f.order {
                    mesh.indices
                        .extend_from_slice(&f.tris[(o & 0xFFFF_FFFF) as usize]);
                }
                f.order_key = Some(order_key);
            }
            f.vertex_key = Some(vertex_key);
            f.shade_key = Some(shade_key);
        }

        painter.add(egui::Shape::Mesh(w.frame.mesh.clone()));

        // The Structure editor's drawn axis, the same white line
        // as in the views, one slice thick.
        if self.module_on(Module::StructureEditor) && struct_tools::axis_live(&self.tools) {
            if let Some((ax, vol)) = self
                .tools
                .axis
                .filter(|a| a.slot == w.slot)
                .zip(self.slots[w.slot].study.as_ref().map(|st| &st.volume))
            {
                let (sy, cy) = w.yaw.sin_cos();
                let (sp, cp) = w.pitch.sin_cos();
                let c = w.center;
                let project = |p: Vec3| -> Pos2 {
                    let (x, y, z) = (p.x as f32 - c[0], p.y as f32 - c[1], p.z as f32 - c[2]);
                    let x1 = cy * x - sy * y;
                    let y1 = sy * x + cy * y;
                    let z2 = sp * y1 + cp * z;
                    Pos2::new(cx + x1 * scale, cyc - z2 * scale)
                };
                let a = project(vol.voxel_to_patient(ax.a[0], ax.a[1], ax.a[2]));
                let b = project(vol.voxel_to_patient(ax.b[0], ax.b[1], ax.b[2]));
                let thick = (vol.spacing[2] as f32 * scale).max(1.5);
                painter.line_segment([a, b], Stroke::new(thick, Color32::WHITE));
            }
        }

        // The deformation field, drawn over the surfaces.
        if w.show_field {
            if let Some((field, method)) = &reg_here {
                let (sy, cy) = w.yaw.sin_cos();
                let (sp, cp) = w.pitch.sin_cos();
                let c = w.center;
                let project = |p: Vec3| -> Pos2 {
                    let (x, y, z) = (p.x as f32 - c[0], p.y as f32 - c[1], p.z as f32 - c[2]);
                    let x1 = cy * x - sy * y;
                    let y1 = sy * x + cy * y;
                    let z2 = sp * y1 + cp * z;
                    Pos2::new(cx + x1 * scale, cyc - z2 * scale)
                };
                let max = field.max_mag.max(1e-6) as f32;
                let mut drawn = 0usize;
                for (a, b, mag) in dvf::glyphs_3d(field, 1500) {
                    let col = {
                        let t = (mag as f32 / max).clamp(0.0, 1.0);
                        let rgb = render::dose_colormap(t);
                        Color32::from_rgba_unmultiplied(rgb[0], rgb[1], rgb[2], 220)
                    };
                    let (pa, pb) = (project(a), project(b));
                    painter.line_segment([pa, pb], Stroke::new(1.2, col));
                    painter.circle_filled(pb, 1.6, col);
                    drawn += 1;
                }
                painter.text(
                    rect.right_bottom() + Vec2::new(-6.0, -6.0),
                    Align2::RIGHT_BOTTOM,
                    format!("{drawn} field arrows · {method} · max {max:.1} mm"),
                    FontId::proportional(11.0),
                    Color32::GRAY,
                );
            }
        }

        painter.text(
            rect.left_bottom() + Vec2::new(6.0, -6.0),
            Align2::LEFT_BOTTOM,
            format!(
                "{} structure(s){}, {} triangles",
                meshes.len() + n_seg,
                if n_other > 0 {
                    format!(" + {n_other} registered")
                } else {
                    String::new()
                },
                w.frame.tris.len()
            ),
            FontId::proportional(11.0),
            Color32::GRAY,
        );
    }
}

#[cfg(test)]
mod tests {
    /// The frame cache in `D3Frame` is keyed on `mesh_gen`, so a set of
    /// meshes that is installed without bumping it is drawn with the
    /// previous set's triangle order: the scene freezes, and the next thing
    /// that touches the camera or the opacity tears the surface apart. That
    /// is exactly what stepping through 4D phases out of the mesh cache did
    /// once, and it is invisible to the compiler, so the guard is here.
    #[test]
    fn every_mesh_set_goes_through_set_meshes() {
        // Everything above this module: the guard must not count itself.
        let src = include_str!("d3.rs");
        let src = src
            .split_once("\n#[cfg(test)]")
            .expect("this module is the last thing in the file")
            .0;
        let sites: Vec<(usize, &str)> = src
            .lines()
            .enumerate()
            .filter(|(_, l)| l.contains("w.meshes = ") && !l.trim_start().starts_with("//"))
            .map(|(n, l)| (n + 1, l.trim()))
            .collect();
        assert_eq!(
            sites.len(),
            1,
            "only set_meshes may assign w.meshes; found {sites:?}"
        );
        assert!(
            src[..src.find(sites[0].1).expect("the line is in the file")]
                .ends_with("fn set_meshes(w: &mut D3Window, meshes: Arc<Vec<RoiMesh>>) {\n    "),
            "the one assignment is not the one inside set_meshes"
        );
        // And that function does bump the generation.
        let body = src
            .split_once("fn set_meshes(")
            .expect("set_meshes exists")
            .1;
        let body = &body[..body.find("\n}").expect("it ends")];
        assert!(body.contains("mesh_gen += 1"), "{body}");
    }
}
