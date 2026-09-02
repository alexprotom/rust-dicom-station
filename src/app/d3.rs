//! The live 3D structure window: mesh cache, camera, and the painter-order
//! software renderer that draws RTSTRUCT and segmentation surfaces.

use super::*;

impl ViewerApp {
    // -- 3D structure windows ----------------------------------------------
    /// Identity of the structure set a 3D window would be built from.
    pub(super) fn d3_key(&self, slot: usize) -> u64 {
        let mut h: u64 = 0x9E3779B97F4A7C15 ^ (slot as u64);
        if let Some(ss) = self.slots[slot].active_structures() {
            for b in ss.sop_instance_uid.bytes().chain(ss.file_name.bytes()) {
                h = h.wrapping_mul(31).wrapping_add(b as u64);
            }
            h ^= (self.slots[slot].active_structs as u64) << 40;
            h ^= ss.rois.len() as u64;
        }
        h
    }

    pub(super) fn open_d3_window(&mut self, slot: usize) {
        let key = self.d3_key(slot);
        if let Some(w) = self.d3_windows.iter_mut().find(|w| w.slot == slot) {
            if w.key == key {
                w.open = true;
                return;
            }
        }
        let ss = self.slots[slot].active_structures().cloned();
        if ss.is_none() && self.slots[slot].segs().is_empty() {
            return;
        }
        // Initial auto-fit from the volume extents; replaced by the meshes'
        // own bounding sphere once structure meshes arrive. Keeps the camera
        // stable for segmentation-only scenes that rebuild while painting.
        let (center, radius) = self.slots[slot]
            .study
            .as_ref()
            .map(|st| {
                let v = &st.volume;
                let d = v.dims;
                let a = v.voxel_to_patient(0.0, 0.0, 0.0);
                let b = v.voxel_to_patient(d[0] as f64 - 1.0, d[1] as f64 - 1.0, d[2] as f64 - 1.0);
                let c = (a + b) * 0.5;
                let r = ((b - a).length() * 0.5).max(10.0);
                ([c.x as f32, c.y as f32, c.z as f32], r as f32)
            })
            .unwrap_or(([0.0; 3], 100.0));
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
            open: true,
            yaw: 0.7,
            pitch: -0.5,
            zoom: 1.0,
            pan: Vec2::ZERO,
            opacity: 1.0,
            meshes: no_structs.then(|| Arc::new(Vec::new())),
            seg_meshes: None,
            seg_job: None,
            seg_built: 0,
            show_other: false,
            other_opacity: 0.55,
            other_meshes: None,
            other_job: None,
            other_key: 0,
            show_field: false,
            frame: D3Frame::default(),
            center,
            radius,
            key,
            job,
        });
    }

    // -- 3D structure windows (render) --------------------------------------
    pub(super) fn d3_windows_ui(&mut self, ctx: &egui::Context) {
        let mut windows = std::mem::take(&mut self.d3_windows);
        for w in &mut windows {
            if !w.open {
                continue;
            }
            // Poll mesh building.
            {
                let mut err = None;
                if let Some(meshes) = poll_job(&mut w.job, ctx, "Meshing", &mut err) {
                    {
                        // Scene bounding sphere for auto-fit.
                        let (mut mn, mut mx) = ([f32::MAX; 3], [f32::MIN; 3]);
                        for m in &meshes {
                            for v in &m.verts {
                                for a in 0..3 {
                                    mn[a] = mn[a].min(v[a]);
                                    mx[a] = mx[a].max(v[a]);
                                }
                            }
                        }
                        if mn[0] < mx[0] {
                            w.center = [
                                (mn[0] + mx[0]) * 0.5,
                                (mn[1] + mx[1]) * 0.5,
                                (mn[2] + mx[2]) * 0.5,
                            ];
                            w.radius = (0..3)
                                .map(|a| (mx[a] - mn[a]) * 0.5)
                                .fold(0.0f32, |acc, v| (acc * acc + v * v).sqrt())
                                .max(10.0);
                        }
                        w.meshes = Some(Arc::new(meshes));
                    }
                }
                self.error = self.error.take().or(err);
            }

            // Live segmentation meshes: rebuilt in the background whenever a
            // mask changes (one build in flight; a newer state simply spawns
            // the next build once the current one lands), so painting shows
            // up in 3D essentially in real time.
            {
                let mut err = None;
                if let Some(m) = poll_job(&mut w.seg_job, ctx, "Segmentation meshing", &mut err) {
                    w.seg_meshes = Some(Arc::new(m));
                }
                self.error = self.error.take().or(err);
                let hash = self.seg_mesh_hash(w.slot);
                if w.seg_job.is_none() && w.seg_built != hash {
                    w.seg_built = hash;
                    if let Some(study) = &self.slots[w.slot].study {
                        let geom = GridGeom::of(&study.volume);
                        let snaps: Vec<_> = self.slots[w.slot]
                            .segs()
                            .iter()
                            .enumerate()
                            .filter_map(|(i, s)| s.mesh_grid().map(|g| (i, s.color, g)))
                            .collect();
                        if snaps.is_empty() {
                            w.seg_meshes = Some(Arc::new(Vec::new()));
                        } else {
                            let progress = Arc::new(Progress::default());
                            let (tx, rx) = mpsc::channel();
                            std::thread::spawn(move || {
                                let meshes: Vec<RoiMesh> = snaps
                                    .into_par_iter()
                                    .filter_map(|(i, color, (grid, gdims, lo, stride))| {
                                        mesh3d::mesh_from_mask(&grid, gdims, lo, stride, &geom).map(
                                            |(verts, normals, tris)| RoiMesh {
                                                roi_index: i,
                                                color,
                                                external: false,
                                                verts,
                                                normals,
                                                tris,
                                            },
                                        )
                                    })
                                    .collect();
                                let _ = tx.send(meshes);
                            });
                            w.seg_job = Some(Job { progress, rx });
                        }
                    }
                }
            }

            // The other dataset's structures, mapped through the active
            // registration. Meshing and mapping both happen once, on a
            // worker: a deformable inverse is a fixed-point iteration per
            // vertex, which is not something a paint loop can afford.
            {
                let mut err = None;
                if let Some(m) = poll_job(&mut w.other_job, ctx, "Registered meshing", &mut err) {
                    w.other_meshes = Some(Arc::new(m));
                }
                self.error = self.error.take().or(err);
                let other = 1 - w.slot;
                let key = self
                    .registration
                    .as_ref()
                    .map(|_| {
                        let mut h = self.d3_key(other);
                        h = mix(h, self.reg_gen);
                        mix(h, w.show_other as u64)
                    })
                    .unwrap_or(0);
                if w.show_other && w.other_job.is_none() && w.other_key != key {
                    w.other_key = key;
                    w.other_meshes = None;
                    let reg = self.registration.as_ref();
                    let ss = self.slots[other].active_structures().cloned();
                    if let (Some(reg), Some(ss)) = (reg, ss) {
                        // The transform maps fixed → moving. Whichever of the
                        // two this window shows, the *other* dataset has to
                        // come the other way round.
                        let inverse = reg.fixed_slot == w.slot;
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

            let visible: &[bool] = &self.slots[w.slot].roi_visible;
            // Snapshot of segmentation display state (visibility + live color).
            let seg_disp: Vec<(bool, [u8; 3])> = self.slots[w.slot]
                .segs()
                .iter()
                .map(|s| (s.visible, s.color))
                .collect();
            let other_visible: Vec<bool> = self.slots[1 - w.slot].roi_visible.clone();
            let has_other = self.slots[1 - w.slot]
                .study
                .as_ref()
                .map(|s| !s.structure_sets.is_empty())
                .unwrap_or(false);
            let reg_here = self
                .registration
                .as_ref()
                .filter(|r| r.fixed_slot == w.slot)
                .map(|r| (r.field.clone(), r.result.method.short()));
            let registered = self.registration.is_some();
            let title = format!("3D structures - dataset {}", SLOT_NAMES[w.slot]);
            let mut open = w.open;
            detach::tool_window(
                ctx,
                &format!("d3_{}", w.slot),
                title,
                &mut open,
                detach::WinOpts::size(640.0, 700.0).no_scroll(),
                |ui| {
                    if let Some(job) = &w.job {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label(job.progress.get());
                        });
                        return;
                    }
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::Slider::new(&mut w.opacity, 0.2..=1.0)
                                .text(format!("Opacity {}", SLOT_NAMES[w.slot])),
                        );
                        if ui.small_button("⟲ Reset view").clicked() {
                            w.yaw = 0.7;
                            w.pitch = -0.5;
                            w.zoom = 1.0;
                            w.pan = Vec2::ZERO;
                        }
                        ui.weak("drag rotate · wheel zoom · middle-drag pan");
                    });
                    if registered {
                        ui.horizontal(|ui| {
                            ui.add_enabled(
                                has_other,
                                egui::Checkbox::new(
                                    &mut w.show_other,
                                    format!(
                                        "Dataset {} through the registration",
                                        SLOT_NAMES[1 - w.slot]
                                    ),
                                ),
                            )
                            .on_hover_text(
                                "Mesh the other dataset's structures and map every vertex \
                                 through the recovered transform, so both anatomies stand \
                                 in one frame of reference - the only way to see what a \
                                 deformable registration actually did to a surface",
                            );
                            if w.show_other {
                                ui.add(
                                    egui::Slider::new(&mut w.other_opacity, 0.1..=1.0)
                                        .text(SLOT_NAMES[1 - w.slot]),
                                );
                            }
                        });
                        if reg_here.is_some() {
                            ui.checkbox(&mut w.show_field, "Deformation field")
                                .on_hover_text(
                                    "Arrows from where the anatomy is to where the \
                                     registration sends it, coloured by magnitude",
                                );
                        }
                        if w.other_job.is_some() {
                            ui.horizontal(|ui| {
                                ui.spinner();
                                ui.weak("Mapping the other dataset");
                            });
                        }
                    }

                    let avail = ui.available_size();
                    let size = Vec2::new(avail.x.max(240.0), avail.y.max(240.0));
                    let (resp, painter) = ui.allocate_painter(size, Sense::click_and_drag());
                    let rect = resp.rect;
                    painter.rect_filled(rect, 0.0, Color32::BLACK);

                    // Interaction.
                    if resp.dragged_by(egui::PointerButton::Primary) {
                        let d = resp.drag_delta();
                        w.yaw += d.x * 0.01;
                        w.pitch = (w.pitch + d.y * 0.01).clamp(-1.55, 1.55);
                    }
                    if resp.dragged_by(egui::PointerButton::Middle) {
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
                    if meshes.is_empty() && n_seg == 0 && n_other == 0 {
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
                    let mut order_key = mix(0x243F6A8885A308D3, Arc::as_ptr(meshes) as u64);
                    order_key = mix(order_key, w.yaw.to_bits() as u64);
                    order_key = mix(order_key, w.pitch.to_bits() as u64);
                    for m in meshes.iter() {
                        let on = visible.get(m.roi_index).copied().unwrap_or(true);
                        order_key = mix(order_key, on as u64);
                    }
                    if let Some(sm) = &seg_meshes {
                        order_key = mix(order_key, Arc::as_ptr(sm) as u64);
                        for m in sm.iter() {
                            let on = seg_disp.get(m.roi_index).map(|d| d.0).unwrap_or(false);
                            order_key = mix(order_key, on as u64);
                        }
                    }
                    if let Some(om) = &other_meshes {
                        order_key = mix(order_key, Arc::as_ptr(om) as u64);
                        for m in om.iter() {
                            let on = other_visible.get(m.roi_index).copied().unwrap_or(true);
                            order_key = mix(order_key, on as u64);
                        }
                    }
                    let mut vertex_key = mix(order_key, scale.to_bits() as u64);
                    vertex_key = mix(vertex_key, cx.to_bits() as u64);
                    vertex_key = mix(vertex_key, cyc.to_bits() as u64);
                    vertex_key = mix(vertex_key, alpha as u64);
                    vertex_key = mix(vertex_key, other_alpha as u64);
                    // Segmentation colors are applied live at draw time.
                    for (_, c) in &seg_disp {
                        vertex_key = mix(
                            vertex_key,
                            (c[0] as u64) | ((c[1] as u64) << 8) | ((c[2] as u64) << 16),
                        );
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
                        if reorder {
                            f.tris.clear();
                        }
                        // One iterator over everything the scene draws: this
                        // dataset's structures, its live segmentations, and
                        // the other dataset's structures already mapped
                        // through the registration - each with its own
                        // opacity, which is the whole point of showing two
                        // datasets at once.
                        let entries = meshes
                            .iter()
                            .map(|m| {
                                (
                                    m,
                                    visible.get(m.roi_index).copied().unwrap_or(true),
                                    m.color,
                                    m.external,
                                    alpha,
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
                            }));
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
                                let d = (f.depth[t[0] as usize]
                                    + f.depth[t[1] as usize]
                                    + f.depth[t[2] as usize])
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
                    }

                    painter.add(egui::Shape::Mesh(w.frame.mesh.clone()));

                    // The deformation field, drawn over the surfaces.
                    if w.show_field {
                        if let Some((field, method)) = &reg_here {
                            let (sy, cy) = w.yaw.sin_cos();
                            let (sp, cp) = w.pitch.sin_cos();
                            let c = w.center;
                            let project = |p: Vec3| -> Pos2 {
                                let (x, y, z) =
                                    (p.x as f32 - c[0], p.y as f32 - c[1], p.z as f32 - c[2]);
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
                },
            );
            w.open = open;
        }
        windows.retain(|w| w.open);
        self.d3_windows = windows;
    }
}
