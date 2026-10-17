//! The planar panes: a workspace's planar images - the DX / CR / RTIMAGE it
//! came with and the DRRs made from its CT - shown in its row, beside the
//! MPR planes, instead of in windows of their own.
//!
//! A row carries up to [`settings::PLANAR_PANES`] of them,
//! [`PaneKind::Planar`] `0` and `1`, chosen under *Settings ▸ View layout*
//! like any pane and off in the standard row. Two, because the comparison a
//! radiograph is taken for is with its DRR, and the two belong side by side
//! at the same scale.
//!
//! A pane shows a **stack** of images ([`PlanarView`]): one, or several to
//! compare. The scrubber along its bottom edge moves through the stack the
//! way a plane's moves through its slices, and the two images either side of
//! it are compared as the pane's [`Compare`] mode says: faded into each
//! other, in a checkerboard, as their difference, or the upper one's edges
//! over the lower. Every image is drawn at the pane's zoom (screen pixels per
//! millimetre of its own pixel spacing) about the same centre, so two
//! renderings of one geometry lie exactly on each other, and each keeps its
//! own window - a kV image and a line integral share no modality window.
//!
//! The data tree drives them: a click on an image shows it alone in a planar
//! pane, adding the pane to the row when it is not there; Ctrl-click adds it
//! to the stack of the pane given an image last, or takes it out again; the
//! right-click menu names the pane. A rendered DRR lands the same way
//! ([`ViewerApp::show_planar`]). A pane whose stack is emptied leaves the row.
//!
//! A planar pane in the layout is drawn only when it has something of its
//! own to show (`planar_drawn`): a stack, or - for the first planar pane of
//! the row while no pane has one - the workspace's first image, so a study
//! that arrives with its RT images shows one at once. A workspace with no
//! planar image draws none at all (`row_panes`), so the layout can keep them
//! without a blank pane for every CT.

use super::*;

/// Window width bounds for a planar image, from its value range: a DRR's
/// line integrals span a few units, a radiograph's raw values tens of
/// thousands, and a fixed floor of one would make the first unusable.
fn width_bounds(img: &crate::extras::PlanarImage) -> (f32, f32) {
    let range = (img.max_value - img.min_value).abs().max(1e-6);
    ((range * 1e-4).max(1e-6), range * 20.0 + 1.0)
}

/// The window an image comes up with: the one it carries.
fn default_window(img: &crate::extras::PlanarImage) -> (f32, f32) {
    let (lo, hi) = width_bounds(img);
    (img.window.0, img.window.1.clamp(lo, hi))
}

fn planar_key(img: &crate::extras::PlanarImage) -> PlanarKey {
    (img.data.as_ptr() as usize, img.cols, img.rows)
}

/// An image's values through a window, as grey levels 0 to 255.
fn windowed(img: &crate::extras::PlanarImage, wl: (f32, f32)) -> Vec<u8> {
    let lo = wl.0 - wl.1 * 0.5;
    let scale = 255.0 / wl.1.max(1e-12);
    img.data
        .iter()
        .map(|&x| ((x - lo) * scale).clamp(0.0, 255.0) as u8)
        .collect()
}

/// The pair of stack positions the scrubber compares, and how far it is
/// from the first towards the second. At the top of the stack the pair is
/// the last two, so a mode that needs two images still has them.
fn pair_at(blend: f32, n: usize) -> (usize, usize, f32) {
    if n < 2 {
        return (0, 0, 0.0);
    }
    let b = blend.clamp(0.0, (n - 1) as f32);
    let lower = (b.floor() as usize).min(n - 2);
    (lower, lower + 1, b - lower as f32)
}

impl Compare {
    const ALL: [Compare; 5] = [
        Compare::Fade,
        Compare::Checker,
        Compare::Difference,
        Compare::Edges,
        Compare::Flicker,
    ];

    fn glyph(self) -> &'static str {
        match self {
            Compare::Fade => "◐",
            Compare::Checker => "▣",
            Compare::Difference => "⊖",
            Compare::Edges => "⬚",
            Compare::Flicker => "⇄",
        }
    }

    fn name(self) -> &'static str {
        match self {
            Compare::Fade => "fade",
            Compare::Checker => "checkerboard",
            Compare::Difference => "difference",
            Compare::Edges => "edges",
            Compare::Flicker => "flicker",
        }
    }

    fn next(self) -> Compare {
        let i = Compare::ALL.iter().position(|c| *c == self).unwrap_or(0);
        Compare::ALL[(i + 1) % Compare::ALL.len()]
    }
}

/// The signed difference of two windowed images, mid-grey where they agree.
fn difference_image(a: &[u8], b: &[u8], size: [usize; 2]) -> ColorImage {
    let pixels = a
        .iter()
        .zip(b)
        .map(|(x, y)| Color32::from_gray((128 + (*x as i32 - *y as i32) / 2).clamp(0, 255) as u8))
        .collect();
    ColorImage::new(size, pixels)
}

/// The edges of a windowed image (Sobel magnitude above its mean plus two
/// standard deviations), opaque orange on transparent.
fn edge_image(g: &[u8], size: [usize; 2]) -> ColorImage {
    let [w, h] = size;
    let mut mag = vec![0.0f32; w * h];
    if w >= 3 && h >= 3 {
        let at = |i: usize, j: usize| g[j * w + i] as f32;
        for j in 1..h - 1 {
            for i in 1..w - 1 {
                let gx = at(i + 1, j - 1) + 2.0 * at(i + 1, j) + at(i + 1, j + 1)
                    - at(i - 1, j - 1)
                    - 2.0 * at(i - 1, j)
                    - at(i - 1, j + 1);
                let gy = at(i - 1, j + 1) + 2.0 * at(i, j + 1) + at(i + 1, j + 1)
                    - at(i - 1, j - 1)
                    - 2.0 * at(i, j - 1)
                    - at(i + 1, j - 1);
                mag[j * w + i] = (gx * gx + gy * gy).sqrt();
            }
        }
    }
    // A sum that decides a threshold: in order, as everywhere else.
    let n = mag.len().max(1) as f64;
    let mean = mag.iter().map(|v| *v as f64).sum::<f64>() / n;
    let var = mag.iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>() / n;
    let cut = (mean + 2.0 * var.sqrt()) as f32;
    let edge = Color32::from_rgba_unmultiplied(255, 170, 0, 230);
    let pixels = mag
        .iter()
        .map(|m| {
            if *m > cut && *m > 0.0 {
                edge
            } else {
                Color32::TRANSPARENT
            }
        })
        .collect();
    ColorImage::new(size, pixels)
}

impl ViewerApp {
    /// How many planar images `slot` holds.
    pub(super) fn planar_count(&self, slot: usize) -> usize {
        self.slots[slot.min(MAX_WORKSPACES - 1)]
            .study
            .as_ref()
            .map(|st| st.planar_images.len())
            .unwrap_or(0)
    }

    /// Is planar pane `k` of `slot` drawn: in the layout, with an image to
    /// show? See the module notes for the rule.
    pub(super) fn planar_drawn(&self, slot: usize, k: usize) -> bool {
        let slot = slot.min(MAX_WORKSPACES - 1);
        let n = self.planar_count(slot);
        let layout = self.row_layout(slot);
        let in_row = |k: usize| layout.contains(&PaneKind::Planar(k as u8));
        if n == 0 || k >= settings::PLANAR_PANES || !in_row(k) {
            return false;
        }
        let s = &self.slots[slot];
        let has = |k: usize| s.planar[k].layers.iter().any(|l| l.image < n);
        if has(k) {
            return true;
        }
        // The stand-in: the first planar pane of the row, while none of them
        // has a stack of its own.
        let first = (0..settings::PLANAR_PANES).find(|j| in_row(*j));
        first == Some(k) && !(0..settings::PLANAR_PANES).any(|j| in_row(j) && has(j))
    }

    /// The images pane `k` of `slot` shows, bottom first: its stack, or the
    /// stand-in's first image.
    fn pane_images(&self, slot: usize, k: usize) -> Vec<usize> {
        if !self.planar_drawn(slot, k) {
            return Vec::new();
        }
        let n = self.planar_count(slot);
        let own: Vec<usize> = self.slots[slot].planar[k]
            .layers
            .iter()
            .map(|l| l.image)
            .filter(|i| *i < n)
            .collect();
        if own.is_empty() {
            vec![0]
        } else {
            own
        }
    }

    /// The planar panes of `slot`'s row that are on screen with image `idx`
    /// in their stack.
    pub(super) fn planar_panes_showing(&self, slot: usize, idx: usize) -> Vec<usize> {
        (0..settings::PLANAR_PANES)
            .filter(|k| self.pane_images(slot, *k).contains(&idx))
            .collect()
    }

    /// The pane a plain click gives an image to, when none was named: one
    /// already showing it, else a planar pane not on screen, else the one
    /// that was given an image longest ago.
    fn planar_target(&self, slot: usize, idx: usize) -> usize {
        self.planar_panes_showing(slot, idx)
            .first()
            .copied()
            .or_else(|| (0..settings::PLANAR_PANES).find(|k| !self.planar_drawn(slot, *k)))
            .unwrap_or((self.slots[slot].planar_last + 1) % settings::PLANAR_PANES)
            .min(settings::PLANAR_PANES - 1)
    }

    /// Show planar image `idx` of `slot` alone in a planar pane - `pane`, or
    /// the one [`Self::planar_target`] picks - putting the pane into the row
    /// when it is not there yet.
    pub(super) fn show_planar(&mut self, slot: usize, idx: usize, pane: Option<usize>) {
        if idx >= self.planar_count(slot) {
            return;
        }
        let k = pane
            .unwrap_or_else(|| self.planar_target(slot, idx))
            .min(settings::PLANAR_PANES - 1);
        let v = &mut self.slots[slot].planar[k];
        let kept = v.layers.drain(..).find(|l| l.image == idx);
        v.layers = vec![kept.unwrap_or_else(|| PlanarLayer::new(idx))];
        v.blend = 0.0;
        v.scroll_accum = 0.0;
        self.slots[slot].planar_last = k;
        self.add_to_row(slot, PaneKind::Planar(k as u8));
    }

    /// Ctrl-click in the tree: put image `idx` on top of a pane's stack to
    /// compare it with what is there, or take it out of a stack it was put
    /// on. Without a pane named, a stack holding it among others gives it
    /// up; otherwise it goes on the pane given an image last that does not
    /// hold it already (so with one rendering in each pane, the other one
    /// comes over to be compared). With no planar pane on screen this is
    /// [`Self::show_planar`].
    pub(super) fn stack_planar(&mut self, slot: usize, idx: usize, pane: Option<usize>) {
        if idx >= self.planar_count(slot) {
            return;
        }
        let last = self.slots[slot].planar_last.min(settings::PLANAR_PANES - 1);
        let order: Vec<usize> = match pane {
            Some(k) => vec![k.min(settings::PLANAR_PANES - 1)],
            None => std::iter::once(last)
                .chain((0..settings::PLANAR_PANES).filter(|k| *k != last))
                .filter(|k| self.planar_drawn(slot, *k))
                .collect(),
        };
        if order.is_empty() {
            self.show_planar(slot, idx, None);
            return;
        }
        let stacked = |k: usize| {
            let imgs = self.pane_images(slot, k);
            imgs.len() >= 2 && imgs.contains(&idx)
        };
        if let Some(k) = order.iter().copied().find(|k| stacked(*k)) {
            self.take_planar_image(slot, k, idx);
            return;
        }
        let Some(k) = order
            .iter()
            .copied()
            .find(|k| !self.pane_images(slot, *k).contains(&idx))
        else {
            return;
        };
        if !self.planar_drawn(slot, k) {
            self.show_planar(slot, idx, Some(k));
            return;
        }
        // The stand-in's image becomes a real one before anything is put on
        // top of it.
        let shown = self.pane_images(slot, k);
        let v = &mut self.slots[slot].planar[k];
        if v.layers.is_empty() {
            v.layers = shown.iter().map(|i| PlanarLayer::new(*i)).collect();
        }
        v.layers.push(PlanarLayer::new(idx));
        // The scrubber goes half way: the two are compared from the start.
        v.blend = v.layers.len() as f32 - 1.5;
        self.slots[slot].planar_last = k;
    }

    /// Take position `at` out of pane `k`'s stack; a pane left with nothing
    /// leaves the row.
    fn take_planar(&mut self, slot: usize, k: usize, at: usize) {
        let v = &mut self.slots[slot].planar[k];
        if at < v.layers.len() {
            v.layers.remove(at);
        }
        v.blend = v.blend.min(v.layers.len().saturating_sub(1) as f32);
        if v.layers.is_empty() {
            self.remove_from_row(slot, PaneKind::Planar(k as u8));
        }
    }

    /// Take image `idx` out of pane `k`.
    pub(super) fn take_planar_image(&mut self, slot: usize, k: usize, idx: usize) {
        let shown = self.pane_images(slot, k);
        let v = &mut self.slots[slot].planar[k];
        if v.layers.is_empty() {
            v.layers = shown.iter().map(|i| PlanarLayer::new(*i)).collect();
        }
        if let Some(at) = v.layers.iter().position(|l| l.image == idx) {
            self.take_planar(slot, k, at);
        }
    }

    /// Planar image `idx` of `slot` is gone: it leaves every stack it was
    /// in, every image past it moves back by one, and a pane left with
    /// nothing leaves the row.
    pub(super) fn planar_removed(&mut self, slot: usize, idx: usize) {
        let layout = self.row_layout(slot);
        for k in 0..settings::PLANAR_PANES {
            let v = &mut self.slots[slot].planar[k];
            let had = !v.layers.is_empty();
            v.layers.retain(|l| l.image != idx);
            for l in &mut v.layers {
                if l.image > idx {
                    l.image -= 1;
                }
            }
            v.blend = v.blend.min(v.layers.len().saturating_sub(1) as f32);
            let emptied = had && v.layers.is_empty();
            if emptied && layout.contains(&PaneKind::Planar(k as u8)) {
                self.remove_from_row(slot, PaneKind::Planar(k as u8));
            }
        }
        // The setups' radiographs and the DRRs drawn for one.
        if let Some(study) = self.slots[slot].study.as_mut() {
            for s in &mut study.setups {
                for i in &mut s.images {
                    if *i == idx {
                        *i = usize::MAX;
                    } else if *i > idx && *i != usize::MAX {
                        *i -= 1;
                    }
                }
            }
        }
        if let Some(st) = self.r2d.as_mut().filter(|st| st.slot == slot) {
            st.drrs.retain(|d| *d != idx);
            for d in &mut st.drrs {
                if *d > idx {
                    *d -= 1;
                }
            }
        }
        if self.planar_count(slot) == 0 {
            if let Some((s, PaneKind::Planar(_))) = self.maximized {
                if s == slot {
                    self.maximized = None;
                }
            }
        }
    }

    /// Put `kind` into `slot`'s row at the right, when it is not there.
    /// The layout is the user's and is remembered, as a tick under
    /// *Settings ▸ View layout* would be.
    pub(super) fn add_to_row(&mut self, slot: usize, kind: PaneKind) {
        let row = self.row_layout(slot);
        if row.contains(&kind) {
            return;
        }
        let panes = &mut self.view_rows[slot];
        if panes.is_empty() {
            *panes = row;
        }
        panes.push(kind);
        self.persist_settings();
    }

    /// Take `kind` out of `slot`'s row. The last pane of a row stays: a row
    /// has to show something. A planar pane forgets its stack.
    pub(super) fn remove_from_row(&mut self, slot: usize, kind: PaneKind) {
        let row = self.row_layout(slot);
        if let PaneKind::Planar(k) = kind {
            if let Some(v) = self.slots[slot].planar.get_mut(k as usize) {
                v.layers.clear();
                v.blend = 0.0;
            }
        }
        if !row.contains(&kind) || row.len() == 1 {
            return;
        }
        let panes = &mut self.view_rows[slot];
        if panes.is_empty() {
            *panes = row;
        }
        panes.retain(|k| *k != kind);
        if self.maximized == Some((slot, kind)) {
            self.maximized = None;
        }
        self.persist_settings();
    }

    /// Rebuild the textures of pane `k`'s stack, and of the pair it
    /// compares, when an image or a window changed.
    fn refresh_planar_tex(&mut self, ctx: &egui::Context, slot: usize, k: usize) {
        let Some(study) = &self.slots[slot].study else {
            return;
        };
        let v = &mut self.slots[slot].planar[k];
        for (li, layer) in v.layers.iter_mut().enumerate() {
            let Some(img) = study.planar_images.get(layer.image) else {
                continue;
            };
            let key = planar_key(img);
            if layer.wl_for != Some(key) {
                layer.wl = default_window(img);
                layer.wl_for = Some(key);
            }
            let tex_key = (key, layer.wl.0.to_bits(), layer.wl.1.to_bits());
            if layer.tex.is_some() && layer.tex_key == Some(tex_key) {
                continue;
            }
            let pixels: Vec<Color32> = windowed(img, layer.wl)
                .into_iter()
                .map(Color32::from_gray)
                .collect();
            let ci = ColorImage::new([img.cols, img.rows], pixels);
            layer.tex_key = Some(tex_key);
            match &mut layer.tex {
                Some(t) => t.set(ci, TextureOptions::LINEAR),
                None => {
                    layer.tex = Some(ctx.load_texture(
                        format!("planar_pane_{slot}_{k}_{li}"),
                        ci,
                        TextureOptions::LINEAR,
                    ))
                }
            }
        }
        // The comparison image of the pair, for the two modes that need one.
        if v.layers.len() < 2 || !matches!(v.compare, Compare::Difference | Compare::Edges) {
            return;
        }
        let (lo, hi, _) = pair_at(v.blend, v.layers.len());
        let (Some(a), Some(b)) = (
            study.planar_images.get(v.layers[lo].image),
            study.planar_images.get(v.layers[hi].image),
        ) else {
            return;
        };
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for x in [
            v.compare as u64,
            planar_key(a).0 as u64,
            planar_key(b).0 as u64,
            v.layers[lo].wl.0.to_bits() as u64,
            v.layers[lo].wl.1.to_bits() as u64,
            v.layers[hi].wl.0.to_bits() as u64,
            v.layers[hi].wl.1.to_bits() as u64,
        ] {
            h = (h ^ x).wrapping_mul(0x100_0000_01b3);
        }
        if v.cmp_tex.is_some() && v.cmp_key == Some(h) {
            return;
        }
        let ci = match v.compare {
            Compare::Difference => {
                if (a.cols, a.rows) != (b.cols, b.rows) {
                    v.cmp_tex = None;
                    v.cmp_key = Some(h);
                    return;
                }
                difference_image(
                    &windowed(a, v.layers[lo].wl),
                    &windowed(b, v.layers[hi].wl),
                    [a.cols, a.rows],
                )
            }
            _ => edge_image(&windowed(b, v.layers[hi].wl), [b.cols, b.rows]),
        };
        v.cmp_key = Some(h);
        match &mut v.cmp_tex {
            Some(t) => t.set(ci, TextureOptions::LINEAR),
            None => {
                v.cmp_tex = Some(ctx.load_texture(
                    format!("planar_cmp_{slot}_{k}"),
                    ci,
                    TextureOptions::LINEAR,
                ))
            }
        }
    }

    /// One planar pane filling `rect`.
    pub(super) fn planar_pane(&mut self, ui: &mut egui::Ui, slot: usize, k: usize, rect: Rect) {
        let ctx = ui.ctx().clone();
        let kind = PaneKind::Planar(k as u8);
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, Color32::BLACK);
        let n = self.planar_count(slot);
        if n == 0 {
            // Reached only through a maximized pane whose images went away
            // this pass; the row itself never draws an empty planar pane.
            painter.text(
                rect.center(),
                Align2::CENTER_CENTER,
                "No planar images in this workspace",
                FontId::proportional(13.0),
                ui.visuals().weak_text_color(),
            );
            return;
        }
        // The stand-in becomes a stack of one, and images that no longer
        // exist leave the stack.
        let shown = self.pane_images(slot, k);
        {
            let v = &mut self.slots[slot].planar[k];
            v.layers.retain(|l| l.image < n);
            if v.layers.is_empty() {
                v.layers = shown.iter().map(|i| PlanarLayer::new(*i)).collect();
            }
            if v.layers.is_empty() {
                v.layers.push(PlanarLayer::new(0));
            }
            v.blend = v.blend.clamp(0.0, (v.layers.len() - 1) as f32);
        }
        self.refresh_planar_tex(&ctx, slot, k);

        let Some(study) = &self.slots[slot].study else {
            return;
        };
        let v = &self.slots[slot].planar[k];
        let nl = v.layers.len();
        let (lo, hi, frac) = pair_at(v.blend, nl);
        // The image the window, the readout and the title speak of: the one
        // the scrubber is nearer.
        let near = if nl < 2 {
            0
        } else if frac >= 0.5 {
            hi
        } else {
            lo
        };
        let near_img = &study.planar_images[v.layers[near].image];
        let geom = |img: &crate::extras::PlanarImage| {
            let sx = img.spacing[0].max(1e-6) as f32;
            let sy = img.spacing[1].max(1e-6) as f32;
            (sx, sy, img.cols as f32 * sx, img.rows as f32 * sy)
        };
        // The fit is the largest image of the stack's, so every one of them
        // is on screen at the start.
        let fit_zoom = v
            .layers
            .iter()
            .filter_map(|l| study.planar_images.get(l.image))
            .map(|img| {
                let (_, _, w, h) = geom(img);
                (rect.width() / w).min(rect.height() / h)
            })
            .fold(f32::INFINITY, f32::min)
            .max(1e-3)
            * 0.97;
        let zoom = if v.zoom > 0.0 { v.zoom } else { fit_zoom };
        let center = rect.center() + v.pan * zoom;
        let rect_of = |img: &crate::extras::PlanarImage| {
            let (_, _, w, h) = geom(img);
            Rect::from_center_size(center, Vec2::new(w * zoom, h * zoom))
        };
        let full_uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));

        // ---- the stack: the lower image, then the upper as the mode says ----
        let lower = &v.layers[if nl < 2 { 0 } else { lo }];
        let lower_img = &study.planar_images[lower.image];
        if let Some(tex) = &lower.tex {
            painter.image(tex.id(), rect_of(lower_img), full_uv, Color32::WHITE);
        }
        let mut mode_note = None;
        if nl >= 2 {
            let upper = &v.layers[hi];
            let upper_img = &study.planar_images[upper.image];
            let up_rect = rect_of(upper_img);
            match v.compare {
                Compare::Fade => {
                    if let Some(tex) = &upper.tex {
                        let a = (frac * 255.0).round() as u8;
                        painter.image(tex.id(), up_rect, full_uv, Color32::from_white_alpha(a));
                    }
                }
                Compare::Checker => {
                    if let Some(tex) = &upper.tex {
                        let tile = 48.0f32;
                        let vis = up_rect.intersect(rect);
                        if vis.is_positive() {
                            let i0 = ((vis.left() - up_rect.left()) / tile).floor() as i64;
                            let i1 = ((vis.right() - up_rect.left()) / tile).ceil() as i64;
                            let j0 = ((vis.top() - up_rect.top()) / tile).floor() as i64;
                            let j1 = ((vis.bottom() - up_rect.top()) / tile).ceil() as i64;
                            for j in j0..j1 {
                                for i in i0..i1 {
                                    if (i + j).rem_euclid(2) == 0 {
                                        continue;
                                    }
                                    let cell = Rect::from_min_size(
                                        up_rect.min + Vec2::new(i as f32 * tile, j as f32 * tile),
                                        Vec2::splat(tile),
                                    )
                                    .intersect(up_rect);
                                    if !cell.is_positive() {
                                        continue;
                                    }
                                    let uv = Rect::from_min_max(
                                        Pos2::new(
                                            (cell.left() - up_rect.left()) / up_rect.width(),
                                            (cell.top() - up_rect.top()) / up_rect.height(),
                                        ),
                                        Pos2::new(
                                            (cell.right() - up_rect.left()) / up_rect.width(),
                                            (cell.bottom() - up_rect.top()) / up_rect.height(),
                                        ),
                                    );
                                    painter.image(tex.id(), cell, uv, Color32::WHITE);
                                }
                            }
                        }
                    }
                }
                Compare::Difference => match &v.cmp_tex {
                    Some(tex) if v.cmp_key.is_some() => {
                        painter.image(tex.id(), up_rect, full_uv, Color32::WHITE);
                    }
                    _ => mode_note = Some("different rasters: no difference"),
                },
                Compare::Edges => {
                    if let Some(tex) = &v.cmp_tex {
                        painter.image(tex.id(), up_rect, full_uv, Color32::WHITE);
                    }
                }
                Compare::Flicker => {
                    let t = ui.input(|i| i.time);
                    if (t * 2.0).floor() as i64 % 2 == 1 {
                        if let Some(tex) = &upper.tex {
                            painter.image(tex.id(), up_rect, full_uv, Color32::WHITE);
                        }
                    }
                    ui.ctx()
                        .request_repaint_after(std::time::Duration::from_millis(100));
                }
            }
        }

        // A setup's isocentre, markers and outlines over its radiographs.
        self.reg2d3d_overlay(&painter, slot, v.layers[near].image, rect_of(near_img));
        // The regions and markers placed by hand on the radiograph of the
        // stack.
        let layer_images: Vec<usize> = v.layers.iter().map(|l| l.image).collect();
        let marked = self
            .reg2d3d_pane_image(slot, &layer_images)
            .and_then(|i| study.planar_images.get(i).map(|img| (i, rect_of(img))));
        if let Some((i, r)) = marked {
            self.reg2d3d_marks(&painter, slot, i, r);
        }

        let (sx, sy, _, _) = geom(near_img);
        let near_rect = rect_of(near_img);
        let wl = v.layers[near].wl;
        let pan = v.pan;
        let scroll_accum = v.scroll_accum;
        let blend = v.blend;
        let compare = v.compare;
        let bounds = width_bounds(near_img);
        let default_wl = default_window(near_img);
        let label_of = |l: &PlanarLayer| {
            study
                .planar_images
                .get(l.image)
                .map(|p| p.label.clone())
                .unwrap_or_default()
        };
        let title = label_of(&v.layers[if nl < 2 { 0 } else { lo }]);
        let subtitle = (nl >= 2).then(|| {
            let what = match compare {
                Compare::Fade => format!("{:.0} %", frac * 100.0),
                c => c.name().to_string(),
            };
            format!("{} {} · {what}", compare.glyph(), label_of(&v.layers[hi]))
        });
        let window_line = format!("C {} W {}", short_value(wl.0), short_value(wl.1));
        let size_line = format!(
            "{} × {} px · {:.2} × {:.2} mm · {window_line}",
            near_img.cols, near_img.rows, near_img.spacing[0], near_img.spacing[1],
        );
        let hover_value = ui
            .input(|i| i.pointer.hover_pos())
            .filter(|p| rect.contains(*p) && near_rect.contains(*p))
            .and_then(|p| {
                let i = ((p.x - near_rect.left()) / (sx * zoom)).floor() as i64;
                let j = ((p.y - near_rect.top()) / (sy * zoom)).floor() as i64;
                (i >= 0 && j >= 0 && (i as usize) < near_img.cols && (j as usize) < near_img.rows)
                    .then(|| {
                        let val = near_img.data[j as usize * near_img.cols + i as usize];
                        format!("({i}, {j})  {}", short_value(val))
                    })
            });

        // ---- annotations: a static title, as every pane has ----
        let has_slider = nl >= 2;
        if self.show_labels {
            let both = self.comparing();
            let room = rect.width() - 70.0;
            let fitted = |text: String, size: f32, color: Color32| {
                let mut job = egui::text::LayoutJob::single_section(
                    text,
                    egui::TextFormat::simple(FontId::proportional(size), color),
                );
                job.wrap = egui::text::TextWrapping::truncate_at_width(room.max(40.0));
                painter.layout_job(job)
            };
            let title = if both {
                format!("{title} · {}", SLOT_NAMES[slot])
            } else {
                title
            };
            painter.galley(
                rect.left_top() + Vec2::new(6.0, 4.0),
                fitted(title, 14.0, Color32::WHITE),
                Color32::WHITE,
            );
            if let Some(sub) = subtitle {
                painter.galley(
                    rect.left_top() + Vec2::new(6.0, 22.0),
                    fitted(sub, 12.0, Color32::LIGHT_GRAY),
                    Color32::LIGHT_GRAY,
                );
            }
            if nl >= 2 {
                painter.text(
                    rect.right_top() + Vec2::new(-6.0, 4.0),
                    Align2::RIGHT_TOP,
                    format!("{}/{nl}", near + 1),
                    FontId::proportional(12.0),
                    Color32::LIGHT_GRAY,
                );
            }
            let bottom = rect.bottom() - if has_slider { 26.0 } else { 6.0 };
            let font = FontId::proportional(11.0);
            let full = painter.layout_no_wrap(size_line.clone(), font.clone(), Color32::LIGHT_GRAY);
            let line = if full.size().x <= rect.width() - 12.0 {
                size_line
            } else {
                window_line
            };
            painter.text(
                Pos2::new(rect.left() + 6.0, bottom),
                Align2::LEFT_BOTTOM,
                line,
                font.clone(),
                Color32::LIGHT_GRAY,
            );
            if let Some(t) = hover_value
                .as_ref()
                .or(mode_note.map(String::from).as_ref())
            {
                painter.text(
                    Pos2::new(rect.left() + 6.0, bottom - 15.0),
                    Align2::LEFT_BOTTOM,
                    t,
                    font,
                    Color32::LIGHT_GRAY,
                );
            }
        }

        // ---- the bar: a plane's buttons in a plane's places, then the mode ----
        let is_max = self.maximized == Some((slot, kind));
        let bsize = egui::vec2(24.0, 20.0);
        let by = rect.top() + 22.0 + if nl >= 2 { 16.0 } else { 0.0 };
        let cell = |right: f32| Rect::from_min_size(Pos2::new(right - bsize.x, by), bsize);
        let max_rect = cell(rect.right() - 4.0);
        let folded = self.pane_buttons_hidden;
        let fold_rect = cell(max_rect.left() - 4.0);
        let fit_rect = cell(fold_rect.left() - 4.0);
        let hand_rect = cell(fit_rect.left() - 4.0);
        let zout_rect = cell(hand_rect.left() - 4.0);
        let zin_rect = cell(zout_rect.left() - 4.0);
        let mode_rect = cell(zin_rect.left() - 4.0);
        let close_rect = if nl >= 2 {
            cell(mode_rect.left() - 4.0)
        } else {
            mode_rect
        };
        let slider_rect = Rect::from_min_max(
            Pos2::new(rect.left() + 10.0, rect.bottom() - 18.0),
            Pos2::new(rect.right() - 10.0, rect.bottom() - 5.0),
        );
        let (pointer_pos, any_click) =
            ui.input(|i| (i.pointer.interact_pos(), i.pointer.any_click()));
        let over_buttons = pointer_pos
            .map(|p| {
                max_rect.contains(p)
                    || fold_rect.contains(p)
                    || (!folded
                        && (fit_rect.contains(p)
                            || hand_rect.contains(p)
                            || zout_rect.contains(p)
                            || zin_rect.contains(p)
                            || close_rect.contains(p)
                            || (nl >= 2 && mode_rect.contains(p))))
                    || (has_slider && slider_rect.expand2(egui::vec2(0.0, 5.0)).contains(p))
            })
            .unwrap_or(false);

        let resp = ui.interact(
            rect,
            egui::Id::new(("planar_pane", slot, k)),
            Sense::click_and_drag(),
        );
        // The 2D/3D section's region and marker tools take the left button.
        let tool_on = marked.is_some_and(|(i, _)| self.reg2d3d_pane_tool(slot, i));
        let pane_edit = match marked {
            Some((i, r)) if !over_buttons => self.reg2d3d_pane_input(slot, i, r, &resp),
            _ => None,
        };
        let max_resp = ui
            .put(
                max_rect,
                egui::Button::new(if is_max { "⊞" } else { "⛶" }).small(),
            )
            .on_hover_text(if is_max {
                "Restore the multi-view layout"
            } else {
                "Maximize this pane to the whole window"
            });
        let fold_resp = ui
            .put(
                fold_rect,
                egui::Button::new(if folded { "◀" } else { "▶" }).small(),
            )
            .on_hover_text(if folded {
                "Show the buttons of this bar. One switch for every pane, and it is \
                 remembered between runs."
            } else {
                "Fold these buttons away, leaving the image. One switch for every pane, \
                 and it is remembered between runs."
            });
        let mut fit_resp = None;
        let mut hand_resp = None;
        let mut zout_resp = None;
        let mut zin_resp = None;
        let mut mode_resp = None;
        let mut close_resp = None;
        if !folded {
            fit_resp = Some(
                ui.put(fit_rect, egui::Button::new("⟲").small())
                    .on_hover_text("Fit zoom, clear pan, each image's own window"),
            );
            hand_resp = Some(
                ui.put(
                    hand_rect,
                    egui::Button::selectable(self.hand_pan, "✋")
                        .frame_when_inactive(true)
                        .small(),
                )
                .on_hover_text(
                    "Move the image with the left button. Off, a middle drag still moves \
                     it. Shared by every view.",
                ),
            );
            zout_resp = Some(
                ui.put(zout_rect, egui::Button::new("➖").small())
                    .on_hover_text("Zoom out"),
            );
            zin_resp = Some(
                ui.put(zin_rect, egui::Button::new("➕").small())
                    .on_hover_text("Zoom in"),
            );
            if nl >= 2 {
                mode_resp = Some(
                    ui.put(mode_rect, egui::Button::new(compare.glyph()).small())
                        .on_hover_text(format!(
                            "Comparing by {}; click for {}",
                            compare.name(),
                            compare.next().name()
                        )),
                );
            }
            close_resp = Some(
                ui.put(close_rect, egui::Button::new("✖").small())
                    .on_hover_text("Take this pane off the row"),
            );
        }
        let mut new_blend = None;
        if has_slider {
            ui.painter().rect_filled(
                Rect::from_min_max(
                    Pos2::new(rect.left(), slider_rect.top() - 4.0),
                    Pos2::new(rect.right(), rect.bottom()),
                ),
                0.0,
                Color32::from_black_alpha(90),
            );
            let mut b = blend;
            let saved = ui.spacing().slider_width;
            ui.spacing_mut().slider_width = slider_rect.width();
            let r = ui.put(
                slider_rect,
                egui::Slider::new(&mut b, 0.0..=(nl - 1) as f32).show_value(false),
            );
            ui.spacing_mut().slider_width = saved;
            if r.changed() {
                new_blend = Some(b);
            }
        }

        let hit = |r: Rect, resp: &Option<egui::Response>| {
            resp.as_ref().is_some_and(|x| x.clicked())
                || (!folded && any_click && pointer_pos.map(|p| r.contains(p)).unwrap_or(false))
        };
        let clicked_max = max_resp.clicked()
            || (any_click && pointer_pos.map(|p| max_rect.contains(p)).unwrap_or(false));
        let clicked_fold = fold_resp.clicked()
            || (any_click && pointer_pos.map(|p| fold_rect.contains(p)).unwrap_or(false));
        let clicked_fit = hit(fit_rect, &fit_resp);
        let clicked_hand = hit(hand_rect, &hand_resp);
        let clicked_zout = hit(zout_rect, &zout_resp);
        let clicked_zin = hit(zin_rect, &zin_resp);
        let clicked_mode = nl >= 2 && hit(mode_rect, &mode_resp);
        let clicked_close = hit(close_rect, &close_resp);

        let mut new_zoom = (clicked_zin || clicked_zout).then(|| {
            let step = if clicked_zin { 1.25 } else { 1.0 / 1.25 };
            (zoom * step).clamp(fit_zoom * 0.2, fit_zoom * 40.0)
        });
        let mut new_pan = None;
        let mut new_wl = None;
        let mut new_accum = None;

        if resp.hovered() && !over_buttons {
            // The wheel steps through the stack, as it steps through the
            // slices of a plane; Ctrl + wheel (or a pinch) zooms about the
            // pointer, as it does there.
            let (lines, zoom_delta, pointer) = ui.input(|i| {
                let mut lines = 0.0f32;
                for e in &i.events {
                    if let egui::Event::MouseWheel {
                        unit,
                        delta,
                        modifiers,
                        ..
                    } = e
                    {
                        let scale = match unit {
                            egui::MouseWheelUnit::Line => 1.0,
                            egui::MouseWheelUnit::Point => 1.0 / 40.0,
                            egui::MouseWheelUnit::Page => 10.0,
                        };
                        if !(modifiers.ctrl || modifiers.command) {
                            lines += delta.y * scale;
                        }
                    }
                }
                (lines, i.zoom_delta(), i.pointer.hover_pos())
            });
            if (zoom_delta - 1.0).abs() > 1e-4 {
                let z1 = (zoom * zoom_delta).clamp(fit_zoom * 0.2, fit_zoom * 40.0);
                if let Some(mp) = pointer {
                    let rel = (mp - rect.center()) / zoom - pan;
                    new_pan = Some((mp - rect.center()) / z1 - rel);
                }
                new_zoom = Some(z1);
            }
            if lines.abs() > 0.0 && nl >= 2 {
                let acc = scroll_accum + lines;
                let steps = acc.trunc() as i64;
                new_accum = Some(acc - steps as f32);
                if steps != 0 {
                    // Whole images, wheel up towards the first.
                    let to = (blend.round() as i64 - steps).clamp(0, nl as i64 - 1);
                    new_blend = Some(to as f32);
                }
            }
        }
        if resp.dragged_by(egui::PointerButton::Secondary) {
            // The CT views' window drag: x widens, y moves the centre.
            let d = resp.drag_delta();
            let w = (wl.1 * (1.0 + d.x * 0.005)).clamp(bounds.0, bounds.1);
            new_wl = Some((wl.0 + d.y * w * 0.002, w));
        }
        if resp.dragged_by(egui::PointerButton::Middle)
            || (self.hand_pan
                && !over_buttons
                && !tool_on
                && resp.dragged_by(egui::PointerButton::Primary))
        {
            new_pan = Some(pan + resp.drag_delta() / zoom);
        }
        if resp.hovered() {
            self.hovered_slot = slot;
        }

        // ---- apply ----
        if clicked_fold {
            self.pane_buttons_hidden = !self.pane_buttons_hidden;
            self.persist_settings();
        }
        if clicked_hand {
            self.hand_pan = !self.hand_pan;
        }
        if clicked_max {
            self.maximized = if is_max { None } else { Some((slot, kind)) };
        }
        if clicked_close {
            self.remove_from_row(slot, kind);
            return;
        }
        if let Some(e) = pane_edit {
            self.reg2d3d_pane_apply(e);
        }
        let v = &mut self.slots[slot].planar[k];
        if clicked_mode {
            v.compare = v.compare.next();
        }
        if let Some(a) = new_accum {
            v.scroll_accum = a;
        }
        if let Some(b) = new_blend {
            v.blend = b;
        }
        if let Some(z) = new_zoom {
            v.zoom = z;
        }
        if let Some(p) = new_pan {
            v.pan = p;
        }
        if let Some(w) = new_wl {
            v.layers[near].wl = w;
        }
        if clicked_fit {
            v.zoom = 0.0;
            v.pan = Vec2::ZERO;
            for l in &mut v.layers {
                l.wl_for = None;
            }
            v.layers[near].wl = default_wl;
        }
    }
}

/// A pixel value or a window bound, as few digits as say it: a DRR's line
/// integrals are a few units, a radiograph's raw values tens of thousands.
fn short_value(v: f32) -> String {
    let a = v.abs();
    if a >= 100.0 || a == 0.0 {
        format!("{v:.0}")
    } else if a >= 1.0 {
        format!("{v:.2}")
    } else {
        format!("{v:.4}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(min: f32, max: f32) -> crate::extras::PlanarImage {
        crate::extras::PlanarImage {
            label: "x".into(),
            modality: "RTIMAGE".into(),
            rows: 1,
            cols: 2,
            spacing: [1.0, 1.0],
            data: vec![min, max],
            min_value: min,
            max_value: max,
            window: ((min + max) * 0.5, (max - min).max(1.0)),
            info: Vec::new(),
            ..Default::default()
        }
    }

    #[test]
    fn the_window_follows_the_scale_of_the_values() {
        // A DRR's optical depths: the floor must sit far below the range,
        // or the window could never be narrowed onto the bone.
        let (lo, hi) = width_bounds(&image(0.0, 6.0));
        assert!(lo < 0.01 && hi > 6.0);
        // A 16-bit radiograph.
        let (lo, hi) = width_bounds(&image(0.0, 65535.0));
        assert!(lo < 10.0 && hi > 65535.0);
        // The window an image carries is the one it comes up with, kept
        // inside those bounds.
        assert_eq!(default_window(&image(0.0, 6.0)), (3.0, 6.0));
        assert_eq!(windowed(&image(0.0, 6.0), (3.0, 6.0)), vec![0, 255]);
    }

    #[test]
    fn the_scrubber_names_the_pair_it_compares() {
        assert_eq!(pair_at(0.0, 1), (0, 0, 0.0));
        assert_eq!(pair_at(0.25, 2), (0, 1, 0.25));
        // At the top of the stack the pair is the last two, wholly the upper.
        assert_eq!(pair_at(1.0, 2), (0, 1, 1.0));
        assert_eq!(pair_at(2.0, 3), (1, 2, 1.0));
        assert_eq!(pair_at(9.0, 3), (1, 2, 1.0), "clamped into the stack");
        assert_eq!(Compare::Edges.next(), Compare::Flicker);
        assert_eq!(Compare::Flicker.next(), Compare::Fade);
    }

    #[test]
    fn the_comparison_images_say_what_they_compare() {
        let d = difference_image(&[100, 100, 0], &[100, 0, 100], [3, 1]);
        assert_eq!(
            d.pixels[0],
            Color32::from_gray(128),
            "agreement is mid-grey"
        );
        assert!(d.pixels[1].r() > 128 && d.pixels[2].r() < 128);
        // A vertical step has edges along it and nowhere else.
        let mut g = vec![0u8; 8 * 8];
        for j in 0..8 {
            for i in 4..8 {
                g[j * 8 + i] = 255;
            }
        }
        let e = edge_image(&g, [8, 8]);
        let opaque: Vec<usize> = (0..64).filter(|p| e.pixels[*p].a() > 0).collect();
        assert!(!opaque.is_empty());
        assert!(opaque.iter().all(|p| (3..=4).contains(&(p % 8))));
    }

    #[test]
    fn values_are_written_with_the_digits_they_need() {
        assert_eq!(short_value(12345.6), "12346");
        assert_eq!(short_value(4.256), "4.26");
        assert_eq!(short_value(0.01234), "0.0123");
        assert_eq!(short_value(0.0), "0");
    }
}
