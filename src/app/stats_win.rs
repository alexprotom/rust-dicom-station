//! *Tools ▶ Structure details*: one row per structure, with the numbers a
//! planner reads off a list rather than off a picture.
//!
//! Volume twice over - by planimetry on the contours and by counting the
//! voxels they fill - because the two disagree on a coarse series and the
//! difference is worth seeing; the grey levels inside the structure, which
//! is how a mis-drawn organ gives itself away; how many slices and points
//! the geometry costs; and whether the structure still matches the recipe
//! it was derived from.
//!
//! The table is computed on demand, not every frame: rasterizing a hundred
//! structures is a second of work, and nothing here changes unless the
//! geometry does.

use crate::contours::Stack;
use crate::derived::Status;
use crate::volume::{Grid, Volume};

use super::seg_engines::{ToolId, ToolInfo};
use super::*;

/// The ninth tool. A clipboard, because this one only reads; the glyph is
/// one egui's bundled fonts carry (see the `glyphs` guard).
pub(super) const DETAILS: ToolInfo = ToolInfo {
    id: ToolId::Details,
    glyph: "📋",
    name: "Structure details",
};

/// One line of the table.
pub(super) struct StatRow {
    pub name: String,
    pub color: [u8; 3],
    pub roi_type: String,
    /// "contours" or "voxels" - the representation the geometry is kept in.
    pub repr: &'static str,
    /// Planimetry on the contours: the area of every slice times the slice
    /// spacing. Empty for a segmentation, which has no contours to measure.
    pub planimetry_cm3: Option<f64>,
    /// The voxels the geometry fills on the lattice it is measured on.
    pub voxel_cm3: f64,
    pub voxels: usize,
    /// Slices carrying geometry, and the points they cost.
    pub slices: usize,
    pub points: usize,
    /// Minimum, mean and maximum image value inside the structure; `None`
    /// when the structure is not on the displayed image's lattice.
    pub grey: Option<[f64; 3]>,
    /// Where a point of interest is, in patient coordinates: a point has no
    /// volume to report, and its position is the only number about it.
    pub point: Option<crate::geometry::Vec3>,
    pub status: Option<Status>,
}

pub(super) struct StatsDialog {
    pub slot: usize,
    pub rows: Vec<StatRow>,
    /// The `settings_gen` the rows were computed at, so the window can say
    /// when it is showing something older than the geometry.
    pub gen: u64,
    /// Set once, so an opened window fills itself without a click.
    pub stale: bool,
}

/// Voxel count and grey-level statistics of one mask, over its own box.
fn measure(mask: &[u8], dims: [usize; 3], vol: Option<&Volume>) -> (usize, Option<[f64; 3]>) {
    let [nx, ny, nz] = dims;
    let same = vol.is_some_and(|v| v.dims == dims);
    let mut count = 0usize;
    let mut lo = f64::MAX;
    let mut hi = f64::MIN;
    let mut sum = 0.0f64;
    for k in 0..nz {
        for j in 0..ny {
            let row = k * nx * ny + j * nx;
            for i in 0..nx {
                if mask[row + i] == 0 {
                    continue;
                }
                count += 1;
                if same {
                    let v = vol.expect("checked").data[row + i] as f64;
                    lo = lo.min(v);
                    hi = hi.max(v);
                    sum += v;
                }
            }
        }
    }
    let grey = if same && count > 0 {
        Some([lo, sum / count as f64, hi])
    } else {
        None
    };
    (count, grey)
}

fn voxel_cm3(spacing: [f64; 3], voxels: usize) -> f64 {
    voxels as f64 * crate::volume::voxel_cm3(spacing)
}

impl ViewerApp {
    pub(super) fn open_stats_dialog(&mut self, slot: usize) {
        self.stats_dialog = Some(StatsDialog {
            slot,
            rows: Vec::new(),
            gen: self.settings_gen,
            stale: true,
        });
    }

    /// Build the table. Every structure is measured on the lattice it is
    /// drawn on: an RT structure on the displayed image, a segmentation on
    /// the series it was painted on, which is why the grey levels are only
    /// filled in for the ones that share the displayed volume's lattice.
    fn stats_rows(&self, slot: usize) -> Vec<StatRow> {
        let Some(study) = self.slots[slot].study.as_ref() else {
            return Vec::new();
        };
        let grid: Grid = study.volume.grid();
        let vol: &Volume = &study.volume;
        let mut rows: Vec<StatRow> = Vec::new();
        for (si, set) in study.structure_sets.iter().enumerate() {
            let active = si == self.slots[slot].active_structs;
            let measured: Vec<StatRow> = set
                .rois
                .par_iter()
                .map(|roi| {
                    // A point of interest has nothing to rasterize; its
                    // position is what there is to say about it.
                    if let Some(p) = roi.point() {
                        return StatRow {
                            name: roi.name.clone(),
                            color: roi.color,
                            roi_type: roi.roi_type.clone(),
                            repr: "point",
                            planimetry_cm3: None,
                            voxel_cm3: 0.0,
                            voxels: 0,
                            slices: 0,
                            points: 1,
                            grey: None,
                            point: Some(p),
                            status: None,
                        };
                    }
                    let st = Stack::from_roi(roi, &grid);
                    let mask = st.rasterize(grid.dims);
                    let (voxels, grey) = measure(&mask, grid.dims, Some(vol));
                    StatRow {
                        name: roi.name.clone(),
                        color: roi.color,
                        roi_type: roi.roi_type.clone(),
                        repr: "contours",
                        planimetry_cm3: Some(st.volume_cm3(grid.spacing)),
                        voxel_cm3: voxel_cm3(grid.spacing, voxels),
                        voxels,
                        slices: st.occupied(),
                        points: roi.contours.iter().map(|c| c.points.len()).sum(),
                        grey,
                        point: None,
                        status: None,
                    }
                })
                .collect();
            for (ri, mut r) in measured.into_iter().enumerate() {
                // The derived cache is only kept for the active set, which
                // is the only one the recipes are refreshed against.
                if active {
                    r.status = self.derived_status(slot, ri);
                }
                rows.push(r);
            }
        }
        for ser in &study.seg_series {
            let dims = ser.grid.dims;
            let on_display = ser.grid.matches(&grid);
            let measured: Vec<StatRow> = ser
                .segs
                .par_iter()
                .map(|seg| {
                    let (voxels, grey) =
                        measure(&seg.mask, dims, if on_display { Some(vol) } else { None });
                    StatRow {
                        name: seg.name.clone(),
                        color: seg.color,
                        roi_type: String::new(),
                        repr: "voxels",
                        planimetry_cm3: None,
                        voxel_cm3: voxel_cm3(ser.grid.spacing, voxels),
                        voxels,
                        slices: 0,
                        points: 0,
                        grey,
                        point: None,
                        status: None,
                    }
                })
                .collect();
            rows.extend(measured);
        }
        rows
    }

    fn stats_csv(rows: &[StatRow]) -> String {
        let mut s = String::from(
            "name,type,representation,planimetry_cm3,voxel_cm3,voxels,slices,points,\
             grey_min,grey_mean,grey_max,derived,point_mm\n",
        );
        for r in rows {
            let g = |i: usize| match r.grey {
                Some(v) => format!("{:.2}", v[i]),
                None => String::new(),
            };
            let place = match r.point {
                Some(p) => format!("{:.3} {:.3} {:.3}", p.x, p.y, p.z),
                None => String::new(),
            };
            s.push_str(&format!(
                "\"{}\",{},{},{},{:.3},{},{},{},{},{},{},{},{}\n",
                r.name.replace('"', "'"),
                r.roi_type,
                r.repr,
                match r.planimetry_cm3 {
                    Some(v) => format!("{v:.3}"),
                    None => String::new(),
                },
                r.voxel_cm3,
                r.voxels,
                r.slices,
                r.points,
                g(0),
                g(1),
                g(2),
                r.status.map(|s| s.label()).unwrap_or(""),
                place
            ));
        }
        s
    }

    pub(super) fn stats_window(&mut self, ctx: &egui::Context) {
        let Some(slot) = self.stats_dialog.as_ref().map(|d| d.slot) else {
            return;
        };
        if self.stats_dialog.as_ref().is_some_and(|d| d.stale) {
            let rows = self.stats_rows(slot);
            let gen = self.settings_gen;
            if let Some(d) = &mut self.stats_dialog {
                d.rows = rows;
                d.gen = gen;
                d.stale = false;
            }
        }
        let has = [self.slots[0].study.is_some(), self.slots[1].study.is_some()];
        let current_gen = self.settings_gen;
        let mut open = true;
        let mut close = false;
        let mut save = false;
        let mut refresh = false;
        let mut switch: Option<usize> = None;
        let d = self.stats_dialog.as_mut().expect("checked above");
        detach::tool_window(
            ctx,
            "structure_details",
            format!("{} {}", DETAILS.glyph, DETAILS.name),
            &mut open,
            detach::WinOpts::width(700.0),
            |ui| {
                switch = seg_engines::dataset_row(ui, d.slot, has, true);
                ui.label(
                    egui::RichText::new(
                        "Volume twice over - the area of the contours times the slice \
                         spacing, and the voxels they fill. They disagree by a few per \
                         cent on a coarse series, and neither number is wrong.",
                    )
                    .weak(),
                );
                if d.gen != current_gen {
                    ui.label(
                        egui::RichText::new(
                            "⚠ A structure has changed since this table was computed.",
                        )
                        .color(theme::warn_color(ui.visuals())),
                    );
                }
                ui.separator();
                egui::ScrollArea::both().max_height(420.0).show(ui, |ui| {
                    egui::Grid::new("stats_grid")
                        .striped(true)
                        .num_columns(9)
                        .show(ui, |ui| {
                            for h in [
                                "Structure",
                                "Type",
                                "Kept as",
                                "Planimetry / place",
                                "Voxels",
                                "Slices",
                                "Points",
                                "Grey (min / mean / max)",
                                "Derived",
                            ] {
                                ui.label(egui::RichText::new(h).strong());
                            }
                            ui.end_row();
                            for r in &d.rows {
                                ui.horizontal(|ui| {
                                    let c = theme::rgb(r.color);
                                    ui.label(egui::RichText::new("■").color(c));
                                    ui.label(r.name.clone());
                                });
                                ui.label(r.roi_type.clone());
                                ui.label(r.repr);
                                match (r.planimetry_cm3, r.point) {
                                    (_, Some(p)) => {
                                        ui.label(format!("{:.1}, {:.1}, {:.1} mm", p.x, p.y, p.z));
                                    }
                                    (Some(v), None) => {
                                        ui.label(format!("{v:.2} cm³"));
                                    }
                                    (None, None) => {
                                        ui.label("-");
                                    }
                                }
                                if r.point.is_some() {
                                    ui.label("-");
                                } else {
                                    ui.label(format!("{:.2} cm³", r.voxel_cm3));
                                }
                                ui.label(if r.slices > 0 {
                                    r.slices.to_string()
                                } else {
                                    "-".into()
                                });
                                ui.label(if r.points > 0 {
                                    r.points.to_string()
                                } else {
                                    "-".into()
                                });
                                ui.label(match r.grey {
                                    Some(g) => {
                                        format!("{:.0} / {:.0} / {:.0}", g[0], g[1], g[2])
                                    }
                                    None => "-".into(),
                                });
                                match r.status {
                                    Some(s) => {
                                        let c = s.color();
                                        ui.label(
                                            egui::RichText::new(format!(
                                                "{} {}",
                                                s.glyph(),
                                                s.label()
                                            ))
                                            .color(theme::rgb(c)),
                                        );
                                    }
                                    None => {
                                        ui.label("-");
                                    }
                                }
                                ui.end_row();
                            }
                        });
                });
                if d.rows.is_empty() {
                    ui.label("This dataset has no structures yet.");
                }
                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button("Refresh").clicked() {
                        refresh = true;
                    }
                    ui.add_enabled_ui(!d.rows.is_empty(), |ui| {
                        if ui.button("Save CSV").clicked() {
                            save = true;
                        }
                    });
                    if ui.button("Close").clicked() {
                        close = true;
                    }
                });
            },
        );
        if let Some(s) = switch {
            if let Some(d) = &mut self.stats_dialog {
                d.slot = s;
                d.stale = true;
            }
        }
        if refresh {
            if let Some(d) = &mut self.stats_dialog {
                d.stale = true;
            }
        }
        if save {
            let csv = self
                .stats_dialog
                .as_ref()
                .map(|d| Self::stats_csv(&d.rows))
                .unwrap_or_default();
            if let Some(path) = rfd::FileDialog::new()
                .set_title("Save the structure table")
                .add_filter("CSV", &["csv"])
                .set_file_name("structure_details.csv")
                .save_file()
            {
                if let Err(e) = std::fs::write(&path, csv) {
                    self.error = Some(format!("Could not write {}: {e}", path.display()));
                }
            }
        }
        if close || !open {
            self.stats_dialog = None;
        }
    }
}
