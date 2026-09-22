//! *File ▸ Save image*: one row of the central area, or both, written out as
//! a PNG or a JPEG at a chosen resolution.
//!
//! The picture is the row's own pixels, taken with egui's screenshot the way
//! [`super::record`] takes a run's frames, so a figure carries the contours,
//! the dose wash, the crosshair, the orientation labels and the 3D surfaces
//! exactly as they are on screen.
//!
//! **What the DPI does.** egui draws into the window's framebuffer and
//! nothing else, so a pane cannot be re-rendered larger for one frame: the
//! capture is at screen resolution, whatever is asked for. The DPI therefore
//! does two things. It is written into the file - `pHYs` in a PNG, the JFIF
//! density in a JPEG - so that Word, LaTeX or InDesign place the figure at
//! its intended physical size instead of guessing; and the image is resampled
//! to match, so the placed figure is not a handful of pixels stretched by the
//! layout program's own filter. Resampling enlarges, it does not invent
//! detail, and the dialog says so.
//!
//! The reference is 96 pixels per inch, on the display's own scale: a pane
//! 600 points wide is 6.25 inches of figure, which is 1875 pixels at 300 DPI.
//! A window already running at 2 points per pixel has half of that in hand
//! before anything is resampled, and the factor accounts for it.

use std::path::PathBuf;

use anyhow::{Context as _, Result};

use super::*;

/// What a saved image holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ImgWhat {
    /// One workspace: its row of the central area.
    Row(usize),
    /// Every row on screen, as they sit one above the other.
    Both,
}

/// What it is written as.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) enum ImgFormat {
    #[default]
    Png,
    Jpeg,
}

impl ImgFormat {
    pub(super) fn label(self) -> &'static str {
        match self {
            ImgFormat::Png => "PNG",
            ImgFormat::Jpeg => "JPEG",
        }
    }

    pub(super) fn ext(self) -> &'static str {
        match self {
            ImgFormat::Png => "png",
            ImgFormat::Jpeg => "jpg",
        }
    }

    pub(super) fn hint(self) -> &'static str {
        match self {
            ImgFormat::Png => {
                "Lossless, and every pixel is the one that was on screen. What a figure \
                 of an image with contours and text on it should be."
            }
            ImgFormat::Jpeg => {
                "Smaller, and lossy: the compression puts rings around high-contrast \
                 edges, which is what a contour line and a slice counter are. For a \
                 quick look rather than a publication."
            }
        }
    }
}

/// The *Save image* dialog's own state.
pub(super) struct SaveImgDialog {
    pub(super) what: ImgWhat,
    pub(super) format: ImgFormat,
    pub(super) dpi: u32,
    /// JPEG only.
    pub(super) quality: u8,
    pub(super) status: Option<String>,
}

impl Default for SaveImgDialog {
    fn default() -> Self {
        SaveImgDialog {
            what: ImgWhat::Row(0),
            format: ImgFormat::default(),
            // 300 DPI is what a journal asks for.
            dpi: 300,
            quality: 92,
            status: None,
        }
    }
}

/// A picture asked for and waiting for its frame.
pub(super) struct PendingShot {
    pub(super) crop: [usize; 4],
    pub(super) dest: PathBuf,
    pub(super) format: ImgFormat,
    pub(super) dpi: u32,
    pub(super) quality: u8,
    /// How many passes to let go by before asking for the picture, so the
    /// dialog that asked for it is off the screen first.
    pub(super) wait: u8,
    /// The command has been sent and the picture is on its way.
    pub(super) asked: bool,
}

/// The reference resolution a point is worth: 96 pixels to the inch, the
/// same one every layout program assumes when a file says nothing.
const REFERENCE_DPI: f32 = 96.0;

/// Pixels of output per pixel of capture.
///
/// The capture is already `ppp` pixels per point, so a window on a
/// high-resolution display has part of the way covered before anything is
/// resampled. Held at 1/8 to 8 so a slip of the DPI field cannot ask for a
/// gigapixel.
pub(super) fn scale_for(dpi: u32, ppp: f32) -> f32 {
    (dpi as f32 / REFERENCE_DPI / ppp.max(0.1)).clamp(0.125, 8.0)
}

/// Write `img` as a PNG carrying `dpi` in its `pHYs` chunk.
pub(super) fn write_png(path: &std::path::Path, img: &egui::ColorImage, dpi: u32) -> Result<()> {
    let (w, h) = (img.size[0] as u32, img.size[1] as u32);
    let file = std::fs::File::create(path).with_context(|| format!("create {}", path.display()))?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w, h);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    // pHYs is per metre, and an inch is 0.0254 of one.
    let ppm = ((dpi as f64) / 0.0254).round() as u32;
    enc.set_pixel_dims(Some(png::PixelDimensions {
        xppu: ppm,
        yppu: ppm,
        unit: png::Unit::Meter,
    }));
    let mut writer = enc
        .write_header()
        .with_context(|| format!("write the header of {}", path.display()))?;
    writer
        .write_image_data(&rgba_bytes(img))
        .with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

/// Write `img` as a JPEG carrying `dpi` in its JFIF density.
pub(super) fn write_jpeg(
    path: &std::path::Path,
    img: &egui::ColorImage,
    dpi: u32,
    quality: u8,
) -> Result<()> {
    use image::codecs::jpeg::{JpegEncoder, PixelDensity};
    let (w, h) = (img.size[0] as u32, img.size[1] as u32);
    // JPEG has no alpha; the views are drawn on an opaque backdrop anyway.
    let mut rgb = Vec::with_capacity((w * h) as usize * 3);
    for p in &img.pixels {
        let [r, g, b, _] = p.to_array();
        rgb.extend_from_slice(&[r, g, b]);
    }
    let file = std::fs::File::create(path).with_context(|| format!("create {}", path.display()))?;
    let mut enc =
        JpegEncoder::new_with_quality(std::io::BufWriter::new(file), quality.clamp(1, 100));
    enc.set_pixel_density(PixelDensity::dpi(dpi.clamp(1, 65535) as u16));
    enc.encode(&rgb, w, h, image::ExtendedColorType::Rgb8)
        .with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

/// Resample to `scale`, or hand the image straight back when there is
/// nothing to do.
pub(super) fn resample(img: &egui::ColorImage, scale: f32) -> egui::ColorImage {
    let (w, h) = (img.size[0] as u32, img.size[1] as u32);
    let nw = ((w as f32 * scale).round() as u32).max(1);
    let nh = ((h as f32 * scale).round() as u32).max(1);
    if (nw, nh) == (w, h) || w == 0 || h == 0 {
        return img.clone();
    }
    let Some(buf) = image::RgbaImage::from_raw(w, h, rgba_bytes(img)) else {
        return img.clone();
    };
    // Lanczos3: the sharpest of the sensible filters, which matters because
    // what is being enlarged is mostly thin bright lines on dark anatomy.
    let out = image::imageops::resize(&buf, nw, nh, image::imageops::FilterType::Lanczos3);
    let pixels = out
        .pixels()
        .map(|p| egui::Color32::from_rgba_unmultiplied(p[0], p[1], p[2], p[3]))
        .collect();
    egui::ColorImage::new([nw as usize, nh as usize], pixels)
}

fn rgba_bytes(img: &egui::ColorImage) -> Vec<u8> {
    let mut out = Vec::with_capacity(img.pixels.len() * 4);
    for p in &img.pixels {
        out.extend_from_slice(&p.to_array());
    }
    out
}

impl ViewerApp {
    /// Remember where a workspace's row was drawn, so *Save image* knows what
    /// part of the window to cut out.
    pub(super) fn note_row_rect(&mut self, slot: usize, rect: Rect) {
        if slot < MAX_WORKSPACES {
            self.row_rects[slot] = Some(rect);
        }
    }

    /// The rectangle a choice asks for, in points.
    fn snap_rect(&self, what: ImgWhat) -> Option<Rect> {
        match what {
            ImgWhat::Row(s) => self.row_rects[s.min(MAX_WORKSPACES - 1)],
            ImgWhat::Both => self
                .row_rects
                .iter()
                .flatten()
                .copied()
                .reduce(|a, b| a.union(b)),
        }
    }

    /// *File ▸ Save image*.
    pub(super) fn save_image_window(&mut self, ctx: &egui::Context) {
        let Some(mut d) = self.save_img.take() else {
            return;
        };
        let rows = self.open_slots();
        let both = rows.len() > 1;
        // A row that is no longer on screen cannot be saved.
        if !matches!(d.what, ImgWhat::Row(s) if rows.contains(&s)) && d.what != ImgWhat::Both {
            d.what = ImgWhat::Row(rows[0]);
        }
        if !both && d.what == ImgWhat::Both {
            d.what = ImgWhat::Row(rows[0]);
        }
        let ppp = ctx.pixels_per_point();
        let scale = scale_for(d.dpi, ppp);
        let size = self.snap_rect(d.what).map(|r| {
            (
                ((r.width() * ppp) * scale).round() as i64,
                ((r.height() * ppp) * scale).round() as i64,
            )
        });
        let mut open = true;
        let mut close = false;
        let mut save = false;
        detach::tool_window(
            ctx,
            "save_image",
            "💾 Save image",
            &mut open,
            detach::WinOpts::default(),
            |ui| {
                ui.set_max_width(460.0);
                ui.label(
                    "A picture of the central area exactly as it is on screen: the panes \
                     as laid out, with their contours, dose, crosshair, labels and 3D \
                     surfaces.",
                );
                ui.add_space(6.0);
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing.x = 4.0;
                    ui.label("Rows");
                    for slot in &rows {
                        ui.selectable_value(
                            &mut d.what,
                            ImgWhat::Row(*slot),
                            format!("Workspace {}", SLOT_NAMES[*slot]),
                        )
                        .on_hover_text("That row on its own");
                    }
                    ui.add_enabled_ui(both, |ui| {
                        ui.selectable_value(&mut d.what, ImgWhat::Both, "All")
                            .on_hover_text("Every row together, one above the other");
                    });
                    if !both {
                        ui.weak("one workspace on screen");
                    }
                });
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing.x = 4.0;
                    ui.label("Format");
                    for f in [ImgFormat::Png, ImgFormat::Jpeg] {
                        ui.selectable_value(&mut d.format, f, f.label())
                            .on_hover_text(f.hint());
                    }
                    if d.format == ImgFormat::Jpeg {
                        ui.add_space(8.0);
                        ui.label("Quality");
                        ui.add(
                            egui::DragValue::new(&mut d.quality)
                                .speed(1)
                                .range(40..=100),
                        );
                    }
                });
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing.x = 4.0;
                    ui.label("Resolution");
                    ui.add(
                        egui::DragValue::new(&mut d.dpi)
                            .speed(10)
                            .range(48..=1200)
                            .suffix(" DPI"),
                    )
                    .on_hover_text(
                        "Written into the file, so the figure is placed at its intended \
                         physical size, and the image is resampled to match.",
                    );
                    for p in [150u32, 300, 600] {
                        if ui.selectable_label(d.dpi == p, format!("{p}")).clicked() {
                            d.dpi = p;
                        }
                    }
                });
                match size {
                    Some((w, h)) if w > 0 && h > 0 => {
                        ui.weak(format!(
                            "{w} × {h} pixels, {:.2} × {:.2} inches at {} DPI",
                            w as f32 / d.dpi as f32,
                            h as f32 / d.dpi as f32,
                            d.dpi
                        ));
                    }
                    _ => {
                        ui.weak("Nothing to save: that row is not on screen.");
                    }
                }
                if scale > 1.01 {
                    ui.weak(format!(
                        "Enlarged ×{scale:.2} from what the window draws. It scales the \
                         picture up so it prints at the right size; it does not add \
                         detail that was never rendered."
                    ));
                }
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(size.is_some(), egui::Button::new("💾 Save"))
                        .on_hover_text("Choose a file, then take the picture")
                        .clicked()
                    {
                        save = true;
                    }
                    if ui.button("Close").clicked() {
                        close = true;
                    }
                });
                if let Some(msg) = &d.status {
                    ui.add_space(4.0);
                    ui.weak(msg);
                }
            },
        );

        if save {
            match self.arm_snapshot(&d, ppp) {
                // The dialog would otherwise be in its own picture.
                Ok(true) => close = true,
                Ok(false) => {}
                Err(e) => d.status = Some(format!("{e:#}")),
            }
        }
        if open && !close {
            self.save_img = Some(d);
        } else {
            self.snap_status = d.status;
        }
    }

    /// Ask for the file, work out the crop, and queue the picture.
    ///
    /// `Ok(false)` means the file dialog was dismissed and nothing is queued.
    fn arm_snapshot(&mut self, d: &SaveImgDialog, ppp: f32) -> Result<bool> {
        let rect = self
            .snap_rect(d.what)
            .context("that row is not on screen")?;
        let name = match d.what {
            ImgWhat::Row(s) => format!(
                "view_{}.{}",
                SLOT_NAMES[s.min(MAX_WORKSPACES - 1)],
                d.format.ext()
            ),
            ImgWhat::Both => format!(
                "view_{}.{}",
                self.open_slots()
                    .into_iter()
                    .map(|s| SLOT_NAMES[s])
                    .collect::<String>(),
                d.format.ext()
            ),
        };
        let crop = [
            (rect.left() * ppp).round().max(0.0) as usize,
            (rect.top() * ppp).round().max(0.0) as usize,
            (rect.width() * ppp).round().max(1.0) as usize,
            (rect.height() * ppp).round().max(1.0) as usize,
        ];
        let (format, dpi, quality) = (d.format, d.dpi, d.quality);
        self.ask_save("Save the image", name, None, None, move |app, dest| {
            app.snap = Some(PendingShot {
                crop,
                dest,
                format,
                dpi,
                quality,
                // One pass for the dialog to leave the screen.
                wait: 1,
                asked: false,
            });
            // Where the dialog outlived the file browser (Android), it has
            // been put back in the meantime and would be in its own picture:
            // close it the way the caller does when the answer is immediate.
            if let Some(d) = app.save_img.take() {
                app.snap_status = d.status;
            }
        });
        Ok(self.snap.is_some())
    }

    /// Ask for the queued picture, and write it when it arrives.
    pub(super) fn snapshot_tick(
        &mut self,
        ctx: &egui::Context,
        shot: Option<&std::sync::Arc<egui::ColorImage>>,
    ) {
        let Some(snap) = self.snap.as_mut() else {
            return;
        };
        if snap.wait > 0 {
            snap.wait -= 1;
            ctx.request_repaint();
            return;
        }
        if !snap.asked {
            snap.asked = true;
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
            ctx.request_repaint();
            return;
        }
        let Some(shot) = shot else { return };
        let snap = self.snap.take().expect("just checked");
        let Some(cut) = record::cut(shot, snap.crop) else {
            self.snap_status = Some(
                "The row moved out from under the picture before it was taken; nothing \
                 was written."
                    .into(),
            );
            return;
        };
        let img = resample(&cut, scale_for(snap.dpi, ctx.pixels_per_point()));
        let res = match snap.format {
            ImgFormat::Png => write_png(&snap.dest, &img, snap.dpi),
            ImgFormat::Jpeg => write_jpeg(&snap.dest, &img, snap.dpi, snap.quality),
        };
        self.snap_status = Some(match res {
            Ok(()) => format!(
                "{} × {} pixels written to {}",
                img.size[0],
                img.size[1],
                snap.dest.display()
            ),
            Err(e) => format!("Could not write the image: {e:#}"),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(w: usize, h: usize) -> egui::ColorImage {
        let pixels = (0..w * h)
            .map(|i| egui::Color32::from_gray((i % 256) as u8))
            .collect();
        egui::ColorImage::new([w, h], pixels)
    }

    #[test]
    fn the_scale_counts_the_pixels_the_window_already_has() {
        // A 1:1 window at the reference resolution needs no resampling.
        assert!((scale_for(96, 1.0) - 1.0).abs() < 1e-6);
        // 300 DPI on that window is a little over three times as many pixels.
        assert!((scale_for(300, 1.0) - 3.125).abs() < 1e-4);
        // The same 300 DPI on a window that already draws two pixels to the
        // point needs half of that.
        assert!((scale_for(300, 2.0) - 1.5625).abs() < 1e-4);
        // And a slip of the field cannot ask for a gigapixel.
        assert!((scale_for(100_000, 1.0) - 8.0).abs() < 1e-6);
    }

    #[test]
    fn resampling_lands_on_the_size_the_scale_asks_for() {
        let img = image(40, 30);
        let out = resample(&img, 2.5);
        assert_eq!(out.size, [100, 75]);
        assert_eq!(out.pixels.len(), 100 * 75);
        // A scale of one is the picture itself, untouched.
        let same = resample(&img, 1.0);
        assert_eq!(same.size, img.size);
        assert_eq!(same.pixels, img.pixels);
    }

    #[test]
    fn a_png_carries_the_resolution_it_was_saved_at() {
        let dir = std::env::temp_dir().join("rds_snap_png_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("view.png");
        write_png(&path, &image(16, 9), 300).expect("write the png");
        let dec = png::Decoder::new(std::io::BufReader::new(
            std::fs::File::open(&path).expect("open it"),
        ));
        let reader = dec.read_info().expect("read the header");
        let info = reader.info();
        assert_eq!([info.width, info.height], [16, 9]);
        let dims = info.pixel_dims.expect("the pHYs chunk is there");
        assert_eq!(dims.unit, png::Unit::Meter);
        // 300 dots per inch is 11811 per metre.
        assert_eq!(dims.xppu, 11811);
        assert_eq!(dims.yppu, dims.xppu);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_jpeg_is_written_and_reads_back_at_its_size() {
        let dir = std::env::temp_dir().join("rds_snap_jpg_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("view.jpg");
        write_jpeg(&path, &image(24, 12), 150, 90).expect("write the jpeg");
        let bytes = std::fs::read(&path).expect("read it back");
        assert_eq!(&bytes[..2], &[0xFF, 0xD8], "it really is a JPEG");
        let back = image::load_from_memory(&bytes).expect("decode it");
        assert_eq!((back.width(), back.height()), (24, 12));
        let _ = std::fs::remove_file(&path);
    }
}
