//! Recording a playback run: what a pane shows, frame by frame, written out
//! as an animated GIF or a numbered PNG sequence.
//!
//! The frames are the pane's own pixels, not a second rendering of the
//! volume: a recording taken this way carries the contours, the dose wash,
//! the crosshair, the annotations and the 3D surfaces exactly as they were
//! on screen, because they *are* what was on screen. egui hands the whole
//! window over on request ([`egui::ViewportCommand::Screenshot`]) and the
//! pane's rectangle is cut out of it.
//!
//! **Arm first, then play.** Pressing *Record* asks where the file goes and
//! leaves the recorder waiting; the next run that starts is the one that is
//! recorded, and it binds to whichever pane that run's button belongs to.
//! The recording ends by itself when the run has come back to the frame it
//! started on - one full cycle, whether that is a loop through the slices or
//! a bounce out and back - and otherwise when the run stops, when the frame
//! limit is reached, or when the user says so.
//!
//! One image per played frame, not per repaint. The run advances on its own
//! clock - a few frames a second - while the window repaints far more often
//! than that, so the recorder waits for the run's own counter to move before
//! asking for the next picture. A frame is requested on one pass and
//! collected on the next, because that is when egui delivers it, and the
//! run's clock is held meanwhile so no played frame goes unrecorded.

use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result};

use super::*;

/// What a recording is written as.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) enum RecFormat {
    /// One animated GIF, timed at the run's own rate.
    #[default]
    Gif,
    /// A numbered PNG per frame, in a folder of their own: full colour, and
    /// the raw material for whatever video tool the user already has.
    Pngs,
}

impl RecFormat {
    pub(super) fn label(self) -> &'static str {
        match self {
            RecFormat::Gif => "GIF",
            RecFormat::Pngs => "PNG frames",
        }
    }

    pub(super) fn hint(self) -> &'static str {
        match self {
            RecFormat::Gif => {
                "One animated GIF at the run's own frame rate, looping. Written here, \
                 with no other program involved - which is also why it is 256 colours \
                 a frame and large for a long run."
            }
            RecFormat::Pngs => {
                "One PNG per frame, numbered, in a folder of their own. Full colour and \
                 lossless, and what a video tool wants as input: ffmpeg, a video editor \
                 or a slide program will turn them into a film."
            }
        }
    }
}

/// The run a recording has latched onto, once one has started.
pub(super) struct Bound {
    /// It ends when this run ends.
    pub(super) target: play::PlayTarget,
    /// The pane, in physical pixels of the window - fixed when the run is
    /// picked up, so every frame of a GIF is the same size even if the
    /// window is resized halfway through.
    pub(super) crop: [usize; 4],
    /// Frames per second the run is playing at: the GIF's frame delay.
    pub(super) fps: f32,
    /// Where the run stood when the first picture was taken. Coming back to
    /// it is a completed cycle, and that is where the recording ends.
    pub(super) start: Option<usize>,
    /// The run has moved off that frame, so the next time it is there again
    /// is a return rather than the frame it began on.
    pub(super) left: bool,
}

/// A recording: armed and waiting for a run, or taking one.
pub(super) struct Recording {
    pub(super) format: RecFormat,
    /// Where it goes: the GIF file, or the folder the PNGs are numbered into.
    pub(super) dest: PathBuf,
    /// `None` while it is armed and nothing is playing yet.
    pub(super) bound: Option<Bound>,
    pub(super) frames: Vec<egui::ColorImage>,
    pub(super) max_frames: usize,
    /// A picture has been asked for and has not come back yet.
    pub(super) pending: bool,
    /// The run's frame counter at the last picture taken, so one played
    /// frame yields one recorded frame however often the window repaints.
    pub(super) at: u64,
}

impl Recording {
    /// Bytes the frames held so far take: shown while recording, because a
    /// long run at a large pane size adds up fast.
    pub(super) fn bytes(&self) -> usize {
        self.frames
            .iter()
            .map(|f| f.pixels.len() * std::mem::size_of::<egui::Color32>())
            .sum()
    }
}

/// Cut `crop` (physical pixels) out of a screenshot.
///
/// Returns `None` when the rectangle no longer fits the window - a pane that
/// has been resized or scrolled out from under the recording. The frame is
/// skipped rather than written at another size, because a GIF whose frames
/// disagree about their size is not a GIF.
pub(super) fn cut(shot: &egui::ColorImage, crop: [usize; 4]) -> Option<egui::ColorImage> {
    let [x, y, w, h] = crop;
    if w == 0 || h == 0 || x + w > shot.size[0] || y + h > shot.size[1] {
        return None;
    }
    let mut pixels = Vec::with_capacity(w * h);
    for row in y..y + h {
        let from = row * shot.size[0] + x;
        pixels.extend_from_slice(&shot.pixels[from..from + w]);
    }
    Some(egui::ColorImage::new([w, h], pixels))
}

/// One frame as RGBA bytes, the way both encoders want it.
fn rgba(frame: &egui::ColorImage) -> Vec<u8> {
    let mut out = Vec::with_capacity(frame.pixels.len() * 4);
    for p in &frame.pixels {
        let [r, g, b, a] = p.to_array();
        out.extend_from_slice(&[r, g, b, a]);
    }
    out
}

/// Write an animated GIF of `frames`, `fps` frames a second, looping.
pub(super) fn write_gif(path: &Path, frames: &[egui::ColorImage], fps: f32) -> Result<()> {
    use image::codecs::gif::{GifEncoder, Repeat};
    let first = frames.first().context("nothing was recorded")?;
    let (w, h) = (first.size[0] as u32, first.size[1] as u32);
    let file = std::fs::File::create(path).with_context(|| format!("create {}", path.display()))?;
    let mut enc = GifEncoder::new(std::io::BufWriter::new(file));
    enc.set_repeat(Repeat::Infinite)
        .context("set the GIF to loop")?;
    // A GIF's delay is stored in hundredths of a second, so the rate it can
    // really carry is coarse; 2 cs (50 fps) is as fast as it goes and 1 s is
    // where a slow run is pinned.
    let delay_ms = (1000.0 / fps.clamp(1.0, 50.0)).clamp(20.0, 1000.0);
    let delay =
        image::Delay::from_saturating_duration(std::time::Duration::from_millis(delay_ms as u64));
    for f in frames {
        if f.size[0] as u32 != w || f.size[1] as u32 != h {
            continue;
        }
        let buf = image::RgbaImage::from_raw(w, h, rgba(f)).context("frame to image")?;
        enc.encode_frame(image::Frame::from_parts(buf, 0, 0, delay))
            .with_context(|| format!("write a frame of {}", path.display()))?;
    }
    Ok(())
}

/// Write `frames` as `frame_0001.png`, `frame_0002.png`, ... into `dir`.
pub(super) fn write_pngs(dir: &Path, frames: &[egui::ColorImage]) -> Result<usize> {
    std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    let mut n = 0;
    for (i, f) in frames.iter().enumerate() {
        let (w, h) = (f.size[0] as u32, f.size[1] as u32);
        let buf = image::RgbaImage::from_raw(w, h, rgba(f)).context("frame to image")?;
        let path = dir.join(format!("frame_{:04}.png", i + 1));
        buf.save(&path)
            .with_context(|| format!("write {}", path.display()))?;
        n += 1;
    }
    Ok(n)
}

impl ViewerApp {
    /// Remember where a pane was drawn this pass, so a recording started
    /// afterwards knows what to cut out of the window.
    pub(super) fn note_pane_rect(&mut self, slot: usize, kind: PaneKind, rect: Rect) {
        if let Some(e) = self
            .pane_rects
            .iter_mut()
            .find(|(s, k, _)| *s == slot && *k == kind)
        {
            e.2 = rect;
            return;
        }
        self.pane_rects.push((slot, kind, rect));
    }

    /// The pane a run is playing in: the one its button was pressed on when
    /// that is known, and otherwise the first pane of that workspace's row.
    pub(super) fn playing_pane(&self, target: play::PlayTarget) -> Option<(usize, PaneKind)> {
        let slot = target.slot();
        if let Some(p) = self.play.from_pane.filter(|(s, _)| *s == slot) {
            return Some(p);
        }
        match target {
            play::PlayTarget::Slices { slot, view } => self.slots[slot]
                .views
                .get(view)
                .map(|v| (slot, PaneKind::Plane(v.plane))),
            _ => self.row_panes(slot).first().map(|k| (slot, *k)),
        }
    }

    /// The rate the run being recorded advances at.
    fn record_fps(&self, target: play::PlayTarget) -> f32 {
        match target {
            play::PlayTarget::Slices { .. } => self.play.slice_fps,
            play::PlayTarget::Phases { .. } => self.play.phase_fps,
            play::PlayTarget::DoseLog { .. } => self.play.log_fps,
        }
    }

    /// Where a run stands, in frames of its own range: the slice, the phase,
    /// or the step of the log. Coming back to it is how a recording knows a
    /// cycle is complete.
    fn play_position(&self, target: play::PlayTarget) -> Option<usize> {
        match target {
            play::PlayTarget::Slices { slot, view } => {
                self.slots[slot].views.get(view).map(|v| v.slice)
            }
            play::PlayTarget::Phases { slot } => self.current_phase(slot).map(|(at, _)| at),
            play::PlayTarget::DoseLog { .. } => self.dose_est.cursor,
        }
    }

    /// The pane of a run, in physical pixels of the window.
    fn run_crop(&self, target: play::PlayTarget, ppp: f32) -> Option<[usize; 4]> {
        let pane = self.playing_pane(target)?;
        let rect = self
            .pane_rects
            .iter()
            .find(|(s, k, _)| (*s, *k) == pane)
            .map(|(_, _, r)| *r)?;
        Some([
            (rect.left() * ppp).round().max(0.0) as usize,
            (rect.top() * ppp).round().max(0.0) as usize,
            (rect.width() * ppp).round().max(1.0) as usize,
            (rect.height() * ppp).round().max(1.0) as usize,
        ])
    }

    /// Arm a recording: ask where it goes, and wait for a run.
    ///
    /// Nothing has to be playing. The next run to start is the one that is
    /// taken, which is also what decides the pane - the button that starts
    /// it belongs to one.
    pub(super) fn start_recording(&mut self, _ctx: &egui::Context) {
        let arm = |app: &mut ViewerApp, dest: PathBuf| {
            app.rec_status = None;
            app.rec = Some(Recording {
                format: app.rec_format,
                dest,
                bound: None,
                frames: Vec::new(),
                max_frames: app.rec_max.max(1),
                pending: false,
                at: u64::MAX,
            });
        };
        match self.rec_format {
            RecFormat::Gif => self.ask_save("Save the recording", "playback.gif", None, None, arm),
            RecFormat::Pngs => self.ask_folder("Folder for the recorded frames", arm),
        }
    }

    /// Latch onto a run when one starts, collect the picture that was asked
    /// for, ask for the next, and write the file when the cycle is done.
    pub(super) fn record_tick(
        &mut self,
        ctx: &egui::Context,
        shot: Option<&std::sync::Arc<egui::ColorImage>>,
    ) {
        if self.rec.is_none() {
            return;
        }
        // Everything the decision needs, read before the recording is held
        // mutably: the borrow checker, and the fact that finishing wants the
        // whole of `self` back.
        let ppp = ctx.pixels_per_point();
        let running = self.play.running.map(|r| r.target);
        let pos = running.and_then(|t| self.play_position(t));
        let crop = running.and_then(|t| self.run_crop(t, ppp));
        let fps = running.map(|t| self.record_fps(t));
        let frame_no = self.play.frame_no;

        let mut finish: Option<bool> = None;
        let mut want_shot = false;
        {
            let Some(rec) = self.rec.as_mut() else { return };
            if let Some(shot) = shot {
                if rec.pending {
                    rec.pending = false;
                    if let Some(b) = rec.bound.as_mut() {
                        if let Some(f) = cut(shot, b.crop) {
                            rec.frames.push(f);
                            // The first picture fixes where the cycle began;
                            // after that, leaving that frame and returning to
                            // it is one full cycle - a loop through the range
                            // or a bounce out and back, either way.
                            match (b.start, pos) {
                                (None, p) => b.start = p,
                                (Some(s), Some(p)) if p != s => b.left = true,
                                (Some(s), Some(p)) if p == s && b.left => {
                                    // Back where it began: the cycle is
                                    // complete. This frame is the first one
                                    // over again, and a GIF that loops would
                                    // show it twice in a row, so it goes.
                                    rec.frames.pop();
                                    finish = Some(false);
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
            match rec.bound.as_ref() {
                // Armed: the next run that starts is the one to take.
                None => {
                    if let (Some(target), Some(crop), Some(fps)) = (running, crop, fps) {
                        rec.bound = Some(Bound {
                            target,
                            crop,
                            fps,
                            start: None,
                            left: false,
                        });
                        rec.at = u64::MAX;
                    }
                }
                Some(b) => {
                    if running != Some(b.target) {
                        // The run it was taking has ended.
                        finish = finish.or(Some(false));
                    } else if rec.frames.len() >= rec.max_frames {
                        finish = Some(true);
                    }
                }
            }
            if finish.is_none() && rec.bound.is_some() && !rec.pending && rec.at != frame_no {
                rec.at = frame_no;
                rec.pending = true;
                want_shot = true;
            }
        }
        if let Some(capped) = finish {
            self.finish_recording(capped);
            return;
        }
        if want_shot {
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
            ctx.request_repaint();
        }
    }

    /// Write what was recorded and say where it went.
    pub(super) fn finish_recording(&mut self, capped: bool) {
        let Some(rec) = self.rec.take() else { return };
        if rec.frames.is_empty() {
            self.rec_status = Some(match rec.bound {
                Some(_) => "Nothing was recorded: the run ended first.".into(),
                None => "Recording cancelled before a run started.".to_string(),
            });
            return;
        }
        let fps = rec.bound.as_ref().map(|b| b.fps).unwrap_or(10.0);
        let n = rec.frames.len();
        let capped = if capped {
            " (the frame limit was reached)"
        } else {
            ""
        };
        let res = match rec.format {
            RecFormat::Gif => write_gif(&rec.dest, &rec.frames, fps).map(|()| n),
            RecFormat::Pngs => write_pngs(&rec.dest, &rec.frames),
        };
        self.rec_status = Some(match res {
            Ok(n) => format!("{n} frames written to {}{capped}", rec.dest.display()),
            Err(e) => format!("Could not write the recording: {e:#}"),
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
    fn a_cut_takes_the_rectangle_it_was_given() {
        let shot = image(10, 8);
        let c = cut(&shot, [2, 3, 4, 2]).expect("fits");
        assert_eq!(c.size, [4, 2]);
        // Row 3 of the source, starting at column 2.
        assert_eq!(c.pixels[0], shot.pixels[3 * 10 + 2]);
        assert_eq!(c.pixels[4], shot.pixels[4 * 10 + 2]);
    }

    #[test]
    fn a_cut_that_hangs_over_the_edge_is_refused() {
        let shot = image(10, 8);
        assert!(cut(&shot, [8, 0, 4, 2]).is_none(), "past the right edge");
        assert!(cut(&shot, [0, 7, 2, 4]).is_none(), "past the bottom");
        assert!(cut(&shot, [0, 0, 0, 2]).is_none(), "empty");
    }

    #[test]
    fn a_gif_is_written_and_reads_back_as_one() {
        let dir = std::env::temp_dir().join("rds_rec_gif_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("run.gif");
        let frames: Vec<egui::ColorImage> = (0..3).map(|_| image(8, 6)).collect();
        write_gif(&path, &frames, 10.0).expect("write the gif");
        let bytes = std::fs::read(&path).expect("read it back");
        assert!(bytes.len() > 6, "the file has content");
        assert_eq!(&bytes[..3], b"GIF", "it really is a GIF");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn every_frame_becomes_its_own_numbered_png() {
        let dir = std::env::temp_dir().join("rds_rec_png_test");
        let _ = std::fs::remove_dir_all(&dir);
        let frames: Vec<egui::ColorImage> = (0..4).map(|_| image(5, 5)).collect();
        let n = write_pngs(&dir, &frames).expect("write the frames");
        assert_eq!(n, 4);
        for i in 1..=4 {
            assert!(
                dir.join(format!("frame_{i:04}.png")).is_file(),
                "frame {i} is there under its number"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
