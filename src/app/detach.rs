//! Every secondary window is a window of the operating system.
//!
//! The archive, the model manager, the DRR, the 3D scenes, the segmentation
//! and motion tools, the export and anonymizer dialogs — each is drawn
//! through [`tool_window`], which puts it in an *immediate viewport*: a real
//! top-level window with its own title bar, task-bar entry and place on
//! whichever monitor the user drags it to. Nothing floats inside the main
//! window any more, so the six viewports always keep the whole of it, and a
//! two- or three-screen reading room can lay the tools out as it likes.
//!
//! Two details make this behave:
//!
//! * **Position and size are set once, when the window is created.** egui
//!   diffs the [`egui::ViewportBuilder`] it is handed against the one it
//!   stored and sends a move / resize command for every field that differs.
//!   Feeding the window's own position back into the builder every frame is
//!   therefore a loop: the user drags, the next frame commands the window
//!   back, and it shakes — sometimes long after the mouse is released. The
//!   geometry is remembered for the session and applied only on the pass
//!   that opens the window; after that the window belongs to the user.
//! * **Titles follow one pattern**, [`window_title`]: `Rust DICOM Station:`
//!   and the tool's name, so a task bar full of them reads as one program.
//!
//! If a backend cannot make native windows at all (there is none such on the
//! desktop, but egui allows it), egui falls back on its own and draws the
//! contents in a window inside the main one; nothing here has to care.

/// egui-memory key prefix of one window's remembered geometry.
const GEOM: &str = "tool_window_geometry";
/// egui-memory key prefix of the pass this window was last drawn in.
const SEEN: &str = "tool_window_last_pass";

/// A window with no height of its own opens this tall.
const DEFAULT_TALL: f32 = 620.0;
/// No tool window opens (or can be resized) smaller than this.
const MIN_SIZE: [f32; 2] = [320.0, 200.0];

/// How big the window opens, and whether its contents scroll.
#[derive(Clone, Copy)]
pub(super) struct WinOpts {
    /// Size the window opens at the first time. A height of `0.0` means
    /// [`DEFAULT_TALL`].
    pub size: [f32; 2],
    /// Scroll the contents when the user makes the window smaller than they
    /// are. Off for the windows that answer the mouse wheel themselves (the
    /// image and 3-D views), where a scroll area would fight them for it.
    pub scroll: bool,
}

impl Default for WinOpts {
    fn default() -> Self {
        Self {
            size: [420.0, 0.0],
            scroll: true,
        }
    }
}

impl WinOpts {
    pub(super) fn width(w: f32) -> Self {
        Self {
            size: [w, 0.0],
            ..Self::default()
        }
    }

    pub(super) fn size(w: f32, h: f32) -> Self {
        Self {
            size: [w, h],
            ..Self::default()
        }
    }

    pub(super) fn no_scroll(mut self) -> Self {
        self.scroll = false;
        self
    }
}

/// One window's last geometry, so it reopens where it was left — on the
/// monitor it was left on.
#[derive(Clone, Copy, Default)]
struct Geometry {
    pos: Option<[f32; 2]>,
    size: Option<[f32; 2]>,
}

fn geometry(ctx: &egui::Context, id: &str) -> Geometry {
    ctx.data(|d| d.get_temp::<Geometry>(egui::Id::new((GEOM, id))))
        .unwrap_or_default()
}

fn set_geometry(ctx: &egui::Context, id: &str, g: Geometry) {
    ctx.data_mut(|d| d.insert_temp(egui::Id::new((GEOM, id)), g));
}

/// The title of every window of this application: the program, then what
/// this particular window is.
///
/// The tool names carry a glyph for the menus (`📦 Downloaded models`); a
/// title bar has the program name in front of it already, so the glyph is
/// dropped here and the name reads plainly.
pub(super) fn window_title(name: &str) -> String {
    let plain = name
        .trim_start_matches(|c: char| !c.is_alphanumeric())
        .trim_start();
    format!("Rust DICOM Station: {plain}")
}

/// Show one tool window in its own window of the operating system and run
/// `contents` inside it.
///
/// `id` must be stable and unique — it keys the native window and its
/// remembered geometry. `open` is cleared when the user closes the window.
pub(super) fn tool_window<R>(
    ctx: &egui::Context,
    id: &str,
    title: impl Into<String>,
    open: &mut bool,
    opts: WinOpts,
    contents: impl FnOnce(&mut egui::Ui) -> R,
) -> Option<R> {
    if !*open {
        return None;
    }
    // Is this the pass that opens the window, or one of the passes that keep
    // it open? Every open window is drawn once per pass of the main window
    // (an immediate viewport repaints with its parent), so a gap in the pass
    // numbers means this window was closed in between and is being created
    // again. The caller usually returns early while its window is shut, so
    // this — not a flag cleared on closing — is what can tell the two apart.
    let seen_key = egui::Id::new((SEEN, id));
    let pass = ctx.cumulative_pass_nr();
    let fresh = ctx
        .data(|d| d.get_temp::<u64>(seen_key))
        .is_none_or(|last| pass.saturating_sub(last) > 1);
    ctx.data_mut(|d| d.insert_temp(seen_key, pass));

    let title = window_title(&title.into());
    let mut builder = egui::ViewportBuilder::default().with_title(&title);
    if fresh {
        // Only on the pass that creates the window. Repeating this every
        // frame is what makes a dragged window shake.
        let geom = geometry(ctx, id);
        let size = geom.size.unwrap_or([
            opts.size[0].max(MIN_SIZE[0]),
            if opts.size[1] > 0.0 {
                opts.size[1]
            } else {
                DEFAULT_TALL
            },
        ]);
        builder = builder
            .with_inner_size(size)
            .with_min_inner_size(MIN_SIZE)
            .with_resizable(true);
        if let Some(pos) = geom.pos {
            builder = builder.with_position(pos);
        }
    }

    // `FnOnce` contents, called from egui's `FnMut` callback: the option
    // hands it over exactly once, on the pass that actually draws.
    let mut contents = Some(contents);
    let mut ret = None;
    let mut close = false;
    let mut seen = None;
    ctx.show_viewport_immediate(
        egui::ViewportId::from_hash_of(("tool_window", id)),
        builder,
        |ui, _class| {
            if opts.scroll {
                egui::ScrollArea::both()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        if let Some(c) = contents.take() {
                            ret = Some(c(ui));
                        }
                    });
            } else if let Some(c) = contents.take() {
                ret = Some(c(ui));
            }
            ui.ctx().input(|i| {
                let info = i.viewport();
                if info.close_requested() {
                    close = true;
                }
                // Where the user has put it — including which monitor, since
                // the position is in desktop coordinates. Read only: it is
                // used the next time this window opens, never fed back into
                // the builder of the window it came from.
                if let Some(outer) = info.outer_rect {
                    seen = Some(Geometry {
                        pos: Some([outer.min.x, outer.min.y]),
                        size: info
                            .inner_rect
                            .map(|r| [r.width(), r.height()])
                            .or(Some([outer.width(), outer.height()])),
                    });
                }
            });
        },
    );
    if let Some(g) = seen {
        set_geometry(ctx, id, g);
    }
    if close {
        *open = false;
    }
    ret
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_window_is_titled_the_same_way() {
        assert_eq!(
            window_title("📦 Downloaded models"),
            "Rust DICOM Station: Downloaded models"
        );
        assert_eq!(
            window_title("🏥 PACS — patient archive"),
            "Rust DICOM Station: PACS — patient archive"
        );
        assert_eq!(
            window_title("⇄ Propagate structures — A ▶ B"),
            "Rust DICOM Station: Propagate structures — A ▶ B",
            "only the leading glyph goes; the ones inside the name stay"
        );
        assert_eq!(
            window_title("3D structures — dataset A"),
            "Rust DICOM Station: 3D structures — dataset A",
            "a name that starts with a digit is untouched"
        );
    }
}
