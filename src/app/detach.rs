//! Tool windows that can step outside the main window.
//!
//! Every secondary window — the archive, the model manager, the DRR, the 3D
//! scenes, the segmentation tools — is drawn through [`tool_window`]. Docked,
//! it is an ordinary [`egui::Window`] floating over the viewports, which is
//! where a single-screen user wants it. Detached, the same contents are drawn
//! into an *immediate viewport*: a real top-level window of the operating
//! system that can be dragged onto a second or third monitor, maximized
//! there, and left open while the main window keeps the images. Nothing about
//! the contents changes — the same closure runs in both cases — so a window
//! can be moved back and forth mid-run.
//!
//! Three things make this work in practice:
//!
//! * the choice is per window and remembered (see [`detached_ids`], which the
//!   application writes to its settings file), so a reading room that always
//!   wants the archive on the right-hand screen gets it there on every start;
//! * the size and position of a detached window are remembered for the
//!   session, so closing and reopening it puts it back on the same monitor
//!   rather than on the main one;
//! * when the backend cannot give us native windows at all, egui says so
//!   through [`egui::ViewportClass::EmbeddedWindow`] and the window simply
//!   stays inside the main one instead of vanishing.

use std::collections::BTreeSet;

/// egui-memory key of the set of detached window ids.
const DETACHED: &str = "detached_tool_windows";
/// egui-memory key prefix of one window's remembered geometry.
const GEOM: &str = "tool_window_geometry";

/// How the window looks while it is docked, and how big its own window opens.
#[derive(Clone, Copy)]
pub(super) struct WinOpts {
    /// Default outer size. A height of `0.0` means "as tall as the contents"
    /// while docked; the detached window then opens at `DEFAULT_TALL`.
    pub size: [f32; 2],
    pub resizable: bool,
    pub collapsible: bool,
    /// Docked, this window is pinned to the middle of the main window (the
    /// dialog-like tools do this so they cannot be lost behind the views).
    pub center: bool,
    /// Scroll the contents when the window is its own window and the user
    /// has made it smaller than they are. Off for the windows that answer
    /// the mouse wheel themselves (the image and 3-D views), where a scroll
    /// area would fight them for it.
    pub scroll: bool,
}

/// A native window with no height of its own opens this tall.
const DEFAULT_TALL: f32 = 620.0;

impl Default for WinOpts {
    fn default() -> Self {
        Self {
            size: [420.0, 0.0],
            resizable: true,
            collapsible: true,
            center: false,
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

    pub(super) fn resizable(mut self, yes: bool) -> Self {
        self.resizable = yes;
        self
    }

    pub(super) fn collapsible(mut self, yes: bool) -> Self {
        self.collapsible = yes;
        self
    }

    pub(super) fn centered(mut self) -> Self {
        self.center = true;
        self
    }

    pub(super) fn no_scroll(mut self) -> Self {
        self.scroll = false;
        self
    }
}

/// One detached window's last geometry, so it reopens where it was left —
/// on the monitor it was left on.
#[derive(Clone, Copy, Default)]
struct Geometry {
    pos: Option<[f32; 2]>,
    size: Option<[f32; 2]>,
}

/// The ids of the windows the user has pulled out, as the application stores
/// them between runs.
pub(super) fn detached_ids(ctx: &egui::Context) -> BTreeSet<String> {
    ctx.data(|d| d.get_temp::<BTreeSet<String>>(egui::Id::new(DETACHED)))
        .unwrap_or_default()
}

/// Seed the set from the settings file at start-up.
pub(super) fn set_detached_ids(ctx: &egui::Context, ids: BTreeSet<String>) {
    ctx.data_mut(|d| d.insert_temp(egui::Id::new(DETACHED), ids));
}

fn is_detached(ctx: &egui::Context, id: &str) -> bool {
    detached_ids(ctx).contains(id)
}

fn set_detached(ctx: &egui::Context, id: &str, yes: bool) {
    let mut ids = detached_ids(ctx);
    if yes {
        ids.insert(id.to_owned());
    } else {
        ids.remove(id);
    }
    set_detached_ids(ctx, ids);
}

fn geometry(ctx: &egui::Context, id: &str) -> Geometry {
    ctx.data(|d| d.get_temp::<Geometry>(egui::Id::new((GEOM, id))))
        .unwrap_or_default()
}

fn set_geometry(ctx: &egui::Context, id: &str, g: Geometry) {
    ctx.data_mut(|d| d.insert_temp(egui::Id::new((GEOM, id)), g));
}

/// The one-line header every tool window carries: the button that moves it
/// out of the main window and back in.
fn detach_row(ui: &mut egui::Ui, out: &mut bool) {
    ui.horizontal(|ui| {
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let (label, tip) = if *out {
                (
                    "Dock",
                    "Put this window back inside the main window.\n\
                     Closing it instead only closes the tool — it opens in its own \
                     window again next time.",
                )
            } else {
                (
                    "Detach",
                    "Give this window its own window of the operating system — \
                     drag it onto a second monitor, resize it there, and it stays \
                     open beside the images. It reopens where you left it, and the \
                     choice is remembered between runs.",
                )
            };
            if ui.small_button(label).on_hover_text(tip).clicked() {
                *out = !*out;
            }
        });
    });
    ui.separator();
}

/// Show one tool window, docked or in its own window of the operating
/// system, and run `contents` in whichever of the two it ended up in.
///
/// `id` must be stable and unique — it keys the detached-window set, the
/// remembered geometry and the native window itself. `open` is cleared when
/// the user closes the window either way.
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
    let title = title.into();
    let mut out = is_detached(ctx, id);
    let was_out = out;
    let mut ret = None;

    if out {
        let geom = geometry(ctx, id);
        let size = geom.size.unwrap_or([
            opts.size[0].max(320.0),
            if opts.size[1] > 0.0 {
                opts.size[1]
            } else {
                DEFAULT_TALL
            },
        ]);
        let mut builder = egui::ViewportBuilder::default()
            .with_title(&title)
            .with_inner_size(size);
        if let Some(pos) = geom.pos {
            builder = builder.with_position(pos);
        }
        // `FnOnce` contents, called from egui's `FnMut` callback: the option
        // hands it over exactly once, on the pass that actually draws.
        let mut contents = Some(contents);
        let mut close = false;
        let mut new_geom = None;
        ctx.show_viewport_immediate(
            egui::ViewportId::from_hash_of(("tool_window", id)),
            builder,
            |ui, class| {
                // No native windows from this backend: draw the contents in
                // the window egui made for us instead of losing them.
                if class == egui::ViewportClass::EmbeddedWindow {
                    out = false;
                }
                detach_row(ui, &mut out);
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
                    // Remember where the user put it — including which
                    // monitor, since the position is in desktop coordinates.
                    if let Some(outer) = info.outer_rect {
                        new_geom = Some(Geometry {
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
        if let Some(g) = new_geom {
            set_geometry(ctx, id, g);
        }
        if close {
            *open = false;
        }
    } else {
        let mut still_open = true;
        let mut win = egui::Window::new(&title)
            .id(egui::Id::new(id))
            .open(&mut still_open)
            .collapsible(opts.collapsible)
            .resizable(opts.resizable);
        win = if opts.size[1] > 0.0 {
            win.default_size(opts.size)
        } else {
            win.default_width(opts.size[0])
        };
        if opts.center {
            win = win.anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO);
        }
        win.show(ctx, |ui| {
            detach_row(ui, &mut out);
            ret = Some(contents(ui));
        });
        if !still_open {
            *open = false;
        }
    }

    if out != was_out {
        set_detached(ctx, id, out);
        // The window that is being left behind would otherwise keep its old
        // size and position for one more pass.
        ctx.request_repaint();
    }
    ret
}
