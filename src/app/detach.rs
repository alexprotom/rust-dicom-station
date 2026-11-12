//! Every secondary window is a window of the operating system.
//!
//! The archive, the model manager, the DRR, the 3D scenes, the segmentation
//! and motion tools, the export and anonymizer dialogs - each is drawn
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
//!   back, and it shakes - sometimes long after the mouse is released. The
//!   geometry is remembered for the session and applied only on the pass
//!   that opens the window; after that the window belongs to the user.
//! * **Titles follow one pattern**, [`window_title`]: `Rust DICOM Station:`
//!   and the tool's name, so a task bar full of them reads as one program.
//! * **Opening an open window brings it to the front.** The button that
//!   opens a tool also calls [`ViewerApp::raise_window`](super::ViewerApp::raise_window)
//!   with its id; a window that is already there, hidden behind the main
//!   one or minimized, is restored and focused instead of staying where
//!   the user cannot see it. A window being created gets the focus anyway.
//! * **A window fits what it shows.** A scrolling tool window measures its
//!   contents every pass and takes their height - smaller when rows fold,
//!   taller when they unfold - and widens when something in it is wider
//!   than the window, never past the monitor it is on (and moved up when
//!   growing would take its bottom off the screen). Lists that would rather
//!   scroll than end at the window's edge size themselves by
//!   [`room_height`], the height the window could still grow by included.
//!   Resizing the window by hand ends the fitting until it is opened again:
//!   from then on its size is the user's.
//!
//! If a backend cannot make native windows at all - Android is one, with
//! its single window per process - the tool is drawn as an [`egui::Window`]
//! inside the main one instead ([`embedded_window`]), with egui's own title
//! bar and close button. egui would embed the viewport by itself, but
//! without a way to close it: the viewport path learns of a close from the
//! native window's button, which the embedded copy does not have. The
//! desktop never takes that branch, since `eframe` reports native windows
//! there.

/// egui-memory key prefix of one window's remembered geometry.
const GEOM: &str = "tool_window_geometry";
/// egui-memory key prefix of the pass this window was last drawn in.
const SEEN: &str = "tool_window_last_pass";
/// egui-memory key prefix of the *Keep on top* switch of one window. It
/// starts off every time a window opens and is not part of the settings.
const ON_TOP: &str = "tool_window_on_top";
/// egui-memory key prefix of a request to bring one window to the front,
/// set by [`raise_window`](super::ViewerApp::raise_window) and taken by the
/// window's next draw.
const RAISE: &str = "tool_window_raise";
/// egui-memory key prefix of one window's fitting ([`Fit`]).
const FIT: &str = "tool_window_fit";
/// egui-memory key prefix of how much taller the window being drawn could
/// still get, by viewport: read by [`room_height`].
const ROOM: &str = "tool_window_room";
/// What a fitted window leaves of the monitor: a margin at the sides, the
/// task bar and the title bar in the height.
const SCREEN_MARGIN: [f32; 2] = [40.0, 90.0];

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

/// One window's last geometry, so it reopens where it was left - on the
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

/// One window's fitting to its contents, kept from pass to pass.
#[derive(Clone, Copy)]
struct Fit {
    /// Still fitting: no resize by hand since the window opened.
    auto: bool,
    /// When the last size was asked for (egui time, seconds): a change of
    /// size soon after is that answer, not the user's hand.
    asked_at: f64,
    /// The inner size seen on the pass before.
    last: Option<egui::Vec2>,
}

impl Default for Fit {
    fn default() -> Self {
        Fit {
            auto: true,
            asked_at: f64::NEG_INFINITY,
            last: None,
        }
    }
}

/// The height a list in a tool window may take: what is left below it, plus
/// what the window could still grow by on its monitor while it fits its
/// contents. A list sized by this grows the window until the screen ends,
/// and only then scrolls. Outside a fitting window it is the height left.
pub(super) fn room_height(ui: &egui::Ui) -> f32 {
    let extra = ui
        .ctx()
        .data(|d| d.get_temp::<f32>(egui::Id::new((ROOM, ui.ctx().viewport_id()))))
        .unwrap_or(0.0);
    ui.available_height() + extra.max(0.0)
}

/// The largest inner size a fitted window may take on its monitor.
fn screen_limit(info: &egui::ViewportInfo) -> Option<egui::Vec2> {
    let m = info.monitor_size?;
    Some(egui::vec2(
        (m.x - SCREEN_MARGIN[0]).max(MIN_SIZE[0]),
        (m.y - SCREEN_MARGIN[1]).max(MIN_SIZE[1]),
    ))
}

/// Ask for the size the contents measured this pass, unless the user has
/// taken the size over: `cur` is the window's inner size, `out` what the
/// scroll area holding the contents reported.
fn fit_to_contents<R>(
    ctx: &egui::Context,
    fit: &mut Fit,
    info: &egui::ViewportInfo,
    cur: egui::Vec2,
    out: &egui::scroll_area::ScrollAreaOutput<R>,
) {
    let now = ctx.input(|i| i.time);
    // A size that changed long after the last one asked for is the user's
    // hand (or a maximize): the window is theirs from now on.
    if let Some(prev) = fit.last {
        if (cur - prev).abs().max_elem() > 2.0 && now - fit.asked_at > 0.6 {
            fit.auto = false;
        }
    }
    fit.last = Some(cur);
    if !fit.auto || info.maximized == Some(true) || info.minimized == Some(true) {
        return;
    }
    // What the window spends outside the scroll area (the *Keep on top*
    // bar, the margins) plus what the contents measure. Height both ways;
    // width only up, since text wraps to whatever width it is given and
    // would shrink the window step by step.
    let overhead = cur - out.inner_rect.size();
    let mut want = egui::vec2(
        cur.x.max(overhead.x + out.content_size.x),
        overhead.y + out.content_size.y,
    )
    .max(egui::vec2(MIN_SIZE[0], MIN_SIZE[1]));
    if let Some(l) = screen_limit(info) {
        want = want.min(l.max(egui::vec2(MIN_SIZE[0], MIN_SIZE[1])));
    }
    let want = want.round();
    if (want - cur).abs().max_elem() <= 2.0 {
        return;
    }
    ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(want));
    fit.asked_at = now;
    // Growing past the bottom of the screen: up, as far as it needs to.
    // Only for a window whose top is on a monitor that starts at the top
    // of the desktop (side by side, the usual case); anything else is left
    // where the user put it.
    if let (Some(outer), Some(m)) = (info.outer_rect, info.monitor_size) {
        let chrome = (outer.height() - cur.y).max(0.0);
        let screen_bottom = m.y - (SCREEN_MARGIN[1] - chrome).max(40.0);
        let bottom = outer.min.y + chrome + want.y;
        if want.y > cur.y && (0.0..m.y).contains(&outer.min.y) && bottom > screen_bottom {
            let y = (screen_bottom - chrome - want.y).max(0.0);
            ctx.send_viewport_cmd(egui::ViewportCommand::OuterPosition(egui::pos2(
                outer.min.x,
                y,
            )));
        }
    }
}

impl super::ViewerApp {
    /// Bring the tool window `id` (the id it passes to [`tool_window`]) to
    /// the front on its next draw. Called next to whatever opens it, so a
    /// click on a button whose window is already open shows that window
    /// rather than doing nothing visible. Harmless for a window that is shut
    /// or being created.
    pub(super) fn raise_window(&mut self, id: &'static str) {
        self.raise_windows.push(id);
    }

    /// Hand the pending [`raise_window`](Self::raise_window) requests to
    /// the windows; once a pass, before they are drawn.
    pub(super) fn pass_raise_requests(&mut self, ctx: &egui::Context) {
        for id in self.raise_windows.drain(..) {
            ctx.data_mut(|d| d.insert_temp(egui::Id::new((RAISE, id)), true));
        }
    }
}

/// Whether the window `id` was asked to come to the front, clearing the ask.
fn take_raise(ctx: &egui::Context, id: &str) -> bool {
    ctx.data_mut(|d| d.remove_temp::<bool>(egui::Id::new((RAISE, id))))
        .unwrap_or(false)
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
/// `id` must be stable and unique - it keys the native window and its
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
    // `FnOnce` contents, called from egui's `FnMut` callbacks: the option
    // hands them over exactly once, on the pass that actually draws. The
    // window itself is [`tool_window_body`], compiled once for every tool
    // rather than once per caller's closure type.
    let mut contents = Some(contents);
    let mut ret = None;
    tool_window_body(ctx, id, title.into(), open, opts, &mut |ui| {
        if let Some(c) = contents.take() {
            ret = Some(c(ui));
        }
    });
    ret
}

/// [`tool_window`] with its contents as a plain `dyn` call.
fn tool_window_body(
    ctx: &egui::Context,
    id: &str,
    title: String,
    open: &mut bool,
    opts: WinOpts,
    contents: &mut dyn FnMut(&mut egui::Ui),
) {
    let title = window_title(&title);
    if ctx.embed_viewports() {
        embedded_window(ctx, id, &title, open, opts, contents);
        return;
    }
    // Is this the pass that opens the window, or one of the passes that keep
    // it open? Every open window is drawn once per pass of the main window
    // (an immediate viewport repaints with its parent), so a gap in the pass
    // numbers means this window was closed in between and is being created
    // again. The caller usually returns early while its window is shut, so
    // this - not a flag cleared on closing - is what can tell the two apart.
    let seen_key = egui::Id::new((SEEN, id));
    let pass = ctx.cumulative_pass_nr();
    let fresh = ctx
        .data(|d| d.get_temp::<u64>(seen_key))
        .is_none_or(|last| pass.saturating_sub(last) > 1);
    ctx.data_mut(|d| d.insert_temp(seen_key, pass));
    let on_top_key = egui::Id::new((ON_TOP, id));
    let fit_key = egui::Id::new((FIT, id));
    if fresh {
        ctx.data_mut(|d| {
            d.insert_temp(on_top_key, false);
            d.insert_temp(fit_key, Fit::default());
        });
    }
    // Taken either way, so an ask made while the window is being created
    // does not linger; a new window is focused by the system already.
    let raise = take_raise(ctx, id) && !fresh;

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

    let mut close = false;
    let mut seen = None;
    ctx.show_viewport_immediate(
        egui::ViewportId::from_hash_of(("tool_window", id)),
        builder,
        |ui, _class| {
            if raise {
                // Restored first: a minimized window cannot take the focus.
                if ui.ctx().input(|i| i.viewport().minimized == Some(true)) {
                    ui.ctx()
                        .send_viewport_cmd(egui::ViewportCommand::Minimized(false));
                }
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::Focus);
            }
            // *Keep on top*: the window stays over the main one (and every
            // other) while the switch is on. The switch is per window, off
            // when the window opens, and never written to the settings.
            let mut on_top = ui
                .ctx()
                .data(|d| d.get_temp::<bool>(on_top_key))
                .unwrap_or(false);
            egui::Panel::top(egui::Id::new(("tool_window_bar", id)))
                .show_separator_line(false)
                .show(ui, |ui| {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .add(egui::Button::new("📌 Keep on top").small().selected(on_top))
                            .on_hover_text(
                                "Keep this window above the main window while it is open. \
                                 Off whenever a window is opened; not remembered.",
                            )
                            .clicked()
                        {
                            on_top = !on_top;
                            ui.ctx().data_mut(|d| d.insert_temp(on_top_key, on_top));
                            ui.ctx()
                                .send_viewport_cmd(egui::ViewportCommand::WindowLevel(if on_top {
                                    egui::WindowLevel::AlwaysOnTop
                                } else {
                                    egui::WindowLevel::Normal
                                }));
                        }
                    });
                });
            if opts.scroll {
                let info = ui.ctx().input(|i| i.viewport().clone());
                let mut fit = ui
                    .ctx()
                    .data(|d| d.get_temp::<Fit>(fit_key))
                    .unwrap_or_default();
                let cur = info.inner_rect.map(|r| r.size());
                // What the lists may count on, set before they are drawn.
                let room = match (fit.auto, cur, screen_limit(&info)) {
                    (true, Some(c), Some(l)) => (l.y - c.y).max(0.0),
                    _ => 0.0,
                };
                let vp = ui.ctx().viewport_id();
                ui.ctx()
                    .data_mut(|d| d.insert_temp(egui::Id::new((ROOM, vp)), room));
                let out = egui::ScrollArea::both()
                    .auto_shrink([false, false])
                    .show(ui, |ui| contents(ui));
                if let Some(c) = cur {
                    fit_to_contents(ui.ctx(), &mut fit, &info, c, &out);
                }
                ui.ctx().data_mut(|d| d.insert_temp(fit_key, fit));
            } else {
                contents(ui);
            }
            ui.ctx().input(|i| {
                let info = i.viewport();
                if info.close_requested() {
                    close = true;
                }
                // Where the user has put it - including which monitor, since
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
}

/// The tool window where the backend has no native windows: an
/// [`egui::Window`] in the main one, opened at the size the tool asks for,
/// resizable, closed by its own title-bar button. egui remembers where the
/// user leaves it, so nothing about geometry has to be kept here; *Keep on
/// top* does not exist, there being nothing to be on top of.
fn embedded_window(
    ctx: &egui::Context,
    id: &str,
    title: &str,
    open: &mut bool,
    opts: WinOpts,
    contents: &mut dyn FnMut(&mut egui::Ui),
) {
    let size = [
        opts.size[0].max(MIN_SIZE[0]),
        if opts.size[1] > 0.0 {
            opts.size[1]
        } else {
            DEFAULT_TALL
        },
    ];
    let win_id = egui::Id::new(("tool_window", id));
    if take_raise(ctx, id) {
        // An `egui::Window` lives on the middle layer under its own id.
        ctx.move_to_top(egui::LayerId::new(egui::Order::Middle, win_id));
    }
    egui::Window::new(title)
        .id(win_id)
        .open(open)
        .collapsible(false)
        .resizable(true)
        .default_size(size)
        .min_size(MIN_SIZE)
        .show(ctx, |ui| {
            if opts.scroll {
                egui::ScrollArea::both()
                    .auto_shrink([false, false])
                    .show(ui, |ui| contents(ui));
            } else {
                contents(ui);
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A headless context has no native windows, which is the Android
    /// case: the window must draw, hand its contents a `Ui`, and close
    /// through the flag rather than stay open for good.
    #[test]
    fn without_native_windows_the_tool_is_a_closable_window_inside_the_main_one() {
        let ctx = egui::Context::default();
        assert!(ctx.embed_viewports());
        let mut open = true;
        let mut drawn = 0;
        let mut out = ctx.run_ui(egui::RawInput::default(), |ui| {
            let r = tool_window(
                ui.ctx(),
                "t",
                "📦 Test",
                &mut open,
                WinOpts::default(),
                |ui| {
                    drawn += 1;
                    ui.label("contents");
                    7
                },
            );
            assert_eq!(r, Some(7));
        });
        // No renderer here to take the font texture; dropping it unhandled
        // is what epaint refuses.
        out.textures_delta.clear();
        assert_eq!(drawn, 1);
        assert!(open);
        // A closed window draws nothing and reports nothing.
        open = false;
        let mut out = ctx.run_ui(egui::RawInput::default(), |ui| {
            let r = tool_window(
                ui.ctx(),
                "t",
                "📦 Test",
                &mut open,
                WinOpts::default(),
                |ui| {
                    drawn += 1;
                    ui.label("contents");
                    7
                },
            );
            assert_eq!(r, None);
        });
        out.textures_delta.clear();
        assert_eq!(drawn, 1);
    }

    #[test]
    fn every_window_is_titled_the_same_way() {
        assert_eq!(
            window_title("📦 Downloaded models"),
            "Rust DICOM Station: Downloaded models"
        );
        assert_eq!(
            window_title("🏥 Picture Archiving and Communication System (PACS)"),
            "Rust DICOM Station: Picture Archiving and Communication System (PACS)"
        );
        assert_eq!(
            window_title("⇄ Propagate structures - A ▶ B"),
            "Rust DICOM Station: Propagate structures - A ▶ B",
            "only the leading glyph goes; the ones inside the name stay"
        );
        assert_eq!(
            window_title("3D structures - workspace A"),
            "Rust DICOM Station: 3D structures - workspace A",
            "a name that starts with a digit is untouched"
        );
    }
}
