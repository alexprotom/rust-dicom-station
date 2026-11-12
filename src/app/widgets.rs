//! Small widgets every window and panel reaches for: a button with a
//! tooltip, a glyph button, a picker over a list.
//!
//! **Names for the widgets that show none.** A button that is one glyph, a
//! tick box with no text beside it, a colour square, a scrubber: on screen
//! the place and the tooltip say what each is, but what egui hands to
//! AccessKit - which a screen reader reads, and which the headless harness
//! (`tests/common/headless.rs`) finds widgets by - is the glyph or nothing.
//! [`Named`] gives such a widget a name there and changes nothing that is
//! drawn. The rule for new code: a widget without words takes its tooltip
//! through [`Named::tip`] instead of `on_hover_text`, which names it by the
//! tooltip's first sentence; where several on screen would then be called
//! the same (one per pane, one per row), [`Named::named`] /
//! [`Named::named_with`] say which. `tests/ui.rs` fails on a wordless
//! widget on the screens it opens.

use super::combine::ItemRef;

/// Whether a text says something in words: a glyph or an empty string does
/// not.
fn has_words(text: &str) -> bool {
    text.chars().any(char::is_alphanumeric)
}

/// The part of a tooltip that names the widget: its first sentence, or what
/// stands before a colon when that is more than one word (`Reset this view:
/// fit zoom, ...` is *Reset this view*; `New: an empty set` stays whole).
fn first_sentence(tip: &str) -> &str {
    let mut chars = tip.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        let ends = match c {
            '\n' => true,
            // A full stop, not the point in `0.5` or `e.g.`.
            '.' => chars.peek().is_none_or(|(_, next)| next.is_whitespace()),
            ':' => tip[..i].trim().contains(' '),
            _ => false,
        };
        if ends {
            return tip[..i].trim();
        }
    }
    tip.trim()
}

/// A name in the accessibility tree for a widget that shows no words (see
/// the module notes). Nothing drawn changes.
pub(super) trait Named: Sized {
    /// Call the widget `name`.
    fn named(self, name: &str) -> Self;

    /// [`named`](Self::named) for a name that has to be put together (a
    /// structure's, a pane's): the closure runs only while something is
    /// reading the tree, so a row of a long list costs nothing otherwise.
    fn named_with(self, name: impl FnOnce() -> String) -> Self;

    /// `on_hover_text(tip)`, and for a widget that shows no words the
    /// tooltip's first sentence as its name. A widget that has words keeps
    /// them, so this is safe on anything.
    fn tip(self, tip: impl Into<egui::WidgetText>) -> Self;
}

impl Named for egui::Response {
    fn named(self, name: &str) -> Self {
        self.ctx
            .accesskit_node_builder(self.id, |node| node.set_label(name));
        self
    }

    fn named_with(self, name: impl FnOnce() -> String) -> Self {
        self.ctx
            .accesskit_node_builder(self.id, |node| node.set_label(name()));
        self
    }

    fn tip(self, tip: impl Into<egui::WidgetText>) -> Self {
        let tip = tip.into();
        self.ctx.accesskit_node_builder(self.id, |node| {
            // A label's words are its value, a button's its label.
            let worded = node.label().is_some_and(has_words) || node.value().is_some_and(has_words);
            let name = first_sentence(tip.text());
            if !worded && has_words(name) {
                node.set_label(name);
            }
        });
        self.on_hover_text(tip)
    }
}

/// Name the arrow of a collapsing header after what it folds. egui draws
/// that arrow as a bare clickable area: with this it is a button called
/// *Fold X* or *Unfold X*, by what a click would do.
pub(super) fn name_fold_arrow(arrow: &egui::Response, open: bool, what: &str) {
    arrow.widget_info(|| {
        let verb = if open { "Fold" } else { "Unfold" };
        egui::WidgetInfo::labeled(
            egui::WidgetType::CollapsingHeader,
            true,
            format!("{verb} {what}"),
        )
    });
}

/// `ui.button(text)` with a tooltip; true when clicked.
pub(super) fn tip_button(
    ui: &mut egui::Ui,
    text: impl Into<egui::WidgetText>,
    tip: impl Into<egui::WidgetText>,
) -> bool {
    ui.button(text).tip(tip).clicked()
}

/// [`tip_button`], small.
pub(super) fn small_tip_button(
    ui: &mut egui::Ui,
    text: impl Into<egui::WidgetText>,
    tip: impl Into<egui::WidgetText>,
) -> bool {
    ui.small_button(text).tip(tip).clicked()
}

/// A *Paste* button beside a text field, for the systems where the window
/// library cannot read the system clipboard (Android, iOS: see
/// [`crate::settings::clipboard`]). Draws nothing on the desktop, where
/// Ctrl+V works. Returns the clipboard's text when pressed, trimmed, and
/// `None` otherwise - also when the clipboard holds no text, which the
/// tooltip says.
pub(super) fn system_paste_button(ui: &mut egui::Ui) -> Option<String> {
    let clipboard = crate::settings::clipboard::get()?;
    let clicked = ui
        .small_button("📋 Paste")
        .on_hover_text("Paste from another app")
        .clicked();
    if !clicked {
        return None;
    }
    clipboard
        .text()
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
}

/// [`tip_button`] that can be greyed out; the tooltip shows either way.
pub(super) fn enabled_tip_button(
    ui: &mut egui::Ui,
    enabled: bool,
    text: impl Into<egui::WidgetText>,
    tip: impl Into<egui::WidgetText>,
) -> bool {
    ui.add_enabled(enabled, egui::Button::new(text))
        .tip(tip)
        .clicked()
}

/// Any widget - a small button, say - with a tooltip, greyed out unless
/// `enabled`; true when clicked.
pub(super) fn tip_widget(
    ui: &mut egui::Ui,
    enabled: bool,
    widget: impl egui::Widget,
    tip: impl Into<egui::WidgetText>,
) -> bool {
    ui.add_enabled(enabled, widget).tip(tip).clicked()
}

/// A small colour square that opens the colour picker when clicked - the
/// same one for RT structures, segmentations and the colour a propagated
/// copy will land in, so a colour is always changed the same way. `of` is
/// whose colour it is (the structure's name), for the square's name in the
/// accessibility tree: a list has one a row. Returns true when the colour
/// changed.
pub(super) fn color_swatch(ui: &mut egui::Ui, color: &mut [u8; 3], of: &str) -> bool {
    ui.scope(|ui| {
        ui.spacing_mut().interact_size = egui::vec2(12.0, 12.0);
        ui.color_edit_button_srgb(color)
            .named_with(|| format!("Colour of {of}"))
            .on_hover_text("Change the colour")
            .changed()
    })
    .inner
}

/// A glyph-only selectable button, the shape every drawing tool takes.
pub(super) fn glyph_button(ui: &mut egui::Ui, on: bool, glyph: &str, tip: &str) -> bool {
    ui.add(egui::Button::selectable(on, glyph))
        .tip(tip)
        .clicked()
}

/// `(pick)` and a combo over `list`, choosing by index - the operand picker
/// of the comparison section and the transfer fold.
pub(super) fn index_picker(
    ui: &mut egui::Ui,
    salt: &str,
    item: &mut Option<usize>,
    list: &[String],
    width: f32,
) {
    let sel = item
        .and_then(|i| list.get(i).cloned())
        .unwrap_or_else(|| "(pick)".into());
    egui::ComboBox::from_id_salt(salt.to_string())
        .width(width)
        .selected_text(sel)
        .show_ui(ui, |ui| {
            for (i, l) in list.iter().enumerate() {
                ui.selectable_value(item, Some(i), l);
            }
        });
}

/// A combo over the structures and segments of a workspace, with a sentinel
/// entry for "none": the picker behind the grow limit and the threshold's
/// limiting structure.
pub(super) fn item_picker(
    ui: &mut egui::Ui,
    salt: &str,
    sel: &mut Option<ItemRef>,
    cands: &[(ItemRef, String)],
    none_label: &str,
    width: f32,
) -> egui::Response {
    let text = sel
        .and_then(|it| cands.iter().find(|(c, _)| *c == it).map(|(_, l)| l.clone()))
        .unwrap_or_else(|| none_label.to_string());
    egui::ComboBox::from_id_salt(salt)
        .width(width)
        .selected_text(text)
        .show_ui(ui, |ui| {
            ui.selectable_value(sel, None, none_label);
            for (item, label) in cands {
                ui.selectable_value(sel, Some(*item), label);
            }
        })
        .response
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The name a tooltip gives: its first sentence, cut at a colon only
    /// when what stands before it is more than a word.
    #[test]
    fn a_tooltip_names_by_its_first_sentence() {
        for (tip, name) in [
            (
                "Zoom out, about the middle of the view",
                "Zoom out, about the middle of the view",
            ),
            ("Reset this view: fit zoom, clear pan", "Reset this view"),
            (
                "New: an empty RT structure set",
                "New: an empty RT structure set",
            ),
            (
                "Show / hide the crosshair.\nHidden: no navigation",
                "Show / hide the crosshair",
            ),
            ("Blur by 0.5 mm. Slower.", "Blur by 0.5 mm"),
            (
                "Fold these buttons away, leaving the image. One switch.",
                "Fold these buttons away, leaving the image",
            ),
        ] {
            assert_eq!(first_sentence(tip), name);
        }
        assert!(has_words("Step 2"));
        assert!(!has_words("⟲"));
        assert!(!has_words(""));
    }
}
