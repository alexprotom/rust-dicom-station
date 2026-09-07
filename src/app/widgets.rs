//! Small widgets every window and panel reaches for: a button with a
//! tooltip, a glyph button, a picker over a list.

use super::combine::ItemRef;

/// `ui.button(text)` with a tooltip; true when clicked.
pub(super) fn tip_button(
    ui: &mut egui::Ui,
    text: impl Into<egui::WidgetText>,
    tip: impl Into<egui::WidgetText>,
) -> bool {
    ui.button(text).on_hover_text(tip).clicked()
}

/// [`tip_button`], small.
pub(super) fn small_tip_button(
    ui: &mut egui::Ui,
    text: impl Into<egui::WidgetText>,
    tip: impl Into<egui::WidgetText>,
) -> bool {
    ui.small_button(text).on_hover_text(tip).clicked()
}

/// [`tip_button`] that can be greyed out; the tooltip shows either way.
pub(super) fn enabled_tip_button(
    ui: &mut egui::Ui,
    enabled: bool,
    text: impl Into<egui::WidgetText>,
    tip: impl Into<egui::WidgetText>,
) -> bool {
    ui.add_enabled(enabled, egui::Button::new(text))
        .on_hover_text(tip)
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
    ui.add_enabled(enabled, widget).on_hover_text(tip).clicked()
}

/// A glyph-only selectable button, the shape every drawing tool takes.
pub(super) fn glyph_button(ui: &mut egui::Ui, on: bool, glyph: &str, tip: &str) -> bool {
    ui.add(egui::Button::selectable(on, glyph))
        .on_hover_text(tip)
        .clicked()
}

/// `(pick)` and a combo over `list`, choosing by index - the operand picker
/// of the compare and transfer windows.
pub(super) fn index_picker(
    ui: &mut egui::Ui,
    salt: &str,
    item: &mut Option<usize>,
    list: &[String],
) {
    let sel = item
        .and_then(|i| list.get(i).cloned())
        .unwrap_or_else(|| "(pick)".into());
    egui::ComboBox::from_id_salt(salt.to_string())
        .width(260.0)
        .selected_text(sel)
        .show_ui(ui, |ui| {
            for (i, l) in list.iter().enumerate() {
                ui.selectable_value(item, Some(i), l);
            }
        });
}

/// A combo over the structures and segments of a dataset, with a sentinel
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
