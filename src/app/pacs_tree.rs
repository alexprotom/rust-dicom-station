//! The tree of the PACS window: cohorts, their patients, studies and
//! series, for this station's PACS and for a paired server alike.
//!
//! A row is selected with a click, added to or taken out of the selection
//! with Ctrl (Cmd on a Mac) and a click, and a range is selected with
//! Shift and a click. The keyboard does the same: the arrows move (Shift
//! held: extend), Left and Right fold and unfold, Ctrl+A selects every row
//! shown, Escape clears, F2 asks to rename and Delete to remove. What a
//! selection is used for (load, pull, move, remove) is the caller's; this
//! file only draws rows and keeps the selection.
//!
//! Rows are drawn only where they are visible ([`egui::ScrollArea::show_rows`]),
//! so a PACS of thousands of series scrolls as quickly as one of ten.

use std::collections::{BTreeSet, HashSet};

/// One row of the model with the rows below it.
#[derive(Clone, Debug, Default)]
pub(super) struct TNode {
    /// Unique among its siblings: a cohort id, a patient key, a UID.
    pub key: String,
    pub icon: &'static str,
    pub text: String,
    /// Counts, drawn at the right edge.
    pub right: String,
    pub hover: String,
    pub files: usize,
    pub children: Vec<TNode>,
}

/// A row, by the keys from the top down: `[cohort]`, `[cohort, patient]`,
/// `[cohort, patient, study]`, `[cohort, patient, study, series]`; a series
/// of a treatment day has the day's row between the study and it
/// (`[..., study, day, series]`, see [`study_rows`]), which
/// [`series_ids`] takes out again before a selection is acted on.
pub(super) type NodeId = Vec<String>;

/// The key of a treatment day's row: `day:20240917`. Never a UID, which is
/// digits and dots.
const DAY_KEY: &str = "day:";

/// One series of a study for [`study_rows`]: its row, its modality and when
/// it was made (`YYYYMMDDHHMMSS`, empty when unknown).
pub(super) struct SeriesRow {
    pub node: TNode,
    pub modality: String,
    pub datetime: String,
}

/// The rows under a study. A study that is a treatment course - it holds
/// treatment records - has a row per day a record was made on
/// (`2024-09-17 · 4 of 16`), holding that day's series in time order: the
/// daily CT, the radiographs, the registrations, the records, the dose
/// reports. Its other series (the planning CT, the plan) stay directly
/// under the study. The viewer's data tree puts a day together by beam
/// ([`crate::sessions`]); the catalog knows no references, so here it is
/// the series' date alone. Any other study lists its series as they come.
pub(super) fn study_rows(series: Vec<SeriesRow>) -> Vec<TNode> {
    let day = |s: &SeriesRow| s.datetime.get(..8).map(str::to_string);
    let days: std::collections::BTreeSet<String> = series
        .iter()
        .filter(|s| s.modality == "RTRECORD")
        .filter_map(day)
        .collect();
    if days.is_empty() {
        return series.into_iter().map(|s| s.node).collect();
    }
    let of = days.len();
    let mut out: Vec<TNode> = Vec::new();
    let mut by_day: std::collections::BTreeMap<String, Vec<SeriesRow>> = Default::default();
    for s in series {
        match day(&s).filter(|d| days.contains(d)) {
            Some(d) => by_day.entry(d).or_default().push(s),
            None => out.push(s.node),
        }
    }
    for (n, (date, mut rows)) in by_day.into_iter().enumerate() {
        rows.sort_by(|a, b| a.datetime.cmp(&b.datetime));
        let files: usize = rows.iter().map(|r| r.node.files).sum();
        let mut kinds: Vec<(String, usize)> = Vec::new();
        for r in &rows {
            match kinds.iter_mut().find(|(m, _)| *m == r.modality) {
                Some((_, k)) => *k += 1,
                None => kinds.push((r.modality.clone(), 1)),
            }
        }
        out.push(TNode {
            key: format!("{DAY_KEY}{date}"),
            icon: "🕐",
            text: format!("{} · {} of {of}", crate::archive::show_date(&date), n + 1),
            right: format!(
                "{} series · {files} file{}",
                rows.len(),
                if files == 1 { "" } else { "s" }
            ),
            hover: format!(
                "A treatment day: the series made that day, by the time they were made\n{}",
                kinds
                    .iter()
                    .map(|(m, k)| format!("{m} {k}"))
                    .collect::<Vec<_>>()
                    .join(" · ")
            ),
            files,
            children: rows.into_iter().map(|r| r.node).collect(),
        });
    }
    out
}

/// The rows a selection means, with the treatment days taken out: a day's
/// row is each of its series, a series under a day the series of its
/// study - so every id is `[cohort]`, `[cohort, patient]`, `[cohort,
/// patient, study]` or `[cohort, patient, study, series]`, as the callers
/// read them.
/// Is this key a treatment day's row (which names no object of its own)?
pub(super) fn is_day(key: &str) -> bool {
    key.starts_with(DAY_KEY)
}

pub(super) fn series_ids(nodes: &[TNode], ids: &[NodeId]) -> Vec<NodeId> {
    let mut out: Vec<NodeId> = Vec::new();
    let mut push = |id: NodeId| {
        if !out.contains(&id) {
            out.push(id);
        }
    };
    for id in ids {
        match id.iter().position(|k| k.starts_with(DAY_KEY)) {
            None => push(id.clone()),
            Some(at) if at + 1 == id.len() => {
                for child in node_at(nodes, id)
                    .map(|n| n.children.as_slice())
                    .unwrap_or(&[])
                {
                    let mut s = id[..at].to_vec();
                    s.push(child.key.clone());
                    push(s);
                }
            }
            Some(at) => {
                let mut s = id.clone();
                s.remove(at);
                push(s);
            }
        }
    }
    out
}

/// Which rows are unfolded and selected, and where the keyboard is.
#[derive(Clone, Debug, Default)]
pub(super) struct TreeView {
    pub expanded: HashSet<NodeId>,
    pub selected: BTreeSet<NodeId>,
    /// Where a Shift-click or Shift-arrow range starts.
    pub anchor: Option<NodeId>,
    /// The row the keyboard is on.
    pub focus: Option<NodeId>,
    /// Rows whose text holds this (any case) are shown, with what leads
    /// to them.
    pub filter: String,
    /// The scroll position and height of the list as last drawn, to bring
    /// a row the keyboard moved to into view.
    offset: f32,
    viewport: f32,
    scroll_to: Option<usize>,
}

impl TreeView {
    /// Forget rows that are no longer in the model (after a rescan).
    pub fn retain_known(&mut self, nodes: &[TNode]) {
        let mut all: HashSet<NodeId> = HashSet::new();
        fn walk(nodes: &[TNode], prefix: &NodeId, all: &mut HashSet<NodeId>) {
            for n in nodes {
                let mut id = prefix.clone();
                id.push(n.key.clone());
                walk(&n.children, &id, all);
                all.insert(id);
            }
        }
        walk(nodes, &Vec::new(), &mut all);
        self.selected.retain(|id| all.contains(id));
        self.expanded.retain(|id| all.contains(id));
        if self.anchor.as_ref().is_some_and(|a| !all.contains(a)) {
            self.anchor = None;
        }
        if self.focus.as_ref().is_some_and(|f| !all.contains(f)) {
            self.focus = None;
        }
    }

    /// The selected rows that no selected row above them already covers.
    pub fn tops(&self) -> Vec<NodeId> {
        self.selected
            .iter()
            .filter(|id| !(1..id.len()).any(|n| self.selected.contains(&id[..n])))
            .cloned()
            .collect()
    }

    pub fn select_only(&mut self, id: NodeId) {
        self.selected.clear();
        self.selected.insert(id.clone());
        self.anchor = Some(id.clone());
        self.focus = Some(id);
    }
}

/// The node an id names.
pub(super) fn node_at<'a>(nodes: &'a [TNode], id: &[String]) -> Option<&'a TNode> {
    let (first, rest) = id.split_first()?;
    let n = nodes.iter().find(|n| &n.key == first)?;
    if rest.is_empty() {
        Some(n)
    } else {
        node_at(&n.children, rest)
    }
}

/// How many files the selection holds (a row inside a selected row is not
/// counted twice).
pub(super) fn selected_files(nodes: &[TNode], view: &TreeView) -> usize {
    view.tops()
        .iter()
        .filter_map(|id| node_at(nodes, id))
        .map(|n| n.files)
        .sum()
}

/// One row as drawn.
struct Row<'a> {
    id: NodeId,
    depth: usize,
    node: &'a TNode,
    open: bool,
}

fn matches(n: &TNode, f: &str) -> bool {
    n.text.to_lowercase().contains(f) || n.hover.to_lowercase().contains(f)
}

fn any_below(n: &TNode, f: &str) -> bool {
    n.children.iter().any(|c| matches(c, f) || any_below(c, f))
}

/// The rows to draw, top to bottom: unfolded rows show their children; a
/// filter shows the rows that match and unfolds what leads to them.
fn visible<'a>(nodes: &'a [TNode], view: &TreeView) -> Vec<Row<'a>> {
    let f = view.filter.trim().to_lowercase();
    let mut out = Vec::new();
    fn walk<'a>(
        nodes: &'a [TNode],
        prefix: &NodeId,
        depth: usize,
        f: &str,
        inherited: bool,
        view: &TreeView,
        out: &mut Vec<Row<'a>>,
    ) {
        for n in nodes {
            let mut id = prefix.clone();
            id.push(n.key.clone());
            let hit = f.is_empty() || inherited || matches(n, f);
            let below = !f.is_empty() && !hit && any_below(n, f);
            if !hit && !below {
                continue;
            }
            let open = !n.children.is_empty() && (below || view.expanded.contains(&id));
            out.push(Row {
                id: id.clone(),
                depth,
                node: n,
                open,
            });
            if open {
                walk(
                    &n.children,
                    &id,
                    depth + 1,
                    f,
                    hit && !f.is_empty(),
                    view,
                    out,
                );
            }
        }
    }
    walk(nodes, &Vec::new(), 0, &f, false, view, &mut out);
    out
}

/// What happened in the tree besides selecting and folding.
#[derive(Default)]
pub(super) struct TreeEvents {
    /// F2 on the focused row.
    pub rename: bool,
    /// Delete with a selection.
    pub delete: bool,
    /// A double click on a row without children.
    pub opened: Option<NodeId>,
}

/// Draw the tree, at most `height` points tall. `menu` fills the context
/// menu of a row (the row is selected first unless it already was).
/// `keys`: the tree answers the keyboard (not while a bar above it asks for
/// a name, whose Enter would otherwise unfold a row as well).
pub(super) fn tree_ui(
    ui: &mut egui::Ui,
    nodes: &[TNode],
    view: &mut TreeView,
    id_salt: &str,
    height: f32,
    keys: bool,
    menu: &mut dyn FnMut(&mut egui::Ui, &NodeId, &TreeView),
) -> TreeEvents {
    let mut ev = TreeEvents::default();
    let rows = visible(nodes, view);
    if keys {
        keyboard(ui, &rows, view, &mut ev);
    }

    let row_h = ui.spacing().interact_size.y.max(18.0);
    let mut area = egui::ScrollArea::vertical()
        .id_salt(id_salt)
        .max_height(height)
        .auto_shrink([false, true]);
    if let Some(i) = view.scroll_to.take() {
        let top = i as f32 * (row_h + ui.spacing().item_spacing.y);
        let bottom = top + row_h;
        if top < view.offset {
            area = area.vertical_scroll_offset(top);
        } else if bottom > view.offset + view.viewport && view.viewport > 0.0 {
            area = area.vertical_scroll_offset(bottom - view.viewport);
        }
    }
    let mut clicked: Option<(NodeId, egui::Modifiers, bool)> = None;
    let mut toggled: Option<NodeId> = None;
    let out = area.show_rows(ui, row_h, rows.len(), |ui, range| {
        for row in &rows[range] {
            let sel = view.selected.contains(&row.id);
            let focus = view.focus.as_ref() == Some(&row.id);
            let width = ui.available_width();
            let (rect, resp) =
                ui.allocate_exact_size(egui::vec2(width, row_h), egui::Sense::click());
            if ui.is_rect_visible(rect) {
                let visuals = ui.visuals();
                let painter = ui.painter_at(rect);
                if sel {
                    painter.rect_filled(rect, 2.0, visuals.selection.bg_fill);
                } else if resp.hovered() {
                    painter.rect_filled(rect, 2.0, visuals.widgets.hovered.weak_bg_fill);
                }
                if focus {
                    painter.rect_stroke(
                        rect.shrink(0.5),
                        2.0,
                        egui::Stroke::new(1.0, visuals.selection.stroke.color),
                        egui::StrokeKind::Inside,
                    );
                }
                let text_color = if sel {
                    visuals.selection.stroke.color
                } else {
                    visuals.text_color()
                };
                let x0 = rect.left() + 4.0 + 16.0 * row.depth as f32;
                if !row.node.children.is_empty() {
                    painter.text(
                        egui::pos2(x0 + 6.0, rect.center().y),
                        egui::Align2::CENTER_CENTER,
                        if row.open { "▼" } else { "▶" },
                        egui::FontId::proportional(10.0),
                        visuals.weak_text_color(),
                    );
                }
                let right = egui::WidgetText::from(egui::RichText::new(&row.node.right).small())
                    .into_galley(
                        ui,
                        Some(egui::TextWrapMode::Truncate),
                        (rect.width() * 0.45).max(40.0),
                        egui::TextStyle::Small,
                    );
                let right_w = right.size().x;
                let text_x = x0 + 16.0;
                let room = (rect.right() - right_w - 12.0 - text_x).max(20.0);
                let left = egui::WidgetText::from(format!("{} {}", row.node.icon, row.node.text))
                    .into_galley(
                        ui,
                        Some(egui::TextWrapMode::Truncate),
                        room,
                        egui::TextStyle::Body,
                    );
                painter.galley(
                    egui::pos2(text_x, rect.center().y - left.size().y * 0.5),
                    left,
                    text_color,
                );
                painter.galley(
                    egui::pos2(
                        rect.right() - right_w - 6.0,
                        rect.center().y - right.size().y * 0.5,
                    ),
                    right,
                    if sel {
                        text_color
                    } else {
                        visuals.weak_text_color()
                    },
                );
            }
            let on_arrow = |r: &egui::Response| {
                r.interact_pointer_pos()
                    .is_some_and(|p| p.x < rect.left() + 4.0 + 16.0 * row.depth as f32 + 14.0)
            };
            if resp.double_clicked() {
                if row.node.children.is_empty() {
                    ev.opened = Some(row.id.clone());
                } else {
                    toggled = Some(row.id.clone());
                }
            } else if resp.clicked() {
                if !row.node.children.is_empty() && on_arrow(&resp) {
                    toggled = Some(row.id.clone());
                } else {
                    clicked = Some((row.id.clone(), ui.input(|i| i.modifiers), false));
                }
            } else if resp.secondary_clicked() && !sel {
                clicked = Some((row.id.clone(), egui::Modifiers::NONE, true));
            }
            let resp = if row.node.hover.is_empty() {
                resp
            } else {
                resp.on_hover_text(&row.node.hover)
            };
            resp.context_menu(|ui| menu(ui, &row.id, view));
        }
    });
    view.offset = out.state.offset.y;
    view.viewport = out.inner_rect.height();

    if let Some(id) = toggled {
        if !view.expanded.remove(&id) {
            view.expanded.insert(id);
        }
    }
    if let Some((id, mods, _right)) = clicked {
        if mods.shift {
            let order: Vec<&NodeId> = rows.iter().map(|r| &r.id).collect();
            let a = view
                .anchor
                .as_ref()
                .and_then(|a| order.iter().position(|x| *x == a));
            let b = order.iter().position(|x| **x == id);
            if let (Some(a), Some(b)) = (a, b) {
                if !mods.command {
                    view.selected.clear();
                }
                for x in &order[a.min(b)..=a.max(b)] {
                    view.selected.insert((*x).clone());
                }
            } else {
                view.select_only(id.clone());
            }
            view.focus = Some(id);
        } else if mods.command {
            if !view.selected.remove(&id) {
                view.selected.insert(id.clone());
            }
            view.anchor = Some(id.clone());
            view.focus = Some(id);
        } else {
            view.select_only(id);
        }
    }
    ev
}

/// The keys the tree answers. Only while no text field has the keyboard.
fn keyboard(ui: &egui::Ui, rows: &[Row], view: &mut TreeView, ev: &mut TreeEvents) {
    if ui.ctx().egui_wants_keyboard_input() || rows.is_empty() {
        return;
    }
    let keys: Vec<(egui::Key, egui::Modifiers)> = ui.input(|i| {
        i.events
            .iter()
            .filter_map(|e| match e {
                egui::Event::Key {
                    key,
                    pressed: true,
                    modifiers,
                    ..
                } => Some((*key, *modifiers)),
                _ => None,
            })
            .collect()
    });
    for (key, mods) in keys {
        let at = view
            .focus
            .as_ref()
            .and_then(|f| rows.iter().position(|r| &r.id == f));
        let go = |to: usize, view: &mut TreeView| {
            let id = rows[to].id.clone();
            if mods.shift {
                let a = view
                    .anchor
                    .as_ref()
                    .and_then(|a| rows.iter().position(|r| &r.id == a))
                    .unwrap_or(to);
                view.selected.clear();
                for r in &rows[a.min(to)..=a.max(to)] {
                    view.selected.insert(r.id.clone());
                }
                view.focus = Some(id);
            } else {
                view.select_only(id);
            }
            view.scroll_to = Some(to);
        };
        match key {
            egui::Key::ArrowDown => {
                let to = at.map_or(0, |i| (i + 1).min(rows.len() - 1));
                go(to, view);
            }
            egui::Key::ArrowUp => {
                let to = at.map_or(0, |i| i.saturating_sub(1));
                go(to, view);
            }
            egui::Key::Home => go(0, view),
            egui::Key::End => go(rows.len() - 1, view),
            egui::Key::ArrowRight => {
                if let Some(i) = at {
                    let r = &rows[i];
                    if !r.node.children.is_empty() {
                        if r.open {
                            if i + 1 < rows.len() {
                                go(i + 1, view);
                            }
                        } else {
                            view.expanded.insert(r.id.clone());
                        }
                    }
                }
            }
            egui::Key::ArrowLeft => {
                if let Some(i) = at {
                    let r = &rows[i];
                    if r.open {
                        view.expanded.remove(&r.id);
                    } else if r.id.len() > 1 {
                        let parent = r.id[..r.id.len() - 1].to_vec();
                        if let Some(p) = rows.iter().position(|x| x.id == parent) {
                            go(p, view);
                        }
                    }
                }
            }
            egui::Key::Enter => {
                if let Some(i) = at {
                    let r = &rows[i];
                    if r.node.children.is_empty() {
                        ev.opened = Some(r.id.clone());
                    } else if !view.expanded.remove(&r.id) {
                        view.expanded.insert(r.id.clone());
                    }
                }
            }
            egui::Key::A if mods.command => {
                view.selected = rows.iter().map(|r| r.id.clone()).collect();
            }
            egui::Key::Escape => {
                view.selected.clear();
            }
            egui::Key::F2 => ev.rename = view.focus.is_some() || !view.selected.is_empty(),
            egui::Key::Delete => ev.delete = !view.selected.is_empty(),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n(key: &str, text: &str, children: Vec<TNode>) -> TNode {
        TNode {
            key: key.into(),
            text: text.into(),
            files: 1 + children.iter().map(|c| c.files).sum::<usize>(),
            children,
            ..Default::default()
        }
    }

    fn model() -> Vec<TNode> {
        vec![
            n(
                "c1",
                "Trial",
                vec![
                    n("p1", "Doe John", vec![n("s1", "CT thorax", vec![])]),
                    n("p2", "Roe Ann", vec![n("s2", "MR brain", vec![])]),
                ],
            ),
            n("unsorted", "Unsorted", vec![n("p3", "Moe", vec![])]),
        ]
    }

    #[test]
    fn folded_rows_hide_their_children_and_a_filter_unfolds_its_way() {
        let m = model();
        let mut v = TreeView::default();
        assert_eq!(visible(&m, &v).len(), 2);
        v.expanded.insert(vec!["c1".into()]);
        assert_eq!(visible(&m, &v).len(), 4);
        v.filter = "brain".into();
        let ids: Vec<NodeId> = visible(&m, &v).into_iter().map(|r| r.id).collect();
        assert_eq!(
            ids,
            vec![
                vec!["c1".to_string()],
                vec!["c1".into(), "p2".into()],
                vec!["c1".into(), "p2".into(), "s2".into()],
            ]
        );
    }

    #[test]
    fn a_selection_inside_a_selected_row_is_not_counted_twice() {
        let m = model();
        let mut v = TreeView::default();
        v.selected.insert(vec!["c1".into()]);
        v.selected.insert(vec!["c1".into(), "p1".into()]);
        v.selected.insert(vec!["unsorted".into(), "p3".into()]);
        assert_eq!(v.tops().len(), 2);
        assert_eq!(selected_files(&m, &v), m[0].files + 1);
        v.retain_known(&m[..1]);
        assert_eq!(v.selected.len(), 2, "rows gone from the model are dropped");
    }

    #[test]
    fn a_course_groups_its_series_by_the_days_a_record_was_made_on() {
        let row = |key: &str, modality: &str, when: &str| SeriesRow {
            node: n(key, key, vec![]),
            modality: modality.into(),
            datetime: when.into(),
        };
        let rows = study_rows(vec![
            row("ct-plan", "CT", "20240903115855"),
            row("rec2", "RTRECORD", "20240917105946"),
            row("ct-day", "CT", "20240917103701"),
            row("rec1", "RTRECORD", "20240916105418"),
            row("undated", "RTPLAN", ""),
        ]);
        let keys: Vec<&str> = rows.iter().map(|r| r.key.as_str()).collect();
        assert_eq!(keys, ["ct-plan", "undated", "day:20240916", "day:20240917"]);
        assert_eq!(rows[3].text, "2024-09-17 · 2 of 2");
        let day: Vec<&str> = rows[3].children.iter().map(|c| c.key.as_str()).collect();
        assert_eq!(day, ["ct-day", "rec2"], "in the order they were made");
        // A study with no record is listed as it comes.
        let plain = study_rows(vec![row("ct", "CT", "20240903115855")]);
        assert_eq!(plain[0].key, "ct");

        // A selection of a day is its series; a series under a day is the
        // study's series.
        let model = vec![n(
            "c",
            "Trial",
            vec![n("p", "P", vec![n("s", "Course", rows)])],
        )];
        let id = |k: &[&str]| k.iter().map(|s| s.to_string()).collect::<NodeId>();
        assert_eq!(
            series_ids(&model, &[id(&["c", "p", "s", "day:20240917"])]),
            vec![id(&["c", "p", "s", "ct-day"]), id(&["c", "p", "s", "rec2"])]
        );
        assert_eq!(
            series_ids(&model, &[id(&["c", "p", "s", "day:20240916", "rec1"])]),
            vec![id(&["c", "p", "s", "rec1"])]
        );
        assert_eq!(
            series_ids(&model, &[id(&["c", "p"])]),
            vec![id(&["c", "p"])]
        );
    }
}
