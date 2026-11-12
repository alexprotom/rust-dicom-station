//! The analytics panel on the right of the PACS window: what the rows in
//! view hold, as bars or as a ring, one chart above the other.
//!
//! - **Above, the patients' words**: the words in the brackets of each
//!   patient's row (`P01-095 (Adult, Thorax)` gives `Adult` and `Thorax`,
//!   the same words the list beside 🔍 offers, [`super::pacs_win`]),
//!   counted in patients, one chart per [`WordKind`] - age group, sex,
//!   site, subject - since `Adult` and `Thorax` are no shares of one
//!   whole. Each chart has the patients whose row says nothing of its kind
//!   as *not given*; a kind no patient in the scope has is left out. A
//!   patient may name two sites, so the site bars may add up to more than
//!   the patients; the ring shares the circle out among the words.
//! - **Below, the modalities** of the series, counted in studies while the
//!   scope holds more than one study and in series inside one study (where
//!   every modality would be "1 study").
//!
//! The scope is the selection in the tree: nothing selected is the whole
//! PACS as the tree shows it (the filter beside 🔍 applies), a cohort row
//! its patients, a patient row that patient, a study or series row that
//! study or series; several rows are their union, and a patient listed in
//! two cohorts, a study reached twice, count once. A 4D set or a treatment
//! day row stands for its series, as everywhere in the window
//! ([`pacs_tree::series_ids`]).
//!
//! Each bar, slice and legend line says its count and share on hover; a
//! click ticks that word or modality in the filter beside 🔍 (and a second
//! click unticks it), so the charts are a way into the tree as well as a
//! picture of it; *✖ Reset filters* under the scope, while anything
//! filters, shows every study again. The figures are worked out again on
//! every click in the
//! tree, or only on *⟳ Recalculate* when *Settings* says so - the stale
//! figures are then marked as such.

use std::collections::{BTreeMap, BTreeSet, HashSet};

use crate::archive::{PatientEntry, SeriesEntry, StudyEntry, Tree};

use super::pacs_tree::{self, NodeId};
use super::pacs_win::{patient_words, Facet, PacsWindow, WordKind};
use super::widgets::Named;

/// One bar, slice or legend line.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Item {
    pub label: String,
    pub count: usize,
    /// What a click ticks in the filter; `None` for *not given* and
    /// *other*, which nothing filters by.
    pub facet: Option<Facet>,
    /// What else the hover says (files, series of a modality).
    pub more: String,
}

/// What the scope holds.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Stats {
    /// `Whole PACS`, `Cohort Trial A`, `Patient P01-001 (Adult)`,
    /// `3 selected rows`.
    pub scope: String,
    pub patients: usize,
    pub studies: usize,
    pub series: usize,
    pub files: usize,
    /// The words of the patients, a chart per kind (in [`WordKind`]'s
    /// order, those the scope has), most patients first, *not given* last.
    pub words: Vec<(WordKind, Vec<Item>)>,
    /// The modalities, most first, counted in [`Stats::modality_unit`].
    pub modalities: Vec<Item>,
    /// `studies` or `series`.
    pub modality_unit: &'static str,
}

/// The figures of `ids` (rows of `shown`, as [`pacs_tree::series_ids`]
/// gives them) - of all of `shown` when there are none. `filtered` says
/// whether `shown` is the PACS through the filter beside 🔍.
pub(super) fn compute(shown: &Tree, ids: &[NodeId], filtered: bool) -> Stats {
    let mut patients: BTreeMap<String, &PatientEntry> = BTreeMap::new();
    let mut studies: BTreeMap<String, &StudyEntry> = BTreeMap::new();
    let mut series: BTreeMap<(String, String), &SeriesEntry> = BTreeMap::new();
    // A study, with all its series or only the one `only` names.
    fn take<'a>(
        s: &'a StudyEntry,
        only: Option<&str>,
        studies: &mut BTreeMap<String, &'a StudyEntry>,
        series: &mut BTreeMap<(String, String), &'a SeriesEntry>,
    ) {
        for se in s
            .series
            .iter()
            .filter(|se| only.is_none_or(|u| u == se.series_uid))
        {
            series.insert((s.study_uid.clone(), se.series_uid.clone()), se);
        }
        studies.insert(s.study_uid.clone(), s);
    }
    let mut scope = String::new();
    if ids.is_empty() {
        scope = if filtered {
            "Whole PACS, as filtered".into()
        } else {
            "Whole PACS".into()
        };
        for p in shown.cohorts.iter().flat_map(|c| c.patients.iter()) {
            patients.insert(p.key(), p);
            for s in &p.studies {
                take(s, None, &mut studies, &mut series);
            }
        }
    }
    for id in ids {
        let Some(c) = shown.cohorts.iter().find(|c| c.id == id[0]) else {
            continue;
        };
        let pats: Vec<&PatientEntry> = match id.get(1) {
            Some(k) => c.patients.iter().filter(|p| &p.key == k).collect(),
            None => c.patients.iter().collect(),
        };
        for p in pats {
            patients.insert(p.key(), p);
            for s in p
                .studies
                .iter()
                .filter(|s| id.get(2).is_none_or(|u| u == &s.study_uid))
            {
                take(s, id.get(3).map(String::as_str), &mut studies, &mut series);
                if ids.len() == 1 {
                    scope = match id.len() {
                        3 => format!("Study {}", s.title()),
                        4 => s
                            .series
                            .iter()
                            .find(|se| Some(&se.series_uid) == id.get(3))
                            .map(|se| format!("Series {}", se.title()))
                            .unwrap_or_default(),
                        _ => scope,
                    };
                }
            }
            if ids.len() == 1 && id.len() == 2 {
                scope = format!("Patient {}", p.title());
            }
        }
        if ids.len() == 1 && id.len() == 1 {
            scope = format!("Cohort {}", c.name);
        }
    }
    if scope.is_empty() {
        scope = format!("{} selected rows", ids.len());
    }

    // The words kind by kind, one count per patient; `Male` and `male`
    // are one word, under its first spelling.
    type Words = BTreeMap<String, (String, usize)>;
    let mut by_kind: BTreeMap<WordKind, (Words, usize)> = BTreeMap::new();
    for p in patients.values() {
        let mut kinds = BTreeSet::new();
        for w in patient_words(p) {
            let k = WordKind::of(&w);
            kinds.insert(k);
            let e = by_kind.entry(k).or_default();
            e.0.entry(w.to_lowercase()).or_insert((w, 0)).1 += 1;
        }
        for k in kinds {
            by_kind.entry(k).or_default().1 += 1;
        }
    }
    let words = by_kind
        .into_iter()
        .map(|(k, (ws, with))| {
            let mut items: Vec<Item> = ws
                .into_values()
                .map(|(w, n)| Item {
                    facet: Some(Facet::Patient(k, w.clone())),
                    label: w,
                    count: n,
                    more: String::new(),
                })
                .collect();
            sort_items(&mut items);
            let none = patients.len() - with;
            if none > 0 {
                items.push(Item {
                    label: "not given".into(),
                    count: none,
                    facet: None,
                    more: format!(
                        "Patients whose row says no {} in its brackets",
                        k.title().to_lowercase()
                    ),
                });
            }
            (k, items)
        })
        .collect();

    // The modalities: studies holding one, or series inside one study.
    let one_study = studies.len() <= 1;
    let mut by_mod: BTreeMap<&str, (HashSet<&str>, usize, usize)> = BTreeMap::new();
    for ((study, _), se) in &series {
        let m = se.modality.trim();
        let m = if m.is_empty() { "?" } else { m };
        let e = by_mod.entry(m).or_default();
        e.0.insert(study.as_str());
        e.1 += 1;
        e.2 += se.files;
    }
    let mut modalities: Vec<Item> = by_mod
        .into_iter()
        .map(|(m, (st, n, files))| Item {
            label: m.to_string(),
            count: if one_study { n } else { st.len() },
            facet: (m != "?").then(|| Facet::Modality(m.to_string())),
            more: format!(
                "{} in {}, {}",
                plural(n, "series", "series"),
                plural(st.len(), "study", "studies"),
                plural(files, "file", "files")
            ),
        })
        .collect();
    sort_items(&mut modalities);

    Stats {
        scope,
        patients: patients.len(),
        studies: studies.len(),
        series: series.len(),
        files: series.values().map(|se| se.files).sum(),
        words,
        modalities,
        modality_unit: if one_study { "series" } else { "studies" },
    }
}

/// Most first; a tie by name.
fn sort_items(v: &mut [Item]) {
    v.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.label.cmp(&b.label)));
}

fn plural(n: usize, one: &str, many: &str) -> String {
    format!("{n} {}", if n == 1 { one } else { many })
}

/// The colours of the ring's slices, readable on the dark and the light
/// theme alike; the eleventh slice on is folded into *other*.
const PALETTE: [egui::Color32; 10] = [
    egui::Color32::from_rgb(76, 114, 176),
    egui::Color32::from_rgb(221, 132, 82),
    egui::Color32::from_rgb(85, 168, 104),
    egui::Color32::from_rgb(196, 78, 82),
    egui::Color32::from_rgb(129, 114, 179),
    egui::Color32::from_rgb(147, 120, 96),
    egui::Color32::from_rgb(218, 139, 195),
    egui::Color32::from_rgb(204, 185, 116),
    egui::Color32::from_rgb(100, 181, 205),
    egui::Color32::from_rgb(140, 140, 140),
];

/// What a click in the panel asks of the filter beside 🔍.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum PanelClick {
    /// Tick this word or modality, or untick it if it is ticked.
    Toggle(Facet),
    /// Show every study again.
    Reset,
}

/// The panel's body; what a click asks of the filter comes back. `auto`
/// is the *Settings* switch, `pie` the chart kind (switched here,
/// remembered by the caller).
pub(super) fn panel_body(
    ui: &mut egui::Ui,
    w: &mut PacsWindow,
    auto: bool,
    pie: &mut bool,
) -> Option<PanelClick> {
    let selected: Vec<NodeId> = w.view.selected.iter().cloned().collect();
    let ids = pacs_tree::series_ids(&w.nodes, &selected);
    let key = (w.tree_gen, ids.clone());
    let mut recalc = false;
    ui.horizontal(|ui| {
        ui.strong("📊 Analytics");
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.small_button("⟳").tip("Recalculate").clicked() {
                recalc = true;
            }
            ui.selectable_value(pie, true, "◐").tip("Show as rings");
            ui.selectable_value(pie, false, "📊").tip("Show as bars");
        });
    });
    if w.stats.is_none() || recalc || (auto && w.stats_for.as_ref() != Some(&key)) {
        w.stats = Some(compute(&w.shown, &ids, !w.study_filter.picked.is_empty()));
        w.stats_for = Some(key.clone());
    }
    let st = w.stats.as_ref()?;
    if w.stats_for.as_ref() != Some(&key) {
        ui.colored_label(
            ui.visuals().warn_fg_color,
            "The selection or the PACS changed: ⟳ to recalculate",
        );
    }
    let mut clicked = None;
    ui.label(egui::RichText::new(&st.scope).strong());
    let filtering = !w.study_filter.picked.is_empty() || !w.view.filter.is_empty();
    if filtering
        && ui
            .small_button("✖ Reset filters")
            .on_hover_text("Show every study")
            .clicked()
    {
        clicked = Some(PanelClick::Reset);
    }
    ui.weak(format!(
        "{} · {} · {} · {}",
        plural(st.patients, "patient", "patients"),
        plural(st.studies, "study", "studies"),
        plural(st.series, "series", "series"),
        plural(st.files, "file", "files")
    ));
    if st.patients == 0 {
        ui.weak("Nothing to count here.");
        return clicked;
    }
    let picked = &w.study_filter.picked;
    if st.words.is_empty() {
        ui.separator();
        ui.weak(
            "No patient row has words in brackets; give them a name such as P01 (Adult, Thorax).",
        );
    }
    for (k, items) in &st.words {
        ui.separator();
        ui.label(format!(
            "{} · {}",
            k.title(),
            plural(st.patients, "patient", "patients")
        ));
        if let Some(f) = chart(ui, items, st.patients, "patients", *pie, picked) {
            clicked = Some(PanelClick::Toggle(f));
        }
    }
    ui.separator();
    ui.label(format!("Modalities · {}", st.modality_unit));
    let total = if st.modality_unit == "series" {
        st.series
    } else {
        st.studies
    };
    if let Some(f) = chart(ui, &st.modalities, total, st.modality_unit, *pie, picked) {
        clicked = Some(PanelClick::Toggle(f));
    }
    clicked
}

/// One chart: bars, or a ring with its legend. `total` is what the
/// shares are of (`patients`, `studies`, `series`: `unit`).
fn chart(
    ui: &mut egui::Ui,
    items: &[Item],
    total: usize,
    unit: &str,
    pie: bool,
    picked: &BTreeSet<Facet>,
) -> Option<Facet> {
    if items.is_empty() {
        ui.weak("none");
        return None;
    }
    let hover = |it: &Item| {
        let pct = 100.0 * it.count as f64 / total.max(1) as f64;
        let mut t = format!("{}: {} of {total} {unit} ({pct:.0}%)", it.label, it.count);
        if !it.more.is_empty() {
            t.push('\n');
            t.push_str(&it.more);
        }
        match &it.facet {
            Some(f) if picked.contains(f) => t.push_str("\nClick: remove from the filter"),
            Some(_) => t.push_str("\nClick: filter by it"),
            None => {}
        }
        t
    };
    if pie {
        ring(ui, items, picked, &hover)
    } else {
        bars(ui, items, picked, &hover)
    }
}

fn bars(
    ui: &mut egui::Ui,
    items: &[Item],
    picked: &BTreeSet<Facet>,
    hover: &dyn Fn(&Item) -> String,
) -> Option<Facet> {
    let max = items.iter().map(|i| i.count).max().unwrap_or(1).max(1);
    let width = ui.available_width().max(120.0);
    let label_w = (width * 0.42).min(150.0);
    let count_w = 40.0;
    let row_h = ui.text_style_height(&egui::TextStyle::Body) + 4.0;
    let font = egui::TextStyle::Body.resolve(ui.style());
    let mut clicked = None;
    for it in items {
        let sense = if it.facet.is_some() {
            egui::Sense::click()
        } else {
            egui::Sense::hover()
        };
        let (rect, resp) = ui.allocate_exact_size(egui::vec2(width, row_h), sense);
        let v = ui.visuals();
        let on = it.facet.as_ref().is_some_and(|f| picked.contains(f));
        let text = if it.facet.is_some() {
            v.text_color()
        } else {
            v.weak_text_color()
        };
        let label_rect = egui::Rect::from_min_size(rect.min, egui::vec2(label_w - 4.0, row_h));
        ui.painter().with_clip_rect(label_rect).text(
            label_rect.left_center(),
            egui::Align2::LEFT_CENTER,
            &it.label,
            font.clone(),
            text,
        );
        let bar_max = (width - label_w - count_w).max(10.0);
        let bar = egui::Rect::from_min_size(
            egui::pos2(rect.min.x + label_w, rect.min.y + 2.0),
            egui::vec2(bar_max * it.count as f32 / max as f32, row_h - 4.0),
        );
        let mut fill = if it.facet.is_some() {
            v.selection.bg_fill
        } else {
            v.widgets.inactive.bg_fill
        };
        if resp.hovered() {
            fill = fill.gamma_multiply(1.3);
        }
        ui.painter().rect_filled(bar, 2.0, fill);
        if on {
            ui.painter().rect_stroke(
                bar.expand(1.0),
                2.0,
                egui::Stroke::new(1.5, v.strong_text_color()),
                egui::StrokeKind::Outside,
            );
        }
        ui.painter().text(
            egui::pos2(rect.max.x, rect.center().y),
            egui::Align2::RIGHT_CENTER,
            it.count.to_string(),
            font.clone(),
            text,
        );
        let resp = resp.on_hover_text(hover(it));
        if resp.clicked() {
            clicked = it.facet.clone();
        }
    }
    clicked
}

/// The angle of a point seen from `c`, clockwise from the top, in
/// `[0, 2π)`.
fn angle_of(c: egui::Pos2, p: egui::Pos2) -> f32 {
    let a = (p.x - c.x).atan2(-(p.y - c.y));
    if a < 0.0 {
        a + std::f32::consts::TAU
    } else {
        a
    }
}

/// The point at angle `a` (clockwise from the top) and radius `r`.
fn at(c: egui::Pos2, r: f32, a: f32) -> egui::Pos2 {
    egui::pos2(c.x + r * a.sin(), c.y - r * a.cos())
}

/// The slices of a ring: the first ten items, then the rest as one
/// *other*.
fn slices(items: &[Item]) -> Vec<Item> {
    let mut out: Vec<Item> = items.iter().take(PALETTE.len() - 1).cloned().collect();
    let rest = &items[out.len()..];
    match rest.len() {
        0 => {}
        1 => out.push(rest[0].clone()),
        n => out.push(Item {
            label: format!("other ({n})"),
            count: rest.iter().map(|i| i.count).sum(),
            facet: None,
            more: rest
                .iter()
                .map(|i| format!("{} {}", i.label, i.count))
                .collect::<Vec<_>>()
                .join(", "),
        }),
    }
    out
}

/// A slice's colour: its place in the palette, grey for what filters
/// nothing (*not given*, *other*).
fn colour_of(i: usize, it: &Item) -> egui::Color32 {
    if it.facet.is_none() {
        PALETTE[PALETTE.len() - 1]
    } else {
        PALETTE[i % (PALETTE.len() - 1)]
    }
}

fn ring(
    ui: &mut egui::Ui,
    items: &[Item],
    picked: &BTreeSet<Facet>,
    hover: &dyn Fn(&Item) -> String,
) -> Option<Facet> {
    let items = slices(items);
    let sum: usize = items.iter().map(|i| i.count).sum::<usize>().max(1);
    let size = ui.available_width().clamp(100.0, 180.0);
    let (rect, resp) =
        ui.allocate_exact_size(egui::vec2(ui.available_width(), size), egui::Sense::click());
    let c = rect.center();
    let (r, hole) = (size * 0.48, size * 0.26);
    // Where each slice starts and ends.
    let mut bounds = Vec::with_capacity(items.len());
    let mut a = 0.0f32;
    for it in &items {
        let b = a + std::f32::consts::TAU * it.count as f32 / sum as f32;
        bounds.push((a, b));
        a = b;
    }
    let under = resp.hover_pos().and_then(|p| {
        let d = p.distance(c);
        (hole..=r).contains(&d).then(|| {
            let t = angle_of(c, p);
            bounds.iter().position(|(a, b)| (*a..*b).contains(&t))
        })?
    });
    let v = ui.visuals().clone();
    for (i, (it, (a, b))) in items.iter().zip(&bounds).enumerate() {
        let colour = colour_of(i, it);
        let rr = if under == Some(i) { r * 1.04 } else { r };
        // Pieces of at most a quarter turn, each a convex fan.
        let mut s = *a;
        while s < *b - 1e-4 {
            let e = (s + std::f32::consts::FRAC_PI_2).min(*b);
            let steps = (((e - s) / 0.05).ceil() as usize).max(1);
            let mut pts = vec![c];
            for k in 0..=steps {
                pts.push(at(c, rr, s + (e - s) * k as f32 / steps as f32));
            }
            ui.painter()
                .add(egui::Shape::convex_polygon(pts, colour, egui::Stroke::NONE));
            s = e;
        }
        if items.len() > 1 {
            ui.painter().line_segment(
                [at(c, hole, *a), at(c, rr, *a)],
                egui::Stroke::new(1.0, v.panel_fill),
            );
        }
        if it.facet.as_ref().is_some_and(|f| picked.contains(f)) {
            let steps = (((b - a) / 0.05).ceil() as usize).max(2);
            let arc: Vec<egui::Pos2> = (0..=steps)
                .map(|k| at(c, rr + 3.0, a + (b - a) * k as f32 / steps as f32))
                .collect();
            ui.painter().add(egui::Shape::line(
                arc,
                egui::Stroke::new(2.0, v.strong_text_color()),
            ));
        }
    }
    ui.painter().circle_filled(c, hole, v.panel_fill);
    ui.painter().text(
        c,
        egui::Align2::CENTER_CENTER,
        sum.to_string(),
        egui::TextStyle::Heading.resolve(ui.style()),
        v.text_color(),
    );
    let mut clicked = None;
    if let Some(i) = under {
        let resp = resp.on_hover_text(hover(&items[i]));
        if resp.clicked() {
            clicked = items[i].facet.clone();
        }
    }
    // The legend: one line a slice, clickable as the slice is.
    for (i, it) in items.iter().enumerate() {
        let colour = colour_of(i, it);
        let r = ui
            .horizontal(|ui| {
                let (sw, _) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                ui.painter().rect_filled(sw, 2.0, colour);
                let on = it.facet.as_ref().is_some_and(|f| picked.contains(f));
                let text = egui::RichText::new(format!("{}  {}", it.label, it.count));
                let text = if on { text.strong() } else { text };
                let sense = if it.facet.is_some() {
                    egui::Sense::click()
                } else {
                    egui::Sense::hover()
                };
                ui.add(egui::Label::new(text).sense(sense))
            })
            .inner
            .on_hover_text(hover(it));
        if r.clicked() {
            clicked = it.facet.clone();
        }
    }
    clicked
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::archive::CohortEntry;

    fn series(uid: &str, modality: &str, files: usize) -> SeriesEntry {
        SeriesEntry {
            series_uid: uid.into(),
            modality: modality.into(),
            files,
            ..Default::default()
        }
    }

    fn tree() -> Tree {
        let study = |uid: &str, se: Vec<SeriesEntry>| StudyEntry {
            study_uid: uid.into(),
            series: se,
            ..Default::default()
        };
        let p = |key: &str, label: &str, studies: Vec<StudyEntry>| PatientEntry {
            key: key.into(),
            name: key.to_uppercase(),
            id: key.to_uppercase(),
            label: label.into(),
            studies,
            ..Default::default()
        };
        let p1 = p(
            "p1",
            "P1 (Adult, Male, Thorax)",
            vec![
                study(
                    "s1",
                    vec![series("a", "CT", 100), series("b", "RTSTRUCT", 1)],
                ),
                study("s2", vec![series("c", "CT", 90), series("d", "CT", 90)]),
            ],
        );
        Tree {
            cohorts: vec![
                CohortEntry {
                    id: "c1".into(),
                    name: "Trial".into(),
                    patients: vec![
                        p1.clone(),
                        p("p2", "", vec![study("s3", vec![series("e", "MR", 50)])]),
                    ],
                    ..Default::default()
                },
                CohortEntry {
                    id: "c2".into(),
                    name: "Phantoms".into(),
                    patients: vec![
                        // Listed again in a second cohort: counted once.
                        p1,
                        p(
                            "p3",
                            "Mattia (Phantom, male)",
                            vec![study("s4", vec![series("f", "CT", 80)])],
                        ),
                    ],
                    ..Default::default()
                },
            ],
        }
    }

    fn counts(v: &[Item]) -> Vec<(&str, usize)> {
        v.iter().map(|i| (i.label.as_str(), i.count)).collect()
    }

    #[test]
    fn the_whole_pacs_counts_each_patient_and_study_once() {
        let s = compute(&tree(), &[], false);
        assert_eq!(s.scope, "Whole PACS");
        assert_eq!((s.patients, s.studies, s.series, s.files), (3, 4, 6, 411));
        // A chart per kind, each of the three patients: Adult and Thorax
        // are not shares of one whole.
        let kinds: Vec<WordKind> = s.words.iter().map(|(k, _)| *k).collect();
        assert_eq!(
            kinds,
            vec![
                WordKind::Age,
                WordKind::Sex,
                WordKind::Site,
                WordKind::Subject
            ]
        );
        assert_eq!(counts(&s.words[0].1), vec![("Adult", 1), ("not given", 2)]);
        assert_eq!(counts(&s.words[1].1), vec![("Male", 2), ("not given", 1)]);
        assert_eq!(counts(&s.words[2].1), vec![("Thorax", 1), ("not given", 2)]);
        assert_eq!(
            counts(&s.words[3].1),
            vec![("Phantom", 1), ("not given", 2)]
        );
        assert_eq!(s.modality_unit, "studies");
        assert_eq!(
            counts(&s.modalities),
            vec![("CT", 3), ("MR", 1), ("RTSTRUCT", 1)]
        );
        assert_eq!(s.modalities[0].more, "4 series in 3 studies, 360 files");
        assert_eq!(
            s.words[1].1[0].facet,
            Some(Facet::Patient(WordKind::Sex, "Male".into()))
        );
        assert_eq!(s.words[1].1[1].facet, None, "not given filters nothing");
    }

    #[test]
    fn a_selected_row_narrows_the_figures_to_it() {
        let id = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<NodeId>();
        let t = tree();
        let c = compute(&t, &[id(&["c2"])], false);
        assert_eq!(c.scope, "Cohort Phantoms");
        assert_eq!((c.patients, c.studies), (2, 3));
        // Both are male: no one is left out of that chart.
        assert_eq!(counts(&c.words[1].1), vec![("Male", 2)]);

        let p = compute(&t, &[id(&["c1", "p1"])], false);
        assert_eq!(p.scope, "Patient P1 (Adult, Male, Thorax)");
        // A kind nobody in the scope has is no chart.
        assert!(p.words.iter().all(|(k, _)| *k != WordKind::Subject));
        assert_eq!(counts(&p.modalities), vec![("CT", 2), ("RTSTRUCT", 1)]);

        // Inside one study the modalities are counted in series.
        let s = compute(&t, &[id(&["c1", "p1", "s2"])], false);
        assert_eq!(s.modality_unit, "series");
        assert_eq!(counts(&s.modalities), vec![("CT", 2)]);
        let se = compute(&t, &[id(&["c1", "p1", "s2", "d"])], false);
        assert_eq!((se.series, se.files), (1, 90));

        let two = compute(&t, &[id(&["c1", "p2"]), id(&["c2", "p3"])], false);
        assert_eq!(two.scope, "2 selected rows");
        assert_eq!(counts(&two.modalities), vec![("CT", 1), ("MR", 1)]);
    }

    #[test]
    fn a_ring_keeps_ten_slices_and_folds_the_rest() {
        let items: Vec<Item> = (0..14)
            .map(|i| Item {
                label: format!("w{i:02}"),
                count: 20 - i,
                facet: Some(Facet::Patient(WordKind::Site, format!("w{i:02}"))),
                more: String::new(),
            })
            .collect();
        let s = slices(&items);
        assert_eq!(s.len(), 10);
        assert_eq!(s[9].label, "other (5)");
        assert_eq!(s[9].count, (9..14).map(|i| 20 - i).sum::<usize>());
        assert_eq!(s[9].facet, None);
        // Clockwise from the top.
        let c = egui::pos2(0.0, 0.0);
        assert!(angle_of(c, egui::pos2(0.0, -1.0)).abs() < 1e-6);
        assert!((angle_of(c, egui::pos2(1.0, 0.0)) - std::f32::consts::FRAC_PI_2).abs() < 1e-6);
        assert!(
            (angle_of(c, egui::pos2(-1.0, 0.0)) - 3.0 * std::f32::consts::FRAC_PI_2).abs() < 1e-6
        );
    }
}
