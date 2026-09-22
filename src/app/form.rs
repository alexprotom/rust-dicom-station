//! A two-column form: labels down the left, controls down the right.
//!
//! A module built out of `ui.horizontal(|ui| { ui.label("Region"); combo })`
//! rows is a ragged stack - every control starts wherever its own label
//! happened to end, and a section of eight parameters has eight different
//! left edges. Reading down such a block means re-finding the controls on
//! every line.
//!
//! So every parameter block in the program is drawn through [`form`]: one
//! `egui::Grid` of two columns, with the label column given the same minimum
//! width everywhere ([`LABEL_W`]). Within a section the controls line up
//! because a grid lines them up; between sections and between modules they
//! line up because the minimum is shared. A row that is not a parameter -
//! a note, a row of buttons, a warning - spans both columns
//! ([`Form::wide`]) rather than pretending to be one.

/// The least width a label column takes, so that forms in different
/// sections - and different modules - share a left edge. A label longer
/// than this widens its own grid, as a grid column does.
pub(super) const LABEL_W: f32 = 104.0;

/// How wide a note or a button row under a form may run before it wraps.
/// Wide enough for a sentence, narrow enough that a module panel at its
/// usual width does not need a horizontal scrollbar for it.
const NOTE_W: f32 = 260.0;

/// Draw a form. The closure gets a [`Form`] to add rows to.
pub(super) fn form<R>(
    ui: &mut egui::Ui,
    id: impl std::hash::Hash + std::fmt::Debug,
    add: impl FnOnce(&mut Form) -> R,
) -> R {
    let mut out = None;
    egui::Grid::new(id)
        .num_columns(2)
        .spacing([8.0, 4.0])
        .show(ui, |ui| {
            out = Some(add(&mut Form { ui }));
        });
    out.expect("the grid's closure runs exactly once")
}

/// The rows of one form.
pub(super) struct Form<'u> {
    ui: &'u mut egui::Ui,
}

impl Form<'_> {
    /// One parameter: its name, then whatever the closure draws.
    pub(super) fn row<R>(&mut self, label: &str, add: impl FnOnce(&mut egui::Ui) -> R) -> R {
        self.name(label, None);
        let r = self.ui.horizontal(add).inner;
        self.ui.end_row();
        r
    }

    /// The same, with a tooltip on the label - for the parameter whose name
    /// is shorter than what it means.
    pub(super) fn row_tip<R>(
        &mut self,
        label: &str,
        tip: &str,
        add: impl FnOnce(&mut egui::Ui) -> R,
    ) -> R {
        self.name(label, Some(tip));
        let r = self.ui.horizontal(add).inner;
        self.ui.end_row();
        r
    }

    /// A row with no parameter name: a note under the block, a row of
    /// buttons, a warning.
    ///
    /// It goes in the control column, under the control it belongs to, with
    /// the label column left empty. A grid has no column spanning, and
    /// putting wrapping text in the label column instead would set that
    /// column's width from the note rather than from the labels - which is
    /// a column of labels one word wide and a wall of text beside it.
    pub(super) fn wide<R>(&mut self, add: impl FnOnce(&mut egui::Ui) -> R) -> R {
        self.ui.label("");
        // A vertical scope, not `horizontal_wrapped`: a grid takes the row's
        // height from what the cell reports, and wrapped text in a
        // horizontal layout reports the height of one line, so the next row
        // is drawn over it.
        let r = self
            .ui
            .vertical(|ui| {
                ui.set_max_width(NOTE_W);
                add(ui)
            })
            .inner;
        self.ui.end_row();
        r
    }

    fn name(&mut self, label: &str, tip: Option<&str>) {
        let r = self
            .ui
            .scope(|ui| {
                ui.set_min_width(LABEL_W);
                ui.label(label);
            })
            .response;
        if let Some(tip) = tip {
            r.on_hover_text(tip);
        }
    }
}
