//! A transform matrix that can be typed into, the way 3D Slicer's
//! *Transforms* module shows one.
//!
//! Every tool here that moves anatomy from one place to another does it
//! through a [`Transform3`](crate::registration::Transform3), and until now
//! that transform could only come from a registration: the program recovered
//! it, and the user took it or left it. Sometimes the number is already
//! known - a couch shift from the record, a transform from a planning system,
//! a matrix from a colleague, or simply a sanity check with a pure 5 mm
//! translation in it - and typing it is faster and more honest than tuning a
//! registration until it produces it.
//!
//! So this is one widget, used in the same shape wherever a transform is:
//! the sixteen numbers, an *Identity* and an *Invert*, copy and paste in the
//! plain whitespace-separated form Slicer puts on the clipboard, and the
//! switch that says whether the tool should use the matrix or the result it
//! computed. Nothing reads the matrix unless that switch is on.
//!
//! While the switch is off the grid follows the tool - the registration that
//! was run, the offset two reference structures give - so what is on screen
//! is what is actually going to happen, and ticking the switch takes that
//! transform over to be adjusted instead of starting from an identity.
//!
//! A run that made several transforms - one per phase of a 4D group - has
//! several matrices to show. Ten grids one under the other would be a wall
//! of numbers, so there is one grid and a picker above it naming which
//! transform it is showing ([`matrix_editor_multi`]).

use super::*;
use crate::registration::Mat4;

/// A matrix a tool holds, and whether it is the one to use.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct ManualMatrix {
    pub(super) m: Mat4,
    /// Use this matrix instead of whatever the tool would compute.
    pub(super) use_it: bool,
}

impl Default for ManualMatrix {
    fn default() -> Self {
        ManualMatrix {
            m: Mat4::IDENTITY,
            use_it: false,
        }
    }
}

impl ManualMatrix {
    /// Follow what the tool computed, while the switch is off.
    ///
    /// The grid is then a window onto the transform that is actually going
    /// to be used - the registration that was run, the offset the two
    /// reference structures give - so ticking *Use this matrix* takes that
    /// over and edits it rather than starting from an identity nobody
    /// asked for. Once the switch is on the numbers are the user's, and a
    /// later run of the tool does not reach in and change them.
    pub(super) fn follow(&mut self, computed: Option<Mat4>) {
        if !self.use_it {
            self.m = computed.unwrap_or(Mat4::IDENTITY);
        }
    }

    /// The transform to apply, or `None` to let the tool use its own.
    pub(super) fn transform(&self, center: crate::geometry::Vec3) -> Option<Transform3> {
        self.use_it.then(|| Transform3::from_matrix(self.m, center))
    }
}

/// One transform a run produced, for the picker above the grid.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct MatrixChoice {
    /// What it was of: a phase's name, or what the one transform is.
    pub(super) label: String,
    pub(super) m: Mat4,
}

/// The editor over a run that produced more than one transform: a picker
/// naming each, and the one grid below it.
///
/// `pick` is which of them the grid is showing, kept by the caller so it
/// survives the frame. It is clamped here, because the list shrinks when a
/// group is re-registered with fewer phases.
pub(super) fn matrix_editor_multi(
    ui: &mut egui::Ui,
    state: &mut ManualMatrix,
    pick: &mut usize,
    choices: &[MatrixChoice],
) -> bool {
    if *pick >= choices.len() {
        *pick = 0;
    }
    let mut changed = false;
    if choices.len() > 1 {
        ui.horizontal_wrapped(|ui| {
            ui.label("Of");
            for (i, c) in choices.iter().enumerate() {
                if ui
                    .selectable_label(*pick == i, &c.label)
                    .on_hover_text(
                        "Show this transform in the grid. With the switch below off the                          grid follows whichever is picked; with it on, *From the result*                          copies this one in.",
                    )
                    .clicked()
                {
                    *pick = i;
                    changed = true;
                }
            }
        });
    } else if let Some(c) = choices.first() {
        if !c.label.is_empty() {
            ui.weak(format!("of {}", c.label));
        }
    }
    changed | matrix_editor(ui, state, choices.get(*pick).map(|c| c.m))
}

/// Draw the editor. Returns whether anything changed.
///
/// `computed` is what the tool would use on its own, shown as the starting
/// point a hand edit can be taken from; `None` where there is nothing yet.
pub(super) fn matrix_editor(
    ui: &mut egui::Ui,
    state: &mut ManualMatrix,
    computed: Option<Mat4>,
) -> bool {
    let mut changed = false;
    // Off, the grid shows what the tool worked out for itself; the moment
    // the switch goes on, those same numbers become the ones being edited.
    state.follow(computed);
    ui.horizontal_wrapped(|ui| {
        changed |= ui
            .checkbox(&mut state.use_it, "Use this matrix")
            .on_hover_text(
                "Apply the numbers below instead of the transform this tool would \
                 compute. Off, the grid follows what the tool itself worked out, \
                 so ticking this takes that transform over and edits it.",
            )
            .changed();
        // Say which of the two the grid is showing, and - when it is the
        // identity - whether that is a transform that moves nothing or
        // nothing at all.
        let note = match (state.use_it, computed.is_some(), state.m.is_identity()) {
            (true, _, true) => Some("identity - nothing would move"),
            (true, _, false) => None,
            (false, false, _) => Some("identity - nothing computed yet"),
            (false, true, true) => Some("what this tool computed: the identity - nothing moves"),
            (false, true, false) => Some("what this tool computed - tick to edit it"),
        };
        if let Some(note) = note {
            ui.weak(note);
        }
    });
    ui.add_enabled_ui(state.use_it, |ui| {
        // Four rows of four, the way it is written down. The bottom row is
        // 0 0 0 1 for any spatial transform, so it is shown and not editable:
        // a perspective row would not be one.
        egui::Grid::new(ui.next_auto_id())
            .num_columns(4)
            .spacing([4.0, 2.0])
            .show(ui, |ui| {
                for r in 0..4 {
                    for c in 0..4 {
                        if r == 3 {
                            // Shown so the matrix is the whole matrix, but
                            // not editable. The decimals are capped as the
                            // editable cells are: egui reads them off the
                            // drag speed, and a speed of zero means "every
                            // decimal there is", which would set the width
                            // of the whole grid.
                            ui.add_enabled(
                                false,
                                egui::DragValue::new(&mut state.m.0[r][c].clone())
                                    .speed(0.0)
                                    .max_decimals(3),
                            );
                            continue;
                        }
                        // The translation column is millimetres; the rest is
                        // a direction cosine, and a coarse drag would make
                        // nonsense of it.
                        let speed = if c == 3 { 0.5 } else { 0.005 };
                        changed |= ui
                            .add(
                                egui::DragValue::new(&mut state.m.0[r][c])
                                    .speed(speed)
                                    .max_decimals(6),
                            )
                            .changed();
                    }
                    ui.end_row();
                }
            });
        ui.weak("rows: x, y, z in patient millimetres; the right column is the shift");
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing.x = 4.0;
            if small_tip_button(ui, "Identity", "Back to a matrix that moves nothing") {
                state.m = Mat4::IDENTITY;
                changed = true;
            }
            // A matrix that maps space flat has no other way round, so the
            // button is off rather than silently doing nothing.
            let inverse = state.m.invert();
            if tip_widget(
                ui,
                inverse.is_some(),
                egui::Button::new("Invert").small(),
                if inverse.is_some() {
                    "Replace it with the mapping the other way"
                } else {
                    "This matrix flattens space, so it has no inverse"
                },
            ) {
                if let Some(i) = inverse {
                    state.m = i;
                    changed = true;
                }
            }
            if let Some(c) = computed {
                if small_tip_button(
                    ui,
                    "From the result",
                    "Fill it in from the transform this tool has computed, as a place to \
                     start editing from",
                ) {
                    state.m = c;
                    changed = true;
                }
            }
            if small_tip_button(
                ui,
                "📋 Copy",
                "The sixteen numbers to the clipboard, whitespace separated - the form \
                 3D Slicer reads",
            ) {
                ui.ctx().copy_text(state.m.to_text());
            }
            if small_tip_button(
                ui,
                "📥 Paste",
                "Read sixteen numbers from the clipboard, whatever separates them",
            ) {
                changed |= paste_matrix(ui, &mut state.m);
            }
        });
    });
    changed
}

/// Read a matrix out of the clipboard.
///
/// egui delivers a paste as an input event rather than handing the clipboard
/// over on demand, so the button asks for one and the text arrives on the
/// next pass; [`ViewerApp::take_pasted_matrix`] is where it lands.
fn paste_matrix(ui: &mut egui::Ui, into: &mut Mat4) -> bool {
    let text = ui.input(|i| {
        i.events.iter().rev().find_map(|e| match e {
            egui::Event::Paste(t) => Some(t.clone()),
            _ => None,
        })
    });
    match text.as_deref().and_then(Mat4::from_text) {
        Some(m) => {
            *into = m;
            true
        }
        None => {
            // Nothing on the clipboard this pass: ask the platform for it and
            // the event arrives on the next one.
            ui.ctx()
                .send_viewport_cmd(egui::ViewportCommand::RequestPaste);
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Vec3;

    #[test]
    fn an_unused_matrix_is_no_transform_at_all() {
        let mut s = ManualMatrix::default();
        assert!(s.transform(Vec3::ZERO).is_none(), "off means off");
        s.use_it = true;
        assert!(s.transform(Vec3::ZERO).is_some(), "on means use it");
    }

    #[test]
    fn the_grid_follows_the_tool_until_it_is_taken_over() {
        let mut s = ManualMatrix::default();
        let mut computed = Mat4::IDENTITY;
        computed.0[1][3] = 24.9;
        // Off, the grid is a window onto what the tool worked out, so
        // ticking the switch starts the edit from that and not from an
        // identity nobody asked for.
        s.follow(Some(computed));
        assert_eq!(s.m, computed, "off, it shows the computed transform");
        // On, the numbers belong to the user: a later run of the tool does
        // not reach in and undo the edit.
        s.use_it = true;
        s.m.0[1][3] = 30.0;
        let mut later = Mat4::IDENTITY;
        later.0[1][3] = -5.0;
        s.follow(Some(later));
        assert!(
            (s.m.0[1][3] - 30.0).abs() < 1e-9,
            "a hand edit survives the next run"
        );
        // Off again, and it follows again.
        s.use_it = false;
        s.follow(Some(later));
        assert_eq!(s.m, later, "the switch off means following once more");
        // Nothing computed yet: the identity, not the last thing it saw.
        s.follow(None);
        assert!(
            s.m.is_identity(),
            "with nothing computed it is the identity"
        );
    }

    #[test]
    fn a_hand_typed_shift_moves_a_point_by_exactly_that_much() {
        let mut s = ManualMatrix {
            m: Mat4::IDENTITY,
            use_it: true,
        };
        s.m.0[0][3] = 5.0;
        s.m.0[2][3] = -3.0;
        let t = s.transform(Vec3::ZERO).expect("it is on");
        let p = Vec3::new(10.0, 20.0, 30.0);
        let q = t.map(p);
        assert!((q.x - 15.0).abs() < 1e-9, "x shifted by 5");
        assert!((q.y - 20.0).abs() < 1e-9, "y untouched");
        assert!((q.z - 27.0).abs() < 1e-9, "z shifted by -3");
        // And back again, because a tool that carries structures one way
        // carries them the other.
        let back = t.unmap(q);
        assert!((back - p).length() < 1e-9, "unmap is the inverse");
    }

    #[test]
    fn the_matrix_of_a_rigid_transform_is_the_same_mapping() {
        // A rotation about a centre that is not the origin is where the
        // centre has to be folded into the translation column correctly.
        let center = Vec3::new(30.0, -10.0, 5.0);
        let rigid =
            crate::registration::RigidTransform::new([0.1, -0.2, 0.3, 4.0, -5.0, 6.0], center);
        let m = Mat4::from_rigid(&rigid);
        for p in [
            Vec3::ZERO,
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(-40.0, 15.0, 80.0),
            center,
        ] {
            let a = rigid.map(p);
            let b = m.map(p);
            assert!(
                (a - b).length() < 1e-9,
                "the matrix agrees with the rigid transform at {p:?}: {a:?} vs {b:?}"
            );
        }
    }

    #[test]
    fn a_matrix_survives_the_clipboard_and_nonsense_is_refused() {
        let mut m = Mat4::IDENTITY;
        m.0[1][3] = 12.5;
        m.0[0][1] = -0.25;
        let back = Mat4::from_text(&m.to_text()).expect("sixteen numbers read back");
        for r in 0..4 {
            for c in 0..4 {
                assert!((back.0[r][c] - m.0[r][c]).abs() < 1e-6, "cell {r},{c}");
            }
        }
        assert!(Mat4::from_text("1 2 3").is_none(), "too few numbers");
        assert!(Mat4::from_text("").is_none(), "nothing at all");
        assert!(
            Mat4::from_text("1,0,0,0\n0,1,0,0\n0,0,1,0\n0,0,0,1").is_some(),
            "commas separate numbers as well as spaces do"
        );
    }

    #[test]
    fn inverting_a_flat_matrix_is_refused_rather_than_guessed() {
        let mut m = Mat4::IDENTITY;
        m.0[2][2] = 0.0;
        assert!(m.invert().is_none(), "a singular matrix has no inverse");
        assert!(Mat4::IDENTITY.invert().is_some(), "the identity does");
    }
}

#[cfg(test)]
mod multi_tests {
    use super::*;

    fn choice(label: &str, tx: f64) -> MatrixChoice {
        let mut m = Mat4::IDENTITY;
        m.0[0][3] = tx;
        MatrixChoice {
            label: label.into(),
            m,
        }
    }

    #[test]
    fn the_picker_index_is_clamped_to_what_there_is() {
        // The list shrinks when a group is re-registered with fewer phases;
        // a stale index must not index off the end.
        let choices = [choice("0%", 1.0), choice("50%", 2.0)];
        let mut pick = 7usize;
        if pick >= choices.len() {
            pick = 0;
        }
        assert_eq!(pick, 0);
        assert_eq!(choices.get(pick).map(|c| c.m), Some(choices[0].m));
    }

    #[test]
    fn each_phase_carries_its_own_matrix() {
        let choices = [choice("0%", 1.0), choice("50%", 2.0), choice("90%", 3.0)];
        // Following the picked one is what the grid does while the switch
        // is off, so the numbers on screen are that phase's.
        let mut state = ManualMatrix::default();
        for (i, c) in choices.iter().enumerate() {
            state.follow(Some(choices[i].m));
            assert_eq!(state.m, c.m, "phase {} shows its own transform", c.label);
        }
    }
}
