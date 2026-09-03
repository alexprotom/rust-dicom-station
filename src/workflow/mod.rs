//! The pipelines, without a window around them.
//!
//! Everything here is what a tool window used to do on its worker thread,
//! moved out so that a second caller (the MCP server `rds-mcp`, or a test)
//! can run the same steps on the same data and get the same answer. The
//! viewer still owns the choice of *what* runs (its dialogs build the
//! requests); this module owns what a run *is*.
//!
//! Rules that keep both callers honest:
//!
//! * nothing in here knows about `egui`, slots or dialogs; a request carries
//!   copies (or `Arc`s) of what it needs and nothing else;
//! * every run takes a [`Progress`](crate::progress::Progress) and honours
//!   its cancel flag;
//! * the numerics are the engines' (`registration`, `propagate`, `motion`);
//!   this layer only sequences them.

pub mod anchored;
pub mod group;
pub mod motion;
pub mod select;

use crate::fourd::{FourDGroup, Role};
use crate::loader::SeriesInfo;

use anyhow::{bail, Result};

/// The phase members of a 4D group as `(label, series)` in temporal order,
/// resolved against the study's current series list.
///
/// Fails when a member's series is gone or fewer than two phases remain -
/// both the 4D pipeline and a run onto a group need at least two.
pub fn phases_of(group: &FourDGroup, series: &[SeriesInfo]) -> Result<Vec<(String, SeriesInfo)>> {
    let resolved = group.resolve(series);
    let mut phases = Vec::new();
    for (mi, m) in group.members.iter().enumerate() {
        if m.role != Role::Phase {
            continue;
        }
        let Some(si) = resolved[mi] else {
            bail!(
                "phase '{}' of {} has no series any more",
                m.label,
                group.name
            );
        };
        phases.push((m.label.clone(), series[si].clone()));
    }
    if phases.len() < 2 {
        bail!("the group '{}' has fewer than two phases", group.name);
    }
    Ok(phases)
}
