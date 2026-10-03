//! Workflows: the program's own steps wired into a graph, saved as a file,
//! and run again on other data.
//!
//! A workflow is what a physicist does by hand in the viewer, written down
//! once: open this folder, find the CT and the target in it, segment the
//! heart and file it as `heart total`, open the 4DCT, do the same on every
//! phase, carry the target across anchored on the heart, measure the motion,
//! build the ITV, write the results out. Each of those is a **node**; the
//! **links** between them say what flows from one step into the next - a
//! study, an image series, a 4D group, a set of structures, a registration,
//! a report. The file is the recipe, not the data: the input folders are
//! parameters of the nodes that read them, and the run dialog offers to
//! point them somewhere else, so the same file is applied to the next
//! patient unchanged.
//!
//! This module is headless, like the rest of `workflow`: it knows nothing
//! about egui, slots or windows.
//!
//! * [`catalog`] names every kind of node - its ports, its parameters, what
//!   it says about itself on the canvas.
//! * [`exec`] runs a workflow on the calling thread, in dependency order,
//!   reporting what it does through a channel. The viewer runs it on a
//!   worker thread and, when asked to show every step, puts each step's data
//!   in front of the user before the next one starts.
//! * [`store`] is where workflow files live in the user's data folder, the
//!   file format's reading and writing, and the examples the program ships.
//!
//! The document model itself ([`Workflow`], [`Node`], [`Link`]) and its
//! checks ([`Workflow::check`], [`Workflow::order`]) are here.
//!
//! **The file.** JSON, pretty-printed so it can be read and diffed:
//!
//! ```json
//! {
//!   "format": "rds-workflow",
//!   "version": 1,
//!   "name": "Heart-anchored target motion",
//!   "nodes": [
//!     { "id": 1, "title": "Cardiac CT", "pos": [40, 60],
//!       "kind": "load_folder", "params": { "path": "D:/data/CCT" } }
//!   ],
//!   "links": [ { "from": [1, 0], "to": [2, 0] } ]
//! }
//! ```
//!
//! Every parameter block is read with defaults for what it does not say, so
//! a file written before a parameter existed still opens; a node kind the
//! program does not know is an error that names it.

pub mod catalog;
pub mod exec;
mod nodes;
pub mod store;

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

pub use catalog::{Category, Kind, Op, PortSpec, PortType};

/// The `format` tag every workflow file carries.
pub const FORMAT: &str = "rds-workflow";

/// The version of the file layout this program writes.
pub const VERSION: u32 = 1;

/// One workflow: its nodes, the links between them, and where a run writes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Workflow {
    #[serde(default = "format_tag")]
    pub format: String,
    #[serde(default = "file_version")]
    pub version: u32,
    pub name: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    #[serde(default)]
    pub output: OutputSettings,
    #[serde(default)]
    pub nodes: Vec<Node>,
    #[serde(default)]
    pub links: Vec<Link>,
    /// Boxes drawn around groups of nodes on the canvas. For the eye only:
    /// a run ignores them.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub frames: Vec<Frame>,
}

/// A titled box around some nodes on the canvas: a row of the example, the
/// phase work, a note to whoever opens the file next. It follows its nodes
/// as they move, and goes when the last of them does.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Frame {
    pub title: String,
    /// sRGB; drawn faint behind the nodes.
    pub color: [u8; 3],
    /// The nodes inside, by id.
    pub nodes: Vec<u32>,
}

impl Default for Frame {
    fn default() -> Self {
        Frame {
            title: "Frame".into(),
            color: [90, 120, 170],
            nodes: Vec::new(),
        }
    }
}

fn format_tag() -> String {
    FORMAT.to_string()
}

fn file_version() -> u32 {
    VERSION
}

/// Where a run puts what it writes.
///
/// Every file a run writes goes under one folder of its own,
/// `<root>/<folder>`, with the folder name made from a template so two runs
/// never write into each other. The export and report nodes name
/// subfolders of it; an absolute path in a node goes where it says.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct OutputSettings {
    /// The folder the run folders are made in. Empty is
    /// `<data folder>/workflow_runs`.
    pub root: String,
    /// The run folder's name: `{workflow}`, `{date}` and `{time}` are
    /// replaced when the run starts.
    pub folder: String,
}

impl Default for OutputSettings {
    fn default() -> Self {
        OutputSettings {
            root: String::new(),
            folder: "{workflow} {date}-{time}".into(),
        }
    }
}

/// One step.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Node {
    /// Stable within the file; links name nodes by it.
    pub id: u32,
    /// What the canvas and the reports call this node. Empty is the kind's
    /// own name.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub title: String,
    /// Where the node sits on the canvas, in canvas units.
    #[serde(default)]
    pub pos: [f32; 2],
    /// What the node does, with its parameters: `kind` and `params` in the
    /// file.
    #[serde(flatten)]
    pub op: Op,
}

impl Node {
    /// The title, or the kind's name when the node has none of its own.
    pub fn label(&self) -> String {
        if self.title.trim().is_empty() {
            self.op.kind().info().name.to_string()
        } else {
            self.title.trim().to_string()
        }
    }
}

/// One wire: output `from.1` of node `from.0` into input `to.1` of node
/// `to.0`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Link {
    pub from: (u32, usize),
    pub to: (u32, usize),
}

/// What [`Workflow::check`] found: errors stop a run, warnings do not.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Findings {
    /// `(node, message)`; `None` for the workflow as a whole.
    pub errors: Vec<(Option<u32>, String)>,
    pub warnings: Vec<(Option<u32>, String)>,
}

impl Findings {
    pub fn ok(&self) -> bool {
        self.errors.is_empty()
    }

    /// The problems of one node, errors first, for its tooltip.
    pub fn of(&self, node: u32) -> Vec<&str> {
        self.errors
            .iter()
            .chain(&self.warnings)
            .filter(|(n, _)| *n == Some(node))
            .map(|(_, m)| m.as_str())
            .collect()
    }

    pub fn node_has_error(&self, node: u32) -> bool {
        self.errors.iter().any(|(n, _)| *n == Some(node))
    }
}

impl Workflow {
    /// An empty workflow with a name.
    pub fn new(name: impl Into<String>) -> Workflow {
        Workflow {
            format: FORMAT.into(),
            version: VERSION,
            name: name.into(),
            description: String::new(),
            output: OutputSettings::default(),
            nodes: Vec::new(),
            links: Vec::new(),
            frames: Vec::new(),
        }
    }

    /// The part of the workflow made of the nodes `ids`: those nodes, the
    /// wires between them (a wire from outside is left behind) and the
    /// frames around nothing else. What *Copy* puts on the clipboard.
    pub fn fragment(&self, ids: &BTreeSet<u32>) -> Workflow {
        let mut wf = Workflow::new(self.name.clone());
        wf.nodes = self
            .nodes
            .iter()
            .filter(|n| ids.contains(&n.id))
            .cloned()
            .collect();
        wf.links = self
            .links
            .iter()
            .filter(|l| ids.contains(&l.from.0) && ids.contains(&l.to.0))
            .copied()
            .collect();
        wf.frames = self
            .frames
            .iter()
            .filter(|f| !f.nodes.is_empty() && f.nodes.iter().all(|n| ids.contains(n)))
            .cloned()
            .collect();
        wf
    }

    /// Add `part`'s nodes, wires and frames, moved by `offset` and given
    /// ids of their own after this workflow's highest. Returns the new id of
    /// each of `part`'s nodes, by its old one.
    pub fn paste(&mut self, part: &Workflow, offset: [f32; 2]) -> BTreeMap<u32, u32> {
        let first = self.nodes.iter().map(|n| n.id).max().unwrap_or(0) + 1;
        let mut ids = BTreeMap::new();
        for (id, n) in (first..).zip(&part.nodes) {
            ids.insert(n.id, id);
            let mut copy = n.clone();
            copy.id = id;
            copy.pos = [n.pos[0] + offset[0], n.pos[1] + offset[1]];
            self.nodes.push(copy);
        }
        for l in &part.links {
            if let (Some(&a), Some(&b)) = (ids.get(&l.from.0), ids.get(&l.to.0)) {
                self.link(a, l.from.1, b, l.to.1);
            }
        }
        for f in &part.frames {
            let nodes: Vec<u32> = f.nodes.iter().filter_map(|n| ids.get(n).copied()).collect();
            if !nodes.is_empty() {
                self.frames.push(Frame { nodes, ..f.clone() });
            }
        }
        ids
    }

    /// Drop from the frames the nodes that are gone, and the frames left
    /// with none.
    pub fn prune_frames(&mut self) {
        let have: BTreeSet<u32> = self.nodes.iter().map(|n| n.id).collect();
        for f in &mut self.frames {
            f.nodes.retain(|n| have.contains(n));
        }
        self.frames.retain(|f| !f.nodes.is_empty());
    }

    /// Read a workflow from its JSON text.
    pub fn from_json(text: &str) -> Result<Workflow> {
        let wf: Workflow = serde_json::from_str(text).context("not a readable workflow file")?;
        if wf.format != FORMAT {
            bail!(
                "this is not a workflow file (its format is '{}', expected '{FORMAT}')",
                wf.format
            );
        }
        if wf.version > VERSION {
            bail!(
                "the file was written by a newer version of the program (workflow format {}, \
                 this program reads up to {VERSION})",
                wf.version
            );
        }
        Ok(wf)
    }

    /// The workflow as pretty-printed JSON, the way it is saved.
    pub fn to_json(&self) -> String {
        let mut wf = self.clone();
        wf.format = FORMAT.into();
        wf.version = VERSION;
        // Links in a stable order, so saving twice writes the same file.
        wf.links.sort();
        wf.links.dedup();
        serde_json::to_string_pretty(&wf).expect("a workflow serializes")
    }

    pub fn node(&self, id: u32) -> Option<&Node> {
        self.nodes.iter().find(|n| n.id == id)
    }

    pub fn node_mut(&mut self, id: u32) -> Option<&mut Node> {
        self.nodes.iter_mut().find(|n| n.id == id)
    }

    /// The next free node id.
    pub fn next_id(&self) -> u32 {
        self.nodes.iter().map(|n| n.id).max().unwrap_or(0) + 1
    }

    /// Add a node with the kind's default parameters; returns its id.
    pub fn add(&mut self, op: Op, title: &str, pos: [f32; 2]) -> u32 {
        let id = self.next_id();
        self.nodes.push(Node {
            id,
            title: title.to_string(),
            pos,
            op,
        });
        id
    }

    /// Wire output `out` of `from` into input `inp` of `to`.
    pub fn link(&mut self, from: u32, out: usize, to: u32, inp: usize) {
        self.links.push(Link {
            from: (from, out),
            to: (to, inp),
        });
    }

    /// The links arriving at input `inp` of node `id`.
    pub fn links_into(&self, id: u32, inp: usize) -> Vec<Link> {
        self.links
            .iter()
            .filter(|l| l.to == (id, inp))
            .copied()
            .collect()
    }

    /// The nodes in the order a run takes them: every node after all the
    /// nodes it reads from. Among nodes that do not depend on each other the
    /// one higher on the canvas goes first (then the one further left, then
    /// the older), so the rows of a drawn workflow run top to bottom and
    /// reruns take the same path.
    ///
    /// Fails on a cycle, naming a node in it.
    pub fn order(&self) -> Result<Vec<u32>> {
        let ids: BTreeSet<u32> = self.nodes.iter().map(|n| n.id).collect();
        let mut indeg: BTreeMap<u32, usize> = ids.iter().map(|&i| (i, 0)).collect();
        let mut out: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
        for l in &self.links {
            if !ids.contains(&l.from.0) || !ids.contains(&l.to.0) {
                continue;
            }
            *indeg.get_mut(&l.to.0).expect("known id") += 1;
            out.entry(l.from.0).or_default().push(l.to.0);
        }
        let key = |id: u32| {
            let n = self.node(id).expect("known id");
            // Positions as integers: a total order that NaN cannot break.
            (n.pos[1].round() as i64, n.pos[0].round() as i64, id)
        };
        let mut ready: BTreeSet<(i64, i64, u32)> = indeg
            .iter()
            .filter(|(_, &d)| d == 0)
            .map(|(&i, _)| key(i))
            .collect();
        let mut order = Vec::with_capacity(ids.len());
        while let Some(first) = ready.iter().next().copied() {
            ready.remove(&first);
            let id = first.2;
            order.push(id);
            for &next in out.get(&id).map(Vec::as_slice).unwrap_or(&[]) {
                let d = indeg.get_mut(&next).expect("known id");
                *d -= 1;
                if *d == 0 {
                    ready.insert(key(next));
                }
            }
        }
        if order.len() != ids.len() {
            let stuck = indeg
                .iter()
                .find(|(i, &d)| d > 0 && !order.contains(i))
                .map(|(&i, _)| i)
                .unwrap_or(0);
            let name = self.node(stuck).map(Node::label).unwrap_or_default();
            bail!("the links go round in a circle through '{name}'; a workflow must flow one way");
        }
        Ok(order)
    }

    /// Would a link from `from` to `to` close a circle?
    pub fn would_cycle(&self, from: u32, to: u32) -> bool {
        if from == to {
            return true;
        }
        // Is `from` reachable from `to` already?
        let mut stack = vec![to];
        let mut seen = BTreeSet::new();
        while let Some(n) = stack.pop() {
            if n == from {
                return true;
            }
            if !seen.insert(n) {
                continue;
            }
            stack.extend(self.links.iter().filter(|l| l.from.0 == n).map(|l| l.to.0));
        }
        false
    }

    /// Everything that would stop a run, and what is merely worth saying.
    ///
    /// The checks are about the graph and the parameters: missing inputs,
    /// wires of the wrong type, a circle, a folder not given. Whether the
    /// folder holds what the workflow expects is only known when it runs.
    pub fn check(&self) -> Findings {
        let mut f = Findings::default();
        if self.nodes.is_empty() {
            f.errors
                .push((None, "the workflow has no steps yet".to_string()));
            return f;
        }
        let mut seen = BTreeSet::new();
        for n in &self.nodes {
            if !seen.insert(n.id) {
                f.errors
                    .push((Some(n.id), format!("node id {} is used twice", n.id)));
            }
        }
        for l in &self.links {
            let (Some(a), Some(b)) = (self.node(l.from.0), self.node(l.to.0)) else {
                f.errors
                    .push((None, "a link names a node that is not there".to_string()));
                continue;
            };
            let outs = a.op.outputs();
            let ins = b.op.inputs();
            let (Some(o), Some(i)) = (outs.get(l.from.1), ins.get(l.to.1)) else {
                f.errors.push((
                    Some(b.id),
                    format!("a link reaches a port '{}' does not have", b.label()),
                ));
                continue;
            };
            if !i.accepts.contains(&o.ty) {
                f.errors.push((
                    Some(b.id),
                    format!(
                        "'{}' takes {} on '{}', not {}",
                        b.label(),
                        i.accepts_text(),
                        i.name,
                        o.ty.label()
                    ),
                ));
            }
        }
        for n in &self.nodes {
            for (k, spec) in n.op.inputs().iter().enumerate() {
                let wired = self.links_into(n.id, k).len();
                if wired == 0 && !spec.optional {
                    f.errors.push((
                        Some(n.id),
                        format!(
                            "'{}' needs something on its '{}' input",
                            n.label(),
                            spec.name
                        ),
                    ));
                }
                if wired > 1 && !spec.many {
                    f.errors.push((
                        Some(n.id),
                        format!("'{}' takes one wire on '{}'", n.label(), spec.name),
                    ));
                }
            }
            for problem in n.op.param_problems() {
                f.errors
                    .push((Some(n.id), format!("{}: {problem}", n.label())));
            }
            let used = self.links.iter().any(|l| l.from.0 == n.id);
            if !used && !n.op.outputs().is_empty() && !n.op.kind().info().ends_a_branch {
                f.warnings.push((
                    Some(n.id),
                    format!("nothing uses what '{}' makes", n.label()),
                ));
            }
        }
        if let Err(e) = self.order() {
            f.errors.push((None, format!("{e:#}")));
        }
        if !self.nodes.iter().any(|n| n.op.kind().info().writes) {
            f.warnings.push((
                None,
                "no step writes anything to disk; add Export DICOM or Save report to keep \
                 the results"
                    .to_string(),
            ));
        }
        f
    }
}

/// Fill a folder or file name template: `{workflow}`, `{date}`, `{time}`,
/// `{input}`, and whatever else `extra` names. Characters no file system
/// takes in a name are replaced, path separators included, since a value is
/// one folder level.
pub fn fill_template(template: &str, vars: &[(&str, &str)]) -> String {
    let mut out = template.to_string();
    for (k, v) in vars {
        out = out.replace(&format!("{{{k}}}"), &safe_name(v));
    }
    out
}

/// Every step that names a folder of its own for what it writes - an
/// absolute path rather than one inside the run folder - by its label.
/// The servers that run workflows for others (the MCP server, the PACS
/// server) refuse those: everything a run writes there stays in its run
/// folder.
pub fn absolute_outputs(wf: &Workflow) -> Vec<String> {
    let mut out = Vec::new();
    for n in &wf.nodes {
        let folder = match &n.op {
            Op::ExportDicom(p) => &p.folder,
            Op::SaveReport(p) => &p.folder,
            Op::Anonymize(p) => &p.folder,
            Op::Drr(p) => &p.folder,
            _ => continue,
        };
        if std::path::Path::new(folder.trim()).is_absolute() {
            out.push(n.label());
        }
    }
    out
}

/// A string as one file or folder name.
pub fn safe_name(s: &str) -> String {
    let t: String = s
        .chars()
        .map(|c| match c {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
            c if c.is_control() => '_',
            c => c,
        })
        .collect();
    let t = t.trim().trim_end_matches('.').trim();
    if t.is_empty() {
        "_".into()
    } else {
        t.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::catalog::*;
    use super::*;

    fn two_rows() -> Workflow {
        let mut wf = Workflow::new("test");
        let a = wf.add(Op::LoadFolder(LoadFolder::default()), "A", [0.0, 0.0]);
        let b = wf.add(Op::SelectImage(SelectImage::default()), "", [200.0, 0.0]);
        let c = wf.add(Op::LoadFolder(LoadFolder::default()), "B", [0.0, 300.0]);
        let d = wf.add(Op::SelectGroup(SelectGroup::default()), "", [200.0, 300.0]);
        wf.link(a, 0, b, 0);
        wf.link(c, 0, d, 0);
        wf
    }

    #[test]
    fn a_copied_part_pastes_with_ids_of_its_own_and_its_inner_wires() {
        let mut wf = two_rows();
        // A wire across the rows, which a copy of one row leaves behind.
        wf.link(1, 0, 4, 0);
        wf.frames.push(Frame {
            title: "row A".into(),
            nodes: vec![1, 2],
            ..Frame::default()
        });
        wf.frames.push(Frame {
            title: "both".into(),
            nodes: vec![1, 2, 3, 4],
            ..Frame::default()
        });
        let part = wf.fragment(&[1, 2].into_iter().collect());
        assert_eq!(part.nodes.len(), 2);
        assert_eq!(
            part.links,
            vec![Link {
                from: (1, 0),
                to: (2, 0)
            }]
        );
        assert_eq!(part.frames.len(), 1, "only the frame around nothing else");
        // Through the clipboard and back.
        let part = Workflow::from_json(&part.to_json()).expect("reads back");

        let ids = wf.paste(&part, [10.0, 20.0]);
        assert_eq!(ids.get(&1), Some(&5));
        assert_eq!(ids.get(&2), Some(&6));
        assert_eq!(wf.nodes.len(), 6);
        assert_eq!(wf.node(5).map(|n| n.pos), Some([10.0, 20.0]));
        assert_eq!(wf.node(5).map(|n| n.title.as_str()), Some("A"));
        assert!(wf.links.contains(&Link {
            from: (5, 0),
            to: (6, 0)
        }));
        assert_eq!(wf.links.len(), 4, "the original three and the copy's one");
        assert_eq!(wf.frames.last().map(|f| f.nodes.clone()), Some(vec![5, 6]));
        assert!(wf.check().errors.iter().all(|(n, _)| *n != Some(6)));

        // The frames follow their nodes out.
        wf.nodes.retain(|n| n.id != 1 && n.id != 2);
        wf.prune_frames();
        assert_eq!(wf.frames.len(), 2, "row A is gone, both keeps 3 and 4");
        assert_eq!(wf.frames[0].nodes, vec![3, 4]);
    }

    #[test]
    fn a_workflow_survives_the_file_round_trip() {
        let mut wf = two_rows();
        wf.description = "two rows".into();
        if let Some(n) = wf.node_mut(1) {
            if let Op::LoadFolder(p) = &mut n.op {
                p.path = "D:/data/CCT".into();
            }
        }
        let text = wf.to_json();
        assert!(text.contains("\"kind\": \"load_folder\""), "{text}");
        assert!(text.contains("\"params\""), "{text}");
        let back = Workflow::from_json(&text).expect("reads back");
        let mut expect = wf.clone();
        expect.links.sort();
        assert_eq!(back, expect);
    }

    #[test]
    fn missing_parameters_take_their_defaults_and_unknown_kinds_are_named() {
        let text = r#"{"format":"rds-workflow","version":1,"name":"x",
            "nodes":[{"id":1,"kind":"select_image","params":{}}]}"#;
        let wf = Workflow::from_json(text).expect("defaults fill in");
        assert_eq!(wf.nodes[0].op, Op::SelectImage(SelectImage::default()));
        let bad = r#"{"format":"rds-workflow","version":1,"name":"x",
            "nodes":[{"id":1,"kind":"teleport","params":{}}]}"#;
        let e = format!("{:#}", Workflow::from_json(bad).unwrap_err());
        assert!(e.contains("teleport"), "{e}");
        let newer = r#"{"format":"rds-workflow","version":99,"name":"x"}"#;
        assert!(Workflow::from_json(newer).is_err());
        let other = r#"{"format":"something-else","version":1,"name":"x"}"#;
        assert!(Workflow::from_json(other).is_err());
    }

    #[test]
    fn independent_rows_run_top_to_bottom_and_dependencies_first() {
        let wf = two_rows();
        assert_eq!(wf.order().unwrap(), vec![1, 2, 3, 4]);
        // Moving the second row above the first changes which starts.
        let mut up = wf.clone();
        up.node_mut(3).unwrap().pos = [0.0, -500.0];
        up.node_mut(4).unwrap().pos = [200.0, -500.0];
        assert_eq!(up.order().unwrap(), vec![3, 4, 1, 2]);
        // A dependency beats the canvas: 2 below 1's row still waits for 1.
        let mut dep = wf.clone();
        dep.node_mut(1).unwrap().pos = [0.0, 900.0];
        let order = dep.order().unwrap();
        let at = |id| order.iter().position(|&x| x == id).unwrap();
        assert!(at(1) < at(2));
    }

    #[test]
    fn circles_and_wrong_wires_are_found() {
        let mut wf = two_rows();
        assert!(wf.would_cycle(2, 1));
        assert!(!wf.would_cycle(1, 4));
        wf.link(2, 0, 1, 0);
        assert!(wf.order().is_err());
        // An image into a node that takes a study.
        let mut wf = two_rows();
        wf.link(2, 0, 4, 0);
        let f = wf.check();
        assert!(!f.ok());
        assert!(f.node_has_error(4), "{:?}", f.errors);
    }

    #[test]
    fn a_node_without_its_required_input_is_an_error() {
        let mut wf = Workflow::new("x");
        wf.add(Op::SelectImage(SelectImage::default()), "", [0.0, 0.0]);
        let f = wf.check();
        assert!(f.node_has_error(1));
        assert!(f.of(1).iter().any(|m| m.contains("needs something")));
    }

    #[test]
    fn templates_fill_and_stay_one_folder_level() {
        let s = fill_template(
            "{workflow} {date}-{time}",
            &[
                ("workflow", "a/b: c"),
                ("date", "20260924"),
                ("time", "101500"),
            ],
        );
        assert_eq!(s, "a_b_ c 20260924-101500");
        assert_eq!(safe_name("  "), "_");
    }
}
