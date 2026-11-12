//! *Tools ▶ 🏥 PACS*: the Picture Archiving and Communication System as a
//! window.
//!
//! The PACS on this station is shown as it is laid out - cohorts, their
//! patients, each patient's studies and each study's series
//! ([`crate::archive`]) - and every row can be selected, one or many
//! (`pacs_tree.rs`), to be loaded into a workspace, renamed, put into
//! another cohort, moved into another folder or removed. The source row at
//! the top switches to a paired PACS server (`pacs_remote.rs`), shown the
//! same way. The *Settings* menu opens the storage folders the PACS reads
//! and the PACS server of this station (`pacs_server_win.rs`) as pages of
//! the window.
//!
//! Loading needs no special path: a selection is a list of DICOM files,
//! read by the same loader as *File ▶ Add DICOM file(s)*, with the same
//! merging and the same progress. Everything that writes (a scan, an
//! import, a name, a cohort, a move, a removal) runs as one background
//! job at a time; nothing here blocks a frame.
//!
//! A study holding a 4DCT says so on its row (🎞 and the number of scans,
//! `🎞?` while only probable) and lists each set as a row of its own
//! ([`pacs_tree`]), which loads the set; the filter beside the search
//! shows only the studies with one, and a set's menu confirms or rejects
//! it in the catalog.
//!
//! *Settings ▶ Export tags* writes what the person decided about the PACS
//! (the names given, the cohorts and who is in which, the 4D groups decided
//! by hand) into a JSON file, and *Import tags* merges such a file from
//! another station into this one's catalog ([`crate::archive::tags`]).

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::PathBuf;

use crate::archive::{Archive, ImportSummary, PatientEntry, Pick, StudyEntry, Tree, UNSORTED};
use crate::pacs::servers::Servers;

use super::pacs_remote::{self, PairDialog, RemoteView, Source};
use super::pacs_tree::{self, NodeId, TNode, TreeView};
use super::*;

/// The window's pages: the PACS itself, and what its menus open.
#[derive(Default)]
pub(super) enum Page {
    #[default]
    Studies,
    /// *Settings ▶ Storage folders*.
    Storage,
    /// *Settings ▶ PACS server*.
    Server,
    /// *Move the files to a folder*: choosing the folder.
    Move(MoveDialog),
}

/// Choosing where a selection's files go.
pub(super) struct MoveDialog {
    picks: Vec<Pick>,
    /// What is moved, in words.
    what: String,
    /// The storage folder (index) the files are in; they stay in it.
    root: usize,
    /// Its folders, once read.
    folders: Option<Vec<String>>,
    /// The folder chosen: a path below the storage folder, empty for the
    /// storage folder itself.
    dest: Option<String>,
    /// A new folder's name, made inside the chosen one.
    new_name: String,
}

/// A name being typed for a row.
pub(super) struct Renaming {
    target: Pick,
    /// The row as it reads now.
    title: String,
    text: String,
    /// The field takes the keyboard when the bar appears.
    fresh: bool,
}

/// A cohort being named; `then`: the patients to put in it once made
/// (`(from cohort, patient key)`, and copied rather than moved).
pub(super) struct NewCohort {
    name: String,
    then: Option<(Vec<(String, String)>, bool)>,
    fresh: bool,
}

/// What the PACS window is showing and doing.
pub(super) struct PacsWindow {
    /// The last scan; `None` until one has been made.
    pub tree: Option<Tree>,
    /// The tree as rows.
    pub nodes: Vec<TNode>,
    pub view: TreeView,
    pub status: Option<String>,
    /// This station's PACS, or a paired server (`pacs_remote.rs`).
    pub source: Source,
    /// The paired servers, as their file says.
    pub servers: Servers,
    /// What is shown of the selected server.
    pub remote: Option<RemoteView>,
    /// *Add server* / *Pair again*, while open.
    pub pairing: Option<PairDialog>,
    pub page: Page,
    pub renaming: Option<Renaming>,
    pub new_cohort: Option<NewCohort>,
    /// Asked once more before files are deleted: the selection and how
    /// many files it holds.
    pub confirm_remove: Option<(Vec<Pick>, usize)>,
    /// Which studies are shown, by what they hold.
    pub study_filter: StudyFilter,
    /// What the last scan found to pick studies by, with how many studies
    /// hold each.
    pub facets: Vec<(Facet, usize)>,
    /// The tree as the filter leaves it: what the rows are made of.
    pub shown: Tree,
    /// Counts the trees taken, so the analytics know theirs is stale.
    pub tree_gen: u64,
    /// The analytics panel is unfolded.
    pub stats_open: bool,
    /// The analytics last worked out, and for which tree and rows.
    pub stats: Option<super::pacs_stats::Stats>,
    pub stats_for: Option<(u64, Vec<NodeId>)>,
}

/// One thing a study can be picked by in the list beside 🔍: a 4D set, a
/// word of its patient's row (`Adult`, `Phantom`, `Female`: what the name
/// given says in brackets) under its [`WordKind`], or a modality one of
/// its series has (CT, MR, DX, RTSTRUCT, SEG, ...). They sort in that
/// order, the words kind by kind.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Facet {
    /// A confirmed 4DCT.
    FourD,
    /// A set only the geometry says is one.
    ProbableFourD,
    /// A word of [`patient_words`], matched without regard to case, and
    /// what it says ([`WordKind::of`]).
    Patient(WordKind, String),
    Modality(String),
}

impl Facet {
    fn label(&self) -> String {
        match self {
            Facet::FourD => "4D".into(),
            Facet::ProbableFourD => "probable 4D".into(),
            Facet::Patient(_, w) | Facet::Modality(w) => w.clone(),
        }
    }

    fn holds(&self, p: &PatientEntry, s: &StudyEntry) -> bool {
        match self {
            Facet::FourD => s.fourd.iter().any(|g| g.confirmed),
            Facet::ProbableFourD => s.fourd.iter().any(|g| !g.confirmed),
            Facet::Patient(_, w) => patient_words(p).iter().any(|x| x.eq_ignore_ascii_case(w)),
            Facet::Modality(m) => s.modalities.iter().any(|x| x == m),
        }
    }
}

/// The words in the brackets that end a patient's row, `P01-095 (Adult,
/// Thorax)` -> `Adult`, `Thorax`, split at commas, leaving out those that
/// only repeat the patient's name, ID or code (`P01-096 (P01-096)`, the
/// ID that ends a row with no name given).
pub(super) fn patient_words(p: &PatientEntry) -> Vec<String> {
    let title = p.title();
    let t = title.trim_end();
    let Some(inner) = t
        .strip_suffix(')')
        .and_then(|t| t.rfind('(').map(|i| (&t[..i], &t[i + 1..])))
    else {
        return Vec::new();
    };
    let (head, words) = inner;
    let same = |w: &str| {
        [head, p.name.as_str(), p.id.as_str(), p.key.as_str()]
            .iter()
            .any(|x| x.replace('^', " ").trim().eq_ignore_ascii_case(w))
    };
    let mut out: Vec<String> = Vec::new();
    for w in words.split(',').map(str::trim) {
        if !w.is_empty() && !same(w) && !out.iter().any(|x| x.eq_ignore_ascii_case(w)) {
            out.push(w.to_string());
        }
    }
    out
}

/// What a word of a patient's row says: the age group, the sex, the kind
/// of subject (a phantom, a volunteer), or - every word none of those
/// knows - the site (`Thorax`, `HN`, `Pancreas`). Words of one kind are
/// one chart of the analytics and one group of the list beside 🔍: `Adult`
/// and `Pancreas` are not shares of one whole.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum WordKind {
    Age,
    Sex,
    Site,
    Subject,
}

impl WordKind {
    /// The kind of `word`, by the words each kind knows (in any case).
    pub fn of(word: &str) -> WordKind {
        const AGE: &[&str] = &[
            "adult",
            "adults",
            "pediatric",
            "paediatric",
            "pediatrics",
            "paediatrics",
            "peds",
            "child",
            "children",
            "kid",
            "infant",
            "neonate",
            "neonatal",
            "newborn",
            "adolescent",
            "teen",
            "teenager",
            "juvenile",
            "young adult",
            "aya",
            "elderly",
            "geriatric",
        ];
        const SEX: &[&str] = &["male", "female", "m", "f", "man", "woman", "boy", "girl"];
        const SUBJECT: &[&str] = &[
            "volunteer",
            "healthy volunteer",
            "cadaver",
            "animal",
            "canine",
            "porcine",
            "mouse",
            "rat",
        ];
        let w = word.trim().to_lowercase();
        if AGE.contains(&w.as_str()) {
            WordKind::Age
        } else if SEX.contains(&w.as_str()) {
            WordKind::Sex
        } else if SUBJECT.contains(&w.as_str()) || w.contains("phantom") {
            WordKind::Subject
        } else {
            WordKind::Site
        }
    }

    pub fn title(self) -> &'static str {
        match self {
            WordKind::Age => "Age group",
            WordKind::Sex => "Sex",
            WordKind::Site => "Site",
            WordKind::Subject => "Subject",
        }
    }
}

/// Which studies the tree shows: every one while nothing is picked, else
/// those holding every picked facet (`all`) or one of them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct StudyFilter {
    pub picked: BTreeSet<Facet>,
    pub all: bool,
}

impl Default for StudyFilter {
    fn default() -> Self {
        StudyFilter {
            picked: BTreeSet::new(),
            all: true,
        }
    }
}

impl StudyFilter {
    fn is_off(&self) -> bool {
        self.picked.is_empty()
    }

    fn keeps(&self, p: &PatientEntry, s: &StudyEntry) -> bool {
        if self.all {
            self.picked.iter().all(|f| f.holds(p, s))
        } else {
            self.is_off() || self.picked.iter().any(|f| f.holds(p, s))
        }
    }

    /// `all studies`, `Pediatric + CT`, `MR or DX`, `4 kinds (all)`.
    fn label(&self) -> String {
        match self.picked.len() {
            0 => "all studies".into(),
            1..=3 => self
                .picked
                .iter()
                .map(Facet::label)
                .collect::<Vec<_>>()
                .join(if self.all { " + " } else { " or " }),
            n => format!("{n} kinds ({})", if self.all { "all" } else { "any" }),
        }
    }
}

/// What the studies of `tree` can be picked by: the 4D sets first, then
/// the words of the patients' rows, then the modalities, each by name and
/// with how many hold it - studies for the 4D sets and modalities,
/// patients for the words. A patient in two cohorts counts once; a word
/// written in two cases is one word, under its first spelling.
fn facets(tree: &Tree) -> Vec<(Facet, usize)> {
    let mut n: BTreeMap<Facet, usize> = BTreeMap::new();
    let mut spelled: BTreeMap<String, String> = BTreeMap::new();
    let (mut patients, mut studies) = (HashSet::new(), HashSet::new());
    for p in tree.cohorts.iter().flat_map(|c| c.patients.iter()) {
        let mut put = |f: Facet| *n.entry(f).or_default() += 1;
        if patients.insert(p.key()) {
            for w in patient_words(p) {
                let w = spelled.entry(w.to_lowercase()).or_insert(w).clone();
                put(Facet::Patient(WordKind::of(&w), w));
            }
        }
        for s in p
            .studies
            .iter()
            .filter(|s| studies.insert(s.study_uid.clone()))
        {
            for f in [Facet::FourD, Facet::ProbableFourD] {
                if f.holds(p, s) {
                    put(f);
                }
            }
            let mods: BTreeSet<&str> = s
                .modalities
                .iter()
                .map(|m| m.trim())
                .filter(|m| !m.is_empty())
                .collect();
            for m in mods {
                put(Facet::Modality(m.to_string()));
            }
        }
    }
    n.into_iter().collect()
}

/// The tree with only the studies `f` keeps (and the patients and cohorts
/// left with one).
fn filtered(tree: &Tree, f: &StudyFilter) -> Tree {
    if f.is_off() {
        return tree.clone();
    }
    let mut t = tree.clone();
    for c in &mut t.cohorts {
        for p in &mut c.patients {
            let whole = p.clone();
            p.studies.retain(|s| f.keeps(&whole, s));
        }
        c.patients.retain(|p| !p.studies.is_empty());
    }
    t.cohorts.retain(|c| !c.patients.is_empty());
    t
}

impl PacsWindow {
    fn new(servers: Servers) -> PacsWindow {
        PacsWindow {
            tree: None,
            nodes: Vec::new(),
            view: TreeView::default(),
            status: None,
            source: Source::Local,
            servers,
            remote: None,
            pairing: None,
            page: Page::Studies,
            renaming: None,
            new_cohort: None,
            confirm_remove: None,
            study_filter: StudyFilter::default(),
            facets: Vec::new(),
            shown: Tree::default(),
            tree_gen: 0,
            stats_open: true,
            stats: None,
            stats_for: None,
        }
    }

    /// Take a new tree, keeping what is unfolded and selected where those
    /// rows are still there, and what is picked in the list beside 🔍 where
    /// the new tree still offers it.
    pub(super) fn set_tree(&mut self, tree: Tree) {
        self.facets = facets(&tree);
        let offered: BTreeSet<&Facet> = self.facets.iter().map(|(f, _)| f).collect();
        self.study_filter.picked.retain(|f| offered.contains(f));
        self.shown = filtered(&tree, &self.study_filter);
        self.nodes = local_nodes(&self.shown);
        self.tree_gen += 1;
        self.view.retain_known(&self.nodes);
        if self.view.expanded.is_empty() {
            // Cohorts open at first: the patients are what one looks for.
            for c in &self.nodes {
                self.view.expanded.insert(vec![c.key.clone()]);
            }
        }
        self.tree = Some(tree);
    }
}

/// The files of PACS tags.
const TAGS_FILES: pick::Filter = pick::Filter {
    name: "PACS tags (JSON)",
    exts: &["json"],
};

/// What a background PACS job answers with.
pub(super) enum PacsOutcome {
    /// A scan: the folders walked and the catalog brought up to date.
    Scanned(Tree),
    /// The catalog as it was: the folders not walked.
    Listed(Tree),
    Imported(ImportSummary, Tree),
    /// A name, a cohort, a move or a removal: what was done, and the PACS
    /// as it is now.
    Changed(String, Tree),
    /// The folders of a storage folder, for the move page; with the folder
    /// just made, if one was.
    Folders(usize, Vec<String>, Option<String>),
}

/// A count with its noun: `1 study`, `3 studies`.
fn count(n: usize, one: &str, many: &str) -> String {
    format!("{n} {}", if n == 1 { one } else { many })
}

/// The PACS as rows: cohorts, patients, studies, series.
fn local_nodes(tree: &Tree) -> Vec<TNode> {
    tree.cohorts
        .iter()
        .map(|c| TNode {
            weak: false,
            key: c.id.clone(),
            icon: "📁",
            text: c.name.clone(),
            right: format!(
                "{} · {}",
                count(c.patients.len(), "patient", "patients"),
                count(c.files(), "file", "files")
            ),
            hover: match &c.folder {
                Some(f) => format!("Cohort folder {}", f.display()),
                None if c.is_unsorted() => {
                    "Patients outside any cohort folder, and those taken out of a cohort".into()
                }
                None => "A cohort made in this window: its patients' files lie elsewhere".into(),
            },
            files: c.files(),
            children: c
                .patients
                .iter()
                .map(|p| TNode {
                    weak: false,
                    key: p.key.clone(),
                    icon: "👤",
                    text: p.title(),
                    right: format!(
                        "{} · {}",
                        count(p.studies.len(), "study", "studies"),
                        count(p.files(), "file", "files")
                    ),
                    hover: format!(
                        "{}{}\n{}",
                        p.dicom_title(),
                        if p.label.is_empty() {
                            String::new()
                        } else {
                            " (DICOM name)".into()
                        },
                        p.dir.display()
                    ),
                    files: p.files(),
                    children: p
                        .studies
                        .iter()
                        .map(|s| {
                            let (badge, about) = pacs_tree::fourd_badge(&s.fourd, &s.fourd_orphans);
                            TNode {
                                weak: false,
                                key: s.study_uid.clone(),
                                icon: "📂",
                                text: s.title(),
                                right: format!(
                                    "{badge}{} · {}",
                                    count(s.series.len(), "series", "series"),
                                    count(s.files, "file", "files")
                                ),
                                hover: format!(
                                    "{}Study UID {}\n{}{about}",
                                    if s.label.is_empty() {
                                        String::new()
                                    } else {
                                        format!("{}\n", s.dicom_title())
                                    },
                                    s.study_uid,
                                    s.dir.display()
                                ),
                                files: s.files,
                                children: pacs_tree::study_rows(
                                    s.series
                                        .iter()
                                        .map(|se| pacs_tree::SeriesRow {
                                            modality: se.modality.clone(),
                                            datetime: se.datetime.clone(),
                                            fourd: se.fourd.clone(),
                                            node: TNode {
                                                weak: false,
                                                key: se.series_uid.clone(),
                                                icon: "🎞",
                                                text: se.title(),
                                                right: count(se.files, "file", "files"),
                                                hover: format!(
                                                    "{}Series UID {}\n{}",
                                                    if se.label.is_empty() {
                                                        String::new()
                                                    } else {
                                                        format!("{}\n", se.dicom_title())
                                                    },
                                                    se.series_uid,
                                                    se.dir.display()
                                                ),
                                                files: se.files,
                                                children: Vec::new(),
                                            },
                                        })
                                        .collect(),
                                    &s.fourd,
                                ),
                            }
                        })
                        .collect(),
                })
                .collect(),
        })
        .collect()
}

/// What a row of this station's PACS stands for.
fn pick_of(id: &NodeId) -> Option<Pick> {
    match id.len() {
        1 => Some(Pick::Cohort(id[0].clone())),
        2 => Some(Pick::Patient(id[1].clone())),
        3 => Some(Pick::Study(id[2].clone())),
        4 => Some(Pick::Series(id[3].clone())),
        _ => None,
    }
}

/// The patients a set of rows means, as `(cohort, patient key)`: a
/// patient row is itself, a cohort row all its patients, a study or series
/// row its patient.
fn patients_of(nodes: &[TNode], ids: &[NodeId]) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    for id in ids {
        let mut add = |c: &str, k: &str| {
            let e = (c.to_string(), k.to_string());
            if !out.contains(&e) {
                out.push(e);
            }
        };
        match id.len() {
            1 => {
                if let Some(c) = pacs_tree::node_at(nodes, id) {
                    for p in &c.children {
                        add(&id[0], &p.key);
                    }
                }
            }
            n if n >= 2 => add(&id[0], &id[1]),
            _ => {}
        }
    }
    out
}

/// What a click in the window asks for; carried out after it is drawn.
enum Act {
    Rescan,
    Import,
    Send(usize),
    Load(usize),
    Page(Page),
    RenameStart(NodeId),
    /// A 4D set: confirmed (`Some(true)`), rejected (`Some(false)`), or what
    /// was decided forgotten (`None`).
    Decide4d(String, Option<bool>),
    Filter(StudyFilter),
    /// *Settings ▶ Export tags* / *Import tags*.
    TagsExport,
    TagsImport,
    RenameSave(Option<String>),
    RenameCancel,
    NewCohortStart(Option<(Vec<(String, String)>, bool)>),
    NewCohortSave,
    NewCohortCancel,
    PutInCohort(Vec<(String, String)>, String, bool),
    TakeOut(Vec<(String, String)>),
    ResetCohorts(Vec<String>),
    DeleteCohort(String),
    AskRemove(Vec<Pick>),
    Remove,
    RemoveCancel,
    MoveStart(Vec<Pick>, String),
    MoveDo,
    NewFolder,
    StorageAdd,
    StorageChangeMain,
    StorageMain(usize),
    StorageDrop(usize),
}

impl ViewerApp {
    pub(super) fn open_pacs_window(&mut self) {
        self.raise_window("pacs");
        if self.pacs.is_none() {
            let servers = self.reload_servers();
            self.pacs = Some(PacsWindow::new(servers));
            if self.pacs_scan_on_open {
                self.start_pacs_scan();
            } else {
                // What the catalog last saw, at once; a PACS with no
                // catalog yet is walked all the same, or there is nothing
                // to show.
                self.pacs_job_with("Reading the catalog", |a, p| {
                    if a.catalog_path().is_file() {
                        Ok(PacsOutcome::Listed(a.tree()))
                    } else {
                        a.scan_tree(p).map(PacsOutcome::Scanned)
                    }
                });
            }
        }
    }

    /// Run `work` on this station's PACS as the one PACS job.
    fn pacs_job_with(
        &mut self,
        what: &str,
        work: impl FnOnce(&Archive, &Progress) -> anyhow::Result<PacsOutcome> + Send + 'static,
    ) {
        if self.pacs_job.is_some() {
            if let Some(w) = self.pacs.as_mut() {
                w.status = Some("⚠ the PACS is busy; try again when it is done".into());
            }
            return;
        }
        let archive = self.station_pacs();
        let progress = Arc::new(Progress::default());
        progress.set(what);
        self.pacs_job = Some(Job::spawn(progress, move |p| work(&archive, p)));
    }

    pub(super) fn start_pacs_scan(&mut self) {
        self.pacs_job_with("Reading the PACS", |a, p| {
            a.scan_tree(p).map(PacsOutcome::Scanned)
        });
    }

    fn start_pacs_import(&mut self, src: PathBuf) {
        self.pacs_job_with("Importing", move |a, p| {
            let sum = a.import(&src, p)?;
            Ok(PacsOutcome::Imported(sum, a.tree()))
        });
    }

    /// A PACS job finished.
    pub(super) fn on_pacs_done(&mut self, outcome: PacsOutcome) {
        let Some(w) = &mut self.pacs else {
            return;
        };
        match outcome {
            PacsOutcome::Scanned(tree) => {
                w.status = Some(format!(
                    "{}, {}, {}",
                    count(tree.cohorts.len(), "cohort", "cohorts"),
                    count(tree.patients().len(), "patient", "patients"),
                    count(tree.studies(), "study", "studies")
                ));
                w.set_tree(tree);
            }
            PacsOutcome::Listed(tree) => {
                w.status = Some(format!(
                    "{}, {}, {} - as the catalog last saw them; ⟲ Rescan looks in the folders",
                    count(tree.cohorts.len(), "cohort", "cohorts"),
                    count(tree.patients().len(), "patient", "patients"),
                    count(tree.studies(), "study", "studies")
                ));
                w.set_tree(tree);
            }
            PacsOutcome::Imported(sum, tree) => {
                w.status = Some(format!("✔ imported: {}", sum.describe()));
                w.set_tree(tree);
            }
            PacsOutcome::Changed(msg, tree) => {
                w.status = Some(msg);
                w.set_tree(tree);
            }
            PacsOutcome::Folders(root, folders, made) => {
                if let Page::Move(d) = &mut w.page {
                    if d.root == root {
                        d.folders = Some(folders);
                        if made.is_some() {
                            d.dest = made;
                            d.new_name.clear();
                        }
                    }
                }
            }
        }
    }

    pub(super) fn pacs_window(&mut self, ctx: &egui::Context) {
        if self.pacs.is_none() {
            return;
        }
        let mut open = true;
        let mut acts: Vec<Act> = Vec::new();
        let busy = self.pacs_job.is_some();
        let loaded: [bool; MAX_WORKSPACES] = std::array::from_fn(|s| self.slots[s].study.is_some());
        // Which workspaces a selection can be read into: the open ones and
        // one new letter, so the row grows with the work rather than showing
        // four buttons on the first run.
        let targets = self.open_plus_new();
        let mut w = self.pacs.take().expect("checked above");
        let mut ractions: Vec<pacs_remote::Action> = Vec::new();
        let rbusy = self.pacs_remote_job.is_some();
        let rprogress = self.pacs_remote_job.as_ref().map(|j| j.progress.clone());
        let job_progress = self.pacs_job.as_ref().map(|j| j.progress.clone());
        let storage = self.station_pacs().roots().to_vec();
        let mut server = self.pacs_server.take();
        let server_busy = self.pacs_server_job.is_some();
        let server_msg = self.pacs_server_job.as_ref().map(|j| j.progress.get());
        let mut server_clicks = None;
        let mut scan_on_open = self.pacs_scan_on_open;
        let mut stats_auto = self.pacs_stats_auto;
        let mut stats_pie = self.pacs_stats_pie;

        detach::tool_window(
            ctx,
            "pacs",
            "🏥 Picture Archiving and Communication System (PACS)",
            &mut open,
            detach::WinOpts::size(820.0, 640.0),
            |ui| {
                egui::MenuBar::new().ui(ui, |ui| {
                    ui.menu_button("⚙ Settings", |ui| {
                        if tip_button(
                            ui,
                            "📁 Storage folders",
                            "The folders the PACS reads; new data goes into the first",
                        ) {
                            acts.push(Act::Page(Page::Storage));
                            ui.close();
                        }
                        ui.checkbox(&mut scan_on_open, "⟲ Rescan the folders on opening")
                            .on_hover_text(
                                "Look through the storage folders for new and changed files                                  each time this window is first opened. Off: the PACS is listed                                  at once as the catalog last saw it, and the folders are read                                  only on ⟲ Rescan (and when there is no catalog yet).",
                            );
                        ui.checkbox(&mut stats_auto, "📊 Recalculate the analytics on every click")
                            .on_hover_text(
                                "Work the analytics panel out again each time the selection \
                                 in the tree changes. Off: only on its ⟳ button, the figures \
                                 marked stale meanwhile.",
                            );
                        ui.separator();
                        if tip_button(
                            ui,
                            "📤 Export tags…",
                            "Write the names given to patients, studies and series, the cohorts and who is in which, and the 4D groups decided by hand into a JSON file, to take them to another station",
                        ) {
                            acts.push(Act::TagsExport);
                            ui.close();
                        }
                        if tip_button(
                            ui,
                            "📥 Import tags…",
                            "Merge a file of tags from another station into this PACS: names go to the patients, studies and series held here, cohorts are found by name or made, 4D decisions kept; what is not held here is counted and left",
                        ) {
                            acts.push(Act::TagsImport);
                            ui.close();
                        }
                        ui.separator();
                        // The server runs on a PC or a Mac; a tablet or a
                        // phone is always a client.
                        if cfg!(not(any(target_os = "android", target_os = "ios")))
                            && tip_button(
                                ui,
                                "🖥 PACS server",
                                "Serve this station's PACS to other stations: start and \
                                 stop, connection details, pairing codes, its settings",
                            )
                        {
                            acts.push(Act::Page(Page::Server));
                            ui.close();
                        }
                    });
                    ui.menu_button("Help", |ui| {
                        ui.label("Selecting rows:");
                        ui.weak("Click - select one row");
                        ui.weak("Ctrl + click - add to or take from the selection");
                        ui.weak("Shift + click - select a range");
                        ui.weak("Up / Down - move (Shift: extend the selection)");
                        ui.weak("Left / Right - fold / unfold");
                        ui.weak("Ctrl + A - select every row shown");
                        ui.weak("Escape - clear the selection");
                        ui.weak("F2 - rename, Delete - remove");
                        ui.weak("Right click - what can be done with the selection");
                    });
                });

                match &mut w.page {
                    Page::Storage => {
                        storage_page(ui, &storage, busy, &mut acts);
                        return;
                    }
                    Page::Server => {
                        if ui.button("◀ Back to the PACS").clicked() {
                            acts.push(Act::Page(Page::Studies));
                        }
                        ui.separator();
                        egui::ScrollArea::vertical()
                            .id_salt("pacs_server_page")
                            .show(ui, |ui| match server.as_mut() {
                                Some(s) => {
                                    server_clicks = Some(super::pacs_server_win::server_body(
                                        ui,
                                        s,
                                        server_busy,
                                        server_msg.clone(),
                                    ));
                                }
                                None => {
                                    ui.weak("reading");
                                }
                            });
                        return;
                    }
                    Page::Move(d) => {
                        move_page(ui, d, busy, &mut acts);
                        if let Some(p) = &job_progress {
                            progress_row(ui, p);
                        }
                        return;
                    }
                    Page::Studies => {}
                }

                // The analytics, on the right of everything below the menu,
                // for this station's PACS.
                if matches!(w.source, Source::Local) && w.pairing.is_none() {
                    let mut clicked = None;
                    let open = w.stats_open;
                    w.stats_open = ViewerApp::edge_panel(
                        ui,
                        super::panels::EdgePanel {
                            id: "pacs_stats",
                            left: false,
                            width: 260.0,
                            name: "analytics",
                            key: "",
                        },
                        open,
                        |ui| {
                            clicked = super::pacs_stats::panel_body(
                                ui,
                                &mut w,
                                stats_auto,
                                &mut stats_pie,
                            );
                        },
                    );
                    use super::pacs_stats::PanelClick;
                    if let Some(PanelClick::Reset) = clicked {
                        reset_filters(&mut w, &mut acts);
                    }
                    if let Some(PanelClick::Toggle(f)) = clicked {
                        // A word as the filter spells it (`male` in the
                        // scope may be `Male` across the PACS).
                        let f = w
                            .facets
                            .iter()
                            .map(|(o, _)| o)
                            .find(|o| match (o, &f) {
                                (Facet::Patient(_, a), Facet::Patient(_, b)) => {
                                    a.eq_ignore_ascii_case(b)
                                }
                                (o, f) => *o == f,
                            })
                            .cloned()
                            .unwrap_or(f);
                        let mut sf = w.study_filter.clone();
                        if !sf.picked.remove(&f) {
                            sf.picked.insert(f);
                        }
                        acts.push(Act::Filter(sf));
                    }
                }

                let rctx = pacs_remote::Ctx {
                    servers: &w.servers,
                    busy: rbusy,
                    progress: rprogress.as_deref(),
                    loaded,
                    targets: targets.clone(),
                };
                pacs_remote::source_row(ui, &w, &rctx, &mut ractions);
                ui.separator();
                if let Some(d) = w.pairing.as_mut() {
                    pacs_remote::pairing_ui(ui, d, &rctx, &mut ractions);
                    return;
                }
                if let Source::Remote(id) = &w.source {
                    match (w.servers.get(id), w.remote.as_mut()) {
                        (Some(entry), Some(v)) => {
                            pacs_remote::remote_ui(ui, v, entry, &rctx, &mut ractions)
                        }
                        _ => {
                            ui.weak("This server is no longer paired.");
                        }
                    }
                    return;
                }
                local_body(ui, &mut w, busy, &job_progress, loaded, &targets, &mut acts);
            },
        );

        if (scan_on_open, stats_auto, stats_pie)
            != (
                self.pacs_scan_on_open,
                self.pacs_stats_auto,
                self.pacs_stats_pie,
            )
        {
            self.pacs_scan_on_open = scan_on_open;
            self.pacs_stats_auto = stats_auto;
            self.pacs_stats_pie = stats_pie;
            self.persist_settings();
        }
        if !open {
            // Closing the window stops asking the server how it is.
            self.pacs_server = None;
            return;
        }
        self.pacs_server = server;
        self.pacs = Some(w);
        if let Some(k) = server_clicks {
            self.server_clicks(ctx, k);
        }
        for a in acts {
            self.pacs_act(a);
        }
        if !ractions.is_empty() {
            self.remote_actions(ractions);
        }
    }

    /// Carry out one click of the window.
    fn pacs_act(&mut self, a: Act) {
        let archive = self.station_pacs();
        match a {
            Act::Rescan => self.start_pacs_scan(),
            Act::Import => {
                self.ask_folder("Folder to file into the PACS", |app, dir| {
                    app.start_pacs_import(dir);
                });
            }
            Act::Send(slot) => self.open_pacs_save(slot, None),
            Act::Load(slot) => {
                let picks: Vec<Pick> = match &self.pacs {
                    Some(w) => pacs_tree::series_ids(&w.nodes, &w.view.tops())
                        .iter()
                        .filter_map(pick_of)
                        .collect(),
                    None => return,
                };
                let files: Vec<PathBuf> = archive
                    .files_of(&picks)
                    .into_iter()
                    .map(|i| i.path)
                    .collect();
                if files.is_empty() {
                    if let Some(w) = self.pacs.as_mut() {
                        w.status = Some("⚠ the selection holds no file".into());
                    }
                    return;
                }
                let sources = archive.sources_of(&files);
                self.start_load_files_noting(slot, files, Some(sources));
            }
            Act::Page(p) => {
                if matches!(p, Page::Server) {
                    self.open_pacs_server_window();
                } else {
                    self.pacs_server = None;
                }
                if let Some(w) = self.pacs.as_mut() {
                    w.page = p;
                }
            }
            Act::RenameStart(id) => {
                let Some(w) = self.pacs.as_mut() else {
                    return;
                };
                let Some(target) = pacs_tree::series_ids(&w.nodes, std::slice::from_ref(&id))
                    .first()
                    .filter(|_| !id.last().is_some_and(|k| pacs_tree::is_group_row(k)))
                    .and_then(pick_of)
                else {
                    return;
                };
                let title = pacs_tree::node_at(&w.nodes, &id)
                    .map(|n| n.text.clone())
                    .unwrap_or_default();
                w.renaming = Some(Renaming {
                    target,
                    text: title.clone(),
                    title,
                    fresh: true,
                });
            }
            Act::Decide4d(group, yes) => {
                self.pacs_job_with("Keeping the decision", move |a, _| {
                    let msg = match yes {
                        Some(true) => {
                            a.confirm_4d(&group, true)?;
                            "✔ confirmed as a 4DCT"
                        }
                        Some(false) => {
                            a.confirm_4d(&group, false)?;
                            "✔ not a 4DCT: the set is dissolved"
                        }
                        None => {
                            a.reset_4d_group(&group)?;
                            "✔ the headers decide about the set again"
                        }
                    };
                    Ok(PacsOutcome::Changed(msg.into(), a.tree()))
                });
            }
            Act::TagsExport => {
                let (date, _) = crate::dicom_export::today();
                self.ask_save(
                    "Export the PACS tags",
                    format!("rds-pacs-tags-{date}.json"),
                    None,
                    Some(TAGS_FILES),
                    |app, path| {
                        app.pacs_job_with("Exporting the tags", move |a, _| {
                            let sum = a.export_tags(&path)?;
                            Ok(PacsOutcome::Changed(
                                format!("✔ {} exported to {}", sum.describe(), path.display()),
                                a.tree(),
                            ))
                        });
                    },
                );
            }
            Act::TagsImport => {
                self.ask_file("Import PACS tags", None, Some(TAGS_FILES), |app, path| {
                    app.pacs_job_with("Importing the tags", move |a, _| {
                        let sum = a.import_tags(&path)?;
                        Ok(PacsOutcome::Changed(
                            format!("✔ imported {}", sum.describe()),
                            a.tree(),
                        ))
                    });
                });
            }
            Act::Filter(f) => {
                if let Some(w) = self.pacs.as_mut() {
                    w.study_filter = f;
                    if let Some(t) = w.tree.take() {
                        w.set_tree(t);
                    }
                }
            }
            Act::RenameCancel => {
                if let Some(w) = self.pacs.as_mut() {
                    w.renaming = None;
                }
            }
            Act::RenameSave(name) => {
                let Some(r) = self.pacs.as_mut().and_then(|w| w.renaming.take()) else {
                    return;
                };
                let name = name.unwrap_or_default();
                self.pacs_job_with("Renaming", move |a, _| {
                    a.rename(&r.target, &name)?;
                    Ok(PacsOutcome::Changed(
                        if name.trim().is_empty() {
                            format!("✔ '{}' has its own name again", r.title)
                        } else {
                            format!("✔ '{}' is now '{}'", r.title, name.trim())
                        },
                        a.tree(),
                    ))
                });
            }
            Act::NewCohortStart(then) => {
                if let Some(w) = self.pacs.as_mut() {
                    w.new_cohort = Some(NewCohort {
                        name: String::new(),
                        then,
                        fresh: true,
                    });
                }
            }
            Act::NewCohortCancel => {
                if let Some(w) = self.pacs.as_mut() {
                    w.new_cohort = None;
                }
            }
            Act::NewCohortSave => {
                let Some(n) = self.pacs.as_mut().and_then(|w| w.new_cohort.take()) else {
                    return;
                };
                self.pacs_job_with("Making the cohort", move |a, _| {
                    let id = a.new_cohort(&n.name)?;
                    if let Some((members, copy)) = &n.then {
                        put_in(a, members, &id, *copy)?;
                    }
                    Ok(PacsOutcome::Changed(
                        format!("✔ cohort '{}' made", n.name.trim()),
                        a.tree(),
                    ))
                });
            }
            Act::PutInCohort(members, to, copy) => {
                self.pacs_job_with("Changing cohorts", move |a, _| {
                    put_in(a, &members, &to, copy)?;
                    Ok(PacsOutcome::Changed(
                        format!(
                            "✔ {} {}; no file moved",
                            count(members.len(), "patient", "patients"),
                            if copy { "copied" } else { "moved" }
                        ),
                        a.tree(),
                    ))
                });
            }
            Act::TakeOut(members) => {
                self.pacs_job_with("Changing cohorts", move |a, _| {
                    let mut by_from: BTreeMap<String, Vec<String>> = BTreeMap::new();
                    for (from, key) in &members {
                        by_from.entry(from.clone()).or_default().push(key.clone());
                    }
                    for (from, keys) in by_from {
                        a.take_out_of_cohort(&keys, &from)?;
                    }
                    Ok(PacsOutcome::Changed(
                        format!(
                            "✔ {} taken out of the cohort",
                            count(members.len(), "patient", "patients")
                        ),
                        a.tree(),
                    ))
                });
            }
            Act::ResetCohorts(keys) => {
                self.pacs_job_with("Changing cohorts", move |a, _| {
                    a.reset_cohorts(&keys)?;
                    Ok(PacsOutcome::Changed(
                        format!(
                            "✔ {} back in the cohorts their folders say",
                            count(keys.len(), "patient", "patients")
                        ),
                        a.tree(),
                    ))
                });
            }
            Act::DeleteCohort(id) => {
                self.pacs_job_with("Deleting the cohort", move |a, _| {
                    a.delete_cohort(&id)?;
                    Ok(PacsOutcome::Changed("✔ cohort deleted".into(), a.tree()))
                });
            }
            Act::AskRemove(picks) => {
                let picks: Vec<Pick> = picks
                    .into_iter()
                    .filter(|p| !matches!(p, Pick::Cohort(_)))
                    .collect();
                let n = archive.files_of(&picks).len();
                if let Some(w) = self.pacs.as_mut() {
                    if picks.is_empty() {
                        w.status = Some(
                            "A cohort is not removed with its files: select its patients to \
                             remove them"
                                .into(),
                        );
                    } else {
                        w.confirm_remove = Some((picks, n));
                    }
                }
            }
            Act::RemoveCancel => {
                if let Some(w) = self.pacs.as_mut() {
                    w.confirm_remove = None;
                }
            }
            Act::Remove => {
                let Some((picks, _)) = self.pacs.as_mut().and_then(|w| w.confirm_remove.take())
                else {
                    return;
                };
                self.pacs_job_with("Removing", move |a, _| {
                    let n = a.remove_picks(&picks)?;
                    Ok(PacsOutcome::Changed(
                        format!("✔ {} removed from the PACS", count(n, "file", "files")),
                        a.tree(),
                    ))
                });
            }
            Act::MoveStart(picks, what) => {
                let Some(root) = archive.storage_of(&picks) else {
                    if let Some(w) = self.pacs.as_mut() {
                        w.status = Some(
                            "⚠ the selection lies in more than one storage folder; files move \
                             only within one"
                                .into(),
                        );
                    }
                    return;
                };
                if let Some(w) = self.pacs.as_mut() {
                    w.page = Page::Move(MoveDialog {
                        picks,
                        what,
                        root,
                        folders: None,
                        dest: None,
                        new_name: String::new(),
                    });
                }
                self.pacs_job_with("Reading the folders", move |a, _| {
                    Ok(PacsOutcome::Folders(root, a.folders(root), None))
                });
            }
            Act::NewFolder => {
                let Some((root, parent, name)) = self.pacs.as_ref().and_then(|w| match &w.page {
                    Page::Move(d) => Some((
                        d.root,
                        d.dest.clone().unwrap_or_default(),
                        d.new_name.clone(),
                    )),
                    _ => None,
                }) else {
                    return;
                };
                self.pacs_job_with("Making the folder", move |a, _| {
                    let rel = a.new_folder(root, &parent, &name)?;
                    Ok(PacsOutcome::Folders(root, a.folders(root), Some(rel)))
                });
            }
            Act::MoveDo => {
                let Some((picks, root, dest)) = self.pacs.as_ref().and_then(|w| match &w.page {
                    Page::Move(d) => Some((d.picks.clone(), d.root, d.dest.clone()?)),
                    _ => None,
                }) else {
                    return;
                };
                if let Some(w) = self.pacs.as_mut() {
                    w.page = Page::Studies;
                }
                self.pacs_job_with("Moving the files", move |a, p| {
                    p.set("Moving the files");
                    let sum = a.move_into(&picks, root, &dest)?;
                    let place = if dest.is_empty() {
                        a.roots()[root].display().to_string()
                    } else {
                        dest.clone()
                    };
                    Ok(PacsOutcome::Changed(
                        format!(
                            "✔ {} moved into {place}{}{}",
                            count(sum.moved, "file", "files"),
                            if sum.unchanged > 0 {
                                format!(", {} there already", sum.unchanged)
                            } else {
                                String::new()
                            },
                            if sum.folders_removed > 0 {
                                format!(
                                    ", {} left empty removed",
                                    count(sum.folders_removed, "folder", "folders")
                                )
                            } else {
                                String::new()
                            }
                        ),
                        a.tree(),
                    ))
                });
            }
            Act::StorageAdd => {
                self.ask_folder("A further storage folder for the PACS", |app, dir| {
                    let main = crate::archive::root_from_setting(&app.archive_dir);
                    if dir != main && !app.archive_more.contains(&dir) {
                        app.archive_more.push(dir);
                        app.persist_settings();
                        app.start_pacs_scan();
                    }
                });
            }
            Act::StorageChangeMain => {
                self.ask_folder("The main storage folder of the PACS", |app, dir| {
                    // The cohorts and names given go along.
                    let old = app.station_pacs();
                    if let Err(e) = old.carry_catalog_to(&dir) {
                        app.error = Some(format!("PACS: {e:#}"));
                        return;
                    }
                    let gone = old.root().to_path_buf();
                    app.archive_more.retain(|m| m != &dir);
                    app.archive_dir = dir.display().to_string();
                    app.persist_settings();
                    if let Some(w) = app.pacs.as_mut() {
                        w.tree = None;
                        w.nodes.clear();
                    }
                    app.pacs_job_with("Reading the PACS", move |a, p| {
                        a.forget_storage(&gone)?;
                        a.scan_tree(p).map(PacsOutcome::Scanned)
                    });
                });
            }
            Act::StorageMain(i) => {
                // The folder chosen becomes the first; the old main one
                // stays in the list.
                let roots = archive.roots().to_vec();
                if let Some(new_main) = roots.get(i).cloned() {
                    if let Err(e) = archive.carry_catalog_to(&new_main) {
                        self.error = Some(format!("PACS: {e:#}"));
                        return;
                    }
                    let mut more: Vec<PathBuf> =
                        roots.into_iter().filter(|r| r != &new_main).collect();
                    more.dedup();
                    self.archive_dir = new_main.display().to_string();
                    self.archive_more = more;
                    self.persist_settings();
                    if let Some(w) = self.pacs.as_mut() {
                        w.status = Some(format!(
                            "✔ {} is the main storage folder now; the catalog went with it",
                            new_main.display()
                        ));
                    }
                    self.start_pacs_scan();
                }
            }
            Act::StorageDrop(i) => {
                let roots = archive.roots().to_vec();
                if i == 0 || i >= roots.len() {
                    return;
                }
                let gone = roots[i].clone();
                self.archive_more.retain(|m| m != &gone);
                self.persist_settings();
                self.pacs_job_with("Updating the catalog", move |a, p| {
                    a.forget_storage(&gone)?;
                    a.scan_tree(p).map(PacsOutcome::Scanned)
                });
            }
        }
    }
}

/// Put patients into a cohort: moved out of the cohort they were picked
/// in, or copied.
fn put_in(a: &Archive, members: &[(String, String)], to: &str, copy: bool) -> anyhow::Result<()> {
    let mut by_from: BTreeMap<&str, Vec<String>> = BTreeMap::new();
    for (from, key) in members {
        by_from.entry(from.as_str()).or_default().push(key.clone());
    }
    for (from, keys) in by_from {
        a.put_in_cohort(&keys, Some(from), to, copy)?;
    }
    Ok(())
}

/// Clear the search text and untick everything in the list beside 🔍.
fn reset_filters(w: &mut PacsWindow, acts: &mut Vec<Act>) {
    w.view.filter.clear();
    acts.push(Act::Filter(StudyFilter::default()));
}

/// The list beside 🔍: what the last scan found the studies to hold (4D
/// sets, the patients' words, the modalities of their series), ticked as
/// many as wanted;
/// it stays open while ticking, so several are picked in one visit.
fn study_filter_list(ui: &mut egui::Ui, w: &PacsWindow, acts: &mut Vec<Act>) {
    let mut f = w.study_filter.clone();
    egui::ComboBox::from_id_salt("pacs_study_filter")
        .selected_text(f.label())
        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
        .show_ui(ui, |ui| {
            if w.facets.is_empty() {
                ui.weak("Nothing scanned yet");
                return;
            }
            ui.horizontal(|ui| {
                ui.label("Studies with");
                ui.selectable_value(&mut f.all, true, "all")
                    .on_hover_text("Every ticked kind: CT + RTSTRUCT is a study holding both");
                ui.selectable_value(&mut f.all, false, "any")
                    .on_hover_text("One of the ticked kinds at least: MR or DX");
                ui.label("of:");
            });
            ui.separator();
            // The 4D sets, the patients' words kind by kind, the
            // modalities: a line and a heading for each group.
            let group = |x: &Facet| match x {
                Facet::FourD | Facet::ProbableFourD => (0, "4D sets"),
                Facet::Patient(k, _) => (1 + *k as usize, k.title()),
                Facet::Modality(_) => (9, "Modality"),
            };
            let mut last = None;
            for (x, n) in &w.facets {
                let (g, title) = group(x);
                if last != Some(g) {
                    if last.is_some() {
                        ui.separator();
                    }
                    ui.weak(title);
                }
                last = Some(g);
                let mut on = f.picked.contains(x);
                let r = ui.checkbox(&mut on, format!("{}  ({n})", x.label()));
                let r = match x {
                    Facet::FourD => r.on_hover_text("A 4DCT the catalog found from the headers"),
                    Facet::ProbableFourD => r.on_hover_text(
                        "One geometry and nothing else says 4D: to confirm or reject",
                    ),
                    Facet::Patient(..) => r.on_hover_text(
                        "Every study of the patients whose row says this in its brackets \
                         (the count is of patients)",
                    ),
                    Facet::Modality(_) => r,
                };
                if r.changed() {
                    if on {
                        f.picked.insert(x.clone());
                    } else {
                        f.picked.remove(x);
                    }
                }
            }
            ui.separator();
            if ui
                .add_enabled(!f.picked.is_empty(), egui::Button::new("Show all studies"))
                .clicked()
            {
                f.picked.clear();
            }
        })
        .response
        .on_hover_text(
            "Only the studies holding what is ticked: a 4DCT as the catalog finds it from the \
             headers, a word in the brackets of their patient's row (Adult, Phantom, Female, \
             ...), or a modality one of their series has. The kinds offered are those the last \
             scan found, with how many hold each.",
        );
    if f != w.study_filter {
        acts.push(Act::Filter(f));
    }
}

/// The body for this station's PACS: the buttons, the tree, the bars that
/// ask for a name or a confirmation, and the load row.
fn local_body(
    ui: &mut egui::Ui,
    w: &mut PacsWindow,
    busy: bool,
    job_progress: &Option<Arc<Progress>>,
    loaded: [bool; MAX_WORKSPACES],
    targets: &[usize],
    acts: &mut Vec<Act>,
) {
    ui.horizontal_wrapped(|ui| {
        if enabled_tip_button(
            ui,
            !busy,
            "📥 Import folder",
            "Copy every DICOM file of a folder into the main storage folder",
        ) {
            acts.push(Act::Import);
        }
        for slot in (0..MAX_WORKSPACES).filter(|s| loaded[*s]) {
            if ui
                .add_enabled(
                    !busy,
                    egui::Button::new(format!("📤 Send workspace {}", SLOT_NAMES[slot])),
                )
                .on_hover_text(
                    "Choose which of this workspace's structure sets and segmentation \
                     series to write into the PACS, beside the study they belong to (new \
                     SOP Instance UIDs, original Study and Frame of Reference UIDs). What \
                     the PACS holds already is not ticked; images are never re-sent.",
                )
                .clicked()
            {
                acts.push(Act::Send(slot));
            }
        }
        if enabled_tip_button(
            ui,
            !busy,
            "⟲ Rescan",
            "Look through the storage folders again; only new files are read",
        ) {
            acts.push(Act::Rescan);
        }
        ui.add(
            egui::TextEdit::singleline(&mut w.view.filter)
                .hint_text("🔍 find a patient, study or series")
                .desired_width(220.0),
        );
        study_filter_list(ui, w, acts);
        let on = !w.study_filter.is_off() || !w.view.filter.is_empty();
        if ui
            .add_enabled(on, egui::Button::new("✖ Reset filters"))
            .on_hover_text("Show every study again: clear the search text and untick every kind")
            .clicked()
        {
            reset_filters(w, acts);
        }
    });
    if let Some(p) = job_progress {
        progress_row(ui, p);
    }
    if let Some(s) = &w.status {
        ui.weak(s);
    }

    // The tree takes no key while a bar asks for a name: the Enter that
    // saves the name would unfold a row too.
    let keys = w.renaming.is_none() && w.new_cohort.is_none();

    // ---- the bars that ask -----------------------------------------------
    if let Some(r) = w.renaming.as_mut() {
        ui.separator();
        ui.horizontal_wrapped(|ui| {
            ui.label(format!("New name for '{}'", r.title));
            let resp = ui.add(egui::TextEdit::singleline(&mut r.text).desired_width(260.0));
            if std::mem::take(&mut r.fresh) {
                resp.request_focus();
            }
            let enter = resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
            if enabled_tip_button(
                ui,
                !busy,
                "✔ Save",
                "Keep this name (the DICOM files are not changed)",
            ) || (enter && !busy)
            {
                acts.push(Act::RenameSave(Some(r.text.clone())));
            }
            if enabled_tip_button(
                ui,
                !busy,
                "The DICOM name",
                "Drop the name given here and show what the DICOM files (or the folder) say",
            ) {
                acts.push(Act::RenameSave(None));
            }
            if ui.button("Cancel").clicked() || ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                acts.push(Act::RenameCancel);
            }
        });
    }
    if let Some(n) = w.new_cohort.as_mut() {
        ui.separator();
        ui.horizontal_wrapped(|ui| {
            ui.label("New cohort");
            let resp = ui.add(
                egui::TextEdit::singleline(&mut n.name)
                    .hint_text("its name")
                    .desired_width(240.0),
            );
            if std::mem::take(&mut n.fresh) {
                resp.request_focus();
            }
            let enter = resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
            let ok = !busy && !n.name.trim().is_empty();
            if enabled_tip_button(
                ui,
                ok,
                "✔ Make it",
                match &n.then {
                    Some((m, true)) => format!(
                        "Make the cohort and put {} into it too (they stay where they are)",
                        count(m.len(), "patient", "patients")
                    ),
                    Some((m, false)) => format!(
                        "Make the cohort and move {} into it (no file moves)",
                        count(m.len(), "patient", "patients")
                    ),
                    None => "Make an empty cohort; move or copy patients into it".into(),
                },
            ) || (enter && ok)
            {
                acts.push(Act::NewCohortSave);
            }
            if ui.button("Cancel").clicked() || ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                acts.push(Act::NewCohortCancel);
            }
        });
    }
    if let Some((_, n)) = &w.confirm_remove {
        ui.separator();
        ui.horizontal_wrapped(|ui| {
            ui.colored_label(
                ui.visuals().warn_fg_color,
                format!(
                    "⚠ Delete {} of the selection from the disk? This cannot be undone.",
                    count(*n, "file", "files")
                ),
            );
            if enabled_tip_button(
                ui,
                !busy,
                "🗑 Delete",
                "Delete the files, and the folders left empty",
            ) {
                acts.push(Act::Remove);
            }
            if ui.button("Cancel").clicked() {
                acts.push(Act::RemoveCancel);
            }
        });
    }
    ui.separator();

    let Some(tree) = &w.tree else {
        ui.weak("reading");
        return;
    };
    if tree.cohorts.is_empty() {
        ui.weak(
            "The PACS is empty. 📥 Import folder files a study into it; Settings > Storage \
             folders adds a folder that holds DICOM files already.",
        );
        return;
    }

    // ---- the tree ----------------------------------------------------------
    let cohorts: Vec<(String, String, bool)> = tree
        .cohorts
        .iter()
        .map(|c| {
            (
                c.id.clone(),
                c.name.clone(),
                c.folder.is_none() && !c.is_unsorted(),
            )
        })
        .collect();
    // Each 4D set: whether it is confirmed, and set by hand.
    let sets: BTreeMap<String, (bool, bool)> = tree
        .cohorts
        .iter()
        .flat_map(|c| &c.patients)
        .flat_map(|p| &p.studies)
        .flat_map(|s| &s.fourd)
        .map(|g| (g.group.clone(), (g.confirmed, g.custom)))
        .collect();
    let height = (ui.available_height() - 60.0).max(140.0);
    let PacsWindow { nodes, view, .. } = &mut *w;
    let nodes_ref: &[TNode] = nodes;
    let mut menu = |ui: &mut egui::Ui, id: &NodeId, view: &TreeView| {
        local_menu(ui, id, view, nodes_ref, &cohorts, &sets, acts);
    };
    let ev = pacs_tree::tree_ui(
        ui,
        nodes_ref,
        view,
        "pacs_local_tree",
        height,
        keys,
        &mut menu,
    );
    if ev.rename {
        let target = if view.selected.len() == 1 {
            view.selected.iter().next().cloned()
        } else {
            view.focus.clone()
        };
        if let Some(id) = target {
            acts.push(Act::RenameStart(id));
        }
    }
    if ev.delete {
        let picks: Vec<Pick> = pacs_tree::series_ids(nodes_ref, &view.tops())
            .iter()
            .filter_map(pick_of)
            .collect();
        acts.push(Act::AskRemove(picks));
    }

    // ---- what to do with the selection --------------------------------------
    ui.separator();
    let any = !view.selected.is_empty();
    ui.horizontal_wrapped(|ui| {
        for slot in targets.iter().copied() {
            if ui
                .add_enabled(
                    !busy && any,
                    egui::Button::new(format!("📩 Load into workspace {}", SLOT_NAMES[slot])),
                )
                .on_hover_text(
                    "Read the selection into this workspace, merging it with whatever is \
                     already there - the same as adding its files",
                )
                .clicked()
            {
                acts.push(Act::Load(slot));
            }
        }
        if any {
            let tops = view.tops();
            ui.weak(format!(
                "{} selected · {}",
                count(tops.len(), "row", "rows"),
                count(pacs_tree::selected_files(nodes_ref, view), "file", "files")
            ));
        } else {
            ui.weak("select patients, studies or series (Ctrl and Shift select several)");
        }
    });
}

/// The context menu of a row of this station's PACS.
fn local_menu(
    ui: &mut egui::Ui,
    id: &NodeId,
    view: &TreeView,
    nodes: &[TNode],
    cohorts: &[(String, String, bool)],
    sets: &BTreeMap<String, (bool, bool)>,
    acts: &mut Vec<Act>,
) {
    // A click on a selected row acts on the whole selection, on another
    // row on that row alone (it is selected by the click).
    let targets: Vec<NodeId> = if view.selected.contains(id) {
        view.tops()
    } else {
        vec![id.clone()]
    };
    let picks: Vec<Pick> = pacs_tree::series_ids(nodes, &targets)
        .iter()
        .filter_map(pick_of)
        .collect();
    let what = if targets.len() == 1 {
        pacs_tree::node_at(nodes, &targets[0])
            .map(|n| format!("'{}'", n.text))
            .unwrap_or_default()
    } else {
        count(targets.len(), "row", "rows")
    };
    let single = targets.len() == 1;
    // A 4D set's row: what the person decides about it.
    if let Some(group) = id.last().and_then(|k| pacs_tree::fourd_group_of(k)) {
        let (confirmed, custom) = sets.get(group).copied().unwrap_or((true, false));
        if !confirmed
            && ui
                .button("✔ Confirm as a 4DCT")
                .on_hover_text("Kept in the catalog: the set is a 4DCT from now on")
                .clicked()
        {
            acts.push(Act::Decide4d(group.to_string(), Some(true)));
            ui.close();
        }
        if ui
            .button("✖ Not a 4DCT")
            .on_hover_text("Dissolve the set; it stays dissolved through rescans")
            .clicked()
        {
            acts.push(Act::Decide4d(group.to_string(), Some(false)));
            ui.close();
        }
        if custom
            && ui
                .button("Let the headers decide again")
                .on_hover_text("Forget what was decided about this set here or by the assistant")
                .clicked()
        {
            acts.push(Act::Decide4d(group.to_string(), None));
            ui.close();
        }
        ui.separator();
    }
    if single
        && !id.last().is_some_and(|k| pacs_tree::is_group_row(k))
        && ui.button("✏ Rename").clicked()
    {
        acts.push(Act::RenameStart(targets[0].clone()));
        ui.close();
    }
    if id.len() == 1 {
        let virtual_cohort = cohorts.iter().any(|(c, _, v)| c == &id[0] && *v);
        if ui.button("➕ New cohort").clicked() {
            acts.push(Act::NewCohortStart(None));
            ui.close();
        }
        if virtual_cohort
            && ui
                .button("🗑 Delete this cohort")
                .on_hover_text(
                    "Its patients go back to the cohorts of their folders; no file is touched",
                )
                .clicked()
        {
            acts.push(Act::DeleteCohort(id[0].clone()));
            ui.close();
        }
    }
    let all_patients = targets.iter().all(|t| t.len() == 2);
    if all_patients {
        let members = patients_of(nodes, &targets);
        ui.separator();
        for copy in [false, true] {
            let label = if copy {
                "Copy to cohort"
            } else {
                "→ Move to cohort"
            };
            ui.menu_button(label, |ui| {
                ui.weak(if copy {
                    "kept in their cohort too; no file is touched"
                } else {
                    "only the PACS's grouping changes; no file is touched"
                });
                for (cid, name, _) in cohorts {
                    let here = members.iter().all(|(from, _)| from == cid);
                    if ui
                        .add_enabled(!here, egui::Button::new(format!("📁 {name}")))
                        .clicked()
                    {
                        acts.push(Act::PutInCohort(members.clone(), cid.clone(), copy));
                        ui.close();
                    }
                }
                if !cohorts.iter().any(|(c, _, _)| c == UNSORTED)
                    && ui.button("📁 Unsorted").clicked()
                {
                    acts.push(Act::PutInCohort(members.clone(), UNSORTED.into(), copy));
                    ui.close();
                }
                ui.separator();
                if ui.button("➕ New cohort").clicked() {
                    acts.push(Act::NewCohortStart(Some((members.clone(), copy))));
                    ui.close();
                }
            });
        }
        if members.iter().any(|(from, _)| from != UNSORTED)
            && ui
                .button("Take out of this cohort")
                .on_hover_text("A patient in no other cohort goes to Unsorted")
                .clicked()
        {
            acts.push(Act::TakeOut(members.clone()));
            ui.close();
        }
        if ui
            .button("Back to the cohort of its folder")
            .on_hover_text("Forget the moves and copies made here: the folders decide again")
            .clicked()
        {
            acts.push(Act::ResetCohorts(
                members.iter().map(|(_, k)| k.clone()).collect(),
            ));
            ui.close();
        }
    }
    ui.separator();
    if ui
        .button("📁 Move the files to a folder")
        .on_hover_text(
            "Move the DICOM files themselves into another folder of the same storage \
             folder (a new one if wanted)",
        )
        .clicked()
    {
        acts.push(Act::MoveStart(picks.clone(), what.clone()));
        ui.close();
    }
    if id.len() > 1
        && ui
            .button("🗑 Remove from the PACS")
            .on_hover_text("Delete the files from the disk (asked once more)")
            .clicked()
    {
        acts.push(Act::AskRemove(picks));
        ui.close();
    }
}

/// *Settings ▶ Storage folders*.
fn storage_page(ui: &mut egui::Ui, roots: &[PathBuf], busy: bool, acts: &mut Vec<Act>) {
    if ui.button("◀ Back to the PACS").clicked() {
        acts.push(Act::Page(Page::Studies));
    }
    ui.separator();
    ui.label(egui::RichText::new("Storage folders").strong());
    ui.label(
        "The PACS reads every folder listed here, however its files are laid out \
         (storage folder / cohort / patient / study / series is the usual way). What this \
         station files - an import, the structures sent back from a workspace - goes into \
         the first folder, the main one, which also keeps the PACS catalog.",
    );
    ui.add_space(4.0);
    egui::Grid::new("pacs_storage")
        .num_columns(3)
        .striped(true)
        .spacing([10.0, 6.0])
        .show(ui, |ui| {
            for (i, r) in roots.iter().enumerate() {
                ui.monospace(r.display().to_string());
                if i == 0 {
                    ui.strong("main");
                } else {
                    ui.weak("");
                }
                ui.horizontal(|ui| {
                    if i == 0 {
                        if enabled_tip_button(
                            ui,
                            !busy,
                            "📂 Change",
                            "Make another folder the main one (this one leaves the list)",
                        ) {
                            acts.push(Act::StorageChangeMain);
                        }
                    } else {
                        if enabled_tip_button(
                            ui,
                            !busy,
                            "⬆ Make main",
                            "New data goes here from now on, and the catalog with it; the \
                             old main folder stays in the list",
                        ) {
                            acts.push(Act::StorageMain(i));
                        }
                        if enabled_tip_button(
                            ui,
                            !busy,
                            "✖ Remove from the list",
                            "The PACS no longer reads this folder; its files stay on the disk",
                        ) {
                            acts.push(Act::StorageDrop(i));
                        }
                    }
                });
                ui.end_row();
            }
        });
    ui.add_space(4.0);
    if enabled_tip_button(
        ui,
        !busy,
        "➕ Add a folder",
        "A folder that holds DICOM files already, read where it is: nothing is copied",
    ) {
        acts.push(Act::StorageAdd);
    }
    ui.add_space(6.0);
    ui.weak(format!(
        "Catalog: {}",
        roots
            .first()
            .map(|r| r.join(crate::archive::CATALOG_FILE).display().to_string())
            .unwrap_or_default()
    ));
    ui.weak(
        "The catalog lists every file once read, the cohorts and the names given here. \
         The DICOM files stay as they are; delete the catalog and the next scan rebuilds \
         it (the names and cohorts made here are lost then).",
    );
}

/// *Move the files to a folder*: the folders of the storage folder, one to
/// pick, and a new one if wanted.
fn move_page(ui: &mut egui::Ui, d: &mut MoveDialog, busy: bool, acts: &mut Vec<Act>) {
    if ui.button("◀ Back to the PACS").clicked() {
        acts.push(Act::Page(Page::Studies));
    }
    ui.separator();
    ui.label(egui::RichText::new(format!("Move the files of {}", d.what)).strong());
    ui.weak(
        "The files keep the folders they had below the folder that holds them all (a \
         study's series folders arrive as series folders). Nothing is moved when a file \
         of the same name is there already. Folders left empty are removed.",
    );
    ui.add_space(4.0);
    let Some(folders) = &d.folders else {
        ui.weak("reading the folders");
        return;
    };
    let height = (ui.available_height() - 90.0).max(120.0);
    egui::ScrollArea::vertical()
        .id_salt("pacs_move_folders")
        .max_height(height)
        .auto_shrink([false, true])
        .show(ui, |ui| {
            if ui
                .selectable_label(
                    d.dest.as_deref() == Some(""),
                    "📁 (the storage folder itself)",
                )
                .clicked()
            {
                d.dest = Some(String::new());
            }
            for f in folders {
                let depth = f.matches('/').count();
                let name = f.rsplit('/').next().unwrap_or(f);
                ui.horizontal(|ui| {
                    ui.add_space(16.0 * (depth + 1) as f32);
                    if ui
                        .selectable_label(
                            d.dest.as_deref() == Some(f.as_str()),
                            format!("📁 {name}"),
                        )
                        .on_hover_text(f)
                        .clicked()
                    {
                        d.dest = Some(f.clone());
                    }
                });
            }
        });
    ui.separator();
    ui.horizontal_wrapped(|ui| {
        ui.label("New folder in the chosen one");
        ui.add(
            egui::TextEdit::singleline(&mut d.new_name)
                .hint_text("its name")
                .desired_width(200.0),
        );
        if enabled_tip_button(
            ui,
            !busy && !d.new_name.trim().is_empty(),
            "📁 Make it",
            "Make the folder (in the chosen folder, or in the storage folder) and choose it",
        ) {
            acts.push(Act::NewFolder);
        }
    });
    ui.horizontal_wrapped(|ui| {
        let chosen = d.dest.clone();
        if enabled_tip_button(
            ui,
            !busy && chosen.is_some(),
            "→ Move here",
            "Move the files into the chosen folder",
        ) {
            acts.push(Act::MoveDo);
        }
        match chosen {
            Some(c) if c.is_empty() => ui.weak("into the storage folder itself"),
            Some(c) => ui.weak(format!("into {c}")),
            None => ui.weak("choose a folder"),
        };
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::archive::fourd::StudyFourD;
    use crate::archive::CohortEntry;

    fn study(uid: &str, mods: &[&str], fourd: Option<bool>) -> StudyEntry {
        StudyEntry {
            study_uid: uid.into(),
            modalities: mods.iter().map(|m| m.to_string()).collect(),
            fourd: fourd
                .map(|confirmed| StudyFourD {
                    confirmed,
                    ..Default::default()
                })
                .into_iter()
                .collect(),
            ..Default::default()
        }
    }

    fn tree() -> Tree {
        // A name given with words in brackets, a header's `name (ID)`
        // (nothing to offer), and words in another case.
        let p = |key: &str, label: &str, studies: Vec<StudyEntry>| PatientEntry {
            key: key.into(),
            name: key.to_uppercase(),
            id: key.to_uppercase(),
            label: label.into(),
            studies,
            ..Default::default()
        };
        let both = p(
            "p1",
            "P1 (Adult, Male, Thorax)",
            vec![study("1", &["CT", "RTSTRUCT"], Some(true))],
        );
        Tree {
            cohorts: vec![
                CohortEntry {
                    id: "a".into(),
                    patients: vec![both.clone(), p("p2", "", vec![study("2", &["MR"], None)])],
                    ..Default::default()
                },
                CohortEntry {
                    id: "b".into(),
                    // p1 again in a second cohort: their study counts once.
                    patients: vec![
                        both,
                        p(
                            "p3",
                            "Mattia (Phantom, male, P3)",
                            vec![study("3", &["DX", "CT", ""], Some(false))],
                        ),
                    ],
                    ..Default::default()
                },
            ],
        }
    }

    fn uids(t: &Tree) -> BTreeSet<String> {
        t.cohorts
            .iter()
            .flat_map(|c| c.patients.iter())
            .flat_map(|p| p.studies.iter())
            .map(|s| s.study_uid.clone())
            .collect()
    }

    #[test]
    fn the_scan_offers_what_the_studies_hold_and_several_can_be_picked() {
        let t = tree();
        let m = |s: &str| Facet::Modality(s.into());
        let w = |s: &str| Facet::Patient(WordKind::of(s), s.into());
        assert_eq!(
            facets(&t),
            vec![
                (Facet::FourD, 1),
                (Facet::ProbableFourD, 1),
                (w("Adult"), 1),
                (w("Male"), 2),
                (w("Thorax"), 1),
                (w("Phantom"), 1),
                (m("CT"), 2),
                (m("DX"), 1),
                (m("MR"), 1),
                (m("RTSTRUCT"), 1),
            ]
        );
        let pick = |all: bool, fs: &[Facet]| StudyFilter {
            picked: fs.iter().cloned().collect(),
            all,
        };
        let shown = |f: &StudyFilter| uids(&filtered(&t, f));
        let set = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<BTreeSet<_>>();
        assert_eq!(shown(&StudyFilter::default()), set(&["1", "2", "3"]));
        assert_eq!(shown(&pick(true, &[m("CT")])), set(&["1", "3"]));
        assert_eq!(shown(&pick(true, &[m("CT"), m("RTSTRUCT")])), set(&["1"]));
        assert_eq!(shown(&pick(false, &[m("MR"), m("DX")])), set(&["2", "3"]));
        assert_eq!(shown(&pick(true, &[m("MR"), m("DX")])), set(&[]));
        assert_eq!(shown(&pick(true, &[Facet::ProbableFourD])), set(&["3"]));
        assert_eq!(shown(&pick(true, &[w("Male")])), set(&["1", "3"]));
        assert_eq!(shown(&pick(true, &[w("Male"), m("DX")])), set(&["3"]));
        assert_eq!(
            shown(&pick(false, &[w("Adult"), m("MR")])),
            set(&["1", "2"])
        );
        // Cohorts and patients left with no study go too.
        let f = filtered(&t, &pick(true, &[m("MR")]));
        assert_eq!(f.cohorts.len(), 1);
        assert_eq!(f.cohorts[0].patients.len(), 1);
        assert_eq!(
            pick(true, &[m("CT"), m("RTSTRUCT")]).label(),
            "CT + RTSTRUCT"
        );
        assert_eq!(pick(false, &[m("MR"), m("DX")]).label(), "DX or MR");
    }

    #[test]
    fn a_word_is_an_age_group_a_sex_a_subject_or_else_a_site() {
        for (w, k) in [
            ("Adult", WordKind::Age),
            ("pediatric", WordKind::Age),
            ("Paediatric", WordKind::Age),
            ("Male", WordKind::Sex),
            ("female", WordKind::Sex),
            ("Phantom", WordKind::Subject),
            ("CIRS phantom", WordKind::Subject),
            ("Volunteer", WordKind::Subject),
            ("Pancreas", WordKind::Site),
            ("HN", WordKind::Site),
            ("Thoraco-abdominal", WordKind::Site),
        ] {
            assert_eq!(WordKind::of(w), k, "{w}");
        }
    }

    #[test]
    fn a_new_scan_drops_what_it_no_longer_offers() {
        let mut w = PacsWindow::new(Servers::default());
        w.study_filter.picked = [Facet::Modality("MR".into()), Facet::Modality("PT".into())]
            .into_iter()
            .collect();
        w.set_tree(tree());
        assert_eq!(
            w.study_filter.picked,
            [Facet::Modality("MR".into())].into_iter().collect()
        );
        assert_eq!(w.nodes.len(), 1, "only the cohort holding the MR study");
    }
}
