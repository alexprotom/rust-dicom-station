//! What the server holds between calls: the open datasets, the transforms,
//! the reports, and the handles the client uses to name them.
//!
//! This is the headless counterpart of the viewer's two study slots, with
//! two differences: there can be more than two datasets (a heart case has
//! three: the cardiac CT, the planning CT and the 4DCT), and every entity
//! gets a stable handle (`ds1`, `reg2`, `greg1`, `run1`) minted here, which
//! is the only way a tool call may refer to it.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use anyhow::{anyhow, bail, Context, Result};

use crate::dicomseg::SegSeries;
use crate::loader::{self, LoadedStudy};
use crate::motion::MotionReport;
use crate::progress::Progress;
use crate::registration::{RegistrationResult, Transform3};
use crate::segmentation::Segmentation;
use crate::volume::{Grid, Volume};
use crate::workflow::select;

use super::config::Config;
use super::phi::{Public, Redactor};

/// The redactor, shared with the protocol layer so progress messages and
/// job listings can be scrubbed while a call holds the session.
pub type SharedRedactor = Arc<RwLock<Redactor>>;

/// One open dataset.
pub struct Dataset {
    pub id: String,
    pub study: LoadedStudy,
    /// Where it was opened from, resolved. Reported only through the
    /// redactor (relative to its root).
    pub origin: PathBuf,
    /// `rootN`, the label of the root it is under.
    pub root_label: String,
    /// Identity tags that held values when it was opened (names only).
    pub phi_tags: Vec<String>,
    /// Volumes of series other than the displayed one, by series index.
    /// Bounded: see [`Session::volume`].
    volumes: HashMap<usize, Arc<Volume>>,
    /// Insertion order of `volumes`, oldest first.
    volume_order: Vec<usize>,
}

/// Volumes kept per dataset beyond the active one.
const EXTRA_VOLUMES: usize = 2;

/// A registration between two series, kept for propagation.
pub struct Registration {
    pub id: String,
    /// Dataset and series index of the fixed image.
    pub fixed: (String, usize),
    /// Dataset and series index of the moving image.
    pub moving: (String, usize),
    pub result: RegistrationResult,
}

/// One phase of a registered 4D group.
pub struct PhaseReg {
    pub label: String,
    pub series_uid: String,
    /// Phase → the moving image (the destination → source direction a
    /// propagation pulls along).
    pub transform: Arc<Transform3>,
    pub metric_line: String,
}

/// The per-phase transforms of one 4D group against one moving volume.
pub struct GroupRegistration {
    pub id: String,
    pub dataset: String,
    pub group: usize,
    pub group_name: String,
    /// Dataset and series index of the moving image.
    pub moving: (String, usize),
    pub phases: Vec<PhaseReg>,
}

/// A finished motion analysis.
pub struct Run {
    pub id: String,
    pub dataset: String,
    pub report: MotionReport,
}

/// Everything the server knows.
pub struct Session {
    pub config: Config,
    pub redactor: SharedRedactor,
    pub datasets: Vec<Dataset>,
    pub registrations: Vec<Registration>,
    pub group_registrations: Vec<GroupRegistration>,
    pub runs: Vec<Run>,
    counters: HashMap<&'static str, usize>,
    /// The per-session output folder, created on first use.
    out_dir: Option<PathBuf>,
}

impl Session {
    pub fn new(config: Config) -> Session {
        let mut redactor = Redactor::new();
        for (i, root) in config.roots.iter().enumerate() {
            redactor.add_root(root, &format!("root{}", i + 1));
        }
        if let Some(out) = &config.output_dir {
            redactor.add_root(out, "output");
        }
        Session {
            config,
            redactor: Arc::new(RwLock::new(redactor)),
            datasets: Vec::new(),
            registrations: Vec::new(),
            group_registrations: Vec::new(),
            runs: Vec::new(),
            counters: HashMap::new(),
            out_dir: None,
        }
    }

    /// The next handle of a kind: `ds1`, `ds2`, `reg1`.
    pub fn mint(&mut self, prefix: &'static str) -> String {
        let n = self.counters.entry(prefix).or_insert(0);
        *n += 1;
        format!("{prefix}{n}")
    }

    // ---- datasets ---------------------------------------------------------

    pub fn add_dataset(
        &mut self,
        study: LoadedStudy,
        origin: PathBuf,
        root_label: String,
        phi_tags: Vec<String>,
    ) -> Result<&Dataset> {
        if self.datasets.len() >= self.config.max_open_datasets {
            bail!(
                "{} datasets are open, the configured maximum; close one first",
                self.datasets.len()
            );
        }
        let id = self.mint("ds");
        self.datasets.push(Dataset {
            id,
            study,
            origin,
            root_label,
            phi_tags,
            volumes: HashMap::new(),
            volume_order: Vec::new(),
        });
        Ok(self.datasets.last().expect("just pushed"))
    }

    pub fn dataset(&self, id: &str) -> Result<&Dataset> {
        self.datasets
            .iter()
            .find(|d| d.id == id)
            .ok_or_else(|| anyhow!("no open dataset '{id}'"))
    }

    pub fn dataset_mut(&mut self, id: &str) -> Result<&mut Dataset> {
        self.datasets
            .iter_mut()
            .find(|d| d.id == id)
            .ok_or_else(|| anyhow!("no open dataset '{id}'"))
    }

    pub fn close_dataset(&mut self, id: &str) -> Result<()> {
        let n = self.datasets.len();
        self.datasets.retain(|d| d.id != id);
        if self.datasets.len() == n {
            bail!("no open dataset '{id}'");
        }
        self.registrations
            .retain(|r| r.fixed.0 != id && r.moving.0 != id);
        self.group_registrations
            .retain(|g| g.dataset != id && g.moving.0 != id);
        Ok(())
    }

    /// The series index a call means: the given 1-based number, or the
    /// displayed series when none was given.
    pub fn series_index(&self, ds: &Dataset, series: Option<u32>) -> Result<usize> {
        match series {
            None => {
                if ds.study.series.is_empty() {
                    bail!("dataset {} holds no image series", ds.id);
                }
                Ok(ds.study.active_series)
            }
            Some(n) => {
                let i = (n as usize)
                    .checked_sub(1)
                    .context("series numbers start at 1")?;
                if i >= ds.study.series.len() {
                    bail!(
                        "dataset {} has {} series; there is no series {n}",
                        ds.id,
                        ds.study.series.len()
                    );
                }
                Ok(i)
            }
        }
    }

    /// The volume of one series of a dataset, loading it if it is not the
    /// displayed one. A few loaded volumes are kept per dataset; beyond
    /// that the oldest goes, since a 4D study would otherwise pin gigabytes.
    pub fn volume(&mut self, id: &str, series: usize, p: &Progress) -> Result<Arc<Volume>> {
        let ds = self.dataset_mut(id)?;
        if series == ds.study.active_series && ds.study.has_volume() {
            return Ok(ds.study.volume.clone());
        }
        if let Some(v) = ds.volumes.get(&series) {
            return Ok(v.clone());
        }
        let info = ds
            .study
            .series
            .get(series)
            .cloned()
            .ok_or_else(|| anyhow!("no series {} in {}", series + 1, id))?;
        let (vol, _, _) = loader::load_series_volume(&info, p)?;
        let vol = Arc::new(vol);
        while ds.volume_order.len() >= EXTRA_VOLUMES {
            let old = ds.volume_order.remove(0);
            ds.volumes.remove(&old);
        }
        ds.volumes.insert(series, vol.clone());
        ds.volume_order.push(series);
        Ok(vol)
    }

    /// The grid a series' masks live on, without loading its pixels when
    /// it is the displayed series.
    pub fn grid(&mut self, id: &str, series: usize, p: &Progress) -> Result<Grid> {
        Ok(self.volume(id, series, p)?.grid())
    }

    // ---- structures -------------------------------------------------------

    /// A structure by name, optionally within one structure set or
    /// segmentation series.
    pub fn structure(&self, id: &str, name: &str, set: Option<&str>) -> Result<select::Structure> {
        let ds = self.dataset(id)?;
        select::find(&ds.study, name, set).ok_or_else(|| {
            let known: Vec<String> = select::list(&ds.study)
                .into_iter()
                .map(|e| e.name)
                .take(12)
                .collect();
            anyhow!(
                "no structure '{name}' in {id}{}; known: {}",
                set.map(|s| format!(" (set '{s}')")).unwrap_or_default(),
                known.join(", ")
            )
        })
    }

    /// File masks as segments of a segmentation series bound to `series`:
    /// an existing series on that lattice is extended, otherwise one is
    /// created, the way the viewer's `ensure_seg_series` does it. Returns
    /// the series label the segments landed in.
    pub fn land_masks(
        &mut self,
        id: &str,
        series: usize,
        grid: &Grid,
        masks: Vec<(String, [u8; 3], Vec<u8>)>,
    ) -> Result<String> {
        let ds = self.dataset_mut(id)?;
        let info = ds
            .study
            .series
            .get(series)
            .ok_or_else(|| anyhow!("no series {} in {id}", series + 1))?;
        let (uid, study_uid) = (info.uid.clone(), info.study_uid.clone());
        let idx = match ds
            .study
            .seg_series
            .iter()
            .position(|s| s.referenced_series_uid == uid && s.grid.matches(grid))
        {
            Some(i) => i,
            None => {
                let label = format!("Segmentations {}", ds.study.seg_series.len() + 1);
                ds.study
                    .seg_series
                    .push(SegSeries::new(label, grid.clone(), uid, study_uid));
                ds.study.seg_series.len() - 1
            }
        };
        let ser = &mut ds.study.seg_series[idx];
        for (name, color, mask) in masks {
            ser.segs.push(Segmentation::from_label_map(
                name, color, grid.dims, &mask, 1,
            ));
        }
        Ok(ser.label.clone())
    }

    // ---- registrations ----------------------------------------------------

    pub fn registration(&self, id: &str) -> Result<&Registration> {
        self.registrations
            .iter()
            .find(|r| r.id == id)
            .ok_or_else(|| anyhow!("no registration '{id}'"))
    }

    pub fn group_registration(&self, id: &str) -> Result<&GroupRegistration> {
        self.group_registrations
            .iter()
            .find(|r| r.id == id)
            .ok_or_else(|| anyhow!("no group registration '{id}'"))
    }

    pub fn run(&self, id: &str) -> Result<&Run> {
        self.runs
            .iter()
            .find(|r| r.id == id)
            .ok_or_else(|| anyhow!("no motion run '{id}'"))
    }

    // ---- the door ---------------------------------------------------------

    /// Scrub one string.
    pub fn text(&self, s: &str) -> Public {
        self.redactor.read().expect("redactor lock").text(s)
    }

    /// Scrub a JSON value.
    pub fn json(&self, v: serde_json::Value) -> serde_json::Value {
        self.redactor.read().expect("redactor lock").json(v)
    }

    /// Remember identifying values.
    pub fn add_values(&self, values: Vec<String>) {
        self.redactor
            .write()
            .expect("redactor lock")
            .add_values(values);
    }

    /// Register a folder whose path must not appear.
    pub fn add_root(&self, path: &std::path::Path, label: &str) {
        self.redactor
            .write()
            .expect("redactor lock")
            .add_root(path, label);
    }

    // ---- output -----------------------------------------------------------

    /// The session's output folder, `output_dir/rds-mcp-YYYYMMDD-HHMMSS`,
    /// created on first use. Every file the server writes goes under it.
    pub fn out_dir(&mut self) -> Result<PathBuf> {
        if let Some(d) = &self.out_dir {
            return Ok(d.clone());
        }
        let root = self.config.output_dir()?.to_path_buf();
        let (date, time) = crate::dicom_export::today();
        let mut dir = root.join(format!("rds-mcp-{date}-{time}"));
        let mut n = 1;
        while dir.exists() {
            n += 1;
            dir = root.join(format!("rds-mcp-{date}-{time}-{n}"));
        }
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("create {}", self.text(&dir.to_string_lossy()).as_str()))?;
        self.redactor
            .write()
            .expect("redactor lock")
            .add_root(&dir, "output/session");
        self.out_dir = Some(dir.clone());
        Ok(dir)
    }

    /// A folder under the output folder that does not exist yet.
    pub fn fresh_out_subdir(&mut self, stem: &str) -> Result<PathBuf> {
        let base = self.out_dir()?;
        let safe: String = stem
            .chars()
            .map(|c| {
                if c.is_alphanumeric() || c == '-' || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        let mut dir = base.join(&safe);
        let mut n = 1;
        while dir.exists() {
            n += 1;
            dir = base.join(format!("{safe}-{n}"));
        }
        Ok(dir)
    }
}
