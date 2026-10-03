//! What the two sides of the PACS link say to each other.
//!
//! Every answer and every request body that is JSON is one of the structs
//! below, used by the server to write it and by the client to read it, so
//! the protocol has one description. Every struct carries
//! `#[serde(default)]`: a newer server may add a field and an older client
//! simply does not see it; a field an older server does not send arrives
//! as its default. A change that cannot be read that way raises
//! [`API_VERSION`], which [`ServerInfo`] carries and the client checks.
//!
//! All routes are under [`PREFIX`]. A request that is not `/server` or
//! `/pair` carries `Authorization: Bearer <token>`.

use serde::{Deserialize, Serialize};

/// The major version of the API. A client refuses a server whose major it
/// does not know.
pub const API_VERSION: u32 = 1;

/// Where every route lives.
pub const PREFIX: &str = "/rds/v1";

/// What a paired client may do. Each role includes the ones before it.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// List the archive and download studies.
    #[default]
    View,
    /// Also send objects and studies into it.
    Edit,
    /// Also hand the server workflows to run.
    Run,
    /// Also remove studies, make pairing codes, see and revoke clients,
    /// read the audit log.
    Admin,
}

impl Role {
    pub const ALL: [Role; 4] = [Role::View, Role::Edit, Role::Run, Role::Admin];

    pub fn label(self) -> &'static str {
        match self {
            Role::View => "view",
            Role::Edit => "edit",
            Role::Run => "run",
            Role::Admin => "admin",
        }
    }

    /// One line for a tooltip or a combo box.
    pub fn describe(self) -> &'static str {
        match self {
            Role::View => "view: list the archive and download studies",
            Role::Edit => "edit: also send structures, segmentations and studies back",
            Role::Run => "run: also hand the server workflows to run",
            Role::Admin => "admin: also remove studies and manage clients",
        }
    }

    pub fn from_label(s: &str) -> Option<Role> {
        Role::ALL
            .into_iter()
            .find(|r| r.label().eq_ignore_ascii_case(s.trim()))
    }

    /// Does this role include `needed`?
    pub fn allows(self, needed: Role) -> bool {
        self >= needed
    }
}

/// `GET /server` (no token): who the server is.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ServerInfo {
    /// What the server calls itself (`pacs.toml` `name`, else the machine's
    /// name).
    pub name: String,
    /// The program's version, `0.11.0`.
    pub version: String,
    pub api_version: u32,
    /// A random identifier made once per server: the name of a client's
    /// mirror folder, so a renamed server keeps its copy.
    pub server_id: String,
    /// SHA-256 of the certificate the server presents, in hex.
    pub fingerprint: String,
    /// The server runs tasks (`pacs.toml` `tasks`).
    pub tasks: bool,
    /// The largest upload it takes, in MB; a client sends in parts below
    /// it. 0 (an older server): no limit known.
    pub max_upload_mb: u64,
}

/// `POST /pair` body.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PairRequest {
    pub code: String,
    /// The name this device asks to be known by (its machine name).
    pub client_name: String,
}

/// `POST /pair` answer: the token, which is shown to nobody but kept by
/// the client.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PairResponse {
    pub token: String,
    pub role: Role,
    pub server_id: String,
    /// The name the server filed the client under (a number is added when
    /// the name was taken).
    pub client_name: String,
}

/// `GET /whoami`: the caller as the server sees it.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WhoAmI {
    pub name: String,
    pub role: Role,
}

/// One study of the archive, as a listing gives it.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RemoteStudy {
    pub study_uid: String,
    pub date: String,
    pub description: String,
    pub modalities: Vec<String>,
    pub files: usize,
}

impl RemoteStudy {
    /// `20260827 - Planning · CT, RTSTRUCT · 214 files`, as the local
    /// archive describes its studies.
    pub fn describe(&self) -> String {
        crate::archive::StudyEntry {
            study_uid: self.study_uid.clone(),
            date: self.date.clone(),
            description: self.description.clone(),
            modalities: self.modalities.clone(),
            files: self.files,
            dir: Default::default(),
        }
        .describe()
    }
}

/// One patient of the archive with their studies.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RemotePatient {
    /// The patient's folder name: what routes name the patient by.
    pub key: String,
    pub name: String,
    pub id: String,
    pub studies: Vec<RemoteStudy>,
}

impl RemotePatient {
    pub fn title(&self) -> String {
        crate::archive::PatientEntry {
            name: self.name.clone(),
            id: self.id.clone(),
            ..Default::default()
        }
        .title()
    }

    pub fn files(&self) -> usize {
        self.studies.iter().map(|s| s.files).sum()
    }
}

/// `GET /patients`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Listing {
    pub patients: Vec<RemotePatient>,
}

/// One instance of a study: enough to tell which ones a copy lacks.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct InstanceInfo {
    pub sop_uid: String,
    pub bytes: u64,
}

/// `GET /studies/{uid}`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Manifest {
    pub study_uid: String,
    pub patient_key: String,
    pub instances: Vec<InstanceInfo>,
}

/// `POST /studies/{uid}/bundle` body: the instances wanted. The answer is a
/// zip of as many of them as fit the server's bundle size; the client asks
/// again for the rest.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BundleRequest {
    pub sops: Vec<String>,
}

/// `POST /studies` answer: what the archive did with an upload (the same
/// counts as a local import, [`crate::archive::ImportSummary`]).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct UploadSummary {
    pub stored: usize,
    pub duplicates: usize,
    pub skipped: usize,
    pub patients: usize,
    pub studies: usize,
}

impl UploadSummary {
    pub fn add(&mut self, o: &UploadSummary) {
        self.stored += o.stored;
        self.duplicates += o.duplicates;
        self.skipped += o.skipped;
        self.patients = self.patients.max(o.patients);
        self.studies = self.studies.max(o.studies);
    }

    pub fn describe(&self) -> String {
        crate::archive::ImportSummary {
            stored: self.stored,
            duplicates: self.duplicates,
            skipped: self.skipped,
            patients: self.patients,
            studies: self.studies,
        }
        .describe()
    }
}

impl From<crate::archive::ImportSummary> for UploadSummary {
    fn from(s: crate::archive::ImportSummary) -> Self {
        UploadSummary {
            stored: s.stored,
            duplicates: s.duplicates,
            skipped: s.skipped,
            patients: s.patients,
            studies: s.studies,
        }
    }
}

/// What a workflow input is, and so what a task binds to it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InputKind {
    /// A *DICOM folder* step: bound to a study (or a whole patient), whose
    /// folder in the archive it reads.
    #[default]
    Folder,
    /// A *DICOM folders* step (a batch): bound to a patient, each study a
    /// case.
    Batch,
    /// A *From the archive* step: bound to a study.
    Archive,
}

/// One input of a workflow the server offers.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct InputInfo {
    pub node: u32,
    pub title: String,
    pub kind: InputKind,
}

/// One workflow the server can run: a saved one of the server's, an
/// example the program ships, or one of the one-step templates.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WorkflowInfo {
    pub name: String,
    /// `file`, `example` or `template`.
    pub source: String,
    pub description: String,
    pub steps: Vec<String>,
    pub inputs: Vec<InputInfo>,
    /// What the server's check found wrong with it; a task of a workflow
    /// with problems is refused.
    pub problems: Vec<String>,
}

/// `GET /workflows`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WorkflowList {
    pub workflows: Vec<WorkflowInfo>,
}

/// One input of a task bound to a study (or a patient) of the archive.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Binding {
    pub node: u32,
    pub patient_key: String,
    /// Empty binds the whole patient (a folder input reads all their
    /// studies; a batch runs once per study).
    pub study_uid: String,
}

/// `POST /tasks` body.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TaskRequest {
    /// The name of a workflow `GET /workflows` lists, or empty when the
    /// workflow travels inline.
    pub workflow: String,
    /// A whole `.rdsflow` file, for a workflow the server does not have.
    pub inline: String,
    pub bindings: Vec<Binding>,
    /// What the task list calls it; empty is the workflow's name.
    pub title: String,
    /// File every DICOM object the run wrote into the archive afterwards,
    /// so the client can pull the results.
    pub file_results: bool,
    /// Run the independent rows at the start of the workflow side by side.
    pub parallel: bool,
}

impl Default for TaskRequest {
    fn default() -> Self {
        TaskRequest {
            workflow: String::new(),
            inline: String::new(),
            bindings: Vec::new(),
            title: String::new(),
            file_results: true,
            parallel: false,
        }
    }
}

/// `POST /tasks` answer.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Submitted {
    pub task_id: String,
    /// Tasks ahead of this one.
    pub position: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TaskState {
    #[default]
    Queued,
    Running,
    Done,
    Failed,
    Cancelled,
}

impl TaskState {
    pub fn label(self) -> &'static str {
        match self {
            TaskState::Queued => "queued",
            TaskState::Running => "running",
            TaskState::Done => "done",
            TaskState::Failed => "failed",
            TaskState::Cancelled => "cancelled",
        }
    }

    /// Will it change again?
    pub fn finished(self) -> bool {
        matches!(
            self,
            TaskState::Done | TaskState::Failed | TaskState::Cancelled
        )
    }
}

/// One task, as the queue reports it.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TaskInfo {
    pub id: String,
    pub title: String,
    pub workflow: String,
    /// The client that handed it in.
    pub client: String,
    pub state: TaskState,
    /// `0..=1` while it runs.
    pub progress: f32,
    pub message: String,
    pub submitted: String,
    pub started: String,
    pub finished: String,
    pub error: Option<String>,
    /// How many log lines there are; `log` holds those from the cursor the
    /// request gave on.
    pub log_len: usize,
    pub log: Vec<String>,
    /// Studies of the archive the run filed objects into.
    pub filed: Vec<String>,
    /// The files the run wrote other than DICOM (reports, tables), by their
    /// path inside the run folder: `GET /tasks/{id}/files/{path}`.
    pub files: Vec<String>,
}

/// `GET /tasks`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TaskList {
    pub tasks: Vec<TaskInfo>,
}

/// `POST /admin/pairing-codes` body.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CodeRequest {
    pub role: Role,
    /// How long the code is good for; 0 is the server's default.
    pub minutes: u64,
}

/// A pairing code, as the operator sees it once.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PairingCode {
    pub code: String,
    pub role: Role,
    pub expires: String,
    pub expires_in_s: u64,
}

/// A paired client, for the operator.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ClientInfo {
    pub name: String,
    pub role: Role,
    pub paired: String,
    pub last_seen: String,
    pub address: String,
}

/// `GET /admin/clients`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ClientList {
    pub clients: Vec<ClientInfo>,
}

/// `GET /admin/audit`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AuditTail {
    pub lines: Vec<String>,
}

/// Every error answer: what went wrong, for a person, and a short code for
/// a program (`unauthorized`, `forbidden`, `not_found`, `bad_request`,
/// `too_large`, `too_many`, `busy`, `failed`).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ApiError {
    pub error: String,
    pub code: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roles_include_the_ones_before_them() {
        assert!(Role::Admin.allows(Role::Run));
        assert!(Role::Run.allows(Role::Edit));
        assert!(Role::Edit.allows(Role::View));
        assert!(!Role::View.allows(Role::Edit));
        assert!(!Role::Run.allows(Role::Admin));
        assert_eq!(Role::from_label("RUN"), Some(Role::Run));
        assert_eq!(Role::from_label("root"), None);
    }

    /// An older client reads a newer server's answer, and a newer client an
    /// older server's: unknown fields are ignored, missing ones defaulted.
    #[test]
    fn answers_read_across_versions() {
        let newer = r#"{"name":"pacs","version":"9.9.9","api_version":1,
                        "server_id":"x","fingerprint":"f","tasks":true,
                        "something_new":{"a":1}}"#;
        let info: ServerInfo = serde_json::from_str(newer).unwrap();
        assert_eq!(info.name, "pacs");
        assert!(info.tasks);
        let older: TaskInfo = serde_json::from_str(r#"{"id":"t1","state":"running"}"#).unwrap();
        assert_eq!(older.state, TaskState::Running);
        assert!(older.files.is_empty());
        let req: TaskRequest = serde_json::from_str(r#"{"workflow":"w"}"#).unwrap();
        assert!(req.file_results, "filing the results is the default");
    }
}
