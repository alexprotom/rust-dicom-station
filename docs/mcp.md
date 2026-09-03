# The MCP server: `rds-mcp`

Rust DICOM Station can be driven by an AI assistant through the Model Context
Protocol. `rds-mcp` is a second executable, built from the same code as the
viewer, with no window: an MCP client (Claude Desktop, Claude Code, or any
other) launches it, and the assistant then loads, segments, registers,
propagates, analyses and exports through a set of tools. The viewer is not
involved and not changed; the server is a way to run the station's
pipelines without clicking through them, over a whole cohort if need be.

The first workflow it was built for is heart target propagation and analysis
for cardiac radioablation (STAR): carry a target from the cardiac CT it was
contoured on to the planning CT, follow it through the 4DCT phases, build the
ITV, evaluate the dose. The `heart_target_propagation` prompt encodes that
sequence.

## Setting it up

1. Build or install the server. It is part of the Windows installer when the
   release was built with it; from source: `cargo build --release --features
   mcp` produces `target/release/rds-mcp` beside the viewer. The viewer's
   *Settings ▶ MCP server* menu says whether it is present.
2. Write the configuration, `mcp.toml`, in the station's configuration folder
   (`%LOCALAPPDATA%\RustDICOMStation` on Windows, `~/.config/RustDICOMStation`
   on Linux; the menu shows the exact path). Without it no dataset can be
   opened:

   ```toml
   roots = ["D:/studies/anonymized"]      # folders that may be read
   output_dir = "D:/studies/rds-mcp-out"  # the one folder results go to
   phi_policy = "refuse"                  # refuse | redact | allow (see below)
   models_dir = ""                        # empty: the viewer's model folder
   allow_model_download = false           # weights are the only network use
   device = "auto"                        # auto | gpu | cpu
   max_open_datasets = 4
   job_timeout_minutes = 60
   audit_log = true                       # data folder /mcp/audit-YYYY-MM-DD.log
   ```

3. Tell the client about it. *Settings ▶ MCP server ▶ Copy client
   configuration* puts the entry on the clipboard; for Claude Desktop it goes
   into `claude_desktop_config.json`:

   ```json
   { "mcpServers": { "rust-dicom-station": { "command": "C:\\...\\rds-mcp.exe", "args": [] } } }
   ```

   `rds-mcp --check` reads the configuration, prints what it says on standard
   error, and exits, which is the quickest way to see that the file parses.

Nothing in the configuration can be changed through the protocol. The
assistant works inside the roots, writes only under the output folder, and
never deletes.

## What the assistant can do

Every entity gets a handle the assistant refers to it by: datasets `ds1`,
registrations `reg1`, 4D group registrations `greg1`, motion runs `run1`.
Structures are named (`Heart`, `TARGET`), with the structure set or
segmentation series added when a name repeats. Series are numbered as
`describe_dataset` lists them, and default to the displayed one.

| Tool | Does |
|---|---|
| `open_dataset`, `describe_dataset`, `list_structures`, `list_4d_groups`, `close_dataset`, `describe_session` | Open a folder (or files) under a root; see its series, 4D groups, structure sets, segmentations, doses, plans |
| `segment_organs` | TotalSegmentator on one series: `fast`, `high` (with `parts` such as `cardiac`) or `preview`; `keep` narrows to named organs |
| `segment_body` | The patient outline, classically or model-assisted |
| `combine_structures` | Union / intersect / subtract with margins in mm (uniform or per patient direction) and cleanup |
| `register`, `describe_registration` | Rigid, elastix B-spline or plastimatch B-spline; `region` makes a deformable run local to a structure of the fixed dataset; `start` refines an earlier registration |
| `propagate` | Carry structures across a registration, to the fixed or the moving side |
| `propagate_to_group` | One series onto every phase of a 4D group, one deformable registration per phase, transforms kept and reused |
| `analyse_motion` | The 4D pipeline: tracks, amplitudes, correlation with a reference structure, per-phase QA, ITVs |
| `compare_structures` | Volumes, centroid offset, Dice, HD95, mean surface distance |
| `compute_dvh` | DVH curves and metrics against a dose grid, protocol constraints, CSV |
| `motion_report` | An earlier run's report again, as JSON and CSV |
| `export`, `export_registration` | DICOM into the output folder: SEG or RTSTRUCT, doses, plans, images; a Deformable Spatial Registration object |
| `import_to_archive`, `open_in_viewer` | File an exported folder into the local archive; launch the viewer on it |
| `anonymize` | An anonymized copy of a folder, written under the output folder |

Every call that can take more than a few seconds has an `_async` twin that
returns a job handle at once; `list_jobs`, `job_result` and `cancel_job` go
with it. The synchronous form reports progress to the client. One computing
call runs at a time; a second one is told the server is busy.

The results are what the viewer would have produced: segmentations land as
segmentation series bound to the image series they were made on, ITVs on the
reference phase, per-phase results on their phase, and `export` writes them
with the same identifiers, so the tree looks the same whether a person or an
assistant made it. `open_in_viewer` on the exported folder is the way to look.

Two prompt and resource conveniences: the prompt `heart_target_propagation`
(arguments: the three folders and the target's name) is the standard sequence
for a STAR case, and the documentation pages on registration, propagation,
4D motion, DVH and structure algebra are served as resources so the assistant
can read what the numbers mean.

## Patient identity never leaves

Whatever the server answers ends up in a language model's context, and for
most clients that context leaves the machine. The server is built so that it
has nothing to leak.

**The gate.** When a dataset is opened, the headers are checked against the
anonymizer's own list of identifying tags (patient name, ID, birth date,
contact details, physicians, institution, accession number). A dataset in
which any of them holds a value the anonymizer did not write is
*identifying*, and `phi_policy` decides what happens:

| Policy | Behaviour |
|---|---|
| `refuse` (default) | The dataset is closed again. The error names the tags, never their values, and points to the `anonymize` tool. |
| `redact` | The dataset opens with the identifying values replaced in memory by the anonymizer's alias, so nothing downstream (a report header, an export field) carries them. |
| `allow` | The dataset opens as it is, so an export carries the study's own identifiers. Everything that leaves the process is still scrubbed. |

There is no `off`.

**The door.** No tool returns patient tags: `describe_dataset` does not read
them. Everything else that could carry a name passes one redactor before it
becomes part of a protocol frame: tool results, error messages (which quote
file names), progress messages, the prompt, the resources, the audit log. The
redactor knows every identifying value seen in any open dataset and replaces
it, and it reports paths relative to the root they are under (`root1/4DCT`),
so a folder named after the patient never appears either. Free text from
DICOM files (descriptions, structure names) is capped at 64 characters,
stripped of control characters, and delivered in its own JSON field, so the
assistant sees it as data rather than as instructions.

This is tested, not asserted: `tests/mcp_phi.rs` gives the synthetic phantom
a patient name, an ID, a birth date, a physician and an institution, runs
every tool against it under all three policies, including the error paths,
and fails if any of those values appears in anything the server produced. A
second test does the same over the real executable's standard output.

**The sandbox.** Paths in tool calls are canonicalized and must lie under a
configured root (or the output folder, so results can be re-opened). Output
goes into a per-session folder under `output_dir`; existing files are never
overwritten and no tool deletes. Model weights are not downloaded unless
`allow_model_download` is on; a missing model is an error that names the
viewer's model manager. The audit log records every call after redaction.

## What it is not

It is not a network service: it speaks over standard input and output to the
client that launched it. It has no PACS side. It does not drive the open
viewer: results are files, and `open_in_viewer` starts a viewer on them
(an attach point inside the running viewer is a possible later step, and
would reuse the same tools and the same safety layer, which live in the
library rather than in the executable). And it is not a medical device: like
the rest of the station, it is for research and QA use.

## For developers

`src/mcp/` is compiled only with the `mcp` feature. `config.rs` is the
operator's file; `phi.rs` the gate and the redactor; `session.rs` the open
datasets and handles; `tools/` the tools as plain functions
`fn(&mut Core, Args, &Progress) -> Result<Value>` whose argument structs
derive the JSON schema the client sees; `server.rs` the `rmcp` glue
(transport, progress, cancellation, the `_async` jobs); `prompts.rs` the
prompt and the resources. `Core::call_public` is the one entry point the
server and the tests use, and the only place redaction happens.

The pipelines themselves live in `src/workflow/` (the 4D motion pipeline,
one volume onto every phase of a group, structures by name), shared with the
viewer's tool windows, so the server and the viewer run the same code.
`tests/workflow.rs` is the guard for that: the phantom's target moves 0 / 6 /
3 mm between phases and the pipeline has to find it.

```
cargo build --release --features mcp
cargo test --features mcp --test workflow --test mcp_tools --test mcp_phi
```
