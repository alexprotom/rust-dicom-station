# Workflows

A workflow is what you do by hand in the viewer, written down once as a
graph of steps and run again on other data. *Open this folder, find the CT
and the target in it, segment the heart and file it as `heart total`, open
the 4DCT, do the same on every phase, carry the target across anchored on
the heart, measure the motion, build the ITV, write everything out* is one
workflow of twelve steps. Saved, it is a small file; run, it reads the
folders it names (or the ones the run dialog points it at), does every step
with the program's own code - the same engines, pipelines and landing rules
the modules use - and writes its results into a folder of its own.

A run either works **in the background**, leaving the workspaces alone, or
**shows every step in the viewer**: each study is put in a workspace and,
after each step, the viewer shows what the step did before the next one
starts, the way a person working through it would see it. A workflow that
starts from *DICOM folders* is a **batch**: it runs once per patient
folder and puts the tables of every case together. The MCP server runs
saved workflows too (`list_workflows`, `run_workflow`; see [mcp.md](mcp.md)).

## The Workflows menu

| Entry | What it does |
|---|---|
| 🔀 New | Opens the workflow editor on an empty canvas. |
| 📂 Load | Opens a workflow file (`.rdsflow`) in the editor. The dialog starts in your workflow folder. |
| 🕐 Last saved | The five workflow files saved last, newest first. |
| Examples | Workflows that ship with the program, opened as unsaved copies to adapt. |
| Show the editor / Show the run | Brings back a closed editor or run window; the workflow and the run stay in memory while their windows are closed. |

Your workflows are kept in `<data folder>/user_data/workflows`
(`%LOCALAPPDATA%\RustDICOMStation\user_data\workflows` on Windows,
`~/.local/share/RustDICOMStation/user_data/workflows` on Linux,
`~/Library/Application Support/RustDICOMStation/user_data/workflows` on
macOS); the menu shows the path. Any other folder works as well - the file
dialogs only start there. The *Last saved* list lives in
`viewer_settings.txt` as `recent_workflows`.

Replacing a workflow that has unsaved changes (New, Load, an example, a
recent file) asks first, in the editor window.

## The editor

A window of its own with four parts:

* **The canvas**, in the middle. Every step is a node: its header, then
  what it gives (output pins on the right edge), a few lines saying what it
  is set to do, then what it takes (input pins on the left edge). A wire
  runs from an output pin to an input pin. Drag a node by its header, pan by
  dragging empty space, zoom with the wheel; **⛶ Fit** (or a double click on
  empty space) brings every step into view. A right click on empty space
  offers every kind of step; a wire dropped on empty space offers the steps
  that can take it and connects the new one. A right click on a node offers
  *Duplicate* and *Remove*; a right click on a pin drops its wires; the ▼ in
  a header folds the node. **Delete** removes the step shown in the
  inspector, or the steps selected with Shift-click or a Shift-drag.
* **The palette**, on the left: the steps by group, one click adds one. Under
  it, *Frames* (below) and *Workflow*: a description, and where runs write
  (below).
* **The inspector**, on the right: the selected step's title, what it does,
  its parameters, and what its pins take and give.
* **The status line**, at the bottom: whether the workflow can run, and if
  not, why - a required input left free, a folder not given, an organ name
  TotalSegmentator does not know, a circle of wires.

**Wires are typed.** A pin's colour says what it carries, and a wire goes
only where its type is accepted; a refused wire says why in the status
line. An input that takes one wire (a round pin) gives up the old one for
a new one; an input that takes several (a square pin) collects them.

| Colour | Carries |
|---|---|
| grey | a **study**: everything read from one folder |
| violet | an **image series** of a study |
| orange | a **4D group**: its phases, in order |
| green | **structures** by name, on one image series or on every phase of a group |
| blue | a **registration** (or one per phase) |
| red | a **report**: what a step measured |

Structures carry where they are. Structures made on every phase of a 4D
group name that group, so a step that wants a group (*Propagate to 4D
group*'s *Onto*, *Motion and ITV*'s *Group*) also takes the structures -
which makes it wait for the step that made them.

Node headers are coloured by group (input, finding, segmenting,
registering, 4D, output). While a run is going each header carries the
step's state: ⏳ running (outlined), ✔ done with its time, ✖ failed with
the reason.

**Editing** with the pointer over the canvas and no text field active:

| Keys | What they do |
|---|---|
| Ctrl+Z, Ctrl+Shift+Z or Ctrl+Y | Undo, redo (also **⟲ Undo** / **⟳ Redo**). A drag or a typed title is one step of the history; the last 100 are kept. |
| Ctrl+C, Ctrl+X | Copy or cut the selected steps (else the one in the inspector) with the wires between them. The clipboard holds them as workflow JSON, so they paste into another workflow, or another copy of the program. |
| Ctrl+V | Paste under the pointer, with ids of their own. |
| Ctrl+D | Duplicate the selected steps. |
| Arrow keys | Move the selected steps by 10 units; with Shift, by a grid square. |
| Delete | Remove the selected steps. |

**Frames.** **⬚ Frame** draws a titled box around the selected steps, behind
them: a row of the example, the phase work, a note for whoever opens the
file next. The box follows its steps; dragging its title moves every step
inside. Title and colour are set under *Frames* in the palette (🗑 removes
a frame, not its steps). Frames are saved in the file and mean nothing to a
run.

**The map.** With *Map* ticked, the canvas's corner shows every step and
frame in small, with the part in view outlined; a click or a drag on it
moves the view there.

**Save** writes to the file the workflow came from, or asks for a name
(in your workflow folder) the first time; **Save as** always asks. The title
bar carries a `*` while there are unsaved changes.

## The steps

| Step | Takes | Gives | What it does |
|---|---|---|---|
| 📂 DICOM folder | - | study | Reads every DICOM file in the folder and its subfolders: patients, studies, series, RT objects, and the 4D groups among the series. *Workspace* is where a run that shows its steps puts the study. |
| 📁 DICOM folders | - | study | A batch: the folder's subfolders that match the pattern (`*`, `P*`, `case ??`) are the cases, one patient each, and the whole workflow runs once per case (see *Batches*). One per workflow. |
| 🏥 From the archive | - | study | A study of the station's local archive (*Tools ▸ PACS*): the patient by ID or name, the study by date or words of its description, or the newest. |
| 🔏 Anonymize | study | study | Writes an anonymized copy of the study's folder into the run folder (identifiers replaced by an alias, dates fixed, private tags removed, UIDs remapped) and goes on with the copy. |
| 🔍 Image series | study | image | One image series by modality, words in the description, *the largest* (most slices), *the first* or *the last*; optionally not a phase of a 4D group. |
| 🎞 4D group | study | group | The 4D group whose name contains the words given (empty: the first). When the loader recognised none, the series of one modality can be grouped instead (by phase percent, temporal position, series number). |
| 🎯 Find structures | study, image or group | structures | Names or patterns with `*` and `?`, separated by commas (`target*, GTV*`), case-insensitive; *only the first match* takes one. On an image it looks at the structure sets and segmentations drawn on that image; on a group, a name counts when every phase has it. |
| 🔬 Auto-segmentation | image or group | structures, report | TotalSegmentator (3 mm, 1.5 mm or 6 mm) on the image or on every phase. *Organs to keep* lists the classes kept and the name each is filed under (`heart` as `heart total`); none listed keeps everything found. With 1.5 mm only the sub-models holding the listed organs run. Filed as RT structures (type ORGAN), segments, or both - in the image's own structure set (each phase's own on a group; a new one when it has none) or always a new set. A listed organ not found on an image stops the run. |
| 👤 Body contour | image or group | structures, report | The body outline (type EXTERNAL), classical or model-assisted, filed like the organs. |
| 💬 Prompt by name | image or group | structures, report | SegVol with text prompts (`liver`, `pancreas`, `aorta`): one line per structure, the prompt and the name it is filed under (empty keeps the prompt). For what TotalSegmentator's classes do not name. |
| ⊕ Combine structures | A, B (optional, several) | structures, report | A alone with a margin (a PTV from an ITV), A ∪ B, A ∩ B or A − B, each side grown or shrunk first (uniform or per direction), the result cleaned (fill holes, close, keep the largest piece, drop small ones). On an image, or on every phase of a group. |
| ✏ Rename or delete | structures | structures | Renames the structures that arrived by rules (`GTV*` to `GTV`), or deletes them, where they are: on an image, on every phase, or in the whole study. When no rule matches anything, the run stops. |
| 📌 Transfer by relationship | target, reference, onto | structures, report | Places the target onto another image (or every phase) at the offset it has from a reference structure both images carry: the target placed by the heart when the images share no registration. |
| 🔁 Copy to each phase | structures, group | structures | Copies structures onto every phase as they are, in patient coordinates, no registration: a fixed margin, a couch, an ITV. |
| ⇄ Register | image, image | registration, report | Rigid or rigid + B-spline (elastix) or B-spline (plastimatch) of the moving image onto the fixed one, with the start (automatic, identity, centres of gravity) and the effort. |
| ➕ Propagate | registration, structures | structures, report | Carries the structures across the registration onto the other image, into its structure set or a segmentation series, with an optional name suffix, closing, filling and *keep the shape*. |
| ⏩ Propagate to 4D group | structures, anchor (optional), group | structures, registrations, report | One image onto every phase, one registration per phase. With an anchor - a structure on the source image that every phase also has, the heart - the run is anchored: centroids matched, a rigid fit on the anchor plus a margin (by contours or by intensity), then a local deformable refinement, and the anchor's own landed copy (`<anchor>_prop`) is compared with each phase's contour (Dice, HD95, centroid distance, a verdict). Without one, a plain deformable run. Filed in each phase's own structure set (or a segmentation series). See [star-target-propagation.md](star-target-propagation.md). |
| 📈 Motion and ITV | targets, reference (optional), group (optional) | structures (the ITVs), report | The 4D motion pipeline of [motion-4d.md](motion-4d.md): each target (and the reference structure) as contoured on every phase when every phase has it, rigidly (around each structure, or one global body) and deformably; centroid tracks, peak-to-peak, correlation with the reference, registration QA; the ITV per target and model with an optional margin, filed on the reference phase. |
| 📊 DVH | structures (several) | report | Dose-volume histograms against a dose of the study (by words of its label, else the first): the metrics listed (`D95%, D2cc, V20Gy, Dmean`), and with a protocol (`Heart Dmean < 5`, one per line, or a protocol file) which constraints hold. Optionally the cumulative curves. |
| 📐 Dose estimation | structures (several) | report | One row of dose metrics per structure against the physical or the RBE-weighted dose, as the *Dose estimation* module shows it. |
| ☢ DRR | image | report | Radiographs of the image at the gantry angles listed, or at the plan's beams: PNG files in the run folder, and optionally planar images of the study. |
| 💾 Export DICOM | studies (several) | - | Writes each study it is given: its structures as RTSTRUCT or SEG - only the sets the run made or added to, or all - and optionally its images, doses and plans, keeping the UIDs or minting new ones. |
| 📋 Save report | reports (several) | - | One CSV per table of every report (the motion report also as its full CSV) and one Markdown summary. |
| 📥 File in the archive | studies (several) | - | Files the studies into the station's local archive: what an *Export DICOM* step of the run wrote of them, or the folders they were read from. |

Folder parameters (*Export DICOM*, *Save report*) are inside the run's
folder unless absolute; `{input}` is the title of the folder node the study
came from, `{workflow}`, `{date}` and `{time}` are the run's.

**Names already taken.** Every step that files structures says what
happens when a set already holds the name (*Name taken*): *add a counter*
(`heart total (2)`, the default, as the modules do), *file it as
name_prop* (`heart total_prop`), or *replace the one there*. On a 4D group the
choice is made once for every phase, so the target lands under one name on
all of them even when only one phase had a structure of that name; the
steps downstream are handed the names as landed.

## Running

**▶ Run** in the editor opens the run window on the workflow as it is on
the canvas.

**Inputs.** One line per folder node, filled with the folder the node
names. Change it here to run the same workflow on other data: the file is
not changed. A relative folder is looked for beside the workflow file, then
in the current folder, then beside the program.

**Results go to.** The run makes a folder of its own there,
`<workflow name> <date>-<time>` (the template is under *Workflow* in the
palette), with ` (2)` added if it exists. The default root is
`<data folder>/workflow_runs`.

**How.**

* *Show every step in the viewer.* Each study the run reads is put in the
  workspace its folder node asks for, else an empty one, else the next one
  the run is not using. A study of yours in a workspace the run takes is
  kept aside as it was (views, cursor, the structures shown) and **Put back
  A and B** in the finished window returns it; loading something else into
  that workspace instead lets it go. After every step the viewer shows what the step
  did: the image it worked on is displayed, the structure set it filed into
  is the active one, the tree lists what it added, a 4D group is stepped
  through phase by phase as it is segmented, and the motion results window
  opens when the motion is measured. The run then waits - *Show each step
  for* a few seconds, or, with *Wait for me after each step*, until *Next
  step*. While the run is going, the workspaces it shows are its own: every
  step replaces the study with the run's copy.
* *In the background.* The workspaces are left alone. When the run is done,
  *Show the studies in the viewer* puts every study it read, with what it
  filed, into the workspaces.

*Download model weights that are missing* lets an engine fetch what it
needs on first use; off, a step that would download stops the run and says
how much.

*Run independent rows side by side* (background runs and batches): the rows
at the start of the canvas that share nothing, the two inputs of the
example, run at the same time, each on its own thread; the results are the
ones the rows give one after the other. It costs the memory of both rows
at once. When the last run's steps can be taken over (below), they are, and
the rows do not run at all; a run whose rows ran side by side keeps nothing
to take over, so the next run starts from what the last plain run left.

*Take over unchanged steps from the last run* (on by default): the window
keeps what every step of its last run left, filed under a fingerprint of
the step, its settings, its wires, the folders' contents, and every step
before it in the run order. The next run takes over each step whose
fingerprint it finds (marked ♻ on the canvas and in the list) and runs the
rest; change the ITV margin and only the motion step and what follows run
again. Steps that write files always run, into the new run's folder.
*Forget* lets go of what is kept (and its memory).

*Keep up to N MB of image volumes between steps*: the phases a step read
stay in memory for the next step that reads them, the least used let go
past the budget.

While it runs, the window shows the step, its progress bar and message,
every step's state and the log; **⏹ Cancel** stops it at the next check.
The first step that fails ends the run: nothing after it runs, and what the
earlier steps made stays (to look at, and in the run folder). When it is
done the window lists every step with what it said, the reports with their
tables (📋 copies one as CSV; 📈 opens a motion report in *Motion
results*), and the run folder's path.

### Batches

A workflow whose first step is **📁 DICOM folders** runs once for every
subfolder of the folder given that matches the pattern, always in the
background. The run dialog says how many cases it found. Each case is a
plain run in a folder of its own, with the folders step reading that
subfolder and titled after it; a case that fails is listed with its reason
and the next one runs. The finished window lists the cases, and every
report table comes once more with a *Case* column and a row per case:

```
<results root>/<workflow> <date>-<time>/
  batch-summary.md          the cases and the combined tables
  batch - Cases.csv         one row per case: result, time, folder
  batch - <report> - <table>.csv
  case 1/ case 2/ ...       each case's own run folder (below)
  workflow.rdsflow
```

### What a run writes

```
<results root>/<workflow> <date>-<time>/
  workflow.rdsflow     the workflow as it ran, with the folders it read
  run-summary.md       inputs, every step with its lines and time, every report
  CCT/ 4DCT/ ...       what the Export DICOM steps wrote
  reports/             what the Save report steps wrote (CSV + Markdown)
```

`workflow.rdsflow` in the run folder opens like any workflow: the run is
reproducible from its own results.

### Without the viewer

The same run from the command line, for batches:

```
cargo run --release --example workflow_cli -- my.rdsflow \
    --input "Cardiac CT=D:/data/P02/CCT" --input "4DCT=D:/data/P02/4DCT" \
    --out D:/results
```

`--input` names a folder node (or a *DICOM folders* node) by its title (or
id); `--list` prints the steps and the folder nodes; `--no-download`
refuses downloads; `--parallel` runs independent rows side by side;
`--memory <MB>` sets the volume budget.

## The example: heart-anchored target motion

*Workflows ▸ Examples ▸ Cardiac CT and 4DCT: heart-anchored target motion*
is the cardiac radioablation case as a workflow:

```
Cardiac CT ─ The CT ─┬─ Find the target (target*, GTV*, CTV*, PTV*) ──────────┐
                     └─ Heart on the CT (heart as "heart total") ─┬─ anchor ───┤
                                                                  └─ CCT results (export)
4DCT ─ The 4D group ─ Heart on every phase (heart as "heart total") ─ onto ────┤
                                                                               ▼
                                            Target onto the phases (anchored on heart total)
                                                            │
                          reference: heart total ─ Heart and target motion (+ ITV)
                                                            │
                                   4DCT results (export) ◀──┴──▶ Reports
```

1. **Cardiac CT** reads the CCT folder (`data-test-complex/UPSTAR/STAR R P01`
   in the example; point it at yours), **The CT** takes its CT series,
   **Find the target** its target (`target_volume` in the example data).
2. **Heart on the CT** runs TotalSegmentator on the CT, keeps the heart
   and files it as `heart total` in the CT's own structure set. **CCT
   results** writes the CT and that set (RTSTRUCT) into `CCT/` of the run
   folder: the input, saved with what the run added to it.
3. **4DCT** reads the 4DCT folder (`data-test-complex/UPSTAR/Lung
   01-052`), **The 4D group** takes its phases, **Heart on every phase**
   segments each phase and files `heart total` in that phase's own
   structure set, connected to that phase.
4. **Target onto the phases** carries the target from the CT onto every
   phase, anchored on `heart total` (by contours, 10 mm margin, rigid then
   local deformable) and files it, with the heart's own landed copy
   `heart total_prop`, in each phase's set. Its report has each phase's
   registration and the heart's Dice.
5. **Heart and target motion** measures the target and the heart through
   the phases - as contoured, rigid and deformable - their correlation, and
   builds the ITV on the reference phase.
6. **4DCT results** writes the phases and every phase's structure set into
   `4DCT/`; **Reports** writes the four reports into `reports/`.

The two export steps write the images too (*The images too*), so each
folder is a complete study whose structure sets reference their slices.
With *The images too* off and the UIDs kept (the default), a structure set
references the images as their source files have them, slice by slice:
the results go back next to the original images, or into the archive that
holds them, and bind to them there. With new UIDs, a structure set written
without its images keeps only its frame of reference; the export says so
either way.

Save it (it opens as an unsaved copy), and run it on the next patient by
changing the two folders in the run dialog - or change what it does: the
organ and its name in the two segmentation steps, the target's pattern,
the motion models, the ITV margin, or replace a step with another.

## The file

JSON, pretty-printed:

```json
{
  "format": "rds-workflow",
  "version": 1,
  "name": "Heart-anchored target motion",
  "description": "...",
  "output": { "root": "", "folder": "{workflow} {date}-{time}" },
  "nodes": [
    { "id": 1, "title": "Cardiac CT", "pos": [0.0, 0.0],
      "kind": "load_folder", "params": { "path": "D:/data/CCT", "workspace": "a" } },
    { "id": 2, "title": "The CT", "pos": [250.0, 0.0],
      "kind": "select_image", "params": { "modality": "CT", "pick": "largest" } }
  ],
  "links": [ { "from": [1, 0], "to": [2, 0] } ]
}
```

`kind` names the step, `params` its parameters (every one optional: what a
file does not say takes its default, so files stay readable as steps gain
parameters); `from` and `to` are `[node id, pin index]`, pins counted from
the top from 0. A file of a newer format version, or with a step this
program does not know, is refused with the reason.

## Order of the steps

A step runs after every step it takes something from. Steps that do not
depend on each other run top to bottom as drawn on the canvas (then left to
right) - so the two input rows of the example run one after the other, and
moving a row changes which runs first. Steps run one at a time, every
engine parallel inside; with *Run independent rows side by side*, the rows
before the first step that joins them run at the same time, and their
results are merged in the order above, so the run's records, reports and
names are those of the serial run.

## For developers

```
src/workflow/
  params.rs    choices shared by the module dialogs, the MCP tools and the steps
               (variant, device, method, landing, name clash) with their file names
  session.rs   the headless core under both the runner and the MCP session:
               the budgeted volume cache (Volumes), UIDs, filing structures
               into a study under the name-clash rule (file_items)
src/workflow/graph/
  mod.rs       the document model: Workflow, Node, Link, Frame; check(), order(),
               fragment() / paste() for the editor's clipboard, templates
  catalog.rs   every kind of step: ports, typed pins, parameter structs (serde), summaries
  exec.rs      the runner: values on wires, the run's copy-on-write state, events,
               reruns (StepCache), parallel rows, batches, the outcome, run files
  nodes.rs     what each step does, and how results are filed in their study
  nodes/more.rs  the archive, anonymize, prompt, combine, rename, transfer,
               copy-to-phases, DVH, dose, archive-import and DRR steps
  store.rs     the user folder, load / save, the recent list, the shipped examples
  examples/    the example workflows, compiled in
src/app/
  workflow_edit.rs   the editor window: the egui-snarl canvas, palette, inspector, file
                     actions, history, clipboard, frames, the map
  workflow_run.rs    the run window, the job, showing steps in the workspaces (and
                     parking the user's studies), reruns, results
src/mcp/tools/workflows.rs   list_workflows, run_workflow
examples/workflow_cli.rs     the headless runner
tests/workflow_graph.rs      the example rebuilt on the 4D phantom, run end to end
tests/workflow_steps.rs      the other steps, reruns, parallel rows, batches
```

The runner is headless like the rest of `workflow`: a value on a wire is a
reference into the run's own state (the studies it read, the registrations
and reports it made), and a step that files structures adds them to the
study they belong to, recording which sets it touched (what *Export DICOM*
with *what the run changed* writes). The state is copy-on-write
(`Vec<Arc<Dataset>>`, `Arc::make_mut` on the one a step changes), which is
what makes keeping it after every step for a rerun, and splitting it
between parallel rows, cheap. The viewer runs `exec::run` on a `Job`; the
MCP server or a test runs it on any thread with a `Channel`.

**Adding a step:**

1. a parameter struct in `catalog.rs` (`#[serde(default)]`, `Default`), a
   variant of `Op` and of `Kind`, its entry in `Kind::ALL`, `info`,
   `inputs`, `outputs`, `default_op`, `Op::kind`, `Op::summary` and, where
   there is something to check, `Op::param_problems`;
2. its function in `nodes.rs`, reached from `run_node`: read the inputs,
   call the engine or the `workflow` pipeline, file results through `land`
   (so the touched sets are recorded), add a `Report`, return the outputs
   in pin order and the studies to show;
3. its form in `workflow_edit.rs` (`params_ui`), with `names_row` when it
   files structures;
4. if it reads files, their signature in `step_fingerprint` (exec.rs), so a
   rerun sees them change; if it writes, `writes: true` in its `KindInfo`;
5. a test, and a line in the table above.

The catalogue test checks that every kind round-trips its defaults through
the file format and sits in one palette group; the store test that every
shipped example opens and passes its own checks.
