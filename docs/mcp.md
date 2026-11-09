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

1. Build or install the server.

   * **Windows installer** - a tick box on the options page, *Install the MCP
     server*, on by default. Untick it and `rds-mcp.exe` is not written at
     all; the finished page prints the path when it was. `rds-setup.exe
     --no-mcp` is the same answer on the command line.
   * **Linux AppImage** - the server is inside the AppImage, and an MCP client
     is pointed at the AppImage rather than at a file within it:

     ```json
     { "mcpServers": { "rust-dicom-station": {
         "command": "/opt/rust-dicom-station.AppImage", "args": ["mcp"] } } }
     ```

     A copy or symlink of the AppImage named `rds-mcp` also starts the server
     directly, for a client that will not pass an argument. Everything else -
     `--config`, `--check` - works as it does on the standalone binary:
     `./rust-dicom-station.AppImage mcp --check`.
   * **macOS disk image** - the server is inside the application bundle,
     beside the viewer, so a client is pointed at it by full path (it has
     spaces in it, so it needs quoting):

     ```json
     { "mcpServers": { "rust-dicom-station": { "command":
         "/Applications/Rust DICOM Station.app/Contents/MacOS/rds-mcp" } } }
     ```

     Installed through Homebrew, plain `rds-mcp` works too: the cask links
     that same file into the prefix ([macos.md](macos.md#the-mcp-server)).
   * **Flatpak** - the server is a second command of the same application:
     `flatpak run --command=rds-mcp io.github.alexprotom.rust-dicom-station`
     ([flatpak.md](flatpak.md#the-mcp-server-in-a-flatpak)). Its roots must
     be paths the sandbox can see.
   * **Snap** - the server is the snap's second command,
     `/snap/bin/rust-dicom-station.rds-mcp`; that path is what the client
     runs ([snap.md](snap.md#the-mcp-server-in-a-snap)). Its roots must be
     folders the snap may read: under the home folder, or on removable media
     once that interface is connected.
   * **From source** - `cargo build --release --features mcp` produces
     `target/release/rds-mcp` beside the viewer.

   The viewer's *Settings ▶ MCP server* menu says whether it is present.
2. Write the configuration, `mcp.toml`, in the station's configuration folder
   (`%LOCALAPPDATA%\RustDICOMStation` on Windows, `~/.config/RustDICOMStation`
   on Linux, `~/Library/Application Support/RustDICOMStation` on macOS,
   `~/snap/rust-dicom-station/common/config` in the snap,
   `~/.var/app/io.github.alexprotom.rust-dicom-station/config/RustDICOMStation`
   in the Flatpak; the menu shows the exact path). Without it no workspace can be opened:

   ```toml
   roots = ["D:/studies/anonymized"]      # folders that may be read
   output_dir = "D:/studies/rds-mcp-out"  # the one folder results go to
   phi_policy = "refuse"                  # refuse | redact | allow | pseudonymized
   models_dir = ""                        # empty: the viewer's model folder
   allow_model_download = false           # weights are the only network use
   device = "auto"                        # auto | gpu | cpu
   max_open_datasets = 4
   job_timeout_minutes = 60
   audit_log = true                       # data folder /mcp/audit-YYYY-MM-DD.log
   volume_cache_mb = 4096                 # image volumes kept between calls
   workflows_dir = ""                     # empty: the viewer's workflow folder
   archive = ""                           # empty: the station's own PACS

   [pacs]                                 # the curation tools; omit to leave them off
   roots = ["D:/Pryanick-PACS"]           # the archive being curated
   quarantine = "D:/Pryanick-PACS-duplicates"
   records = "D:/Pryanick-PACS-reorg-records"  # manifests, journals, pixel index
   patient_codes = ['^P01-\d{3}$']     # what a pseudonymized patient is called
   ```

   `volume_cache_mb` is the memory for series other than a workspace's
   displayed one (the phases of a 4D group, say), kept between calls so a
   phase is read from disk once rather than by every tool; past it the
   least used are let go. `workflows_dir` is where `list_workflows` looks
   for saved workflows. `archive` is the PACS the archive tools list and
   open: the station's own as the viewer's settings name it, or storage
   folders joined by `|`.
   `[pacs]` turns on the curation tools (below): `roots` are the folders
   they check and correct, `quarantine` where duplicates are moved,
   `records` where every plan, journal and the pixel-hash index is kept;
   all three are readable like a root, and paths under them are reported
   as `pacs1/...`, `quarantine/...`, `records/...`. `patient_codes` are
   regular expressions a pseudonymized PatientName and PatientID match
   whole, for the `pseudonymized` policy.
   `records/pacs-rules.toml` (optional; the defaults otherwise) holds the
   naming rules `plan_reorganize` sorts by, and the patient-folder rules
   every curation tool maps folder names to patient codes with:

   ```toml
   version = 1
   upright_scanners = ["Brilliance Big Bore", "RayStation", "P-Cure"]
   position_keyword = true                # Upright/Supine in study folder names
   max_note = 48                          # longest note in a folder name
   [[patient_folders]]                    # folder name -> code, first match wins
   pattern = 'patient(\d{3})'
   code = 'P01-$1'
   [[patient_folders]]
   pattern = '(\d{2}-\d{3})(_.*)?'        # 01-031_case1_t -> P01-031
   code = 'P$1'
   [study_notes]                          # Study Instance UID -> note
   [series_notes]                         # Series UID, or a folder's name -> note
   "plan_1_arc_lung_med" = "plan-1-arc-lung-med"
   [cohorts.Trial]
   keep_patient_tags = false              # PatientName/ID become the code
   position_keyword = true
   [cohorts.Trial.patient_code_map]       # before the patterns
   "special" = "P01-099"
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

Every entity gets a handle the assistant refers to it by: workspaces `ds1`,
registrations `reg1`, 4D group registrations `greg1`, motion runs `run1`.
Structures are named (`Heart`, `TARGET`), with the structure set or
segmentation series added when a name repeats. Series are numbered as
`describe_dataset` lists them, and default to the displayed one.

| Tool | Does |
|---|---|
| `open_dataset`, `describe_dataset`, `list_structures`, `list_4d_groups`, `close_dataset`, `describe_session` | Open a folder (or files) under a root; see its series, 4D groups, structure sets, segmentations, doses, plans, planar images and reports (a CT dose SR and a scanner's protocol page as tables: the acquisitions with kV, mA, pitch, CTDIvol and DLP, the totals as facts; an SR item that names a person reads `(name)`; and the treatment sessions of a course, `sessions`: per session its number, `day` counted from the first session's and never a date or a time, the day's image modalities and registrations, and per beam the setup pairs, the records' termination status, the correction (translation in mm, rotations in degrees), the verification pairs and dose reports) |
| `list_models` | Every automatic segmentation model: key, modality, classes, sub-models, licence, whether its weights are present or what is left to download |
| `segment_organs` | Any of those models on one series by `model` (`total_mr`, `mrsegmentator`, `lung_vessels`, `lungmask_lobes`, `vista3d`, ...); without it, `variant` picks TotalSegmentator CT: `fast`, `high` (with `parts` such as `cardiac`) or `preview` for the v2 weights, `fast_v3`, `high_v3`, `preview_v3`, `small_v3`, `small_high_v3` for v3; `keep` narrows to named organs, `tg263` names them by AAPM TG-263 |
| `segment_body` | The patient outline, classically or model-assisted |
| `synthesize_ct` | A CT synthesized from an MR or CBCT series (TotalSynth: `app` `sct_mr`, `sct_cbct`, `sct_mr_cbct`, `sct_mr_dixon` or a checkpoint folder's key; `checkpoints` or `all_checkpoints`, `tta` 0 to 2, `uncertainty`; `mask` `network`, `classical`, `none` or a structure name), added to the dataset as a new CT series written under the output folder; the weights are CC BY-NC 4.0 ([synthetic-ct.md](synthetic-ct.md)) |
| `evaluate_sct` | The sCT against a reference CT (same frame of reference, or through `reg`): MAE, ME, RMSE per HU class, PSNR, SSIM inside `mask` (`body`, `covered` or a structure), optionally `wet` (with `hlut`) and `structures` (TotalSegmentator Dice); the numbers and a CSV |
| `combine_structures` | Union / intersect / subtract with margins in mm (uniform or per patient direction) and cleanup |
| `register`, `describe_registration` | Rigid, elastix B-spline or plastimatch B-spline; `region` makes a run local to a structure of the fixed workspace; `start` refines an earlier registration; `init` says where the search starts (automatic, the identity, the centres of gravity, or the centroids of a structure contoured on both) |
| `register_structures` | Registration by structures: the moving series aligned onto the fixed one by structures contoured on both, the intensities left out (each surface laid onto its partner's signed distance map, both ways). `structures` pairs them by name (`moving` when it is named differently there, `weight` for how much one counts); `dof` is `rigid` or `translation`; `refine` adds a local B-spline per structure on the distance maps (`elastix_bspline`, `plastimatch_bspline`), with `start` refining an earlier registration of the same pair; `robust_mm` makes the fit Huber-robust; `init` is `auto` (the structures' centroids), `identity`, or a structure whose centroids are matched. Returns a `reg` handle like `register`, plus the mean surface distance and Dice of every structure before and after |
| `register_2d3d`, `describe_setup`, `export_setup` | 2D/3D registration ([registration-2d3d.md](registration-2d3d.md)): the pose of a CT series whose DRRs match planar images of its dataset (`describe_dataset` lists them, with whether each states its geometry), by `method` `intensity` (`metric` `gc`, `ncc`, `gd`, `pi`, `lncc`, `mi`), `markers` (the MARKER points of interest, named ones, or `detect_markers`) or `hybrid`; each image's `geometry` its own (`image`), a plan `beam`, a calibrated `room` view or `typed`, its `region` (polygons of pixel positions: the part compared) and `picks` (markers placed by hand: `marker` and `at`); `clip_mm` (a cube about the isocentre) or `clip_structure` with `clip_margin_mm` (the box whose shadow is compared; `clip_rays` makes the rays integrate inside it only), `threshold_hu`, `dof`, `isocenter`, `patient`; the renderings run where the server's `device` says; `file_drrs` adds the DRRs at the answer to the dataset. Returns a `setup` handle, the correction in patient coordinates and as couch moves, the displacement, the similarity per view, marker residuals and any warning. `export_setup` writes the rigid Spatial Registration, the DRRs as RT Images and the report into the output folder |
| `propagate` | Carry structures across a registration, to the fixed or the moving side |
| `propagate_to_group` | One series onto every phase of a 4D group, one deformable registration per phase, transforms kept and reused. With `anchor` (a structure contoured on the source and on every phase, the heart say) the run is anchored on it: centroids matched, a rigid fit on the structure plus a margin, a local deformable refinement, and the anchor's Dice against each phase's own contour as the check. This is how a cardiac CT meets a 4DCT. `land` files the results as a segmentation series per phase or as contours in each phase's own structure set; `anchor_landed_as` names the anchor's own copy (`<anchor>_prop` by default); `keep_shape` carries each structure as the rigid fit of the transform over it (shape and volume kept, `rigid_residual_mm` reported) |
| `analyse_motion` | The 4D pipeline: tracks, amplitudes, correlation with a reference structure, per-phase QA, one ITV per target and model. The rigid model is a local fit around each structure (`local_rigid_margin_mm`, 15 by default; 0 for one global fit); a target contoured on every phase is also tracked *as contoured*, straight from the phases' own contours (`contoured`); `phases` restricts the run to some of them |
| `compare_structures` | Volumes, centroid offset, Dice, HD95, mean surface distance |
| `compute_dvh` | DVH curves and metrics against a dose grid, protocol constraints, CSV |
| `motion_report` | An earlier run's report again, as JSON and CSV |
| `export`, `export_registration` | DICOM into the output folder: SEG or RTSTRUCT, doses, plans, images; a Deformable Spatial Registration object |
| `import_to_archive`, `open_in_viewer` | File an exported folder into the local archive; launch the viewer on it |
| `list_archive`, `describe_archive_study`, `open_archive_study` | The station's PACS: every patient as an alias (`anon_...`, never a name or an ID; under `pseudonymized` a patient filed under a code is listed by it) with their studies by Study Instance UID, the year, the modalities, the description and the series - `summary` for counts per cohort only, `cohort` and `patient` to narrow it, `offset` and `limit` to page through it (25 patients a page by default, `next_offset` saying where the next starts), so a listing stays small however large the archive; one study series by series from the headers alone, with what each series becomes when loaded (image volume, planar image, additional object, structure set, ...), its SOP class, image type, pixels, frames and whether its files carry a position, which is how to see why a scanner's protocol page is no radiograph without opening it; and a study opened as a workspace through the same identity gate as `open_dataset`. Built for course studies of tens of thousands of files: the description reads one header per series and lists the sessions by day, the opening can take only some series (`series`); a listing also counts each patient's attachments (files in their folder that are not DICOM). Every study comes with its **4D sets** as the catalog finds them from the headers (`fourd`: group id, name, scan, phases, reconstructions, evidence, confirmed; `fourd_scans`; a series' phase or role; `fourd_orphans`), and the labels typed into the PACS window under `pseudonymized`; `filter: {"has_4d": true \| "probable" \| false}` keeps the studies with a 4DCT, with a probable set or without, and with `summary` gives one line per such study, paged. `open_archive_study` with `fourd_group` opens one set with the structure sets drawn on its phases, and the dataset's `list_4d_groups` lists it under the same id |
| `list_archive_4d_groups` | One study's 4D sets member by member from the catalog alone: each group's id, name, scan, evidence (`phase-labels`, `header-4d`, `same-acquisition`, `geometry-only` = probable, `description`, `custom`) and members (Series Instance UID, number, role, phase, inferred, files), with the structure sets drawn on its phases |
| `anonymize` | An anonymized copy of a folder, written under the output folder |
| `audit_folder`, `check_references` | Curation, read-only, on a folder under the `[pacs]` folders: files that do not parse or parse with a problem, pixel data shorter than its header asks for, slice gaps and doubles, mixed matrices or frames of reference in a series, missing instance numbers, structure sets with fewer StructureSetROI than ROIContour items or their frame of reference in (0008,1155), references to series, instances and frames that are in neither the folder nor the archive, contour planes on no slice, structured reports filed as `DX` - grouped by study, each `broken`, `dangling-reference` or `warning`; `check_references` lists only what does not resolve, for every kind of object |
| `find_duplicates` | Which series of an import a `[pacs]` folder holds already: `identical` (same SOP Instance UID, the same raw elements but for `ignore_tags`), `pixel-twin` (new UIDs, the same SHA-256 of the pixel data at the same position, within one patient folder: a re-anonymized copy), `hu-twin` (the same values - stored value × slope + intercept, HU for a CT - at the same position in other bytes: a copy another program re-encoded, as plastimatch does), `partial`, `ambiguous` (no one archive series holds all its copies; a slice two archive series share, as GE 4D phases share their edge slices, goes to the one that holds the rest), `new`, with the archive series each matches. Inside the import: repeated SOP Instance UIDs (byte-identical, or versions of which the largest is kept) and series that re-export another (the one kept by `prefer`: a full export over a case copy over a plan folder). It saves the UID map the copies give (for `remap_references`), reports splits (a source study or frame of reference with copies in two archive ones), and keeps the hashes in `records`, so a later run reads only what changed. `quarantine_copies` plans moving the copies away |
| `dedupe_rt_objects` | The RT objects of an import (structure sets, plans, doses, RT images, registrations, records, reports) that the archive holds under other UIDs: by a content fingerprint whose referenced UIDs go through the twins' map first (structure sets, then plans, then the rest, each match extending the map, so contours copied onto two 4D phases stay two), else by content without UIDs told apart by the creation stamp, a unique stamp, or a plan's beams and spots or a structure set's contours; copies inside the import keep the largest. Saves the extended map |
| `relink_structure_set`, `remap_references`, `merge_studies`, `merge_frames` | Curation that rewrites references, through three typed UID maps - objects, studies, frames of reference - each UID given to one by where it sits, never by its value (a Philips 4D series shares its UID with its frame); an object's own SOP and series UIDs and private elements are never touched. `relink_structure_set` points a structure set at a CT series of the archive (images mapped through pixel twins, or by slice position after checking the series lie the same way, or by contour z where its CT is missing, a slice no contour lies on by its place in the series-level list, which must run straight in z), filling the frame of ROIs that have none, re-pointing the export's plans and doses with it (`companions`: a plan to this structure set, a dose to the one plan), optionally moving the 4D phases that shared its frame into the target study and frame - only when a series of that frame is a verified twin; `remap_references` applies a map built from twins, saved, or given by hand; `merge_studies` / `merge_frames` merge an earlier split. Each rewrite is run dry first: an old UID left anywhere refuses the plan, and references that would lead nowhere are listed (the relink refuses them) |
| `phi_report`, `plan_pseudonymize`, `patient_demographics`, `set_patient_demographics` | Identified imports: where each patient's names and IDs occur (tag places and counts, never values); a pseudonymization as the archive keeps its trial data - PatientName and PatientID the patient's code (from `pacs-rules.toml` or given), the birth date `YYYY0101`, the identifying tags of the `pseudonymized` policy removed, dates, UIDs, staff names and (by default) private elements kept, in place or on the way to `move_to` - refused while any token of any of the patient's names or IDs (three characters or more, numbers of nine digits or more, a stop-list aside) is left in any text element of the result, nested and private included; patients whose files disagree on birth date or sex (counts only), and a verified rewrite of the two |
| `plan_reorganize`, `apply_reorganize` | Sorting an import (or a cohort) into `cohort/patient/study/series/file`: study folders `<date>_<Upright\|Supine>-<note>_StudyUID-<last 12 digits>` (the earliest session's date and description, Upright by the imaging scanner, `Procedure#n` / `CT-Simulation` where there is no description), series folders `Series-<n>_<Modality>_<note>_SeriesUID-<last 12>` (SeriesDescription, the RT label, `Dose-<type>`, ProtocolName, the date and time), files `<CT\|MR\|RS\|RP\|RD\|RI\|RE\|RR\|...>.<SOP UID>.dcm`, one set of character rules (`_` between parts, `-` within, decimal commas as points, umlauts spelled out); a study the archive has is merged into its folder, a byte-identical copy goes to quarantine, PatientName/ID become the patient folder's code unless the cohort keeps them, attachments (non-DICOM files) are reported or moved; paths stay under 260 characters, the notes giving way first |
| `set_4d_group`, `confirm_4d` | The 4D sets of the station's PACS catalog decided by hand, as the PACS window does: a group made of given series (phases in order), replacing the sets it shares a series with and keeping their id; a set confirmed (a probable one becomes a 4DCT), dissolved, or given back to the rules (`reset`); a copy of the catalog kept first |
| `catalog_relink`, `set_label`, `fill_study_labels`, `catalog_ignore`, `check_patient_identity` | The station's PACS catalog after a curation: one patient entry per patient folder (labels and cohort memberships carried over), a label for a patient or a study, and labels `<YYYY-MM-DD\|undated> (<Upright\|Supine>: <note>) · <modalities>` for every unlabelled study of a cohort (never over a typed one); the folder name patterns a scan never looks into, beside the built-in `to_be_sorted`, `*-duplicates` and `*_duplicates`; each change written the station's way, a copy kept in `records/catalog-backups` first. `check_patient_identity` (read-only) finds one Patient ID given to two people in two cohorts - cohort folders whose files share no birth date and sex - with the cohorts, the folders (for codes), the study counts and how many variants, never the values; one cohort's folder is then recoded with `set_patient_identity` |
| `quarantine`, `set_patient_identity`, `apply_manifest`, `undo_journal` | Curation that writes, always in two steps. `quarantine` (files, or the files of some series or studies, moved to `quarantine/<path below the root>`) and `set_patient_identity` (PatientName and PatientID set wherever they are, in the items of a scanner's own sequences too, the old values never returned), like every planning tool above, only plan: they save a manifest in `records` and return its id. `apply_manifest` does it, journaling each operation (a run cut short resumes where it stopped); a rewrite changes the elements it names and leaves every other byte of the file as it was, is read back and compared element by element before it replaces the file - and must come out byte for byte as the dry run computed it (its SHA-256 is in the manifest), so what the dry run checked is what is written - and keeps a copy for `undo_journal`, which takes everything back, newest first, each file only if nobody changed it since |
| `list_workflows`, `run_workflow` | The saved workflows (and the program's examples) with their steps and inputs, and a run of one: a folder under a root for each input, by its title; the reports come back as tables, the files go into a folder under the output folder, and `open_results` keeps the studies the run read open as workspaces. A *DICOM folders* input runs a batch over its subfolders |

Every call that can take more than a few seconds has an `_async` twin that
returns a job handle at once; `list_jobs`, `job_result` and `cancel_job` go
with it. The synchronous form reports progress to the client. One computing
call runs at a time; a second one is told the server is busy.

The results are what the viewer would have produced: segmentations land as
segmentation series bound to the image series they were made on, ITVs on the
reference phase, per-phase results on their phase, and `export` writes them
with the same identifiers, so the tree looks the same whether a person or an
assistant made it. `open_in_viewer` on the exported folder is the way to look.

A workflow runs under the same rules as the tools. Every folder it reads is
under a root and passes the identity gate first (each case of a batch that
its pattern takes), and under a policy other than `allow` it must be
anonymized already: a workflow reads its folders as they are, so there is
nothing to redact in memory. What it writes goes under the output folder
(a step that names an absolute folder is refused), it reads the station's
archive only when that is under a root, and it files only into the
station's own archive. The folders a workflow file names are never told:
they are its author's, and a folder name can carry a patient's.

Two prompt and resource conveniences: the prompt `heart_target_propagation`
(arguments: the three folders and the target's name) is the standard sequence
for a STAR case, and the documentation pages on registration, propagation,
4D motion, DVH and structure algebra are served as resources so the assistant
can read what the numbers mean.

### Planar images

`project_structures` (structures onto planar images as 2D contours, at the
plan or at a setup), `list_planar_structures`, `draw_planar_structure`
(polygons in pixels), `register_planar` (2D/2D: intensity, points,
markers, deformable), `carry_planar_structures`, `setup_from_planar` (two
matches as one 3D correction, a setup handle) and `save_planar_structures`
(a Grayscale Softcopy Presentation State and the RT Images it needs, file
names without labels). See [planar-2d.md](planar-2d.md).

## Patient identity never leaves

Whatever the server answers ends up in a language model's context, and for
most clients that context leaves the machine. The server is built so that it
has nothing to leak.

**The gate.** When a workspace is opened, the headers are checked against the
anonymizer's own list of identifying tags (patient name, ID, birth date,
contact details, physicians, institution, accession number). A workspace in
which any of them holds a value the anonymizer did not write is
*identifying*, and `phi_policy` decides what happens:

| Policy | Behaviour |
|---|---|
| `refuse` (default) | The workspace is closed again. The error names the tags, never their values, and points to the `anonymize` tool. |
| `redact` | The workspace opens with the identifying values replaced in memory by the anonymizer's alias, so nothing downstream (a report header, an export field) carries them. |
| `allow` | The workspace opens as it is, so an export carries the study's own identifiers. Everything that leaves the process is still scrubbed. |
| `pseudonymized` | For an archive whose patients are codes already. The workspace opens when every PatientName and PatientID matches one of `[pacs] patient_codes`, the birth date is empty or the year alone (`YYYY0101`), and none of the tags a pseudonymization removes holds a value: other names and IDs, birth and mother's names, address, telephone, the medical record locator, the issuer of the ID, the physicians, patient comments. The codes are no secret: they come back as they are, in paths too (`pacs1/Trial/P01-039/...`). Anything else is refused, as under `refuse`, naming the tags; staff and institution names stay in the files and are redacted like any value. |

There is no `off`.

**The door.** No tool returns patient tags: `describe_dataset` does not read
them, and a report's content is returned as its tables with every `PNAME`
item withheld and a page's free lines left out. Everything else that could carry a name passes one redactor before it
becomes part of a protocol frame: tool results, error messages (which quote
file names), progress messages, the prompt, the resources, the audit log. The
redactor knows every identifying value seen in any open workspace and replaces
it, and it reports paths relative to the root they are under (`root1/4DCT`),
so a folder named after the patient never appears either. It works in one
pass, the longest value at each position first, and never scans its own
replacement again. A value of under five characters (a two-letter family
name, initials) counts only as a whole token, with no letter or digit beside
it, so it does not tear `Supine` or `AXIAL` apart; a longer one is found
anywhere, a name glued into a folder name (`Doe-Jane_2024`) included; and no
value is found strictly inside a number or a UID, so an ID or a date does not
cut a Study Instance UID in two. Free text from
DICOM files (descriptions, structure names) is capped at 64 characters,
stripped of control characters, and delivered in its own JSON field, so the
assistant sees it as data rather than as instructions.

This is tested, not asserted: `tests/mcp_phi.rs` gives the synthetic phantom
a patient name, an ID, a birth date, a physician and an institution, runs
every tool against it under all three policies, including the error paths,
and fails if any of those values appears in anything the server produced,
then files the phantom in a PACS and runs the archive tools over it the
same way. A second test does the same over the real executable's standard
output.

**The archive.** The station's PACS is the one place that holds patients
under their own names, so the archive tools name nobody: a patient is an
alias made from the key the catalog files them under (the same in every
call), a study is its UID and its year, the labels typed into the PACS
window are never returned, and every name, ID and label the catalog knows
goes to the redactor before the first answer leaves. Opening a study from
the archive is `open_dataset` with the files taken from the catalog: the
same gate, the same policy.

**The sandbox.** Paths in tool calls are canonicalized and must lie under a
configured root, the output folder (so results can be re-opened) or a
storage folder of the archive the configuration names (which the archive
tools read anyway; what is opened from it passes the same gate). Nothing
else is read. A path may be given the way the server reports it, under its
label: `root1/4DCT`, `output/session/anonymized`, `archive/...`, so what
one tool answers is exactly what the next takes; a `..` behind a label is
canonicalized like any other and cannot climb out. Output
goes into a per-session folder under `output_dir`; existing files are never
overwritten and no tool deletes. The curation tools are the one exception to
"writes only under the output folder", and only within `[pacs]`: they move
and rewrite files under its `roots` and `quarantine` (and move attachments
into `records/attachments`), rewrite the station's catalog
(`catalog_relink`, `set_label`, `fill_study_labels`, `catalog_ignore`, a copy kept first), never in a root's top
level (where the catalog would file a stray file as "other"), never to a path
of 260 characters or more (a warning from 250), and only by applying a
manifest a planning call saved in `records`, which keeps the journal of each
operation and a copy of each rewritten file; they delete nothing either - a
duplicate is moved, not removed. Model weights are not downloaded unless
`allow_model_download` is on; a missing model is an error that names the
viewer's model manager. The audit log records every call after redaction.

## What it is not

It is not a network service: it speaks over standard input and output to the
client that launched it. It has no PACS side of its own (the station's
archive is served to other stations by the separate PACS server,
[pacs-server.md](pacs-server.md)). It does not drive the open
viewer: results are files, and `open_in_viewer` starts a viewer on them
(an attach point inside the running viewer is a possible later step, and
would reuse the same tools and the same safety layer, which live in the
library rather than in the executable). And it is not a medical device: like
the rest of the station, it is for research and QA use.

## For developers

`crates/rds-domain/src/mcp/` is compiled only with the `mcp` feature. `config.rs` is the
operator's file; `phi.rs` the gate and the redactor; `session.rs` the open
workspaces and handles; `tools/` the tools as plain functions
`fn(&mut Core, Args, &Progress) -> Result<Value>` whose argument structs
derive the JSON schema the client sees; `server.rs` the `rmcp` glue
(transport, progress, cancellation, the `_async` jobs); `prompts.rs` the
prompt and the resources. `Core::call_public` is the one entry point the
server and the tests use, and the only place redaction happens.

The pipelines themselves live in `crates/rds-domain/src/workflow/` (the 4D motion pipeline,
one volume onto every phase of a group, structures by name), shared with the
viewer's tool windows, so the server and the viewer run the same code. So
does the headless core under the session: `workflow::session` holds the
budgeted volume cache and the filing of structures under a name rule, and
`workflow::params` the choices (variant, device, method, landing) that the
tools, the dialogs and the workflow steps read by the same names.
`tests/workflow.rs` is the guard for that: the phantom's target moves 0 / 6 /
3 mm between phases and the pipeline has to find it.

```
cargo build --release --features mcp
cargo test --features mcp --test workflow --test mcp_tools --test mcp_phi --test curation
```
