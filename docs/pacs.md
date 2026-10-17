# The PACS

*Tools ▶ 🏥 PACS* opens the **Picture Archiving and Communication System**
window: this station's own store of DICOM data, laid out the way a person
keeps it - cohorts, their patients, each patient's studies, each study's
series - and every PACS server the station is paired with.

It is a PACS in the sense that matters at a workstation - a persistent place
where patients live between sessions, listed without opening a file - not in
the sense of DICOM networking: no association negotiation, no C-FIND, C-MOVE
or C-STORE. The PACS is one or more folders on disk.

One station's PACS can be served to others: the optional PACS server,
`rds-pacs`, puts an HTTPS door on it, and every station can pair with such a
server, pull its studies (or single series), work on them and send results
back, or hand it workflows to run. The servers a station is paired with
appear in this same window, beside *This station*; see
[pacs-server.md](pacs-server.md) for the server, the setup on a local
network and over the internet, and the two ways of working.

## The window

| | |
|---|---|
| **Source** | **🏥 This station** (this page), one **🔗** button per paired server ([pacs-server.md](pacs-server.md#working-on-a-servers-studies)), **➕ Add server** |
| **📥 Import folder** | Copy every DICOM file under a folder into the main storage folder |
| **📤 Send workspace A / B** | Write that workspace's structure sets and segmentation series back into the PACS, attached to the study they belong to |
| **⟲ Rescan** | Look through the storage folders again; only files not seen before are read |
| **🔍** | Show only the rows whose name (or UID, or folder) holds the text, with what leads to them |
| **📩 Load into workspace A / B** | Read the selected rows into a viewer workspace |
| **⚙ Settings ▶ Storage folders** | The folders the PACS reads (below) |
| **⚙ Settings ▶ PACS server** | The server on this station: start, stop, connection details, pairing, its settings ([pacs-server.md](pacs-server.md)) |

The tree has four levels:

```text
📁 Compassionate studies        4 patients · 2239 files
  👤 GC (X24)                   2 studies · 824 files
    📂 2021-03-02 - CT Abdomen · CT, RTSTRUCT     3 series · 261 files
      🎞 CT 2 - Thorax 3 mm     258 files
      🎞 RTSTRUCT 1 - Clinical  1 file
📁 Unsorted                     1 patient · 756 files
```

**Selecting.** A click selects one row; Ctrl and a click (Cmd on a Mac) adds
a row to the selection or takes it out; Shift and a click selects a range.
The keyboard does the same: Up and Down move (with Shift they extend the
selection), Left and Right fold and unfold, Home and End jump, Ctrl+A selects
every row shown, Escape clears, F2 renames and Delete removes. A row stands
for everything below it: a cohort for its patients, a patient for their
studies, a study for its series. *Load into workspace* reads exactly the
files of the selection - one series out of a study of twelve, three studies
of different patients - each file once, however the rows overlap. *Help* in
the window's menu bar lists the keys.

**Right-click** a row for what can be done with it, or with the whole
selection when the row is part of it:

| | |
|---|---|
| **✏ Rename** | Give a cohort, patient, study or series a name of your own (below) |
| **→ Move to cohort** / **Copy to cohort** | Patients only: put them into another cohort, or into another one as well; **➕ New cohort** makes one on the way |
| **Take out of this cohort** | Patients only; one left in no cohort goes to *Unsorted* |
| **Back to the cohort of its folder** | Forget the moves and copies made here for these patients |
| **➕ New cohort** / **🗑 Delete this cohort** | On a cohort row; only a cohort made here can be deleted, and its patients go back to their folders' cohorts |
| **📁 Move the files to a folder** | Move the DICOM files themselves (below) |
| **🗑 Remove from the PACS** | Delete the files from the disk, asked once more |

## Cohorts

A **cohort** is a group of patients: a trial, a retrospective study, the
patients of one collaboration. The PACS reads them from the folders:

```text
<storage folder>/
  rds-pacs-catalog.json             the catalog (main folder only)
  <cohort>/                         a folder holding several patients
    <patient>/
      <study>/
        <series>/ or the files
  <patient>/                        a patient's folder outside any cohort
```

A folder directly in a storage folder that holds the files of **several
patients** is a cohort, named after the folder. A folder holding **one**
patient is that patient's own folder (and the patient is in *Unsorted*) when
it holds files itself or is named after the patient - their ID, or their
family name - and a cohort that holds one patient so far otherwise. Files
lying directly in a storage folder are in *Unsorted* too.

After that the person decides, and **no file moves for it**: a cohort is
renamed, patients are moved or copied between cohorts (a patient can be in
several), new cohorts are made for any grouping, and a cohort made here is
deleted again. All of it lives in the catalog. Moving the files themselves
is a separate act.

## Names

*Rename* gives a cohort, a patient, a study or a series a name of your
choosing - `Case 24` for `GC (X24)`, `Planning CT` for
`2021-03-02 - CT Abdomen`. The name is kept in the catalog; **nothing is
written into a DICOM file**, and the tooltip of a renamed row shows what the
files say. *The DICOM name* in the rename bar goes back to it. A cohort
named after a folder can be renamed without renaming the folder.

## Several storage folders

*⚙ Settings ▶ Storage folders* lists the folders the PACS reads. Any number
can be added - a second disk, a share, the folder a collaborator's data
arrived in - and each is read where it is; nothing is copied. The **first is
the main folder**: what this station files (an import, the structures sent
back from a workspace) goes there, and the catalog lives there. *Make main*
makes another one the first; *Remove from the list* stops reading a folder
and leaves its files on the disk.

The folders are kept in `viewer_settings.txt`: `archive_dir` names the main
one, one `archive_more` line each of the others. The main folder defaults to
`archive` inside the platform data directory (`%LOCALAPPDATA%\RustDICOMStation`
on Windows, `~/.local/share/RustDICOMStation` on Linux, `~/Library/Application
Support/RustDICOMStation` on macOS).

## The catalog

Listing a PACS must stay quick however large it grows, and reading the
headers of ten thousand files is not. So every file is read once and
remembered in **one file for the whole PACS**, `rds-pacs-catalog.json` in the
main folder: per file its path, size, time and SOP Instance UID, and per
patient, study and series what the window shows. Beside that it keeps what
the person decided: the cohorts, who is in which, the names given.

* A scan walks the folders (quick) and reads only the files it has not seen
  or that changed (size or time); the rest comes from the catalog. The first
  scan of a large PACS reads every header once, in parallel. A scan that
  finds nothing new and nothing gone writes nothing.
* An import reads the headers of what it files in parallel and writes the
  catalog once at the end; a pull from a server, bundle after bundle, writes
  it once per study.
* **JSON**, one line per patient, study and series, so that any tool reads it
  and a person can open it.
* The DICOM files are the truth: delete the catalog and the next scan builds
  it again, from the files alone. Only the names, the cohorts and who is in
  which are lost then, being the catalog's alone.
* Two programs may write it at once - the viewer and a PACS server serving
  the same folders. Each write takes a lock file beside it
  (`.rds-pacs-catalog.lock`), reads the catalog afresh, changes it and puts
  it back in one rename, so a reader never sees half of it and neither
  writer loses the other's change.
* Earlier versions wrote `STUDY.txt`, `PATIENT.txt` and `INDEX.txt` into
  every folder. A scan removes those, recognising them by name and content,
  so that a person's own file of the same name stays.

Patients, studies and series come from the **headers**, never from where a
file lies: a patient's CD with the files of four studies in one folder is
listed as four studies, a study whose series sit in folders of the exporting
system's naming (`0142`, `IM000001`) as that study, a folder holding two
patients as two patients. A patient is known by their Patient ID (by their
name when there is none).

## Moving files

*📁 Move the files to a folder* moves the DICOM files of the selection
themselves into another folder **of the same storage folder**: the page that
opens lists its folders, one is chosen (or a new one made inside it, with
*📁 Make it*), and *→ Move here* moves them. The files keep the folders they
had below the folder that held them all, so a study's series folders arrive
as series folders. Nothing is moved when a file of the same name is already
at its place; folders left empty are removed. The catalog follows the move,
so no header is read again.

## Filing

Import copies; it never moves. Each file is opened as far as the pixel data
(not into it), and its Patient ID, Study Instance UID and SOP Instance UID
decide where it lands: beside the study's files when the main folder holds
the study already, else in `<main folder>/<patient>/<study uid>/` as
`<sop instance uid>.dcm`. A file whose SOP Instance UID the PACS holds
already is left alone, so importing the same folder twice is a no-op and
re-importing a folder that has grown files only the new ones. Anything that
will not open as DICOM is counted as skipped and reported, not treated as an
error. A filed instance is copied beside its place and renamed into it, so a
reader - the PACS server sending the study on - never takes half a file.

Only the patient folder name comes from free text and so is sanitized:
anything outside ASCII letters, digits, `.`, `-` and `_` becomes `_`, capped
at 96 characters.

## Taking a selection into the viewer

*Load into workspace A / B* reads the files of the selection through the same
loader as *File ▶ Add DICOM file(s)*, with the same classification, the same
patient ▶ study ▶ series tree and the same merging into whatever the workspace
already holds. The session remembers the folders the selection covers (a
folder whose every file was loaded) rather than each file.

## Sending changes back

*Send workspace A / B* writes back **derived objects only** - the structure sets
and the segmentation series. The images are already in the PACS; re-sending
them would duplicate hundreds of megabytes under new Instance UIDs.

Each written object gets

* a **fresh SOP Instance UID and Series Instance UID** - a new object, not a
  replacement, which keeps the PACS append-only and a mistaken upload
  harmless;
* the **original Study Instance UID**, from the object itself where it says so
  and from the series it references otherwise, filing it under the patient and
  study it belongs to;
* the **original Frame of Reference UID**, so the contours and masks still sit
  on the images they were drawn on;
* a **reference to the image series** it was drawn on.

The objects are written to a scratch folder (removed afterwards) and imported
through the ordinary import path, so a half-failed upload leaves the PACS
untouched. Every send creates new instances, so sending twice leaves two
structure sets rather than overwriting one; the unwanted one can be removed
through the data tree, or as a series from the PACS.

## What it is not

* **Not a DICOM network node.** No SCP, no SCU, no AE titles. Files move by
  the file system, or over HTTPS through the PACS server
  ([pacs-server.md](pacs-server.md)).
* **Not multi-user by itself.** Two instances pointed at the same folders
  will not corrupt them - files are written under unique UIDs, the catalog is
  locked and renamed into place - but a listing can go stale until rescanned.
  Sharing a PACS between people is what the PACS server is for.
* **Not an anonymizer.** What goes in is what comes out. *Tools ▶ Anonymize*
  ([export-and-tools.md](export-and-tools.md)) is the pass to run before
  filing anything that must leave the department.

## Verification

`tests/archive.rs` runs the whole round trip on the synthetic phantom study
([example-data.md](example-data.md)): generate, file, list, load the filed
study back through the ordinary loader, draw a segmentation, send the derived
objects back, and assert they joined the same patient and study - no second
patient or study, the Study Instance UID unchanged, the file count grown by
exactly the objects written, `SEG` now among the study's modalities, the
segmentation present on reload. Two more tests take a study copied in by hand
(series folders, files named by the exporting system) and a patient's CD with
two studies in one folder: listed by their real UIDs, served and loaded each
with its own files, removed one at a time, no sidecar written anywhere. A
fourth lays out a PACS the way a person does (cohort folders, a patient
folder outside any, a further storage folder) and checks the cohorts read off
the folders, renaming, moving and copying patients between cohorts with no
file touched and all of it surviving a rescan, what a selection stands for,
moving a study's files into a new folder with its series folders kept, the
catalog staying in the main folder, and removal taking the emptied folders
with it.

Unit tests in `crates/rds-domain/src/archive.rs` and `crates/rds-domain/src/archive/catalog.rs` cover the parts
with no round trip: the folder-name sanitizer, a missing folder reading as an
empty PACS, the cohort rule on a set of folders, the catalog reading back what
was written (and its lock going with the write), only this program's old
sidecars being removed, and `remove` refusing any path outside the storage
folders. `src/app/pacs_tree.rs` tests the tree's folding, filtering and
selection counting.
