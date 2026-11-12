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
| **📤 Send workspace A / B** | Opens [Save to the PACS](#saving-to-the-pacs) on that workspace: its structure sets and segmentation series, the new ones ticked, written beside the study they belong to |
| **⟲ Rescan** | Look through the storage folders again; only files not seen before are read |
| **🔍** | Show only the rows whose name (or UID, or folder) holds the text, with what leads to them |
| **✖ Reset filters** | Clear the search text and untick everything in the list beside 🔍: every study shows again (also under the scope in the [analytics](#analytics) while anything filters) |
| **📩 Load into workspace A / B** | Read the selected rows into a viewer workspace |
| **⚙ Settings ▶ Storage folders** | The folders the PACS reads (below) |
| **⚙ Settings ▶ ⟲ Rescan the folders on opening** | On (a fresh installation): opening the window walks the folders, as ⟲ Rescan does. Off: the window lists the PACS at once as the catalog last saw it, and the folders are read only on ⟲ Rescan, or when there is no catalog yet. Remembered as `pacs_scan_on_open` in `viewer_settings.txt` |
| **⚙ Settings ▶ 📊 Recalculate the analytics on every click** | On (a fresh installation): the [analytics](#analytics) follow the selection. Off: they are worked out on their ⟳ only, and say when they are stale. `pacs_stats_auto` |
| **▶ / ◀ on the right edge** | Show or hide the [analytics](#analytics) panel |
| **⚙ Settings ▶ 📤 Export tags… / 📥 Import tags…** | What was decided about the PACS - names, cohorts, 4D decisions - to and from a JSON file, to share between stations ([Tags](#tags)) |
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

A study holding a 4DCT says so before anything is loaded ([4D sets](#4d-sets)):
its row carries **🎞** and the number of scans (`🎞2`; `🎞?` while a set is
only probable, `🎞 phases not in PACS` for a 4D scout or an ungated series
whose phases never arrived), its tooltip lists each scan (date, scanner,
phases, reconstructions, evidence), and each set is a row of its own under
the study holding its phases in order (`30%`, `~30%` when the phase was
inferred from the series' place), its reconstructions (`untagged`, `AVG`,
`MIP`), its respiratory trace and the structure sets drawn on a phase.
Selecting that row and *Load* loads the set alone. The list beside 🔍
filters the studies by what they hold: each scan offers the kinds it found,
with how many hold each - **4D** and **probable 4D**, then the words in the
brackets of the patients' rows (`P01-095 (Adult, Thorax)` offers `Adult` and
`Thorax`, counted in patients; words that only repeat the patient's name, ID
or code are left out, and `Male` / `male` are one word), grouped by what
they say - **age group** (Adult, Pediatric, Child, ...), **sex** (Male,
Female), **site** (every word that is none of the others: Thorax, HN,
Pancreas, ...) and **subject** (Phantom, Volunteer, ...) - then every
modality of their series (`CT`, `MR`, `DX`, `RTIMAGE`, `RTSTRUCT`, `RTDOSE`,
`SEG`, `REG`, ...). Several can be ticked in one visit, and the studies shown
hold **all** of them (`Pediatric + CT`) or **any** (`MR or DX`); nothing
ticked shows all studies, and a kind the next scan no longer finds is
unticked. The search finds `4DCT` too. A probable set is drawn muted, and its menu
**✔ Confirms** it or says **✖ Not a 4DCT** (the set is dissolved and stays
so); a set decided on offers **Let the headers decide again**.

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

## Analytics

The panel on the right of the window (its ◀ / ▶ strip folds it away, as
the viewer's side panels do; it can be dragged wider) counts what the rows
in view hold, in charts one above the other:

- **Patients' words**, above: the words in the brackets of the patients'
  rows - those the filter offers - counted in patients, one chart per kind
  of word: **age group**, **sex**, **site**, **subject** (Adult and
  Pancreas are no shares of one whole). Each chart has the patients whose
  row says nothing of its kind as *not given*, and a kind no patient of the
  scope has is left out. A patient may name two sites, so the site bars can
  add up to more than the patients.
- **Modalities**, below: the modalities of the series, counted in studies,
  or in series once the scope is one study.

**📊** draws bars, **◐** rings (the slices past the tenth are one *other*);
the choice is remembered (`pacs_stats_rings`). The scope is the
selection: nothing selected is the whole PACS as the tree shows it (the
filter applies, and the heading says so), a cohort row its patients, a
patient, study or series row that one; several rows are their union, a
patient listed in two cohorts counting once. The heading names the scope
and its patients, studies, series and files. Hovering a bar, a slice or a
legend line gives its count and share (and the series and files of a
modality); clicking it ticks that word or modality in the filter beside 🔍,
and clicking it again unticks it - the ticked ones are outlined - and
**✖ Reset filters** under the scope shows every study again. The
figures follow every click unless *Settings* says otherwise; then **⟳**
works them out for the rows selected at that moment.

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

**Treatment days.** A study that holds treatment records is a course, and
its series are grouped under a row per day a record was made on
(`2024-09-17 · 4 of 16`, its tooltip counting the day's series by modality),
in the order they were made: the daily CT, the radiographs, the
registrations, the records, the dose reports. The series of any other day
(the planning CT, the plan) stay directly under the study. A day's row is
selected, loaded, pulled or removed as its series are. It is the series'
date alone - the catalog keeps no references; the viewer's data tree puts a
day together by beam once it is loaded. A study without records whose
series were made on more than one day - one Study Instance UID over weeks of
sessions, each its own *Procedure #N* - has a row per **session** day the
same way, all its dated series under them. A server of an earlier version
sends no series dates, and its studies are listed as before.

## Names

*Rename* gives a cohort, a patient, a study or a series a name of your
choosing - `Case 24` for `GC (X24)`, `Planning CT` for
`2021-03-02 - CT Abdomen`. The name is kept in the catalog; **nothing is
written into a DICOM file**, and the tooltip of a renamed row shows what the
files say. *The DICOM name* in the rename bar goes back to it. A cohort
named after a folder can be renamed without renaming the folder.

## Tags

The names given, the cohorts and who is in which, and the 4D groups made,
confirmed or dissolved by hand live only in the catalog. *Settings ▶ Export
tags…* writes them into one JSON file (`rds-pacs-tags-<date>.json`,
format `rds-pacs-tags`), naming every object by what travels with its
files - a study or series by its UID, a patient by their Patient ID, a
cohort by its name - and no path. *Settings ▶ Import tags…* merges such a
file from another station holding copies of the same studies: a name goes
to the patient, study or series this station holds (a name of its own the
file does not mention stays), a cohort is found by name or made and its
patients put into it beside the cohorts they are in, a 4D decision is kept
when one of its series is here. What this station does not hold is
counted in the status line and left; importing a file twice is importing
it once.

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
* A series keeps when it was made (`datetime`: its SeriesDate/Time, else the
  acquisition's, else the content's, the earliest of its files). Catalog
  version 2 added it; a catalog of version 1 has every header read again,
  once, by the next scan, and keeps its names, cohorts and memberships.
  Version 3 added what the [4D sets](#4d-sets) are found from.
* A file read again (it changed) says where it belongs: a series whose files
  now name another study moves to it, a study whose files now name another
  patient moves to them, and a study left with no series goes - so a study
  merged or pseudonymized in place is where its files say after one scan.
  A series that lost a file, or one of whose files changed, has all its
  files read again, so what it and its study belong to is never carried
  over from the catalog of before (copies once filed under it, gone since,
  would leave their patient on it). A study's date, time and description
  are its earliest session's (the smallest StudyDate of its files: a course
  study spans weeks); the names given by hand stay.
* A study whose files all lie in one patient's folder (`cohort/patient/...`
  or a patient's folder directly in a storage folder) belongs to the
  patient that folder's code names (its first word, by Patient ID), even
  when its files still carry another code (an export's own pseudonym): the
  folder is the person's sorting and wins. A study spread over folders, or
  in a folder no patient's ID names, stays where its files put it.
* A scan never looks into folders where imports wait or duplicates are put
  aside: `to_be_sorted`, `*-duplicates`, `*_duplicates`, and the patterns in
  the catalog's `ignore` list (`*` any run of characters, case ignored; the
  MCP tool `catalog_ignore` edits it). Their files are copies of what the
  PACS holds under the source's patient codes, and filed they would relink
  sorted studies to those codes; a folder newly ignored leaves the catalog
  with the next scan, and nothing on disk is touched.
* The catalog keys a patient by Patient ID alone, so one code given in two
  cohorts to two people lists them as one. The MCP tool
  `check_patient_identity` finds such codes (the cohorts' files share no
  birth date and sex) and they are split by recoding one cohort's folder
  with `set_patient_identity`.
* The DICOM files are the truth: delete the catalog and the next scan builds
  it again, from the files alone. Only the names, the cohorts and who is in
  which are lost then, being the catalog's alone.
* Two programs may write it at once - the viewer and a PACS server serving
  the same folders. Each write takes a lock file beside it
  (`.rds-pacs-catalog.lock`), reads the catalog afresh, changes it and puts
  it back in one rename, so a reader never sees half of it and neither
  writer loses the other's change.
* A file that is not DICOM (a plan's screenshot, a detector's headerless
  frame) is remembered as an *other* file, so it is not read again, and is
  attached to the patient whose patient folder holds it (`cohort/patient/...`,
  or a patient's folder directly in a storage folder): listings count a
  patient's attachments. A DICOM file the standard reader refuses for a
  broken private length or a header naming the wrong VR encoding is read
  past it, and filed like any other.
* Earlier versions wrote `STUDY.txt`, `PATIENT.txt` and `INDEX.txt` into
  every folder. A scan removes those, recognising them by name and content,
  so that a person's own file of the same name stays.

## 4D sets

The catalog also says which studies hold a 4DCT, how many scans, which
series are the phases and which the reconstructions, without a study being
loaded: the header rules of the viewer's 4D groups
([motion-4d.md](motion-4d.md#4d-groups-cratesrds-domainsrcfourd)) run over
what it keeps of every series, after every write, and the result is a
study's `fourd` sets and a series' `fourd` role. Catalog version 3 keeps for
that the first file's 4D header fields of each CT (ImageType, ProtocolName,
ImageComments, the acquisition date and time, Rows, the scanner, four
private texts, read as bytes), the slice position of every CT file, and the
series each structure set references; a catalog of version 2 has the files
of its CT and structure-set series read again, once.

* What a person decided wins: a group made, corrected, confirmed or
  dissolved - in the PACS window, by the assistant (`set_4d_group`,
  `confirm_4d`) or in the viewer's data tree for a study of this PACS - is
  kept in `fourd_custom` under the group's id and replaces whatever the
  rules find that shares a series with it, through rescans.
* A structure set drawn on one phase is listed under that phase; a 4D
  scout or ungated series with no set in its block is listed as a study's
  `fourd_orphans`.
* The same series give the same sets: a scan that finds nothing new still
  writes nothing.
* The `4DCT ...` labels a script wrote into catalogs before version 3 are
  compared once with the sets found: a label that agrees is cleared (the
  structure says it now), one that says `4DCT` of a set the rules call
  probable is taken as the person's confirmation, any other is kept; each
  is noted in `.rds-pacs-fourd-labels.jsonl` beside the catalog.

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
so no header is read again. No file is moved to a path over 259 characters,
the limit much DICOM software on Windows still has.

## Filing

Import copies; it never moves. Each file is opened as far as the pixel data
(not into it), and its Patient ID, Study Instance UID and SOP Instance UID
decide where it lands: beside the study's files when the main folder holds
the study already, else in `<main folder>/<patient>/<study uid>/` as
`<sop instance uid>.dcm` ([Save to the PACS](#saving-to-the-pacs) files
beside the study in whichever storage folder holds it). A file whose SOP Instance UID the PACS holds
already is left alone, so importing the same folder twice is a no-op and
re-importing a folder that has grown files only the new ones. Anything that
will not open as DICOM is counted as skipped and reported, not treated as an
error. A filed instance is copied beside its place and renamed into it, so a
reader - the PACS server sending the study on - never takes half a file.

Only the patient folder name comes from free text and so is sanitized:
anything outside ASCII letters, digits, `.`, `-` and `_` becomes `_`, capped
at 96 characters. A file whose place would pass 259 characters is not filed
and is counted as left out.

## Taking a selection into the viewer

*Load into workspace A / B* reads the files of the selection through the same
loader as *File ▶ Add DICOM file(s)*, with the same classification, the same
patient ▶ study ▶ series tree and the same merging into whatever the workspace
already holds. The session remembers the folders the selection covers (a
folder whose every file was loaded) rather than each file.

## Saving to the PACS

What is made on a study of the PACS - a structure set drawn or filled by a
model, a segmentation painted - is saved into the PACS itself, beside the
study, through **Save to the PACS**. Three ways open it:

* **File ▶ 📤 Save to the PACS ▶** a workspace;
* **📤 Save to the PACS** in the menu of a structure set or segmentation row
  of the data tree (that one alone ticked);
* **📤 Send workspace A / B** in this window.

The window lists the workspace's structure sets and segmentation series,
each with what the PACS knows of it:

| State | Meaning | Ticked |
|---|---|---|
| **new** | made here and never saved | yes |
| **not in the PACS** | read from a file the PACS does not hold | yes |
| **in the PACS** | the PACS holds an object with its SOP Instance UID (it was loaded from there, or saved before) | no; ticked, it is saved again as a new object beside the old one - how an edited set is kept, the old version staying |
| **empty** | nothing drawn in it | cannot be |
| **study not in the PACS** | the study it belongs to is not in the PACS: import the study first | cannot be |

Under the list, a line per study says the folder they go into. That is the
folder holding the study's files - the patient's folder in its cohort folder,
in whichever [storage folder](#several-storage-folders) holds the study (the
one holding most of it, should it be split), not only the main one. The
catalog learns the new files in the same step, so the tree lists them at
once, without a rescan. Once saved, a new object takes on the identity of its
copy in the PACS (its SOP and Series Instance UIDs, its file name): it reads
**in the PACS** from then on and is not offered a second time. An object
loaded from the PACS and saved again keeps the identity it was loaded with,
which a plan or a registration may refer to.

Only **derived objects** are written - the structure sets and the
segmentation series. The images are already in the PACS; re-sending them
would duplicate hundreds of megabytes under new Instance UIDs.

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

The objects are written to a scratch folder (removed afterwards) and filed
through the ordinary import path, told to file a study's files beside the
ones the PACS holds wherever they are, so a half-failed save leaves the PACS
untouched. Every save creates new instances, so saving a set twice leaves two
structure sets rather than overwriting one; the unwanted one can be removed
as a series from the PACS.

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
