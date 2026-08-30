# The patient archive

*Tools ▶ 🏥 PACS — patient archive…* opens the application's own store of
DICOM studies: every patient ever filed into it, listed in one window, ready
to be taken into a dataset and given back the structures and segmentations
drawn on them.

It is a PACS in the sense that matters at a workstation — a persistent place
where patients live between sessions, separate from whatever folder the data
originally came off — and not in the sense of DICOM networking. There is no
listener, no association negotiation, no C-FIND, C-MOVE or C-STORE, and
nothing on the network at all: the archive is a folder on disk that this
application owns. Everything below follows from that choice.

## The three gestures

| | |
|---|---|
| **📥 Import folder…** | Copy every DICOM file under a folder into the archive, filed by patient and study |
| **📩 Load into dataset A / B** | Read the selected patient or study into a viewer dataset |
| **📤 Send dataset A / B** | Write that dataset's structure sets and segmentation series back into the archive, attached to the study they belong to |

The patient list shows one row per patient — `Doe John (P0001)   3 study(ies)
· 642 file(s)` — that expands into its studies, newest first, each described
as `20260827 — Planning · CT, RTSTRUCT, SEG · 214 files`. Selecting a patient
row means the whole patient; selecting a study row means that study. The
right-click menu on either row removes it from the archive.

The archive folder is shown at the top and can be pointed anywhere — an
external drive, a network share the operating system has already mounted. It
defaults to `archive` inside the platform data directory — the same place the
downloaded model weights live (`%LOCALAPPDATA%\RustDICOMStation` on Windows,
`~/.local/share/RustDICOMStation` on Linux, `~/Library/Application
Support/RustDICOMStation` on macOS) — and the choice is remembered in
`viewer_settings.txt` under `archive_dir`.

## Layout

```text
<root>/
  <patient id>/           PATIENT.txt   name, id
    <study uid>/          STUDY.txt     uid, date, description, modalities, files
      <sop instance uid>.dcm
```

Nothing here is a database. The folder names are the DICOM identifiers, the
files keep their own Instance UIDs as names, and the two sidecars are plain
`key = value` text in the same shape as the settings file. Anyone can look at
the archive with a file manager, copy a study folder onto a stick, or hand it
to another DICOM application, and nothing is lost — which is the property a
proprietary index would have taken away.

The sidecars exist for one reason: listing the archive must stay instant
however large it grows, and reading headers out of ten thousand files is not
instant. They are a cache and never the truth. A study folder that arrived
without one — copied in by hand — has it rebuilt from the headers the first
time it is listed, and is fast from then on; delete every sidecar and the
archive rebuilds itself.

Only the patient folder name comes from free text and so is sanitized:
anything outside ASCII letters, digits, `.`, `-` and `_` becomes `_`, capped
at 96 characters. That can map two identifiers onto one folder, which merges
two patients who already shared an identifier — the correct reading — and is
the reason the folder name is never the authority. `PATIENT.txt` is.

## Filing

Import copies; it never moves, so importing does not take the source folder
apart. Each file is opened as far as the pixel data (not into it), and its
Patient ID, Study Instance UID and SOP Instance UID decide where it lands. A
file already stored under the same SOP Instance UID in the same study is a
duplicate and is left alone, so importing the same folder twice is a no-op
rather than a second copy, and re-importing a folder that has grown files
only the new ones. Anything that will not open as DICOM is counted as skipped
and reported, not treated as an error.

Sidecars are rewritten once per touched study at the end rather than per
file — the counts and the modality list are only right once everything is in.

## Taking a patient into the viewer

There is no special path: **a study folder in the archive is a DICOM
folder**, so *Load into dataset A / B* runs it through the same
`loader::load_directory` as *File ▶ Add DICOM folder*, with the same
classification, the same patient ▶ study ▶ series tree and the same merging
into whatever the dataset already holds. Selecting the patient row loads all
of their studies at once, which is how a planning CT and a later verification
CT end up in one dataset.

This is deliberately unremarkable. Anything the archive could do that the
ordinary loader could not would be a second import path to keep correct.

## Sending changes back

*Send dataset A / B* writes back **derived objects only**: the structure sets
and the segmentation series. The images are already in the archive; re-sending
them would duplicate hundreds of megabytes and, worse, create a second copy of
the same anatomy under new Instance UIDs.

Each written object gets

* a **fresh SOP Instance UID and Series Instance UID** — it is a new object,
  not a claim to replace one, which is what keeps the archive append-only and
  a mistaken upload harmless;
* the **original Study Instance UID**, taken from the object itself where it
  says so and from the series it references otherwise — this is what files it
  under the patient and study it belongs to rather than beside them;
* the **original Frame of Reference UID**, so the contours and masks still sit
  on the images they were drawn on;
* a **reference to the image series** it was drawn on, as a cross-reference to
  data already in the archive.

The objects are written to a scratch folder and then imported through the
ordinary import path, so the filing rule lives in exactly one place and an
upload that fails half way leaves the archive untouched rather than
half-written. The scratch folder is removed afterwards.

Because every send creates new instances, sending the same dataset twice
leaves two structure sets in the study rather than overwriting one. That is
the honest behaviour for an archive — a record of what was drawn, when — and
the unwanted one can be removed from the study through the data tree, or the
study's older objects removed from the archive.

## What it is not

* **Not a DICOM network node.** No SCP, no SCU, no AE titles. Files move by
  the file system.
* **Not multi-user.** One application owns the folder. Two instances pointed
  at the same archive will not corrupt it — files are written under unique
  UIDs — but their listings can go stale until rescanned.
* **Not an anonymizer.** What goes in is what comes out. *Tools ▶ Anonymize*
  ([export-and-tools.md](export-and-tools.md)) is the pass to run before
  filing anything that must leave the department.

## Verification

`tests/archive.rs` runs the whole round trip on the synthetic phantom study
([example-data.md](example-data.md)): generate it, file it, list it from the
sidecars alone, load the archived folder back through the ordinary loader,
draw a segmentation on it, send the derived objects back, and assert that they
joined the same patient and the same study — no second patient, no second
study, the Study Instance UID unchanged, the file count grown by exactly the
number of objects written, `SEG` now among the study's modalities, and the
segmentation present again when the study is reloaded. Removing the patient
empties the archive.

The unit tests in `src/archive.rs` cover the parts that have no round trip:
the folder-name sanitizer against what acquiring systems actually write, a
missing root reading as an empty archive rather than an error, a study reading
back from its sidecar, and `remove` refusing any path outside the archive root.
