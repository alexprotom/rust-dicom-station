# The patient archive

*Tools ▶ 🏥 PACS — patient archive…* opens the application's own store of DICOM
studies: every patient ever filed into it, listed in one window.

It is a PACS in the sense that matters at a workstation — a persistent place
where patients live between sessions — not in the sense of DICOM networking:
no listener, no association negotiation, no C-FIND, C-MOVE or C-STORE. The
archive is a folder on disk that this application owns.

## The three gestures

| | |
|---|---|
| **📥 Import folder…** | Copy every DICOM file under a folder into the archive, filed by patient and study |
| **📩 Load into dataset A / B** | Read the selected patient or study into a viewer dataset |
| **📤 Send dataset A / B** | Write that dataset's structure sets and segmentation series back into the archive, attached to the study they belong to |

The patient list shows one row per patient — `Doe John (P0001)   3 study(ies)
· 642 file(s)` — expanding into its studies, newest first, each as
`20260827 — Planning · CT, RTSTRUCT, SEG · 214 files`. A patient row means
the whole patient, a study row that study; right-click removes either from
the archive.

The archive folder is shown at the top and can be pointed anywhere. It
defaults to `archive` inside the platform data directory — the same place the
downloaded model weights live (`%LOCALAPPDATA%\RustDICOMStation` on Windows,
`~/.local/share/RustDICOMStation` on Linux, `~/Library/Application
Support/RustDICOMStation` on macOS) — and is remembered in
`viewer_settings.txt` under `archive_dir`.

## Layout

```text
<root>/
  <patient id>/           PATIENT.txt   name, id
    <study uid>/          STUDY.txt     uid, date, description, modalities, files
      <sop instance uid>.dcm
```

Nothing here is a database: the folder names are the DICOM identifiers, the
files keep their own Instance UIDs as names, and the two sidecars are plain
`key = value` text in the same shape as the settings file. Anyone can browse
the archive with a file manager, copy a study folder onto a stick, or hand it
to another DICOM application, and nothing is lost.

The sidecars exist so that listing stays instant however large the archive
grows — reading headers out of ten thousand files is not. They are a cache,
never the truth: a study folder that arrived without one — copied in by hand —
has it rebuilt from the headers the first time it is listed, and deleting
every sidecar rebuilds the whole archive.

Only the patient folder name comes from free text and so is sanitized:
anything outside ASCII letters, digits, `.`, `-` and `_` becomes `_`, capped
at 96 characters. That can map two identifiers onto one folder, so the folder
name is never the authority; `PATIENT.txt` is.

## Filing

Import copies; it never moves. Each file is opened as far as the pixel data
(not into it), and its Patient ID, Study Instance UID and SOP Instance UID
decide where it lands. A file already stored under the same SOP Instance UID
in the same study is left alone, so importing the same folder twice is a no-op
and re-importing a folder that has grown files only the new ones. Anything
that will not open as DICOM is counted as skipped and reported, not treated as
an error. Sidecars are rewritten once per touched study at the end — the
counts and the modality list are only right once everything is in.

## Taking a patient into the viewer

**A study folder in the archive is a DICOM folder**, so
*Load into dataset A / B* runs it through the same `loader::load_directory`
as *File ▶ Add DICOM folder*, with the same classification, the same
patient ▶ study ▶ series tree and the same merging into whatever the dataset
already holds. Selecting the patient row loads all of their studies at once.

## Sending changes back

*Send dataset A / B* writes back **derived objects only** — the structure sets
and the segmentation series. The images are already in the archive; re-sending
them would duplicate hundreds of megabytes under new Instance UIDs.

Each written object gets

* a **fresh SOP Instance UID and Series Instance UID** — a new object, not a
  replacement, which keeps the archive append-only and a mistaken upload
  harmless;
* the **original Study Instance UID**, from the object itself where it says so
  and from the series it references otherwise, filing it under the patient and
  study it belongs to;
* the **original Frame of Reference UID**, so the contours and masks still sit
  on the images they were drawn on;
* a **reference to the image series** it was drawn on.

The objects are written to a scratch folder (removed afterwards) and imported
through the ordinary import path, so a half-failed upload leaves the archive
untouched. Every send creates new instances, so sending twice leaves two
structure sets rather than overwriting one; the unwanted one can be removed
through the data tree, or the study's older objects from the archive.

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
([example-data.md](example-data.md)): generate, file, list from the sidecars
alone, load the archived folder back through the ordinary loader, draw a
segmentation, send the derived objects back, and assert they joined the same
patient and study — no second patient or study, the Study Instance UID
unchanged, the file count grown by exactly the objects written, `SEG` now
among the study's modalities, the segmentation present on reload. Removing the
patient empties the archive.

Unit tests in `src/archive.rs` cover the parts with no round trip: the
folder-name sanitizer against what acquiring systems actually write, a missing
root reading as an empty archive, a study reading back from its sidecar, and
`remove` refusing any path outside the archive root.
