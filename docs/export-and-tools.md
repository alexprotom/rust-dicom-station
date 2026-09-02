# DICOM export, model manager, anonymizer and test-data generator

The *Tools* menu also holds the three segmentation engines - see
[auto-segmentation.md](auto-segmentation.md), [segvol.md](segvol.md) and
[medsam2.md](medsam2.md) - plus structure [propagation](propagation.md),
[DRR generation](drr.md) and the local [patient archive](pacs.md), which have
their own documents. Export writes a folder; the archive writes into the
application's own store, with the same DICOM writer underneath - see
[pacs.md](pacs.md) for when to reach for which.

## DICOM export

*File ▶ 💾 Export DICOM* opens one window for everything that is loaded. What
goes out is chosen inside it, not by which menu entry was clicked: the tree
lists **dataset ▸ patient ▸ study ▸ series and RT objects** for both A and B
at once, every row has a tick box, and a tick on a study or a patient takes
everything under it. One run can therefore write two patients, or three
series out of forty, or the structure sets alone.

```
[x] Dataset A
   [x] 👤 STAR_Rambam_2  (STAR_Rambam_2)         2 study(ies)
      [x] 📁 20250728  CCT                        1 series, 1 object(s)
           StudyInstanceUID  [1.2.840.113619.…]  ↺ ⟳
      [x] 📈 4DCT (10 phases)                     10 series, 10 phase(s)
           [x] CT 4DCT 0%   [0%]                  318 file(s)
                SeriesInstanceUID   [1.3.12.2.1107.…]  ↺ ⟳
                FrameOfReferenceUID [1.3.12.2.1107.…]  ↺ ⟳
      [x] ▣ CCT RTSTRUCT   12 ROI(s)     (•) RTSTRUCT ( ) SEG
```

### Identifiers

Every identifier is on screen and every one of them is editable - an export
whose UIDs you cannot read is one you cannot file. Study, series, frame of
reference and SOP instance UIDs each get a row with `↺` (back to what the
data says) and `⟳` (a newly generated one).

**Identifiers ▸ keep the original UIDs** (the default) writes the UIDs the
data already has - study, series, frame of reference, and the SOP instance of
every slice and every RT object, each back in the series it came from: the
export *is* the same study, so re-importing it where it came from updates that
study instead of duplicating it, and references from objects outside the
export still resolve. **Generate new UIDs** re-fills every row with a fresh
`2.25.…` UID, for the edited copy that has to live beside its source. Either
way the cross-references are rewritten to match, so the export is internally
consistent - and single rows can still be overridden by hand.

One exception is made on your behalf. An object whose format you *converted*
is written under a different SOP class, so it is a new instance and not the
one that was read; its SOP Instance UID switches to the generated one as soon
as you move the radio, and switches back if you move it back. Two objects of
different SOP classes sharing one instance UID is the one thing an archive
cannot forgive.

A rendered image series is the other place slice-level identity cannot be
kept: the reconstructed volume no longer knows which file each slice came
from, so its slices get new SOP Instance UIDs. Copied series - the default -
keep theirs.

### What keeps the objects together

An RT Structure Set that names nothing but a frame of reference is what
"losing the link to the CT" looks like; a planning system follows
*ReferencedFrameOfReference ▶ RTReferencedStudy ▶ RTReferencedSeries* and the
*ContourImage* of each contour before it will draw contours on a scan. Every
export now writes that chain in full: the study, the image series, every
slice of it, and per contour the image it lies on. So do the other links -
SEG ▸ image series and its frames, RTPLAN ▸ RTSTRUCT, RTDOSE ▸ RTPLAN, and one
frame of reference across all of them.

If a structure set goes out without its images, that is not silently
degraded: the object is still written and the run reports what it could not
link.

### Structures as RTSTRUCT or SEG

Each set of structures carries its own radio. Contours are rasterised onto
the image lattice for SEG, masks are contoured (marching squares, exactly as
the viewer draws them) for RTSTRUCT, so anything can go out as either;
*Structures ▸ all RTSTRUCT / all SEG* sets the whole run at once. An ROI with
no contour inside the image volume is reported rather than written empty.

### 4D acquisitions

A recognised 4D group is one node of the tree and one tick takes every phase.
Its phases go out into one study, keep their own series identity, their
descriptions and their Temporal Position Identifier, so the export regroups as
the same acquisition when it is read back. Selecting only part of a group is
allowed but reported - half a 4DCT is not a 4DCT.

### Images are copied, not re-encoded

A series that still has its source files is copied file by file with only the
identifying attributes patched. Private tags, acquisition parameters, the
padding value, the transfer syntax and every bit of pixel data pass through
untouched, which is what keeps 4D acquisitions, dual-energy series and vendor
extensions intact. Only a series the application invented - a simulation, a
resampled volume - is rendered from its voxels; *Rewrite images from the
voxels* forces that for everything, and is needed only when the voxels
themselves were changed.

On a copied series the **Common tags** table applies only the rows you
actually change, so the scanner's own equipment tags are not overwritten by
this application's defaults.

### The rest of the window

*Folders* chooses `patient / study / series` subfolders (the default), one
folder per study, or everything flat. The **Common tags** section holds the
attributes the tree does not own - birth date, sex, accession number,
referring physician, institution, station, manufacturer - each with the same
`↺` and a tick box that leaves the tag out of the files altogether. Export
runs on a background thread with progress, and finishes with the file count
and any notes.

A single segmentation series can still be written on its own: right-click it
in the data tree and choose *💾 Export as DICOM SEG…*. To write only what was
*drawn* - the structure sets and segmentation series, with the images left
where they are - use *📤 Send dataset* in the [patient archive](pacs.md)
window instead. The exports round-trip through this viewer and pydicom; they
are QA/research objects, not guaranteed-complete clinical IODs.

## Model manager

Each segmentation engine downloads its weights on first use; *Tools ▶ 📦
Downloaded models…* is the one inventory of what is on this machine, what it
costs in disk, and where to re-fetch a checkpoint after a bad download.

Every model of every engine gets a row: its state (ready / partly downloaded /
missing), its size on disk or to fetch, and the buttons that act on it.

| | |
|---|---|
| ⬇ | download and convert this model |
| ⟳ | remove it and fetch it again - the published files carry no version, so an update *is* a fresh download |
| ♻ | delete the source checkpoint the converted cache was made from; the model keeps running |
| 🗑 | delete every file of this model |

and, over the whole inventory, **⬇ Download all missing**, **⟳ Update all**
and **♻ Free …**, which reports what the redundant source checkpoints cost
before you drop them. The model folder is editable here (the setting the
three tool windows show); the header counts ready models and total size.

Two details worth knowing:

* Preparing a model runs the **engine's own first-use path** - the same
  download, checkpoint conversion and cache - so a model fetched here is bit
  for bit the one a run would have fetched.
* Removal deletes only the file names the inventory lists, never a whole
  folder, so anything else kept in the model folder survives; the model's own
  sub-folder is removed afterwards if it came out empty.

Each engine's weight licence is stated above its rows: TotalSegmentator's
are Apache-2.0, SegVol's carry no licence declaration, and MedSAM2's are
CC-BY-SA-4.0 with a research-only model card. None is redistributed with the
program.

## DICOM anonymizer

*Tools ▶ 🔏 Anonymize DICOM folder…* is an interactive anonymizer for folders
on disk (independent of what is loaded):

1. **Scan** (recursive, background thread): the dialog lists every
   identifying tag present - patient identity, birth date/sex, dates and
   times, accession number, physicians, institution, station, device - with
   its current value(s) and a proposed replacement: a deterministic
   `anon_xxxxxx` patient alias derived from the original PatientID, the fixed
   date `20000101` / time `000000`, or a cleared value. Every proposal is
   editable, each row can be unchecked, and Study/Series descriptions are
   offered opt-in.
2. **Apply** (parallel, background thread) with three switches:
   * **regenerate UIDs** - every non-standard UID (study, series, SOP
     instances, frame of reference, and every reference to them inside
     sequences) is replaced with a fresh `2.25.` UID, consistently across all
     files, so the reference chains stay intact;
   * **remove private elements** - drops all odd-group vendor tags, including
     inside sequences;
   * **mark as de-identified** - writes `PatientIdentityRemoved=YES` and
     `DeidentificationMethod`.

Output goes to a separate folder (relative paths kept; default `<input>_anon`)
or in place; files are written via a temp file so an interrupted run never
corrupts an original, and pixel data is copied through byte-identical.
`tests/anonymize.rs` verifies the pipeline end-to-end: identity gone,
reference chains resolve, volume unchanged. Known limitation: value
replacements apply to top-level elements; identifying strings nested inside
sequences (e.g. operator names in beam session sequences) are not yet
rewritten (UID remapping and private-tag removal do recurse).

## Synthetic test-data generator

*File ▶ 📐 Generate test data…* (also offered on the empty start screen) writes
a complete, analytically known RT study into `test_data/` next to the
executable and loads it straight away - no Python, no external tooling:

* CT - 40 slices, 96 × 96, 2 mm isotropic; water cylinder (r = 70 mm),
  spherical target (r = 25 mm, HU 100), cord (r = 8 mm, HU 40);
* RTSTRUCT - BODY (EXTERNAL), TARGET (PTV), CORD (ORGAN);
* RTDOSE - 3D Gaussian, 60 Gy at isocenter, σ = 20 mm, 32-bit, 4 mm grid;
* RTPLAN - ion (proton) plan, 2 beams, 60 Gy / 30 fx;
* optionally DX, RTIMAGE (DRR), REG and an RT Ion Beams Treatment Record.

The dialog exposes the dose peak, a target Y shift, a whole-phantom X/Y shift,
the plan label and the REG translation, so a deliberately misaligned second
study for comparison-mode and registration testing is one more generation into
another folder:

```
# rigid scenario: whole phantom translated (12, −9) mm
cargo run --release -- test_data test_data_shifted
```

a rigid run in the *Image registration* module should then recover the
(12, −9, 0) mm shift to within a fraction of a millimeter. The phantom is
analytically known, which is what the integration tests assert against - see
[architecture.md](architecture.md#testing).
