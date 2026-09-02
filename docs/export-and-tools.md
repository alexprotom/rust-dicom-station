# DICOM export, model manager, anonymizer and test-data generator

The *Tools* menu also holds the three segmentation engines - see
[auto-segmentation.md](auto-segmentation.md), [segvol.md](segvol.md) and
[medsam2.md](medsam2.md) - plus structure [propagation](propagation.md),
[DRR generation](drr.md) and the local [patient archive](pacs.md), which have
their own documents. Export writes a folder; the archive writes into the
application's own store, with the same DICOM writer underneath - see
[pacs.md](pacs.md) for when to reach for which.

## DICOM export

Any loaded dataset - original, simulated or with converted segmentations -
can be exported as DICOM files via *File ▶ 💾 Export dataset A/B as DICOM…*:
one CT Image Storage file per slice plus RTSTRUCT, one binary Segmentation
(SEG) object per segmentation series, RTDOSE (16-bit with `DoseGridScaling`)
and an RTPLAN skeleton (photon or ion), written with `dicom-rs` in Explicit
VR Little Endian and preserving the RTSTRUCT ▶ series, SEG ▶ series and
RTDOSE ▶ RTPLAN ▶ RTSTRUCT reference chains. A SEG claims the exported image
slices as its source only when it sits on their lattice; the new objects get
fresh `2.25.…` UIDs. Export runs on a background thread with progress.

The dialog shows, in the anonymizer's shape, an output folder and every
patient / study / equipment attribute written into all exported files -

| Tag | Default |
|---|---|
| PatientName, PatientID | from the loaded study |
| PatientBirthDate, PatientSex | empty, `O` |
| StudyID, StudyDescription, StudyDate, StudyTime | `1`, study's own, study's own date (today if absent), now |
| AccessionNumber, ReferringPhysicianName | empty |
| SeriesDescription | from the active series (written on the image series only) |
| InstitutionName, StationName | empty |
| Manufacturer, ManufacturerModelName | `rust-dicom-station`, `DICOM export` |

Every value is editable, `↺` restores the study's own value (`↺ all` the whole
table), and unchecking a row leaves that tag out of the files. *StudyDate* /
*StudyTime* also stamp the RTSTRUCT and RTPLAN date/time. **Keep the source
Frame of Reference UID** (on by default) keeps the export spatially linked to
its source; switched off, a fresh frame of reference is generated.

A single segmentation series can be written on its own: right-click it in
the data tree and choose *💾 Export as DICOM SEG…*. To write only what was
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
