# DICOM export, model manager, anonymizer and test-data generator

Four tools round out the workflow: writing datasets back out as DICOM,
managing the downloaded network weights, anonymizing folders on disk, and
generating a fully synthetic RT study for testing. (The *Tools* menu also
holds the three segmentation engines — see
[auto-segmentation.md](auto-segmentation.md), [segvol.md](segvol.md) and
[medsam2.md](medsam2.md) — plus structure
[propagation](propagation.md) and [DRR generation](drr.md), which have their
own documents.)

## DICOM export

Any loaded dataset — original, simulated or with converted segmentations —
can be exported as a set of DICOM files via *File ▶ 💾 Export dataset A/B as
DICOM…*: one CT Image Storage file per slice plus RTSTRUCT, one binary
Segmentation (SEG) object per segmentation series, RTDOSE (16-bit with
`DoseGridScaling`) and an RTPLAN skeleton (photon or ion), written with
`dicom-rs` in Explicit VR Little Endian and preserving the RTSTRUCT ▶ series,
SEG ▶ series and RTDOSE ▶ RTPLAN ▶ RTSTRUCT reference chains. A SEG only
claims the exported image slices as its source when it sits on their
lattice. Fresh `2.25.…` UIDs are generated
for the new objects. Export runs on a background thread with progress.

The dialog first shows what will be written, in the same shape as the
anonymizer: an output folder, then every patient / study / equipment
attribute that goes into all exported files —

| Tag | Default |
|---|---|
| PatientName, PatientID | from the loaded study |
| PatientBirthDate, PatientSex | empty, `O` |
| StudyID, StudyDescription, StudyDate, StudyTime | `1`, study's own, study's own date (today if absent), now |
| AccessionNumber, ReferringPhysicianName | empty |
| SeriesDescription | from the active series (written on the image series only) |
| InstitutionName, StationName | empty |
| Manufacturer, ManufacturerModelName | `rust-dicom-station`, `DICOM export` |

Every value is editable, `↺` restores the study's own value (`↺ all` restores
the whole table), and unchecking a row leaves that tag out of the files
entirely. *StudyDate* / *StudyTime* also stamp the RTSTRUCT and RTPLAN
date/time. **Keep the source Frame of Reference UID** (on by default) keeps
the export spatially linked to its source, so the two load as a comparable
pair; switching it off generates a fresh frame of reference.

A single segmentation series can also be written on its own, without
exporting the dataset around it: right-click the series in the data tree and
choose *💾 Export as DICOM SEG…*.

The exports round-trip through this viewer and pydicom; they are
QA/research objects, not guaranteed-complete clinical IODs.

## Model manager

The three segmentation engines each download their own weights on first use,
which is convenient right up to the moment somebody asks what is actually on
this machine, how much disk it costs, or wants a checkpoint re-fetched after
a bad download. *Tools ▶ 📦 Downloaded models…* answers all three from one
inventory.

Every model of every engine gets a row: its state (ready / partly downloaded
/ missing), what it occupies on disk or would cost to fetch, and the buttons
that act on it.

| | |
|---|---|
| ⬇ | download and convert this model |
| ⟳ | remove it and fetch it again — the published files carry no version, so an update *is* a fresh download |
| 🧹 | delete the source checkpoint the converted cache was made from; the model keeps running |
| 🗑 | delete every file of this model |

and, over the whole inventory, **⬇ Download all missing**, **⟳ Update all**
and **🧹 Free …**, which reports how much the redundant source checkpoints
are costing before you drop them. The model folder itself is editable here
(it is the same setting the three tool windows show) and the header reports
how many models are ready and how much the lot occupies.

Two details worth knowing:

* Preparing a model runs the **engine's own first-use path** — the same
  download, the same native checkpoint conversion, the same cache. A model
  fetched here is bit for bit the one a run would have fetched; there is no
  second download route to keep in step.
* Removal only ever deletes the file names the inventory lists, never a
  whole folder, so a model folder you also keep something else in survives
  intact. The model's own sub-folder is removed afterwards if it came out
  empty.

The licence of each engine's weights is stated above its rows, because it
differs: TotalSegmentator's are Apache-2.0, SegVol's carry no licence
declaration at all, and MedSAM2's are CC-BY-SA-4.0 with a research-only
model card. None of them is ever redistributed with the program.

## DICOM anonymizer

*Tools ▶ 🔏 Anonymize DICOM folder…* is an interactive anonymizer for
folders on disk (independent of what is loaded):

1. **Scan** (recursive, background thread): the dialog lists every
   identifying tag actually present — patient identity, birth date/sex,
   dates and times, accession number, physicians, institution, station,
   device — with its current value(s) across the files and a proposed
   replacement: a deterministic `anon_xxxxxx` patient alias derived from
   the original PatientID, the fixed date `20000101` / time `000000`, or a
   cleared value. Every proposal is editable, each row can be unchecked,
   and Study/Series descriptions are offered opt-in.
2. **Apply** (parallel, background thread) with three switches:
   * **regenerate UIDs** — every non-standard UID (study, series, SOP
     instances, frame of reference, and every reference to them inside
     sequences) is replaced with a fresh `2.25.` UID, consistently across
     all files, so the reference chains stay intact;
   * **remove private elements** — drops all odd-group vendor tags,
     including inside sequences;
   * **mark as de-identified** — writes `PatientIdentityRemoved=YES` and
     `DeidentificationMethod`.

Output goes to a separate folder (files keep their relative paths; default
`<input>_anon`) or in place; files are written via a temp file so an
interrupted run never corrupts an original, and pixel data is copied
through byte-identical. `tests/anonymize.rs` verifies the pipeline
end-to-end: identity gone, reference chains resolve, volume unchanged.

Known limitation: value replacements are applied to top-level elements;
identifying strings nested inside sequences (e.g. operator names in beam
session sequences) are not yet rewritten (UID remapping and private-tag
removal do recurse).

## Synthetic test-data generator

*File ▶ 🧪 Generate test data…* (also offered on the empty start screen)
writes a complete, analytically known RT study into `test_data/` next to
the executable and loads it straight away — no Python, no external
tooling:

* CT — 40 slices, 96 × 96, 2 mm isotropic; water cylinder (r = 70 mm),
  spherical target (r = 25 mm, HU 100), cord (r = 8 mm, HU 40);
* RTSTRUCT — BODY (EXTERNAL), TARGET (PTV), CORD (ORGAN);
* RTDOSE — 3D Gaussian, 60 Gy at isocenter, σ = 20 mm, 32-bit, 4 mm grid;
* RTPLAN — ion (proton) plan, 2 beams, 60 Gy / 30 fx;
* optionally DX, RTIMAGE (DRR), REG and an RT Ion Beams Treatment Record.

The dialog exposes the dose peak, a target Y shift, a whole-phantom X/Y
shift, the plan label and the REG translation — so a deliberately
misaligned second study for comparison-mode and registration testing is a
matter of generating once more into another folder:

```
# rigid scenario: whole phantom translated (12, −9) mm
cargo run --release -- test_data test_data_shifted
```

*Registration ▶ Rigid* should then recover the (12, −9, 0) mm shift to
within a fraction of a millimeter. The whole phantom is analytically
known, which is what the integration tests assert against — see
[architecture.md](architecture.md#testing).
