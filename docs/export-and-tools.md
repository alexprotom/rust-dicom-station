# DICOM export, anonymizer and test-data generator

Three tools round out the workflow: writing datasets back out as DICOM,
anonymizing folders on disk, and generating a fully synthetic RT study for
testing.

## DICOM export

Any loaded dataset — original, simulated or with converted segmentations —
can be exported as a set of DICOM files via *File ▶ 💾 Export dataset A/B as
DICOM…*: one CT Image Storage file per slice plus RTSTRUCT, RTDOSE (16-bit
with `DoseGridScaling`) and an RTPLAN skeleton (photon or ion), written with
`dicom-rs` in Explicit VR Little Endian and preserving the RTSTRUCT ▶ series,
RTDOSE ▶ RTPLAN ▶ RTSTRUCT reference chain. Fresh `2.25.…` UIDs are generated
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

The exports round-trip through this viewer and pydicom; they are
QA/research objects, not guaranteed-complete clinical IODs.

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

The original one-off script that prepared the bundled example data,
`tools/anonymize_dicom.py` (pure standard library), remains in the repo;
the interactive tool is its generalized Rust successor.

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
