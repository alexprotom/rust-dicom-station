# Bundled example data

`data-test/` holds a real patient - the whole of TCIA **4D-Lung** patient
**P102** (about 980 MB in 1840 files), so the viewer can be exercised on
clinical data and not only on the synthetic phantom: a ten-phase 4DFBCT
with an RT Structure Set per phase, and the ten-phase 4DCBCT acquired from
the same patient nineteen days later. An installed copy of the program,
which has no source tree, fetches the same folder from GitHub with
*Tools ▶ 📥 Download test data*
([export-and-tools.md](export-and-tools.md#real-test-data-from-github)):

```
data-test/TCIA_4D-LUNG/P102/
  4DFBCT+RTS/                          study "4DFBCT+RTS"
    1_CT_4DFBCT__Gated__0.0_A/           133 CT slices, CT_0000.dcm ..
    1_CT_4DFBCT__Gated__10.0_A/          CT_0132.dcm
    ...                                  ten phases, 0 % .. 90 %
    1_CT_4DFBCT__Gated__90.0_A/
    RS_RTS__0.0_A.dcm                    one RTSTRUCT per phase,
    ...                                  "RTS, 0.0%A" .. "RTS, 90.0%A"
    RS_RTS__90.0_A.dcm
  4DCBCT/                              study "4DCBCT"
    500_CT_4DCBCT__Gated__0.0_A/         50 CT slices each
    ...                                  ten phases, 0 % .. 90 %
    509_CT_4DCBCT__Gated__90.0_A/
```

Both studies are one patient: `PatientID 4D-LUNG_TCIA_P102`, name `P102`,
`StudyID 1`, and no `StudyDate` on either (TCIA strips it; the slices keep
a Content Date).

| | 4DFBCT | 4DCBCT |
|---|---|---|
| Series | 10, all `SeriesNumber 1` | 10, `SeriesNumber 500` .. `509` |
| Slices per phase | 133 | 50 |
| Matrix / in-plane | 512 × 512, 0.9766 mm | 512 × 512, 0.8789 mm |
| Slice thickness | 3 mm (399 mm of coverage) | 3 mm (150 mm) |
| Description | `4DFBCT, Gated, <p>%A` | `4DCBCT, Gated, <p>%A` |
| Scanner | ADAC Pinnacle3 | Varian Trilogy Cone Beam CT, 125 kVp |
| Content date | 1998-03-25 | 1998-04-13 |
| Frame of Reference | one, shared by all ten phases | **one per phase** |

The percent in every series description is what the 4D detection keys on
([motion-4d.md](motion-4d.md)), so each study appears as a single ten-phase
group - `4DCT - 4DFBCT, Gated, A (10 phases)` and `4DCT - 4DCBCT, Gated, A
(10 phases)` - rather than as twenty loose series.

The structure sets carry twelve ROIs each - spinal cord, both lungs, heart,
esophagus, carina, a lymph node (`LN`), the tumor and four implanted gold
fiducial markers - plus a vertebra contour that exists only on phase 0, so
thirteen there. Every name carries its phase (`Tumor_c00`, `Tumor_c50`,
`_c90` …), and each set references the 133 images of its own phase.

The 4DCBCT is worth its own note: the ten phases sit on an identical image
grid (same position, spacing and size) but each carries a **different
Frame of Reference UID** - phase 0 keeps TCIA's, phases 10 % .. 90 % have
`2.25.…` ones. That is exactly the case the Frame-of-Reference hints and
*Transfer by relationship* exist for ([contours.md](contours.md)), and it
means a naive contour copy between CBCT phases is refused until a
relationship is given.

## What it is a test case for

```
cargo run --release -- data-test/TCIA_4D-LUNG/P102/4DFBCT+RTS data-test/TCIA_4D-LUNG/P102/4DCBCT
```

loads the planning 4DFBCT into workspace A and the 4DCBCT into workspace B,
which is the inter-study, inter-modality case: different geometry, different
scanner, different Frame of Reference, nineteen days apart. Equivalently,
open `data-test/` alone and both studies appear in one workspace, to be sent
on with right-click ▶ *Copy series to workspace …*.

Inside one study there is real respiratory motion to work with: the tumor
and the four markers move visibly between the phases, the 4D module plays
them as a loop, and the deformable methods of *Image registration* have
something anatomically real to recover. With an RTSTRUCT on every phase, the
structure propagation can be checked against a real contour rather than
against itself - propagate 0 % ▶ 50 % and compare with `RS_RTS__50.0_A`. It
is also the data the auto-segmentation was validated on
([auto-segmentation.md](auto-segmentation.md#validation)).

## Source and citation

The data is patient **P102** from the public **4D-Lung** collection on The
Cancer Imaging Archive (TCIA), a longitudinal 4D fan-beam CT / 4D
cone-beam CT collection of 20 locally advanced NSCLC patients treated with
chemoradiotherapy:

<https://www.cancerimagingarchive.net/collection/4d-lung/>

It is redistributed here under **CC BY 3.0**, the license of the original
collection. If you use it, cite the data and the associated publications:

> **Data.** Hugo, G. D., Weiss, E., Sleeman, W. C., Balik, S., Keall,
> P. J., Lu, J., & Williamson, J. F. (2016). *Data from 4D Lung Imaging of
> NSCLC Patients* (Version 2) [Data set]. The Cancer Imaging Archive.
> <https://doi.org/10.7937/K9/TCIA.2016.ELN8YGLE>
>
> **Publication.** Hugo, G. D., Weiss, E., Sleeman, W. C., Balik, S.,
> Keall, P. J., Lu, J., & Williamson, J. F. (2017). A longitudinal
> four-dimensional computed tomography and cone beam computed tomography
> dataset for image-guided radiation therapy research in lung cancer.
> *Medical Physics*, 44(2), 762-771. <https://doi.org/10.1002/mp.12059>
>
> **TCIA.** Clark, K., Vendt, B., Smith, K., Freymann, J., Kirby, J.,
> Koppel, P., Moore, S., Phillips, S., Maffitt, D., Pringle, M., Tarbox,
> L., & Prior, F. (2013). The Cancer Imaging Archive (TCIA): Maintaining
> and Operating a Public Information Repository. *Journal of Digital
> Imaging*, 26(6), 1045-1057. <https://doi.org/10.1007/s10278-013-9622-6>

## Anonymization

The copy here is TCIA's, byte for byte: the collection is already
de-identified - patient `P102` under the pseudonymous ID
`4D-LUNG_TCIA_P102`, no birth date, no institution or station name, study
dates removed - and nothing was rewritten on top of it, so the UIDs,
scanner tags and acquisition dates are the real ones a clinical archive
would hand over. That makes it a truthful test of the loader: original
UIDs of full length, a CBCT whose phases disagree about their Frame of
Reference, and `SeriesNumber 1` repeated across all ten 4DFBCT phases.

If you need a scrubbed copy - of this or of anything else - the built-in
anonymizer (*Tools ▶ 🔏 Anonymize DICOM folder…*, see
[export-and-tools.md](export-and-tools.md)) rewrites a folder to minimal,
readable identifiers while keeping pixel data, geometry, ROI names, colors,
types and contour points untouched, and every RTSTRUCT image reference
still resolving.
