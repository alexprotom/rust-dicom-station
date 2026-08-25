# Bundled example data

`example_data/` holds a small real patient study (137 MB) so the viewer
can be exercised on clinical data and not only on the synthetic phantom —
two breathing phases of a 4DCT, each with its own RT Structure Set:

```
example_data/
  lung_p1_4DCT_phase_000/   133 CT slices + 1-1.dcm (RTSTRUCT, 13 ROIs)
  lung_p1_4DCT_phase_050/   133 CT slices + 1-1.dcm (RTSTRUCT, 12 ROIs)
```

512 × 512, 0.977 mm in-plane, 3 mm slices, 396 mm of coverage. ROIs are
cord, both lungs, heart, esophagus, carina, lymph node (`LN`), tumor and
four implanted gold fiducial markers, plus a vertebra contour that exists
only on phase 000 (`_c00` / `_c50` suffix = breathing phase). Both series
share one Study Instance UID and one Frame of Reference, so they load as
inhale/exhale of the same study:

```
cargo run --release -- example_data/lung_p1_4DCT_phase_000 example_data/lung_p1_4DCT_phase_050
```

That is a ready-made comparison-mode and registration test case with real
respiratory motion: the tumor and the markers move visibly between the
phases, and *Registration ▶ Deformable* has something anatomically real to
recover. Equivalently, load the whole `example_data/` folder as dataset A
(both phases appear as two series of one study) and right-click one phase
▶ *Copy series to dataset B*. It is also the dataset the auto-segmentation
was validated on ([auto-segmentation.md](auto-segmentation.md#validation)).

## Source and citation

The data is patient **P102** from the public **4D-Lung** collection on The
Cancer Imaging Archive (TCIA), a longitudinal 4D fan-beam CT / 4D
cone-beam CT dataset of 20 locally advanced NSCLC patients treated with
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
> *Medical Physics*, 44(2), 762–771. <https://doi.org/10.1002/mp.12059>
>
> **TCIA.** Clark, K., Vendt, B., Smith, K., Freymann, J., Kirby, J.,
> Koppel, P., Moore, S., Phillips, S., Maffitt, D., Pringle, M., Tarbox,
> L., & Prior, F. (2013). The Cancer Imaging Archive (TCIA): Maintaining
> and Operating a Public Information Repository. *Journal of Digital
> Imaging*, 26(6), 1045–1057. <https://doi.org/10.1007/s10278-013-9622-6>

## Anonymization

The TCIA data is already de-identified; the copy here was additionally
rewritten to minimal, readable identifiers — patient `lung_p1`, and a UID
tree that is easy to read in a debugger:

| | phase_000 | phase_050 |
|---|---|---|
| CT series | `1.2.3.4.5.10` | `1.2.3.4.5.20` |
| CT slices | `1.2.3.4.5.10.<InstanceNumber>` | `1.2.3.4.5.20.<InstanceNumber>` |
| RTSTRUCT series / instance | `1.2.3.4.5.11` / `.11.1` | `1.2.3.4.5.21` / `.21.1` |

with `1.2.3.4.5.1` as the shared Study Instance UID and `1.2.3.4.5.2` as
the shared Frame of Reference UID. Everything not needed to render the
images and contours — accession number, device manufacturer and model,
software versions, acquisition dates and private tags — was dropped; pixel
data, geometry, ROI names, colors, types and contour points are untouched,
and every RTSTRUCT image reference still resolves to a slice of its own
series. The built-in anonymizer (*Tools ▶ 🔏 Anonymize DICOM folder…*, see
[export-and-tools.md](export-and-tools.md)) does the same to any folder.
