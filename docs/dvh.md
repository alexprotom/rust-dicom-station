# Dose-volume histograms

Cumulative and differential DVHs of any structures against any loaded dose
objects, with the metrics table, protocol constraint checking and CSV export.

## Opening it

*Tools ▶ 📊 Dose-volume histograms…*, or tick structures in the data tree,
right-click and choose **📊 Plot … on a DVH**: the window opens with them
already picked and the viewport's dose object already selected.

Like every tool window it goes through [the detach
mechanism](architecture.md#tool-windows): ⧉ puts it on its own top-level
window - for a DVH the normal way to work, curves on one screen, images on the
other.

## What it computes

For every (structure, dose object) pair:

* the **cumulative** histogram - volume receiving at least each dose, the
  curve every constraint is read off;
* the **differential** histogram - volume per dose bin, where a cold spot
  inside a target shows up as a second hump;
* the statistics: minimum, mean and maximum;
* whatever metrics the table is asked for.

Structures may come from either dataset and either kind - RT structure or
segmentation - and any number of dose objects may be overlaid. Structures keep
their own colour and the dose object picks the line style, so two plans over
the same organs read as one colour in two dashes.

## Four things it is careful about

**Where it samples.** The structure's own lattice, not the dose grid: a CT
mask is 1 mm and a dose grid 2-3 mm, so walking the mask and interpolating the
dose gives a curve at the structure's resolution. The walk is affine, the
dose-grid coordinates stepped rather than recomputed per voxel - three adds
instead of three dot products.

**What falls outside the dose grid.** Counted, kept, and said out loud: those
voxels enter the histogram at zero dose - the honest reading of "not
irradiated by *this* dose object" - and, since a DVH silently computed over 60
% of a structure looks cold rather than truncated, a warning line names every
structure that extends outside the grid and by how much.

**Statistics from the samples, not the bins.** Minimum, mean and maximum are
accumulated during the walk; reading them off a binned histogram costs half a
bin width of accuracy for nothing.

**Interpolation inside a bin - except the lowest.** D95 % is almost never
exactly at a bin edge, so the cumulative curve is interpolated linearly
between edges. The lowest bin holds exact zeros, so a reading inside it
returns 0 rather than a few hundredths of a Gy that would look like a real
dose.

The histogram uses 2000 bins over the dose maximum - 3 cGy on a 60 Gy plan,
finer than any constraint is quoted to.

## The axes

Both are switchable, independently:

| | |
|---|---|
| **Dose** | Gy, or per cent of a reference dose |
| **Volume** | per cent of each structure, or cm³ |

The reference defaults to the prescription of the first plan that declares
one, and the window says which plan that was; ↺ restores it after you have
typed something else. Dose-valued table columns follow the same switch, so
`Dmean` reads in per cent when the axis does.

## The metrics table

One row per curve. It starts with volume, minimum, mean, maximum, D95 % and D2
%, and takes any column you type:

| You type | You get |
|---|---|
| `Dmean`, `Dmax`, `Dmin` | the statistics |
| `D95%`, `D2%` | dose to at least that percentage of the structure |
| `D2cc`, `D0.1cc` | dose to at least that absolute volume |
| `V20`, `V20Gy` | percentage of the structure at or above that dose |
| `V20cc` | the same as an absolute volume |

## Constraint checking

A protocol is a plain text file, one constraint per line - human-editable on
purpose - that is how a department keeps them:

```
# head and neck, 30 fractions
PTV*         D95%   >= 57
PTV*         Dmax   <= 63
Cord         Dmax   <= 45
"Parotid L"  Dmean  <= 26
Lung*        V20Gy  <= 30
```

The structure name is matched case-insensitively; a leading or trailing `*`
matches loosely, so `PTV*` catches `PTV_5400`; names with spaces are quoted.
The header line of the collapsing section says how many constraints are met,
and each row shows ✔ or ✖ against the value.

A constraint that matches **no** structure is reported with a dash and does
**not** pass - a line that quietly evaluates to "fine" because the structure
was never contoured is the worst failure mode a checker can have.

## Export

**Export curves…** writes the cumulative curves as CSV: one dose column, then
one volume column per structure, following the volume axis currently shown.
Curves against different dose objects may differ in bin width, so they are
resampled onto one dose axis at the finest of them rather than assumed to
share one.

**Export table…** writes the metrics table as it stands.

## Verification

`src/dvh.rs`'s own tests check the arithmetic on grids built in the test: a
uniform dose gives a step and exact statistics; a linear ramp gives a DVH
linear to within 2 %, read in both directions; voxels outside the dose grid
are counted, reported, and drag D60 % to zero without disturbing the
statistics of what *was* irradiated; metric names round-trip through `label()`
and `parse()`; a protocol survives a write and re-read, quoted name included;
and the CSV puts curves with different bin widths on one axis.

`tests/dvh.rs` goes through the whole path: the synthetic RT study is written
as DICOM, read back through the loader, its contours rasterized, and the
histogram taken against the RTDOSE as parsed. The phantom's dose is an
analytic Gaussian centred on a spherical target, so the target's DVH is known
in closed form - the volume above dose `D` is the ball of radius
`σ·√(2·ln(peak/D))` - and the test compares against that formula, not a
previous run, at six doses and three volume levels, to within 5 % of volume
and 2 Gy of dose. It also checks that a cumulative curve never rises, that a
structure contained in another is nowhere hotter in absolute volume, and that
a protocol reads the phantom the way a physicist would.

As with everything in this viewer: research and QA use - not a medical device,
not for clinical decision-making.
