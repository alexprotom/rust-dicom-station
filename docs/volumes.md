# How volumes are calculated

Every volume the station reports comes from one of three calculations:

| Name in the interface | What is measured | Where it appears |
|---|---|---|
| **Voxels-based** | the number of voxels a structure occupies on an image lattice, times the voxel volume | *Structure details*, the propagation report, DVH, structure comparison, motion, segment lists, every MCP tool |
| **Surface-based** | the volume enclosed by a closed triangle surface reconstructed from the planar contours | *Structure details*, the propagation report |
| **Planimetry** | contour area on each slice times the slice spacing | the drawing panel, while a structure is edited; the confirmations of tools that create a structure |

This page gives the exact rules and formulas. Code: `src/segmentation.rs` (`rasterize_roi`,
`mask_to_roi`), `src/contours.rs` (`Stack`, `Region::area`), `src/rt_surface.rs`,
`src/propagate.rs`, `src/app/stats_win.rs`, `src/app/run_report.rs`.

---

## Common ground

**Units.** Coordinates are patient millimetres (DICOM LPS). Volumes are reported in cm³
(`1 cm³ = 1000 mm³`), shown to two decimals.

**Voxel volume.** For an image lattice with spacings `sx`, `sy`, `sz` (mm):

```
v = sx · sy · sz / 1000            [cm³]
```

- `sx` and `sy` come from Pixel Spacing (0028,0030). `sx` is its second value, the distance
  between columns along a row (index `i`). `sy` is its first value, the distance between rows
  (index `j`).
- `sz` is the median of the differences between consecutive slice positions, measured along the
  slice normal `n = row direction × column direction`:
  `zₖ = ImagePositionPatientₖ · n`, `sz = median(zₖ₊₁ − zₖ)`.
  - Slices closer together than 0.01 mm count as duplicates.
  - When the spacing varies by more than 1 % of the median, the loader warns and still uses the
    median.
  - A single slice uses Slice Thickness (0018,0050), or 1 mm if that is missing.

**Patient → voxel indices.** With `o` the centre of voxel (0, 0, 0) and `r`, `c`, `n` the row,
column and slice directions:

```
i = (p − o)·r / sx      j = (p − o)·c / sy      k = (p − o)·n / sz
```

Voxel `(i, j, k)` is the box centred on those integer indices.

---

## 1. Voxels-based volume

```
V_vox = N · v
```

`N` is the number of voxels set in the structure's mask on the lattice in question.

### A segment

A segment (a painted or model-made mask) is already a mask. `N` is its count of set voxels, and
`v` comes from the series the segment belongs to.

### An RT structure: from contours to a mask

An RTSTRUCT ROI is first rasterized onto the lattice (`segmentation::rasterize_roi`). The rule:
**a voxel is inside when its centre is inside the contours of its slice, by the even-odd rule.**

1. Each contour with at least three points is mapped to voxel indices `(i, j, k)` with the
   formula above. It belongs to slice `k = round(mean kᵥ)` over its vertices. Contours outside
   `0 ≤ k < nz` are ignored, as are POINT contours.
2. A path that passes through one of its own vertices again (bit-identical coordinates) is cut
   there into separate loops, so each loop counts.
3. A loop whose shoelace area, in voxel units, is `≤ 10⁻⁶` is dropped:

   ```
   A = ½ · | Σₘ (iₘ · jₘ₊₁ − iₘ₊₁ · jₘ) |          (indices cyclic)
   ```

4. All remaining loops of one slice are filled together, one row at a time. Each row `j` is
   scanned along the line through its voxel centres, `y = j`. Every loop edge `(p, q)` with
   `(p_y ≤ y) ≠ (q_y ≤ y)` gives one crossing. The test is half-open, so a vertex lying exactly
   on the line is counted once:

   ```
   x = p_x + (y − p_y) · (q_x − p_x) / (q_y − p_y)
   ```

   The crossings are sorted and taken in pairs `(x₁, x₂)`. Voxels `i` with
   `ceil(x₁) ≤ i ≤ floor(x₂)` are set, clipped to the image. All loops of a slice share one
   list of crossings, so a contour inside another is a hole.

5. `N` is the number of set voxels, and `V_vox = N · v` on that lattice.

In *Structure details* the lattice is the image the workspace displays. In a propagation it is
the source image on the source side and the destination image on the deformed side.

### From a mask back to contours

When a mask is filed as contours (*→RS*, or a propagation landed as a structure set),
`segmentation::mask_to_roi` writes, on every slice, the outline of the set voxels themselves,
traced along the voxel edges, with collinear points merged. The outline encloses exactly the
voxels, so rasterizing it with the rule above returns the same mask, and its voxels-based volume
equals the mask's to the voxel.

---

## 2. Planimetry

Used by the drawing panel (the volume shown under the structure's name, recomputed every frame)
and by the confirmations of tools that create a structure.

1. Contours are read onto the image lattice as in §1, steps 1–2.
   - The stack axis is the lattice axis along which the contours have the smallest total extent
     (axial on a tie).
   - Each loop goes to slice `round(mean level)`.
2. On each slice, loops with an area `≤ 10⁻⁶` voxel² are dropped. Each remaining loop gets its
   nesting depth `d`: how many other loops of the same slice contain it.
3. Slice area, in voxel units, with holes subtracted:

   ```
   Aₛ = max(0, Σ_loops (−1)^d · A_loop)
   ```

4. Volume, with `u` and `w` the in-plane axes and `a` the stack axis:

   ```
   V_plan = (Σₛ Aₛ) · s_u · s_w · s_a / 1000        [cm³]
   ```

   Each slice counts for one full slice spacing, and only slices inside the image count.

For contours written from a mask (§1, *From a mask back to contours*) this equals `V_vox`
exactly.

---

## 3. Surface-based volume

The contours are joined into a closed triangle surface, and the volume inside it is integrated.
The surface does not depend on any image lattice. Only its default slice spacing (§3.4) is taken
from the image.

### 3.1 Reading

- Each contour becomes one *line*.
- Its points `(x, y, z)` are converted to `(−x, −y, z)`, which turns LPS into RAS, and rounded
  to single precision (32-bit floats).
- The line is closed by repeating its first point, so a contour of `n` points gives `n + 1`
  vertices.
- An ROI with fewer than two points has no surface.

### 3.2 Orientation

- For each line with more than six distinct points (or for every line, if none has that many),
  the plane normal is the eigenvector of the point covariance matrix with the smallest
  eigenvalue.
- These normals are summed (each one flipped to agree with the running sum) and normalised.
- If the sum does not point along `z`, all points are rotated so that it does. For axial
  contours the rotation is the identity.

### 3.3 Order, keyholes, winding

- **Order.** Lines are sorted by `z_mid = (z_min + z_max)/2` of their bounding box. Ties keep
  the input order.
- **Keyholes.** Take a line of `n` vertices. Two positions `a < b` form a keyhole pair when
  their points lie within `ε = 0.001 mm` of each other and are more than 3 positions apart,
  `min(b − a, n − 1 − b + a) > 3`.
  - A line with keyhole pairs is walked once, keeping a stack of open sub-lines.
  - Reaching the first point of a pair opens a new sub-line.
  - Reaching the second point closes the current sub-line.
  - Each resulting sub-line is closed by repeating its first point. Sub-lines of fewer than two
    points are dropped.
- **Winding.** A line is clockwise when `Σₘ (xₘ₊₁ − xₘ)(yₘ₊₁ + yₘ) > 0`, and clockwise lines
  are reversed. **Every line ends up counter-clockwise, holes included.** A contour nested inside
  another is therefore joined as a solid of its own: the reconstruction has no notion of a hole.

### 3.4 Slice spacing and planes

```
dᵢ = | z_mid(lineᵢ₊₁) − z_mid(lineᵢ) |      kept when dᵢ > 0.01 mm
m  = mean(dᵢ)
Δ  = mean of the dᵢ with |dᵢ − m| < m/10      (m itself if none qualify)
```

If there is no `dᵢ` at all (one line, or all lines on one plane), `Δ` is the slice spacing `sz`
of the image the structure is measured on, rounded to six significant digits.

A *plane* is a run of consecutive sorted lines whose `z_mid` lies within `0.1 Δ` of the run's
first line.

### 3.5 Joining neighbouring planes

For every pair of consecutive planes:

- **Overlap.** Line `L₁` on the lower plane and line `L₂` on the upper plane overlap when their
  xy bounding boxes overlap strictly:
  `x₁,min < x₂,max ∧ x₁,max > x₂,min ∧ y₁,min < y₂,max ∧ y₁,max > y₂,min`.
- **Branching.** A line that overlaps exactly one line on the other plane is used whole. A line
  that overlaps several is divided among them:
  - Each of its points goes to the overlapping line that has the nearest point to it (squared
    Euclidean distance; the first such line on a tie).
  - The part for `L₂` is the points assigned to `L₂`, plus the first point after each run of
    them.
  - If the line was closed, the part is closed by repeating its first point (a *chord* back to
    its start).
- **Strip.** When both parts have at least two points, they are joined by a strip of triangles.
  `L₁` is then marked *joined above*, and `L₂` *joined below*.

The strip is a dynamic-programming match between the two point sequences.

- Let `P₀…P_{n₁−1}` and `Q₀…Q_{n₂−1}` be the two parts.
- Let `c₁₂(i)` be the index of the point of `Q` nearest to `Pᵢ`, and `c₂₁(j)` the reverse.
- `P` starts at `P₀` and `Q` at `Q_{c₁₂(0)}`. Both advance cyclically, skipping the repeated
  closing point of a closed line.
- The score table `S` is built from squared distances `δ(i, j) = |Pᵢ − Qⱼ|²`:

```
S(0,0) = δ(0,0)
S(0,j) = S(0,j−1) + δ(0,j)                          (step along Q)
S(i,0) = S(i−1,0) + δ(i,0)                          (step along P)
S(i,j) = δ(i,j) + S(i,j−1)                          if Pᵢ is P's nearest point to the previous Q point
       = δ(i,j) + S(i−1,j)                          else if Qⱼ is Q's nearest point to the previous P point
       = δ(i,j) + min(S(i,j−1), S(i−1,j))           otherwise (S(i,j−1) on a tie)
```

- "Previous" means the point of the preceding column (for `Q`) or the preceding row (for `P`).
- The position along `Q` is not reset between rows: each row continues from where the previous
  row ended.
- Walking back from the last cell to `(0,0)`, always taking the step that filled the cell, gives
  one triangle per step: `(Pᵢ, Qⱼ, Qⱼ₋₁)` for a step along `Q`, and `(Pᵢ, Qⱼ, Pᵢ₋₁)` for a step
  along `P`.

### 3.6 End caps

A line not joined above gets a cap at `z + Δ/2`, and a line not joined below gets one at
`z − Δ/2`. The cap is a shrunken copy of the line:

1. **Raster.** A 2-D image covers the line's bounding box plus two pixels on each side.
   - The pixel size is `s = min(1 mm, extent/28)` along each axis. If either extent is 0, it is
     1 mm on both axes.
   - A pixel is set when its centre is inside the line (even-odd rule), with a tolerance of
     `7.63·10⁻⁶` pixel on the row positions and on the span ends.
2. **Erosion.** A pixel stays set only when every pixel of the kernel
   `{(dx, dy) : (dx/2.5)² + (dy/2.5)² ≤ 1}` around it is set. This kernel is the 5 × 5 square
   without its four corners (21 pixels). Pixels outside the image are ignored. The erosion is
   repeated while more than half of the originally set pixels remain and the last pass removed
   something: `while N > ⌊N₀/2⌋ and ΔN > 0`.
3. **Outline.** The boundary of the remaining pixels is traced at the level 1 of the 0/1 image
   (marching squares). This gives segments between the centres of the boundary pixels, which are
   chained into poly-lines. Poly-lines of two points or fewer are dropped, and so are poly-lines
   that turn back on themselves at their third point.
4. **Thinning.** Points of each outline are removed one at a time, the point with the smallest
   error first. A point's error is its squared distance from the line through its two
   neighbours. Removal continues while more than three points are queued and either:
   - the smallest error is below machine epsilon, or
   - the fraction of the outline's points still kept exceeds `f = (L · n + 1) / M`, where `L` is
     the number of outlines, `n` the number of vertices of the original line and `M` the number
     of boundary points traced.

   One queue serves all outlines of a cap.
5. **Placement.** The outlines are made counter-clockwise, closed, and placed at `z ± Δ/2`. If
   erosion leaves nothing, the cap is the line itself moved to `z ± Δ/2`.

Each cap is filled with triangles: normals up for a cap above, down for a cap below. Consecutive
vertices closer than `10⁻⁶ ×` the cap's bounding-box diagonal are merged first. Any triangulation
of a flat polygon contributes the same to the volume. The cap is then joined to its line by a
strip as in §3.5, and a cap that came out in several pieces divides the line among them as in
*Branching*.

### 3.7 Closing the seams

The strips alone do not always close the surface.

- **When the chords cancel.** A line divided between two lines on the next plane, whose stretches
  follow one another along it, gives two parts. Their chords (§3.5) are the same edge walked in
  both directions, so they cancel.
- **When they don't.** A line divided among three or more lines, or between two whose stretches
  alternate along it, gives chords that do not cancel. The surface then has a gap there: a
  *seam*. The same can happen where a cap came out in several pieces.

Every seam is closed before the volume is measured:

1. Each triangle contributes its three directed edges. For every pair of vertices,
   `net(a, b) = #(a→b) − #(b→a)`. On a closed surface every `net` is 0. The edges with
   `net ≠ 0`, taken `|net|` times in the direction of the surplus, form the seams' boundaries.
2. These edges are chained into closed loops `v₁ … v_m`. Each loop is covered by a fan of
   triangles from its centroid `c = (1/m) Σ vₖ`, walked the opposite way: `(c, vₖ₊₁, vₖ)`.

The area added is `A_seam = Σₖ ½ |(vₖ − c) × (vₖ₊₁ − c)|`.

### 3.8 The enclosed volume

For each triangle `(p₀, p₁, p₂)` of the closed surface:

```
u      = (p₁ − p₀) × (p₂ − p₀),    n̂ = u / |u|
a, b, c = |p₂ − p₀|, |p₁ − p₀|, |p₂ − p₁|,    s = (a + b + c)/2
A      = √| s (s − a)(s − b)(s − c) |                          (Heron)
x̄, ȳ, z̄ = the means of the three vertices' coordinates
```

```
Vx = Σ A · n̂x · x̄      Vy = Σ A · n̂y · ȳ      Vz = Σ A · n̂z · z̄
```

Each triangle counts towards the axis its normal points along most. `mx`, `my` and `mz` count
the triangles whose `|n̂x|`, `|n̂y|` or `|n̂z|` is strictly the largest. Ties are counted in
`w_xyz` (all three equal), `w_xy`, `w_xz` and `w_yz`. With `T` triangles:

```
kx = (mx + w_xyz/3 + (w_xy + w_xz)/2) / T
ky = (my + w_xyz/3 + (w_xy + w_yz)/2) / T
kz = (mz + w_xyz/3 + (w_xz + w_yz)/2) / T

V_surf = | kx·Vx + ky·Vy + kz·Vz | / 1000        [cm³]
```

On a closed surface each of `Vx`, `Vy` and `Vz` is the enclosed volume (the divergence
theorem), so `V_surf = |Σ p₀·(p₁ × p₂)| / 6 / 1000` over the triangles. The weights do not
change the result, and neither does where the structure lies.

**Why the seams have to be closed.** Take an open surface and move the structure by `t` along
`z`. Its vector area `Σ A·n̂` is not zero, so

```
Vz(t) = Vz(0) + t · Σ A·n̂z
```

Without closing, the result would change linearly with the structure's distance from the
coordinate origin. On a 4DCT a metre down the couch (z ≈ −1080 mm), a heart whose contours split
and merge from slice to slice would read hundreds of cm³ too low. Closed, it reads the same
wherever it lies.

### 3.9 Surface-based against voxels-based

- **End caps.** The surface ends at a shrunken cap half a slice beyond the first and last
  contours, while the voxels on those slices count a full slice spacing. A small structure, or
  one exported voxel by voxel as a thin sheet, therefore reads smaller as a surface. Most of such
  a structure is ends.
- **Keyhole pairs.** A pair of voxels chained through a shared corner into one path (§3.3) comes
  apart into two triangles of half a voxel each. As voxels it counts two full voxels.
- **Holes.** A contour nested inside another is joined as a solid (§3.3), so a structure with
  holes reads larger as a surface than as voxels, by roughly what the holes hold.
- **Points.** A point of interest has no volume. *Structure details* shows its position instead.

---

## 4. Propagation

`src/propagate.rs`. A structure on the source (moving) image is carried onto the destination
(fixed) image through a transform `T` that maps destination points to source points.

**Source volume.** The source structure as a mask on the source lattice. Contours are
rasterized as in §1; a segment on another lattice is resampled.

```
V_src = N_src · v_src
```

**Occupancy.** Each destination voxel `q` is sampled at `n_x · n_y · n_z` sub-points, where
`n_a = clamp(round(s_dst,a / s_src,a), 1, 4)`. The sub-points sit at offsets
`(t + ½)/n_a − ½` voxel, for `t = 0 … n_a − 1`, about the voxel centre:

```
o(q) = (1 / n_x n_y n_z) · Σ_sub-points  m̃( T(q + offset) )
```

- `m̃` is the trilinear interpolation of the 0/1 source mask at the fractional source voxel
  indices of the mapped point, and 0 outside the source image.
- `T` is evaluated on a lattice of nodes about 3 mm apart over the destination box and
  interpolated trilinearly in between. This is exact for a rigid `T`.

**Deformed (mapped) volume**, the structure as the transform maps it:

```
V_map = v_dst · Σ_q o(q)
```

**Filed volume.** The mask gets the `K = round(Σ o(q))` voxels with the largest occupancy, ties
broken by voxel index. It therefore holds `V_map` to within half a voxel:

```
V_res = K · v_dst
```

The ratio of the two volumes is the mean local volume change of the transform over the
structure:

```
V_src ≈ v_dst · Σ_{q ∈ mask} det ∇T(q)        so        V_res / V_src ≈ 1 / mean det ∇T
```

`det ∇T` is the Jacobian determinant of the destination → source map. A value below 1 means the
destination is smaller than the source there.

**Close gaps / fill**, when asked for:

- *Close gaps* dilates the mask by `r`, then erodes it by `r/2`.
- *Fill* fills each slice between those two steps, and the erosion is then `r`.
- *Fill* without *close gaps* fills each slice of the mask directly.
- The result is combined with what landed (logical OR).

The filed volume is then recounted, `V_res = N · v_dst`. `V_map` stays what the transform made of
the structure.

**Keep shape.** Over the structure, `T` is replaced by its best rigid fit: orthogonal Procrustes
on up to 4000 of the structure's voxel centres. The structure keeps its volume, up to resampling
onto the destination lattice. The report gives the RMS distance between the fit and `T` over
those points.

**Change.**

```
Δ% = 100 · (V_after − V_before) / V_before       (0 when V_before ≤ 10⁻⁹)
```

**Landed as a structure set.** The filed mask is written as contours (§1, *From a mask back to
contours*), and the report measures both sides both ways:

| Column | Source | Deformed |
|---|---|---|
| Surface-based | `V_surf` of the source ROI, with `Δ` defaulting to the source image's `sz` | `V_surf` of the filed ROI, with the destination image's `sz` |
| Voxels-based | `V_src` | `V_vox` of the filed ROI on the destination image (= `V_res`) |

The voxels-based `Δ%` is the transform's volume change: it compares two masks that were counted
the same way. The surface-based `Δ%` compares two surfaces. When the source contours and the
filed contours are of different kinds (a thin sheet of voxel squares against a compact filed
outline, say), their end caps and keyhole pairs (§3.9) take away different fractions, and the
surface-based `Δ%` also contains that difference.

---

## 5. Other volumes

- **DVH** (`src/dvh.rs`): `V = (N_in + N_out) · v` over the structure's mask on the image
  lattice. `N_out` counts voxels outside the dose grid. They go into the lowest dose bin and are
  reported as *outside*.
- **Structure comparison, motion and ITV** (`src/motion.rs`): voxels-based, on the lattice the
  comparison is made on.
- **Segments listed in a workspace**: `N · v` of the segment's own series.
- **MCP tools** (`volume_cm3`, `source_cm3`, `mapped_cm3`, `result_cm3`): voxels-based, as above.
- **Structures created by a tool** (a grey-level window, a shape, an isodose): the confirmation
  gives the planimetry (§2) of the outline written for the new structure.

---

## Worked numbers

The STAR test patient: a cardiac CT (0.547 × 0.547 × 0.6 mm) and its 4DCT
(1.17 × 1.17 × 2 mm, a metre down the couch at z ≈ −1080 mm).

| Structure | Voxels-based, cm³ | Surface-based, cm³ | Seams closed, mm² |
|---|---|---|---|
| CCT target (1031 voxel squares) | 0.62 | 0.53 | 0 |
| CCT heart_total | 905.49 | 913.47 | 82 |
| 4DCT 0 % heart_total (the clinic's contour) | 923.70 | 936.15 | 838 |
| heart_total carried onto the 4DCT (anchored, deformable) | 910.32 | 918.92 | 2858 |

The target is a sheet of 1 mm voxel squares: nearly all of it is ends, so its surface stops
short of the voxels (§3.9). The hearts are compact, and there the two measures agree to about a
per cent. The carried heart has 2858 mm² of seams. Without closing them, its volume would read
350 cm³ at its position and 706 cm³ with the contours moved 700 mm up `z`: `0.508 cm³/mm` of
drift, the `Σ A·n̂z` term of §3.8. With them closed, it reads 918.92 cm³ in both places.

The UPSTAR pair: the target and heart of a cardiac CT (0.715 × 0.715 × 1 mm) carried, anchored on
the heart, onto another patient's 4DCT (0.764 × 0.764 × 0.5 mm).

| Structure | Voxels-based, cm³ | Surface-based, cm³ | Mean `det ∇T` over the result |
|---|---|---|---|
| target, source | 2.89 | 1.95 | |
| target, carried | 1.89 (−34.4 %) | 1.76 (−9.8 %) | 1.53 |
| target, carried rigidly (no deformation) | 2.89 (+0.1 %) | 2.72 (+39.2 %) | 1.00 |
| heart, source | 947.20 | 948.10 | |
| heart, carried | 794.32 (−16.1 %) | 796.88 (−16.0 %) | 1.19 |

- The voxels-based change is the transform's (§4): `v_dst · Σ det ∇T` over the carried target is
  2.90 cm³, the source volume to 0.6 %. The heart is matched onto a smaller heart (805 cm³), and
  the target, 9 mm under its surface, is squeezed more than the heart on average.
- The surface-based change also contains a change of shape class (§3.9). The source target is a
  sheet of 1 mm squares on 1 mm slices, and its surface holds 68 % of its voxels. The carried
  target is a compact outline on 0.5 mm slices, and its surface holds 93 %. With no deformation
  at all, the surface-based figure still moves by +39 %.

`src/rt_surface.rs`'s unit tests check the reconstruction on shapes of known volume:

- a stack of squares, which gives a closed surface with no seams;
- a block that splits into four posts, which leaves a 1600 mm² seam that is closed, after which
  the volume no longer moves with the origin;
- a single contour, which takes its thickness from the image;
- a path through one vertex twice, which is read as two loops.
