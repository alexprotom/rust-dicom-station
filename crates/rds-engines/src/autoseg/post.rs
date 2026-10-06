//! The label post-processing TotalSegmentator applies to some tasks, on the
//! model grid, before the labels are mapped back to the scan
//! (`totalsegmentator/postprocessing.py`).
//!
//! * `body`: the trunk keeps its largest connected piece, and pieces of the
//!   extremities smaller than 50 cm³ are dropped.
//! * `vertebrae_pp`: the vertebral-body model labels each body by level; where
//!   two different levels touch, the labelling leaked, so the bodies are
//!   split into connected pieces again and numbered anatomically from the
//!   bottom up (from the top down on a head scan that shows C1 but not L5).
//!   Each label is then grown by 3 mm into the background, undoing the
//!   erosion the training labels were made with.
//!
//! Connectivity is scipy's `ndimage.label` default, the six face
//! neighbours, and components are numbered in raster order as scipy numbers
//! them, so ties resolve the way upstream's do.

/// Six-connected components of `mask` on a C-order grid of `dims`.
/// Returns the component id per voxel (0 = background, ids from 1 in raster
/// order) and the voxel count per id (index 0 unused).
pub fn components(mask: &[bool], dims: [usize; 3]) -> (Vec<u32>, Vec<u64>) {
    let [d0, d1, d2] = dims;
    let n = d0 * d1 * d2;
    debug_assert_eq!(mask.len(), n);
    let mut id = vec![0u32; n];
    let mut sizes = vec![0u64];
    let mut stack: Vec<usize> = Vec::new();
    for start in 0..n {
        if !mask[start] || id[start] != 0 {
            continue;
        }
        let c = sizes.len() as u32;
        sizes.push(0);
        id[start] = c;
        stack.push(start);
        while let Some(v) = stack.pop() {
            sizes[c as usize] += 1;
            let a = v / (d1 * d2);
            let b = (v / d2) % d1;
            let e = v % d2;
            let mut visit = |w: usize| {
                if mask[w] && id[w] == 0 {
                    id[w] = c;
                    stack.push(w);
                }
            };
            if a > 0 {
                visit(v - d1 * d2);
            }
            if a + 1 < d0 {
                visit(v + d1 * d2);
            }
            if b > 0 {
                visit(v - d2);
            }
            if b + 1 < d1 {
                visit(v + d2);
            }
            if e > 0 {
                visit(v - 1);
            }
            if e + 1 < d2 {
                visit(v + 1);
            }
        }
    }
    (id, sizes)
}

/// Keep only the largest connected piece of `label` (the first of equal
/// size, as `np.argmax` picks).
pub fn keep_largest(data: &mut [u8], dims: [usize; 3], label: u8) {
    let mask: Vec<bool> = data.iter().map(|&l| l == label).collect();
    let (id, sizes) = components(&mask, dims);
    if sizes.len() <= 1 {
        return;
    }
    let mut best = 1;
    for c in 2..sizes.len() {
        if sizes[c] > sizes[best] {
            best = c;
        }
    }
    for (l, &c) in data.iter_mut().zip(&id) {
        if c != 0 && c as usize != best {
            *l = 0;
        }
    }
}

/// Drop the connected pieces of `label` with `min_voxels` voxels or fewer
/// (upstream's interval test is `count <= lower`).
pub fn remove_small(data: &mut [u8], dims: [usize; 3], label: u8, min_voxels: f64) {
    let mask: Vec<bool> = data.iter().map(|&l| l == label).collect();
    let (id, sizes) = components(&mask, dims);
    for (l, &c) in data.iter_mut().zip(&id) {
        if c != 0 && sizes[c as usize] as f64 <= min_voxels {
            *l = 0;
        }
    }
}

/// The `body` task's post-processing. `spacing` is the model grid's, in
/// mm; `classes` the task's class table.
pub fn body(data: &mut [u8], dims: [usize; 3], spacing: [f64; 3], classes: &[&str]) {
    let label = |name: &str| classes.iter().position(|c| *c == name).map(|i| i as u8 + 1);
    if let Some(trunk) = label("body_trunc") {
        keep_largest(data, dims, trunk);
    }
    if let Some(ext) = label("body_extremities") {
        let voxel_mm3 = spacing[0] * spacing[1] * spacing[2];
        remove_small(data, dims, ext, 50_000.0 / voxel_mm3);
    }
}

/// True when two different non-zero labels share a face anywhere.
fn labels_touch(data: &[u8], dims: [usize; 3]) -> bool {
    let [_, d1, d2] = dims;
    let strides = [d1 * d2, d2, 1];
    for (axis, &st) in strides.iter().enumerate() {
        for v in 0..data.len() {
            let coord = match axis {
                0 => v / (d1 * d2),
                1 => (v / d2) % d1,
                _ => v % d2,
            };
            if coord + 1 >= dims[axis] {
                continue;
            }
            let (a, b) = (data[v], data[v + st]);
            if a != 0 && b != 0 && a != b {
                return true;
            }
        }
    }
    false
}

/// Grow every label of `labels` (in ascending order) by `radius_mm` into the
/// background, through an ellipsoid of that radius in millimetres. Labels
/// already placed are never overwritten, and each label grows from where it
/// stood before any growing (`dilate_vertebrae_labels`).
fn dilate_labels(
    data: &[u8],
    dims: [usize; 3],
    spacing: [f64; 3],
    radius_mm: f64,
    labels: &[u8],
) -> Vec<u8> {
    let mut out = data.to_vec();
    if radius_mm <= 0.0 {
        return out;
    }
    // The structuring element exactly as numpy builds it:
    // `np.ogrid[-r:r+1]` per axis, kept where sum((v / r)^2) <= 1, centred
    // on index n // 2 as scipy centres it.
    let radii: [f64; 3] = std::array::from_fn(|a| radius_mm / spacing[a]);
    let vals: [Vec<f64>; 3] = std::array::from_fn(|a| {
        let r = radii[a];
        let n = (2.0 * r + 1.0).ceil() as usize;
        (0..n).map(|e| -r + e as f64).collect()
    });
    let mut offsets: Vec<[isize; 3]> = Vec::new();
    for (e0, v0) in vals[0].iter().enumerate() {
        for (e1, v1) in vals[1].iter().enumerate() {
            for (e2, v2) in vals[2].iter().enumerate() {
                let s = (v0 / radii[0]).powi(2) + (v1 / radii[1]).powi(2) + (v2 / radii[2]).powi(2);
                if s <= 1.0 {
                    offsets.push([
                        e0 as isize - (vals[0].len() / 2) as isize,
                        e1 as isize - (vals[1].len() / 2) as isize,
                        e2 as isize - (vals[2].len() / 2) as isize,
                    ]);
                }
            }
        }
    }
    let [d0, d1, d2] = dims;
    let idx = |a: usize, b: usize, c: usize| (a * d1 + b) * d2 + c;
    for &label in labels {
        let mut grow: Vec<usize> = Vec::new();
        for a in 0..d0 {
            for b in 0..d1 {
                for c in 0..d2 {
                    if data[idx(a, b, c)] != label {
                        continue;
                    }
                    for o in &offsets {
                        let (x, y, z) = (a as isize + o[0], b as isize + o[1], c as isize + o[2]);
                        if x < 0
                            || y < 0
                            || z < 0
                            || x >= d0 as isize
                            || y >= d1 as isize
                            || z >= d2 as isize
                        {
                            continue;
                        }
                        grow.push(idx(x as usize, y as usize, z as usize));
                    }
                }
            }
        }
        for v in grow {
            if out[v] == 0 {
                out[v] = label;
            }
        }
    }
    out
}

/// The `vertebrae_pp` post-processing (`postprocess_vertebrae_pp`).
///
/// `s_axis` is the grid axis that increases toward the head; `classes` the
/// task's table (`vertebrae_C1` .. `vertebrae_L5`).
pub fn vertebrae_pp(
    data: &mut [u8],
    dims: [usize; 3],
    spacing: [f64; 3],
    s_axis: usize,
    classes: &[&str],
) {
    const MIN_SIZE_MM3: f64 = 100.0;
    const DILATION_MM: f64 = 3.0;
    let all: Vec<u8> = (1..=classes.len() as u8).collect();
    if !labels_touch(data, dims) {
        let out = dilate_labels(data, dims, spacing, DILATION_MM, &all);
        data.copy_from_slice(&out);
        return;
    }
    let voxel_mm3 = spacing[0] * spacing[1] * spacing[2];
    let mask: Vec<bool> = data.iter().map(|&l| l != 0).collect();
    let (id, sizes) = components(&mask, dims);
    let keep: Vec<bool> = sizes
        .iter()
        .enumerate()
        .map(|(c, &n)| c != 0 && n as f64 * voxel_mm3 >= MIN_SIZE_MM3)
        .collect();
    let mut present = [false; 256];
    for (l, &c) in data.iter().zip(&id) {
        if c != 0 && keep[c as usize] && (*l as usize) <= classes.len() && *l != 0 {
            present[*l as usize] = true;
        }
    }
    let present: Vec<u8> = (1..=255u8).filter(|l| present[*l as usize]).collect();
    if present.is_empty() {
        data.fill(0);
        return;
    }
    let label_of = |name: &str| classes.iter().position(|c| *c == name).map(|i| i as u8 + 1);
    let c1 = label_of("vertebrae_C1").unwrap_or(1);
    let l5 = label_of("vertebrae_L5").unwrap_or(classes.len() as u8);
    let l5_in_image = present.contains(&l5);
    let from_top = !l5_in_image && present.contains(&c1);
    // Centre of every kept component along the head-foot axis.
    let [_, d1, d2] = dims;
    let mut sum = vec![0f64; sizes.len()];
    for (v, &c) in id.iter().enumerate() {
        if c != 0 && keep[c as usize] {
            let coord = match s_axis {
                0 => v / (d1 * d2),
                1 => (v / d2) % d1,
                _ => v % d2,
            };
            sum[c as usize] += coord as f64;
        }
    }
    let mut centres: Vec<(usize, f64)> = (1..sizes.len())
        .filter(|c| keep[*c])
        .map(|c| (c, sum[c] / sizes[c] as f64))
        .collect();
    let labels: Vec<u8> = if from_top {
        // Stable, so equal centres keep component order as Python's does.
        centres.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        (c1..=classes.len() as u8).collect()
    } else {
        centres.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let top = *present.last().expect("non-empty");
        (1..=top).rev().collect()
    };
    let mut assign = vec![0u8; sizes.len()];
    for ((c, _), l) in centres.iter().zip(labels) {
        assign[*c] = l;
    }
    let relabelled: Vec<u8> = id.iter().map(|&c| assign[c as usize]).collect();
    let out = dilate_labels(&relabelled, dims, spacing, DILATION_MM, &all);
    data.copy_from_slice(&out);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn components_are_six_connected_and_numbered_in_raster_order() {
        // Two voxels touching only on an edge are two components.
        let dims = [1, 3, 3];
        let mut m = vec![false; 9];
        m[0] = true; // (0,0,0)
        m[4] = true; // (0,1,1)
        m[5] = true; // (0,1,2)
        let (id, sizes) = components(&m, dims);
        assert_eq!(sizes, vec![0, 1, 2]);
        assert_eq!((id[0], id[4], id[5], id[1]), (1, 2, 2, 0));
    }

    #[test]
    fn the_largest_trunk_piece_survives_and_small_limbs_go() {
        let dims = [1, 1, 10];
        let classes = ["body_trunc", "body_extremities"];
        let mut d = vec![1, 1, 1, 0, 1, 0, 2, 0, 2, 2];
        body(&mut d, dims, [10.0, 10.0, 10.0], &classes);
        // Trunk: the three-voxel piece stays, the single voxel goes. Limbs:
        // 50 cm3 is 50 voxels of 1 cm3, so both pieces go.
        assert_eq!(d, vec![1, 1, 1, 0, 0, 0, 0, 0, 0, 0]);
        let mut d = vec![2; 10];
        body(&mut d, dims, [40.0, 40.0, 40.0], &classes);
        assert_eq!(d, vec![2; 10]);
    }

    #[test]
    fn touching_vertebrae_are_renumbered_from_the_bottom_up() {
        // A column of three bodies along axis 0 (axis 0 = superior), two
        // voxels apart, the middle one wrongly labelled with two levels.
        let classes: Vec<&str> = [
            "vertebrae_C1",
            "vertebrae_C2",
            "vertebrae_C3",
            "vertebrae_C4",
            "vertebrae_C5",
            "vertebrae_C6",
            "vertebrae_C7",
            "vertebrae_T1",
            "vertebrae_T2",
            "vertebrae_T3",
            "vertebrae_T4",
            "vertebrae_T5",
            "vertebrae_T6",
            "vertebrae_T7",
            "vertebrae_T8",
            "vertebrae_T9",
            "vertebrae_T10",
            "vertebrae_T11",
            "vertebrae_T12",
            "vertebrae_L1",
            "vertebrae_L2",
            "vertebrae_L3",
            "vertebrae_L4",
            "vertebrae_L5",
        ]
        .to_vec();
        let dims = [12, 1, 1];
        let mut d = vec![0u8; 12];
        // rows 0-1: L5 (24); rows 4-5: 23 and 22 touching; rows 8-9: 21
        d[0] = 24;
        d[1] = 24;
        d[4] = 23;
        d[5] = 22;
        d[8] = 21;
        d[9] = 21;
        // 10 mm voxels: every piece is >= 100 mm3; the 3 mm growth is less
        // than a voxel, so nothing grows.
        vertebrae_pp(&mut d, dims, [10.0, 10.0, 10.0], 0, &classes);
        assert_eq!(d, vec![24, 24, 0, 0, 23, 23, 0, 0, 22, 22, 0, 0]);
    }

    #[test]
    fn labels_grow_into_the_background_only() {
        let dims = [1, 1, 7];
        let d = vec![0, 0, 1, 0, 0, 2, 0];
        // 1.5 mm voxels, 3 mm: two voxels each way along the line.
        let out = dilate_labels(&d, dims, [1.5, 1.5, 1.5], 3.0, &[1, 2]);
        assert_eq!(out, vec![1, 1, 1, 1, 1, 2, 2]);
    }
}
