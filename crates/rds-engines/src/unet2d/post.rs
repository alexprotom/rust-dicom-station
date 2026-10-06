//! lungmask's volume post-processing (`utils.postprocessing`), on the stack
//! of 256 x 256 slice answers before they go back into their boxes.
//!
//! Every connected piece of every label (26-connected, as scikit-image's
//! `label` connects by default) that is not the largest piece of its label
//! is handed to the neighbouring piece it shares the most border with,
//! smallest pieces first; pieces under three voxels are dropped instead.
//! Then each label keeps its largest 26-connected piece and has its
//! enclosed holes filled (`fill_voids`: background shut off from the edge
//! of the volume through face-connected background).
//!
//! [`postprocess_spare`] is the same with *spare* labels, lungmask's label
//! fusion (`LTRCLobes_R231`): pieces of a spare label are always handed to
//! a neighbour, and whatever stays spare is cleared at the end.

use std::collections::{HashMap, HashSet};

/// Pieces this small are not merged, only dropped (`skip_below`).
const SKIP_BELOW: usize = 3;

fn index(d: [usize; 3], z: usize, y: usize, x: usize) -> usize {
    (z * d[1] + y) * d[2] + x
}

/// The 13 neighbours of the 26 that come earlier in raster order.
fn back_offsets() -> Vec<[isize; 3]> {
    let mut v = Vec::new();
    for dz in -1isize..=1 {
        for dy in -1isize..=1 {
            for dx in -1isize..=1 {
                let earlier = dz < 0 || (dz == 0 && (dy < 0 || (dy == 0 && dx < 0)));
                if earlier {
                    v.push([dz, dy, dx]);
                }
            }
        }
    }
    v
}

fn find(parent: &mut [u32], mut a: u32) -> u32 {
    while parent[a as usize] != a {
        let p = parent[parent[a as usize] as usize];
        parent[a as usize] = p;
        a = p;
    }
    a
}

/// 26-connected pieces of equal non-zero value, numbered from 1 in raster
/// order of their first voxel.
pub fn label26(values: &[u8], d: [usize; 3]) -> (Vec<u32>, usize) {
    let n = values.len();
    // Provisional ids by union-find over the earlier neighbours.
    let mut prov = vec![0u32; n];
    let mut parent: Vec<u32> = vec![0];
    let offs = back_offsets();
    for z in 0..d[0] {
        for y in 0..d[1] {
            for x in 0..d[2] {
                let v = index(d, z, y, x);
                let val = values[v];
                if val == 0 {
                    continue;
                }
                let mut mine = 0u32;
                for o in &offs {
                    let (zz, yy, xx) = (z as isize + o[0], y as isize + o[1], x as isize + o[2]);
                    if zz < 0 || yy < 0 || xx < 0 || yy >= d[1] as isize || xx >= d[2] as isize {
                        continue;
                    }
                    let u = index(d, zz as usize, yy as usize, xx as usize);
                    if values[u] != val {
                        continue;
                    }
                    let other = find(&mut parent, prov[u]);
                    if mine == 0 {
                        mine = other;
                    } else if other != mine {
                        let (a, b) = (mine.min(other), mine.max(other));
                        parent[b as usize] = a;
                        mine = a;
                    }
                }
                if mine == 0 {
                    mine = parent.len() as u32;
                    parent.push(mine);
                }
                prov[v] = mine;
            }
        }
    }
    // Final ids in order of first appearance.
    let mut final_id = vec![0u32; parent.len()];
    let mut next = 0u32;
    let mut out = vec![0u32; n];
    for v in 0..n {
        if prov[v] == 0 {
            continue;
        }
        let root = find(&mut parent, prov[v]) as usize;
        if final_id[root] == 0 {
            next += 1;
            final_id[root] = next;
        }
        out[v] = final_id[root];
    }
    (out, next as usize)
}

/// Keep the largest 26-connected piece of `mask` (the last of equal size,
/// as `np.argsort(sizes)[-1]` picks for a short list).
fn keep_largest(mask: &[bool], d: [usize; 3]) -> Vec<bool> {
    let vals: Vec<u8> = mask.iter().map(|&m| m as u8).collect();
    let (id, n) = label26(&vals, d);
    if n == 0 {
        return mask.to_vec();
    }
    let mut size = vec![0usize; n + 1];
    for &k in &id {
        size[k as usize] += 1;
    }
    let mut best = 1;
    for k in 2..=n {
        if size[k] >= size[best] {
            best = k;
        }
    }
    id.iter().map(|&k| k as usize == best).collect()
}

/// `fill_voids.fill`: background not reachable from the volume's faces
/// through face-connected background becomes foreground.
pub fn fill_voids(mask: &[bool], d: [usize; 3]) -> Vec<bool> {
    let mut outside = vec![false; mask.len()];
    let mut stack = Vec::new();
    for z in 0..d[0] {
        for y in 0..d[1] {
            for x in 0..d[2] {
                let edge =
                    z == 0 || y == 0 || x == 0 || z + 1 == d[0] || y + 1 == d[1] || x + 1 == d[2];
                let v = index(d, z, y, x);
                if edge && !mask[v] {
                    outside[v] = true;
                    stack.push((z, y, x));
                }
            }
        }
    }
    while let Some((z, y, x)) = stack.pop() {
        let mut go = |zz: usize, yy: usize, xx: usize| {
            let u = index(d, zz, yy, xx);
            if !mask[u] && !outside[u] {
                outside[u] = true;
                stack.push((zz, yy, xx));
            }
        };
        if z > 0 {
            go(z - 1, y, x);
        }
        if z + 1 < d[0] {
            go(z + 1, y, x);
        }
        if y > 0 {
            go(z, y - 1, x);
        }
        if y + 1 < d[1] {
            go(z, y + 1, x);
        }
        if x > 0 {
            go(z, y, x - 1);
        }
        if x + 1 < d[2] {
            go(z, y, x + 1);
        }
    }
    outside.iter().map(|o| !o).collect()
}

/// `utils.postprocessing(label_image)` with no spare labels.
pub fn postprocess(labels: &[u8], d: [usize; 3]) -> Vec<u8> {
    postprocess_spare(labels, d, &[])
}

/// `utils.postprocessing(label_image, spare)`: as [`postprocess`], but every
/// piece of a `spare` label is handed to a neighbour (not only the pieces
/// smaller than the label's largest), and what is still spare afterwards
/// is cleared.
///
/// Upstream skips a neighbour when `n not in spare`, where `n` is the
/// neighbour's *piece number*, not its label - so the piece numbered like
/// the spare label is the one never chosen, and a small lobe piece can be
/// handed to a spare piece (and cleared with it). That is reproduced as
/// it is: the point is the reference's answer.
pub fn postprocess_spare(labels: &[u8], d: [usize; 3], spare: &[u8]) -> Vec<u8> {
    let is_spare = |l: u8| spare.contains(&l);
    let (mut rm, n) = label26(labels, d);
    if n == 0 {
        return labels.to_vec();
    }
    // regionprops: area, value (intensity_max), voxels.
    let mut area = vec![0usize; n + 1];
    let mut value = vec![0u8; n + 1];
    let mut voxels: Vec<Vec<u32>> = vec![Vec::new(); n + 1];
    for (v, &k) in rm.iter().enumerate() {
        if k != 0 {
            area[k as usize] += 1;
            value[k as usize] = labels[v];
            voxels[k as usize].push(v as u32);
        }
    }
    // Sorted by area, ties in id order (Python's sort is stable).
    let mut order: Vec<usize> = (1..=n).collect();
    order.sort_by_key(|&k| area[k]);
    let mut maxsub = [0usize; 256];
    let mut lobemap = vec![0u8; n + 1];
    for &k in &order {
        let lab = value[k] as usize;
        if area[k] > maxsub[lab] {
            maxsub[lab] = area[k];
            lobemap[k] = value[k];
        }
    }
    for &k in &order {
        let lab = value[k] as usize;
        if !((area[k] < maxsub[lab] || is_spare(value[k])) && area[k] >= SKIP_BELOW) {
            continue;
        }
        // The piece's neighbours through one face-dilation, counted.
        let mut counts: HashMap<u32, usize> = HashMap::new();
        let mut seen: HashSet<u32> = HashSet::new();
        for &v in &voxels[k] {
            let v = v as usize;
            let (z, y, x) = (v / (d[1] * d[2]), (v / d[2]) % d[1], v % d[2]);
            let mut look = |u: usize| {
                if seen.insert(u as u32) {
                    *counts.entry(rm[u]).or_insert(0) += 1;
                }
            };
            look(v);
            if z > 0 {
                look(v - d[1] * d[2]);
            }
            if z + 1 < d[0] {
                look(v + d[1] * d[2]);
            }
            if y > 0 {
                look(v - d[2]);
            }
            if y + 1 < d[1] {
                look(v + d[2]);
            }
            if x > 0 {
                look(v - 1);
            }
            if x + 1 < d[2] {
                look(v + 1);
            }
        }
        let mut ids: Vec<u32> = counts.keys().copied().collect();
        ids.sort_unstable();
        let mut mapto = k as u32;
        let mut maxmap = 0usize;
        let mut myarea = 0usize;
        for nb in ids {
            let c = counts[&nb];
            let numbered_spare = spare.iter().any(|&s| u32::from(s) == nb);
            if nb != 0 && nb != k as u32 && c > maxmap && !numbered_spare {
                maxmap = c;
                mapto = nb;
                myarea = area[k];
            }
        }
        if mapto as usize == k {
            continue;
        }
        let moved = std::mem::take(&mut voxels[k]);
        for &v in &moved {
            rm[v as usize] = mapto;
        }
        let t = mapto as usize;
        voxels[t].extend(moved);
        let tl = value[t] as usize;
        if area[t] == maxsub[tl] {
            maxsub[tl] += myarea;
        }
        area[t] += myarea;
    }
    let mapped: Vec<u8> = rm
        .iter()
        .map(|&k| lobemap[k as usize])
        .map(|l| if is_spare(l) { 0 } else { l })
        .collect();
    let mut present = [false; 256];
    for &m in &mapped {
        present[m as usize] = true;
    }
    let mut out = vec![0u8; labels.len()];
    for lab in 1..=255u8 {
        if !present[lab as usize] {
            continue;
        }
        let mask: Vec<bool> = mapped.iter().map(|&m| m == lab).collect();
        let filled = fill_voids(&keep_largest(&mask, d), d);
        for (o, f) in out.iter_mut().zip(filled) {
            if f {
                *o = lab;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pieces_connect_through_corners_and_number_in_raster_order() {
        // 1 x 3 x 3: two diagonal voxels of 1 are one piece; a 2 elsewhere.
        let d = [1, 3, 3];
        let v = [1u8, 0, 2, 0, 1, 0, 0, 0, 0];
        let (id, n) = label26(&v, d);
        assert_eq!(n, 2);
        assert_eq!((id[0], id[4], id[2]), (1, 1, 2));
    }

    #[test]
    fn a_stray_piece_joins_its_neighbour_and_holes_close() {
        // A 1 x 6 x 6 slab: a big block of 1, a 3-voxel strip of 2 touching
        // it (merged into 1), a lone 2 elsewhere that is the largest 2
        // (kept), and a hole in the block (filled).
        let d = [3, 6, 6];
        let mut v = vec![0u8; 3 * 36];
        for z in 0..3 {
            for y in 1..5 {
                for x in 0..4 {
                    v[index(d, z, y, x)] = 1;
                }
            }
        }
        v[index(d, 1, 2, 1)] = 0; // the hole
        for z in 0..3 {
            v[index(d, z, 1, 4)] = 2; // strip of 3, touching the block
        }
        for z in 0..3 {
            for y in 3..6 {
                v[index(d, z, y, 5)] = 2; // the largest 2: 9 voxels
            }
        }
        let out = postprocess(&v, d);
        assert_eq!(out[index(d, 1, 2, 1)], 1, "hole filled");
        assert_eq!(out[index(d, 0, 1, 4)], 1, "strip merged into the block");
        assert_eq!(out[index(d, 0, 4, 5)], 2, "largest 2 kept");
    }

    /// 1 x 4 x 8: label 1 on the left (x 0-1), label 2 on the right (x
    /// 4-5), a `spare` strip at x = 3 touching only label 2 (x = 2 is
    /// empty), and a lone `spare` piece at x = 7 touching nothing. Pieces
    /// number 1 (label 1), 2 (strip), 3 (label 2), 4 (lone).
    fn fused_case(spare: u8) -> Vec<u8> {
        let d = [1, 4, 8];
        let mut v = vec![0u8; 32];
        for y in 0..4 {
            v[index(d, 0, y, 0)] = 1;
            v[index(d, 0, y, 1)] = 1;
            v[index(d, 0, y, 3)] = spare;
            v[index(d, 0, y, 4)] = 2;
            v[index(d, 0, y, 5)] = 2;
        }
        for y in 0..3 {
            v[index(d, 0, y, 7)] = spare;
        }
        v
    }

    #[test]
    fn spare_pieces_go_to_their_neighbour_and_the_rest_is_cleared() {
        let d = [1, 4, 8];
        let v = fused_case(6);
        let out = postprocess_spare(&v, d, &[6]);
        assert!(
            (0..4).all(|y| out[index(d, 0, y, 3)] == 2),
            "strip joins label 2"
        );
        assert!(
            (0..3).all(|y| out[index(d, 0, y, 7)] == 0),
            "lone spare cleared"
        );
        assert!(!out.contains(&6));
        // Without the spare list the strip is the largest 6 and stays.
        assert!(postprocess(&v, d).contains(&6));
    }

    #[test]
    fn the_piece_numbered_like_the_spare_label_is_never_a_target() {
        // Spare label 3: piece 3 is label 2, the strip's only neighbour, so
        // upstream leaves the strip spare and clears it.
        let d = [1, 4, 8];
        let out = postprocess_spare(&fused_case(3), d, &[3]);
        assert!((0..4).all(|y| out[index(d, 0, y, 3)] == 0));
        assert!((0..4).all(|y| out[index(d, 0, y, 4)] == 2));
    }

    #[test]
    fn voids_open_to_a_face_stay_empty() {
        let d = [3, 3, 3];
        let mut m = vec![true; 27];
        m[index(d, 1, 1, 1)] = false;
        assert!(fill_voids(&m, d)[index(d, 1, 1, 1)]);
        m[index(d, 0, 1, 1)] = false;
        assert!(!fill_voids(&m, d)[index(d, 1, 1, 1)]);
    }
}
