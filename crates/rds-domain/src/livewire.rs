//! Live-wire: a contour that snaps to the edge under the pointer.
//!
//! The idea is Mortensen and Barrett's *intelligent scissors* (1995), and it
//! is the one drawing tool that pays for itself immediately: click once on
//! the boundary of an organ, move the pointer along it, and the curve
//! between the two follows the image gradient rather than the straight line
//! the hand would have drawn.
//!
//! The machinery is a shortest path on the pixel graph. Every step from one
//! pixel to a neighbour costs
//!
//! ```text
//! l(p, q) = 0.43 · fZ(q) + 0.43 · fG(q) + 0.14 · fD(p, q)
//! ```
//!
//! where **fZ** is zero on a Laplacian zero crossing and one elsewhere (a
//! crossing is where an edge is, to sub-pixel accuracy), **fG** is the
//! inverted gradient magnitude (strong edges are cheap), and **fD** is the
//! direction term that keeps the path running *along* an edge instead of
//! hopping across it. The weights are the paper's.
//!
//! Two deliberate differences from the paper:
//!
//! * the local cost is multiplied by the length of the step (1 or √2), so a
//!   staircase of diagonals is not cheaper than the straight line it
//!   approximates;
//! * the search is bounded to a box around the anchor ([`REACH`]), because a
//!   planner never wants a path that wanders across the whole slice, and it
//!   keeps one click well under the frame budget.
//!
//! The interaction is one Dijkstra per *anchor*, not per frame: from a fixed
//! anchor the shortest-path tree over the whole neighbourhood is computed
//! once ([`Costs::tree`]), and the path to wherever the pointer happens to
//! be is then a walk up the parent pointers ([`Tree::path_to`]).
//!
//! Training is the other half of the tool. The costs above know nothing
//! about *which* edge is wanted: bone and air both have large gradients. So
//! every accepted segment feeds a histogram of the gradient magnitudes it
//! ran along ([`Costs::train`]), and afterwards an edge that looks like the
//! ones already accepted is cheaper than an equally strong edge that does
//! not.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// How far from the anchor the search may run, in pixels. A contour segment
/// longer than this is two clicks, which is what a hand does anyway.
pub const REACH: usize = 220;

const W_ZERO: f32 = 0.43;
const W_GRAD: f32 = 0.43;
const W_DIR: f32 = 0.14;

/// The number of histogram bins the training uses over the normalized
/// gradient magnitude.
const BINS: usize = 32;

/// One slice, pre-processed into everything the path cost needs.
pub struct Costs {
    pub w: usize,
    pub h: usize,
    /// Normalized gradient magnitude, 0-1 (1 = the strongest edge here).
    grad: Vec<f32>,
    /// Unit gradient direction per pixel.
    gx: Vec<f32>,
    gy: Vec<f32>,
    /// 0 on a Laplacian zero crossing, 1 elsewhere.
    zero: Vec<f32>,
    /// Accepted-edge histogram over `grad`; empty until something is
    /// trained. Normalized so that the most-visited bin is 1.
    trained: Vec<f32>,
}

impl Costs {
    /// Pre-process one slice. `img` is row-major, `w` by `h`, in whatever
    /// units the image carries; `window` is the display window as
    /// `(level, width)`, so the tool follows the edges the user can
    /// actually see rather than ones buried outside the window.
    pub fn new(img: &[f32], w: usize, h: usize, window: (f32, f32)) -> Costs {
        let n = w * h;
        let mut grad = vec![0.0f32; n];
        let mut gx = vec![0.0f32; n];
        let mut gy = vec![0.0f32; n];
        let mut zero = vec![1.0f32; n];
        if n == 0 || img.len() < n || w < 3 || h < 3 {
            return Costs {
                w,
                h,
                grad,
                gx,
                gy,
                zero,
                trained: Vec::new(),
            };
        }
        // The windowed image, 0-1: everything below sees the picture on the
        // screen, which is what "level and window dependent" means.
        let (lo, span) = (
            window.0 - window.1 * 0.5,
            if window.1.abs() < 1e-6 { 1.0 } else { window.1 },
        );
        let v: Vec<f32> = img[..n]
            .iter()
            .map(|&x| ((x - lo) / span).clamp(0.0, 1.0))
            .collect();
        // Sobel, and the Laplacian for the zero crossings.
        let at = |x: usize, y: usize| v[y * w + x];
        let mut lap = vec![0.0f32; n];
        let mut gmax = 1e-6f32;
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let sx = (at(x + 1, y - 1) + 2.0 * at(x + 1, y) + at(x + 1, y + 1))
                    - (at(x - 1, y - 1) + 2.0 * at(x - 1, y) + at(x - 1, y + 1));
                let sy = (at(x - 1, y + 1) + 2.0 * at(x, y + 1) + at(x + 1, y + 1))
                    - (at(x - 1, y - 1) + 2.0 * at(x, y - 1) + at(x + 1, y - 1));
                let m = (sx * sx + sy * sy).sqrt();
                let i = y * w + x;
                grad[i] = m;
                gmax = gmax.max(m);
                if m > 1e-9 {
                    gx[i] = sx / m;
                    gy[i] = sy / m;
                }
                lap[i] = at(x + 1, y) + at(x - 1, y) + at(x, y + 1) + at(x, y - 1) - 4.0 * at(x, y);
            }
        }
        for g in grad.iter_mut() {
            *g /= gmax;
        }
        // A zero crossing is a pixel whose Laplacian has a different sign
        // from a 4-neighbour's; of the two, the one nearer zero carries it.
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let i = y * w + x;
                let a = lap[i];
                for j in [i - 1, i + 1, i - w, i + w] {
                    let b = lap[j];
                    if a * b < 0.0 && a.abs() <= b.abs() {
                        zero[i] = 0.0;
                        break;
                    }
                }
                if a == 0.0 {
                    zero[i] = 0.0;
                }
            }
        }
        Costs {
            w,
            h,
            grad,
            gx,
            gy,
            zero,
            trained: Vec::new(),
        }
    }

    /// The gradient-magnitude term, with the training applied when there is
    /// any: an edge that looks like the ones already accepted is cheap even
    /// if a stronger edge runs beside it.
    fn f_grad(&self, i: usize) -> f32 {
        let g = self.grad[i];
        if self.trained.is_empty() {
            return 1.0 - g;
        }
        // Half the static cost, half what has been learned: a trained tool
        // prefers the kind of edge it was shown, but a strong edge it has
        // never seen is still cheaper than flat tissue - which is what keeps
        // training from turning into a trap when the user moves on to the
        // next organ.
        let b = ((g * BINS as f32) as usize).min(BINS - 1);
        0.5 * (1.0 - g) + 0.5 * (1.0 - self.trained[b])
    }

    /// The direction term of a step: cheap along an edge, dear across it.
    fn f_dir(&self, p: usize, q: usize, dx: f32, dy: f32) -> f32 {
        // D(p) is the gradient turned by 90°, i.e. the edge's own direction.
        let (dpx, dpy) = (self.gy[p], -self.gx[p]);
        let (dqx, dqy) = (self.gy[q], -self.gx[q]);
        let len = (dx * dx + dy * dy).sqrt().max(1e-6);
        let (lx, ly) = (dx / len, dy / len);
        // Orient the link so the first dot product is not negative; the
        // paper's L(p, q).
        let s = if dpx * lx + dpy * ly >= 0.0 {
            1.0
        } else {
            -1.0
        };
        let dp = (dpx * lx + dpy * ly) * s;
        let dq = (lx * dqx + ly * dqy) * s;
        (dp.clamp(-1.0, 1.0).acos() + dq.clamp(-1.0, 1.0).acos()) / std::f32::consts::PI
    }

    fn cost(&self, p: usize, q: usize, dx: f32, dy: f32) -> f32 {
        let step = (dx * dx + dy * dy).sqrt();
        step * (W_ZERO * self.zero[q] + W_GRAD * self.f_grad(q) + W_DIR * self.f_dir(p, q, dx, dy))
    }

    /// The shortest-path tree rooted at `from`, over a box of [`REACH`]
    /// pixels around it.
    pub fn tree(&self, from: [usize; 2]) -> Tree {
        let n = self.w * self.h;
        let mut parent = vec![u32::MAX; n];
        let mut dist = vec![f32::MAX; n];
        let mut done = vec![false; n];
        if self.w == 0 || self.h == 0 || from[0] >= self.w || from[1] >= self.h {
            return Tree {
                w: self.w,
                h: self.h,
                root: from,
                parent,
            };
        }
        let x0 = from[0].saturating_sub(REACH);
        let y0 = from[1].saturating_sub(REACH);
        let x1 = (from[0] + REACH).min(self.w - 1);
        let y1 = (from[1] + REACH).min(self.h - 1);
        let start = from[1] * self.w + from[0];
        dist[start] = 0.0;
        let mut heap: BinaryHeap<Node> = BinaryHeap::new();
        heap.push(Node {
            cost: 0.0,
            idx: start as u32,
        });
        while let Some(Node { idx, .. }) = heap.pop() {
            let p = idx as usize;
            if done[p] {
                continue;
            }
            done[p] = true;
            let (px, py) = (p % self.w, p / self.w);
            for (dx, dy) in [
                (-1i32, -1i32),
                (0, -1),
                (1, -1),
                (-1, 0),
                (1, 0),
                (-1, 1),
                (0, 1),
                (1, 1),
            ] {
                let qx = px as i32 + dx;
                let qy = py as i32 + dy;
                if qx < x0 as i32 || qy < y0 as i32 || qx > x1 as i32 || qy > y1 as i32 {
                    continue;
                }
                let q = qy as usize * self.w + qx as usize;
                if done[q] {
                    continue;
                }
                let d = dist[p] + self.cost(p, q, dx as f32, dy as f32);
                if d < dist[q] {
                    dist[q] = d;
                    parent[q] = p as u32;
                    heap.push(Node {
                        cost: d,
                        idx: q as u32,
                    });
                }
            }
        }
        Tree {
            w: self.w,
            h: self.h,
            root: from,
            parent,
        }
    }

    /// Learn from an accepted segment: what the edges the user keeps look
    /// like. The histogram is smoothed once, because a handful of pixels
    /// otherwise makes a comb of it.
    pub fn train(&mut self, path: &[[usize; 2]]) {
        if path.is_empty() {
            return;
        }
        let mut hist = if self.trained.len() == BINS {
            self.trained.clone()
        } else {
            vec![0.0f32; BINS]
        };
        // Existing knowledge fades slowly, so a tool used on a second organ
        // stops insisting on the first one's edges.
        for h in hist.iter_mut() {
            *h *= 0.6;
        }
        for p in path {
            if p[0] >= self.w || p[1] >= self.h {
                continue;
            }
            let g = self.grad[p[1] * self.w + p[0]];
            let b = ((g * BINS as f32) as usize).min(BINS - 1);
            hist[b] += 1.0;
        }
        let mut smooth = hist.clone();
        for b in 0..BINS {
            let lo = hist[b.saturating_sub(1)];
            let hi = hist[(b + 1).min(BINS - 1)];
            smooth[b] = 0.25 * lo + 0.5 * hist[b] + 0.25 * hi;
        }
        let max = smooth.iter().cloned().fold(0.0f32, f32::max).max(1e-6);
        for s in smooth.iter_mut() {
            *s /= max;
        }
        self.trained = smooth;
    }

    /// Whether anything has been trained yet.
    pub fn is_trained(&self) -> bool {
        !self.trained.is_empty()
    }

    /// Forget what was trained.
    pub fn untrain(&mut self) {
        self.trained.clear();
    }
}

/// The gradient ridges of a small piece of one slice, and the one thing
/// they are used for: pulling a curve onto them.
///
/// This is the *smart* half of smart interpolation. An interpolated contour
/// is a good guess at where the boundary runs between two drawn slices, and
/// it is usually a pixel or two off the boundary the image actually shows.
/// Moving every vertex along the curve's own normal to the strongest edge
/// within a short reach fixes that without changing the shape the
/// interpolation found.
pub struct Edges {
    w: usize,
    h: usize,
    /// Origin of the crop in the coordinates the ring is expressed in.
    origin: [f64; 2],
    grad: Vec<f32>,
}

impl Edges {
    /// `img` is a crop of the slice, row-major `w` by `h`, whose pixel
    /// (0, 0) is at `origin` in the ring's coordinates.
    pub fn new(img: &[f32], w: usize, h: usize, origin: [f64; 2], window: (f32, f32)) -> Edges {
        let n = w * h;
        let mut grad = vec![0.0f32; n];
        if n == 0 || img.len() < n || w < 3 || h < 3 {
            return Edges { w, h, origin, grad };
        }
        let span = if window.1.abs() < 1e-6 { 1.0 } else { window.1 };
        let lo = window.0 - window.1 * 0.5;
        let v: Vec<f32> = img[..n]
            .iter()
            .map(|&x| ((x - lo) / span).clamp(0.0, 1.0))
            .collect();
        let at = |x: usize, y: usize| v[y * w + x];
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let sx = (at(x + 1, y - 1) + 2.0 * at(x + 1, y) + at(x + 1, y + 1))
                    - (at(x - 1, y - 1) + 2.0 * at(x - 1, y) + at(x - 1, y + 1));
                let sy = (at(x - 1, y + 1) + 2.0 * at(x, y + 1) + at(x + 1, y + 1))
                    - (at(x - 1, y - 1) + 2.0 * at(x, y - 1) + at(x + 1, y - 1));
                grad[y * w + x] = (sx * sx + sy * sy).sqrt();
            }
        }
        Edges { w, h, origin, grad }
    }

    /// Bilinear gradient magnitude at a point of the ring's coordinates.
    fn grad_at(&self, p: [f64; 2]) -> f32 {
        let x = p[0] - self.origin[0];
        let y = p[1] - self.origin[1];
        if x < 0.0 || y < 0.0 || self.w < 2 || self.h < 2 {
            return 0.0;
        }
        let (x0, y0) = (x.floor() as usize, y.floor() as usize);
        if x0 + 1 >= self.w || y0 + 1 >= self.h {
            return 0.0;
        }
        let (fx, fy) = ((x - x0 as f64) as f32, (y - y0 as f64) as f32);
        let g = |i: usize, j: usize| self.grad[j * self.w + i];
        let a = g(x0, y0) * (1.0 - fx) + g(x0 + 1, y0) * fx;
        let b = g(x0, y0 + 1) * (1.0 - fx) + g(x0 + 1, y0 + 1) * fx;
        a * (1.0 - fy) + b * fy
    }

    /// Every vertex of a closed ring moved along the ring's normal to the
    /// strongest edge within `reach` pixels, then smoothed once.
    ///
    /// A vertex that finds nothing stays where it was: the whole point is
    /// to correct a guess, not to replace it with noise. Candidates nearer
    /// the original position are preferred slightly, so a weak edge under
    /// the curve wins over a strong one at arm's length.
    pub fn snap(&self, pts: &[[f64; 2]], reach: f64) -> Vec<[f64; 2]> {
        let n = pts.len();
        if n < 3 || reach <= 0.0 {
            return pts.to_vec();
        }
        let steps = (reach * 2.0).ceil().max(2.0) as i32;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let prev = pts[(i + n - 1) % n];
            let next = pts[(i + 1) % n];
            let (tx, ty) = (next[0] - prev[0], next[1] - prev[1]);
            let len = (tx * tx + ty * ty).sqrt();
            if len < 1e-9 {
                out.push(pts[i]);
                continue;
            }
            // The normal of the curve at this vertex.
            let (nx, ny) = (-ty / len, tx / len);
            let mut best = pts[i];
            let mut score = self.grad_at(pts[i]);
            for s in -steps..=steps {
                let d = reach * s as f64 / steps as f64;
                let c = [pts[i][0] + nx * d, pts[i][1] + ny * d];
                let near = 1.0 - 0.4 * (d.abs() / reach) as f32;
                let v = self.grad_at(c) * near;
                if v > score {
                    score = v;
                    best = c;
                }
            }
            out.push(best);
        }
        // One pass of averaging: snapping is per vertex, and neighbouring
        // vertices can pick different sides of a two-pixel-wide edge.
        let mut smooth = Vec::with_capacity(n);
        for i in 0..n {
            let a = out[(i + n - 1) % n];
            let b = out[i];
            let c = out[(i + 1) % n];
            smooth.push([
                0.25 * a[0] + 0.5 * b[0] + 0.25 * c[0],
                0.25 * a[1] + 0.5 * b[1] + 0.25 * c[1],
            ]);
        }
        smooth
    }
}

/// The shortest-path tree from one anchor.
pub struct Tree {
    pub w: usize,
    pub h: usize,
    pub root: [usize; 2],
    parent: Vec<u32>,
}

impl Tree {
    /// The path from the anchor to `to`, anchor first. Empty when `to` was
    /// never reached (outside the search box, or outside the image).
    pub fn path_to(&self, to: [usize; 2]) -> Vec<[usize; 2]> {
        if to[0] >= self.w || to[1] >= self.h {
            return Vec::new();
        }
        let mut out = Vec::new();
        let mut cur = to[1] * self.w + to[0];
        let root = self.root[1] * self.w + self.root[0];
        // A path is at most the box's perimeter times a small factor; the
        // bound only guards against a corrupted tree.
        for _ in 0..(4 * REACH * REACH).min(self.parent.len() * 2) + 8 {
            out.push([cur % self.w, cur / self.w]);
            if cur == root {
                out.reverse();
                return out;
            }
            let p = self.parent[cur];
            if p == u32::MAX {
                return Vec::new();
            }
            cur = p as usize;
        }
        Vec::new()
    }
}

/// Heap entry: smallest cost first, so the ordering is reversed.
struct Node {
    cost: f32,
    idx: u32,
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost && self.idx == other.idx
    }
}
impl Eq for Node {}
impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost
            .total_cmp(&self.cost)
            .then_with(|| other.idx.cmp(&self.idx))
    }
}
impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A disc of one value on a background of another, with a little noise.
    fn disc(w: usize, h: usize, r: f64, inside: f32, outside: f32) -> Vec<f32> {
        let mut img = vec![outside; w * h];
        let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
        for y in 0..h {
            for x in 0..w {
                let d = ((x as f64 - cx).powi(2) + (y as f64 - cy).powi(2)).sqrt();
                if d <= r {
                    img[y * w + x] = inside;
                }
                // Deterministic ripple, so the edge is not the only
                // structure in the picture.
                img[y * w + x] += 4.0 * ((x * 7 + y * 13) % 5) as f32;
            }
        }
        img
    }

    #[test]
    fn the_path_follows_the_edge_instead_of_the_straight_line() {
        let (w, h) = (128, 128);
        let r = 40.0;
        let img = disc(w, h, r, 900.0, 0.0);
        let costs = Costs::new(&img, w, h, (450.0, 1000.0));
        // Two points on the rim, a quarter of the circle apart.
        let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
        let on_rim = |deg: f64| {
            let a = deg.to_radians();
            [
                (cx + r * a.cos()).round() as usize,
                (cy + r * a.sin()).round() as usize,
            ]
        };
        let a = on_rim(0.0);
        let b = on_rim(90.0);
        let path = costs.tree(a).path_to(b);
        assert!(path.len() > 40, "path of {} points", path.len());
        assert_eq!(path.first().copied(), Some(a));
        assert_eq!(path.last().copied(), Some(b));
        // Every point of it sits on the rim, to within a pixel and a half.
        let worst = path
            .iter()
            .map(|p| {
                let d = ((p[0] as f64 - cx).powi(2) + (p[1] as f64 - cy).powi(2)).sqrt();
                (d - r).abs()
            })
            .fold(0.0f64, f64::max);
        assert!(worst < 2.0, "worst deviation {worst:.2} px");
        // And it really is the long way round, not the chord: the chord
        // would be about 57 points.
        assert!(path.len() > 55);
    }

    #[test]
    fn a_path_outside_the_reach_is_empty_rather_than_wrong() {
        let (w, h) = (64, 64);
        let img = vec![0.0f32; w * h];
        let costs = Costs::new(&img, w, h, (0.0, 100.0));
        let t = costs.tree([2, 2]);
        assert!(t.path_to([w, h]).is_empty());
        // Inside the image the tree always reaches, even on a flat picture.
        assert!(!t.path_to([40, 40]).is_empty());
    }

    #[test]
    fn snapping_pulls_a_ring_onto_the_edge_it_missed() {
        let (w, h) = (100, 100);
        let r = 30.0;
        let img = disc(w, h, r, 800.0, 0.0);
        let edges = Edges::new(&img, w, h, [0.0, 0.0], (400.0, 900.0));
        // A circle of the right centre and the wrong radius: what an
        // interpolation between two slices of a cone looks like.
        let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
        let ring: Vec<[f64; 2]> = (0..90)
            .map(|i| {
                let a = i as f64 / 90.0 * std::f64::consts::TAU;
                [cx + (r - 2.5) * a.cos(), cy + (r - 2.5) * a.sin()]
            })
            .collect();
        let before = ring
            .iter()
            .map(|p| (((p[0] - cx).powi(2) + (p[1] - cy).powi(2)).sqrt() - r).abs())
            .sum::<f64>()
            / ring.len() as f64;
        let snapped = edges.snap(&ring, 4.0);
        assert_eq!(snapped.len(), ring.len());
        let after = snapped
            .iter()
            .map(|p| (((p[0] - cx).powi(2) + (p[1] - cy).powi(2)).sqrt() - r).abs())
            .sum::<f64>()
            / snapped.len() as f64;
        assert!(after < 0.4 * before, "{after:.2} px off, was {before:.2}");
    }

    #[test]
    fn snapping_a_ring_on_a_flat_picture_leaves_it_where_it_was() {
        let (w, h) = (40, 40);
        let img = vec![100.0f32; w * h];
        let edges = Edges::new(&img, w, h, [0.0, 0.0], (100.0, 200.0));
        let ring: Vec<[f64; 2]> = (0..40)
            .map(|i| {
                let a = i as f64 / 40.0 * std::f64::consts::TAU;
                [20.0 + 10.0 * a.cos(), 20.0 + 10.0 * a.sin()]
            })
            .collect();
        let snapped = edges.snap(&ring, 3.0);
        for (a, b) in ring.iter().zip(&snapped) {
            assert!((a[0] - b[0]).abs() < 0.2 && (a[1] - b[1]).abs() < 0.2);
        }
    }

    #[test]
    fn training_makes_the_accepted_kind_of_edge_cheaper() {
        let (w, h) = (96, 96);
        let img = disc(w, h, 30.0, 600.0, 0.0);
        let mut costs = Costs::new(&img, w, h, (300.0, 800.0));
        let path = costs.tree([48, 18]).path_to([48, 78]);
        assert!(!path.is_empty());
        // The kind of edge the path ran along: the busiest histogram bin,
        // and one pixel that belongs to it.
        let bin_of = |c: &Costs, p: [usize; 2]| {
            ((c.grad[p[1] * w + p[0]] * BINS as f32) as usize).min(BINS - 1)
        };
        let mut hist = [0usize; BINS];
        for p in &path {
            hist[bin_of(&costs, *p)] += 1;
        }
        let modal = (0..BINS).max_by_key(|&b| hist[b]).expect("bins");
        let sample = *path
            .iter()
            .find(|p| bin_of(&costs, **p) == modal)
            .expect("a pixel in the busiest bin");
        let i = sample[1] * w + sample[0];
        let before = costs.f_grad(i);
        let flat_before = costs.f_grad(0);

        costs.train(&path);
        assert!(costs.is_trained());
        assert!(
            costs.f_grad(i) < before,
            "trained {} vs untrained {before}",
            costs.f_grad(i)
        );
        // A pixel of flat background is not the kind of edge that was
        // accepted, so training did not make it cheaper.
        assert!(costs.f_grad(0) >= flat_before - 1e-6);
        costs.untrain();
        assert!(!costs.is_trained());
        assert!((costs.f_grad(i) - before).abs() < 1e-6);
    }
}
