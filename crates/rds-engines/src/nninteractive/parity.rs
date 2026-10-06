//! The session against the reference session (`nninteractive` 2.6.0) on a
//! synthetic image, with one 3x3x3 convolution standing in for the network
//! in both (`tests/data/nninteractive-session.safetensors`, written by a
//! generator kept outside the repository). Every prompt kind, the AutoZoom
//! loop, refinement with the greedy cover and its random fallback, an
//! initial segmentation kept as it is and one refined whole.

use std::cell::RefCell;

use safetensors::SafeTensors;

use super::session::{Kind, Predictor, Session, Settings};
use crate::autoseg::cpu::{conv3d, Act};
use crate::progress::Progress;

const DIMS: [usize; 3] = [30, 40, 26];
const PATCH: [usize; 3] = [12, 16, 10];

struct Mock {
    w: Vec<f32>,
    b: Vec<f32>,
    /// Per pass: per input channel, the sum and the sum of magnitudes.
    calls: RefCell<Vec<Vec<[f64; 2]>>>,
}

impl Predictor for Mock {
    fn predict(&self, input: &[f32], p: [usize; 3]) -> anyhow::Result<Vec<u8>> {
        let n = p[0] * p[1] * p[2];
        let c = input.len() / n;
        self.calls.borrow_mut().push(
            (0..c)
                .map(|ch| {
                    let s = &input[ch * n..(ch + 1) * n];
                    [
                        s.iter().map(|&v| f64::from(v)).sum(),
                        s.iter().map(|&v| f64::from(v).abs()).sum(),
                    ]
                })
                .collect(),
        );
        let x = Act {
            c,
            d: p[0],
            h: p[1],
            w: p[2],
            data: input.to_vec(),
        };
        let y = conv3d(&x, &self.w, &self.b, 2, [3, 3, 3], [1, 1, 1]).data;
        Ok((0..n).map(|v| u8::from(y[n + v] > y[v])).collect())
    }
}

fn f32s(t: &SafeTensors, name: &str) -> Vec<f32> {
    let v = t.tensor(name).unwrap();
    v.data()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn f64s(t: &SafeTensors, name: &str) -> (Vec<usize>, Vec<f64>) {
    let v = t.tensor(name).unwrap();
    (
        v.shape().to_vec(),
        v.data()
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect(),
    )
}

#[test]
fn the_session_matches_nninteractive() {
    let bytes = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/data/nninteractive-session.safetensors"
    ))
    .expect("tests/data/nninteractive-session.safetensors");
    let t = SafeTensors::deserialize(&bytes).unwrap();
    let image = f32s(&t, "image");
    let net = Mock {
        w: f32s(&t, "weight"),
        b: f32s(&t, "bias"),
        calls: RefCell::new(Vec::new()),
    };
    let settings = Settings {
        refine_margin: 2,
        ..Settings::v1(PATCH)
    };
    let mut s = Session::new(image, DIMS, settings).unwrap();
    let progress = Progress::default();

    let check = |s: &Session, step: usize| {
        let want = t.tensor(&format!("mask_{step}")).unwrap();
        let want = want.data();
        let differ = want.iter().zip(&s.mask).filter(|(a, b)| *a != *b).count();
        let calls = net.calls.replace(Vec::new());
        let (shape, ref_calls) = f64s(&t, &format!("calls_{step}"));
        assert_eq!(
            calls.len(),
            shape[0],
            "step {step}: {} network passes, the reference made {}",
            calls.len(),
            shape[0]
        );
        for (ci, call) in calls.iter().enumerate() {
            for (ch, got) in call.iter().enumerate() {
                for k in 0..2 {
                    let want = ref_calls[(ci * 8 + ch) * 2 + k];
                    let tol = 1e-4 * want.abs().max(1.0);
                    assert!(
                        (got[k] - want).abs() <= tol,
                        "step {step}, pass {ci}, channel {ch}: {} vs {want}",
                        got[k]
                    );
                }
            }
        }
        assert_eq!(differ, 0, "step {step}: {differ} voxels differ");
    };

    let predict = |s: &mut Session, force: bool| {
        s.predict(&net, force, &progress).unwrap();
    };

    s.add_point([15.0, 20.0, 13.0], true).unwrap();
    predict(&mut s, false);
    check(&s, 1);
    s.add_point([15.0, 31.0, 13.0], false).unwrap();
    predict(&mut s, false);
    check(&s, 2);
    s.add_box([8.0, 10.0, 7.0], [22.0, 34.0, 8.0], true)
        .unwrap();
    predict(&mut s, false);
    check(&s, 3);
    let mut scr = vec![0u8; 8 * 9];
    for i in 0..8 {
        scr[i * 9 + (i + 1).min(8)] = 1;
        scr[i * 9 + i.min(8)] = 1;
    }
    s.add_mask(Kind::Scribble, [1, 6, 16], [8, 1, 9], scr, true)
        .unwrap();
    predict(&mut s, false);
    check(&s, 4);
    let mut las = vec![0u8; 12 * 10];
    for r in 0..12usize {
        let w = r.min(11 - r).min(5);
        for c in 5 - w..5 + w {
            las[r * 10 + c] = 1;
        }
    }
    s.add_mask(Kind::Lasso, [20, 14, 8], [1, 12, 10], las, false)
        .unwrap();
    predict(&mut s, false);
    check(&s, 5);
    s.add_point([5.0, 6.0, 20.0], true).unwrap();
    predict(&mut s, false);
    check(&s, 6);
    let block = |lo: [usize; 3], hi: [usize; 3]| -> Vec<u8> {
        let mut m = vec![0u8; DIMS[0] * DIMS[1] * DIMS[2]];
        for i in lo[0]..hi[0] {
            for j in lo[1]..hi[1] {
                for k in lo[2]..hi[2] {
                    m[(i * DIMS[1] + j) * DIMS[2] + k] = 1;
                }
            }
        }
        m
    };
    s.set_initial(&block([10, 12, 8], [20, 28, 18]), false)
        .unwrap();
    check(&s, 7);
    s.add_point([14.0, 19.0, 12.0], true).unwrap();
    predict(&mut s, false);
    check(&s, 8);
    s.set_initial(&block([4, 8, 5], [26, 33, 22]), true)
        .unwrap();
    predict(&mut s, true);
    check(&s, 9);
}
