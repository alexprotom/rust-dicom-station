//! The naive reference kernels against the fixture PyTorch wrote.
//!
//! `tests/data/medsam2-ops.safetensors` was produced by PyTorch, PIL and
//! SAM 2 themselves (a PyTorch script the 0.8 releases carried in `tools/`).
//! `tests/common/ops_ref.rs` re-derives every `y` in it from the fixture's
//! own inputs; agreement here is what lets `examples/gen_ops_fixtures.rs`
//! regenerate the file without Python and still stand for the frameworks'
//! semantics, and what keeps the kernels in `src/medsam2` checked against
//! something that is not themselves.

#[path = "common/ops_ref.rs"]
mod ops_ref;

use std::collections::HashMap;
use std::path::Path;

use ops_ref::*;
use rust_dicom_station::nn::cache::{load_safetensors, WTensor};

fn fixture() -> HashMap<String, WTensor> {
    load_safetensors(Path::new("tests/data/medsam2-ops.safetensors")).expect("the op fixtures")
}

fn get<'a>(f: &'a HashMap<String, WTensor>, key: &str) -> &'a WTensor {
    f.get(key).unwrap_or_else(|| panic!("fixture {key}"))
}

/// Assert one recomputed tensor against the fixture, to `tol`.
fn check(f: &HashMap<String, WTensor>, key: &str, got: WTensor, tol: f64) {
    let want = get(f, key);
    let diff = max_abs_diff(&got, want);
    assert!(diff <= tol, "{key}: max |diff| {diff:e} > {tol:e}");
}

#[test]
fn the_convolutions_and_pooling_reproduce_pytorch() {
    let f = fixture();
    for (name, stride, pad, groups) in [
        ("conv_k7s4p3", 4, 3, 1),
        ("conv_k3s2p1", 2, 1, 1),
        ("conv_k1", 1, 0, 1),
        ("conv_dw", 1, 3, 6),
    ] {
        let y = conv2d(
            get(&f, &format!("{name}.x")),
            get(&f, &format!("{name}.w")),
            get(&f, &format!("{name}.b")),
            stride,
            pad,
            groups,
        );
        check(&f, &format!("{name}.y"), y, 2e-5);
    }
    let y = conv_transpose2d(
        get(&f, "convt_k2s2.x"),
        get(&f, "convt_k2s2.w"),
        get(&f, "convt_k2s2.b"),
        2,
    );
    check(&f, "convt_k2s2.y", y, 2e-5);
    check(&f, "maxpool2x2.y", maxpool2x2(get(&f, "maxpool2x2.x")), 0.0);
}

#[test]
fn the_interpolations_reproduce_pytorch_and_pil() {
    let f = fixture();
    check(
        &f,
        "interp_bilinear.y",
        interp_bilinear(get(&f, "interp_bilinear.x"), 20, 20),
        2e-6,
    );
    check(
        &f,
        "interp_nearest.y",
        interp_nearest2x(get(&f, "interp_nearest.x")),
        0.0,
    );
    check(
        &f,
        "interp_bicubic.y",
        interp_bicubic(get(&f, "interp_bicubic.x"), 32, 32),
        2e-6,
    );
    check(
        &f,
        "torch_bilinear_aa.y",
        interp_bilinear_aa(get(&f, "torch_bilinear_aa.x"), 4, 4),
        2e-6,
    );
    check(
        &f,
        "pil_up.y",
        pil_resize_f32(get(&f, "pil_up.x"), 16, 13),
        2e-6,
    );
    check(
        &f,
        "pil_down.y",
        pil_resize_f32(get(&f, "pil_down.x"), 12, 12),
        2e-6,
    );
    // The byte path is exact: fixed-point taps and a rounded, clipped byte
    // after each pass, so the resized bytes have to match one for one.
    let (pil, y) = preprocess(get(&f, "preprocess.u8"), 64);
    check(&f, "preprocess.pil_u8", pil, 0.0);
    check(&f, "preprocess.y", y, 2e-6);
}

#[test]
fn the_pointwise_and_normalization_ops_reproduce_pytorch() {
    let f = fixture();
    check(&f, "gelu.y", gelu(get(&f, "gelu.x")), 2e-6);
    check(&f, "relu.y", relu(get(&f, "relu.x")), 0.0);
    check(&f, "sigmoid.y", sigmoid(get(&f, "sigmoid.x")), 2e-6);
    check(
        &f,
        "layernorm_last.y",
        layer_norm_last(
            get(&f, "layernorm_last.x"),
            get(&f, "layernorm_last.w"),
            get(&f, "layernorm_last.b"),
            1e-6,
        ),
        2e-5,
    );
    check(
        &f,
        "layernorm2d.y",
        layer_norm_2d(
            get(&f, "layernorm2d.x"),
            get(&f, "layernorm2d.w"),
            get(&f, "layernorm2d.b"),
            1e-6,
        ),
        2e-5,
    );
    check(&f, "softmax.y", softmax_last(get(&f, "softmax.x")), 2e-6);
    check(
        &f,
        "matmul.y",
        matmul(get(&f, "matmul.a"), get(&f, "matmul.b")),
        2e-5,
    );
    check(
        &f,
        "sdpa.y",
        sdpa(get(&f, "sdpa.q"), get(&f, "sdpa.k"), get(&f, "sdpa.v")),
        2e-5,
    );
}

#[test]
fn the_positional_encodings_reproduce_sam2() {
    let f = fixture();
    check(&f, "pe_sine.y", pe_sine(8, 10000.0, 3, 4), 2e-6);
    let g = get(&f, "pe_random.gaussian");
    check(&f, "pe_random.dense", pe_random_dense(g, 3, 4), 2e-6);
    check(
        &f,
        "pe_random.y",
        pe_random_coords(g, get(&f, "pe_random.coords"), (64, 64)),
        2e-6,
    );
    let (re, im) = axial_cis(16, 4, 3, 10000.0);
    check(&f, "rope.freqs_real", re.clone(), 2e-6);
    check(&f, "rope.freqs_imag", im.clone(), 2e-6);
    check(&f, "rope.q_out", rope(get(&f, "rope.q"), &re, &im), 2e-5);
    check(&f, "rope.k_out", rope(get(&f, "rope.k"), &re, &im), 2e-5);
    check(
        &f,
        "rope_repeat.k_out",
        rope(get(&f, "rope_repeat.k"), &re, &im),
        2e-5,
    );
    check(
        &f,
        "rope_repeat.q_out",
        rope(get(&f, "rope.q"), &re, &im),
        2e-5,
    );
}

/// The error function the GELU rests on, against tabulated values.
#[test]
fn erf_is_accurate_on_both_branches() {
    for (x, want) in [
        (0.0, 0.0),
        (0.5, 0.520_499_877_813_046_5),
        (1.0, 0.842_700_792_949_714_9),
        (2.0, 0.995_322_265_018_952_7),
        (2.5, 0.999_593_047_982_555),
        (3.0, 0.999_977_909_503_001_4),
        (4.0, 0.999_999_984_582_742_1),
    ] {
        assert!((erf(x) - want).abs() < 1e-12, "erf({x}) = {}", erf(x));
        assert!((erf(-x) + want).abs() < 1e-12);
    }
}

/// The generator writes the same inventory PyTorch did - every key, every
/// shape - and its own outputs are what its kernels say, so a regenerated
/// file is a drop-in replacement.
#[test]
fn the_generator_reproduces_the_fixture_inventory() {
    let want = fixture();
    let got = generate(1);
    let mut missing: Vec<&String> = want.keys().filter(|k| !got.contains_key(*k)).collect();
    missing.sort();
    assert!(missing.is_empty(), "missing {missing:?}");
    let mut extra: Vec<&String> = got.keys().filter(|k| !want.contains_key(*k)).collect();
    extra.sort();
    assert!(extra.is_empty(), "extra {extra:?}");
    for (k, v) in &want {
        assert_eq!(v.shape, got[k].shape, "{k}");
        assert!(got[k].data.iter().all(|x| x.is_finite()), "{k} finite");
    }
    // Deterministic from the seed, different across seeds.
    assert_eq!(generate(1)["conv_k1.x"].data, got["conv_k1.x"].data);
    assert_ne!(generate(2)["conv_k1.x"].data, got["conv_k1.x"].data);
    // The bytes of the preprocessing input are bytes.
    assert!(got["preprocess.u8"]
        .data
        .iter()
        .all(|v| *v >= 0.0 && *v <= 255.0 && v.fract() == 0.0));
}
