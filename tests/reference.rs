//! Numerical agreement with the reference implementation.
//!
//! These tests need the two files produced by
//! `tools/gen_reference_activations.py` — a randomly initialized SAM 2.1-T at
//! 512 and its activations. They are ~160 MB and are not committed, so every
//! test here is skipped (and says so) when `MEDSAM2_REF` does not point at
//! them:
//!
//! ```text
//! python3 tools/gen_reference_activations.py /tmp/ref
//! MEDSAM2_REF=/tmp/ref cargo test --release --test reference -- --nocapture
//! ```

use std::collections::HashMap;
use std::path::PathBuf;

use burn::tensor::{Device, Tensor};
use rust_dicom_station::medsam2::{config, hiera::Hiera, layout, neck::Neck, ops};
use rust_dicom_station::nn::cache::{load_safetensors, WTensor};
use rust_dicom_station::nn::params::Params;

type B = burn::backend::NdArray;

struct Reference {
    params: Params,
    acts: HashMap<String, WTensor>,
    device: Device<B>,
}

fn reference() -> Option<Reference> {
    reference_at("")
}

/// The propagation dump, which is a second model instance with its own
/// weights (see `propagation()` in the generator).
fn track_reference() -> Option<Reference> {
    reference_at("-track")
}

fn reference_at(suffix: &str) -> Option<Reference> {
    let stem = std::env::var("MEDSAM2_REF").ok()?;
    let weights = PathBuf::from(format!("{stem}{suffix}-weights.safetensors"));
    let acts = PathBuf::from(format!("{stem}{suffix}-acts.safetensors"));
    if !weights.is_file() || !acts.is_file() {
        eprintln!("skipping: {} not found", weights.display());
        return None;
    }
    Some(Reference {
        params: Params::new(load_safetensors(&weights).expect("reference weights")),
        acts: load_safetensors(&acts).expect("reference activations"),
        device: Default::default(),
    })
}

macro_rules! track_reference_or_skip {
    () => {
        match track_reference() {
            Some(r) => r,
            None => {
                eprintln!("skipping: set MEDSAM2_REF to a reference dump");
                return;
            }
        }
    };
}

macro_rules! reference_or_skip {
    () => {
        match reference() {
            Some(r) => r,
            None => {
                eprintln!("skipping: set MEDSAM2_REF to a reference dump");
                return;
            }
        }
    };
}

impl Reference {
    fn act<const D: usize>(&self, key: &str) -> Tensor<B, D> {
        let t = self
            .acts
            .get(key)
            .unwrap_or_else(|| panic!("activation {key}"));
        assert_eq!(t.shape.len(), D, "{key} has rank {}", t.shape.len());
        let mut shape = [1usize; D];
        shape.copy_from_slice(&t.shape);
        ops::from_slice(&t.data, shape, &self.device)
    }

    /// Worst relative deviation, in the same normalization the op-parity
    /// tests use.
    fn compare<const D: usize>(&self, got: Tensor<B, D>, key: &str) -> f32 {
        let want = self.act::<D>(key);
        assert_eq!(got.dims(), want.dims(), "{key}: shape");
        let g = ops::to_vec(got);
        let w = ops::to_vec(want);
        let mut worst = 0.0f32;
        for (a, b) in g.iter().zip(w.iter()) {
            worst = worst.max((a - b).abs() / (1.0 + b.abs()));
        }
        println!("  {key:28} worst relative deviation {worst:.3e}");
        worst
    }
}

#[test]
fn the_checkpoint_layout_matches_the_reference_state_dict() {
    let r = reference_or_skip!();
    let mut actual: Vec<layout::TensorInfo> = Vec::new();
    for key in r.params.keys() {
        let t = r.acts.get(key);
        assert!(t.is_none(), "activations and weights should not share keys");
        actual.push(layout::TensorInfo {
            name: key.to_string(),
            shape: Vec::new(),
            dtype: "f32",
        });
    }
    // shapes have to come from the params themselves
    let mut actual: Vec<layout::TensorInfo> = actual
        .into_iter()
        .map(|mut t| {
            let want = layout::expected();
            let shape = want.get(&t.name).cloned().unwrap_or_default();
            // prove the tensor really has that shape by fetching it
            if !shape.is_empty() {
                r.params
                    .get(&t.name, &shape)
                    .unwrap_or_else(|e| panic!("{e}"));
            }
            t.shape = shape;
            t
        })
        .collect();
    actual.sort_by(|a, b| a.name.cmp(&b.name));
    let problems = layout::problems(&actual);
    assert!(problems.is_empty(), "{problems:#?}");
    assert_eq!(r.params.len(), layout::TENSOR_COUNT);
    assert_eq!(r.params.elements(), layout::STATE_ELEMENTS);
}

#[test]
fn the_trunk_and_neck_reproduce_the_reference_features() {
    let r = reference_or_skip!();
    let trunk = Hiera::<B>::load(&r.params, &r.device).expect("build trunk");
    let neck = Neck::<B>::load(&r.params, &r.device).expect("build neck");

    let stages = trunk.forward(r.act::<4>("img"));
    assert_eq!(stages.len(), 4);
    let mut worst = 0.0f32;
    for (i, s) in stages.iter().enumerate() {
        worst = worst.max(r.compare(s.clone(), &format!("trunk.{i}")));
    }
    let levels = neck.forward(&stages);
    for (i, l) in levels.iter().enumerate() {
        worst = worst.max(r.compare(l.clone(), &format!("neck.{i}")));
    }
    worst = worst.max(r.compare(levels[2].clone(), "vision_features"));
    assert!(worst < 1e-3, "worst deviation {worst:e}");
}

#[test]
fn the_neck_position_encoding_matches_the_reference() {
    let r = reference_or_skip!();
    let pe = rust_dicom_station::medsam2::neck::sine_pos_embed::<B>(
        config::EMBED_GRID,
        config::EMBED_GRID,
        config::D_MODEL,
        &r.device,
    );
    assert!(r.compare(pe, "neck_pos.2") < 1e-5);
}

#[test]
fn the_high_resolution_projections_match_the_reference() {
    let r = reference_or_skip!();
    let decoder =
        rust_dicom_station::medsam2::decoder::MaskDecoder::<B>::load(&r.params, &r.device).unwrap();
    let [s0, s1] = decoder.project_high_res(r.act::<4>("neck.0"), r.act::<4>("neck.1"));
    assert!(r.compare(s0, "high_res_s0") < 1e-5);
    assert!(r.compare(s1, "high_res_s1") < 1e-5);
}

#[test]
fn the_prompt_encoder_matches_the_reference() {
    use rust_dicom_station::medsam2::prompt::{Point, PromptEncoder};
    let r = reference_or_skip!();
    let enc = PromptEncoder::<B>::load(&r.params, &r.device).unwrap();

    assert!(r.compare(enc.dense_pe(), "prompt_dense_pe") < 1e-5);

    // a box: two labelled corners plus the padding token the reference adds
    let coords = ops::to_vec(r.act::<3>("prompt_box_coords"));
    let corners = Point::box_corners(coords[0], coords[1], coords[2], coords[3]);
    let boxed = enc.encode(&corners, None);
    assert_eq!(boxed.sparse.dims(), [1, 3, 256]);
    assert!(r.compare(boxed.sparse, "prompt_sparse_box") < 1e-5);
    assert!(r.compare(boxed.dense, "prompt_dense_none") < 1e-5);

    // no prompt at all: one synthesized padding point, plus the appended one
    let empty = enc.encode(&Point::none(), None);
    assert_eq!(empty.sparse.dims(), [1, 2, 256]);
    assert!(r.compare(empty.sparse, "prompt_sparse_empty") < 1e-5);

    // a mask prompt, already downsampled to 128 x 128 by the reference
    let masked = enc.encode(&Point::none(), Some(r.act::<4>("mask_prompt_downsampled")));
    assert!(r.compare(masked.dense, "prompt_dense_mask") < 1e-5);
}

#[test]
fn the_sam_head_matches_the_reference() {
    use rust_dicom_station::medsam2::prompt::Point;
    use rust_dicom_station::medsam2::sam::SamHead;
    let r = reference_or_skip!();
    let head = SamHead::<B>::load(&r.params, &r.device).unwrap();
    let high_res = [r.act::<4>("high_res_s0"), r.act::<4>("high_res_s1")];
    let pix_feat = r.act::<4>("pix_feat");

    // ---- a box on a conditioning slice: single-mask output ---------------
    let coords = ops::to_vec(r.act::<3>("prompt_box_coords"));
    let corners = Point::box_corners(coords[0], coords[1], coords[2], coords[3]);
    assert!(!SamHead::<B>::use_multimask(corners.len()));
    let out = head.forward(pix_feat.clone(), &high_res, &corners, None, false);
    let mut worst = r.compare(out.low_res_multimasks, "sam_box.low_res_multimasks");
    worst = worst.max(r.compare(out.low_res_masks, "sam_box.low_res_masks"));
    worst = worst.max(r.compare(out.high_res_masks, "sam_box.high_res_masks"));
    worst = worst.max(r.compare(out.ious, "sam_box.ious"));
    worst = worst.max(r.compare(out.obj_ptr, "sam_box.obj_ptr"));
    worst = worst.max(r.compare(out.object_score_logits, "sam_box.object_score_logits"));

    // ---- a propagated slice: three masks, the best by predicted IoU ------
    assert!(SamHead::<B>::use_multimask(0));
    let out = head.forward(pix_feat, &high_res, &[], None, true);
    worst = worst.max(r.compare(out.low_res_multimasks, "sam_track.low_res_multimasks"));
    worst = worst.max(r.compare(out.low_res_masks, "sam_track.low_res_masks"));
    worst = worst.max(r.compare(out.high_res_masks, "sam_track.high_res_masks"));
    worst = worst.max(r.compare(out.ious, "sam_track.ious"));
    worst = worst.max(r.compare(out.obj_ptr, "sam_track.obj_ptr"));
    worst = worst.max(r.compare(out.object_score_logits, "sam_track.object_score_logits"));
    assert!(worst < 1e-3, "worst deviation {worst:e}");
}

#[test]
fn the_mask_decoder_reproduces_the_reference_masks() {
    use rust_dicom_station::medsam2::decoder::MaskDecoder;
    use rust_dicom_station::medsam2::prompt::{Point, PromptEncoder};
    use rust_dicom_station::medsam2::sam::SamHead;
    let r = reference_or_skip!();
    let enc = PromptEncoder::<B>::load(&r.params, &r.device).unwrap();
    let decoder = MaskDecoder::<B>::load(&r.params, &r.device).unwrap();
    let head = SamHead::<B>::load(&r.params, &r.device).unwrap();
    let high_res = [r.act::<4>("high_res_s0"), r.act::<4>("high_res_s1")];
    let pix_feat = r.act::<4>("pix_feat");

    // The object-score head is usually negative with random weights, so
    // `_forward_sam_heads` blanks the logits. Comparing the decoder directly
    // is what actually exercises the two-way transformer, the upscaling and
    // the hypernetworks.
    let coords = ops::to_vec(r.act::<3>("prompt_box_coords"));
    let corners = Point::box_corners(coords[0], coords[1], coords[2], coords[3]);
    let prompts = enc.encode(&corners, None);
    let decoded = decoder.forward(
        pix_feat.clone(),
        enc.dense_pe(),
        prompts.sparse,
        prompts.dense,
        &high_res,
    );
    let selected = decoder.select(decoded, false);
    let mut worst = r.compare(selected.masks, "dec_box.masks");
    worst = worst.max(r.compare(selected.ious, "dec_box.ious"));
    worst = worst.max(r.compare(selected.sam_tokens.clone(), "dec_box.tokens"));
    worst = worst.max(r.compare(selected.object_score_logits, "dec_box.obj_score"));
    let token = selected
        .sam_tokens
        .slice([0..1, 0..1, 0..256])
        .reshape([1, 256]);
    worst = worst.max(r.compare(head.project_obj_ptr(token), "dec_box.obj_ptr"));

    // and the tracking path: no prompt, three masks kept
    let prompts = enc.encode(&Point::none(), None);
    let decoded = decoder.forward(
        pix_feat,
        enc.dense_pe(),
        prompts.sparse,
        prompts.dense,
        &high_res,
    );
    let selected = decoder.select(decoded, true);
    worst = worst.max(r.compare(selected.masks, "dec_track.masks"));
    worst = worst.max(r.compare(selected.ious, "dec_track.ious"));
    worst = worst.max(r.compare(selected.sam_tokens, "dec_track.tokens"));
    assert!(worst < 1e-3, "worst deviation {worst:e}");
}

#[test]
fn the_memory_encoder_matches_the_reference() {
    use rust_dicom_station::medsam2::memory::MemoryEncoder;
    let r = reference_or_skip!();
    let enc = MemoryEncoder::<B>::load(&r.params, &r.device).unwrap();
    let pix_feat = r.act::<4>("pix_feat");

    // a varied mask with the object present: the downsampler and the fuser
    let mask = r.act::<4>("memenc.mask_rand");
    let hard = enc.encode(pix_feat.clone(), mask.clone(), true, true);
    let mut worst = r.compare(hard.features, "memenc.features_rand");
    worst = worst.max(r.compare(hard.pos, "memenc.pos"));
    let soft = enc.encode(pix_feat.clone(), mask, false, true);
    worst = worst.max(r.compare(soft.features, "memenc.features_rand_soft"));

    // and the absent-object path, which adds `no_obj_embed_spatial` on top
    let absent = enc.encode(pix_feat, r.act::<4>("memenc.mask_rand"), true, false);
    worst = worst.max(r.compare(absent.features, "memenc.features_absent"));
    assert!(worst < 1e-3, "worst deviation {worst:e}");
}

#[test]
fn the_memory_attention_matches_the_reference() {
    use rust_dicom_station::medsam2::memattn::MemoryAttention;
    let r = reference_or_skip!();
    let attn = MemoryAttention::<B>::load(&r.params, &r.device).unwrap();
    // the reference carries sequences as [tokens, batch, channels]
    let seq = |key: &str| r.act::<3>(key).swap_dims(0, 1);

    // three memory frames plus eight object-pointer tokens
    let out = attn.forward(
        seq("memattn.curr"),
        seq("memattn.curr_pos"),
        seq("memattn.memory"),
        seq("memattn.memory_pos"),
        8,
    );
    let mut worst = r.compare(out.swap_dims(0, 1), "memattn.out");

    // and one frame with no pointers at all — the first propagated slice
    let out = attn.forward(
        seq("memattn.curr"),
        seq("memattn.curr_pos"),
        seq("memattn1.memory"),
        seq("memattn1.memory_pos"),
        0,
    );
    worst = worst.max(r.compare(out.swap_dims(0, 1), "memattn1.out"));
    assert!(worst < 1e-3, "worst deviation {worst:e}");
}

#[test]
fn the_tracker_reproduces_the_reference_propagation() {
    use rust_dicom_station::medsam2::model::Medsam2;
    use rust_dicom_station::medsam2::prompt::Point;
    use rust_dicom_station::medsam2::track::{Prompt, Tracker};

    let r = track_reference_or_skip!();
    let model = Medsam2::<B>::load(&r.params, &r.device).expect("build the network");

    let frames = r.act::<4>("frames");
    let n = frames.dims()[0];
    let size = config::IMAGE_SIZE;
    let frame = |i: usize| frames.clone().slice([i..i + 1, 0..3, 0..size, 0..size]);
    let prompt_slice = ops::to_vec(r.act::<1>("prompt_frame"))[0] as usize;
    let b = ops::to_vec(r.act::<2>("box"));
    let prompt = Prompt::Points(Point::box_corners(b[0], b[1], b[2], b[3]).to_vec());

    // The prompted slice is encoded once and used by both passes, exactly as
    // `infer::propagate` does.
    let anchor = model.encode_slice(frame(prompt_slice));
    let mut worst = 0.0f32;
    // the reference yields `[objects, height, width]` per frame
    let as_ref_shape = |m: Tensor<B, 4>| m.reshape([1, size, size]);

    let mut forward = Tracker::new(&model, n);
    let out = forward.prompt(prompt_slice, &anchor, &prompt);
    worst = worst.max(r.compare(
        as_ref_shape(out.high_res_masks),
        &format!("fwd.{prompt_slice}"),
    ));
    for i in prompt_slice + 1..n {
        let feats = model.encode_slice(frame(i));
        let out = forward.track(i, &feats, false);
        worst = worst.max(r.compare(as_ref_shape(out.high_res_masks), &format!("fwd.{i}")));
    }

    let mut reverse = Tracker::new(&model, n);
    reverse.prompt(prompt_slice, &anchor, &prompt);
    for i in (0..prompt_slice).rev() {
        let feats = model.encode_slice(frame(i));
        let out = reverse.track(i, &feats, true);
        worst = worst.max(r.compare(as_ref_shape(out.high_res_masks), &format!("rev.{i}")));
    }
    assert!(worst < 1e-3, "worst deviation {worst:e}");
}
