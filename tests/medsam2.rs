//! The MedSAM2 engine, end to end, without a checkpoint.
//!
//! Every module is built from a synthetic state dict generated out of
//! [`layout::expected`] — which means these tests fail if a loader asks for a
//! key or a shape the derived inventory does not contain, and they run in CI
//! with no download and no network. What they do *not* check is arithmetic;
//! that is `tests/reference.rs`, which needs a reference dump.

use std::collections::HashMap;

use burn::tensor::{Device, Tensor};
use rust_dicom_station::medsam2::{
    config, hiera::Hiera, layout, memattn::MemoryAttention, memory::MemoryEncoder, neck::Neck,
    ops, prompt::Point, sam::SamHead,
};
use rust_dicom_station::nn::cache::WTensor;
use rust_dicom_station::nn::params::Params;

type B = burn::backend::NdArray;

/// A state dict with the right keys and shapes and made-up values.
///
/// Normalization gains start at one and everything else is small, so a
/// forward pass through twelve residual blocks stays finite — the point is to
/// exercise the plumbing, not to produce a mask.
fn synthetic_params() -> Params {
    let mut state = 0x2026_0825_u64;
    let mut noise = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.05
    };
    let mut tensors: HashMap<String, WTensor> = HashMap::new();
    for (key, shape) in layout::expected() {
        let n: usize = shape.iter().product();
        let gain = shape.len() == 1 && key.ends_with(".weight");
        let data = (0..n)
            .map(|_| if gain { 1.0 + noise() } else { noise() })
            .collect();
        tensors.insert(key, WTensor { shape, data });
    }
    Params::new(tensors)
}

fn device() -> Device<B> {
    Default::default()
}

#[test]
fn every_module_loads_from_the_derived_inventory() {
    let p = synthetic_params();
    let d = device();
    assert_eq!(p.len(), layout::TENSOR_COUNT);
    Hiera::<B>::load(&p, &d).expect("trunk");
    Neck::<B>::load(&p, &d).expect("neck");
    SamHead::<B>::load(&p, &d).expect("prompt encoder, decoder and pointer projection");
    MemoryEncoder::<B>::load(&p, &d).expect("memory encoder");
    MemoryAttention::<B>::load(&p, &d).expect("memory attention");
}

#[test]
fn a_missing_tensor_is_reported_by_name_rather_than_panicking() {
    let mut tensors: HashMap<String, WTensor> = HashMap::new();
    for (key, shape) in layout::expected() {
        if key == "image_encoder.trunk.blocks.7.attn.qkv.weight" {
            continue;
        }
        let n = shape.iter().product();
        tensors.insert(
            key,
            WTensor {
                shape,
                data: vec![0.0; n],
            },
        );
    }
    let err = match Hiera::<B>::load(&Params::new(tensors), &device()) {
        Ok(_) => panic!("a missing tensor must fail the build"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("blocks.7.attn.qkv.weight"), "{err}");
}

#[test]
fn a_slice_flows_through_the_engine_with_the_documented_shapes() {
    let p = synthetic_params();
    let d = device();
    let size = config::IMAGE_SIZE;
    let grid = config::EMBED_GRID;

    let trunk = Hiera::<B>::load(&p, &d).unwrap();
    let neck = Neck::<B>::load(&p, &d).unwrap();
    let head = SamHead::<B>::load(&p, &d).unwrap();
    let mem_enc = MemoryEncoder::<B>::load(&p, &d).unwrap();
    let mem_attn = MemoryAttention::<B>::load(&p, &d).unwrap();

    // ---- encode one slice ------------------------------------------------
    let slice: Tensor<B, 4> = Tensor::zeros([1, 3, size, size], &d);
    let stages = trunk.forward(slice);
    assert_eq!(stages[0].dims(), [1, 96, 128, 128]);
    assert_eq!(stages[1].dims(), [1, 192, 64, 64]);
    assert_eq!(stages[2].dims(), [1, 384, grid, grid]);
    assert_eq!(stages[3].dims(), [1, 768, 16, 16]);

    let levels = neck.forward(&stages);
    assert_eq!(levels.len(), config::FPN_LEVELS);
    assert_eq!(levels[2].dims(), [1, config::D_MODEL, grid, grid]);
    let high_res = head
        .decoder
        .project_high_res(levels[0].clone(), levels[1].clone());
    assert_eq!(high_res[0].dims(), [1, config::HIGH_RES_S0_CH, 128, 128]);
    assert_eq!(high_res[1].dims(), [1, config::HIGH_RES_S1_CH, 64, 64]);

    // ---- prompt it with a box -------------------------------------------
    let corners = Point::box_corners(100.0, 120.0, 300.0, 340.0);
    assert!(!SamHead::<B>::use_multimask(corners.len()));
    let out = head.forward(
        levels[2].clone(),
        &high_res,
        &corners,
        None,
        false,
    );
    assert_eq!(out.low_res_masks.dims(), [1, 1, config::LOW_RES, config::LOW_RES]);
    assert_eq!(out.high_res_masks.dims(), [1, 1, size, size]);
    assert_eq!(out.obj_ptr.dims(), [1, config::D_MODEL]);
    assert_eq!(out.object_score_logits.dims(), [1, 1]);
    assert!(ops::to_vec(out.low_res_masks.clone())
        .iter()
        .all(|v| v.is_finite()));

    // ---- turn the answer into a memory ----------------------------------
    let memory = mem_enc.encode(
        levels[2].clone(),
        out.high_res_masks.clone(),
        true,
        out.object_present(),
    );
    assert_eq!(memory.features.dims(), [1, config::MEM_DIM, grid, grid]);
    assert_eq!(memory.pos.dims(), [1, config::MEM_DIM, grid, grid]);

    // ---- and condition the next slice on two of them ---------------------
    let tokens = grid * grid;
    let flat = |t: Tensor<B, 4>, c: usize| t.reshape([1, c, tokens]).swap_dims(1, 2);
    let frame = flat(memory.features.clone(), config::MEM_DIM);
    let frame_pos = flat(memory.pos.clone(), config::MEM_DIM);
    // one object pointer, split into four 64-wide sub-tokens
    let ptr = out.obj_ptr.clone().reshape([1, config::PTR_TOKENS, config::MEM_DIM]);
    let bank = Tensor::cat(vec![frame.clone(), frame.clone(), ptr.clone()], 1);
    let bank_pos = Tensor::cat(
        vec![
            frame_pos.clone(),
            frame_pos,
            Tensor::zeros([1, config::PTR_TOKENS, config::MEM_DIM], &d),
        ],
        1,
    );
    assert_eq!(bank.dims(), [1, 2 * tokens + config::PTR_TOKENS, config::MEM_DIM]);

    let curr = flat(levels[2].clone(), config::D_MODEL);
    let curr_pos = flat(
        rust_dicom_station::medsam2::neck::sine_pos_embed::<B>(grid, grid, config::D_MODEL, &d),
        config::D_MODEL,
    );
    let conditioned = mem_attn.forward(curr, curr_pos, bank, bank_pos, config::PTR_TOKENS);
    assert_eq!(conditioned.dims(), [1, tokens, config::D_MODEL]);
    assert!(ops::to_vec(conditioned).iter().all(|v| v.is_finite()));
}

#[test]
fn the_engine_accepts_a_mask_prompt_at_either_resolution() {
    let p = synthetic_params();
    let d = device();
    let head = SamHead::<B>::load(&p, &d).unwrap();
    let full: Tensor<B, 4> = Tensor::zeros([1, 1, config::IMAGE_SIZE, config::IMAGE_SIZE], &d);
    assert_eq!(
        head.downsample_mask(full).dims(),
        [1, 1, config::MASK_PROMPT_SIZE, config::MASK_PROMPT_SIZE]
    );
    let already: Tensor<B, 4> = Tensor::zeros(
        [1, 1, config::MASK_PROMPT_SIZE, config::MASK_PROMPT_SIZE],
        &d,
    );
    assert_eq!(
        head.downsample_mask(already).dims(),
        [1, 1, config::MASK_PROMPT_SIZE, config::MASK_PROMPT_SIZE]
    );
}
