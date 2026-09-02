//! SegVol integration tests.
//!
//! The fast tests run against `tests/data/segvol-tensors.csv` - the tensor
//! inventory of the real `pytorch_model.bin`, recorded by
//! `examples/segvol_probe`. That fixture lets the network assembly be
//! developed and checked against the exact published key names and shapes
//! without a 724 MB download, which is also what makes these tests runnable
//! in CI.
//!
//! The `#[ignore]`d test checks the actual checkpoint. Enable it locally with
//!
//! ```text
//! RDS_SEGVOL_MODEL=path/to/models/segvol \
//!   cargo test --release --test segvol -- --ignored
//! ```
//!
//! (the weights are downloaded into that folder on first use).

use rust_dicom_station::nn::pickle::Dtype;
use rust_dicom_station::segvol::layout::{self, Inventory, TensorInfo};

/// One row of the recorded inventory.
struct Row {
    name: String,
    dtype: Dtype,
    shape: Vec<usize>,
}

fn recorded() -> Vec<Row> {
    let csv = include_str!("data/segvol-tensors.csv");
    let mut rows = Vec::new();
    for (i, line) in csv.lines().enumerate().skip(1) {
        if line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        assert_eq!(f.len(), 4, "line {}: {line}", i + 1);
        let dtype = match f[1] {
            "F32" => Dtype::F32,
            "F64" => Dtype::F64,
            "F16" => Dtype::F16,
            "I64" => Dtype::I64,
            "I32" => Dtype::I32,
            other => panic!("line {}: unknown dtype {other}", i + 1),
        };
        let shape: Vec<usize> = f[2]
            .split_whitespace()
            .map(|d| d.parse().unwrap())
            .collect();
        let numel: usize = f[3].parse().unwrap();
        assert_eq!(
            shape.iter().product::<usize>(),
            numel,
            "line {}: shape does not match numel",
            i + 1
        );
        rows.push(Row {
            name: f[0].to_string(),
            dtype,
            shape,
        });
    }
    rows
}

fn inventory(rows: &[Row]) -> Inventory {
    Inventory::of(rows.iter().map(|r| TensorInfo {
        name: &r.name,
        dtype: r.dtype,
        shape: &r.shape,
        contiguous: None, // the recorded inventory carries no strides
    }))
}

#[test]
fn the_recorded_inventory_matches_the_expected_layout() {
    let rows = recorded();
    let inv = inventory(&rows);
    let problems = inv.problems();
    assert!(problems.is_empty(), "{problems:#?}");
    assert_eq!(inv.tensors, layout::EXPECTED_TENSORS);
    assert_eq!(inv.params, layout::EXPECTED_PARAMS);
    assert_eq!(inv.dead_params, layout::DEAD_PARAMS);
    assert_eq!(
        inv.live_params(),
        layout::EXPECTED_PARAMS - layout::DEAD_PARAMS
    );
}

#[test]
fn every_key_is_unique_and_carries_the_wrapper_prefix() {
    let rows = recorded();
    let mut seen = std::collections::HashSet::new();
    for r in &rows {
        assert!(
            r.name.starts_with("model."),
            "{} lacks the SegVolModel wrapper prefix",
            r.name
        );
        assert!(seen.insert(r.name.clone()), "duplicate key {}", r.name);
    }
}

#[test]
fn the_image_encoder_is_a_12_block_768_wide_vit() {
    let rows = recorded();
    let inv = inventory(&rows);
    for b in 0..layout::EXPECTED_VIT_BLOCKS {
        let p = format!("image_encoder.blocks.{b}");
        // fused qkv, no bias - MONAI's SABlock hardcodes bias=False
        assert_eq!(
            inv.shape(&format!("{p}.attn.qkv.weight")),
            Some(&[2304, 768][..])
        );
        assert_eq!(inv.shape(&format!("{p}.attn.qkv.bias")), None);
        // the output projection does have one
        assert_eq!(
            inv.shape(&format!("{p}.attn.out_proj.weight")),
            Some(&[768, 768][..])
        );
        assert_eq!(
            inv.shape(&format!("{p}.attn.out_proj.bias")),
            Some(&[768][..])
        );
        // pre-norm: two LayerNorms per block, MLP ratio 4
        assert_eq!(inv.shape(&format!("{p}.norm1.weight")), Some(&[768][..]));
        assert_eq!(inv.shape(&format!("{p}.norm2.weight")), Some(&[768][..]));
        assert_eq!(
            inv.shape(&format!("{p}.mlp.linear1.weight")),
            Some(&[3072, 768][..])
        );
        assert_eq!(
            inv.shape(&format!("{p}.mlp.linear2.weight")),
            Some(&[768, 3072][..])
        );
    }
    // one block past the end must not exist
    assert_eq!(
        inv.shape(&format!(
            "image_encoder.blocks.{}.attn.qkv.weight",
            layout::EXPECTED_VIT_BLOCKS
        )),
        None
    );
    assert_eq!(inv.shape("image_encoder.norm.weight"), Some(&[768][..]));
}

#[test]
fn the_prompt_encoder_is_seven_live_tensors_plus_a_dead_2d_branch() {
    let rows = recorded();
    let inv = inventory(&rows);
    // the Fourier matrix is a buffer and must be loaded, not regenerated
    assert_eq!(
        inv.shape("prompt_encoder.pe_layer.positional_encoding_gaussian_matrix"),
        Some(&[3, 384][..])
    );
    // four point embeddings: negative, positive, box-min corner, box-max corner
    for i in 0..4 {
        assert_eq!(
            inv.shape(&format!("prompt_encoder.point_embeddings.{i}.weight")),
            Some(&[1, 768][..])
        );
    }
    assert_eq!(inv.shape("prompt_encoder.point_embeddings.4.weight"), None);
    assert_eq!(
        inv.shape("prompt_encoder.not_a_point_embed.weight"),
        Some(&[1, 768][..])
    );
    assert_eq!(
        inv.shape("prompt_encoder.no_mask_embed.weight"),
        Some(&[1, 768][..])
    );

    // the dead branch is 2-D throughout: 4-element kernel shapes, not 5
    let dead: Vec<&Row> = rows
        .iter()
        .filter(|r| layout::is_dead_weight(&r.name))
        .collect();
    assert_eq!(dead.len(), 10);
    for r in &dead {
        assert!(r.shape.len() <= 4, "{} is {:?}", r.name, r.shape);
    }
    let dead_params: usize = dead.iter().map(|r| r.shape.iter().product::<usize>()).sum();
    assert_eq!(dead_params, layout::DEAD_PARAMS);
    // live prompt-encoder parameters: the buffer plus six 768-d embeddings
    assert_eq!(
        inv.per_group["prompt_encoder"].1 - dead_params,
        1152 + 6 * 768
    );
}

#[test]
fn the_mask_decoder_is_a_depth_2_two_way_transformer() {
    let rows = recorded();
    let inv = inventory(&rows);
    for l in 0..layout::EXPECTED_DECODER_LAYERS {
        let p = format!("mask_decoder.transformer.layers.{l}");
        // self-attention at full width, both cross-attentions at half
        assert_eq!(
            inv.shape(&format!("{p}.self_attn.q_proj.weight")),
            Some(&[768, 768][..])
        );
        for a in ["cross_attn_token_to_image", "cross_attn_image_to_token"] {
            assert_eq!(
                inv.shape(&format!("{p}.{a}.q_proj.weight")),
                Some(&[384, 768][..])
            );
            assert_eq!(
                inv.shape(&format!("{p}.{a}.out_proj.weight")),
                Some(&[768, 384][..])
            );
            // unlike the ViT, every decoder projection carries a bias
            assert_eq!(inv.shape(&format!("{p}.{a}.q_proj.bias")), Some(&[384][..]));
        }
        // four LayerNorms per layer, MLP 768 -> 2048 -> 768
        for n in 1..=4 {
            assert_eq!(inv.shape(&format!("{p}.norm{n}.weight")), Some(&[768][..]));
        }
        assert_eq!(
            inv.shape(&format!("{p}.mlp.lin1.weight")),
            Some(&[2048, 768][..])
        );
        assert_eq!(
            inv.shape(&format!("{p}.mlp.lin2.weight")),
            Some(&[768, 2048][..])
        );
    }
    assert_eq!(
        inv.shape(&format!(
            "mask_decoder.transformer.layers.{}.self_attn.q_proj.weight",
            layout::EXPECTED_DECODER_LAYERS
        )),
        None
    );
    // the final token-to-image attention and its norm
    assert_eq!(
        inv.shape("mask_decoder.transformer.final_attn_token_to_image.q_proj.weight"),
        Some(&[384, 768][..])
    );
    assert_eq!(
        inv.shape("mask_decoder.transformer.norm_final_attn.weight"),
        Some(&[768][..])
    );

    // one iou token + four mask tokens; inference reads mask channel 0 only
    assert_eq!(
        inv.shape("mask_decoder.iou_token.weight"),
        Some(&[1, 768][..])
    );
    assert_eq!(
        inv.shape("mask_decoder.mask_tokens.weight"),
        Some(&[4, 768][..])
    );
    // one hypernetwork MLP per mask token, 768 -> 768 -> 768 -> 96
    for i in 0..4 {
        let p = format!("mask_decoder.output_hypernetworks_mlps.{i}.layers");
        assert_eq!(inv.shape(&format!("{p}.0.weight")), Some(&[768, 768][..]));
        assert_eq!(inv.shape(&format!("{p}.1.weight")), Some(&[768, 768][..]));
        assert_eq!(inv.shape(&format!("{p}.2.weight")), Some(&[96, 768][..]));
    }
    assert_eq!(
        inv.shape("mask_decoder.output_hypernetworks_mlps.4.layers.0.weight"),
        None
    );
    // IoU head 768 -> 256 -> 256 -> 4
    assert_eq!(
        inv.shape("mask_decoder.iou_prediction_head.layers.2.weight"),
        Some(&[4, 256][..])
    );
}

#[test]
fn the_decoder_layer_norm_normalizes_over_all_four_trailing_dims() {
    // This is the single most easily mis-ported tensor in the checkpoint: a
    // LayerNorm whose normalized_shape is (C, D, H, W), giving it 3.1 M
    // affine values in each of weight and bias rather than the 192 a
    // channel-wise norm would have. It is also what freezes the input shape.
    let rows = recorded();
    let inv = inventory(&rows);
    let want = &[192, 16, 32, 32][..];
    assert_eq!(
        inv.shape("mask_decoder.output_upscaling.1.weight"),
        Some(want)
    );
    assert_eq!(
        inv.shape("mask_decoder.output_upscaling.1.bias"),
        Some(want)
    );
    assert_eq!(want.iter().product::<usize>(), 3_145_728);
    // it alone is 21% of the mask decoder
    let ln = 2 * 3_145_728;
    let decoder = inv.per_group["mask_decoder"].1;
    assert!(
        (0.20..0.22).contains(&(ln as f64 / decoder as f64)),
        "{ln} of {decoder}"
    );
}

#[test]
fn the_text_tower_is_clip_vit_b_32() {
    let rows = recorded();
    let inv = inventory(&rows);
    let emb = "text_encoder.clip_text_model.text_model.embeddings";
    // vocab 49408, width 512, 77 positions - the ViT-B/32 text tower
    assert_eq!(
        inv.shape(&format!("{emb}.token_embedding.weight")),
        Some(&[49408, 512][..])
    );
    assert_eq!(
        inv.shape(&format!("{emb}.position_embedding.weight")),
        Some(&[77, 512][..])
    );
    // position_ids is the checkpoint's only integer tensor
    let ints: Vec<&Row> = rows
        .iter()
        .filter(|r| matches!(r.dtype, Dtype::I64 | Dtype::I32))
        .collect();
    assert_eq!(ints.len(), 1);
    assert_eq!(
        layout::normalize_key(&ints[0].name),
        format!("{emb}.position_ids")
    );
    assert_eq!(ints[0].shape, vec![1, 77]);

    for l in 0..layout::EXPECTED_CLIP_LAYERS {
        let p = format!("text_encoder.clip_text_model.text_model.encoder.layers.{l}");
        assert_eq!(
            inv.shape(&format!("{p}.self_attn.q_proj.weight")),
            Some(&[512, 512][..])
        );
        assert_eq!(
            inv.shape(&format!("{p}.mlp.fc1.weight")),
            Some(&[2048, 512][..])
        );
        assert_eq!(
            inv.shape(&format!("{p}.layer_norm1.weight")),
            Some(&[512][..])
        );
    }
    assert_eq!(
        inv.shape("text_encoder.clip_text_model.text_model.final_layer_norm.weight"),
        Some(&[512][..])
    );
    // and the projection that lifts a pooled 512-d CLIP embedding to the
    // network's 768-d prompt width
    assert_eq!(
        inv.shape("text_encoder.dim_align.weight"),
        Some(&[768, 512][..])
    );
    assert_eq!(inv.shape("text_encoder.dim_align.bias"), Some(&[768][..]));
}

#[test]
fn a_layout_deviation_is_reported_rather_than_ignored() {
    // Guard the guard: if a future checkpoint drops a tensor the port needs,
    // problems() must say so.
    let mut rows = recorded();
    rows.retain(|r| {
        !r.name
            .ends_with("prompt_encoder.pe_layer.positional_encoding_gaussian_matrix")
    });
    let problems = inventory(&rows).problems();
    assert!(
        problems
            .iter()
            .any(|p| p.contains("positional_encoding_gaussian_matrix")),
        "{problems:#?}"
    );
    // and an unexpected qkv bias must be caught too
    let mut rows = recorded();
    rows.push(Row {
        name: "model.image_encoder.blocks.0.attn.qkv.bias".into(),
        dtype: Dtype::F32,
        shape: vec![2304],
    });
    let problems = inventory(&rows).problems();
    assert!(
        problems.iter().any(|p| p.contains("qkv.bias")),
        "{problems:#?}"
    );
}

/// End-to-end against the published checkpoint. Ignored by default: it needs
/// the 724 MB download.
#[test]
#[ignore]
fn the_real_checkpoint_matches_the_recorded_inventory() {
    use rust_dicom_station::segvol::weights;
    let dir =
        std::path::PathBuf::from(std::env::var("RDS_SEGVOL_MODEL").expect("set RDS_SEGVOL_MODEL"));
    let path = weights::CHECKPOINT
        .ensure(&dir, &rust_dicom_station::progress::Stderr)
        .unwrap();
    let reader = weights::open_checkpoint(&path).unwrap();
    let live = Inventory::of(reader.tensors.iter().map(|(name, meta)| TensorInfo {
        name,
        dtype: meta.dtype,
        shape: &meta.shape,
        contiguous: Some(meta.is_contiguous()),
    }));
    let problems = live.problems();
    assert!(problems.is_empty(), "{problems:#?}");

    // and it agrees with the recorded fixture tensor for tensor
    let rows = recorded();
    assert_eq!(reader.tensors.len(), rows.len());
    for r in &rows {
        let found = reader
            .tensors
            .iter()
            .find(|(n, _)| n == &r.name)
            .unwrap_or_else(|| panic!("checkpoint is missing {}", r.name));
        assert_eq!(found.1.shape, r.shape, "{}", r.name);
        assert_eq!(found.1.dtype, r.dtype, "{}", r.name);
    }
}

// ---------------------------------------------------------------------------
// Network assembly and forward passes.
//
// The recorded inventory gives the exact published key names and shapes, so a
// checkpoint with the right structure and arbitrary values can be synthesized
// here. That exercises the whole assembly path -- every key the builders ask
// for, every shape assertion -- and a real forward pass, without the 724 MB
// download. Only the numbers are fake; the architecture is not.

use rust_dicom_station::nn::cache::WTensor;
use rust_dicom_station::nn::params::Params;
use rust_dicom_station::segvol::config::*;
use rust_dicom_station::segvol::prompt::{Point, PromptEncoder};
use rust_dicom_station::segvol::{decoder::MaskDecoder, net::SegVolNet, vit::Vit};

/// Deterministic small values: large enough to exercise the arithmetic,
/// small enough that twelve residual blocks do not overflow.
fn fill(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (((s >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5) * 0.05
        })
        .collect()
}

/// Synthesize a checkpoint containing exactly the recorded tensors whose keys
/// match `keep`.
fn synthetic(keep: impl Fn(&str) -> bool) -> Params {
    let mut m = std::collections::HashMap::new();
    for (i, r) in recorded().iter().enumerate() {
        let key = layout::normalize_key(&r.name);
        if !keep(key) {
            continue;
        }
        let n: usize = r.shape.iter().product();
        m.insert(
            key.to_string(),
            WTensor {
                shape: r.shape.clone(),
                data: fill(i as u64 + 1, n),
            },
        );
    }
    Params::new(m)
}

#[test]
fn the_prompt_encoder_and_mask_decoder_assemble_and_run() {
    // Everything except the image encoder and the text tower: the prompt
    // encoder, the depth-2 two-way transformer, the (C,D,H,W) LayerNorm, both
    // transposed convolutions, the four hypernetworks and the IoU head.
    let p = synthetic(|k| k.starts_with("prompt_encoder.") || k.starts_with("mask_decoder."));
    let prompt = PromptEncoder::build(&p).expect("prompt encoder");
    let decoder = MaskDecoder::build(&p).expect("mask decoder");

    let image =
        rust_dicom_station::nn::tensor::Mat::from_vec(TOKENS, EMBED, fill(999, TOKENS * EMBED));
    let image_pe = prompt.dense_pe();
    let text: Vec<f32> = fill(7, EMBED);

    let prompts = prompt.encode(
        &[Point::foreground([16.0, 128.0, 128.0])],
        &[[4.0, 40.0, 40.0, 28.0, 200.0, 200.0]],
        Some(&text),
    );
    // one point + two box corners + one text token
    assert_eq!(prompts.sparse.rows, 4);

    let out = decoder.forward(
        &image,
        &image_pe,
        &prompts.sparse,
        &prompts.dense,
        Some(&text),
    );
    assert_eq!(out.masks.c, NUM_MASK_TOKENS);
    assert_eq!([out.masks.d, out.masks.h, out.masks.w], MASK_SHAPE);
    assert_eq!(out.iou.len(), NUM_MASK_TOKENS);
    assert!(
        out.masks.data.iter().all(|v| v.is_finite()),
        "non-finite logits"
    );
    assert!(out.iou.iter().all(|v| v.is_finite()));
    // the four mask channels are genuinely different filters
    let ch = |i: usize| &out.masks.data[i * out.masks.spatial()..(i + 1) * out.masks.spatial()];
    assert_ne!(ch(0), ch(1));
    // inference keeps channel 0
    assert_eq!(out.best().data, ch(0));
    assert_eq!(out.best().c, 1);
}

#[test]
fn the_text_similarity_path_actually_changes_the_logits() {
    // Text enters twice. Dropping the additive similarity map is a silent
    // accuracy loss, so assert it moves the output.
    let p = synthetic(|k| k.starts_with("prompt_encoder.") || k.starts_with("mask_decoder."));
    let prompt = PromptEncoder::build(&p).unwrap();
    let decoder = MaskDecoder::build(&p).unwrap();
    let image =
        rust_dicom_station::nn::tensor::Mat::from_vec(TOKENS, EMBED, fill(31, TOKENS * EMBED));
    let pe = prompt.dense_pe();
    let text: Vec<f32> = fill(41, EMBED);
    let boxes = [[4.0, 40.0, 40.0, 28.0, 200.0, 200.0]];

    // identical sparse tokens, text similarity on and off
    let pr = prompt.encode(&[], &boxes, Some(&text));
    let with = decoder.forward(&image, &pe, &pr.sparse, &pr.dense, Some(&text));
    let without = decoder.forward(&image, &pe, &pr.sparse, &pr.dense, None);
    assert_ne!(with.masks.data, without.masks.data);
    // the difference is the same map added to every channel
    let sp = with.masks.spatial();
    let d0: Vec<f32> = (0..sp)
        .map(|i| with.masks.data[i] - without.masks.data[i])
        .collect();
    let d1: Vec<f32> = (0..sp)
        .map(|i| with.masks.data[sp + i] - without.masks.data[sp + i])
        .collect();
    for (a, b) in d0.iter().zip(d1.iter()) {
        assert!(
            (a - b).abs() < 1e-3,
            "similarity map must be shared: {a} vs {b}"
        );
    }
}

#[test]
fn the_image_encoder_assembles_from_the_published_key_names() {
    let p = synthetic(|k| k.starts_with("image_encoder."));
    Vit::build(&p).expect("image encoder");
}

#[test]
fn assembly_rejects_a_checkpoint_with_a_qkv_bias() {
    // MONAI builds the fused qkv with bias=False. A checkpoint that has one
    // is a different network and must be refused, not quietly accommodated.
    let mut m = std::collections::HashMap::new();
    for r in recorded() {
        let key = layout::normalize_key(&r.name).to_string();
        if !key.starts_with("image_encoder.") {
            continue;
        }
        let n: usize = r.shape.iter().product();
        m.insert(
            key,
            WTensor {
                shape: r.shape.clone(),
                data: vec![0.0; n],
            },
        );
    }
    m.insert(
        "image_encoder.blocks.0.attn.qkv.bias".to_string(),
        WTensor {
            shape: vec![3 * EMBED],
            data: vec![0.0; 3 * EMBED],
        },
    );
    let e = Vit::build(&Params::new(m))
        .map(|_| ())
        .unwrap_err()
        .to_string();
    assert!(e.contains("bias=False"), "{e}");
}

#[test]
#[ignore]
fn the_whole_network_runs_a_forward_pass() {
    // Heavy: the full 181 M-parameter assembly plus a real image-encoder pass
    // (2.5e11 MACs). Run with `cargo test --release -- --ignored`.
    let p = synthetic(|k| !k.starts_with("text_encoder."));
    let net = SegVolNet::build(&p).expect("network");
    let volume: Vec<f32> = fill(5, ROI[0] * ROI[1] * ROI[2]);
    let out = net.forward(&volume, &[], &[[4.0, 40.0, 40.0, 28.0, 200.0, 200.0]], None);
    assert_eq!([out.masks.d, out.masks.h, out.masks.w], MASK_SHAPE);
    assert!(out.masks.data.iter().all(|v| v.is_finite()));
    // re-decoding against a cached embedding must match a full forward
    let image = net.encode_image(&volume);
    let again = net.decode(&image, &[], &[[4.0, 40.0, 40.0, 28.0, 200.0, 200.0]], None);
    assert_eq!(out.masks.data, again.masks.data);
}

// ---------------------------------------------------------------------------
// The tokenizer against the published vocabulary.
//
// vocab.json and merges.txt cannot be synthesized -- the merge table *is* the
// algorithm -- so this is the one part of the port with no offline check. The
// round-trip below is strong evidence in its place: decoding recovers the
// input only if the byte alphabet, the merge ranks and the vocabulary all
// line up.

#[test]
#[ignore]
fn the_real_tokenizer_round_trips() {
    use rust_dicom_station::segvol::bpe::{self, Bpe, BOS, EOS};
    let dir =
        std::path::PathBuf::from(std::env::var("RDS_SEGVOL_MODEL").expect("set RDS_SEGVOL_MODEL"));
    let t = Bpe::from_dir(&dir).expect("vocab.json + merges.txt");
    assert_eq!(t.vocab_size(), 49408);

    for name in [
        "liver",
        "spleen",
        "left kidney",
        "pancreas",
        "aorta",
        "gallbladder",
        "urinary bladder",
        "esophagus",
    ] {
        let text = bpe::prompt_for(name);
        let ids = t.encode(&text);
        assert_eq!(ids.first(), Some(&BOS));
        assert_eq!(ids.last(), Some(&EOS));
        assert!(
            ids.len() >= 8 && ids.len() <= 32,
            "{name}: {} tokens",
            ids.len()
        );
        // decoding must recover the cleaned text exactly
        assert_eq!(t.decode(&ids), text.to_lowercase(), "round trip for {name}");
    }
    // common single words tokenize to one symbol plus the two markers
    for w in ["liver", "the", "of"] {
        assert_eq!(t.encode(w).len(), 3, "{w} should be a single token");
    }
    // every digit separately
    assert_eq!(t.encode("2024").len(), 6);
}
