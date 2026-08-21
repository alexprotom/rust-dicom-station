//! The published checkpoint's tensor layout, and the checks that decide
//! whether a given file really is the model we think it is.
//!
//! Everything here was read off the real `pytorch_model.bin` with
//! `examples/segvol_probe`, and is recorded so that the network assembly can
//! be developed and tested against the exact key names and shapes without the
//! 724 MB download. `tests/data/segvol-tensors.csv` is the full inventory;
//! the constants below are the parts the port's correctness turns on.

use std::collections::BTreeMap;

use crate::nn::pickle::Dtype;

/// The Hugging Face export wraps the network in a `SegVolModel` whose only
/// field is `model`, so every state-dict key is prefixed. Strip it once on
/// load and the rest of the code sees the architecture's own names.
pub fn normalize_key(key: &str) -> &str {
    key.strip_prefix("model.").unwrap_or(key)
}

/// Which part of the network a (normalized) key belongs to.
pub fn group_of(key: &str) -> &'static str {
    let k = normalize_key(key);
    if k.starts_with("image_encoder.") {
        "image_encoder"
    } else if k.starts_with("prompt_encoder.") {
        "prompt_encoder"
    } else if k.starts_with("mask_decoder.") {
        "mask_decoder"
    } else if k.starts_with("text_encoder.") {
        "text_encoder"
    } else {
        "other"
    }
}

/// True for tensors the inference path never touches.
///
/// `prompt_encoder.mask_downscaling` is SAM's 2-D mask-input branch. SegVol
/// never passes a mask prompt, so the branch is dead — but its 13,388
/// parameters are still in the checkpoint, and reproducing them would mean
/// implementing `Conv2d` for nothing.
pub fn is_dead_weight(key: &str) -> bool {
    normalize_key(key).starts_with("prompt_encoder.mask_downscaling.")
}

// ---- what the real checkpoint contains ----------------------------------

pub const EXPECTED_TENSORS: usize = 475;

/// Every value in the state dict, buffers included.
pub const EXPECTED_PARAMS: usize = 180_891_293;

/// Learnable parameters only — the figure the paper reports as "181 M".
///
/// The difference from [`EXPECTED_PARAMS`] is two non-learnable buffers that
/// `state_dict()` carries but that no parameter count includes: the prompt
/// encoder's 3x384 random-Fourier matrix (1,152) and CLIP's `position_ids`
/// (77 int64 values).
pub const EXPECTED_LEARNABLE: usize = 180_890_064;

/// Parameters in the dead 2-D mask branch (see [`is_dead_weight`]).
pub const DEAD_PARAMS: usize = 13_388;

/// Int64 values in the checkpoint: CLIP's `position_ids` buffer, and nothing
/// else. Everything the network computes with is fp32.
pub const EXPECTED_INT_VALUES: usize = 77;

/// Parameters per group, as measured.
pub const EXPECTED_PER_GROUP: &[(&str, usize)] = &[
    ("image_encoder", 87_388_416),
    ("mask_decoder", 29_923_716),
    ("prompt_encoder", 19_148),
    ("text_encoder", 63_560_013),
];

/// Total tensor bytes: fp32 values plus the one int64 buffer.
pub const PAYLOAD_BYTES: u64 =
    (EXPECTED_PARAMS - EXPECTED_INT_VALUES) as u64 * 4 + EXPECTED_INT_VALUES as u64 * 8;

/// Tensors whose shapes pin down the parts of the architecture that the
/// published description gets wrong or leaves ambiguous. If these match, the
/// network in the file is the network the port assumes.
pub const PINNED_SHAPES: &[(&str, &[usize])] = &[
    // A learned absolute position embedding over 2048 tokens: no
    // interpolation logic exists, so the input is hard-locked to 32x256x256.
    (
        "image_encoder.patch_embedding.position_embeddings",
        &[1, 2048, 768],
    ),
    // The patch embedding is a Linear over flattened (4,16,16) patches
    // (4*16*16 = 1024), not a Conv3d.
    (
        "image_encoder.patch_embedding.patch_embeddings.1.weight",
        &[768, 1024],
    ),
    // Fused qkv, 3*768 = 2304. See ABSENT_KEYS: it has no bias.
    ("image_encoder.blocks.0.attn.qkv.weight", &[2304, 768]),
    ("image_encoder.blocks.0.mlp.linear1.weight", &[3072, 768]),
    // The random-Fourier prompt encoding is a *buffer*: it must be loaded
    // from the checkpoint, never regenerated.
    (
        "prompt_encoder.pe_layer.positional_encoding_gaussian_matrix",
        &[3, 384],
    ),
    // 3 multimask outputs + 1, of which inference uses only channel 0.
    ("mask_decoder.mask_tokens.weight", &[4, 768]),
    // The two-way transformer runs its cross-attentions at an internal
    // dimension of 384 (downsample_rate 2) and self-attention at 768.
    (
        "mask_decoder.transformer.layers.0.self_attn.q_proj.weight",
        &[768, 768],
    ),
    (
        "mask_decoder.transformer.layers.0.cross_attn_token_to_image.q_proj.weight",
        &[384, 768],
    ),
    (
        "mask_decoder.transformer.layers.0.mlp.lin1.weight",
        &[2048, 768],
    ),
    // Upscaling: ConvTranspose3d 768->192, then a LayerNorm whose
    // normalized_shape is the full (C,D,H,W) — 3.1 M affine values in each of
    // weight and bias, and the second reason the input shape is frozen — then
    // ConvTranspose3d 192->96.
    (
        "mask_decoder.output_upscaling.0.weight",
        &[768, 192, 2, 2, 2],
    ),
    ("mask_decoder.output_upscaling.1.weight", &[192, 16, 32, 32]),
    ("mask_decoder.output_upscaling.1.bias", &[192, 16, 32, 32]),
    (
        "mask_decoder.output_upscaling.3.weight",
        &[192, 96, 2, 2, 2],
    ),
    // Hypernetwork MLPs project a mask token down to the 96 upscaled channels.
    (
        "mask_decoder.output_hypernetworks_mlps.0.layers.2.weight",
        &[96, 768],
    ),
    // Text is injected twice; this is the additive-similarity path, and the
    // easiest one to leave out by accident.
    (
        "mask_decoder.txt_align_upscaled_embedding.weight",
        &[96, 768],
    ),
    // CLIP ViT-B/32 text tower: vocab 49408, width 512, 77 positions.
    (
        "text_encoder.clip_text_model.text_model.embeddings.token_embedding.weight",
        &[49408, 512],
    ),
    (
        "text_encoder.clip_text_model.text_model.embeddings.position_embedding.weight",
        &[77, 512],
    ),
    (
        "text_encoder.clip_text_model.text_model.encoder.layers.0.mlp.fc1.weight",
        &[2048, 512],
    ),
    ("text_encoder.dim_align.weight", &[768, 512]),
];

/// Keys that must **not** exist. Their absence is as load-bearing as any
/// shape: MONAI's `SABlock` builds its qkv projection with `bias=False`,
/// while the decoder's attention projections all carry biases. A port that
/// adds a qkv bias "for symmetry" is silently wrong.
pub const ABSENT_KEYS: &[&str] = &[
    "image_encoder.blocks.0.attn.qkv.bias",
    "image_encoder.blocks.11.attn.qkv.bias",
];

/// Repeat counts: 12 ViT blocks, a depth-2 two-way transformer, 12 CLIP
/// text layers.
pub const EXPECTED_VIT_BLOCKS: usize = 12;
pub const EXPECTED_DECODER_LAYERS: usize = 2;
pub const EXPECTED_CLIP_LAYERS: usize = 12;

// ---- inventory ----------------------------------------------------------

/// One tensor, as either the checkpoint reader or the recorded inventory
/// describes it.
#[derive(Clone, Copy, Debug)]
pub struct TensorInfo<'a> {
    pub name: &'a str,
    pub dtype: Dtype,
    pub shape: &'a [usize],
    /// `None` when contiguity is unknown (the recorded inventory has no
    /// strides); `Some(false)` is always a problem.
    pub contiguous: Option<bool>,
}

/// A checkpoint's tensor table, summarized.
#[derive(Debug, Default)]
pub struct Inventory {
    pub tensors: usize,
    pub params: usize,
    pub dead_params: usize,
    pub int_values: usize,
    pub per_group: BTreeMap<&'static str, (usize, usize)>,
    pub non_contiguous: Vec<String>,
    shapes: BTreeMap<String, Vec<usize>>,
}

impl Inventory {
    pub fn of<'a>(items: impl IntoIterator<Item = TensorInfo<'a>>) -> Inventory {
        let mut inv = Inventory::default();
        for t in items {
            let n: usize = t.shape.iter().product();
            let key = normalize_key(t.name);
            inv.tensors += 1;
            inv.params += n;
            if is_dead_weight(key) {
                inv.dead_params += n;
            }
            if !matches!(t.dtype, Dtype::F32 | Dtype::F64 | Dtype::F16) {
                inv.int_values += n;
            }
            let e = inv.per_group.entry(group_of(key)).or_default();
            e.0 += 1;
            e.1 += n;
            if t.contiguous == Some(false) {
                inv.non_contiguous.push(key.to_string());
            }
            inv.shapes.insert(key.to_string(), t.shape.to_vec());
        }
        inv
    }

    pub fn live_params(&self) -> usize {
        self.params - self.dead_params
    }

    fn count_indexed(&self, prefix: &str, suffix: &str) -> usize {
        (0..)
            .take_while(|i| self.shapes.contains_key(&format!("{prefix}{i}{suffix}")))
            .count()
    }

    /// Compare against the recorded layout. An empty result means this really
    /// is the SegVol checkpoint the port was written for.
    pub fn problems(&self) -> Vec<String> {
        let mut p = Vec::new();
        if self.tensors != EXPECTED_TENSORS {
            p.push(format!(
                "{} tensors, expected {EXPECTED_TENSORS}",
                self.tensors
            ));
        }
        if self.params != EXPECTED_PARAMS {
            p.push(format!(
                "{} values, expected {EXPECTED_PARAMS} (difference {})",
                self.params,
                self.params as i64 - EXPECTED_PARAMS as i64
            ));
        }
        if self.dead_params != DEAD_PARAMS {
            p.push(format!(
                "{} parameters in the dead 2-D mask branch, expected {DEAD_PARAMS}",
                self.dead_params
            ));
        }
        if self.int_values != EXPECTED_INT_VALUES {
            p.push(format!(
                "{} integer values, expected {EXPECTED_INT_VALUES}",
                self.int_values
            ));
        }
        for (g, want) in EXPECTED_PER_GROUP {
            let got = self.per_group.get(g).map(|e| e.1).unwrap_or(0);
            if got != *want {
                p.push(format!("group {g}: {got} parameters, expected {want}"));
            }
        }
        if let Some(other) = self.per_group.get("other") {
            p.push(format!(
                "{} tensors ({} values) fall outside every known group",
                other.0, other.1
            ));
        }
        for name in &self.non_contiguous {
            p.push(format!("{name} is a non-contiguous view"));
        }
        for (key, want) in PINNED_SHAPES {
            match self.shapes.get(*key) {
                Some(got) if got == want => {}
                Some(got) => p.push(format!("{key} is {got:?}, expected {want:?}")),
                None => p.push(format!("{key} is missing")),
            }
        }
        for key in ABSENT_KEYS {
            if self.shapes.contains_key(*key) {
                p.push(format!("{key} exists but must not"));
            }
        }
        for (label, count, want) in [
            (
                "ViT blocks",
                self.count_indexed("image_encoder.blocks.", ".attn.qkv.weight"),
                EXPECTED_VIT_BLOCKS,
            ),
            (
                "decoder transformer layers",
                self.count_indexed(
                    "mask_decoder.transformer.layers.",
                    ".self_attn.q_proj.weight",
                ),
                EXPECTED_DECODER_LAYERS,
            ),
            (
                "CLIP text layers",
                self.count_indexed(
                    "text_encoder.clip_text_model.text_model.encoder.layers.",
                    ".mlp.fc1.weight",
                ),
                EXPECTED_CLIP_LAYERS,
            ),
        ] {
            if count != want {
                p.push(format!("{count} {label}, expected {want}"));
            }
        }
        p
    }

    /// Shape of one (normalized) key, if present.
    pub fn shape(&self, key: &str) -> Option<&[usize]> {
        self.shapes.get(normalize_key(key)).map(|v| v.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrapper_prefix_is_stripped_once() {
        assert_eq!(
            normalize_key("model.image_encoder.norm.weight"),
            "image_encoder.norm.weight"
        );
        // already-normalized keys pass through
        assert_eq!(
            normalize_key("image_encoder.norm.weight"),
            "image_encoder.norm.weight"
        );
        // only the outermost prefix goes; nothing deeper is touched
        assert_eq!(
            normalize_key("model.mask_decoder.transformer.layers.0.mlp.lin1.weight"),
            "mask_decoder.transformer.layers.0.mlp.lin1.weight"
        );
    }

    #[test]
    fn grouping_and_dead_weights_work_on_raw_keys() {
        assert_eq!(group_of("model.image_encoder.norm.weight"), "image_encoder");
        assert_eq!(group_of("text_encoder.dim_align.weight"), "text_encoder");
        assert_eq!(group_of("nonsense"), "other");
        assert!(is_dead_weight(
            "model.prompt_encoder.mask_downscaling.0.weight"
        ));
        assert!(!is_dead_weight("model.prompt_encoder.no_mask_embed.weight"));
    }

    #[test]
    fn the_recorded_constants_agree_with_each_other() {
        let group_total: usize = EXPECTED_PER_GROUP.iter().map(|(_, n)| n).sum();
        assert_eq!(group_total, EXPECTED_PARAMS);
        // the two buffers state_dict() carries but the paper's count does not
        assert_eq!(EXPECTED_PARAMS - EXPECTED_LEARNABLE, 1152 + 77);
        // the analytic module-by-module total from the plan
        assert_eq!(
            EXPECTED_LEARNABLE,
            87_388_416 + 17_996 + 29_923_716 + 63_165_952 + 393_984
        );
        // prompt encoder = the analytic 17,996 plus its Fourier buffer
        assert_eq!(EXPECTED_PER_GROUP[2].1, 17_996 + 1_152);
        assert_eq!(PAYLOAD_BYTES, 723_565_480);
    }
}
