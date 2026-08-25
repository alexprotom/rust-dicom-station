//! The published checkpoint's tensor layout, and the checks that decide
//! whether a given file really is the model we think it is.
//!
//! SegVol's equivalent module records an inventory read off the real file.
//! Here the inventory is *derived* instead: SAM 2.1's parameter set follows
//! mechanically from the geometry in [`super::config`], so [`expected`] builds
//! all 471 key/shape pairs from it and the tests assert that they sum to the
//! published parameter count. A checkpoint is then accepted only if it agrees
//! with that derivation key for key — which catches both a drifted upstream
//! file and a mistake in our own understanding of the architecture, and needs
//! no 156 MB download to run.
//!
//! Keys are given in normalized form: the `.pt` wraps the state dict in a
//! `"model"` entry, which [`crate::nn::pickle::PthReader`] unwraps on open, so
//! what reaches here are the architecture's own names.

use std::collections::BTreeMap;

use super::config::*;

/// Total elements across the state dict, parameters plus the one buffer.
pub const STATE_ELEMENTS: usize = 38_962_754;
/// Trainable parameters (`STATE_ELEMENTS` minus the 256-element random-Fourier
/// buffer), as reported by the reference implementation.
pub const PARAMETERS: usize = 38_962_498;
/// Entries in the state dict.
pub const TENSOR_COUNT: usize = 471;
/// Bytes of tensor data in the fp32 checkpoint, `STATE_ELEMENTS * 4`.
pub const PAYLOAD_BYTES: u64 = STATE_ELEMENTS as u64 * 4;

/// Strip prefixes a re-save might have added. The published files are already
/// bare once `"model"` is unwrapped; `module.` appears if someone saves a
/// `DistributedDataParallel` wrapper by hand.
pub fn normalize_key(key: &str) -> &str {
    key.strip_prefix("module.").unwrap_or(key)
}

/// Which part of the network a key belongs to.
pub fn group_of(key: &str) -> &'static str {
    let k = normalize_key(key);
    for (prefix, group) in [
        ("image_encoder.trunk.", "trunk"),
        ("image_encoder.neck.", "neck"),
        ("sam_prompt_encoder.", "prompt_encoder"),
        ("sam_mask_decoder.", "mask_decoder"),
        ("memory_attention.", "memory_attention"),
        ("memory_encoder.", "memory_encoder"),
    ] {
        if k.starts_with(prefix) {
            return group;
        }
    }
    "other"
}

/// True for tensors the inference path never touches.
///
/// `no_mem_pos_enc` is the positional encoding of the dummy memory token used
/// when `directly_add_no_mem_embed` is false. MedSAM2's config sets it true,
/// so the conditioning slice takes the short-circuit path (`no_mem_embed`
/// added straight onto the features) and this tensor is never read. It is the
/// only dead weight in the checkpoint — note in particular that
/// `mask_downsample` is *not* dead here, because this engine does accept mask
/// prompts.
pub fn is_dead_weight(key: &str) -> bool {
    normalize_key(key) == "no_mem_pos_enc"
}

// ---- inventory construction ---------------------------------------------

struct Builder(BTreeMap<String, Vec<usize>>);

impl Builder {
    fn put(&mut self, key: String, shape: Vec<usize>) {
        assert!(self.0.insert(key.clone(), shape).is_none(), "dup {key}");
    }
    /// `nn.Linear(inp, out)`.
    fn linear(&mut self, p: &str, out: usize, inp: usize) {
        self.put(format!("{p}.weight"), vec![out, inp]);
        self.put(format!("{p}.bias"), vec![out]);
    }
    /// `nn.LayerNorm(n)` and the channels-first `LayerNorm2d(n)` — same pair.
    fn norm(&mut self, p: &str, n: usize) {
        self.put(format!("{p}.weight"), vec![n]);
        self.put(format!("{p}.bias"), vec![n]);
    }
    /// `nn.Conv2d(inp, out, k, groups=g)`; PyTorch stores `[out, inp/g, k, k]`.
    fn conv(&mut self, p: &str, out: usize, inp: usize, k: usize, groups: usize) {
        self.put(format!("{p}.weight"), vec![out, inp / groups, k, k]);
        self.put(format!("{p}.bias"), vec![out]);
    }
    /// `nn.ConvTranspose2d(inp, out, k)`; stored `[inp, out, k, k]`.
    fn conv_t(&mut self, p: &str, out: usize, inp: usize, k: usize) {
        self.put(format!("{p}.weight"), vec![inp, out, k, k]);
        self.put(format!("{p}.bias"), vec![out]);
    }
    /// SAM 2's `MLP` helper: `dims.len() - 1` linear layers under `.layers.N`.
    fn mlp(&mut self, p: &str, dims: &[usize]) {
        for i in 0..dims.len() - 1 {
            self.linear(&format!("{p}.layers.{i}"), dims[i + 1], dims[i]);
        }
    }
    /// One `Attention` from `sam/transformer.py`, internal width
    /// `D_MODEL / downsample`.
    fn attention(&mut self, p: &str, dim: usize, downsample: usize, kv_in: usize) {
        let internal = dim / downsample;
        self.linear(&format!("{p}.q_proj"), internal, dim);
        self.linear(&format!("{p}.k_proj"), internal, kv_in);
        self.linear(&format!("{p}.v_proj"), internal, kv_in);
        self.linear(&format!("{p}.out_proj"), dim, internal);
    }
}

/// Every key the checkpoint must contain, with its exact shape.
pub fn expected() -> BTreeMap<String, Vec<usize>> {
    let mut b = Builder(BTreeMap::new());

    // ---- image encoder: trunk -------------------------------------------
    let t = "image_encoder.trunk";
    b.put(
        format!("{t}.pos_embed"),
        vec![1, EMBED_DIM, POS_EMBED_BKG, POS_EMBED_BKG],
    );
    b.put(
        format!("{t}.pos_embed_window"),
        vec![1, EMBED_DIM, POS_EMBED_WINDOW, POS_EMBED_WINDOW],
    );
    b.conv(
        &format!("{t}.patch_embed.proj"),
        EMBED_DIM,
        3,
        PATCH_KERNEL,
        1,
    );
    for (i, blk) in blocks().iter().enumerate() {
        let p = format!("{t}.blocks.{i}");
        b.norm(&format!("{p}.norm1"), blk.dim_in);
        b.linear(&format!("{p}.attn.qkv"), 3 * blk.dim_out, blk.dim_in);
        b.linear(&format!("{p}.attn.proj"), blk.dim_out, blk.dim_out);
        b.norm(&format!("{p}.norm2"), blk.dim_out);
        b.mlp(
            &format!("{p}.mlp"),
            &[blk.dim_out, 4 * blk.dim_out, blk.dim_out],
        );
        if blk.dim_in != blk.dim_out {
            b.linear(&format!("{p}.proj"), blk.dim_out, blk.dim_in);
        }
    }

    // ---- image encoder: neck --------------------------------------------
    for (i, ch) in BACKBONE_CHANNELS.iter().enumerate() {
        b.conv(
            &format!("image_encoder.neck.convs.{i}.conv"),
            D_MODEL,
            *ch,
            1,
            1,
        );
    }

    // ---- prompt encoder --------------------------------------------------
    let p = "sam_prompt_encoder";
    b.put(
        format!("{p}.pe_layer.positional_encoding_gaussian_matrix"),
        vec![2, PE_GAUSSIAN],
    );
    for i in 0..4 {
        b.put(format!("{p}.point_embeddings.{i}.weight"), vec![1, D_MODEL]);
    }
    b.put(format!("{p}.not_a_point_embed.weight"), vec![1, D_MODEL]);
    b.put(format!("{p}.no_mask_embed.weight"), vec![1, D_MODEL]);
    let md = format!("{p}.mask_downscaling");
    b.conv(&format!("{md}.0"), MASK_IN_CHANS / 4, 1, 2, 1);
    b.norm(&format!("{md}.1"), MASK_IN_CHANS / 4);
    b.conv(&format!("{md}.3"), MASK_IN_CHANS, MASK_IN_CHANS / 4, 2, 1);
    b.norm(&format!("{md}.4"), MASK_IN_CHANS);
    b.conv(&format!("{md}.6"), D_MODEL, MASK_IN_CHANS, 1, 1);

    // ---- mask decoder ----------------------------------------------------
    let d = "sam_mask_decoder";
    for i in 0..DEC_LAYERS {
        let l = format!("{d}.transformer.layers.{i}");
        b.attention(&format!("{l}.self_attn"), D_MODEL, 1, D_MODEL);
        b.norm(&format!("{l}.norm1"), D_MODEL);
        b.attention(
            &format!("{l}.cross_attn_token_to_image"),
            D_MODEL,
            DEC_DOWNSAMPLE,
            D_MODEL,
        );
        b.norm(&format!("{l}.norm2"), D_MODEL);
        b.mlp(&format!("{l}.mlp"), &[D_MODEL, DEC_MLP, D_MODEL]);
        b.norm(&format!("{l}.norm3"), D_MODEL);
        b.attention(
            &format!("{l}.cross_attn_image_to_token"),
            D_MODEL,
            DEC_DOWNSAMPLE,
            D_MODEL,
        );
        b.norm(&format!("{l}.norm4"), D_MODEL);
    }
    b.attention(
        &format!("{d}.transformer.final_attn_token_to_image"),
        D_MODEL,
        DEC_DOWNSAMPLE,
        D_MODEL,
    );
    b.norm(&format!("{d}.transformer.norm_final_attn"), D_MODEL);
    b.put(format!("{d}.iou_token.weight"), vec![1, D_MODEL]);
    b.put(
        format!("{d}.mask_tokens.weight"),
        vec![NUM_MASK_TOKENS, D_MODEL],
    );
    b.put(format!("{d}.obj_score_token.weight"), vec![1, D_MODEL]);
    b.conv_t(&format!("{d}.output_upscaling.0"), D_MODEL / 4, D_MODEL, 2);
    b.norm(&format!("{d}.output_upscaling.1"), D_MODEL / 4);
    b.conv_t(
        &format!("{d}.output_upscaling.3"),
        UPSCALED_CH,
        D_MODEL / 4,
        2,
    );
    for i in 0..NUM_MASK_TOKENS {
        b.mlp(
            &format!("{d}.output_hypernetworks_mlps.{i}"),
            &[D_MODEL, D_MODEL, D_MODEL, HYPER_DIM],
        );
    }
    b.mlp(
        &format!("{d}.iou_prediction_head"),
        &[D_MODEL, D_MODEL, D_MODEL, NUM_MASK_TOKENS],
    );
    b.mlp(
        &format!("{d}.pred_obj_score_head"),
        &[D_MODEL, D_MODEL, D_MODEL, 1],
    );
    b.conv(&format!("{d}.conv_s0"), HIGH_RES_S0_CH, D_MODEL, 1, 1);
    b.conv(&format!("{d}.conv_s1"), HIGH_RES_S1_CH, D_MODEL, 1, 1);

    // ---- memory attention -------------------------------------------------
    for i in 0..MEM_ATTN_LAYERS {
        let l = format!("memory_attention.layers.{i}");
        b.attention(&format!("{l}.self_attn"), D_MODEL, 1, D_MODEL);
        b.attention(&format!("{l}.cross_attn_image"), D_MODEL, 1, MEM_DIM);
        b.linear(&format!("{l}.linear1"), MEM_MLP, D_MODEL);
        b.linear(&format!("{l}.linear2"), D_MODEL, MEM_MLP);
        for n in 1..=3 {
            b.norm(&format!("{l}.norm{n}"), D_MODEL);
        }
    }
    b.norm("memory_attention.norm", D_MODEL);

    // ---- memory encoder ---------------------------------------------------
    let e = "memory_encoder.mask_downsampler.encoder";
    let mut ch = 1;
    for i in 0..MASK_DOWN_LAYERS {
        let out = ch * MASK_DOWN_STRIDE * MASK_DOWN_STRIDE;
        b.conv(&format!("{e}.{}", 3 * i), out, ch, 3, 1);
        b.norm(&format!("{e}.{}", 3 * i + 1), out);
        ch = out;
    }
    assert_eq!(ch, D_MODEL);
    b.conv(
        &format!("{e}.{}", 3 * MASK_DOWN_LAYERS),
        D_MODEL,
        D_MODEL,
        1,
        1,
    );
    b.conv("memory_encoder.pix_feat_proj", D_MODEL, D_MODEL, 1, 1);
    for i in 0..FUSER_LAYERS {
        let l = format!("memory_encoder.fuser.layers.{i}");
        b.conv(&format!("{l}.dwconv"), D_MODEL, D_MODEL, CX_KERNEL, D_MODEL);
        b.norm(&format!("{l}.norm"), D_MODEL);
        b.linear(&format!("{l}.pwconv1"), CX_MLP, D_MODEL);
        b.linear(&format!("{l}.pwconv2"), D_MODEL, CX_MLP);
        b.put(format!("{l}.gamma"), vec![D_MODEL]);
    }
    b.conv("memory_encoder.out_proj", MEM_DIM, D_MODEL, 1, 1);

    // ---- top level --------------------------------------------------------
    b.mlp("obj_ptr_proj", &[D_MODEL, D_MODEL, D_MODEL, D_MODEL]);
    b.linear("obj_ptr_tpos_proj", MEM_DIM, D_MODEL);
    b.conv("mask_downsample", 1, 1, 4, 1);
    b.put("maskmem_tpos_enc".into(), vec![NUM_MASKMEM, 1, 1, MEM_DIM]);
    b.put("no_mem_embed".into(), vec![1, 1, D_MODEL]);
    b.put("no_mem_pos_enc".into(), vec![1, 1, D_MODEL]);
    b.put("no_obj_ptr".into(), vec![1, D_MODEL]);
    b.put("no_obj_embed_spatial".into(), vec![1, MEM_DIM]);

    b.0
}

/// Elements in a shape.
fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Per-group element counts of [`expected`], for the probe's report.
pub fn group_totals() -> BTreeMap<&'static str, usize> {
    let mut out: BTreeMap<&'static str, usize> = BTreeMap::new();
    for (k, s) in expected() {
        *out.entry(group_of(&k)).or_default() += numel(&s);
    }
    out
}

/// One tensor as found in a checkpoint.
#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: &'static str,
}

/// Everything wrong with a checkpoint, or an empty list.
///
/// Reported rather than returned as an error so the probe can print all of it
/// at once instead of one problem per run.
pub fn problems(actual: &[TensorInfo]) -> Vec<String> {
    let want = expected();
    let mut out = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    for t in actual {
        let key = normalize_key(&t.name).to_string();
        if !seen.insert(key.clone()) {
            out.push(format!("{key}: appears twice"));
            continue;
        }
        match want.get(&key) {
            None => out.push(format!("{key}: unexpected tensor {:?}", t.shape)),
            Some(s) if *s != t.shape => {
                out.push(format!("{key}: shape {:?}, expected {s:?}", t.shape))
            }
            Some(_) => {}
        }
        if t.dtype != "f32" && t.dtype != "f16" {
            out.push(format!("{key}: unsupported dtype {}", t.dtype));
        }
    }
    for k in want.keys() {
        if !seen.contains(k) {
            out.push(format!("{k}: missing"));
        }
    }
    let total: usize = actual.iter().map(|t| numel(&t.shape)).sum();
    if out.is_empty() && total != STATE_ELEMENTS {
        out.push(format!("{total} elements, expected {STATE_ELEMENTS}"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_derived_inventory_matches_the_published_totals() {
        let e = expected();
        assert_eq!(e.len(), TENSOR_COUNT, "tensor count");
        let total: usize = e.values().map(|s| numel(s)).sum();
        assert_eq!(total, STATE_ELEMENTS, "state-dict elements");
        assert_eq!(PARAMETERS, STATE_ELEMENTS - 2 * PE_GAUSSIAN);
    }

    #[test]
    fn every_group_matches_the_published_subtotal() {
        // Reference figures, obtained by instantiating SAM 2.1-T.
        let want: [(&str, usize); 7] = [
            ("trunk", 26_849_472),
            ("neck", 369_664),
            ("memory_attention", 5_922_304),
            ("mask_decoder", 4_215_109),
            ("memory_encoder", 1_384_608),
            ("prompt_encoder", 6_476),
            // obj_ptr_proj + obj_ptr_tpos_proj + mask_downsample +
            // maskmem_tpos_enc + the four singleton embeddings
            ("other", 197_376 + 16_448 + 17 + 448 + 256 + 256 + 256 + 64),
        ];
        let got = group_totals();
        for (g, n) in want {
            assert_eq!(got.get(g), Some(&n), "group {g}");
        }
        assert_eq!(got.len(), want.len());
    }

    #[test]
    fn the_image_encoder_is_two_groups_that_sum_to_the_published_figure() {
        let g = group_totals();
        assert_eq!(g["trunk"] + g["neck"], 27_219_136);
    }

    #[test]
    fn a_missing_or_misshapen_tensor_is_reported_by_name() {
        let mut actual: Vec<TensorInfo> = expected()
            .into_iter()
            .map(|(name, shape)| TensorInfo {
                name,
                shape,
                dtype: "f32",
            })
            .collect();
        assert!(problems(&actual).is_empty());

        actual[0].shape.push(3);
        let bad = actual[0].name.clone();
        let p = problems(&actual);
        assert_eq!(p.len(), 1);
        assert!(p[0].starts_with(&bad), "{p:?}");

        actual.remove(0);
        let p = problems(&actual);
        assert_eq!(p.len(), 1);
        assert!(p[0].contains("missing"), "{p:?}");
    }

    #[test]
    fn an_unexpected_tensor_is_reported_too() {
        let mut actual: Vec<TensorInfo> = expected()
            .into_iter()
            .map(|(name, shape)| TensorInfo {
                name,
                shape,
                dtype: "f32",
            })
            .collect();
        actual.push(TensorInfo {
            name: "image_encoder.trunk.blocks.12.norm1.weight".into(),
            shape: vec![768],
            dtype: "f32",
        });
        let p = problems(&actual);
        assert_eq!(p.len(), 1);
        assert!(p[0].contains("unexpected"), "{p:?}");
    }

    #[test]
    fn keys_are_normalized_and_grouped() {
        assert_eq!(normalize_key("module.no_mem_embed"), "no_mem_embed");
        assert_eq!(group_of("image_encoder.trunk.pos_embed"), "trunk");
        assert_eq!(group_of("image_encoder.neck.convs.0.conv.weight"), "neck");
        assert_eq!(group_of("maskmem_tpos_enc"), "other");
        assert!(is_dead_weight("no_mem_pos_enc"));
        assert!(!is_dead_weight("no_mem_embed"));
        assert!(!is_dead_weight("mask_downsample.weight"));
    }

    #[test]
    fn only_one_tensor_is_dead() {
        let dead = expected().keys().filter(|k| is_dead_weight(k)).count();
        assert_eq!(dead, 1);
    }
}
