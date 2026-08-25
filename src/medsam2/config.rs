//! Fixed dimensions of the MedSAM2 network.
//!
//! MedSAM2 is SAM 2.1 Hiera-Tiny with the input resolution halved: the
//! published `sam2.1_hiera_t512.yaml` differs from Meta's `sam2.1_hiera_t.yaml`
//! in exactly three values — `image_size` 1024 -> 512 and the two
//! `RoPEAttention.feat_sizes` [64,64] -> [32,32], which is only 512/16 spelled
//! out. Everything below is therefore SAM 2.1-T's own geometry, evaluated at
//! 512.
//!
//! Unlike the sibling engine [`crate::segvol`], none of this is baked
//! into the checkpoint: the weights are resolution-independent, and every
//! quantity that depends on `IMAGE_SIZE` (the interpolated background
//! position embedding, the dense prompt encoding, the RoPE frequency table)
//! is computed at build time. Changing `IMAGE_SIZE` here would therefore
//! produce a *working* network — just not the one MedSAM2 was fine-tuned as.

/// Input edge, in pixels, of one slice as the network sees it.
pub const IMAGE_SIZE: usize = 512;

// ---- image encoder: Hiera-T trunk ---------------------------------------

/// Patch embedding: `Conv2d(3, 96, kernel 7, stride 4, padding 3)`.
pub const PATCH_KERNEL: usize = 7;
pub const PATCH_STRIDE: usize = 4;
pub const PATCH_PADDING: usize = 3;
/// Token grid after the patch embedding, `IMAGE_SIZE / PATCH_STRIDE`.
pub const TRUNK_GRID: usize = IMAGE_SIZE / PATCH_STRIDE;
/// Width of the first stage.
pub const EMBED_DIM: usize = 96;
/// Blocks per stage, `[1, 2, 7, 2]` — twelve in total.
pub const STAGES: [usize; 4] = [1, 2, 7, 2];
pub const NUM_BLOCKS: usize = 12;
/// Index of the last block of each stage, `cumsum(STAGES) - 1`.
pub const STAGE_ENDS: [usize; 4] = [0, 2, 9, 11];
/// Blocks that pool their queries (and their residual) by 2 x 2.
pub const Q_POOL_BLOCKS: [usize; 3] = [1, 3, 10];
/// Attention window per stage. Note the reference's off-by-one-block quirk:
/// the *first* block of a stage keeps the *previous* stage's window.
pub const WINDOW_SPEC: [usize; 4] = [8, 4, 14, 7];
/// Blocks with global (unwindowed) attention.
pub const GLOBAL_ATT_BLOCKS: [usize; 3] = [5, 7, 9];
/// Head width — constant across the trunk, because width and head count
/// double together.
pub const HEAD_DIM: usize = 96;
/// Learned background position embedding, tiled window embedding.
pub const POS_EMBED_BKG: usize = 7;
pub const POS_EMBED_WINDOW: usize = 8;

// ---- image encoder: FPN neck --------------------------------------------

/// Trunk output channels, lowest resolution first — the order the neck's
/// convolutions are declared in.
pub const BACKBONE_CHANNELS: [usize; 4] = [768, 384, 192, 96];
/// Width of everything downstream of the neck.
pub const D_MODEL: usize = 256;
/// Levels the neck emits.
pub const FPN_LEVELS: usize = 4;
/// Levels that receive the top-down addition.
pub const FPN_TOP_DOWN_LEVELS: [usize; 2] = [2, 3];
/// How many of the lowest-resolution levels the image encoder throws away:
/// the neck emits [`FPN_LEVELS`], the model consumes the first
/// `FPN_LEVELS - SCALP`.
pub const SCALP: usize = 1;
/// The levels the model actually consumes.
pub const USED_LEVELS: usize = FPN_LEVELS - SCALP;

/// Grid of the feature map the SAM head and the memory attention consume
/// (FPN level 2, stride 16).
pub const EMBED_GRID: usize = IMAGE_SIZE / 16;
/// Grid of the mask logits the decoder produces (stride 4).
pub const LOW_RES: usize = IMAGE_SIZE / 4;
/// Channels of the two high-resolution features after `conv_s0` / `conv_s1`.
pub const HIGH_RES_S0_CH: usize = 32;
pub const HIGH_RES_S1_CH: usize = 64;

// ---- prompt encoder ------------------------------------------------------

/// Intermediate channels of the mask-prompt downscaling stack.
pub const MASK_IN_CHANS: usize = 16;
/// Edge of the mask a mask prompt is expected at, `4 * EMBED_GRID`.
pub const MASK_PROMPT_SIZE: usize = 4 * EMBED_GRID;
/// Random-Fourier positional encoding: `[2, D_MODEL / 2]`.
pub const PE_GAUSSIAN: usize = D_MODEL / 2;

// ---- mask decoder --------------------------------------------------------

pub const DEC_LAYERS: usize = 2;
pub const DEC_HEADS: usize = 8;
pub const DEC_MLP: usize = 2048;
/// The decoder's cross-attentions run at `D_MODEL / DEC_DOWNSAMPLE`.
pub const DEC_DOWNSAMPLE: usize = 2;
/// One IoU token, four mask tokens, one object-score token.
pub const NUM_MASK_TOKENS: usize = 4;
/// Channels after the two transposed convolutions, `D_MODEL / 8`.
pub const UPSCALED_CH: usize = D_MODEL / 8;
/// Width of the hypernetwork MLPs' output.
pub const HYPER_DIM: usize = UPSCALED_CH;
/// Hidden layers of the IoU prediction head (`D_MODEL` wide, then the
/// per-mask output).
pub const IOU_HEAD_DEPTH: usize = 3;

// ---- memory attention ----------------------------------------------------

pub const MEM_ATTN_LAYERS: usize = 4;
/// One head of width `D_MODEL`. Not a typo: `num_heads: 1`.
pub const MEM_ATTN_HEADS: usize = 1;
pub const MEM_MLP: usize = 2048;
/// Width of a memory entry, and of the object-pointer sub-tokens.
pub const MEM_DIM: usize = 64;
/// The input's positional encoding is added scaled by this, not by 1.
pub const POS_ENC_INPUT_SCALE: f32 = 0.1;
pub const ROPE_THETA: f32 = 10000.0;
/// Temperature of the sine positional encodings.
pub const PE_TEMPERATURE: f32 = 10000.0;

// ---- memory encoder ------------------------------------------------------

/// `MaskDownSampler` stride per layer and total stride: four `k3 s2 p1`
/// convolutions take a 512 mask to the 32 x 32 embedding grid.
pub const MASK_DOWN_LAYERS: usize = 4;
pub const MASK_DOWN_STRIDE: usize = 2;
/// Fuser depth, and the CXBlock's depthwise kernel.
pub const FUSER_LAYERS: usize = 2;
pub const CX_KERNEL: usize = 7;
/// Pointwise expansion inside a CXBlock.
pub const CX_MLP: usize = 4 * D_MODEL;

// ---- tracking ------------------------------------------------------------

/// Spatial memories kept: the most recent `NUM_MASKMEM - 1` tracked slices
/// plus the conditioning slices, addressed through a `[7, 1, 1, 64]` table of
/// temporal encodings.
pub const NUM_MASKMEM: usize = 7;
/// Object pointers carried into the memory attention.
pub const MAX_OBJ_PTRS: usize = 16;
/// A 256-d pointer is split into this many `MEM_DIM`-wide tokens.
pub const PTR_TOKENS: usize = D_MODEL / MEM_DIM;
/// The mask handed to the memory encoder is `sigmoid(x) * SCALE + BIAS`, or —
/// on a prompted slice — the hard binarization `(x > 0) * SCALE + BIAS`.
pub const SIGMOID_SCALE: f32 = 20.0;
pub const SIGMOID_BIAS: f32 = -10.0;
/// Logit written everywhere when the object-score head says "absent".
pub const NO_OBJ_SCORE: f32 = -1024.0;
/// `_use_multimask`: multi-mask output is used when the prompt has between
/// these many points, inclusive — so a click (1) qualifies and a box (2)
/// does not.
pub const MULTIMASK_MIN_PT: usize = 0;
pub const MULTIMASK_MAX_PT: usize = 1;
/// `_dynamic_multimask_via_stability` thresholds.
pub const STABILITY_DELTA: f32 = 0.05;
pub const STABILITY_THRESH: f32 = 0.98;

// ---- preprocessing -------------------------------------------------------

pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

// ---- derived block table -------------------------------------------------

/// One Hiera block's geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Block {
    /// Width in.
    pub dim_in: usize,
    /// Width out — twice `dim_in` on the first block of a new stage.
    pub dim_out: usize,
    pub heads: usize,
    /// Attention window, or 0 for global attention.
    pub window: usize,
    /// Whether queries and the residual are 2 x 2 max-pooled.
    pub q_stride: bool,
    /// Token grid the block consumes.
    pub grid_in: usize,
    /// Token grid it produces.
    pub grid_out: usize,
}

/// The twelve blocks, derived exactly as `Hiera.__init__` derives them.
pub fn blocks() -> Vec<Block> {
    let mut out = Vec::with_capacity(NUM_BLOCKS);
    let mut dim = EMBED_DIM;
    let mut heads = 1;
    let mut cur_stage = 1;
    let mut grid = TRUNK_GRID;
    for i in 0..NUM_BLOCKS {
        let mut dim_out = dim;
        // "lags by a block": the window comes from the stage we are *leaving*.
        let mut window = WINDOW_SPEC[cur_stage - 1];
        if GLOBAL_ATT_BLOCKS.contains(&i) {
            window = 0;
        }
        if i > 0 && STAGE_ENDS.contains(&(i - 1)) {
            dim_out = dim * 2;
            heads *= 2;
            cur_stage += 1;
        }
        let q_stride = Q_POOL_BLOCKS.contains(&i);
        let grid_out = if q_stride { grid / 2 } else { grid };
        out.push(Block {
            dim_in: dim,
            dim_out,
            heads,
            window,
            q_stride,
            grid_in: grid,
            grid_out,
        });
        dim = dim_out;
        grid = grid_out;
    }
    out
}

/// Channels of each trunk stage output, highest resolution first.
pub fn stage_dims() -> [usize; 4] {
    [EMBED_DIM, EMBED_DIM * 2, EMBED_DIM * 4, EMBED_DIM * 8]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_block_table_matches_the_published_schedule() {
        let b = blocks();
        assert_eq!(b.len(), 12);
        // (dim_in, dim_out, heads, window, q_stride, grid_in, grid_out)
        let want = [
            (96, 96, 1, 8, false, 128, 128),
            (96, 192, 2, 8, true, 128, 64),
            (192, 192, 2, 4, false, 64, 64),
            (192, 384, 4, 4, true, 64, 32),
            (384, 384, 4, 14, false, 32, 32),
            (384, 384, 4, 0, false, 32, 32),
            (384, 384, 4, 14, false, 32, 32),
            (384, 384, 4, 0, false, 32, 32),
            (384, 384, 4, 14, false, 32, 32),
            (384, 384, 4, 0, false, 32, 32),
            (384, 768, 8, 14, true, 32, 16),
            (768, 768, 8, 7, false, 16, 16),
        ];
        for (i, w) in want.iter().enumerate() {
            let g = b[i];
            assert_eq!(
                (g.dim_in, g.dim_out, g.heads, g.window, g.q_stride, g.grid_in, g.grid_out),
                *w,
                "block {i}"
            );
        }
    }

    #[test]
    fn every_block_has_the_same_head_width() {
        for (i, b) in blocks().iter().enumerate() {
            assert_eq!(b.dim_out / b.heads, HEAD_DIM, "block {i}");
        }
    }

    #[test]
    fn the_stage_outputs_match_the_necks_channel_list() {
        let dims = stage_dims();
        let mut reversed = BACKBONE_CHANNELS;
        reversed.reverse();
        assert_eq!(dims, reversed);
        let b = blocks();
        for (s, end) in STAGE_ENDS.iter().enumerate() {
            assert_eq!(b[*end].dim_out, dims[s], "stage {s}");
        }
    }

    #[test]
    fn the_grids_halve_once_per_stage() {
        let b = blocks();
        assert_eq!(b[STAGE_ENDS[0]].grid_out, 128);
        assert_eq!(b[STAGE_ENDS[1]].grid_out, 64);
        assert_eq!(b[STAGE_ENDS[2]].grid_out, EMBED_GRID);
        assert_eq!(b[STAGE_ENDS[3]].grid_out, 16);
        assert_eq!(EMBED_GRID, 32);
        assert_eq!(LOW_RES, 128);
        assert_eq!(MASK_PROMPT_SIZE, 128);
    }

    #[test]
    fn the_stage_lengths_agree_with_the_stage_ends() {
        let mut acc = 0;
        for (s, n) in STAGES.iter().enumerate() {
            acc += n;
            assert_eq!(STAGE_ENDS[s], acc - 1);
        }
        assert_eq!(acc, NUM_BLOCKS);
    }
}
