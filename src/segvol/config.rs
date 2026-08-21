//! Fixed dimensions of the SegVol network.
//!
//! Every one of these is baked into the published weights. The input shape in
//! particular cannot be varied: the image encoder's position embedding is a
//! learned `[1, 2048, 768]` parameter with no interpolation logic, and the
//! mask decoder's `output_upscaling.1` is a `LayerNorm` whose
//! `normalized_shape` is the literal `(192, 16, 32, 32)` activation. Feeding
//! anything but a 32x256x256 volume is a shape error, not a resize.

/// Input volume accepted by one forward pass, `[axis0, axis1, axis2]`.
pub const ROI: [usize; 3] = [32, 256, 256];
/// Patch size the volume is cut into.
pub const PATCH: [usize; 3] = [4, 16, 16];
/// Patch grid, `ROI / PATCH`.
pub const GRID: [usize; 3] = [8, 16, 16];
/// Tokens per forward pass, `GRID.product()`.
pub const TOKENS: usize = 2048;
/// Values per patch, `PATCH.product() * in_channels`.
pub const PATCH_FEATURES: usize = 1024;

/// Width of the image encoder, the prompt tokens and the mask decoder.
pub const EMBED: usize = 768;
pub const VIT_BLOCKS: usize = 12;
pub const VIT_HEADS: usize = 12;
pub const VIT_MLP: usize = 3072;

pub const DEC_LAYERS: usize = 2;
pub const DEC_HEADS: usize = 8;
pub const DEC_MLP: usize = 2048;
/// The two-way transformer's cross-attentions run at `EMBED / 2`; its
/// self-attention runs at the full width.
pub const DEC_ATTN_DOWNSAMPLE: usize = 2;
/// One IoU token plus `NUM_MASK_TOKENS` mask tokens are prepended to the
/// prompt tokens.
pub const NUM_MASK_TOKENS: usize = 4;

/// Channels after `output_upscaling`, `EMBED / 8`.
pub const UPSCALED_CHANNELS: usize = 96;
/// Spatial shape between the two transposed convolutions — the shape the
/// decoder's LayerNorm is sized for.
pub const FEAT_SHAPE: [usize; 3] = [16, 32, 32];
/// Spatial shape of the logits the decoder produces: full resolution along
/// axis 0 and a quarter in-plane, a consequence of the anisotropic patch.
pub const MASK_SHAPE: [usize; 3] = [32, 64, 64];

/// CLIP ViT-B/32 text tower.
pub const CLIP_WIDTH: usize = 512;
pub const CLIP_LAYERS: usize = 12;
pub const CLIP_HEADS: usize = 8;
pub const CLIP_MLP: usize = 2048;
pub const CLIP_VOCAB: usize = 49408;
pub const CLIP_MAX_POSITIONS: usize = 77;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_dimensions_are_mutually_consistent() {
        for a in 0..3 {
            assert_eq!(ROI[a], GRID[a] * PATCH[a]);
            // two stride-2 transposed convolutions over the token grid
            assert_eq!(FEAT_SHAPE[a], GRID[a] * 2);
            assert_eq!(MASK_SHAPE[a], GRID[a] * 4);
        }
        assert_eq!(TOKENS, GRID[0] * GRID[1] * GRID[2]);
        assert_eq!(PATCH_FEATURES, PATCH[0] * PATCH[1] * PATCH[2]);
        assert_eq!(UPSCALED_CHANNELS, EMBED / 8);
        assert_eq!(VIT_MLP, EMBED * 4);
        // head widths: 64 in the encoder, 96 and 48 in the decoder
        assert_eq!(EMBED / VIT_HEADS, 64);
        assert_eq!(EMBED / DEC_HEADS, 96);
        assert_eq!(EMBED / DEC_ATTN_DOWNSAMPLE / DEC_HEADS, 48);
    }
}
