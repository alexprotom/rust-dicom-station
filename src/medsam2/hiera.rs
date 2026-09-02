//! The image encoder's trunk: Hiera-T.
//!
//! A hierarchical vision transformer in four stages. Unlike a plain ViT the
//! token grid shrinks - 128² -> 64² -> 32² -> 16² for a 512 x 512 slice - while
//! the width doubles, and most blocks attend inside a window rather than
//! globally. Three details of the reference are easy to miss and expensive to
//! get wrong, so they are called out where they happen below:
//!
//! 1. the residual projection of a width-changing block consumes the
//!    **normalized** activation, not the block input;
//! 2. that residual is **max-pooled** with the same 2 x 2 pooling as the
//!    queries, so the skip connection lands on the smaller grid;
//! 3. the attention window of a block "lags by one" - the first block of a
//!    new stage keeps the previous stage's window (see
//!    [`super::config::blocks`]) - and after query pooling the window used to
//!    put the tiles back is half the one used to cut them.
//!
//! Tokens are carried as `[batch, h, w, channels]` throughout, as in the
//! reference; only the stage outputs are permuted to `[n, c, h, w]` for the
//! neck.

use anyhow::Result;
use burn::tensor::activation::gelu;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::nn::params::Params;

use super::layers::{Lin, Norm};

use super::config::{self, Block};
use super::ops;

/// One `MultiScaleBlock`.
struct HieraBlock<B: Backend> {
    spec: Block,
    norm1: Norm<B>,
    qkv: Lin<B>,
    proj: Lin<B>,
    norm2: Norm<B>,
    mlp0: Lin<B>,
    mlp1: Lin<B>,
    /// Present exactly on the blocks that change width.
    residual: Option<Lin<B>>,
}

/// `[b, h, w, c]` -> 2 x 2 max pool -> `[b, h/2, w/2, c]`.
fn pool_hwc<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    ops::max_pool2x2(x.permute([0, 3, 1, 2])).permute([0, 2, 3, 1])
}

impl<B: Backend> HieraBlock<B> {
    fn load(p: &Params, i: usize, spec: Block, dev: &B::Device) -> Result<HieraBlock<B>> {
        let base = format!("image_encoder.trunk.blocks.{i}");
        let (di, dout) = (spec.dim_in, spec.dim_out);
        Ok(HieraBlock {
            spec,
            norm1: Norm::load6(p, &format!("{base}.norm1"), di, dev)?,
            qkv: Lin::load(p, &format!("{base}.attn.qkv"), 3 * dout, di, dev)?,
            proj: Lin::load(p, &format!("{base}.attn.proj"), dout, dout, dev)?,
            norm2: Norm::load6(p, &format!("{base}.norm2"), dout, dev)?,
            mlp0: Lin::load(p, &format!("{base}.mlp.layers.0"), 4 * dout, dout, dev)?,
            mlp1: Lin::load(p, &format!("{base}.mlp.layers.1"), dout, 4 * dout, dev)?,
            residual: if di != dout {
                Some(Lin::load(p, &format!("{base}.proj"), dout, di, dev)?)
            } else {
                None
            },
        })
    }

    /// `MultiScaleAttention`: one projection to q, k and v, an optional 2 x 2
    /// pooling of the queries only, then attention per head.
    fn attention(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [b, h, w, _] = x.dims();
        let (heads, dout) = (self.spec.heads, self.spec.dim_out);
        let hd = dout / heads;
        let n = h * w;

        let qkv = self
            .qkv
            .apply(x.reshape([b, n, self.spec.dim_in]))
            .reshape([b, n, 3, heads, hd]);
        let take = |i: usize| {
            qkv.clone()
                .slice([0..b, 0..n, i..i + 1, 0..heads, 0..hd])
                .reshape([b, n, heads, hd])
        };
        let (q, k, v) = (take(0), take(1), take(2));

        // Queries are pooled with the heads folded back into the channel
        // axis; max pooling is per channel, so that is exact.
        let (q, hq, wq) = if self.spec.q_stride {
            let pooled = pool_hwc(q.reshape([b, h, w, heads * hd]));
            let [_, h2, w2, _] = pooled.dims();
            (pooled.reshape([b, h2 * w2, heads, hd]), h2, w2)
        } else {
            (q, h, w)
        };

        let out = ops::sdpa(q.swap_dims(1, 2), k.swap_dims(1, 2), v.swap_dims(1, 2));
        self.proj
            .apply(out.swap_dims(1, 2).reshape([b, hq, wq, heads * hd]))
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [_, h, w, _] = x.dims();
        let normed = self.norm1.apply(x.clone());

        // (1) and (2): the projection sees the normalized activation, and the
        // result is pooled exactly as the queries are.
        let shortcut = match &self.residual {
            None => x,
            Some(res) => {
                let projected = res.apply(normed.clone());
                if self.spec.q_stride {
                    pool_hwc(projected)
                } else {
                    projected
                }
            }
        };

        let (windowed, pad) = if self.spec.window > 0 {
            let (tiles, pad) = ops::window_partition(normed, self.spec.window);
            (tiles, pad)
        } else {
            (normed, [h, w])
        };
        let attended = self.attention(windowed);

        // (3): after pooling, both the window and the grid have halved, and
        // the padding has to be recomputed for the smaller window.
        let mut window = self.spec.window;
        let mut pad_hw = pad;
        let mut hw = [h, w];
        if self.spec.q_stride {
            window /= 2;
            let [_, hs, ws, _] = shortcut.dims();
            hw = [hs, ws];
            if window > 0 {
                let ph = (window - hs % window) % window;
                let pw = (window - ws % window) % window;
                pad_hw = [hs + ph, ws + pw];
            }
        }
        let attended = if self.spec.window > 0 {
            ops::window_unpartition(attended, window, pad_hw, hw)
        } else {
            attended
        };

        let x = shortcut + attended;
        let mlp = self
            .mlp1
            .apply(gelu(self.mlp0.apply(self.norm2.apply(x.clone()))));
        x + mlp
    }
}

/// The trunk, weights resident.
pub struct Hiera<B: Backend> {
    patch_weight: Tensor<B, 4>,
    patch_bias: Tensor<B, 1>,
    /// The position embedding, already interpolated, tiled and permuted to
    /// `[1, grid, grid, EMBED_DIM]`. It depends only on the weights and the
    /// input size, so it is built once - which also keeps the one bicubic
    /// interpolation in the whole engine off the hot path.
    pos_embed: Tensor<B, 4>,
    blocks: Vec<HieraBlock<B>>,
}

impl<B: Backend> Hiera<B> {
    pub fn load(p: &Params, dev: &B::Device) -> Result<Hiera<B>> {
        let t = "image_encoder.trunk";
        let (pw, pb) = p.conv2d(
            &format!("{t}.patch_embed.proj"),
            config::EMBED_DIM,
            3,
            config::PATCH_KERNEL,
            1,
        )?;
        let blocks = config::blocks()
            .into_iter()
            .enumerate()
            .map(|(i, spec)| HieraBlock::load(p, i, spec, dev))
            .collect::<Result<Vec<_>>>()?;
        Ok(Hiera {
            patch_weight: ops::from_slice(
                pw,
                [
                    config::EMBED_DIM,
                    3,
                    config::PATCH_KERNEL,
                    config::PATCH_KERNEL,
                ],
                dev,
            ),
            patch_bias: ops::from_slice(pb, [config::EMBED_DIM], dev),
            pos_embed: Self::build_pos_embed(p, dev)?,
            blocks,
        })
    }

    /// `pos_embed` bicubically stretched over the token grid, plus
    /// `pos_embed_window` **tiled** over it - periodic, not interpolated.
    fn build_pos_embed(p: &Params, dev: &B::Device) -> Result<Tensor<B, 4>> {
        let (c, bkg, win, grid) = (
            config::EMBED_DIM,
            config::POS_EMBED_BKG,
            config::POS_EMBED_WINDOW,
            config::TRUNK_GRID,
        );
        let background: Tensor<B, 4> = ops::from_slice(
            p.get("image_encoder.trunk.pos_embed", &[1, c, bkg, bkg])?,
            [1, c, bkg, bkg],
            dev,
        );
        let window: Tensor<B, 4> = ops::from_slice(
            p.get("image_encoder.trunk.pos_embed_window", &[1, c, win, win])?,
            [1, c, win, win],
            dev,
        );
        let background = ops::resize_bicubic(background, [grid, grid]);
        let tiles = grid / win;
        let window = window.repeat_dim(2, tiles).repeat_dim(3, tiles);
        Ok((background + window).permute([0, 2, 3, 1]))
    }

    /// Encode one slice, returning the four stage outputs as `[n, c, h, w]`,
    /// highest resolution first.
    pub fn forward(&self, image: Tensor<B, 4>) -> Vec<Tensor<B, 4>> {
        let x = ops::conv2d(
            image,
            &self.patch_weight,
            Some(&self.patch_bias),
            config::PATCH_STRIDE,
            config::PATCH_PADDING,
            1,
        );
        let mut x = x.permute([0, 2, 3, 1]) + self.pos_embed.clone();
        let mut out = Vec::with_capacity(4);
        for (i, block) in self.blocks.iter().enumerate() {
            x = block.forward(x);
            if config::STAGE_ENDS.contains(&i) {
                out.push(x.clone().permute([0, 3, 1, 2]));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Bk = burn::backend::NdArray;

    #[test]
    fn tiling_is_periodic_not_stretched() {
        let dev: burn::tensor::Device<Bk> = Default::default();
        let x: Tensor<Bk, 4> = ops::from_slice(&[1.0, 2.0, 3.0, 4.0], [1, 1, 2, 2], &dev);
        let tiled = x.repeat_dim(2, 2).repeat_dim(3, 2);
        assert_eq!(tiled.dims(), [1, 1, 4, 4]);
        assert_eq!(
            ops::to_vec(tiled),
            vec![
                1.0, 2.0, 1.0, 2.0, //
                3.0, 4.0, 3.0, 4.0, //
                1.0, 2.0, 1.0, 2.0, //
                3.0, 4.0, 3.0, 4.0,
            ]
        );
    }

    #[test]
    fn pooling_a_channels_last_tensor_pools_the_grid_not_the_channels() {
        let dev: burn::tensor::Device<Bk> = Default::default();
        // [1, 2, 2, 3]: a 2 x 2 grid of 3-channel tokens
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let x: Tensor<Bk, 4> = ops::from_slice(&data, [1, 2, 2, 3], &dev);
        let pooled = pool_hwc(x);
        assert_eq!(pooled.dims(), [1, 1, 1, 3]);
        // per channel maxima are the last token's values
        assert_eq!(ops::to_vec(pooled), vec![9.0, 10.0, 11.0]);
    }
}
