//! Memory attention: how a slice sees the slices already segmented.
//!
//! Four pre-norm layers, each self-attending over the current slice's 1024
//! image tokens and then cross-attending into the memory bank — the spatial
//! memories of up to seven earlier slices, 64-dimensional, plus a tail of
//! object-pointer tokens. This is the only part of SAM 2 that is inherently
//! sequential, and at a full bank it costs about as much arithmetic as the
//! image encoder.
//!
//! Five details decide whether it is right:
//!
//! * the input adds its positional encoding scaled by **0.1**;
//! * queries never get a positional encoding, keys always do — and the keys'
//!   is added in the **64-dimensional** memory space, *before* `k_proj`;
//!   values get none;
//! * both attentions are single-headed at the full width of 256;
//! * rotary encodings are **tiled** across memory frames, so every frame
//!   reuses the same 1024 spatial rotations and is distinguished only by its
//!   temporal encoding;
//! * the object-pointer tail is excluded from the rotation entirely.

use anyhow::Result;
use burn::tensor::activation::relu;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::nn::params::Params;

use super::config::{self, D_MODEL, MEM_DIM};
use super::layers::{Lin, Norm, SamAttention};
use super::ops;

/// Axial rotary position encodings for one square token grid.
///
/// Real dimensions 0..128 (complex pairs 0..64) carry the **x** rotation and
/// 128..256 the **y** rotation, in the interleaved adjacent-pair convention:
/// pair `j` of a vector is `(v[2j], v[2j+1])`, rotated by `angle[j]`.
pub struct Rope<B: Backend> {
    cos: Tensor<B, 4>,
    sin: Tensor<B, 4>,
    tokens: usize,
}

impl<B: Backend> Rope<B> {
    /// `compute_axial_cis(dim, end_x, end_y, theta)`.
    pub fn new(dim: usize, end_x: usize, end_y: usize, theta: f64, dev: &B::Device) -> Rope<B> {
        let quarter = dim / 4;
        let pairs = 2 * quarter;
        let tokens = end_x * end_y;
        let mut cos = vec![0f32; tokens * pairs];
        let mut sin = vec![0f32; tokens * pairs];
        for t in 0..tokens {
            let tx = (t % end_x) as f64;
            let ty = (t / end_x) as f64;
            for k in 0..quarter {
                let freq = theta.powf(-((4 * k) as f64) / dim as f64);
                for (offset, pos) in [(0, tx), (quarter, ty)] {
                    let angle = pos * freq;
                    cos[t * pairs + offset + k] = angle.cos() as f32;
                    sin[t * pairs + offset + k] = angle.sin() as f32;
                }
            }
        }
        Rope {
            cos: ops::from_slice(&cos, [1, 1, tokens, pairs], dev),
            sin: ops::from_slice(&sin, [1, 1, tokens, pairs], dev),
            tokens,
        }
    }

    pub fn tokens(&self) -> usize {
        self.tokens
    }

    /// Rotate `[b, heads, repeat * tokens, dim]`, reusing the same rotations
    /// for each repeat — which is what makes every memory frame carry the
    /// same spatial encoding.
    pub fn apply(&self, x: Tensor<B, 4>, repeat: usize) -> Tensor<B, 4> {
        let [b, heads, n, dim] = x.dims();
        assert_eq!(
            n,
            repeat * self.tokens,
            "rope covers {} tokens",
            self.tokens
        );
        let pairs = dim / 2;
        let (cos, sin) = if repeat == 1 {
            (self.cos.clone(), self.sin.clone())
        } else {
            (
                self.cos.clone().repeat_dim(2, repeat),
                self.sin.clone().repeat_dim(2, repeat),
            )
        };
        let x = x.reshape([b, heads, n, pairs, 2]);
        let even = x.clone().slice([0..b, 0..heads, 0..n, 0..pairs, 0..1]);
        let odd = x.slice([0..b, 0..heads, 0..n, 0..pairs, 1..2]);
        let cos = cos.reshape([1, 1, n, pairs, 1]);
        let sin = sin.reshape([1, 1, n, pairs, 1]);
        let out_even = even.clone() * cos.clone() - odd.clone() * sin.clone();
        let out_odd = even * sin + odd * cos;
        Tensor::cat(vec![out_even, out_odd], 4).reshape([b, heads, n, dim])
    }
}

/// `RoPEAttention`.
pub struct RopeAttention<B: Backend> {
    attn: SamAttention<B>,
    rope: Rope<B>,
    /// Whether keys longer than the queries tile the rotations (the
    /// cross-attention) or are an error (the self-attention).
    k_repeat: bool,
}

impl<B: Backend> RopeAttention<B> {
    pub fn load(
        p: &Params,
        prefix: &str,
        kv_in: usize,
        k_repeat: bool,
        dev: &B::Device,
    ) -> Result<RopeAttention<B>> {
        Ok(RopeAttention {
            attn: SamAttention::load(p, prefix, D_MODEL, config::MEM_ATTN_HEADS, 1, kv_in, dev)?,
            rope: Rope::new(
                D_MODEL / config::MEM_ATTN_HEADS,
                config::EMBED_GRID,
                config::EMBED_GRID,
                f64::from(config::ROPE_THETA),
                dev,
            ),
            k_repeat,
        })
    }

    /// `num_k_exclude_rope` keys at the **end** of the sequence — the object
    /// pointers — pass through unrotated.
    pub fn forward(
        &self,
        q: Tensor<B, 3>,
        k: Tensor<B, 3>,
        v: Tensor<B, 3>,
        num_k_exclude_rope: usize,
    ) -> Tensor<B, 3> {
        let q = self.attn.split(self.attn.q.apply(q));
        let k = self.attn.split(self.attn.k.apply(k));
        let v = self.attn.split(self.attn.v.apply(v));

        let [b, heads, n_k, dim] = k.dims();
        let n_q = q.dims()[2];
        let q = self.rope.apply(q, 1);

        let n_rope = n_k - num_k_exclude_rope;
        let repeat = if self.k_repeat { n_rope / n_q } else { 1 };
        let rotated = self
            .rope
            .apply(k.clone().slice([0..b, 0..heads, 0..n_rope, 0..dim]), repeat);
        let k = if num_k_exclude_rope == 0 {
            rotated
        } else {
            Tensor::cat(
                vec![rotated, k.slice([0..b, 0..heads, n_rope..n_k, 0..dim])],
                2,
            )
        };

        self.attn.out.apply(self.attn.merge(ops::sdpa(q, k, v)))
    }
}

/// One `MemoryAttentionLayer`.
pub struct MemoryAttentionLayer<B: Backend> {
    self_attn: RopeAttention<B>,
    cross_attn: RopeAttention<B>,
    norm1: Norm<B>,
    norm2: Norm<B>,
    norm3: Norm<B>,
    linear1: Lin<B>,
    linear2: Lin<B>,
}

impl<B: Backend> MemoryAttentionLayer<B> {
    fn load(p: &Params, prefix: &str, dev: &B::Device) -> Result<MemoryAttentionLayer<B>> {
        Ok(MemoryAttentionLayer {
            self_attn: RopeAttention::load(p, &format!("{prefix}.self_attn"), D_MODEL, false, dev)?,
            cross_attn: RopeAttention::load(
                p,
                &format!("{prefix}.cross_attn_image"),
                MEM_DIM,
                true,
                dev,
            )?,
            norm1: Norm::load(p, &format!("{prefix}.norm1"), D_MODEL, dev)?,
            norm2: Norm::load(p, &format!("{prefix}.norm2"), D_MODEL, dev)?,
            norm3: Norm::load(p, &format!("{prefix}.norm3"), D_MODEL, dev)?,
            linear1: Lin::load(
                p,
                &format!("{prefix}.linear1"),
                config::MEM_MLP,
                D_MODEL,
                dev,
            )?,
            linear2: Lin::load(
                p,
                &format!("{prefix}.linear2"),
                D_MODEL,
                config::MEM_MLP,
                dev,
            )?,
        })
    }

    fn forward(
        &self,
        tgt: Tensor<B, 3>,
        memory: &Tensor<B, 3>,
        memory_pos: &Tensor<B, 3>,
        num_obj_ptr_tokens: usize,
    ) -> Tensor<B, 3> {
        // self-attention: no positional encoding on either side
        let t2 = self.norm1.apply(tgt.clone());
        let tgt = tgt + self.self_attn.forward(t2.clone(), t2.clone(), t2, 0);

        // cross-attention: keys carry their encoding, values do not
        let t2 = self.norm2.apply(tgt.clone());
        let keys = memory.clone() + memory_pos.clone();
        let tgt = tgt
            + self
                .cross_attn
                .forward(t2, keys, memory.clone(), num_obj_ptr_tokens);

        // feed-forward
        let t2 = self.norm3.apply(tgt.clone());
        tgt + self.linear2.apply(relu(self.linear1.apply(t2)))
    }
}

/// The four-layer stack.
pub struct MemoryAttention<B: Backend> {
    layers: Vec<MemoryAttentionLayer<B>>,
    norm: Norm<B>,
}

impl<B: Backend> MemoryAttention<B> {
    pub fn load(p: &Params, dev: &B::Device) -> Result<MemoryAttention<B>> {
        let mut layers = Vec::with_capacity(config::MEM_ATTN_LAYERS);
        for i in 0..config::MEM_ATTN_LAYERS {
            layers.push(MemoryAttentionLayer::load(
                p,
                &format!("memory_attention.layers.{i}"),
                dev,
            )?);
        }
        Ok(MemoryAttention {
            layers,
            norm: Norm::load(p, "memory_attention.norm", D_MODEL, dev)?,
        })
    }

    /// Condition `curr` (`[1, tokens, 256]`) on `memory` (`[1, entries, 64]`).
    ///
    /// The last `num_obj_ptr_tokens` memory entries are object pointers; the
    /// rest must be a whole number of spatial frames.
    pub fn forward(
        &self,
        curr: Tensor<B, 3>,
        curr_pos: Tensor<B, 3>,
        memory: Tensor<B, 3>,
        memory_pos: Tensor<B, 3>,
        num_obj_ptr_tokens: usize,
    ) -> Tensor<B, 3> {
        let mut out = curr + curr_pos.mul_scalar(config::POS_ENC_INPUT_SCALE);
        for layer in &self.layers {
            out = layer.forward(out, &memory, &memory_pos, num_obj_ptr_tokens);
        }
        self.norm.apply(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::cache::load_safetensors;
    use std::path::Path;

    type Bk = burn::backend::NdArray;

    fn fixture(name: &str, dev: &burn::tensor::Device<Bk>) -> Tensor<Bk, 4> {
        let f = load_safetensors(Path::new("tests/data/medsam2-ops.safetensors")).unwrap();
        let t = f.get(name).unwrap_or_else(|| panic!("fixture {name}"));
        let mut shape = [1usize; 4];
        shape.copy_from_slice(&t.shape);
        ops::from_slice(&t.data, shape, dev)
    }

    fn worst(a: Tensor<Bk, 4>, b: Tensor<Bk, 4>) -> f32 {
        assert_eq!(a.dims(), b.dims());
        ops::to_vec(a)
            .iter()
            .zip(ops::to_vec(b).iter())
            .map(|(x, y)| (x - y).abs() / (1.0 + y.abs()))
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn the_rotation_matches_the_reference() {
        let dev: burn::tensor::Device<Bk> = Default::default();
        // head dim 16 over a 4 x 3 grid, as the fixture was generated
        let rope = Rope::<Bk>::new(16, 4, 3, 10000.0, &dev);
        assert_eq!(rope.tokens(), 12);
        let q = fixture("rope.q", &dev);
        let k = fixture("rope.k", &dev);
        assert!(worst(rope.apply(q, 1), fixture("rope.q_out", &dev)) < 1e-6);
        assert!(worst(rope.apply(k, 1), fixture("rope.k_out", &dev)) < 1e-6);
    }

    #[test]
    fn tiling_the_rotation_matches_repeat_freqs_k() {
        let dev: burn::tensor::Device<Bk> = Default::default();
        let rope = Rope::<Bk>::new(16, 4, 3, 10000.0, &dev);
        // keys three times as long: three "memory frames" of the same grid
        let k = fixture("rope_repeat.k", &dev);
        assert_eq!(k.dims()[2], 36);
        assert!(worst(rope.apply(k, 3), fixture("rope_repeat.k_out", &dev)) < 1e-6);
    }

    #[test]
    fn the_frequency_table_puts_x_before_y() {
        let dev: burn::tensor::Device<Bk> = Default::default();
        let rope = Rope::<Bk>::new(16, 4, 3, 10000.0, &dev);
        // token 7 is (x = 3, y = 1); pairs 0..4 rotate by x, 4..8 by y, and
        // the first frequency of each half is 1.0
        let cos = ops::to_vec(rope.cos.clone());
        let pairs = 8;
        assert!((cos[7 * pairs] - 3.0f32.cos()).abs() < 1e-6);
        assert!((cos[7 * pairs + 4] - 1.0f32.cos()).abs() < 1e-6);
    }
}
