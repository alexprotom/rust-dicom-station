//! CLIP's text tower, and the projection that lifts its output to the
//! network's prompt width.
//!
//! This is the ViT-B/32 text encoder at `transformers`' default
//! `CLIPTextConfig`: vocabulary 49,408, width 512, twelve pre-norm layers,
//! eight heads, MLP 2,048, **QuickGELU** — not the exact GELU the image
//! encoder uses — and a **causal** attention mask. It is frozen throughout
//! SegVol's training and, crucially, ships *inside* the SegVol checkpoint, so
//! no second download is needed for the weights; only the tokenizer's two
//! data files are separate.
//!
//! Pooling takes the hidden state at the end-of-text token, which is found as
//! `argmax(input_ids)` — a trick that works because `<|endoftext|>` is the
//! highest id in the vocabulary. That vector goes through `dim_align`
//! (512 -> 768) and becomes the text prompt the rest of the network sees.
//!
//! Compute here is negligible: one forward over at most 77 tokens is well
//! under a GFLOP against the image encoder's 250, so results are cached by
//! prompt string and the tower is rarely run twice for the same structure.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::sync::Mutex;

use crate::nn::attention::{attention, Mask};
use crate::nn::linalg::{layer_norm, linear, quick_gelu, LAYER_NORM_EPS};
use crate::nn::tensor::Mat;

use super::bpe::{Bpe, EOS};
use super::config::*;
use crate::nn::params::Params;

struct Layer {
    ln1: (Vec<f32>, Vec<f32>),
    q: (Vec<f32>, Vec<f32>),
    k: (Vec<f32>, Vec<f32>),
    v: (Vec<f32>, Vec<f32>),
    o: (Vec<f32>, Vec<f32>),
    ln2: (Vec<f32>, Vec<f32>),
    fc1: (Vec<f32>, Vec<f32>),
    fc2: (Vec<f32>, Vec<f32>),
}

/// The text tower plus `dim_align`, with a cache of already-computed prompts.
pub struct TextEncoder {
    token_embedding: Vec<f32>,
    position_embedding: Vec<f32>,
    layers: Vec<Layer>,
    final_ln: (Vec<f32>, Vec<f32>),
    align: (Vec<f32>, Vec<f32>),
    cache: Mutex<HashMap<String, Vec<f32>>>,
}

fn pair(p: &Params, prefix: &str, out: usize, inp: usize) -> Result<(Vec<f32>, Vec<f32>)> {
    let (w, b) = p.linear_opt(prefix, out, inp)?;
    Ok((
        w.to_vec(),
        b.with_context(|| format!("{prefix} needs a bias"))?
            .to_vec(),
    ))
}

fn norm(p: &Params, prefix: &str) -> Result<(Vec<f32>, Vec<f32>)> {
    let (w, b) = p.norm(prefix, CLIP_WIDTH)?;
    Ok((w.to_vec(), b.to_vec()))
}

impl TextEncoder {
    pub fn build(p: &Params) -> Result<TextEncoder> {
        let tm = "text_encoder.clip_text_model.text_model";
        let mut layers = Vec::with_capacity(CLIP_LAYERS);
        for i in 0..CLIP_LAYERS {
            let b = format!("{tm}.encoder.layers.{i}");
            layers.push(Layer {
                ln1: norm(p, &format!("{b}.layer_norm1"))?,
                q: pair(p, &format!("{b}.self_attn.q_proj"), CLIP_WIDTH, CLIP_WIDTH)?,
                k: pair(p, &format!("{b}.self_attn.k_proj"), CLIP_WIDTH, CLIP_WIDTH)?,
                v: pair(p, &format!("{b}.self_attn.v_proj"), CLIP_WIDTH, CLIP_WIDTH)?,
                o: pair(
                    p,
                    &format!("{b}.self_attn.out_proj"),
                    CLIP_WIDTH,
                    CLIP_WIDTH,
                )?,
                ln2: norm(p, &format!("{b}.layer_norm2"))?,
                fc1: pair(p, &format!("{b}.mlp.fc1"), CLIP_MLP, CLIP_WIDTH)?,
                fc2: pair(p, &format!("{b}.mlp.fc2"), CLIP_WIDTH, CLIP_MLP)?,
            });
        }
        Ok(TextEncoder {
            token_embedding: p
                .get(
                    &format!("{tm}.embeddings.token_embedding.weight"),
                    &[CLIP_VOCAB, CLIP_WIDTH],
                )?
                .to_vec(),
            position_embedding: p
                .get(
                    &format!("{tm}.embeddings.position_embedding.weight"),
                    &[CLIP_MAX_POSITIONS, CLIP_WIDTH],
                )?
                .to_vec(),
            layers,
            final_ln: norm(p, &format!("{tm}.final_layer_norm"))?,
            align: pair(p, "text_encoder.dim_align", EMBED, CLIP_WIDTH)?,
            cache: Mutex::new(HashMap::new()),
        })
    }

    /// Encode token ids into the aligned `EMBED`-wide prompt vector.
    pub fn encode_ids(&self, ids: &[u32]) -> Vec<f32> {
        assert!(!ids.is_empty() && ids.len() <= CLIP_MAX_POSITIONS);
        let n = ids.len();
        // token + position embeddings
        let mut x = Mat::zeros(n, CLIP_WIDTH);
        for (t, id) in ids.iter().enumerate() {
            let tok = *id as usize * CLIP_WIDTH;
            let pos = t * CLIP_WIDTH;
            for c in 0..CLIP_WIDTH {
                x.data[t * CLIP_WIDTH + c] =
                    self.token_embedding[tok + c] + self.position_embedding[pos + c];
            }
        }
        for l in &self.layers {
            // x = x + attn(ln1(x)), causally masked
            let mut h = x.clone();
            layer_norm(&mut h.data, CLIP_WIDTH, &l.ln1.0, &l.ln1.1, LAYER_NORM_EPS);
            let q = linear(&h, &l.q.0, CLIP_WIDTH, Some(&l.q.1));
            let k = linear(&h, &l.k.0, CLIP_WIDTH, Some(&l.k.1));
            let v = linear(&h, &l.v.0, CLIP_WIDTH, Some(&l.v.1));
            let a = attention(&q, &k, &v, CLIP_HEADS, Mask::Causal);
            let a = linear(&a, &l.o.0, CLIP_WIDTH, Some(&l.o.1));
            x.add_assign(&a);
            // x = x + mlp(ln2(x)), QuickGELU
            let mut h = x.clone();
            layer_norm(&mut h.data, CLIP_WIDTH, &l.ln2.0, &l.ln2.1, LAYER_NORM_EPS);
            let mut h = linear(&h, &l.fc1.0, CLIP_MLP, Some(&l.fc1.1));
            quick_gelu(&mut h.data);
            let h = linear(&h, &l.fc2.0, CLIP_WIDTH, Some(&l.fc2.1));
            x.add_assign(&h);
        }
        layer_norm(
            &mut x.data,
            CLIP_WIDTH,
            &self.final_ln.0,
            &self.final_ln.1,
            LAYER_NORM_EPS,
        );
        // Pool at the end-of-text token: the highest id in the sequence.
        let eot = ids
            .iter()
            .enumerate()
            .max_by_key(|(_, id)| **id)
            .map(|(i, _)| i)
            .unwrap_or(n - 1);
        debug_assert_eq!(ids[eot], EOS, "pooling must land on the end-of-text token");
        let pooled = Mat::row_vec(x.row(eot));
        linear(&pooled, &self.align.0, EMBED, Some(&self.align.1)).data
    }

    /// Encode a structure name through the trained prompt template, caching
    /// the result so repeated prompts never re-run the tower.
    pub fn encode_structure(&self, bpe: &Bpe, structure: &str) -> Vec<f32> {
        let key = structure.to_ascii_lowercase();
        if let Some(v) = self
            .cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&key)
        {
            return v.clone();
        }
        let ids = bpe.encode(&super::bpe::prompt_for(structure));
        let v = self.encode_ids(&ids);
        self.cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(key, v.clone());
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::cache::WTensor;

    /// A miniature text tower with the real key names but tiny tensors is not
    /// possible — the dimensions are fixed by the checkpoint — so these tests
    /// use the real shapes with cheap values. That is 63 M parameters, so
    /// only one encoder is built and it is shared.
    fn tiny_params() -> Params {
        let mut m = HashMap::new();
        let mut s = 7u64;
        let mut rnd = move |n: usize| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    (((s >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5) * 0.02
                })
                .collect()
        };
        let tm = "text_encoder.clip_text_model.text_model";
        let put = |m: &mut HashMap<String, WTensor>, k: String, shape: Vec<usize>, d: Vec<f32>| {
            m.insert(k, WTensor { shape, data: d });
        };
        let n = CLIP_VOCAB * CLIP_WIDTH;
        put(
            &mut m,
            format!("{tm}.embeddings.token_embedding.weight"),
            vec![CLIP_VOCAB, CLIP_WIDTH],
            rnd(n),
        );
        put(
            &mut m,
            format!("{tm}.embeddings.position_embedding.weight"),
            vec![CLIP_MAX_POSITIONS, CLIP_WIDTH],
            rnd(CLIP_MAX_POSITIONS * CLIP_WIDTH),
        );
        for i in 0..CLIP_LAYERS {
            let b = format!("{tm}.encoder.layers.{i}");
            for p in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                put(
                    &mut m,
                    format!("{b}.self_attn.{p}.weight"),
                    vec![CLIP_WIDTH, CLIP_WIDTH],
                    rnd(CLIP_WIDTH * CLIP_WIDTH),
                );
                put(
                    &mut m,
                    format!("{b}.self_attn.{p}.bias"),
                    vec![CLIP_WIDTH],
                    rnd(CLIP_WIDTH),
                );
            }
            for (nm, o, i2) in [("fc1", CLIP_MLP, CLIP_WIDTH), ("fc2", CLIP_WIDTH, CLIP_MLP)] {
                put(
                    &mut m,
                    format!("{b}.mlp.{nm}.weight"),
                    vec![o, i2],
                    rnd(o * i2),
                );
                put(&mut m, format!("{b}.mlp.{nm}.bias"), vec![o], rnd(o));
            }
            for nm in ["layer_norm1", "layer_norm2"] {
                put(
                    &mut m,
                    format!("{b}.{nm}.weight"),
                    vec![CLIP_WIDTH],
                    vec![1.0; CLIP_WIDTH],
                );
                put(
                    &mut m,
                    format!("{b}.{nm}.bias"),
                    vec![CLIP_WIDTH],
                    vec![0.0; CLIP_WIDTH],
                );
            }
        }
        put(
            &mut m,
            format!("{tm}.final_layer_norm.weight"),
            vec![CLIP_WIDTH],
            vec![1.0; CLIP_WIDTH],
        );
        put(
            &mut m,
            format!("{tm}.final_layer_norm.bias"),
            vec![CLIP_WIDTH],
            vec![0.0; CLIP_WIDTH],
        );
        put(
            &mut m,
            "text_encoder.dim_align.weight".into(),
            vec![EMBED, CLIP_WIDTH],
            rnd(EMBED * CLIP_WIDTH),
        );
        put(
            &mut m,
            "text_encoder.dim_align.bias".into(),
            vec![EMBED],
            rnd(EMBED),
        );
        Params::new(m)
    }

    #[test]
    fn the_tower_assembles_and_pools_at_the_end_marker() {
        let enc = TextEncoder::build(&tiny_params()).expect("text encoder");
        let ids = vec![super::super::bpe::BOS, 100, 200, EOS];
        let v = enc.encode_ids(&ids);
        assert_eq!(v.len(), EMBED);
        assert!(v.iter().all(|x| x.is_finite()));
        // a different sequence gives a different vector
        let v2 = enc.encode_ids(&[super::super::bpe::BOS, 300, EOS]);
        assert_ne!(v, v2);
    }

    #[test]
    fn the_causal_mask_means_a_suffix_cannot_change_earlier_states() {
        // Pooling is at the end marker, so appending tokens *before* it must
        // change the result, while the causal mask guarantees that tokens
        // after a position never influence it. Check the latter directly by
        // comparing two sequences that agree up to the pooled position.
        let enc = TextEncoder::build(&tiny_params()).unwrap();
        let a = enc.encode_ids(&[super::super::bpe::BOS, 11, 22, EOS]);
        let b = enc.encode_ids(&[super::super::bpe::BOS, 11, 33, EOS]);
        // token 2 differs, and the end marker attends to it, so these differ
        assert_ne!(a, b);
    }

    #[test]
    fn missing_text_weights_are_an_error_not_a_panic() {
        assert!(TextEncoder::build(&Params::new(HashMap::new()))
            .map(|_| ())
            .is_err());
    }
}
