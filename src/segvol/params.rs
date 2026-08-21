//! Shape-checked access to the checkpoint's tensors.
//!
//! Every weight is fetched by its published key and validated against the
//! shape the architecture expects, so a checkpoint that has drifted fails
//! while the network is being assembled rather than producing a plausible
//! mask from a mis-shaped tensor. Keys are given in normalized form (without
//! the `model.` wrapper prefix); see [`super::layout::normalize_key`].

use anyhow::{bail, Result};
use std::collections::HashMap;

use crate::nn::cache::WTensor;

use super::layout::normalize_key;

/// The checkpoint's tensors, keyed by normalized name.
pub struct Params {
    tensors: HashMap<String, WTensor>,
}

impl Params {
    /// Take ownership of a tensor map, normalizing every key.
    pub fn new(tensors: HashMap<String, WTensor>) -> Params {
        Params {
            tensors: tensors
                .into_iter()
                .map(|(k, v)| (normalize_key(&k).to_string(), v))
                .collect(),
        }
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn contains(&self, key: &str) -> bool {
        self.tensors.contains_key(key)
    }

    /// Fetch a tensor and assert its shape.
    pub fn get(&self, key: &str, shape: &[usize]) -> Result<&[f32]> {
        let Some(t) = self.tensors.get(key) else {
            bail!("checkpoint is missing {key}");
        };
        if t.shape != shape {
            bail!("{key} has shape {:?}, expected {shape:?}", t.shape);
        }
        Ok(&t.data)
    }

    /// Fetch a 1-D tensor of the given length.
    pub fn vec(&self, key: &str, len: usize) -> Result<&[f32]> {
        self.get(key, &[len])
    }

    /// Fetch a `Linear`'s weight `[out, in]` and its bias, if it has one.
    pub fn linear(&self, prefix: &str, out: usize, inp: usize) -> Result<(&[f32], Option<&[f32]>)> {
        let w = self.get(&format!("{prefix}.weight"), &[out, inp])?;
        let b_key = format!("{prefix}.bias");
        let b = if self.contains(&b_key) {
            Some(self.vec(&b_key, out)?)
        } else {
            None
        };
        Ok((w, b))
    }

    /// Fetch a `LayerNorm`'s affine pair, both of `shape`.
    pub fn norm(&self, prefix: &str, shape: &[usize]) -> Result<(&[f32], &[f32])> {
        Ok((
            self.get(&format!("{prefix}.weight"), shape)?,
            self.get(&format!("{prefix}.bias"), shape)?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(items: &[(&str, Vec<usize>)]) -> Params {
        Params::new(
            items
                .iter()
                .map(|(k, s)| {
                    (
                        k.to_string(),
                        WTensor {
                            shape: s.clone(),
                            data: vec![0.0; s.iter().product()],
                        },
                    )
                })
                .collect(),
        )
    }

    #[test]
    fn keys_are_normalized_on_construction() {
        let p = params(&[("model.image_encoder.norm.weight", vec![768])]);
        assert!(p.contains("image_encoder.norm.weight"));
        assert!(!p.contains("model.image_encoder.norm.weight"));
        assert_eq!(p.len(), 1);
        assert!(!p.is_empty());
    }

    #[test]
    fn a_wrong_shape_is_an_error_naming_both_shapes() {
        let p = params(&[("a.weight", vec![4, 5])]);
        let e = p.get("a.weight", &[5, 4]).unwrap_err().to_string();
        assert!(e.contains("[4, 5]") && e.contains("[5, 4]"), "{e}");
        assert!(p.get("a.weight", &[4, 5]).is_ok());
    }

    #[test]
    fn a_missing_tensor_is_an_error_naming_the_key() {
        let p = params(&[]);
        let e = p.get("nope", &[1]).unwrap_err().to_string();
        assert!(e.contains("nope"), "{e}");
    }

    #[test]
    fn linear_reports_a_missing_bias_rather_than_failing() {
        // The image encoder's fused qkv projection genuinely has no bias.
        let p = params(&[
            ("q.weight", vec![6, 3]),
            ("r.weight", vec![6, 3]),
            ("r.bias", vec![6]),
        ]);
        let (w, b) = p.linear("q", 6, 3).unwrap();
        assert_eq!(w.len(), 18);
        assert!(b.is_none());
        let (_, b) = p.linear("r", 6, 3).unwrap();
        assert_eq!(b.unwrap().len(), 6);
    }
}
