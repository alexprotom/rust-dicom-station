//! Shape-checked access to a checkpoint's tensors.
//!
//! Every weight is fetched by its published key and validated against the
//! shape the architecture expects, so a checkpoint that has drifted fails
//! while the network is being assembled rather than producing a plausible
//! mask from a mis-shaped tensor.
//!
//! This is the generalization of `segvol::params`, which predates it and
//! normalizes SegVol's own `model.` prefix on construction; new engines
//! should use this one and pass a key normalizer if they need it.

use anyhow::{bail, Result};
use std::collections::HashMap;

use super::cache::WTensor;

/// A checkpoint's tensors, keyed by name.
pub struct Params {
    tensors: HashMap<String, WTensor>,
}

impl Params {
    /// Take ownership of a tensor map as-is.
    pub fn new(tensors: HashMap<String, WTensor>) -> Params {
        Params { tensors }
    }

    /// Take ownership of a tensor map, rewriting every key.
    pub fn with_keys(
        tensors: HashMap<String, WTensor>,
        normalize: impl Fn(&str) -> String,
    ) -> Params {
        Params {
            tensors: tensors
                .into_iter()
                .map(|(k, v)| (normalize(&k), v))
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

    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(|s| s.as_str())
    }

    /// Total elements held.
    pub fn elements(&self) -> usize {
        self.tensors.values().map(|t| t.numel()).sum()
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

    /// Fetch a `Linear`'s weight `[out, in]` and its bias.
    pub fn linear(&self, prefix: &str, out: usize, inp: usize) -> Result<(&[f32], &[f32])> {
        Ok((
            self.get(&format!("{prefix}.weight"), &[out, inp])?,
            self.vec(&format!("{prefix}.bias"), out)?,
        ))
    }

    /// Fetch a `Linear`'s weight and its optional bias.
    pub fn linear_opt(
        &self,
        prefix: &str,
        out: usize,
        inp: usize,
    ) -> Result<(&[f32], Option<&[f32]>)> {
        let w = self.get(&format!("{prefix}.weight"), &[out, inp])?;
        let b_key = format!("{prefix}.bias");
        let b = if self.contains(&b_key) {
            Some(self.vec(&b_key, out)?)
        } else {
            None
        };
        Ok((w, b))
    }

    /// Fetch a `LayerNorm`'s (or `LayerNorm2d`'s) affine pair.
    pub fn norm(&self, prefix: &str, n: usize) -> Result<(&[f32], &[f32])> {
        Ok((
            self.get(&format!("{prefix}.weight"), &[n])?,
            self.get(&format!("{prefix}.bias"), &[n])?,
        ))
    }

    /// Fetch a `Conv2d`'s weight `[out, in / groups, k, k]` and its bias.
    pub fn conv2d(
        &self,
        prefix: &str,
        out: usize,
        inp: usize,
        k: usize,
        groups: usize,
    ) -> Result<(&[f32], &[f32])> {
        Ok((
            self.get(&format!("{prefix}.weight"), &[out, inp / groups, k, k])?,
            self.vec(&format!("{prefix}.bias"), out)?,
        ))
    }

    /// Fetch a `ConvTranspose2d`'s weight `[in, out, k, k]` and its bias.
    pub fn conv_transpose2d(
        &self,
        prefix: &str,
        out: usize,
        inp: usize,
        k: usize,
    ) -> Result<(&[f32], &[f32])> {
        Ok((
            self.get(&format!("{prefix}.weight"), &[inp, out, k, k])?,
            self.vec(&format!("{prefix}.bias"), out)?,
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
    fn a_wrong_shape_is_an_error_naming_both_shapes() {
        let p = params(&[("a.weight", vec![4, 5])]);
        let e = p.get("a.weight", &[5, 4]).unwrap_err().to_string();
        assert!(e.contains("[4, 5]") && e.contains("[5, 4]"), "{e}");
        assert!(p.get("a.weight", &[4, 5]).is_ok());
        assert_eq!(p.elements(), 20);
    }

    #[test]
    fn a_missing_tensor_is_an_error_naming_the_key() {
        let p = params(&[]);
        assert!(p.is_empty());
        let e = p.get("nope", &[1]).unwrap_err().to_string();
        assert!(e.contains("nope"), "{e}");
    }

    #[test]
    fn conv_shapes_follow_pytorchs_storage_order() {
        let p = params(&[
            ("c.weight", vec![8, 2, 3, 3]),
            ("c.bias", vec![8]),
            ("d.weight", vec![256, 1, 7, 7]),
            ("d.bias", vec![256]),
            ("t.weight", vec![256, 64, 2, 2]),
            ("t.bias", vec![64]),
        ]);
        assert!(p.conv2d("c", 8, 4, 3, 2).is_ok(), "groups divide the input");
        assert!(p.conv2d("d", 256, 256, 7, 256).is_ok(), "depthwise");
        assert!(p.conv_transpose2d("t", 64, 256, 2).is_ok());
        assert!(p.conv2d("c", 8, 4, 3, 1).is_err(), "groups must be applied");
    }

    #[test]
    fn keys_can_be_rewritten_on_construction() {
        let mut m = HashMap::new();
        m.insert(
            "module.x".to_string(),
            WTensor {
                shape: vec![2],
                data: vec![0.0; 2],
            },
        );
        let p = Params::with_keys(m, |k| {
            k.strip_prefix("module.").unwrap_or(k).to_string()
        });
        assert!(p.contains("x"));
        assert_eq!(p.len(), 1);
        assert_eq!(p.keys().count(), 1);
    }

    #[test]
    fn a_missing_bias_is_optional_only_where_asked() {
        let p = params(&[("q.weight", vec![6, 3])]);
        assert!(p.linear("q", 6, 3).is_err());
        let (w, b) = p.linear_opt("q", 6, 3).unwrap();
        assert_eq!(w.len(), 18);
        assert!(b.is_none());
    }
}
