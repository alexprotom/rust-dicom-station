//! What a TotalSynth checkpoint holds, and the network read out of it.
//!
//! The network is `segmentation_models_pytorch.UnetPlusPlus(encoder_name=
//! "resnet34", in_channels=5, classes=1, activation=None)` followed by a
//! `Tanh`. KonfAI writes it under three sets of names, all the same layers
//! in the same order:
//!
//! * smp's own (`encoder.layer2.0.downsample.0.weight`,
//!   `decoder.blocks.x_0_1.conv1.1.running_var`, `segmentation_head.0.bias`),
//!   from the `Model.py` form (`model.` in front, under the network's name);
//! * the declarative `UNetPlusPlus.yml` the ImpactSynth apps predict with
//!   (`Stem.Conv_0.weight`, `Encoder_2.Block_0.SkipNorm.bias`,
//!   `x_0_1.Conv.Norm_1.running_var`, `SegmentationHead.weight`);
//! * KonfAI's parametric `UNetPlusPlus` (`StemConv.weight`,
//!   `Encoder_2_Block_0.Conv1.weight`, `x_0_1_conv.Conv_0.weight`).
//!
//! Whatever leads them (`Model.UNetPlusPlus.`, `UNetpp5.model.`) is dropped.
//! Every convolution but the head's is followed by a batch norm, which is
//! folded into the convolution here, once, at conversion: the cache holds
//! plain convolutions under one naming of its own ([`canonical`]) and the
//! networks have no norm layers.
//!
//! The widths are read from the weights rather than assumed, so the
//! miniature network the tests carry (the same graph at a fraction of the
//! width, one block per stage but a two-block third stage) runs through the
//! same code as the published 26.1 M-parameter one.

use std::collections::{BTreeMap, HashMap};

use anyhow::{bail, Context, Result};

use crate::nn::cache::WTensor;
use crate::nn::params::Params;

/// `nn.BatchNorm2d`'s default eps.
const BN_EPS: f64 = 1e-5;

/// The decoder of a five-feature (depth-4) UNet++: nodes `x_{d}_{l}` in
/// smp's forward order, each with the column `l` and row `d` it sits at.
pub const NODES: [(usize, usize); 11] = [
    (0, 0),
    (1, 1),
    (2, 2),
    (3, 3),
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 2),
    (1, 3),
    (0, 3),
    (0, 4),
];

/// A batch-norm statistic or affine parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Bn {
    Weight,
    Bias,
    Mean,
    Var,
}

/// What one checkpoint tensor is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Role {
    ConvWeight,
    ConvBias,
    Norm(Bn),
}

fn bn_role(leaf: &str) -> Option<Role> {
    Some(Role::Norm(match leaf {
        "weight" => Bn::Weight,
        "bias" => Bn::Bias,
        "running_mean" => Bn::Mean,
        "running_var" => Bn::Var,
        _ => return None,
    }))
}

fn conv_role(leaf: &str) -> Option<Role> {
    match leaf {
        "weight" => Some(Role::ConvWeight),
        "bias" => Some(Role::ConvBias),
        _ => None,
    }
}

/// `x_3_1` → `(3, 1)`.
fn node_of(s: &str) -> Option<(usize, usize)> {
    let rest = s.strip_prefix("x_")?;
    let (d, l) = rest.split_once('_')?;
    Some((d.parse().ok()?, l.parse().ok()?))
}

/// `"1"` → 1, for the trailing digit of `conv1` / `bn2` / `Conv_0`.
fn digit(s: &str) -> Option<usize> {
    s.parse().ok()
}

/// Where a checkpoint's own naming starts: the first component one of the
/// three namings begins with. What comes before it (the network's name,
/// smp's `model.`) is a wrapper.
fn anchored(key: &str) -> Option<Vec<&str>> {
    let parts: Vec<&str> = key.split('.').collect();
    let start = parts.iter().position(|p| {
        matches!(
            *p,
            "encoder" | "decoder" | "segmentation_head" | "Stem" | "StemConv" | "StemNorm"
        ) || *p == "SegmentationHead"
            || p.starts_with("Encoder_")
            || node_of(p.trim_end_matches("_conv")).is_some()
    })?;
    Some(parts[start..].to_vec())
}

/// The canonical layer a checkpoint key belongs to and what it is, or
/// `None` for anything that is not a weight of the network
/// (`num_batches_tracked`).
fn classify(key: &str) -> Option<(String, Role)> {
    let p = anchored(key)?;
    match p.as_slice() {
        // smp
        ["encoder", "conv1", l] => Some(("stem".into(), conv_role(l)?)),
        ["encoder", "bn1", l] => Some(("stem".into(), bn_role(l)?)),
        ["encoder", layer, b, kind, l] => {
            let s: usize = layer.strip_prefix("layer")?.parse().ok()?;
            let b: usize = b.parse().ok()?;
            let (name, role) = if let Some(k) = kind.strip_prefix("conv") {
                (format!("conv{}", digit(k)?), conv_role(l)?)
            } else {
                let k = kind.strip_prefix("bn")?;
                (format!("conv{}", digit(k)?), bn_role(l)?)
            };
            Some((format!("enc{s}.{b}.{name}"), role))
        }
        ["encoder", layer, b, "downsample", i, l] => {
            let s: usize = layer.strip_prefix("layer")?.parse().ok()?;
            let role = match *i {
                "0" => conv_role(l)?,
                "1" => bn_role(l)?,
                _ => return None,
            };
            Some((format!("enc{s}.{b}.skip"), role))
        }
        ["decoder", "blocks", node, conv, i, l] => {
            let (d, n) = node_of(node)?;
            let k = digit(conv.strip_prefix("conv")?)?;
            let role = match *i {
                "0" => conv_role(l)?,
                "1" => bn_role(l)?,
                _ => return None,
            };
            Some((format!("x_{d}_{n}.conv{k}"), role))
        }
        ["segmentation_head", "0", l] => Some(("head".into(), conv_role(l)?)),
        // KonfAI, declarative
        ["Stem", "Conv_0", l] | ["StemConv", l] => Some(("stem".into(), conv_role(l)?)),
        ["Stem", "Norm_0", l] | ["StemNorm", l] => Some(("stem".into(), bn_role(l)?)),
        ["SegmentationHead", l] => Some(("head".into(), conv_role(l)?)),
        [enc, block, kind, l] if enc.starts_with("Encoder_") => {
            let s: usize = enc.strip_prefix("Encoder_")?.parse().ok()?;
            let b: usize = block.strip_prefix("Block_")?.parse().ok()?;
            block_layer(s, b, kind, l)
        }
        // KonfAI, parametric: `Encoder_2_Block_0.Conv1.weight`
        [enc, kind, l] if enc.starts_with("Encoder_") => {
            let (s, b) = enc.strip_prefix("Encoder_")?.split_once("_Block_")?;
            block_layer(s.parse().ok()?, b.parse().ok()?, kind, l)
        }
        [node, "Conv", layer, l] | [node, layer, l]
            if node_of(node.trim_end_matches("_conv")).is_some() =>
        {
            let (d, n) = node_of(node.trim_end_matches("_conv"))?;
            let (kind, j) = layer.split_once('_')?;
            let k = digit(j)? + 1;
            let role = match kind {
                "Conv" => conv_role(l)?,
                "Norm" => bn_role(l)?,
                _ => return None,
            };
            Some((format!("x_{d}_{n}.conv{k}"), role))
        }
        _ => None,
    }
}

/// A residual block's layer under KonfAI's names (`Conv1`, `Norm2`,
/// `SkipConv`, `SkipNorm`).
fn block_layer(s: usize, b: usize, kind: &str, leaf: &str) -> Option<(String, Role)> {
    let (name, role) = match kind {
        "SkipConv" => ("skip".to_string(), conv_role(leaf)?),
        "SkipNorm" => ("skip".to_string(), bn_role(leaf)?),
        _ => {
            if let Some(k) = kind.strip_prefix("Conv") {
                (format!("conv{}", digit(k)?), conv_role(leaf)?)
            } else {
                let k = kind.strip_prefix("Norm")?;
                (format!("conv{}", digit(k)?), bn_role(leaf)?)
            }
        }
    };
    Some((format!("enc{s}.{b}.{name}"), role))
}

/// Whether checkpoint keys are a TotalSynth network: a stem and a head
/// under one of the three namings, and every decoder node.
pub fn recognised<'a>(keys: impl IntoIterator<Item = &'a str>) -> bool {
    let mut stem = false;
    let mut head = false;
    let mut nodes = [false; NODES.len()];
    for k in keys {
        let Some((name, _)) = classify(k) else {
            continue;
        };
        stem |= name == "stem";
        head |= name == "head";
        if let Some(n) = name.split('.').next().and_then(node_of) {
            if let Some(i) = NODES.iter().position(|&x| x == n) {
                nodes[i] = true;
            }
        }
    }
    stem && head && nodes.iter().all(|&n| n)
}

/// The checkpoint's tensors under the canonical names of the cache, every
/// batch norm folded into the convolution before it:
///
/// `stem`, `enc{s}.{b}.conv{1,2}`, `enc{s}.{b}.skip`, `x_{d}_{l}.conv{1,2}`
/// and `head`, each a `.weight` and a `.bias`. The fold is
/// `w' = w * g / sqrt(var + eps)`, `b' = (b - mean) * g / sqrt(var + eps) +
/// beta`, in double precision.
pub fn canonical(tensors: HashMap<String, WTensor>) -> Result<HashMap<String, WTensor>> {
    let mut layers: BTreeMap<String, BTreeMap<Role, WTensor>> = BTreeMap::new();
    let mut unknown = Vec::new();
    for (key, t) in tensors {
        if key.ends_with("num_batches_tracked") {
            continue;
        }
        match classify(&key) {
            Some((layer, role)) => {
                layers.entry(layer).or_default().insert(role, t);
            }
            None => unknown.push(key),
        }
    }
    if layers.is_empty() {
        unknown.sort();
        bail!(
            "not a TotalSynth (UNet++) checkpoint: none of its {} tensors has a known name \
             (first: {})",
            unknown.len(),
            unknown.first().map(String::as_str).unwrap_or("-")
        );
    }
    let mut out = HashMap::new();
    for (layer, mut parts) in layers {
        let w = parts
            .remove(&Role::ConvWeight)
            .with_context(|| format!("checkpoint has no convolution weight for {layer}"))?;
        let shape = w.shape.clone();
        let cout = *shape.first().context("scalar convolution weight")?;
        let bias = parts.remove(&Role::ConvBias).map(|b| b.data);
        let norm = [Bn::Weight, Bn::Bias, Bn::Mean, Bn::Var].map(|r| parts.remove(&Role::Norm(r)));
        let (w, b) = match norm {
            [Some(g), Some(beta), Some(mean), Some(var)] => {
                for (name, v) in [
                    ("weight", &g),
                    ("bias", &beta),
                    ("mean", &mean),
                    ("var", &var),
                ] {
                    if v.data.len() != cout {
                        bail!(
                            "{layer}: norm {name} holds {} values for {cout} channels",
                            v.data.len()
                        );
                    }
                }
                let per = w.data.len() / cout;
                let mut wf = Vec::with_capacity(w.data.len());
                let mut bf = Vec::with_capacity(cout);
                for o in 0..cout {
                    let s = g.data[o] as f64 / (var.data[o] as f64 + BN_EPS).sqrt();
                    wf.extend(
                        w.data[o * per..(o + 1) * per]
                            .iter()
                            .map(|&v| (v as f64 * s) as f32),
                    );
                    let b0 = bias.as_ref().map_or(0.0, |b| b[o] as f64);
                    bf.push(((b0 - mean.data[o] as f64) * s + beta.data[o] as f64) as f32);
                }
                (wf, bf)
            }
            [None, None, None, None] => (w.data, bias.unwrap_or_else(|| vec![0.0; cout])),
            _ => bail!("{layer}: an incomplete batch norm"),
        };
        out.insert(format!("{layer}.weight"), WTensor { shape, data: w });
        out.insert(
            format!("{layer}.bias"),
            WTensor {
                shape: vec![cout],
                data: b,
            },
        );
    }
    Ok(out)
}

/// One convolution as plain data: `[cout, cin, k, k]` weights and a bias,
/// the batch norm folded in.
#[derive(Clone, Debug)]
pub struct ConvW {
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub cin: usize,
    pub cout: usize,
    pub k: usize,
    pub stride: usize,
    pub pad: usize,
}

/// A torchvision `BasicBlock`: conv 3 x 3 (stride s) → ReLU → conv 3 x 3,
/// plus the input (through a 1 x 1 stride-s projection when the stride or
/// the width changes), then ReLU.
#[derive(Clone, Debug)]
pub struct BasicBlockW {
    pub conv1: ConvW,
    pub conv2: ConvW,
    pub skip: Option<ConvW>,
}

/// A UNet++ node: nearest 2x upsampling of its coarser input, concatenated
/// with its skips, then two conv 3 x 3 → ReLU.
#[derive(Clone, Debug)]
pub struct NodeW {
    pub conv1: ConvW,
    pub conv2: ConvW,
}

/// The whole network as plain data, read once from the canonical tensors;
/// the burn network ([`super::net`]) and the CPU one ([`super::cpu`]) are
/// both built from it.
#[derive(Clone, Debug)]
pub struct Weights {
    /// Input channels: the slices of a 2.5-D stack (5 for TotalSynth).
    pub in_ch: usize,
    pub stem: ConvW,
    /// The four residual stages.
    pub stages: Vec<Vec<BasicBlockW>>,
    /// The decoder nodes, in [`NODES`] order.
    pub nodes: Vec<NodeW>,
    pub head: ConvW,
}

fn conv(p: &Params, name: &str, stride: usize) -> Result<ConvW> {
    let shape = p
        .shape(&format!("{name}.weight"))
        .with_context(|| format!("checkpoint has no {name}"))?
        .to_vec();
    let [cout, cin, kh, kw] = shape[..] else {
        bail!("{name}.weight is {shape:?}, not a 2-D convolution");
    };
    if kh != kw || kh % 2 == 0 {
        bail!("{name}: a {kh} x {kw} kernel");
    }
    let w = p.get(&format!("{name}.weight"), &shape)?.to_vec();
    let b = p.vec(&format!("{name}.bias"), cout)?.to_vec();
    Ok(ConvW {
        w,
        b,
        cin,
        cout,
        k: kh,
        stride,
        pad: kh / 2,
    })
}

impl Weights {
    /// Assemble from the canonical (folded) tensors.
    pub fn load(p: &Params) -> Result<Weights> {
        let stem = conv(p, "stem", 2)?;
        let mut stages = Vec::with_capacity(4);
        let mut width = stem.cout;
        for s in 1..=4 {
            let n = (0..)
                .take_while(|b| p.contains(&format!("enc{s}.{b}.conv1.weight")))
                .count();
            if n == 0 {
                bail!("checkpoint has no encoder stage {s}");
            }
            let mut blocks = Vec::with_capacity(n);
            for b in 0..n {
                let stride = if s > 1 && b == 0 { 2 } else { 1 };
                let skip_key = format!("enc{s}.{b}.skip");
                let skip = if p.contains(&format!("{skip_key}.weight")) {
                    Some(conv(p, &skip_key, stride)?)
                } else {
                    None
                };
                let c1 = conv(p, &format!("enc{s}.{b}.conv1"), stride)?;
                let c2 = conv(p, &format!("enc{s}.{b}.conv2"), 1)?;
                if c1.cin != width || c2.cin != c1.cout {
                    bail!("encoder stage {s} block {b}: widths do not chain");
                }
                if skip.is_none() && (stride != 1 || c1.cin != c2.cout) {
                    bail!("encoder stage {s} block {b} changes shape but has no projection");
                }
                width = c2.cout;
                blocks.push(BasicBlockW {
                    conv1: c1,
                    conv2: c2,
                    skip,
                });
            }
            stages.push(blocks);
        }
        let nodes = NODES
            .iter()
            .map(|&(d, l)| {
                Ok(NodeW {
                    conv1: conv(p, &format!("x_{d}_{l}.conv1"), 1)?,
                    conv2: conv(p, &format!("x_{d}_{l}.conv2"), 1)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let head = conv(p, "head", 1)?;
        if head.cout != 1 {
            bail!("the head has {} output channels, not one", head.cout);
        }
        let w = Weights {
            in_ch: stem.cin,
            stem,
            stages,
            nodes,
            head,
        };
        w.check()?;
        Ok(w)
    }

    /// The widths every concatenation produces, checked against the
    /// convolutions that read them, so a mis-assembled checkpoint fails
    /// here and not inside a convolution.
    fn check(&self) -> Result<()> {
        let enc: Vec<usize> = std::iter::once(self.stem.cout)
            .chain(
                self.stages
                    .iter()
                    .map(|s| s.last().map_or(0, |b| b.conv2.cout)),
            )
            .collect();
        // features finest first: c1, l1, l2, l3, l4
        let mut width: HashMap<(usize, usize), usize> = HashMap::new();
        for (i, &(d, l)) in NODES.iter().enumerate() {
            let (up, skips) = node_inputs(d, l);
            let up_w = match up {
                Input::Enc(e) => enc[e],
                Input::Node(a, b) => width[&(a, b)],
            };
            let skip_w: usize = skips
                .iter()
                .map(|s| match *s {
                    Input::Enc(e) => enc[e],
                    Input::Node(a, b) => width[&(a, b)],
                })
                .sum();
            let n = &self.nodes[i];
            if n.conv1.cin != up_w + skip_w || n.conv2.cin != n.conv1.cout {
                bail!(
                    "decoder node x_{d}_{l} reads {} channels where its inputs hold {}",
                    n.conv1.cin,
                    up_w + skip_w
                );
            }
            width.insert((d, l), n.conv2.cout);
        }
        if self.head.cin != width[&(0, 4)] {
            bail!(
                "the head reads {} channels, x_0_4 gives {}",
                self.head.cin,
                width[&(0, 4)]
            );
        }
        Ok(())
    }
}

/// One input of a decoder node.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Input {
    /// Encoder feature `e`: 0 the stem (1/2), 1 to 4 the stages (1/4..1/32).
    Enc(usize),
    /// The output of node `x_{d}_{l}`.
    Node(usize, usize),
}

/// The inputs of node `x_{d}_{l}` as smp's `UnetPlusPlusDecoder` wires a
/// depth-4 grid: the coarser input it upsamples, then its skips in
/// concatenation order.
pub fn node_inputs(d: usize, l: usize) -> (Input, Vec<Input>) {
    // smp's features after dropping the input and reversing: f[0] = the
    // deepest stage (enc 4) ... f[4] = the stem (enc 0).
    let f = |i: usize| Input::Enc(4 - i);
    if l == 4 {
        return (Input::Node(0, 3), Vec::new());
    }
    if d == l {
        return (f(d), vec![f(d + 1)]);
    }
    let mut skips: Vec<Input> = (d + 1..=l).map(|i| Input::Node(i, l)).collect();
    skips.push(f(l + 1));
    (Input::Node(d, l - 1), skips)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_three_namings_meet() {
        let cases = [
            (
                "UNetpp5.model.encoder.conv1.weight",
                "stem",
                Role::ConvWeight,
            ),
            (
                "Model.UNetPlusPlus.Stem.Norm_0.running_var",
                "stem",
                Role::Norm(Bn::Var),
            ),
            ("StemNorm.bias", "stem", Role::Norm(Bn::Bias)),
            (
                "model.encoder.layer3.5.bn2.weight",
                "enc3.5.conv2",
                Role::Norm(Bn::Weight),
            ),
            (
                "model.encoder.layer2.0.downsample.0.weight",
                "enc2.0.skip",
                Role::ConvWeight,
            ),
            (
                "model.encoder.layer2.0.downsample.1.running_mean",
                "enc2.0.skip",
                Role::Norm(Bn::Mean),
            ),
            (
                "UNetPlusPlus.Encoder_2.Block_0.SkipNorm.bias",
                "enc2.0.skip",
                Role::Norm(Bn::Bias),
            ),
            (
                "Encoder_4_Block_2.Conv2.weight",
                "enc4.2.conv2",
                Role::ConvWeight,
            ),
            (
                "model.decoder.blocks.x_1_3.conv2.1.weight",
                "x_1_3.conv2",
                Role::Norm(Bn::Weight),
            ),
            ("x_0_4.Conv.Conv_0.weight", "x_0_4.conv1", Role::ConvWeight),
            (
                "x_2_3_conv.Norm_1.running_var",
                "x_2_3.conv2",
                Role::Norm(Bn::Var),
            ),
            ("model.segmentation_head.0.bias", "head", Role::ConvBias),
            ("SegmentationHead.weight", "head", Role::ConvWeight),
        ];
        for (key, layer, role) in cases {
            assert_eq!(classify(key), Some((layer.to_string(), role)), "{key}");
        }
        assert_eq!(classify("epoch"), None);
        assert_eq!(classify("model.encoder.fc.weight"), None);
    }

    #[test]
    fn the_decoder_is_wired_as_smp_wires_it() {
        assert_eq!(node_inputs(0, 0), (Input::Enc(4), vec![Input::Enc(3)]));
        assert_eq!(node_inputs(3, 3), (Input::Enc(1), vec![Input::Enc(0)]));
        assert_eq!(
            node_inputs(0, 1),
            (Input::Node(0, 0), vec![Input::Node(1, 1), Input::Enc(2)])
        );
        assert_eq!(
            node_inputs(1, 3),
            (
                Input::Node(1, 2),
                vec![Input::Node(2, 3), Input::Node(3, 3), Input::Enc(0)]
            )
        );
        assert_eq!(
            node_inputs(0, 3),
            (
                Input::Node(0, 2),
                vec![
                    Input::Node(1, 3),
                    Input::Node(2, 3),
                    Input::Node(3, 3),
                    Input::Enc(0)
                ]
            )
        );
        assert_eq!(node_inputs(0, 4), (Input::Node(0, 3), vec![]));
    }

    #[test]
    fn a_folded_norm_is_the_norm_after_the_conv() {
        let t = |shape: Vec<usize>, data: Vec<f32>| WTensor { shape, data };
        let mut m = HashMap::new();
        m.insert(
            "x.Stem.Conv_0.weight".into(),
            t(vec![2, 1, 1, 1], vec![2.0, -1.0]),
        );
        m.insert("x.Stem.Norm_0.weight".into(), t(vec![2], vec![0.5, 3.0]));
        m.insert("x.Stem.Norm_0.bias".into(), t(vec![2], vec![0.1, -0.2]));
        m.insert(
            "x.Stem.Norm_0.running_mean".into(),
            t(vec![2], vec![1.0, 0.0]),
        );
        m.insert(
            "x.Stem.Norm_0.running_var".into(),
            t(vec![2], vec![4.0 - 1e-5, 1.0 - 1e-5]),
        );
        m.insert(
            "x.Stem.Norm_0.num_batches_tracked".into(),
            t(vec![], vec![]),
        );
        let c = canonical(m).unwrap();
        // channel 0: (2x - 1) * 0.5 / 2 + 0.1 = 0.5x - 0.15
        let w = &c["stem.weight"].data;
        let b = &c["stem.bias"].data;
        assert!((w[0] - 0.5).abs() < 1e-6 && (b[0] + 0.15).abs() < 1e-6);
        // channel 1: (-x - 0) * 3 + (-0.2)
        assert!((w[1] + 3.0).abs() < 1e-6 && (b[1] + 0.2).abs() < 1e-6);
    }
}
