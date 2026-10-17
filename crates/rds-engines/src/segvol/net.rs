//! Assembly of the image encoder, prompt encoder and mask decoder into one
//! network, and the single forward pass over a `ROI`-shaped volume.
//!
//! The text tower is not part of this: text prompts arrive here already
//! projected to `EMBED` width, so the network can be built, run and tested
//! without CLIP. Wiring the tokenizer and text encoder in is a later step.
//!
//! The image encoder is about 97 % of a forward pass and does not see the
//! prompt, so the network keeps the encodings of the last windows it ran
//! ([`Encodings`]): prompting the same region again - a point added, a box
//! moved - runs the decoder only. A window is recognised by its content,
//! so a volume prepared anew for each run (as the viewer does) still finds
//! its encodings.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::Result;

use crate::nn::tensor::Mat;

use super::decoder::{Decoded, MaskDecoder};
use super::prompt::{BBox, Point, PromptEncoder};
use super::vit::Vit;
use crate::nn::params::Params;

/// The assembled network, weights resident.
pub struct SegVolNet {
    pub vit: Vit,
    /// GPU image encoder, attached after build when a usable adapter exists.
    /// The CPU encoder stays resident regardless - it is the fallback when a
    /// window fails on the device mid-run.
    #[cfg(feature = "gpu")]
    gpu_vit: Option<super::gpu::GpuVit>,
    pub prompt: PromptEncoder,
    pub decoder: MaskDecoder,
    /// The dense positional encoding, computed once at build time: it depends
    /// only on the grid and the Fourier buffer, never on the input.
    image_pe: Mat,
    encodings: Mutex<Encodings>,
    /// Windows the image encoder has run since the network was built.
    encodes: AtomicUsize,
}

/// Windows kept with their encodings: enough for the coarse view and the
/// refinement windows of one region (a crop of 64 x 384 x 384 is twelve
/// windows), 6.3 MB each.
pub const KEEP_ENCODINGS: usize = 16;

/// The encodings of the last windows the image encoder ran, most recent
/// last, each under a 128-bit digest of its input.
#[derive(Default)]
pub struct Encodings {
    kept: Vec<([u64; 2], Arc<Mat>)>,
}

impl Encodings {
    /// The kept encoding of an input with digest `key`, made the most
    /// recent.
    pub fn get(&mut self, key: [u64; 2]) -> Option<Arc<Mat>> {
        let i = self.kept.iter().position(|(k, _)| *k == key)?;
        let hit = self.kept.remove(i);
        let m = hit.1.clone();
        self.kept.push(hit);
        Some(m)
    }

    /// Keep `m` as the encoding of `key`, dropping the least recent beyond
    /// [`KEEP_ENCODINGS`].
    pub fn put(&mut self, key: [u64; 2], m: Arc<Mat>) {
        self.kept.retain(|(k, _)| *k != key);
        self.kept.push((key, m));
        if self.kept.len() > KEEP_ENCODINGS {
            self.kept.remove(0);
        }
    }

    pub fn len(&self) -> usize {
        self.kept.len()
    }

    pub fn is_empty(&self) -> bool {
        self.kept.is_empty()
    }

    pub fn clear(&mut self) {
        self.kept.clear();
    }
}

/// Two independent 64-bit digests of a window's values (their bit
/// patterns), so that two different windows are never taken for one.
pub fn digest(values: &[f32]) -> [u64; 2] {
    let mut a: u64 = 0x243F_6A88_85A3_08D3 ^ values.len() as u64;
    let mut b: u64 = 0x1319_8A2E_0370_7344 ^ (values.len() as u64).rotate_left(32);
    for v in values {
        let x = v.to_bits() as u64;
        a = (a ^ x).wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(29);
        b = (b.rotate_left(17) ^ x.wrapping_mul(0xC2B2_AE3D_27D4_EB4F))
            .wrapping_add(0x1656_67B1_9E37_79F9);
    }
    [a, b]
}

impl SegVolNet {
    pub fn build(p: &Params) -> Result<SegVolNet> {
        let prompt = PromptEncoder::build(p)?;
        let image_pe = prompt.dense_pe();
        Ok(SegVolNet {
            vit: Vit::build(p)?,
            #[cfg(feature = "gpu")]
            gpu_vit: None,
            prompt,
            decoder: MaskDecoder::build(p)?,
            image_pe,
            encodings: Mutex::new(Encodings::default()),
            encodes: AtomicUsize::new(0),
        })
    }

    /// Windows the image encoder has run since the network was built; a
    /// re-prompted window leaves it unchanged.
    pub fn encode_count(&self) -> usize {
        self.encodes.load(Ordering::Relaxed)
    }

    /// Forget the kept encodings.
    pub fn clear_encodings(&self) {
        self.encodings
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clear();
    }

    /// [`encode_image`](Self::encode_image) through the kept encodings: the
    /// same window again costs a digest of its values, not an encoder pass.
    pub fn encode_kept(&self, volume: &[f32]) -> Arc<Mat> {
        let key = digest(volume);
        let lock = || self.encodings.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(m) = lock().get(key) {
            return m;
        }
        let m = Arc::new(self.encode_image(volume));
        lock().put(key, m.clone());
        m
    }

    /// Attach a GPU image encoder; every subsequent window encodes on the
    /// device instead of the CPU.
    #[cfg(feature = "gpu")]
    pub fn attach_gpu(&mut self, gpu: super::gpu::GpuVit) {
        self.gpu_vit = Some(gpu);
    }

    /// Encode one `ROI`-shaped volume. The result can be reused across
    /// several prompts, which is what makes interactive re-prompting cheap:
    /// the image encoder is ~97% of the compute.
    pub fn encode_image(&self, volume: &[f32]) -> Mat {
        self.encodes.fetch_add(1, Ordering::Relaxed);
        #[cfg(feature = "gpu")]
        if let Some(gpu) = &self.gpu_vit {
            match gpu.forward(volume) {
                Ok(m) => return m,
                Err(e) => {
                    eprintln!("segvol: GPU encode failed ({e:#}); falling back to the CPU")
                }
            }
        }
        self.vit.forward(volume)
    }

    /// Decode one prompt against an already-encoded volume.
    pub fn decode(
        &self,
        image: &Mat,
        points: &[Point],
        boxes: &[BBox],
        text: Option<&[f32]>,
    ) -> Decoded {
        let prompts = self.prompt.encode(points, boxes, text);
        self.decoder
            .forward(image, &self.image_pe, &prompts.sparse, &prompts.dense, text)
    }

    /// Encode (or find the window's kept encoding) and decode in one call.
    pub fn forward(
        &self,
        volume: &[f32],
        points: &[Point],
        boxes: &[BBox],
        text: Option<&[f32]>,
    ) -> Decoded {
        let image = self.encode_kept(volume);
        self.decode(&image, points, boxes, text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kept_encodings_are_found_by_content_and_the_oldest_goes() {
        let m = |v: f32| Arc::new(Mat::from_vec(1, 1, vec![v]));
        let mut e = Encodings::default();
        let windows: Vec<Vec<f32>> = (0..KEEP_ENCODINGS + 1)
            .map(|i| vec![i as f32, 0.5, -0.0])
            .collect();
        for (i, w) in windows.iter().enumerate() {
            assert!(e.get(digest(w)).is_none());
            e.put(digest(w), m(i as f32));
        }
        assert_eq!(e.len(), KEEP_ENCODINGS);
        assert!(e.get(digest(&windows[0])).is_none(), "the oldest went");
        // A copy of a window is the same window; a value that differs in
        // its sign alone is not.
        let copy = windows[3].clone();
        assert_eq!(e.get(digest(&copy)).unwrap().data, vec![3.0]);
        let mut flipped = windows[3].clone();
        flipped[2] = 0.0;
        assert!(e.get(digest(&flipped)).is_none());
        // A hit is the most recent: the next to go is window 1, not 3.
        e.put(digest(&[9.0]), m(9.0));
        assert!(e.get(digest(&windows[1])).is_none());
        assert!(e.get(digest(&windows[3])).is_some());
        e.clear();
        assert!(e.is_empty());
    }
}
