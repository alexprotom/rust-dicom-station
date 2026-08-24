//! Assembly of the image encoder, prompt encoder and mask decoder into one
//! network, and the single forward pass over a `ROI`-shaped volume.
//!
//! The text tower is not part of this: text prompts arrive here already
//! projected to `EMBED` width, so the network can be built, run and tested
//! without CLIP. Wiring the tokenizer and text encoder in is a later step.

use anyhow::Result;

use crate::nn::tensor::{Act, Mat};

use super::decoder::{Decoded, MaskDecoder};
use super::params::Params;
use super::prompt::{BBox, Point, PromptEncoder};
use super::vit::Vit;

/// The assembled network, weights resident.
pub struct SegVolNet {
    pub vit: Vit,
    /// GPU image encoder, attached after build when a usable adapter exists.
    /// The CPU encoder stays resident regardless — it is the fallback when a
    /// window fails on the device mid-run.
    #[cfg(feature = "gpu")]
    gpu_vit: Option<super::gpu::GpuVit>,
    pub prompt: PromptEncoder,
    pub decoder: MaskDecoder,
    /// The dense positional encoding, computed once at build time: it depends
    /// only on the grid and the Fourier buffer, never on the input.
    image_pe: Mat,
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
        })
    }

    /// Attach a GPU image encoder; every subsequent window encodes on the
    /// device instead of the CPU.
    #[cfg(feature = "gpu")]
    pub fn attach_gpu(&mut self, gpu: super::gpu::GpuVit) {
        self.gpu_vit = Some(gpu);
    }

    /// Whether a GPU encoder is attached.
    pub fn on_gpu(&self) -> bool {
        #[cfg(feature = "gpu")]
        {
            self.gpu_vit.is_some()
        }
        #[cfg(not(feature = "gpu"))]
        {
            false
        }
    }

    /// Encode one `ROI`-shaped volume. The result can be reused across
    /// several prompts, which is what makes interactive re-prompting cheap:
    /// the image encoder is ~97% of the compute.
    pub fn encode_image(&self, volume: &[f32]) -> Mat {
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

    /// Encode and decode in one call.
    pub fn forward(
        &self,
        volume: &[f32],
        points: &[Point],
        boxes: &[BBox],
        text: Option<&[f32]>,
    ) -> Decoded {
        let image = self.encode_image(volume);
        self.decode(&image, points, boxes, text)
    }

    /// The logit volume a caller actually wants: mask channel 0, at
    /// `MASK_SHAPE`.
    pub fn logits(decoded: &Decoded) -> Act {
        decoded.best()
    }
}
