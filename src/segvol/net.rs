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
            prompt,
            decoder: MaskDecoder::build(p)?,
            image_pe,
        })
    }

    /// Encode one `ROI`-shaped volume. The result can be reused across
    /// several prompts, which is what makes interactive re-prompting cheap:
    /// the image encoder is ~97% of the compute.
    pub fn encode_image(&self, volume: &[f32]) -> Mat {
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
