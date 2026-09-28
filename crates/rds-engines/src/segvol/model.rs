//! A SegVol network loaded once and kept between prompts.
//!
//! Loading is most of what a single prompt used to cost: the converted
//! checkpoint read from disk, the network assembled from it, the image
//! encoder uploaded to the GPU. The prompt window is a conversation - point,
//! look, point again - and a run over the phases of a 4D group is the same
//! prompt ten times, so the loaded network is kept (by the caller, behind an
//! `Arc`) and every run after the first starts at the image.
//!
//! The text tower is only needed for text prompts, so it is built the first
//! time one arrives, from a second read of the checkpoint, and kept from then
//! on; box and point prompts never pay for it.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use anyhow::{bail, Context, Result};

use crate::nn::device::DevicePref;
use crate::progress::{Progress, CANCELLED};

use super::bpe::Bpe;
use super::clip::TextEncoder;
use super::net::SegVolNet;
use super::weights;

/// The network and what it runs on.
pub struct Model {
    pub net: SegVolNet,
    /// Where the image encoder runs, e.g. "GPU (wgpu)".
    pub device: String,
    models_dir: PathBuf,
    text: OnceLock<(Bpe, TextEncoder)>,
    /// Held while the text tower is built, so two runs that both need it
    /// build it once.
    text_build: Mutex<()>,
}

impl Model {
    /// Load the weights (downloading and converting them the first time),
    /// assemble the network and put the image encoder on the device `pref`
    /// resolves to, falling back to the CPU when there is none.
    pub fn load(models_dir: &Path, pref: DevicePref, progress: &Progress) -> Result<Model> {
        let params = weights::load(models_dir, progress)?;
        if progress.cancelled() {
            bail!(CANCELLED);
        }
        progress.set("Assembling the network");
        #[cfg_attr(not(feature = "gpu"), allow(unused_mut))]
        let mut net = SegVolNet::build(&params).context("assemble the SegVol network")?;

        // Put the image encoder on the GPU when asked and a usable adapter
        // exists; fall back to the CPU otherwise. Either way the device is
        // reported, both in the progress row and in the finished-run status.
        progress.set("Choosing the compute device");
        let gpu = pref.resolve()?;
        let device = match gpu {
            #[cfg(feature = "gpu")]
            Some(ctx) => {
                let vit =
                    super::gpu::GpuVit::new(&ctx, &params).context("upload the image encoder")?;
                net.attach_gpu(vit);
                ctx.describe()
            }
            #[cfg(not(feature = "gpu"))]
            Some(ctx) => ctx.unreachable(),
            None => crate::nn::device::describe_cpu(),
        };
        Ok(Model {
            net,
            device,
            models_dir: models_dir.to_path_buf(),
            text: OnceLock::new(),
            text_build: Mutex::new(()),
        })
    }

    /// A structure name as the network's text prompt. The text tower is
    /// built on first use (see the module notes); the encoder keeps the
    /// prompts it has already encoded.
    pub fn encode_text(&self, structure: &str, progress: &Progress) -> Result<Vec<f32>> {
        if structure.trim().is_empty() {
            bail!("enter a structure name");
        }
        if self.text.get().is_none() {
            let _building = self
                .text_build
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if self.text.get().is_none() {
                progress.set("Loading the text encoder");
                for f in &weights::CLIP_FILES {
                    f.ensure(&self.models_dir, progress)?;
                }
                let bpe = Bpe::from_dir(&self.models_dir)?;
                let params = weights::load(&self.models_dir, progress)?;
                let enc = TextEncoder::build(&params)?;
                let _ = self.text.set((bpe, enc));
            }
        }
        let (bpe, enc) = self.text.get().expect("built above");
        progress.set("Encoding the text prompt");
        Ok(enc.encode_structure(bpe, structure))
    }
}
