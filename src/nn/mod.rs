//! Shared neural-network infrastructure.
//!
//! Nothing in here knows about a particular architecture. It is the part of
//! running a published PyTorch model natively that every engine needs: read
//! the checkpoint, fetch it, cache the converted tensors. [`autoseg`] (the
//! nnU-Net re-implementation) and the SegVol engine are both built on top of
//! it, and both follow the same path — download once, parse the torch pickle,
//! convert to `safetensors`, load from that cache ever after. No Python, no
//! libtorch, no ONNX Runtime.
//!
//! [`autoseg`]: crate::autoseg

pub mod cache;
pub mod half;
pub mod pickle;
