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
//! Alongside the loading machinery are the dense kernels a transformer needs:
//! [`tensor`] holds the two concrete shapes, [`linalg`] the matrix multiply,
//! normalizations and activations, and [`attention`] the one attention
//! routine every part of the network shares. Convolutions specific to a
//! U-Net remain in [`crate::autoseg::cpu`].
//!
//! [`params`] is the shape-checked view of a loaded state dict every engine
//! assembles itself from.
//!
//! [`autoseg`]: crate::autoseg

pub mod attention;
pub mod cache;
pub mod half;
pub mod linalg;
pub mod params;
pub mod pickle;
pub mod tensor;
