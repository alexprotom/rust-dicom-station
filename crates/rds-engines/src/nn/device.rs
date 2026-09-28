//! Where a network runs: the user's preference, and the one wgpu device.
//!
//! Every engine offers the same choice - the GPU when there is one, else the
//! CPU - and every GPU path has the same two hazards: `burn`/`wgpu` report a
//! missing or broken adapter by panicking rather than returning an error, and
//! a readback can fail. [`GpuContext::try_new`] proves the device works with
//! a tiny computation before anything depends on it, and [`guarded`] turns a
//! backend panic into an error, so a worker thread survives it and the user
//! sees a message instead of a vanished job.

use anyhow::Result;

/// Device preference for inference.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DevicePref {
    /// GPU when available, CPU otherwise.
    #[default]
    Auto,
    Cpu,
    Gpu,
}

impl DevicePref {
    pub const ALL: [DevicePref; 3] = [DevicePref::Auto, DevicePref::Gpu, DevicePref::Cpu];

    pub fn label(self) -> &'static str {
        match self {
            DevicePref::Auto => "Auto",
            DevicePref::Gpu => "GPU",
            DevicePref::Cpu => "CPU",
        }
    }

    /// The command-line spelling, and the reverse of it.
    pub fn from_key(key: &str) -> Option<DevicePref> {
        match key {
            "auto" => Some(DevicePref::Auto),
            "cpu" => Some(DevicePref::Cpu),
            "gpu" => Some(DevicePref::Gpu),
            _ => None,
        }
    }

    /// Resolve the preference: `Some(ctx)` to run on the GPU, `None` for the
    /// CPU. `Gpu` fails when no usable adapter exists; `Auto` falls back.
    #[cfg(feature = "gpu")]
    pub fn resolve(self) -> Result<Option<GpuContext>> {
        match self {
            DevicePref::Cpu => Ok(None),
            DevicePref::Gpu => GpuContext::try_new()
                .map(Some)
                .map_err(|e| anyhow::anyhow!("GPU requested but not available: {e}")),
            DevicePref::Auto => Ok(GpuContext::try_new().ok()),
        }
    }

    /// Without the `gpu` feature only the CPU exists, and asking for the GPU
    /// is an error rather than a silent downgrade.
    #[cfg(not(feature = "gpu"))]
    pub fn resolve(self) -> Result<Option<GpuContext>> {
        match self {
            DevicePref::Gpu => {
                anyhow::bail!("this build has no GPU support (compiled without the 'gpu' feature)")
            }
            _ => Ok(None),
        }
    }
}

/// What to show the user for a CPU run.
pub fn describe_cpu() -> String {
    format!("CPU ({} threads)", rayon::current_num_threads())
}

/// Run `f`, turning a panic inside the GPU backend into an error.
pub fn guarded<T>(f: impl FnOnce() -> Result<T>) -> Result<T> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))
        .map_err(|_| anyhow::anyhow!("GPU inference failed (backend panic)"))?
}

/// A validated wgpu device (Vulkan / DX12 / Metal - no vendor toolkit).
#[cfg(feature = "gpu")]
pub struct GpuContext {
    device: burn::backend::wgpu::WgpuDevice,
}

#[cfg(feature = "gpu")]
impl GpuContext {
    /// Initialize the default wgpu device and prove it works with a tiny
    /// computation. Backend initialization failures surface as panics inside
    /// wgpu/cubecl, so they are caught here and turned into errors.
    pub fn try_new() -> Result<GpuContext> {
        use burn::backend::wgpu::WgpuDevice;
        use burn::backend::Wgpu;
        use burn::tensor::{Tensor, TensorData};
        let result = std::panic::catch_unwind(|| {
            let device = WgpuDevice::default();
            let t =
                Tensor::<Wgpu, 1>::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0], [3]), &device);
            let s: f32 = t.sum().into_scalar();
            (device, s)
        });
        match result {
            Ok((device, s)) if (s - 6.0).abs() < 1e-3 => Ok(GpuContext { device }),
            Ok((_, s)) => anyhow::bail!("GPU self-test returned {s}, expected 6"),
            Err(_) => anyhow::bail!("no usable wgpu adapter found"),
        }
    }

    pub fn device(&self) -> &burn::backend::wgpu::WgpuDevice {
        &self.device
    }

    /// What to show the user for a GPU run.
    pub fn describe(&self) -> String {
        "GPU (wgpu)".to_string()
    }
}

/// The type exists in every build so callers can name it; without the `gpu`
/// feature no value of it can be made.
#[cfg(not(feature = "gpu"))]
pub struct GpuContext {
    _never: std::convert::Infallible,
}

#[cfg(not(feature = "gpu"))]
impl GpuContext {
    pub fn describe(&self) -> String {
        match self._never {}
    }
    /// A value of this type cannot exist, so a match arm holding one is dead.
    pub fn unreachable<T>(&self) -> T {
        match self._never {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preferences_round_trip_their_keys() {
        for p in DevicePref::ALL {
            assert_eq!(
                DevicePref::from_key(p.label().to_ascii_lowercase().as_str()),
                Some(p)
            );
        }
        assert_eq!(DevicePref::from_key("tpu"), None);
        assert_eq!(DevicePref::default(), DevicePref::Auto);
    }

    #[test]
    fn the_cpu_preference_never_needs_a_gpu() {
        assert!(DevicePref::Cpu.resolve().unwrap().is_none());
        assert!(describe_cpu().starts_with("CPU ("));
    }

    #[test]
    fn a_backend_panic_becomes_an_error() {
        let r: Result<()> = guarded(|| panic!("wgpu exploded"));
        assert!(r.unwrap_err().to_string().contains("backend panic"));
        assert_eq!(guarded(|| Ok(3)).unwrap(), 3);
    }
}
