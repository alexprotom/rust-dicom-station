//! Where a network runs: the user's preference, and the one wgpu device.
//!
//! Every engine offers the same choice - the GPU when there is one, else the
//! CPU - and every GPU path has the same two hazards: `burn`/`wgpu` report a
//! missing or broken adapter by panicking rather than returning an error, and
//! a readback can fail. [`GpuContext::try_new`] opens the device itself,
//! trying every adapter of every backend the platform has, and proves it
//! works with a tiny computation before anything depends on it; [`guarded`]
//! turns a backend panic into an error, so a worker thread survives it and
//! the user sees what the backend said instead of a vanished job.
//!
//! The device's memory is held per thread (see [`memory`]): cubecl gives
//! every thread a stream of its own with pools of its own, and those pools
//! keep each buffer they ever made until asked to let go. Every run happens
//! on a new thread, so without [`memory`] each run of a model added its
//! whole working set to what the previous ones still held, and the second
//! run of a large model ran out of GPU memory.

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

/// Run `f`, turning a panic inside the GPU backend into an error that says
/// what the backend said (see [`panics::cause`]).
pub fn guarded<T>(f: impl FnOnce() -> Result<T>) -> Result<T> {
    #[cfg(feature = "gpu")]
    memory::enter();
    let mark = panics::mark();
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).map_err(|p| {
        let cause = panics::cause(mark, panic_text(p.as_ref()));
        anyhow::anyhow!("GPU inference failed (backend panic: {})", explain(&cause))
    })?
}

/// Whether an error is the GPU running out of memory: the device refusing
/// a buffer (wgpu's "Out of Memory") or one larger than it gives at once.
/// The CPU has the room such a run needs.
pub fn is_out_of_memory(e: &anyhow::Error) -> bool {
    let text = format!("{e:#}").to_ascii_lowercase();
    text.contains("out of memory") || text.contains("allocate buffer of size")
}

/// A backend message, with what it means when that is not obvious.
fn explain(msg: &str) -> String {
    // The wrapping every `unwrap` in the backend adds says nothing.
    let msg = msg
        .strip_prefix("called `Result::unwrap()` on an `Err` value: ")
        .unwrap_or(msg);
    if msg.contains("allocate buffer of size") {
        format!(
            "{msg}; the network needs a larger block of GPU memory than this device \
             gives one buffer - run it on the CPU"
        )
    } else {
        msg.to_string()
    }
}

/// The panic messages of every thread, numbered, from the moment the GPU is
/// first opened: burn and cubecl fail on their own worker threads, and the
/// panic the calling thread sees only says that one of them did. The hook
/// is chained in front of the one already installed (the Android and iOS
/// front ends keep a panic file), so nothing that was logged before stops
/// being logged.
pub mod panics {
    use std::sync::{Mutex, Once};

    /// Messages kept after a mark. The first are the ones that matter: one
    /// failure inside burn's autotune repeats itself hundreds of times.
    const KEEP: usize = 32;

    /// The next message's number, and the first [`KEEP`] messages since the
    /// latest [`mark`].
    static SEEN: Mutex<(u64, Vec<(u64, String)>)> = Mutex::new((0, Vec::new()));
    static INSTALL: Once = Once::new();

    /// Start recording (once per process).
    pub fn record() {
        INSTALL.call_once(|| {
            let previous = std::panic::take_hook();
            std::panic::set_hook(Box::new(move |info| {
                if let Ok(mut s) = SEEN.lock() {
                    let n = s.0;
                    s.0 += 1;
                    if s.1.len() < KEEP {
                        s.1.push((n, super::panic_text(info.payload())));
                    }
                }
                previous(info);
            }));
        });
    }

    /// Begin a call whose failure may be reported: forget what came before
    /// and return the number the next message will get.
    pub fn mark() -> u64 {
        match SEEN.lock() {
            Ok(mut s) => {
                s.1.clear();
                s.0
            }
            Err(_) => 0,
        }
    }

    /// What a failure was. burn and cubecl fail on their own worker
    /// threads, and what then reaches the caller is a symptom - their echo,
    /// a message ending in "CallError", or a tensor the failed thread never
    /// made ("Should have handle for tensor"). The cause is the first message recorded
    /// since `mark` that is not an echo; `own`, the caller's, only when
    /// nothing else was recorded.
    pub fn cause(mark: u64, own: String) -> String {
        match SEEN.lock() {
            Ok(s) => pick(s.1.iter(), mark, own),
            Err(_) => own,
        }
    }

    /// [`cause`] over a given record.
    pub(super) fn pick<'a>(
        recorded: impl Iterator<Item = &'a (u64, String)>,
        mark: u64,
        own: String,
    ) -> String {
        recorded
            .filter(|(n, m)| *n >= mark && !m.contains(ECHO))
            .map(|(_, m)| m.clone())
            .next()
            .unwrap_or(own)
    }

    /// What cubecl's channel says when the thread it waits on has failed.
    const ECHO: &str = "CallError";
}

/// A validated wgpu device (Vulkan / DX12 / Metal / OpenGL - no vendor
/// toolkit).
#[cfg(feature = "gpu")]
#[derive(Clone)]
pub struct GpuContext {
    device: burn::backend::wgpu::WgpuDevice,
    /// The adapter and its backend, as the user should read it.
    adapter: String,
    /// The same device as wgpu has it, for the kernels written in WGSL
    /// rather than through burn (the 2D/3D registration's projector).
    raw: (wgpu::Device, wgpu::Queue),
    /// A renderer in software (llvmpipe, WARP, SwiftShader): it runs the
    /// networks, but a kernel the CPU path already does well is faster
    /// there.
    software: bool,
}

/// The one inference device of the process, or why there is none. Made on
/// first use and kept: wgpu allows one device per adapter, and an attempt
/// that failed by panicking inside a driver is not one to repeat per run.
#[cfg(feature = "gpu")]
static GPU: std::sync::OnceLock<std::result::Result<GpuContext, String>> =
    std::sync::OnceLock::new();

#[cfg(feature = "gpu")]
impl GpuContext {
    /// The inference device, opened on first use (see [`open`]).
    pub fn try_new() -> Result<GpuContext> {
        match GPU.get_or_init(open::first_working) {
            Ok(ctx) => {
                memory::enter();
                Ok(ctx.clone())
            }
            Err(why) => anyhow::bail!("{why}"),
        }
    }

    /// The device, for burn tensors: the calling thread is noted as one
    /// that holds memory on it (see [`memory`]).
    pub fn device(&self) -> &burn::backend::wgpu::WgpuDevice {
        memory::enter();
        &self.device
    }

    /// The device and its queue for a compute shader of one's own. Shared
    /// with burn: wgpu serialises the submissions.
    pub fn wgpu(&self) -> (&wgpu::Device, &wgpu::Queue) {
        (&self.raw.0, &self.raw.1)
    }

    /// Whether the adapter is a renderer in software.
    pub fn is_software(&self) -> bool {
        self.software
    }

    /// What to show the user for a GPU run.
    pub fn describe(&self) -> String {
        format!("GPU ({})", self.adapter)
    }
}

/// Opening the inference device ourselves rather than through
/// `WgpuDevice::default()`.
///
/// `cubecl` (under `burn`) picks one graphics API per platform at compile
/// time - Metal on macOS, Vulkan everywhere else, iOS included - asks it for
/// one adapter, and panics when anything on that one path fails, with a
/// message that never reaches the user. So a tablet whose Vulkan adapter
/// cannot run the shaders, a Windows machine whose Vulkan driver is broken
/// (the reason `gfx.rs` exists) and every iPad were all "no GPU", and
/// nothing said why.
///
/// Here every adapter of every backend this platform has is tried in turn:
/// the device is requested exactly as `cubecl` requests it (every feature
/// the adapter offers but mappable primary buffers, its own limits), handed
/// to `cubecl` with [`init_device`](burn::backend::wgpu::init_device), and
/// proved with a small computation of the kind the networks do. The first
/// that passes is the device; when none does, the error names each adapter
/// and what stopped it.
#[cfg(feature = "gpu")]
mod open {
    use burn::backend::wgpu::{init_device, RuntimeOptions, WgpuDevice, WgpuSetup};

    /// The backends to try, in order: those `WGPU_BACKEND` names first (the
    /// viewer exports its *Graphics backend* setting there, and a user may
    /// set it by hand), then the platform's own.
    pub(super) fn backends() -> Vec<wgpu::Backend> {
        use wgpu::Backend as B;
        let platform: &[B] = if cfg!(any(target_os = "macos", target_os = "ios")) {
            &[B::Metal]
        } else if cfg!(windows) {
            &[B::Vulkan, B::Dx12]
        } else {
            // Linux and Android. Not OpenGL unless the user chose it: burn's
            // reductions need float atomics that OpenGL lacks, so it fails
            // the self-test (tried on Mesa), and on a phone a GL driver is
            // one more thing that can take the program down with it.
            &[B::Vulkan]
        };
        order(wgpu::Backends::from_env(), platform)
    }

    /// The backends `env` names, then the rest of `platform`'s.
    pub(super) fn order(
        env: Option<wgpu::Backends>,
        platform: &[wgpu::Backend],
    ) -> Vec<wgpu::Backend> {
        use wgpu::Backend as B;
        let mut order = Vec::new();
        if let Some(env) = env {
            for b in [B::Vulkan, B::Metal, B::Dx12, B::Gl] {
                if env.contains(b.into()) {
                    order.push(b);
                }
            }
        }
        for b in platform {
            if !order.contains(b) {
                order.push(*b);
            }
        }
        order
    }

    /// What the user calls a backend.
    pub(super) fn backend_name(b: wgpu::Backend) -> &'static str {
        match b {
            wgpu::Backend::Vulkan => "Vulkan",
            wgpu::Backend::Metal => "Metal",
            wgpu::Backend::Dx12 => "DirectX 12",
            wgpu::Backend::Gl => "OpenGL",
            _ => "other",
        }
    }

    /// Adapters in the order worth trying: a discrete GPU before an
    /// integrated one, a software renderer last (it works, and is how a
    /// machine without a GPU still runs the GPU path, but slowly).
    pub(super) fn rank(t: wgpu::DeviceType) -> u8 {
        match t {
            wgpu::DeviceType::DiscreteGpu => 0,
            wgpu::DeviceType::IntegratedGpu => 1,
            wgpu::DeviceType::VirtualGpu => 2,
            wgpu::DeviceType::Other => 3,
            wgpu::DeviceType::Cpu => 4,
        }
    }

    pub(super) fn first_working() -> Result<super::GpuContext, String> {
        // burn runs kernels on threads of its own; when one of those fails,
        // the caller only learns that "a call" did. The first panic of any
        // thread during an attempt is the reason it reports.
        super::panics::record();
        let mut tried: Vec<String> = Vec::new();
        for backend in backends() {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: backend.into(),
                flags: wgpu::InstanceFlags::default(),
                ..wgpu::InstanceDescriptor::new_without_display_handle()
            });
            let mut adapters = block_on(instance.enumerate_adapters(backend.into()));
            if adapters.is_empty() {
                tried.push(format!("{}: no adapter", backend_name(backend)));
                continue;
            }
            adapters.sort_by_key(|a| rank(a.get_info().device_type));
            for adapter in adapters {
                let info = adapter.get_info();
                let name = format!("{}, {}", info.name.trim(), backend_name(backend));
                let mark = super::panics::mark();
                let attempt = try_adapter(&instance, adapter, backend)
                    .map_err(|why| super::panics::cause(mark, why));
                match attempt {
                    Ok((device, raw)) => {
                        eprintln!(
                            "inference GPU: {name} ({} {})",
                            info.driver, info.driver_info
                        );
                        return Ok(super::GpuContext {
                            device,
                            adapter: name,
                            raw,
                            software: info.device_type == wgpu::DeviceType::Cpu,
                        });
                    }
                    Err(why) => {
                        eprintln!("inference GPU {name} not usable: {why}");
                        tried.push(format!("{name}: {why}"));
                    }
                }
            }
        }
        Err(format!("no usable GPU ({})", tried.join("; ")))
    }

    /// Request the device, give it to `cubecl` and prove it adds up.
    fn try_adapter(
        instance: &wgpu::Instance,
        adapter: wgpu::Adapter,
        backend: wgpu::Backend,
    ) -> Result<(WgpuDevice, (wgpu::Device, wgpu::Queue)), String> {
        // `cubecl`'s own request (cubecl-wgpu, backend/wgsl.rs): it reads
        // the adapter's features to decide which shader types to emit, so the
        // device must have all of them.
        let request = adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("rds inference"),
            required_features: adapter
                .features()
                .difference(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS),
            required_limits: adapter.limits(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
            // SAFETY: the same opt-in `cubecl` makes for the features above;
            // nothing here uses an experimental feature itself.
            experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
        });
        let (device, queue) =
            block_on(request).map_err(|e| format!("the device request was refused ({e})"))?;
        let raw = (device.clone(), queue.clone());
        let setup = WgpuSetup {
            instance: instance.clone(),
            adapter,
            device,
            queue,
            backend,
        };
        // On a thread of its own, with a time limit: a driver that hangs
        // in a shader compile must cost a minute, not the job.
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::Builder::new()
            .name("gpu self-test".into())
            .spawn(move || {
                let made = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let device = init_device(setup, RuntimeOptions::default());
                    let tested = self_test(&device);
                    // This thread ends here; what its test held goes back.
                    super::memory::release(&device);
                    tested.map(|()| device)
                }));
                let _ = tx.send(made.unwrap_or_else(|p| Err(super::panic_text(p.as_ref()))));
            })
            .map_err(|e| format!("the self-test could not start ({e})"))?;
        rx.recv_timeout(SELF_TEST_LIMIT)
            .unwrap_or_else(|_| {
                Err(format!(
                    "the self-test did not finish in {} s",
                    SELF_TEST_LIMIT.as_secs()
                ))
            })
            .map(|device| (device, raw))
    }

    /// How long the self-test may take: its first run also autotunes
    /// burn's kernels, which on a slow GPU takes seconds.
    const SELF_TEST_LIMIT: std::time::Duration = std::time::Duration::from_secs(120);

    /// A 3-D convolution summed over its few thousand outputs, and a
    /// matrix product: the operations the networks are made of, at sizes
    /// that make burn's autotune pick the kernels a real run uses. A
    /// three-value sum is not enough: OpenGL adds three numbers and then
    /// fails the first real reduction, which needs atomics it lacks.
    fn self_test(device: &WgpuDevice) -> Result<(), String> {
        use burn::backend::Wgpu;
        use burn::tensor::module::conv3d;
        use burn::tensor::ops::ConvOptions;
        use burn::tensor::Tensor;
        let x = Tensor::<Wgpu, 5>::ones([1, 2, 8, 8, 8], device);
        let w = Tensor::<Wgpu, 5>::full([4, 2, 3, 3, 3], 0.5, device);
        let y = conv3d(x, w, None, ConvOptions::new([1; 3], [1; 3], [1; 3], 1));
        let got: f32 = y.sum().into_scalar();
        // Per axis the 3-tap window with one voxel of padding covers 2 taps
        // at the two ends and 3 at the six inside: 22 in all, 22^3 taps,
        // each 0.5 x 2 channels, for 4 output channels.
        let want = 22f32.powi(3) * 0.5 * 2.0 * 4.0;
        if (got - want).abs() > want * 1e-4 {
            return Err(format!(
                "a test convolution summed to {got}, expected {want}"
            ));
        }
        let a = Tensor::<Wgpu, 2>::full([64, 32], 0.25, device);
        let b = Tensor::<Wgpu, 2>::full([32, 16], 2.0, device);
        let got: f32 = a.matmul(b).sum().into_scalar();
        let want = 64.0 * 16.0 * 32.0 * 0.5;
        if (got - want).abs() > want * 1e-4 {
            return Err(format!(
                "a test matrix product summed to {got}, expected {want}"
            ));
        }
        Ok(())
    }

    /// Wait for one of wgpu's futures. On native platforms they are ready
    /// when made; the loop is for form.
    fn block_on<F: std::future::Future>(f: F) -> F::Output {
        let mut f = std::pin::pin!(f);
        let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
        loop {
            if let std::task::Poll::Ready(v) = f.as_mut().poll(&mut cx) {
                return v;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }
}

/// GPU memory, thread by thread.
///
/// cubecl (under burn) keeps one stream per thread that uses the device -
/// the thread's number modulo 128 - and each stream has its own memory
/// pools. The pools burn configures (`MemoryConfiguration::SubSlices`) never
/// give a buffer back by themselves: a freed tensor's slice is only marked
/// free for the next tensor *of the same stream*, and the buffer goes back
/// to the driver only on an explicit cleanup, which acts on the calling
/// thread's stream alone. A run of a model happens on a thread of its own
/// (a job, an MCP call), and the nnU-Net path forwards each part's windows
/// on yet another (`autoseg::infer`, the look-ahead), so when those threads
/// ended their pools stayed full: one run of TotalSegmentator's `total`
/// left five parts' worth of activations behind, and the second run ran
/// out of memory in its second part.
///
/// So every thread that works on the device is noted ([`enter`], called by
/// [`GpuContext::device`], [`GpuContext::try_new`] and [`guarded`]); when it
/// ends, its stream is put on a list, and the next thread to use the device
/// finishes that stream's queued work (burn-fusion queues a tensor's
/// release until the stream next runs) and has its pools give back every
/// buffer nothing uses, acting as that stream through
/// [`StreamId::executes`]. Nothing of cubecl runs inside the thread-local's
/// destructor itself, where the backend's own thread-locals may already be
/// gone. A thread that lives on (the command-line tools, a rayon worker)
/// keeps its pools for its next run, which reuses them; [`release_all`]
/// gives them back at a point of the caller's choosing.
///
/// One leak remains that this cannot reach: burn-fusion does not release a
/// tensor dropped while a panic unwinds, so a run that failed inside the
/// backend (an out-of-memory panic) leaves its live tensors allocated.
#[cfg(feature = "gpu")]
pub mod memory {
    use std::cell::OnceCell;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Mutex;

    use burn::backend::wgpu::WgpuDevice;
    use burn::backend::Wgpu;
    use burn::cubecl::stream_id::StreamId;
    use burn::tensor::backend::Backend;

    /// The streams of the threads that used the device and have ended.
    static ENDED: Mutex<Vec<u64>> = Mutex::new(Vec::new());
    /// Whether [`ENDED`] has anything: checked on every [`enter`].
    static ANY_ENDED: AtomicBool = AtomicBool::new(false);

    /// A thread's stream, put on [`ENDED`] when the thread ends.
    struct Noted(u64);

    impl Drop for Noted {
        fn drop(&mut self) {
            ENDED.lock().unwrap_or_else(|p| p.into_inner()).push(self.0);
            ANY_ENDED.store(true, Ordering::Release);
        }
    }

    std::thread_local! {
        static NOTED: OnceCell<Noted> = const { OnceCell::new() };
    }

    /// The device, when it has been opened; nothing here opens it.
    fn device() -> Option<&'static WgpuDevice> {
        match super::GPU.get() {
            Some(Ok(ctx)) => Some(&ctx.device),
            _ => None,
        }
    }

    /// Note the calling thread as one that holds memory on the device, and
    /// give back what the threads that have ended still held.
    pub fn enter() {
        let Some(device) = device() else { return };
        // `try_with`: a destructor running at the thread's end may drop a
        // network after this thread-local is gone.
        let _ = NOTED.try_with(|n| {
            n.get_or_init(|| Noted(StreamId::current().value));
        });
        if ANY_ENDED.load(Ordering::Acquire) {
            reclaim(device);
        }
    }

    /// Give back what the streams of ended threads hold.
    fn reclaim(device: &WgpuDevice) {
        let ended = {
            let mut list = ENDED.lock().unwrap_or_else(|p| p.into_inner());
            ANY_ENDED.store(false, Ordering::Release);
            std::mem::take(&mut *list)
        };
        for value in ended {
            StreamId { value }.executes(|| release(device));
        }
    }

    /// Run the calling stream's queued work, then give back every buffer of
    /// its pools that holds no live tensor. A failure (a device already
    /// lost) is logged, not raised: nothing depends on the memory coming
    /// back but the next run.
    pub fn release(device: &WgpuDevice) {
        let done = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let synced = <Wgpu as Backend>::sync(device);
            <Wgpu as Backend>::memory_cleanup(device);
            synced
        }));
        match done {
            Ok(Ok(())) => {}
            Ok(Err(e)) => eprintln!("GPU memory release: {e:?}"),
            Err(p) => eprintln!("GPU memory release: {}", super::panic_text(p.as_ref())),
        }
    }

    /// Give back what the calling thread and every ended one hold: at the
    /// end of a run on a thread that lives on, or before trying again after
    /// running out of memory.
    pub fn release_all() {
        if let Some(device) = device() {
            reclaim(device);
            release(device);
        }
    }

    /// The bytes the device's pools hold, over every stream, and how many
    /// streams hold any (diagnostics: `autoseg_cli --repeat`).
    pub fn held() -> Option<(u64, usize)> {
        use burn::backend::wgpu::WgpuRuntime;
        use burn::cubecl::config::RuntimeConfig;
        use burn::cubecl::Runtime;
        let device = device()?;
        let client = WgpuRuntime::client(device);
        let streams = burn::cubecl::config::CubeClRuntimeConfig::get()
            .streaming
            .max_streams as u64;
        let (mut bytes, mut holding) = (0, 0);
        for value in 0..streams {
            let usage = StreamId { value }.executes(|| client.memory_usage());
            if let Ok(u) = usage {
                if u.bytes_reserved > 0 {
                    bytes += u.bytes_reserved;
                    holding += 1;
                }
            }
        }
        Some((bytes, holding))
    }
}

/// Without the `gpu` feature there is no device memory to give back.
#[cfg(not(feature = "gpu"))]
pub mod memory {
    pub fn release_all() {}
}

/// The message a panic carried, for an error the user reads.
pub fn panic_text(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "a panic without a message".to_string()
    }
}

/// The type exists in every build so callers can name it; without the `gpu`
/// feature no value of it can be made.
#[cfg(not(feature = "gpu"))]
#[derive(Clone)]
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
    pub fn is_software(&self) -> bool {
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

    /// The backend the viewer exports comes first and is not tried twice;
    /// the platform's own follow in their order.
    #[cfg(feature = "gpu")]
    #[test]
    fn the_exported_backend_is_tried_first() {
        use wgpu::{Backend as B, Backends};
        let linux = [B::Vulkan, B::Gl];
        assert_eq!(open::order(None, &linux), linux);
        assert_eq!(open::order(Some(Backends::GL), &linux), [B::Gl, B::Vulkan]);
        assert_eq!(open::order(Some(Backends::VULKAN), &linux), linux);
        assert_eq!(
            open::order(Some(Backends::DX12), &[B::Vulkan, B::Dx12]),
            [B::Dx12, B::Vulkan]
        );
        assert_eq!(
            open::order(Some(Backends::from_comma_list("gles")), &[B::Metal]),
            [B::Gl, B::Metal]
        );
        assert!(
            open::rank(wgpu::DeviceType::DiscreteGpu) < open::rank(wgpu::DeviceType::IntegratedGpu)
        );
        assert!(open::rank(wgpu::DeviceType::IntegratedGpu) < open::rank(wgpu::DeviceType::Cpu));
        assert_eq!(open::backend_name(B::Gl), "OpenGL");
    }

    #[test]
    fn a_panic_says_what_it_carried() {
        let p = std::panic::catch_unwind(|| panic!("plain")).unwrap_err();
        assert_eq!(panic_text(p.as_ref()), "plain");
        let p = std::panic::catch_unwind(|| panic!("formatted {}", 7)).unwrap_err();
        assert_eq!(panic_text(p.as_ref()), "formatted 7");
        let p = std::panic::catch_unwind(|| std::panic::panic_any(3u8)).unwrap_err();
        assert_eq!(panic_text(p.as_ref()), "a panic without a message");
    }

    /// A failure on a worker thread is reported by its own message, not by
    /// the symptom the calling thread panics with; a message recorded before
    /// the call is not its cause, and with nothing recorded the caller's
    /// message stands.
    #[test]
    fn the_cause_of_an_echoed_failure_is_the_worker_s_message() {
        let echo = "called `Result::unwrap()` on an `Err` value: CallError".to_string();
        let seen = [
            (3, "an older failure".to_string()),
            (
                7,
                "Can't have an error when submitting a task: CallError".to_string(),
            ),
            (8, "can't allocate buffer of size: 205520896".to_string()),
            (9, "index out of bounds".to_string()),
        ];
        assert_eq!(
            panics::pick(seen.iter(), 5, echo.clone()),
            "can't allocate buffer of size: 205520896"
        );
        let symptom = "Should have handle for tensor TensorId { value: 388 }".to_string();
        assert_eq!(
            panics::pick(seen.iter(), 5, symptom),
            "can't allocate buffer of size: 205520896"
        );
        assert_eq!(panics::pick(seen.iter(), 10, echo.clone()), echo);
        assert_eq!(panics::pick(seen.iter(), 10, "plain".into()), "plain");
        // What the user reads for the commonest GPU failure on a tablet.
        let r: Result<()> = guarded(|| {
            panic!(
                "called `Result::unwrap()` on an `Err` value: can't allocate buffer of size: 205520896"
            )
        });
        let e = r.unwrap_err().to_string();
        assert!(
            e.contains("205520896") && e.contains("run it on the CPU"),
            "{e}"
        );
        assert!(!e.contains("unwrap"), "{e}");
    }

    /// What a thread that ended held on the GPU stays held (cubecl's pools
    /// are per thread and keep their buffers), until the next user of the
    /// device gives it back. Needs a GPU:
    /// `RDS_GPU_TESTS=1 cargo test -p rds-engines --release -- --ignored gpu_memory`.
    #[cfg(feature = "gpu")]
    #[test]
    #[ignore]
    fn gpu_memory_of_an_ended_thread_comes_back() {
        if std::env::var_os("RDS_GPU_TESTS").is_none() {
            eprintln!("RDS_GPU_TESTS is not set");
            return;
        }
        use burn::backend::Wgpu;
        use burn::tensor::Tensor;
        const MB: u64 = 1 << 20;
        let ctx = GpuContext::try_new().expect("a GPU");
        memory::release_all();
        let (before, _) = memory::held().unwrap();
        let worker = ctx.clone();
        std::thread::spawn(move || {
            let t = Tensor::<Wgpu, 1>::ones([(64 * MB / 4) as usize], worker.device());
            let s: f32 = (t * 2.0).sum().into_scalar();
            assert_eq!(s, (64 * MB / 2) as f32);
        })
        .join()
        .unwrap();
        let (left, streams) = memory::held().unwrap();
        eprintln!(
            "before {} MB; after the thread ended {} MB in {streams} stream(s)",
            before / MB,
            left / MB
        );
        assert!(
            left >= before + 64 * MB,
            "the thread's pool held its tensors"
        );
        memory::release_all();
        let (after, _) = memory::held().unwrap();
        eprintln!("given back: {} MB left", after / MB);
        assert!(after <= before, "{after} bytes left of {before}");
    }

    #[test]
    fn running_out_of_gpu_memory_is_recognised() {
        let e =
            anyhow::anyhow!("GPU inference failed (backend panic: wgpu error: Out of Memory\n)")
                .context("GPU forward")
                .context("inference (v3 1.5 mm vertebrae (2/5))");
        assert!(is_out_of_memory(&e));
        let r: Result<()> = guarded(|| {
            panic!("called `Result::unwrap()` on an `Err` value: can't allocate buffer of size: 7")
        });
        assert!(is_out_of_memory(&r.unwrap_err()));
        assert!(!is_out_of_memory(&anyhow::anyhow!("GPU readback failed")));
    }

    #[test]
    fn a_backend_panic_becomes_an_error() {
        let r: Result<()> = guarded(|| panic!("wgpu exploded"));
        let e = r.unwrap_err().to_string();
        assert!(
            e.contains("backend panic") && e.contains("wgpu exploded"),
            "{e}"
        );
        let r: Result<()> = guarded(|| panic!("{} buffers", 3));
        assert!(r.unwrap_err().to_string().contains("3 buffers"));
        assert_eq!(guarded(|| Ok(3)).unwrap(), 3);
    }
}
