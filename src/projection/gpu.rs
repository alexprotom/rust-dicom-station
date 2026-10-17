//! [`render_posed`](super::render_posed) on the GPU: the same ray-cast as a
//! WGSL compute shader, one invocation per pixel.
//!
//! The shader is the CPU loop transcribed: the ray from the source to the
//! pixel centre in the volume's continuous index space, clipped to the box
//! `[0, n − 1]` along each axis, cut into `⌈span / step⌉` equal pieces and
//! summed by the midpoint rule with the same trilinear read (zero outside,
//! the last voxel repeated at the far edges), in `f32` as the CPU's inner
//! loop is. The camera is converted to index space once per rendering on the
//! CPU, in `f64`. What differs is the arithmetic of the ray's set-up, done
//! per pixel in `f32` here: the two images agree to a few parts in a million
//! of their range (`tests/registration_2d3d.rs`, run with `RDS_GPU_TESTS`).
//!
//! The attenuation volume is uploaded once, as a 3-D `R32Float` texture
//! read with `textureLoad` (no sampler: a float texture need not be
//! filterable, and the interpolation must be the CPU's). Each rendering
//! writes one buffer of pixels and reads it back; the similarity is
//! computed on the CPU from there, so a registration's evaluation costs one
//! dispatch and one readback per view.
//!
//! The device is the one [`GpuContext`] opened and proved for the networks,
//! shared with burn. A failure of the backend (a lost device, a panic in
//! the driver glue) is an error, never a crash: the registration then goes
//! on with the CPU when the device was its own choice.

#[cfg(not(feature = "gpu"))]
use super::{MuVolume, ProjectionGeometry};
#[cfg(not(feature = "gpu"))]
use crate::nn::device::GpuContext;
#[cfg(not(feature = "gpu"))]
use anyhow::Result;

#[cfg(feature = "gpu")]
pub use imp::GpuProjector;

#[cfg(feature = "gpu")]
mod imp {
    use std::collections::HashMap;
    use std::sync::Mutex;

    use anyhow::{anyhow, bail, Result};
    use rds_engines::wgpu;

    use crate::geometry::Vec3;
    use crate::iec::{self, M3};
    use crate::nn::device::{panic_text, GpuContext};
    use crate::projection::{MuVolume, ProjectionGeometry};

    /// The ray-cast, as `render_posed` and `MuVolume::integrate` have it.
    const SHADER: &str = r#"
struct Camera {
    source: vec4<f32>,
    origin: vec4<f32>,
    row: vec4<f32>,
    col: vec4<f32>,
    ray_mm: vec4<f32>,
    row_mm: vec4<f32>,
    col_mm: vec4<f32>,
    step: vec4<f32>,
    detector: vec4<u32>,
    volume: vec4<u32>,
};

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var mu: texture_3d<f32>;
@group(0) @binding(2) var<storage, read_write> pixels: array<f32>;

fn at(i: i32, j: i32, k: i32) -> f32 {
    return textureLoad(mu, vec3<i32>(i, j, k), 0).x;
}

fn sample(p: vec3<f32>) -> f32 {
    if (!(p.x >= 0.0 && p.y >= 0.0 && p.z >= 0.0)) {
        return 0.0;
    }
    let n = vec3<i32>(cam.volume.xyz);
    let f = floor(p);
    let t = p - f;
    let a = vec3<i32>(f);
    if (a.x >= n.x || a.y >= n.y || a.z >= n.z) {
        return 0.0;
    }
    let b = min(a + vec3<i32>(1), n - vec3<i32>(1));
    let c00 = at(a.x, a.y, a.z) + (at(b.x, a.y, a.z) - at(a.x, a.y, a.z)) * t.x;
    let c10 = at(a.x, b.y, a.z) + (at(b.x, b.y, a.z) - at(a.x, b.y, a.z)) * t.x;
    let c01 = at(a.x, a.y, b.z) + (at(b.x, a.y, b.z) - at(a.x, a.y, b.z)) * t.x;
    let c11 = at(a.x, b.y, b.z) + (at(b.x, b.y, b.z) - at(a.x, b.y, b.z)) * t.x;
    let c0 = c00 + (c10 - c00) * t.y;
    let c1 = c01 + (c11 - c01) * t.y;
    return c0 + (c1 - c0) * t.z;
}

// The part [t.x, t.y] of a + t d (t in [0, 1]) inside [0, hi] along one
// axis; empty when the ray runs beside the box.
fn slab(a: f32, d: f32, hi: f32, t: vec2<f32>) -> vec2<f32> {
    if (abs(d) < 1e-12) {
        if (a < 0.0 || a > hi) {
            return vec2<f32>(1.0, 0.0);
        }
        return t;
    }
    var ta = (0.0 - a) / d;
    var tb = (hi - a) / d;
    if (ta > tb) {
        let s = ta;
        ta = tb;
        tb = s;
    }
    return vec2<f32>(max(t.x, ta), min(t.y, tb));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let cols = cam.detector.x;
    let rows = cam.detector.y;
    if (id.x >= cols || id.y >= rows) {
        return;
    }
    let fi = f32(id.x);
    let fj = f32(id.y);
    let a = cam.source.xyz;
    let d = cam.origin.xyz + cam.row.xyz * fi + cam.col.xyz * fj - a;
    let hi = vec3<f32>(max(cam.volume.xyz, vec3<u32>(1u)) - vec3<u32>(1u));
    var t = vec2<f32>(0.0, 1.0);
    t = slab(a.x, d.x, hi.x, t);
    t = slab(a.y, d.y, hi.y, t);
    t = slab(a.z, d.z, hi.z, t);
    var value = 0.0;
    if (t.x < t.y) {
        let total = length(cam.ray_mm.xyz + cam.row_mm.xyz * fi + cam.col_mm.xyz * fj);
        let span = (t.y - t.x) * total;
        if (span > 0.0) {
            let n = max(u32(ceil(span / max(cam.step.x, 1e-3))), 1u);
            let piece = span / f32(n);
            let dt = (t.y - t.x) / f32(n);
            var p = a + d * (t.x + 0.5 * dt);
            let dp = d * dt;
            var acc = 0.0;
            for (var s = 0u; s < n; s = s + 1u) {
                acc = acc + sample(p);
                p = p + dp;
            }
            value = acc * piece;
        }
    }
    pixels[id.y * cols + id.x] = value;
}
"#;

    /// Pixels per workgroup side (the shader's `workgroup_size`).
    const GROUP: u32 = 8;

    /// The uniform block: ten `vec4`s.
    const UNIFORM_BYTES: u64 = 160;

    /// A volume on the GPU and the pipeline that projects it.
    pub struct GpuProjector {
        device: wgpu::Device,
        queue: wgpu::Queue,
        pipeline: wgpu::ComputePipeline,
        view: wgpu::TextureView,
        _texture: wgpu::Texture,
        dims: [usize; 3],
        origin: Vec3,
        inv: M3,
        /// Buffers per detector size, kept between renderings; the lock also
        /// makes the renderings of one projector one at a time.
        scratch: Mutex<HashMap<usize, Scratch>>,
    }

    struct Scratch {
        uniform: wgpu::Buffer,
        pixels: wgpu::Buffer,
        read: wgpu::Buffer,
        bind: wgpu::BindGroup,
    }

    /// Run `f`, a panic in the backend becoming an error.
    fn caught<T>(f: impl FnOnce() -> Result<T>) -> Result<T> {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))
            .map_err(|p| anyhow!("the GPU projector failed ({})", panic_text(p.as_ref())))?
    }

    /// The validation and memory errors of what `f` asks of the device.
    fn scoped<T>(device: &wgpu::Device, f: impl FnOnce() -> T) -> Result<T> {
        let memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let out = f();
        let v = block_on(validation.pop());
        let m = block_on(memory.pop());
        if let Some(e) = v.or(m) {
            bail!("the GPU refused the projector: {e}");
        }
        Ok(out)
    }

    impl GpuProjector {
        /// Upload `mu` and build the pipeline.
        pub fn new(ctx: &GpuContext, mu: &MuVolume) -> Result<GpuProjector> {
            let (device, queue) = ctx.wgpu();
            let (device, queue) = (device.clone(), queue.clone());
            caught(|| {
                let [nx, ny, nz] = mu.dims;
                let largest = device.limits().max_texture_dimension_3d as usize;
                if nx.max(ny).max(nz) > largest {
                    bail!(
                        "the volume ({nx} × {ny} × {nz}) is larger than the GPU's 3-D \
                         textures ({largest})"
                    );
                }
                if nx * ny * nz == 0 {
                    bail!("the volume is empty");
                }
                let (texture, pipeline) = scoped(&device, || {
                    let texture = device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("rds projector volume"),
                        size: wgpu::Extent3d {
                            width: nx as u32,
                            height: ny as u32,
                            depth_or_array_layers: nz as u32,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D3,
                        format: wgpu::TextureFormat::R32Float,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                        view_formats: &[],
                    });
                    // A few slices at a time: one write of a whole CT
                    // would stage it all at once.
                    let plane = nx * ny;
                    let slices = (16 << 20) / (plane * 4).max(1);
                    let slices = slices.clamp(1, nz);
                    let mut k = 0;
                    while k < nz {
                        let n = slices.min(nz - k);
                        let bytes: Vec<u8> = mu.data[k * plane..(k + n) * plane]
                            .iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect();
                        queue.write_texture(
                            wgpu::TexelCopyTextureInfo {
                                texture: &texture,
                                mip_level: 0,
                                origin: wgpu::Origin3d {
                                    x: 0,
                                    y: 0,
                                    z: k as u32,
                                },
                                aspect: wgpu::TextureAspect::All,
                            },
                            &bytes,
                            wgpu::TexelCopyBufferLayout {
                                offset: 0,
                                bytes_per_row: Some(nx as u32 * 4),
                                rows_per_image: Some(ny as u32),
                            },
                            wgpu::Extent3d {
                                width: nx as u32,
                                height: ny as u32,
                                depth_or_array_layers: n as u32,
                            },
                        );
                        k += n;
                    }
                    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("rds projector"),
                        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
                    });
                    let pipeline =
                        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                            label: Some("rds projector"),
                            layout: None,
                            module: &module,
                            entry_point: Some("main"),
                            compilation_options: Default::default(),
                            cache: None,
                        });
                    (texture, pipeline)
                })?;
                let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
                Ok(GpuProjector {
                    device: device.clone(),
                    queue: queue.clone(),
                    pipeline,
                    view,
                    _texture: texture,
                    dims: mu.dims,
                    origin: mu.origin,
                    inv: mu.inv,
                    scratch: Mutex::new(HashMap::new()),
                })
            })
        }

        fn scratch(&self, pixels: usize) -> Result<Scratch> {
            let device = &self.device;
            scoped(device, || {
                let bytes = (pixels * 4) as u64;
                let uniform = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("rds projector camera"),
                    size: UNIFORM_BYTES,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let out = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("rds projector pixels"),
                    size: bytes,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                let read = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("rds projector readback"),
                    size: bytes,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("rds projector"),
                    layout: &self.pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: uniform.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&self.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: out.as_entire_binding(),
                        },
                    ],
                });
                Scratch {
                    uniform,
                    pixels: out,
                    read,
                    bind,
                }
            })
        }

        /// The camera in the volume's index space, as the shader reads it.
        fn uniform(&self, cam: &ProjectionGeometry, step_mm: f64) -> Vec<u8> {
            let ix = |p: Vec3| iec::mul_vec(&self.inv, p - self.origin);
            let dir = |v: Vec3| iec::mul_vec(&self.inv, v);
            let mut out = Vec::with_capacity(UNIFORM_BYTES as usize);
            let mut v4 = |v: Vec3| {
                for c in [v.x, v.y, v.z, 0.0] {
                    out.extend_from_slice(&(c as f32).to_le_bytes());
                }
            };
            v4(ix(cam.source));
            v4(ix(cam.origin));
            v4(dir(cam.row_step));
            v4(dir(cam.col_step));
            v4(cam.origin - cam.source);
            v4(cam.row_step);
            v4(cam.col_step);
            v4(Vec3::new(step_mm, 0.0, 0.0));
            for u in [cam.dims[0], cam.dims[1], 0, 0].into_iter().chain([
                self.dims[0],
                self.dims[1],
                self.dims[2],
                0,
            ]) {
                out.extend_from_slice(&(u as u32).to_le_bytes());
            }
            out
        }

        /// The line integral of every pixel of `cam`, row-major, as
        /// `render_posed` gives it.
        pub fn render(&self, cam: &ProjectionGeometry, step_mm: f64) -> Result<Vec<f32>> {
            let [w, h] = cam.dims;
            if w == 0 || h == 0 {
                bail!("the detector has no pixels");
            }
            let n = w * h;
            caught(|| {
                let mut cache = self
                    .scratch
                    .lock()
                    .map_err(|_| anyhow!("the GPU projector failed earlier"))?;
                if let std::collections::hash_map::Entry::Vacant(e) = cache.entry(n) {
                    e.insert(self.scratch(n)?);
                }
                let s = &cache[&n];
                self.queue
                    .write_buffer(&s.uniform, 0, &self.uniform(cam, step_mm));
                let mut enc = self
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("rds projector"),
                    });
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("rds projector"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.pipeline);
                    pass.set_bind_group(0, &s.bind, &[]);
                    pass.dispatch_workgroups(
                        (w as u32).div_ceil(GROUP),
                        (h as u32).div_ceil(GROUP),
                        1,
                    );
                }
                let bytes = (n * 4) as u64;
                enc.copy_buffer_to_buffer(&s.pixels, 0, &s.read, 0, bytes);
                self.queue.submit(Some(enc.finish()));
                let slice = s.read.slice(..bytes);
                let (tx, rx) = std::sync::mpsc::channel();
                slice.map_async(wgpu::MapMode::Read, move |r| {
                    let _ = tx.send(r);
                });
                self.device
                    .poll(wgpu::PollType::wait_indefinitely())
                    .map_err(|e| anyhow!("the GPU did not finish the projection ({e})"))?;
                rx.recv()
                    .map_err(|_| anyhow!("the GPU never answered"))?
                    .map_err(|e| anyhow!("the projection could not be read back ({e})"))?;
                let out: Vec<f32> = {
                    let view = slice.get_mapped_range();
                    view.chunks_exact(4)
                        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                        .collect()
                };
                s.read.unmap();
                Ok(out)
            })
        }
    }

    /// Wait for one of wgpu's futures; on native platforms they are ready
    /// once made.
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

/// Without the `gpu` feature there is no device, so no projector.
#[cfg(not(feature = "gpu"))]
pub struct GpuProjector {
    never: std::convert::Infallible,
}

#[cfg(not(feature = "gpu"))]
impl GpuProjector {
    pub fn new(ctx: &GpuContext, _mu: &MuVolume) -> Result<GpuProjector> {
        ctx.unreachable()
    }

    pub fn render(&self, _cam: &ProjectionGeometry, _step_mm: f64) -> Result<Vec<f32>> {
        match self.never {}
    }
}
