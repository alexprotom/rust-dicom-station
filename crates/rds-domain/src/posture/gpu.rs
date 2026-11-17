//! [`render_planes`](super::apply::render_planes) on the GPU: the three
//! slices on screen while the slider moves, as a WGSL compute shader, one
//! invocation per pixel - the way [`crate::projection::gpu`] runs the DRR.
//!
//! What is on the GPU and what is not. The states' images are uploaded once
//! (each a 3-D `R16Sint` texture, read with `textureLoad`: the
//! interpolation must be the CPU's, and a sampler would do its own). The
//! path at a time - [`super::PostureModel::frame`], a scaling and squaring
//! on the lattice - stays on the CPU, in `f64`: it is a few hundred
//! thousand nodes, not the millions of pixels, and it is what the
//! structures and meshes are carried with anyway. Each call uploads the
//! frame's two pull-back fields (and, for the Jacobian, the field to the
//! first state and the determinant per node) as storage buffers of `vec4`,
//! and the planes are dispatched in one command buffer: one submission and
//! one wait for all three.
//!
//! The shader is [`super::apply::Sampler::value`] transcribed: the point of
//! the pixel, its two pull-backs through the lattice fields with
//! [`super::svf::Lattice`]'s cell rule (linear extension for two cells past
//! the faces, then constant), each state's trilinear read with
//! [`super::apply::sample`]'s rule at the edges, and the five ways of
//! showing them. The affine maps (anchor frame to each state's voxels, to
//! the lattice's node coordinates) are composed on the CPU in `f64` and
//! handed over in `f32`; what differs from the CPU is that arithmetic in
//! single precision. The planes agree with the CPU's to a fraction of a
//! grey level almost everywhere (`tests/posture.rs`, run with
//! `RDS_GPU_TESTS`); where a pull-back lands within rounding of a scan's
//! edge one side may read the scan and the other the fill value.
//!
//! Failures of the backend (a lost device, a driver panic) are errors,
//! never crashes; [`render_planes_with`] then answers on the CPU and says
//! why.

use std::sync::Arc;

use anyhow::Result;

use super::apply::{self, Plane, Show};
use super::{Frame, PostureModel};
use crate::volume::{Grid, Volume};

#[cfg(not(feature = "gpu"))]
use crate::nn::device::GpuContext;

#[cfg(feature = "gpu")]
pub use imp::GpuSlicer;

/// The planes on the GPU when there is a slicer and it works, else on the
/// CPU; the second value says which ran, the third why the GPU did not
/// when it was asked.
pub fn render_planes_with(
    slicer: Option<&GpuSlicer>,
    model: &PostureModel,
    states: &[Arc<Volume>],
    frame: &Frame,
    show: Show,
    onto: &Grid,
    planes: &[Plane],
) -> Result<(Vec<Vec<f32>>, bool, Option<String>)> {
    let mut why = None;
    if let Some(g) = slicer {
        match g.render_planes(model, frame, show, onto, planes) {
            Ok(v) => return Ok((v, true, None)),
            Err(e) => why = Some(format!("{e:#}")),
        }
    }
    let v = apply::render_planes(model, states, frame, show, onto, planes)?;
    Ok((v, false, why))
}

#[cfg(feature = "gpu")]
mod imp {
    use std::sync::{Arc, Mutex};

    use anyhow::{anyhow, bail, Result};
    use rds_engines::wgpu;

    use crate::geometry::Vec3;
    use crate::nn::device::{panic_text, GpuContext};
    use crate::posture::apply::{Plane, Show};
    use crate::posture::svf::Field;
    use crate::posture::{Frame, PostureModel};
    use crate::volume::{Grid, Volume};

    /// The parts every field and image read shares.
    const SHADER_HEAD: &str = r#"
struct Params {
    origin: vec4<f32>,
    du: vec4<f32>,
    dv: vec4<f32>,
    lat0: vec4<f32>,
    lat1: vec4<f32>,
    lat2: vec4<f32>,
    a0: vec4<f32>,
    a1: vec4<f32>,
    a2: vec4<f32>,
    b0: vec4<f32>,
    b1: vec4<f32>,
    b2: vec4<f32>,
    lat_dims: vec4<u32>,
    a_dims: vec4<u32>,
    b_dims: vec4<u32>,
    size: vec4<u32>,
    misc: vec4<f32>,
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var img_a: texture_3d<i32>;
@group(0) @binding(2) var img_b: texture_3d<i32>;
@group(0) @binding(3) var<storage, read> to_prev: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> to_next: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> to_first: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> jac: array<f32>;
@group(0) @binding(7) var<storage, read_write> out: array<f32>;

fn lat_coords(p: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(P.lat0.xyz, p) + P.lat0.w,
        dot(P.lat1.xyz, p) + P.lat1.w,
        dot(P.lat2.xyz, p) + P.lat2.w,
    );
}

// Lattice::cell along one axis: (base, weight), the weight extending two
// cells past the faces.
fn axis_cell(u: f32, n: u32) -> vec2<f32> {
    if (n < 2u) {
        return vec2<f32>(0.0, 0.0);
    }
    let top = f32(n - 2u);
    let lo = clamp(floor(u), 0.0, top);
    return vec2<f32>(lo, clamp(u - lo, -2.0, 3.0));
}

fn node(i: u32, j: u32, k: u32) -> u32 {
    return i + P.lat_dims.x * (j + P.lat_dims.y * k);
}

fn l3(a: vec3<f32>, b: vec3<f32>, t: f32) -> vec3<f32> {
    return a + (b - a) * t;
}

// node_scalar: the Jacobian per node, trilinear, clamped to the nodes.
fn node_scalar(p: vec3<f32>) -> f32 {
    let u = lat_coords(p);
    let n = P.lat_dims.xyz;
    var b = vec3<u32>(0u, 0u, 0u);
    var w = vec3<f32>(0.0, 0.0, 0.0);
    if (n.x >= 2u) {
        let x = clamp(u.x, 0.0, f32(n.x - 1u));
        b.x = min(u32(floor(x)), n.x - 2u);
        w.x = x - f32(b.x);
    }
    if (n.y >= 2u) {
        let x = clamp(u.y, 0.0, f32(n.y - 1u));
        b.y = min(u32(floor(x)), n.y - 2u);
        w.y = x - f32(b.y);
    }
    if (n.z >= 2u) {
        let x = clamp(u.z, 0.0, f32(n.z - 1u));
        b.z = min(u32(floor(x)), n.z - 2u);
        w.z = x - f32(b.z);
    }
    var acc = 0.0;
    for (var dk = 0u; dk < 2u; dk = dk + 1u) {
        for (var dj = 0u; dj < 2u; dj = dj + 1u) {
            for (var di = 0u; di < 2u; di = di + 1u) {
                let i = min(b.x + di, n.x - 1u);
                let j = min(b.y + dj, n.y - 1u);
                let k = min(b.z + dk, n.z - 1u);
                var wt = 1.0;
                if (di == 1u) { wt = wt * w.x; } else { wt = wt * (1.0 - w.x); }
                if (dj == 1u) { wt = wt * w.y; } else { wt = wt * (1.0 - w.y); }
                if (dk == 1u) { wt = wt * w.z; } else { wt = wt * (1.0 - w.z); }
                acc = acc + wt * jac[node(i, j, k)];
            }
        }
    }
    return acc;
}

// apply::sample along one axis: (low corner, weight, inside).
fn img_axis(c: f32, n: u32) -> vec3<f32> {
    if (n == 0u || c < -1e-4 || c > f32(n - 1u) + 1e-4) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let u = clamp(c, 0.0, f32(n - 1u));
    let lo = min(floor(u), f32(max(n, 2u) - 2u));
    var f = u - lo;
    if (n == 1u) {
        f = 0.0;
    }
    return vec3<f32>(lo, f, 1.0);
}
"#;

    /// One field's trilinear read; `FIELD` is replaced by the buffer.
    const FIELD_READ: &str = r#"
fn sample_FIELD(p: vec3<f32>) -> vec3<f32> {
    let u = lat_coords(p);
    let n = P.lat_dims.xyz;
    let cx = axis_cell(u.x, n.x);
    let cy = axis_cell(u.y, n.y);
    let cz = axis_cell(u.z, n.z);
    let b = vec3<u32>(u32(cx.x), u32(cy.x), u32(cz.x));
    let f = vec3<f32>(cx.y, cy.y, cz.y);
    let i1 = min(b.x + 1u, n.x - 1u);
    let j1 = min(b.y + 1u, n.y - 1u);
    let k1 = min(b.z + 1u, n.z - 1u);
    let x00 = l3(FIELD[node(b.x, b.y, b.z)].xyz, FIELD[node(i1, b.y, b.z)].xyz, f.x);
    let x10 = l3(FIELD[node(b.x, j1, b.z)].xyz, FIELD[node(i1, j1, b.z)].xyz, f.x);
    let x01 = l3(FIELD[node(b.x, b.y, k1)].xyz, FIELD[node(i1, b.y, k1)].xyz, f.x);
    let x11 = l3(FIELD[node(b.x, j1, k1)].xyz, FIELD[node(i1, j1, k1)].xyz, f.x);
    return l3(l3(x00, x10, f.y), l3(x01, x11, f.y), f.z);
}
"#;

    /// One state's trilinear read; `IMG`, `R0..R2` and `DIMS` replaced.
    const IMAGE_READ: &str = r#"
fn at_IMG(i: u32, j: u32, k: u32) -> f32 {
    let n = P.DIMS.xyz;
    let c = vec3<i32>(vec3<u32>(min(i, n.x - 1u), min(j, n.y - 1u), min(k, n.z - 1u)));
    return f32(textureLoad(IMG, c, 0).x);
}

// (value, inside) of the state at an anchor-frame point.
fn read_IMG(p: vec3<f32>) -> vec2<f32> {
    let c = vec3<f32>(
        dot(P.R0.xyz, p) + P.R0.w,
        dot(P.R1.xyz, p) + P.R1.w,
        dot(P.R2.xyz, p) + P.R2.w,
    );
    let n = P.DIMS.xyz;
    let ax = img_axis(c.x, n.x);
    let ay = img_axis(c.y, n.y);
    let az = img_axis(c.z, n.z);
    if (ax.z < 0.5 || ay.z < 0.5 || az.z < 0.5) {
        return vec2<f32>(0.0, 0.0);
    }
    let i = u32(ax.x);
    let j = u32(ay.x);
    let k = u32(az.x);
    let x00 = at_IMG(i, j, k) + (at_IMG(i + 1u, j, k) - at_IMG(i, j, k)) * ax.y;
    let x10 = at_IMG(i, j + 1u, k) + (at_IMG(i + 1u, j + 1u, k) - at_IMG(i, j + 1u, k)) * ax.y;
    let x01 = at_IMG(i, j, k + 1u) + (at_IMG(i + 1u, j, k + 1u) - at_IMG(i, j, k + 1u)) * ax.y;
    let x11 = at_IMG(i, j + 1u, k + 1u) + (at_IMG(i + 1u, j + 1u, k + 1u) - at_IMG(i, j + 1u, k + 1u)) * ax.y;
    let y0 = x00 + (x10 - x00) * ay.y;
    let y1 = x01 + (x11 - x01) * ay.y;
    return vec2<f32>(y0 + (y1 - y0) * az.y, 1.0);
}
"#;

    const SHADER_MAIN: &str = r#"
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let nu = P.size.x;
    let nv = P.size.y;
    if (id.x >= nu || id.y >= nv) {
        return;
    }
    let y = P.origin.xyz + P.du.xyz * f32(id.x) + P.dv.xyz * f32(id.y);
    let show = P.size.z;
    let s = P.misc.x;
    let low = P.misc.y;
    var value = low;
    if (show == 4u) {
        value = 1000.0 * node_scalar(y + sample_to_first(y));
    } else {
        let a = read_img_a(y + sample_to_prev(y));
        let b = read_img_b(y + sample_to_next(y));
        let ha = a.y > 0.5;
        let hb = b.y > 0.5;
        if (show == 0u) {
            value = select(low, a.x, ha);
        } else if (show == 1u) {
            value = select(low, b.x, hb);
        } else if (show == 2u) {
            if (ha && hb) {
                value = (1.0 - s) * a.x + s * b.x;
            } else if (ha) {
                value = a.x;
            } else if (hb) {
                value = b.x;
            }
        } else {
            value = 0.0;
            if (ha && hb) {
                value = a.x - b.x;
            }
        }
    }
    out[id.y * nu + id.x] = value;
}
"#;

    fn shader() -> String {
        let mut s = SHADER_HEAD.to_string();
        for f in ["to_prev", "to_next", "to_first"] {
            s.push_str(&FIELD_READ.replace("FIELD", f));
        }
        for (img, r, dims) in [("img_a", "a", "a_dims"), ("img_b", "b", "b_dims")] {
            s.push_str(
                &IMAGE_READ
                    .replace("IMG", img)
                    .replace("R0", &format!("{r}0"))
                    .replace("R1", &format!("{r}1"))
                    .replace("R2", &format!("{r}2"))
                    .replace("DIMS", dims),
            );
        }
        s.push_str(SHADER_MAIN);
        s
    }

    /// Pixels per workgroup side (the shader's `workgroup_size`).
    const GROUP: u32 = 8;
    /// The uniform block: seventeen `vec4`s.
    const UNIFORM_BYTES: u64 = 17 * 16;

    /// One state's image on the GPU and where it lies.
    struct State {
        _texture: wgpu::Texture,
        view: wgpu::TextureView,
        grid: Grid,
    }

    /// The states' images on the GPU and the pipeline that slices them.
    pub struct GpuSlicer {
        device: wgpu::Device,
        queue: wgpu::Queue,
        pipeline: wgpu::ComputePipeline,
        states: Vec<State>,
        /// Each state's lowest value: a segment fills with the lower of its
        /// two, as the CPU's sampler does.
        lows: Vec<f32>,
        name: String,
        /// One rendering at a time.
        lock: Mutex<()>,
    }

    /// Run `f`, a panic in the backend becoming an error.
    fn caught<T>(f: impl FnOnce() -> Result<T>) -> Result<T> {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))
            .map_err(|p| anyhow!("the GPU slicer failed ({})", panic_text(p.as_ref())))?
    }

    /// The validation and memory errors of what `f` asks of the device.
    fn scoped<T>(device: &wgpu::Device, f: impl FnOnce() -> T) -> Result<T> {
        let memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let out = f();
        let v = block_on(validation.pop());
        let m = block_on(memory.pop());
        if let Some(e) = v.or(m) {
            bail!("the GPU refused the slicer: {e}");
        }
        Ok(out)
    }

    /// Rows of the affine map `x -> f(x)` (`f` affine), with the offset in
    /// `w`, as three `vec4`s.
    fn affine(f: impl Fn(Vec3) -> [f64; 3]) -> [[f32; 4]; 3] {
        let o = f(Vec3::ZERO);
        let cols = [
            f(Vec3::new(1.0, 0.0, 0.0)),
            f(Vec3::new(0.0, 1.0, 0.0)),
            f(Vec3::new(0.0, 0.0, 1.0)),
        ];
        std::array::from_fn(|r| {
            [
                (cols[0][r] - o[r]) as f32,
                (cols[1][r] - o[r]) as f32,
                (cols[2][r] - o[r]) as f32,
                o[r] as f32,
            ]
        })
    }

    fn v4(out: &mut Vec<u8>, v: [f32; 4]) {
        for c in v {
            out.extend_from_slice(&c.to_le_bytes());
        }
    }

    fn u4(out: &mut Vec<u8>, v: [u32; 4]) {
        for c in v {
            out.extend_from_slice(&c.to_le_bytes());
        }
    }

    fn p4(p: Vec3) -> [f32; 4] {
        [p.x as f32, p.y as f32, p.z as f32, 0.0]
    }

    fn field_bytes(f: &Field) -> Vec<u8> {
        f.d.iter()
            .flat_map(|v| [v.x as f32, v.y as f32, v.z as f32, 0.0])
            .flat_map(|c| c.to_le_bytes())
            .collect()
    }

    impl GpuSlicer {
        /// Upload the states' images and build the pipeline.
        pub fn new(ctx: &GpuContext, states: &[Arc<Volume>]) -> Result<GpuSlicer> {
            let (device, queue) = ctx.wgpu();
            let (device, queue) = (device.clone(), queue.clone());
            let name = ctx.describe();
            caught(|| {
                let largest = device.limits().max_texture_dimension_3d as usize;
                let mut up = Vec::with_capacity(states.len());
                for v in states {
                    let [nx, ny, nz] = v.dims;
                    if nx.max(ny).max(nz) > largest {
                        bail!(
                            "a state ({nx} × {ny} × {nz}) is larger than the GPU's 3-D \
                             textures ({largest})"
                        );
                    }
                    if nx * ny * nz == 0 {
                        bail!("a state's image is empty");
                    }
                    let texture = scoped(&device, || {
                        let texture = device.create_texture(&wgpu::TextureDescriptor {
                            label: Some("rds posture state"),
                            size: wgpu::Extent3d {
                                width: nx as u32,
                                height: ny as u32,
                                depth_or_array_layers: nz as u32,
                            },
                            mip_level_count: 1,
                            sample_count: 1,
                            dimension: wgpu::TextureDimension::D3,
                            format: wgpu::TextureFormat::R16Sint,
                            usage: wgpu::TextureUsages::TEXTURE_BINDING
                                | wgpu::TextureUsages::COPY_DST,
                            view_formats: &[],
                        });
                        // A few slices at a time, as the projector uploads.
                        let plane = nx * ny;
                        let slices = ((16 << 20) / (plane * 2).max(1)).clamp(1, nz);
                        let mut k = 0;
                        while k < nz {
                            let n = slices.min(nz - k);
                            let bytes: Vec<u8> = v.data[k * plane..(k + n) * plane]
                                .iter()
                                .flat_map(|x| x.to_le_bytes())
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
                                    bytes_per_row: Some(nx as u32 * 2),
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
                        texture
                    })?;
                    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
                    up.push(State {
                        _texture: texture,
                        view,
                        grid: v.grid(),
                    });
                }
                let source = shader();
                let pipeline = scoped(&device, || {
                    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("rds posture slices"),
                        source: wgpu::ShaderSource::Wgsl(source.into()),
                    });
                    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("rds posture slices"),
                        layout: None,
                        module: &module,
                        entry_point: Some("main"),
                        compilation_options: Default::default(),
                        cache: None,
                    })
                })?;
                Ok(GpuSlicer {
                    device: device.clone(),
                    queue: queue.clone(),
                    pipeline,
                    states: up,
                    lows: states.iter().map(|v| v.min_value as f32).collect(),
                    name: name.clone(),
                    lock: Mutex::new(()),
                })
            })
        }

        /// The device, as the user should read it.
        pub fn describe(&self) -> &str {
            &self.name
        }

        fn storage(&self, label: &str, bytes: &[u8]) -> wgpu::Buffer {
            // A buffer may not be empty, and a vec4 array not shorter than one.
            let size = (bytes.len() as u64).max(16).next_multiple_of(4);
            let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            if !bytes.is_empty() {
                self.queue.write_buffer(&buf, 0, bytes);
            }
            buf
        }

        /// The uniform block of one plane.
        #[allow(clippy::too_many_arguments)]
        fn uniform(
            &self,
            model: &PostureModel,
            frame: &Frame,
            show: Show,
            onto: &Grid,
            plane: &Plane,
            low: f32,
        ) -> Vec<u8> {
            let [ua, va] = plane.along();
            let step = |a: usize| -> Vec3 {
                match a {
                    0 => onto.row_dir * onto.spacing[0],
                    1 => onto.col_dir * onto.spacing[1],
                    _ => onto.normal * onto.spacing[2],
                }
            };
            let [i, j, k] = plane.voxel(0, 0);
            let origin = onto.voxel_to_patient(i as f64, j as f64, k as f64);
            let lat = &frame.forward.lattice;
            let lat_rows = affine(|p| lat.coords(p));
            let seg = frame.segment;
            let (ga, gb) = (&self.states[seg].grid, &self.states[seg + 1].grid);
            let (ta, tb) = (model.to_state[seg], model.to_state[seg + 1]);
            let a_rows = affine(|p| ga.patient_to_voxel(ta.map(p)));
            let b_rows = affine(|p| gb.patient_to_voxel(tb.map(p)));
            let [nu, nv] = plane.size(onto);
            let code = match show {
                Show::StartForward => 0,
                Show::EndBackward => 1,
                Show::CrossFade => 2,
                Show::Difference => 3,
                Show::Jacobian => 4,
            };
            let d = |g: &Grid| [g.dims[0] as u32, g.dims[1] as u32, g.dims[2] as u32, 0];
            let mut out = Vec::with_capacity(UNIFORM_BYTES as usize);
            v4(&mut out, p4(origin));
            v4(&mut out, p4(step(ua)));
            v4(&mut out, p4(step(va)));
            for r in lat_rows.iter().chain(&a_rows).chain(&b_rows) {
                v4(&mut out, *r);
            }
            u4(
                &mut out,
                [
                    lat.dims[0] as u32,
                    lat.dims[1] as u32,
                    lat.dims[2] as u32,
                    0,
                ],
            );
            u4(&mut out, d(ga));
            u4(&mut out, d(gb));
            u4(&mut out, [nu as u32, nv as u32, code, 0]);
            v4(&mut out, [frame.s as f32, low, 0.0, 0.0]);
            out
        }

        /// [`crate::posture::apply::render_planes`] of `frame`, on the GPU.
        pub fn render_planes(
            &self,
            model: &PostureModel,
            frame: &Frame,
            show: Show,
            onto: &Grid,
            planes: &[Plane],
        ) -> Result<Vec<Vec<f32>>> {
            let seg = frame.segment;
            if self.states.len() < seg + 2 {
                bail!("the slicer holds {} states", self.states.len());
            }
            for pl in planes {
                if !pl.fits(onto) {
                    bail!("plane {pl:?} is not on the grid");
                }
            }
            if planes.is_empty() {
                return Ok(Vec::new());
            }
            let _one = self
                .lock
                .lock()
                .map_err(|_| anyhow!("the GPU slicer failed earlier"))?;
            caught(|| {
                let low = self.lows[seg].min(self.lows[seg + 1]);
                let jac_bytes: Vec<u8> = if show == Show::Jacobian {
                    frame
                        .forward
                        .jacobians()
                        .iter()
                        .flat_map(|j| (*j as f32).to_le_bytes())
                        .collect()
                } else {
                    Vec::new()
                };
                let first_bytes = if show == Show::Jacobian {
                    field_bytes(&frame.to_first)
                } else {
                    Vec::new()
                };
                let (bufs, outs) = scoped(&self.device, || {
                    let prev = self.storage("rds posture to_prev", &field_bytes(&frame.to_prev));
                    let next = self.storage("rds posture to_next", &field_bytes(&frame.to_next));
                    let first = self.storage("rds posture to_first", &first_bytes);
                    let jac = self.storage("rds posture jacobian", &jac_bytes);
                    let outs: Vec<(wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, usize)> = planes
                        .iter()
                        .map(|pl| {
                            let [nu, nv] = pl.size(onto);
                            let bytes = (nu * nv * 4) as u64;
                            let uniform = self.device.create_buffer(&wgpu::BufferDescriptor {
                                label: Some("rds posture plane"),
                                size: UNIFORM_BYTES,
                                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                                mapped_at_creation: false,
                            });
                            self.queue.write_buffer(
                                &uniform,
                                0,
                                &self.uniform(model, frame, show, onto, pl, low),
                            );
                            let out = self.device.create_buffer(&wgpu::BufferDescriptor {
                                label: Some("rds posture pixels"),
                                size: bytes,
                                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                                mapped_at_creation: false,
                            });
                            let read = self.device.create_buffer(&wgpu::BufferDescriptor {
                                label: Some("rds posture readback"),
                                size: bytes,
                                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                                mapped_at_creation: false,
                            });
                            (uniform, out, read, nu * nv)
                        })
                        .collect();
                    ([prev, next, first, jac], outs)
                })?;
                let layout = self.pipeline.get_bind_group_layout(0);
                let (a, b) = (&self.states[seg].view, &self.states[seg + 1].view);
                let mut enc = self
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("rds posture slices"),
                    });
                let binds = scoped(&self.device, || {
                    outs.iter()
                        .map(|(uniform, out, _, _)| {
                            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("rds posture slices"),
                                layout: &layout,
                                entries: &[
                                    wgpu::BindGroupEntry {
                                        binding: 0,
                                        resource: uniform.as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 1,
                                        resource: wgpu::BindingResource::TextureView(a),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 2,
                                        resource: wgpu::BindingResource::TextureView(b),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 3,
                                        resource: bufs[0].as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 4,
                                        resource: bufs[1].as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 5,
                                        resource: bufs[2].as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 6,
                                        resource: bufs[3].as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 7,
                                        resource: out.as_entire_binding(),
                                    },
                                ],
                            })
                        })
                        .collect::<Vec<_>>()
                })?;
                for ((pl, (_, out, read, n)), bind) in planes.iter().zip(&outs).zip(&binds) {
                    let [nu, nv] = pl.size(onto);
                    {
                        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("rds posture slices"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&self.pipeline);
                        pass.set_bind_group(0, bind, &[]);
                        pass.dispatch_workgroups(
                            (nu as u32).div_ceil(GROUP),
                            (nv as u32).div_ceil(GROUP),
                            1,
                        );
                    }
                    enc.copy_buffer_to_buffer(out, 0, read, 0, (*n * 4) as u64);
                }
                self.queue.submit(Some(enc.finish()));
                let answers: Vec<_> = outs
                    .iter()
                    .map(|(_, _, read, n)| {
                        let (tx, rx) = std::sync::mpsc::channel();
                        read.slice(..(*n * 4) as u64)
                            .map_async(wgpu::MapMode::Read, move |r| {
                                let _ = tx.send(r);
                            });
                        rx
                    })
                    .collect();
                self.device
                    .poll(wgpu::PollType::wait_indefinitely())
                    .map_err(|e| anyhow!("the GPU did not finish the slices ({e})"))?;
                let mut result = Vec::with_capacity(planes.len());
                for ((_, _, read, n), rx) in outs.iter().zip(answers) {
                    rx.recv()
                        .map_err(|_| anyhow!("the GPU never answered"))?
                        .map_err(|e| anyhow!("the slices could not be read back ({e})"))?;
                    let px: Vec<f32> = {
                        let view = read.slice(..(*n * 4) as u64).get_mapped_range();
                        view.as_chunks::<4>()
                            .0
                            .iter()
                            .map(|b| f32::from_le_bytes(*b))
                            .collect()
                    };
                    read.unmap();
                    result.push(px);
                }
                Ok(result)
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

/// Without the `gpu` feature there is no device, so no slicer.
#[cfg(not(feature = "gpu"))]
pub struct GpuSlicer {
    never: std::convert::Infallible,
}

#[cfg(not(feature = "gpu"))]
impl GpuSlicer {
    pub fn new(ctx: &GpuContext, _states: &[Arc<Volume>]) -> Result<GpuSlicer> {
        ctx.unreachable()
    }

    pub fn describe(&self) -> &str {
        match self.never {}
    }

    pub fn render_planes(
        &self,
        _model: &PostureModel,
        _frame: &Frame,
        _show: Show,
        _onto: &Grid,
        _planes: &[Plane],
    ) -> Result<Vec<Vec<f32>>> {
        match self.never {}
    }
}
