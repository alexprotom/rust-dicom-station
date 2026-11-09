//! 2D/2D registration of two planar images, headless
//! ([`rust_dicom_station::registration::planar`]).
//!
//! ```text
//! cargo run --release --example reg2d_cli -- <fixed.dcm> <moving.dcm> [options]
//! cargo run --release --example reg2d_cli -- <image.dcm> --simulate tx,ty,deg[,scale] [options]
//! cargo run --release --example reg2d_cli -- <image.dcm> --bench
//! ```
//!
//! Options: `--method intensity|points|points_then_intensity|deformable`,
//! `--metric ncc|gc|gd|pi|lncc|mi`, `--dof shift|shift_rotation|similarity`,
//! `--levels N`, `--pixels N`, `--range MM`, `--refine`, `--noise F` (with
//! `--simulate`, Gaussian noise at that fraction of the range, default
//! 0.01), `--json` (the report as JSON).
//!
//! `--simulate` is the self-test on any radiograph: the moving image is the
//! fixed one resampled through the inverse of the transform given (shift in
//! mm at the detector, rotation in degrees, scale), so registering it back
//! should answer that transform; the error is printed. `--bench` times the
//! resampler and each measure on the image at 256 pixels.

use std::time::Instant;

use anyhow::{anyhow, bail, Result};
use rust_dicom_station::extras::load_planar;
use rust_dicom_station::progress::Stderr;
use rust_dicom_station::registration::metric2d::Metric2d;
use rust_dicom_station::registration::planar::{
    self as p2, Dof2d, Method2d, Planar2d, Request2d, Rigid2d, Transform2d,
};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    // The options that take a value; everything else not an option is a
    // file.
    const VALUED: [&str; 8] = [
        "--method",
        "--metric",
        "--dof",
        "--levels",
        "--pixels",
        "--range",
        "--noise",
        "--simulate",
    ];
    let opt = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let has = |name: &str| args.iter().any(|a| a == name);
    let mut files = Vec::new();
    let mut k = 0;
    while k < args.len() {
        if VALUED.contains(&args[k].as_str()) {
            k += 2;
            continue;
        }
        if !args[k].starts_with("--") {
            files.push(args[k].clone());
        }
        k += 1;
    }
    let Some(first) = files.first() else {
        bail!("usage: reg2d_cli <fixed.dcm> <moving.dcm> | <image.dcm> --simulate tx,ty,deg[,s] | <image.dcm> --bench");
    };
    let fixed_img = load_planar(std::path::Path::new(first))?;
    let fixed = Planar2d::of(&fixed_img);
    if has("--bench") {
        return bench(&fixed);
    }
    let simulate = opt("--simulate");
    let truth = simulate
        .as_deref()
        .map(|s| -> Result<Rigid2d> {
            let v: Vec<f64> = s
                .split(',')
                .map(|x| x.trim().parse::<f64>())
                .collect::<std::result::Result<_, _>>()
                .map_err(|e| anyhow!("--simulate {s}: {e}"))?;
            if v.len() < 3 {
                bail!("--simulate takes tx,ty,deg[,scale]");
            }
            Ok(Rigid2d {
                tx: v[0],
                ty: v[1],
                theta: v[2].to_radians(),
                scale: v.get(3).copied().unwrap_or(1.0),
                center: [0.0, 0.0],
            })
        })
        .transpose()?;
    let moving = match (&truth, files.get(1)) {
        (Some(t), _) => {
            let noise: f64 = opt("--noise").and_then(|v| v.parse().ok()).unwrap_or(0.01);
            let mut img = p2::resample(&fixed, &Transform2d::rigid(t.inverse()), &fixed);
            let (lo, hi) = img
                .data
                .iter()
                .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
            let mut state = 0x2D2Du64;
            let mut gauss = || {
                let mut u = || {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
                };
                let (a, b) = (u(), u());
                (-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()
            };
            for v in &mut img.data {
                *v += (noise * (hi - lo) as f64 * gauss()) as f32;
            }
            Planar2d {
                image: img,
                ..fixed.clone()
            }
        }
        (None, Some(path)) => Planar2d::of(&load_planar(std::path::Path::new(path))?),
        (None, None) => bail!("give a moving image, or --simulate"),
    };
    // A simulation compares about the image centre: no camera to turn
    // about the isocentre's projection.
    let fixed = if truth.is_some() {
        Planar2d {
            camera: None,
            isocenter: None,
            ..fixed
        }
    } else {
        fixed
    };
    let moving = if truth.is_some() {
        Planar2d {
            camera: None,
            isocenter: None,
            ..moving
        }
    } else {
        moving
    };
    let mut req = Request2d::new(fixed, moving);
    if let Some(m) = opt("--method") {
        req.method = Method2d::from_key(&m).ok_or_else(|| anyhow!("--method {m}"))?;
    }
    if let Some(m) = opt("--metric") {
        req.metric = Metric2d::from_key(&m).ok_or_else(|| anyhow!("--metric {m}"))?;
    }
    if let Some(d) = opt("--dof") {
        req.dof = Dof2d::from_key(&d).ok_or_else(|| anyhow!("--dof {d}"))?;
    }
    if let Some(v) = opt("--levels").and_then(|v| v.parse().ok()) {
        req.levels = v;
    }
    if let Some(v) = opt("--pixels").and_then(|v| v.parse().ok()) {
        req.max_pixels = v;
    }
    if let Some(v) = opt("--range").and_then(|v| v.parse().ok()) {
        req.range_mm = v;
    }
    req.refine = has("--refine");
    let r = p2::register(&req, &Stderr)?;
    if has("--json") {
        println!("{}", serde_json::to_string_pretty(&r.json())?);
    } else {
        println!("{}", r.line());
        println!(
            "{} {:.4} -> {:.4}, {} evaluations, {:.2} s",
            r.metric.label(),
            r.before,
            r.after,
            r.evaluations,
            r.elapsed_secs
        );
    }
    if let Some(t) = truth {
        let g = r.transform.rigid.recentered(t.center);
        let at = |p: [f64; 2]| {
            let (a, b) = (g.map(p), t.map(p));
            ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
        };
        println!(
            "error: {:.3} mm at the centre, {:.3} deg, scale {:+.3} %",
            at([0.0, 0.0]),
            (g.theta - t.theta).to_degrees().abs(),
            (g.scale / t.scale - 1.0) * 100.0
        );
    }
    Ok(())
}

/// Time the resampler and the measures on the image at 256 pixels.
fn bench(img: &Planar2d) -> Result<()> {
    let mut small = img.image.clone();
    let mut f = 1.0;
    while small.dims[0].max(small.dims[1]) > 256 {
        small = small.halved();
        f *= 2.0;
    }
    let p = Planar2d {
        image: small,
        spacing: [img.spacing[0] * f, img.spacing[1] * f],
        camera: None,
        isocenter: None,
        transmission: false,
    };
    let t = Transform2d::rigid(Rigid2d {
        tx: 3.0,
        ty: -2.0,
        theta: 0.02,
        ..Rigid2d::identity([0.0, 0.0])
    });
    let n = 50;
    let t0 = Instant::now();
    let mut moved = p2::resample(&p, &t, &p);
    for _ in 1..n {
        moved = p2::resample(&p, &t, &p);
    }
    println!(
        "resample {} x {}: {:.3} ms",
        p.image.dims[0],
        p.image.dims[1],
        t0.elapsed().as_secs_f64() * 1e3 / n as f64
    );
    for m in Metric2d::ALL {
        let t0 = Instant::now();
        let mut v = 0.0;
        for _ in 0..n {
            v = m.similarity(&p.image.data, &moved.data, p.image.dims, None, 7);
        }
        println!(
            "{:<28} {:.3} ms ({v:.4})",
            m.label(),
            t0.elapsed().as_secs_f64() * 1e3 / n as f64
        );
    }
    Ok(())
}
