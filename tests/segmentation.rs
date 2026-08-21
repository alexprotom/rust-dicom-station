//! Segmentation tool tests: brush geometry, undo, region growing, slice
//! overlay display conventions, RTSTRUCT conversion and surface meshing.

use rust_dicom_station::geometry::Vec3;
use rust_dicom_station::mesh3d::{self, GridGeom};
use rust_dicom_station::segmentation::{self, GrowState, Segmentation};
use rust_dicom_station::volume::{ViewPlane, Volume};

/// A synthetic volume with anisotropic spacing (1 x 1 x 2 mm) — the brush
/// and meshing math must honor per-axis spacing.
fn test_volume(dims: [usize; 3], fill: i16) -> Volume {
    Volume {
        data: vec![fill; dims[0] * dims[1] * dims[2]],
        dims,
        spacing: [1.0, 1.0, 2.0],
        origin: Vec3::new(0.0, 0.0, 0.0),
        row_dir: Vec3::new(1.0, 0.0, 0.0),
        col_dir: Vec3::new(0.0, 1.0, 0.0),
        normal: Vec3::new(0.0, 0.0, 1.0),
        frame_of_reference_uid: "1.2.3".into(),
        min_value: fill.min(0),
        max_value: fill.max(1000),
    }
}

#[test]
fn brush_paints_a_spacing_aware_sphere_and_undo_restores() {
    let vol = test_volume([40, 40, 24], 0);
    let mut seg = Segmentation::new("s".into(), [255, 0, 0], vol.dims);
    let c = [20.0, 20.0, 12.0];
    seg.paint_capsule(&vol, c, c, 5.0, false, None);
    // Ellipsoid of voxel radii (5, 5, 2.5): (4/3)π·5·5·2.5 ≈ 262 voxels.
    assert!(
        (180..350).contains(&seg.count),
        "unexpected sphere voxel count {}",
        seg.count
    );
    // Every set voxel lies within the radius (in mm).
    let [nx, ny, _] = vol.dims;
    for (idx, &m) in seg.mask.iter().enumerate() {
        if m == 0 {
            continue;
        }
        let k = idx / (nx * ny);
        let j = (idx % (nx * ny)) / nx;
        let i = idx % nx;
        let d = [
            (i as f64 - c[0]) * vol.spacing[0],
            (j as f64 - c[1]) * vol.spacing[1],
            (k as f64 - c[2]) * vol.spacing[2],
        ];
        let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        assert!(r <= 5.0 + 1e-9, "voxel at {r:.2} mm outside the brush");
    }
    seg.end_stroke();
    assert!(seg.can_undo());
    assert!(seg.undo_last());
    assert_eq!(seg.count, 0);
    assert!(seg.mask.iter().all(|&m| m == 0));
}

#[test]
fn brush_2d_stays_on_its_slice_and_erase_subtracts() {
    let vol = test_volume([40, 40, 24], 0);
    let mut seg = Segmentation::new("s".into(), [255, 0, 0], vol.dims);
    let c = [20.0, 20.0, 10.0];
    seg.paint_capsule(&vol, c, c, 6.0, false, Some((ViewPlane::Axial, 10)));
    assert!(seg.count > 0);
    let [nx, ny, _] = vol.dims;
    for (idx, &m) in seg.mask.iter().enumerate() {
        if m != 0 {
            assert_eq!(idx / (nx * ny), 10, "2D brush left its slice");
        }
    }
    seg.end_stroke();
    let painted = seg.count;
    seg.paint_capsule(&vol, c, c, 3.0, true, Some((ViewPlane::Axial, 10)));
    seg.end_stroke();
    assert!(seg.count < painted, "erasing did not remove voxels");
}

#[test]
fn geodesic_grow_suggests_the_organ_and_respects_its_boundary() {
    let mut vol = test_volume([30, 30, 20], 0);
    // A 5×5×5 "organ" of 1000 HU in a 0 HU background.
    for k in 5..10 {
        for j in 5..10 {
            for i in 5..10 {
                vol.data[k * 30 * 30 + j * 30 + i] = 1000;
            }
        }
    }
    let mut grow = GrowState::default();
    grow.seed(&vol, [7, 7, 7]);
    // At the default reach the whole organ — and only the organ — is
    // suggested: the intensity jump at its boundary is a geodesic barrier.
    assert_eq!(grow.voxels.len(), 125);
    assert!(!grow.capped);
    // Dragging down shrinks the selection to an inner subset…
    grow.set_level(&vol, 0.15);
    assert!(!grow.voxels.is_empty() && grow.voxels.len() < 125);
    // …dragging back up re-extends it instantly (incremental front)…
    grow.set_level(&vol, 1.0);
    assert_eq!(grow.voxels.len(), 125);
    // …and even a 20× reach does not cross the strong boundary.
    grow.set_level(&vol, 20.0);
    assert_eq!(grow.voxels.len(), 125);
    grow.release();
    assert!(grow.voxels.is_empty());
}

#[test]
fn slicewise_hole_filling_closes_enclosed_gaps() {
    let dims = [10, 10, 3];
    let (nx, ny) = (dims[0], dims[1]);
    let sl = nx * ny;
    // A square ring on slice k=1: border of 3..=6 × 3..=6, hollow inside.
    let mut voxels: Vec<u32> = Vec::new();
    for j in 3..=6 {
        for i in 3..=6 {
            if i == 3 || i == 6 || j == 3 || j == 6 {
                voxels.push((sl + j * nx + i) as u32);
            }
        }
    }
    assert_eq!(voxels.len(), 12);
    segmentation::fill_holes_slicewise(&mut voxels, dims);
    // The 2×2 enclosed hole is filled; the outer background is not.
    assert_eq!(voxels.len(), 16);
    for j in 4..=5 {
        for i in 4..=5 {
            assert!(voxels.contains(&((sl + j * nx + i) as u32)));
        }
    }
}

#[test]
fn overlay_matches_extract_slice_conventions() {
    let dims = [7, 6, 5];
    let (nx, ny, nz) = (dims[0], dims[1], dims[2]);
    let mut mask = vec![0u8; nx * ny * nz];
    let (i, j, k) = (3usize, 4usize, 2usize);
    mask[k * nx * ny + j * nx + i] = 1;
    let clear = egui::Color32::TRANSPARENT;
    let hit = |out: &[egui::Color32]| {
        out.iter()
            .enumerate()
            .filter(|(_, c)| **c != clear)
            .map(|(p, _)| p)
            .collect::<Vec<_>>()
    };

    let mut out = vec![clear; nx * ny];
    segmentation::overlay_slice(&mask, dims, ViewPlane::Axial, k, [255, 0, 0], 90, &mut out);
    assert_eq!(hit(&out), vec![j * nx + i]);

    // Sagittal: horizontal = j, vertical = k flipped (row = nz-1-k).
    let mut out = vec![clear; ny * nz];
    segmentation::overlay_slice(
        &mask,
        dims,
        ViewPlane::Sagittal,
        i,
        [255, 0, 0],
        90,
        &mut out,
    );
    assert_eq!(hit(&out), vec![(nz - 1 - k) * ny + j]);

    // Coronal: horizontal = i, vertical = k flipped.
    let mut out = vec![clear; nx * nz];
    segmentation::overlay_slice(
        &mask,
        dims,
        ViewPlane::Coronal,
        j,
        [255, 0, 0],
        90,
        &mut out,
    );
    assert_eq!(hit(&out), vec![(nz - 1 - k) * nx + i]);
}

#[test]
fn mask_converts_to_closed_rtstruct_contours() {
    let vol = test_volume([40, 40, 24], 0);
    let mut seg = Segmentation::new("disk".into(), [0, 255, 0], vol.dims);
    let c = [20.0, 20.0, 5.0];
    seg.paint_capsule(&vol, c, c, 5.0, false, Some((ViewPlane::Axial, 5)));
    let roi = segmentation::mask_to_roi(&seg, &vol, 7);
    assert_eq!(roi.number, 7);
    assert_eq!(roi.contours.len(), 1, "one disk → one closed contour");
    let contour = &roi.contours[0];
    assert_eq!(contour.geometric_type, "CLOSED_PLANAR");
    assert!(contour.points.len() >= 8);
    for p in &contour.points {
        // On the slice plane (k=5, 2 mm spacing → z = 10 mm)…
        assert!((p.z - 10.0).abs() < 1e-6);
        // …and on the rim of the 5 mm disk (within half a voxel).
        let r = ((p.x - 20.0).powi(2) + (p.y - 20.0).powi(2)).sqrt();
        assert!((4.0..=6.0).contains(&r), "contour point at radius {r:.2}");
    }
}

#[test]
fn mask_meshes_into_a_sphere_surface() {
    let vol = test_volume([40, 40, 24], 0);
    let mut seg = Segmentation::new("ball".into(), [0, 0, 255], vol.dims);
    let c = [20.0, 20.0, 12.0];
    seg.paint_capsule(&vol, c, c, 6.0, false, None);
    let (grid, gdims, lo, stride) = seg.mesh_grid().expect("non-empty mask meshes");
    assert_eq!(stride, 1, "small mask should mesh at native resolution");
    let (verts, normals, tris) =
        mesh3d::mesh_from_mask(&grid, gdims, lo, stride, &GridGeom::of(&vol))
            .expect("sphere produces a surface");
    assert!(!verts.is_empty() && !tris.is_empty());
    assert_eq!(verts.len(), normals.len());
    // Vertices sit near the 6 mm sphere around the patient-space center
    // (voxel (20, 20, 12) → (20, 20, 24) mm), within a cell of the surface.
    for v in &verts {
        let r = ((v[0] - 20.0).powi(2) + (v[1] - 20.0).powi(2) + (v[2] - 24.0).powi(2)).sqrt();
        assert!((3.0..=9.0).contains(&r), "vertex at {r:.2} mm from center");
    }
    for n in &normals {
        let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        assert!((l - 1.0).abs() < 1e-3, "normal not unit length: {l}");
    }
    // Triangle indices are in range.
    for t in &tris {
        assert!(t.iter().all(|&i| (i as usize) < verts.len()));
    }
}
