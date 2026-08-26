//! DICOM Segmentation round-trip: build segments in memory → export the
//! study → reload it with the normal directory scanner → the masks must come
//! back voxel for voxel.
//!
//! This exercises the whole path the application uses — the SEG writer, the
//! loader's classification of SEG files, the frame-position lattice
//! reconstruction and the resampling back onto the displayed volume — rather
//! than any one of them in isolation.

use rust_dicom_station::dicom_export::{self, ExportParams};
use rust_dicom_station::dicomseg::{self, SegSeries};
use rust_dicom_station::gen_test_data::{self, GenParams};
use rust_dicom_station::loader;
use rust_dicom_station::progress::Progress;
use rust_dicom_station::segmentation::Segmentation;

fn source_study(tag: &str) -> (std::path::PathBuf, loader::LoadedStudy) {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("target/{tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    gen_test_data::generate(&dir, &GenParams::default(), &Progress::default())
        .expect("test data generation succeeds");
    let study = loader::load_directory(&dir, &Progress::default()).expect("synthetic study loads");
    (dir, study)
}

/// A solid ellipsoid around the volume centre.
fn ball(dims: [usize; 3], radius: f64) -> Vec<u8> {
    let [nx, ny, nz] = dims;
    let c = [
        (nx as f64 - 1.0) * 0.5,
        (ny as f64 - 1.0) * 0.5,
        (nz as f64 - 1.0) * 0.5,
    ];
    let mut m = vec![0u8; nx * ny * nz];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let d = ((i as f64 - c[0]).powi(2)
                    + (j as f64 - c[1]).powi(2)
                    + (k as f64 - c[2]).powi(2))
                .sqrt();
                if d <= radius {
                    m[k * nx * ny + j * nx + i] = 1;
                }
            }
        }
    }
    m
}

/// An axis-aligned box, deliberately off-centre and touching neither the
/// first nor the last slice, so the writer's "only occupied slices become
/// frames" rule is actually exercised.
fn brick(dims: [usize; 3]) -> Vec<u8> {
    let [nx, ny, nz] = dims;
    let mut m = vec![0u8; nx * ny * nz];
    for k in 5..nz.saturating_sub(9) {
        for j in 10..20 {
            for i in 30..44 {
                m[k * nx * ny + j * nx + i] = 1;
            }
        }
    }
    m
}

#[test]
fn seg_export_import_roundtrip() {
    let (_dir, mut study) = source_study("test_data_seg");
    let dims = study.volume.dims;

    let mut ser = SegSeries::new(
        "QA segments".into(),
        study.volume.grid(),
        study.series[study.active_series].uid.clone(),
        study.series[study.active_series].study_uid.clone(),
    );
    ser.segs.push(Segmentation::from_mask(
        "Ball".into(),
        [220, 40, 40],
        dims,
        ball(dims, 9.0),
    ));
    ser.segs.push(Segmentation::from_mask(
        "Brick".into(),
        [40, 180, 90],
        dims,
        brick(dims),
    ));
    let want: Vec<(String, [u8; 3], usize, Vec<u8>)> = ser
        .segs
        .iter()
        .map(|s| (s.name.clone(), s.color, s.count, s.mask.clone()))
        .collect();
    assert!(want[0].2 > 0 && want[1].2 > 0, "the fixtures are not empty");
    study.seg_series.push(ser);

    // ---- export ----------------------------------------------------------
    let out = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/test_data_seg_out");
    let _ = std::fs::remove_dir_all(&out);
    let params = ExportParams::for_study(&study);
    dicom_export::export_study(&study, &out, &params, &Progress::default())
        .expect("export succeeds");
    let seg_files: Vec<_> = std::fs::read_dir(&out)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().starts_with("SEG_"))
        .collect();
    assert_eq!(seg_files.len(), 1, "one SEG file per segmentation series");

    // ---- reload ----------------------------------------------------------
    let back = loader::load_directory(&out, &Progress::default()).expect("export reloads");
    assert_eq!(
        back.seg_series.len(),
        1,
        "the SEG file was classified as SEG"
    );
    let mut got = back.seg_series[0].clone();
    assert_eq!(got.segs.len(), 2);
    assert_eq!(
        got.referenced_series_uid, back.series[back.active_series].uid,
        "the SEG points back at the exported image series"
    );

    // The frames only cover the occupied slices, so the reloaded lattice is a
    // sub-grid; putting it back on the volume must restore the exact masks.
    assert!(
        got.grid.dims[2] < dims[2],
        "empty slices must not be written as frames"
    );
    got.rebind(&back.volume);
    assert_eq!(got.grid.dims, dims);
    for (seg, (name, color, count, mask)) in got.segs.iter().zip(&want) {
        assert_eq!(&seg.name, name, "segment label survives");
        // CIELab is lossy at 16 bits per channel, but not by more than a step.
        for (a, b) in seg.color.iter().zip(color) {
            assert!(
                (*a as i32 - *b as i32).abs() <= 2,
                "segment colour {:?} vs {:?}",
                seg.color,
                color
            );
        }
        assert_eq!(seg.count, *count, "'{name}' voxel count");
        assert!(seg.mask == *mask, "'{name}' mask differs voxel for voxel");
    }
}

#[test]
fn resampling_between_lattices_is_exact_on_a_sub_grid() {
    let (_dir, study) = source_study("test_data_seg_grid");
    let vol = &study.volume;
    let dims = vol.dims;
    let mask = ball(dims, 7.0);

    // A lattice with the same orientation and spacing but half the slices,
    // starting at slice 4 — the shape a SEG file's frames produce.
    let mut sub = vol.grid();
    sub.dims = [dims[0], dims[1], dims[2] / 2];
    sub.origin = vol.voxel_to_patient(0.0, 0.0, 4.0);
    let down = dicomseg::resample_mask(&mask, &vol.grid(), &sub);
    let up = dicomseg::resample_mask(&down, &sub, &vol.grid());

    let [nx, ny, _] = dims;
    for k in 4..4 + sub.dims[2] {
        for j in 0..ny {
            for i in 0..nx {
                let idx = k * nx * ny + j * nx + i;
                assert_eq!(up[idx], mask[idx], "voxel ({i},{j},{k}) after two hops");
            }
        }
    }
    // Outside the sub-grid's slice range nothing may be invented.
    assert!(up[..4 * nx * ny].iter().all(|v| *v == 0));
}
