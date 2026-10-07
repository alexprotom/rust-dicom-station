//! The older name of [`seg_cli`](../seg_cli.rs), kept for one release so
//! scripts that call it keep working.
//!
//! ```text
//! cargo run --release --example autoseg_cli -- <dicom_dir> <out_prefix>
//!     [--variant fast3|highres|preview6|fast3-v3|highres-v3|preview6-v3|small3-v3|small15-v3]
//!     [--models DIR] [--device auto|gpu|cpu]
//!     [--parts organs,vertebrae,cardiac,muscles,ribs]
//! ```
//!
//! Every argument means what it always did: `--variant` names a
//! TotalSegmentator CT model (any key `seg_cli --list` prints is accepted
//! too), and `--models` may still be the `totalsegmentator/` folder itself.

#[path = "seg_cli.rs"]
mod seg_cli;

fn main() {
    if let Err(e) = seg_cli::run(std::env::args().skip(1).collect()) {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}
