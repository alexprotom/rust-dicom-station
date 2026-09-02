//! Build script: give the setup executable the application's icon.
//!
//! The installer is the first thing a user sees of this program, often as a
//! downloaded file in Explorer, so it carries the same icon as the viewer it
//! installs - from the same file, `assets/rust-dicom-station.ico` in the
//! repository root (this crate is its own workspace one level below).
fn main() {
    println!("cargo:rerun-if-changed=../assets/rust-dicom-station.ico");
    #[cfg(windows)]
    {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("../assets/rust-dicom-station.ico");
        if let Err(e) = res.compile() {
            println!("cargo:warning=could not embed the application icon: {e}");
        }
    }
}
