//! Build script: give the Windows executable its icon.
//!
//! On Windows an application's icon is a resource compiled into the `.exe`
//! itself — Explorer, the task bar, Alt+Tab and every shortcut that points at
//! the executable read it from there, which is also what the installer's
//! shortcuts and its *Add or remove programs* entry rely on. The runtime
//! window icon comes from the same picture (see `app_icon` in `main.rs`),
//! so the program looks the same before and after it starts.
//!
//! Everywhere else this script does nothing: on Linux and macOS the icon
//! travels with the desktop entry / bundle instead.
fn main() {
    println!("cargo:rerun-if-changed=assets/rust-dicom-station.ico");
    #[cfg(windows)]
    {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("assets/rust-dicom-station.ico");
        // A missing resource compiler is not worth failing a build over: the
        // program runs perfectly well with the default executable icon.
        if let Err(e) = res.compile() {
            println!("cargo:warning=could not embed the application icon: {e}");
        }
    }
}
