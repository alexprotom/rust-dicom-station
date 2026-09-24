//! Folders outside the app's sandbox, granted in the system's folder picker.
//!
//! iOS and iPadOS give an app its own container and nothing else. A folder in
//! iCloud Drive, in another app's space, on a USB drive or on a file server
//! can only be read through a *security-scoped* URL: the one
//! `UIDocumentPickerViewController` hands over when the user picks the
//! folder, or one resolved later from a bookmark made of it. While
//! `startAccessingSecurityScopedResource` is in force for such a URL, the
//! folder and everything below it are ordinary paths to `std::fs`, which is
//! all the viewer's loaders, exporters and archive need.
//!
//! What this module does:
//!
//! * [`Files::request`] presents the picker (folders only, opened in place,
//!   not copied) over whatever is on screen;
//! * the picker's delegate turns a pick into three things: access started,
//!   a bookmark written to `<settings folder>/places/`, and the path added
//!   to the list the file browser reads (`app/pick.rs` polls
//!   [`Files::granted`] while it is open);
//! * [`restore`], at start-up, resolves those bookmarks again, so a folder
//!   granted once stays granted across launches; a stale bookmark is
//!   renewed, one that no longer resolves (the drive is gone, the folder
//!   was deleted) is left on disk and skipped, so it comes back with the
//!   drive;
//! * [`Files::forget`] stops the access and deletes the bookmark.
//!
//! Every UIKit call here runs on the main thread, which is where `winit`
//! runs the event loop and therefore where egui draws the browser that
//! calls in. The URLs themselves stay in a thread-local on that thread; the
//! list of paths the viewer sees is behind a mutex, since
//! [`rust_dicom_station::settings::ios::Places`] must be `Send + Sync`.

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::{define_class, msg_send, ClassType, MainThreadMarker, MainThreadOnly, Message};
use objc2_foundation::{
    NSArray, NSData, NSObject, NSObjectProtocol, NSURLBookmarkCreationOptions,
    NSURLBookmarkResolutionOptions, NSURL,
};
use objc2_ui_kit::{
    UIApplication, UIDevice, UIDocumentPickerDelegate, UIDocumentPickerViewController,
    UISceneActivationState, UIUserInterfaceIdiom, UIViewController, UIWindowScene,
};
use objc2_uniform_type_identifiers::{UTType, UTTypeFolder};

use rust_dicom_station::settings;

/// Where the bookmarks live, one file per granted folder, named by the
/// time of the grant so that a listing sorts oldest first.
fn bookmark_dir() -> PathBuf {
    settings::config_dir().join("places")
}

/// One granted folder as the viewer sees it.
#[derive(Clone, Debug)]
struct Granted {
    path: PathBuf,
    bookmark: PathBuf,
}

/// The folders granted so far, oldest first.
static GRANTED: Mutex<Vec<Granted>> = Mutex::new(Vec::new());

thread_local! {
    /// The security-scoped URL of every granted folder, kept so that its
    /// access can be stopped again. Main thread only.
    static URLS: RefCell<Vec<(PathBuf, Retained<NSURL>)>> = const { RefCell::new(Vec::new()) };
    /// The picker's delegate. UIKit holds a delegate weakly, so something
    /// has to keep it alive while the picker is on screen.
    static DELEGATE: RefCell<Option<Retained<PickerDelegate>>> = const { RefCell::new(None) };
}

/// The viewer's handle on all of the above
/// (`rust_dicom_station::settings::ios::set_places`).
pub struct Files {
    /// An iPhone rather than an iPad: only the name of the app's own
    /// folder differs, as the Files app spells it.
    phone: bool,
}

impl Files {
    /// Called from `main`, on the main thread.
    pub fn new() -> Self {
        let phone = MainThreadMarker::new()
            .map(|mtm| {
                UIDevice::currentDevice(mtm).userInterfaceIdiom() == UIUserInterfaceIdiom::Phone
            })
            .unwrap_or(false);
        Self { phone }
    }
}

impl settings::ios::Places for Files {
    fn home_label(&self) -> &'static str {
        if self.phone {
            "On My iPhone"
        } else {
            "On My iPad"
        }
    }

    fn granted(&self) -> Vec<PathBuf> {
        lock().iter().map(|g| g.path.clone()).collect()
    }

    fn request(&self) {
        match MainThreadMarker::new() {
            Some(mtm) => present_picker(mtm),
            None => log::error!("the folder picker can only be shown from the main thread"),
        }
    }

    fn forget(&self, path: &Path) {
        let gone: Vec<Granted> = {
            let mut list = lock();
            let (gone, keep) = list.drain(..).partition(|g| g.path == path);
            *list = keep;
            gone
        };
        for g in gone {
            if let Err(e) = std::fs::remove_file(&g.bookmark) {
                log::warn!("could not remove {}: {e}", g.bookmark.display());
            }
        }
        if MainThreadMarker::new().is_some() {
            URLS.with(|urls| {
                urls.borrow_mut().retain(|(p, url)| {
                    if p == path {
                        // SAFETY: balanced with the start in `grant`.
                        unsafe { url.stopAccessingSecurityScopedResource() };
                        false
                    } else {
                        true
                    }
                })
            });
        }
    }
}

fn lock() -> std::sync::MutexGuard<'static, Vec<Granted>> {
    // A panic while the list was held leaves it as consistent as it was:
    // every change is one push or one replacement.
    GRANTED.lock().unwrap_or_else(|e| e.into_inner())
}

/// Resolve the bookmarks written by earlier runs and start access to each
/// folder they name. Called once, from `main`, before the viewer starts.
pub fn restore() {
    let dir = bookmark_dir();
    let Ok(rd) = std::fs::read_dir(&dir) else {
        return;
    };
    let mut files: Vec<PathBuf> = rd
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "bookmark"))
        .collect();
    files.sort();
    for file in files {
        match resolve(&file) {
            Ok(path) => log::info!("granted folder restored: {}", path.display()),
            Err(e) => log::warn!("{}: {e}", file.display()),
        }
    }
}

/// One bookmark file: resolve it, start access, renew it when stale.
fn resolve(file: &Path) -> Result<PathBuf, String> {
    let bytes = std::fs::read(file).map_err(|e| e.to_string())?;
    let data = NSData::with_bytes(&bytes);
    let mut stale = objc2::runtime::Bool::NO;
    // SAFETY: `stale` is a valid, writable `Bool` for the duration of the call.
    let url = unsafe {
        NSURL::URLByResolvingBookmarkData_options_relativeToURL_bookmarkDataIsStale_error(
            &data,
            NSURLBookmarkResolutionOptions(0),
            None,
            &mut stale,
        )
    }
    .map_err(|e| {
        format!(
            "the bookmark no longer resolves ({})",
            e.localizedDescription()
        )
    })?;
    // SAFETY: plain message send on a URL just returned by Foundation.
    if !unsafe { url.startAccessingSecurityScopedResource() } {
        return Err("iOS refused access to the folder".into());
    }
    let path = url
        .to_file_path()
        .ok_or_else(|| "the bookmark names no file path".to_owned())?;
    if stale.as_bool() {
        // The folder moved or was renamed: the URL still reaches it, the
        // stored bookmark would not next time.
        if let Err(e) = write_bookmark(&url, file) {
            log::warn!("could not renew {}: {e}", file.display());
        }
    }
    remember(path.clone(), file.to_path_buf(), url);
    Ok(path)
}

/// Record a folder whose access has been started.
fn remember(path: PathBuf, bookmark: PathBuf, url: Retained<NSURL>) {
    {
        let mut list = lock();
        if list.iter().any(|g| g.path == path) {
            // Granted again: keep the first entry, drop the new bookmark.
            drop(list);
            if bookmark.exists() {
                let _ = std::fs::remove_file(&bookmark);
            }
            // SAFETY: balanced with the start done by the caller.
            unsafe { url.stopAccessingSecurityScopedResource() };
            return;
        }
        list.push(Granted {
            path: path.clone(),
            bookmark,
        });
    }
    URLS.with(|urls| urls.borrow_mut().push((path, url)));
}

/// Write a bookmark of `url` to `file`.
fn write_bookmark(url: &NSURL, file: &Path) -> Result<(), String> {
    let data = url
        .bookmarkDataWithOptions_includingResourceValuesForKeys_relativeToURL_error(
            NSURLBookmarkCreationOptions(0),
            None,
            None,
        )
        .map_err(|e| e.localizedDescription().to_string())?;
    if let Some(dir) = file.parent() {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    std::fs::write(file, data.to_vec()).map_err(|e| e.to_string())
}

/// A folder the user just picked: start access, keep a bookmark, offer it.
fn grant(url: &NSURL) {
    // SAFETY: plain message send on a URL the picker handed over.
    if !unsafe { url.startAccessingSecurityScopedResource() } {
        log::warn!("iOS refused access to the folder just picked");
        return;
    }
    let Some(path) = url.to_file_path() else {
        log::warn!("the folder just picked has no file path");
        // SAFETY: balanced with the start above.
        unsafe { url.stopAccessingSecurityScopedResource() };
        return;
    };
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or_default();
    let file = bookmark_dir().join(format!("{stamp:015}.bookmark"));
    if let Err(e) = write_bookmark(url, &file) {
        // Still usable for this run; it just will not be back next time.
        log::warn!("could not keep a bookmark of {}: {e}", path.display());
    }
    log::info!("folder granted: {}", path.display());
    remember(path, file, url.retain());
}

define_class!(
    // SAFETY: NSObject has no subclassing requirements, and the class
    // implements no `Drop`.
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[name = "RDSFolderPickerDelegate"]
    struct PickerDelegate;

    unsafe impl NSObjectProtocol for PickerDelegate {}

    unsafe impl UIDocumentPickerDelegate for PickerDelegate {
        #[unsafe(method(documentPicker:didPickDocumentsAtURLs:))]
        fn did_pick(&self, _controller: &UIDocumentPickerViewController, urls: &NSArray<NSURL>) {
            for url in urls.to_vec() {
                grant(&url);
            }
        }

        #[unsafe(method(documentPickerWasCancelled:))]
        fn was_cancelled(&self, _controller: &UIDocumentPickerViewController) {}
    }
);

impl PickerDelegate {
    fn new(mtm: MainThreadMarker) -> Retained<Self> {
        let this = Self::alloc(mtm).set_ivars(());
        // SAFETY: NSObject's designated initialiser.
        unsafe { msg_send![super(this), init] }
    }
}

/// Show the folder picker over the view controller that is in front.
fn present_picker(mtm: MainThreadMarker) {
    let Some(front) = front_view_controller(mtm) else {
        log::error!("no window to show the folder picker over");
        return;
    };
    // SAFETY: an immutable static Foundation initialises before `main`.
    let folder: &UTType = unsafe { UTTypeFolder };
    let types = NSArray::from_slice(&[folder]);
    let picker = UIDocumentPickerViewController::initForOpeningContentTypes_asCopy(
        mtm.alloc(),
        &types,
        false,
    );
    picker.setAllowsMultipleSelection(false);
    let delegate = DELEGATE.with(|d| {
        d.borrow_mut()
            .get_or_insert_with(|| PickerDelegate::new(mtm))
            .clone()
    });
    picker.setDelegate(Some(ProtocolObject::from_ref(&*delegate)));
    front.presentViewController_animated_completion(&picker, true, None);
}

/// The key window's root view controller, or whatever it is presenting
/// already, walked to the front. The same scene search `egui-winit` does
/// for the safe-area insets.
fn front_view_controller(mtm: MainThreadMarker) -> Option<Retained<UIViewController>> {
    let app = UIApplication::sharedApplication(mtm);
    let mut root = None;
    for scene in app.connectedScenes().to_vec() {
        if !scene.isKindOfClass(UIWindowScene::class()) {
            continue;
        }
        if !matches!(
            scene.activationState(),
            UISceneActivationState::ForegroundActive | UISceneActivationState::ForegroundInactive
        ) {
            continue;
        }
        // SAFETY: the class was checked just above.
        let scene = unsafe { Retained::cast_unchecked::<UIWindowScene>(scene) };
        let window = scene
            .keyWindow()
            .or_else(|| scene.windows().to_vec().into_iter().next());
        if let Some(vc) = window.and_then(|w| w.rootViewController()) {
            root = Some(vc);
            break;
        }
    }
    let mut front = root?;
    while let Some(next) = front.presentedViewController() {
        if next.isBeingDismissed() {
            break;
        }
        front = next;
    }
    Some(front)
}

/// Keep the app's own folders out of iCloud and computer backups, as the
/// Android package does with `allowBackup="false"`: the archive, imported
/// studies and exports live there, and a patient study has no business in
/// a cloud backup. Everything below a folder marked this way is excluded.
pub fn exclude_from_backup(dir: &Path) {
    let Some(url) = NSURL::from_directory_path(dir) else {
        return;
    };
    let yes = objc2_foundation::NSNumber::numberWithBool(true);
    // SAFETY: the key's documented value type is a boolean NSNumber.
    let r = unsafe {
        url.setResourceValue_forKey_error(
            Some(&yes),
            objc2_foundation::NSURLIsExcludedFromBackupKey,
        )
    };
    if let Err(e) = r {
        log::warn!(
            "could not exclude {} from backups: {}",
            dir.display(),
            e.localizedDescription()
        );
    }
}
