//! The system clipboard (`UIPasteboard`), for the viewer's *Paste* buttons
//! and its copy buttons.
//!
//! The window library reaches no clipboard on iOS, so without this a
//! connection line copied from Mail cannot be pasted into *Tools > PACS >
//! Add server*, and what the viewer's copy buttons copy never leaves the
//! program. The general pasteboard is read and written here; the viewer
//! shows its *Paste* buttons only where a clipboard is registered
//! (`rust_dicom_station::settings::clipboard`). Both calls come from the
//! viewer's own frame, on the main thread.

use objc2_foundation::NSString;
use objc2_ui_kit::UIPasteboard;
use rust_dicom_station::settings::clipboard::SystemClipboard;

pub struct Clipboard;

impl SystemClipboard for Clipboard {
    fn text(&self) -> Option<String> {
        // SAFETY: the general pasteboard is the process's shared one, and the
        // read happens on the main thread, from the viewer's frame.
        let text = unsafe { UIPasteboard::generalPasteboard().string() }?;
        Some(text.to_string())
    }

    fn set_text(&self, text: &str) {
        let text = NSString::from_str(text);
        // SAFETY: as in `text`.
        unsafe { UIPasteboard::generalPasteboard().setString(Some(&*text)) };
    }
}
