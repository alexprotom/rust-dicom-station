//! The system clipboard, for the viewer's *Paste* buttons and its copy
//! buttons.
//!
//! The window library reaches no clipboard on Android, so without this a
//! connection line copied from a mail cannot be pasted into *Tools > PACS >
//! Add server*, and what the viewer's copy buttons put on egui's own
//! clipboard never leaves the program. Both go through the Java
//! `ClipboardManager` over JNI: `getPrimaryClip().getItemAt(0)
//! .coerceToText(context)` to read, `setPrimaryClip(ClipData
//! .newPlainText(..))` to write. Android lets a program read the clipboard
//! only while it is the one in the foreground with the focus, which a
//! button press guarantees; a read at any other time answers nothing.
//!
//! Registered with [`settings::clipboard`](rust_dicom_station::settings::clipboard)
//! at start-up; the viewer shows its *Paste* buttons only where a clipboard
//! is registered.

use android_activity::AndroidApp;
use jni::objects::{JObject, JString, JValue};
use jni::{jni_sig, jni_str};
use rust_dicom_station::settings::clipboard::SystemClipboard;

pub struct Clipboard {
    pub app: AndroidApp,
}

impl SystemClipboard for Clipboard {
    fn text(&self) -> Option<String> {
        match read(&self.app) {
            Ok(t) => t,
            Err(e) => {
                log::warn!("the clipboard could not be read: {e}");
                None
            }
        }
    }

    fn set_text(&self, text: &str) {
        if let Err(e) = write(&self.app, text) {
            log::warn!("the clipboard could not be written: {e}");
        }
    }
}

/// `(ClipboardManager) activity.getSystemService("clipboard")`.
fn manager<'l>(env: &mut jni::Env<'l>, activity: &JObject<'l>) -> jni::errors::Result<JObject<'l>> {
    let name = env.new_string("clipboard")?;
    env.call_method(
        activity,
        jni_str!("getSystemService"),
        jni_sig!("(Ljava/lang/String;)Ljava/lang/Object;"),
        &[JValue::Object(&name)],
    )?
    .l()
}

fn read(app: &AndroidApp) -> Result<Option<String>, String> {
    let activity_raw = app.activity_as_ptr();
    if activity_raw.is_null() {
        return Err("the activity object is not available".into());
    }
    super::permission::vm(app)?
        .attach_current_thread(|env| {
            // SAFETY: `activity_as_ptr` is the activity's `jobject`, a global
            // reference the glue keeps alive; it is only borrowed here.
            let activity = unsafe { JObject::from_raw(env, activity_raw.cast()) };
            let manager = manager(env, &activity)?;
            if manager.is_null() {
                return Ok(None);
            }
            let has = env
                .call_method(&manager, jni_str!("hasPrimaryClip"), jni_sig!("()Z"), &[])?
                .z()?;
            if !has {
                return Ok(None);
            }
            let clip = env
                .call_method(
                    &manager,
                    jni_str!("getPrimaryClip"),
                    jni_sig!("()Landroid/content/ClipData;"),
                    &[],
                )?
                .l()?;
            if clip.is_null() {
                return Ok(None);
            }
            let count = env
                .call_method(&clip, jni_str!("getItemCount"), jni_sig!("()I"), &[])?
                .i()?;
            if count < 1 {
                return Ok(None);
            }
            let item = env
                .call_method(
                    &clip,
                    jni_str!("getItemAt"),
                    jni_sig!("(I)Landroid/content/ClipData$Item;"),
                    &[JValue::Int(0)],
                )?
                .l()?;
            // `coerceToText` turns a URI or an intent item into text too; a
            // plain text item comes back as it is.
            let text = env
                .call_method(
                    &item,
                    jni_str!("coerceToText"),
                    jni_sig!("(Landroid/content/Context;)Ljava/lang/CharSequence;"),
                    &[JValue::Object(&activity)],
                )?
                .l()?;
            if text.is_null() {
                return Ok(None);
            }
            let text = env
                .call_method(
                    &text,
                    jni_str!("toString"),
                    jni_sig!("()Ljava/lang/String;"),
                    &[],
                )?
                .l()?;
            let text: JString = env.cast_local::<JString>(text)?;
            Ok(Some(text.try_to_string(env)?))
        })
        .map_err(|e: jni::errors::Error| e.to_string())
}

fn write(app: &AndroidApp, text: &str) -> Result<(), String> {
    let activity_raw = app.activity_as_ptr();
    if activity_raw.is_null() {
        return Err("the activity object is not available".into());
    }
    super::permission::vm(app)?
        .attach_current_thread(|env| {
            // SAFETY: as in `read`.
            let activity = unsafe { JObject::from_raw(env, activity_raw.cast()) };
            let manager = manager(env, &activity)?;
            if manager.is_null() {
                return Ok(());
            }
            let label = env.new_string("Rust DICOM Station")?;
            let value = env.new_string(text)?;
            let clip = env
                .call_static_method(
                    jni_str!("android/content/ClipData"),
                    jni_str!("newPlainText"),
                    jni_sig!("(Ljava/lang/CharSequence;Ljava/lang/CharSequence;)Landroid/content/ClipData;"),
                    &[JValue::Object(&label), JValue::Object(&value)],
                )?
                .l()?;
            env.call_method(
                &manager,
                jni_str!("setPrimaryClip"),
                jni_sig!("(Landroid/content/ClipData;)V"),
                &[JValue::Object(&clip)],
            )?;
            Ok(())
        })
        .map_err(|e: jni::errors::Error| e.to_string())
}
