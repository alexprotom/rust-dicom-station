//! The two calls into the Java side: is *all files access* granted, and
//! open the settings page where it is.
//!
//! `MANAGE_EXTERNAL_STORAGE` is not a permission a program can request in a
//! dialog. The user turns it on for the app in the system settings, and the
//! program can only ask `Environment.isExternalStorageManager()` and send
//! them to that page (`ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION` with
//! the package as the data URI). Both go through JNI on the activity's
//! Java VM; neither needs a class of our own.

use android_activity::AndroidApp;
use jni::objects::{JObject, JValue};
use jni::{jni_sig, jni_str, JavaVM};

/// Must match `package` in `AndroidManifest.xml`.
pub const PACKAGE: &str = "io.github.alexprotom.rustdicomstation";

pub(crate) fn vm(app: &AndroidApp) -> Result<JavaVM, String> {
    let raw = app.vm_as_ptr();
    if raw.is_null() {
        return Err("the activity has no Java VM".into());
    }
    // SAFETY: `vm_as_ptr` is the `JavaVM*` the activity was created with,
    // valid for the life of the process.
    Ok(unsafe { JavaVM::from_raw(raw.cast()) })
}

/// `Environment.isExternalStorageManager()`.
pub fn all_files_access(app: &AndroidApp) -> Result<bool, String> {
    vm(app)?
        .attach_current_thread(|env| {
            env.call_static_method(
                jni_str!("android/os/Environment"),
                jni_str!("isExternalStorageManager"),
                jni_sig!("()Z"),
                &[],
            )?
            .z()
        })
        .map_err(|e: jni::errors::Error| e.to_string())
}

/// `activity.startActivity(new Intent(ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION,
/// Uri.parse("package:" + PACKAGE)))`.
pub fn open_all_files_access_settings(app: &AndroidApp) -> Result<(), String> {
    let activity_raw = app.activity_as_ptr();
    if activity_raw.is_null() {
        return Err("the activity object is not available".into());
    }
    vm(app)?
        .attach_current_thread(|env| {
            // SAFETY: `activity_as_ptr` is the activity's `jobject`, a global
            // reference the glue keeps alive; it is only borrowed here.
            let activity = unsafe { JObject::from_raw(env, activity_raw.cast()) };
            let action =
                env.new_string("android.settings.MANAGE_APP_ALL_FILES_ACCESS_PERMISSION")?;
            let uri_text = env.new_string(format!("package:{PACKAGE}"))?;
            let uri = env
                .call_static_method(
                    jni_str!("android/net/Uri"),
                    jni_str!("parse"),
                    jni_sig!("(Ljava/lang/String;)Landroid/net/Uri;"),
                    &[JValue::Object(&uri_text)],
                )?
                .l()?;
            let intent = env.new_object(
                jni_str!("android/content/Intent"),
                jni_sig!("(Ljava/lang/String;Landroid/net/Uri;)V"),
                &[JValue::Object(&action), JValue::Object(&uri)],
            )?;
            env.call_method(
                &activity,
                jni_str!("startActivity"),
                jni_sig!("(Landroid/content/Intent;)V"),
                &[JValue::Object(&intent)],
            )?;
            Ok(())
        })
        .map_err(|e: jni::errors::Error| e.to_string())
}
