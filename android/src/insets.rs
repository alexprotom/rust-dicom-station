//! Where the system draws over the window: the status bar, the navigation
//! bar, a camera cut-out.
//!
//! A program that targets Android 15 is laid out edge to edge, with the
//! status bar drawn over the top of its window rather than above it. The
//! window library reports the whole surface, so without help the viewer's
//! menu bar sits under the clock. The insets come from the Java side:
//! `getWindow().getDecorView().getRootWindowInsets().getInsets(systemBars
//! | displayCutout)`, a read of cached values, and [`Shell`](crate::Shell)
//! keeps that much of the window free on each side.

use android_activity::AndroidApp;
use jni::objects::{JObject, JValue};
use jni::{jni_sig, jni_str};

/// Pixels the system covers on each side of the window.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Insets {
    pub left: i32,
    pub top: i32,
    pub right: i32,
    pub bottom: i32,
}

/// The current system-bar and cut-out insets of the activity's window;
/// zero before the window is attached.
pub fn system_insets(app: &AndroidApp) -> Result<Insets, String> {
    let activity_raw = app.activity_as_ptr();
    if activity_raw.is_null() {
        return Err("the activity object is not available".into());
    }
    super::permission::vm(app)?
        .attach_current_thread(|env| {
            // SAFETY: `activity_as_ptr` is the activity's `jobject`, a global
            // reference the glue keeps alive; it is only borrowed here.
            let activity = unsafe { JObject::from_raw(env, activity_raw.cast()) };
            let window = env
                .call_method(
                    &activity,
                    jni_str!("getWindow"),
                    jni_sig!("()Landroid/view/Window;"),
                    &[],
                )?
                .l()?;
            let decor = env
                .call_method(
                    &window,
                    jni_str!("getDecorView"),
                    jni_sig!("()Landroid/view/View;"),
                    &[],
                )?
                .l()?;
            let root = env.call_method(
                &decor,
                jni_str!("getRootWindowInsets"),
                jni_sig!("()Landroid/view/WindowInsets;"),
                &[],
            )?;
            if root.is_null() {
                return Ok(Insets::default());
            }
            let root = root.l()?;
            let bars = env
                .call_static_method(
                    jni_str!("android/view/WindowInsets$Type"),
                    jni_str!("systemBars"),
                    jni_sig!("()I"),
                    &[],
                )?
                .i()?;
            let cutout = env
                .call_static_method(
                    jni_str!("android/view/WindowInsets$Type"),
                    jni_str!("displayCutout"),
                    jni_sig!("()I"),
                    &[],
                )?
                .i()?;
            let insets = env
                .call_method(
                    &root,
                    jni_str!("getInsets"),
                    jni_sig!("(I)Landroid/graphics/Insets;"),
                    &[JValue::Int(bars | cutout)],
                )?
                .l()?;
            let field = |env: &mut jni::Env, name| -> jni::errors::Result<i32> {
                env.get_field(&insets, name, jni_sig!("I"))?.i()
            };
            Ok(Insets {
                left: field(env, jni_str!("left"))?,
                top: field(env, jni_str!("top"))?,
                right: field(env, jni_str!("right"))?,
                bottom: field(env, jni_str!("bottom"))?,
            })
        })
        .map_err(|e: jni::errors::Error| e.to_string())
}
