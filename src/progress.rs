//! One progress handle for every background job.
//!
//! Loading, registration, meshing, export, anonymization and the three
//! segmentation engines all run on worker threads and all need the same
//! things from the thread that started them: a message to show, a fraction
//! for a progress bar, a cancel flag to poll, and — for the engines — the
//! name of the device the work ended up on. Rather than five near-identical
//! structs, there is one, and the workers see it through [`ProgressSink`].
//!
//! A phase window ([`Progress::set_phase`]) lets a multi-step job map each
//! step's own 0‥1 onto its slice of the overall bar, so a step that reports
//! through [`ProgressSink::report`] never has to know where it sits.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Mutex;

/// What a long operation reports to whoever started it. Implemented by
/// [`Progress`] for the application, and by [`Quiet`] / [`Stderr`] for tests
/// and the headless examples.
pub trait ProgressSink: Sync {
    /// `frac` is the operation's own progress in `0..=1`.
    fn report(&self, _frac: f32, _msg: &str) {}
    /// Polled between steps; a `true` makes the worker stop with an error
    /// whose message contains `cancelled`.
    fn cancelled(&self) -> bool {
        false
    }
}

/// A sink that ignores everything.
pub struct Quiet;
impl ProgressSink for Quiet {}

/// A sink that prints each report to standard error — for the examples.
pub struct Stderr;
impl ProgressSink for Stderr {
    fn report(&self, frac: f32, msg: &str) {
        eprintln!("[{:5.1}%] {msg}", frac * 100.0);
    }
}

/// Message, fraction, device label, cancel flag, phase window.
pub struct Progress {
    msg: Mutex<String>,
    device: Mutex<String>,
    /// `f32` bits of the overall fraction (`0..=1`).
    frac: AtomicU32,
    cancel: AtomicBool,
    /// The window `report`'s fraction is mapped onto: `[base, base + span]`.
    phase_base: AtomicU32,
    phase_span: AtomicU32,
}

impl Default for Progress {
    fn default() -> Self {
        Progress {
            msg: Mutex::new(String::new()),
            device: Mutex::new(String::new()),
            frac: AtomicU32::new(0f32.to_bits()),
            cancel: AtomicBool::new(false),
            phase_base: AtomicU32::new(0f32.to_bits()),
            phase_span: AtomicU32::new(1f32.to_bits()),
        }
    }
}

impl Progress {
    pub fn set(&self, m: impl Into<String>) {
        *self.msg.lock().unwrap_or_else(|e| e.into_inner()) = m.into();
    }
    pub fn get(&self) -> String {
        self.msg.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
    /// Which device the work runs on, e.g. `GPU (wgpu)`; empty until known.
    pub fn set_device(&self, d: impl Into<String>) {
        *self.device.lock().unwrap_or_else(|e| e.into_inner()) = d.into();
    }
    pub fn device(&self) -> String {
        self.device
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }
    /// Overall fraction in `0..=1`.
    pub fn frac(&self) -> f32 {
        f32::from_bits(self.frac.load(Ordering::Relaxed))
    }
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
    pub fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }
    /// Map subsequent [`ProgressSink::report`] fractions onto
    /// `[base, base + span]` of the overall bar, and move the bar to `base`.
    pub fn set_phase(&self, base: f32, span: f32) {
        self.phase_base.store(base.to_bits(), Ordering::Relaxed);
        self.phase_span.store(span.to_bits(), Ordering::Relaxed);
        self.frac.store(base.to_bits(), Ordering::Relaxed);
    }
}

impl ProgressSink for Progress {
    fn report(&self, frac: f32, msg: &str) {
        let base = f32::from_bits(self.phase_base.load(Ordering::Relaxed));
        let span = f32::from_bits(self.phase_span.load(Ordering::Relaxed));
        self.frac.store(
            (base + span * frac.clamp(0.0, 1.0)).to_bits(),
            Ordering::Relaxed,
        );
        if !msg.is_empty() {
            self.set(msg);
        }
    }
    fn cancelled(&self) -> bool {
        Progress::cancelled(self)
    }
}

/// Every engine signals cancellation by bailing with this text; the
/// application recognises it so a cancelled run is not shown as a failure.
pub const CANCELLED: &str = "cancelled";

/// True when an error is the worker acknowledging [`ProgressSink::cancelled`].
pub fn is_cancellation(e: &anyhow::Error) -> bool {
    format!("{e:#}").contains(CANCELLED)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_phase_window_maps_the_fraction_and_keeps_the_message() {
        let p = Progress::default();
        p.set("start");
        p.report(0.5, "");
        assert_eq!(p.frac(), 0.5);
        assert_eq!(p.get(), "start", "an empty message leaves the old one");
        p.set_phase(0.2, 0.5);
        assert_eq!(p.frac(), 0.2, "entering a phase moves the bar to its base");
        p.report(0.5, "half");
        assert!((p.frac() - 0.45).abs() < 1e-6);
        assert_eq!(p.get(), "half");
        p.report(2.0, "");
        assert!((p.frac() - 0.7).abs() < 1e-6, "fractions are clamped");
    }

    #[test]
    fn cancellation_is_visible_through_the_sink() {
        let p = Progress::default();
        assert!(!ProgressSink::cancelled(&p));
        p.cancel();
        assert!(ProgressSink::cancelled(&p));
        assert!(is_cancellation(&anyhow::anyhow!("{CANCELLED}")));
        assert!(!is_cancellation(&anyhow::anyhow!("out of memory")));
    }
}
