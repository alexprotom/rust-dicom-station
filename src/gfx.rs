//! Which graphics backend the program draws and computes with.
//!
//! Everything here exists because of one failure that is common in the field
//! and impossible for a user to diagnose: a Windows machine advertises a
//! Vulkan driver that does not work. `wgpu` enumerates it, prefers it, and
//! the program dies before it has drawn anything - the same program that
//! runs perfectly on the next desk. The only escape used to be knowing to
//! type `$env:WGPU_BACKEND = "dx12"` before starting it, which is not a
//! thing to ask of a clinical physicist.
//!
//! So the choice is made explicit in three places, in this order of
//! authority:
//!
//! 1. the `WGPU_BACKEND` environment variable, if the user has set one -
//!    it stays the escape hatch and it still wins;
//! 2. the `graphics_backend` line in the settings file, which the installer
//!    writes from the page it asks on and the *Settings > Graphics backend* menu
//!    changes afterwards;
//! 3. failing both, whatever `wgpu` picks.
//!
//! And whatever is chosen, [`candidates`] gives the order to *fall back*
//! through when it does not work. That is the part that actually fixes the
//! reported problem: the program tries the next backend by itself instead of
//! presenting a stack trace.
//!
//! One subtlety worth stating. The program creates two independent `wgpu`
//! instances: `eframe` draws the interface with one, and `burn` runs the
//! networks on another. The first takes its backends as a typed argument;
//! the second is several layers down inside `cubecl` and takes them only
//! from the environment. So [`Backend::export`] sets `WGPU_BACKEND` for this
//! process, once, before anything has started a thread - which is both the
//! documented contract and exactly the workaround that was already known to
//! work.

/// A graphics backend the program can be asked to use.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Backend {
    /// Let `wgpu` choose. Correct on a healthy machine, and the reason this
    /// module exists on an unhealthy one.
    #[default]
    Auto,
    /// Cross-platform, and the fastest on most hardware - when the driver
    /// works.
    Vulkan,
    /// Windows only. Present and dependable wherever Windows 10 or later is.
    Dx12,
    /// Apple platforms only.
    Metal,
    /// The last resort: old, slow and almost always available.
    OpenGl,
}

impl Backend {
    /// Every backend, in the order the interface lists them.
    pub const ALL: [Backend; 5] = [
        Backend::Auto,
        Backend::Vulkan,
        Backend::Dx12,
        Backend::Metal,
        Backend::OpenGl,
    ];

    /// The ones worth offering on this platform.
    pub fn offered() -> Vec<Backend> {
        Backend::ALL
            .into_iter()
            .filter(|b| b.available_here())
            .collect()
    }

    /// False for backends this platform cannot have at all - DirectX on
    /// Linux, Metal anywhere but Apple. Offering them would only invite a
    /// choice that cannot work.
    pub fn available_here(self) -> bool {
        match self {
            Backend::Auto | Backend::OpenGl => true,
            Backend::Vulkan => !cfg!(target_os = "macos"),
            Backend::Dx12 => cfg!(target_os = "windows"),
            Backend::Metal => cfg!(target_os = "macos"),
        }
    }

    /// What the settings file and the command line spell it.
    pub fn key(self) -> &'static str {
        match self {
            Backend::Auto => "auto",
            Backend::Vulkan => "vulkan",
            Backend::Dx12 => "dx12",
            Backend::Metal => "metal",
            Backend::OpenGl => "opengl",
        }
    }

    /// The reverse, accepting the spellings people actually type.
    pub fn from_key(key: &str) -> Option<Backend> {
        match key.trim().to_ascii_lowercase().as_str() {
            "auto" | "default" | "" => Some(Backend::Auto),
            "vulkan" | "vk" => Some(Backend::Vulkan),
            "dx12" | "d3d12" | "directx" | "directx12" | "dx" => Some(Backend::Dx12),
            "metal" | "mtl" => Some(Backend::Metal),
            "opengl" | "gl" | "gles" => Some(Backend::OpenGl),
            _ => None,
        }
    }

    /// What the menu and the installer page call it.
    pub fn label(self) -> &'static str {
        match self {
            Backend::Auto => "Automatic",
            Backend::Vulkan => "Vulkan",
            Backend::Dx12 => "DirectX 12",
            Backend::Metal => "Metal",
            Backend::OpenGl => "OpenGL",
        }
    }

    /// Which of these a running `wgpu` adapter is.
    ///
    /// The menu says what the program is drawing with *now*, which is not
    /// always what was asked for: a machine whose Vulkan driver does not
    /// start falls through to the next backend by itself.
    pub fn from_wgpu(backend: eframe::wgpu::Backend) -> Backend {
        match backend {
            eframe::wgpu::Backend::Vulkan => Backend::Vulkan,
            eframe::wgpu::Backend::Dx12 => Backend::Dx12,
            eframe::wgpu::Backend::Metal => Backend::Metal,
            eframe::wgpu::Backend::Gl => Backend::OpenGl,
            _ => Backend::Auto,
        }
    }

    /// One line saying when to pick it.
    pub fn hint(self) -> &'static str {
        match self {
            Backend::Auto => "Let the graphics library decide. Right on most machines.",
            Backend::Vulkan => {
                "Usually the fastest. A few Windows machines advertise a Vulkan driver \
                 that does not work - if the program will not start, this is why."
            }
            Backend::Dx12 => {
                "Windows' own. Slightly slower than Vulkan on some cards, and \
                 dependable on every machine with Windows 10 or later."
            }
            Backend::Metal => "Apple's own, and the only one on macOS.",
            Backend::OpenGl => "Old and slow. Worth trying only when nothing else starts.",
        }
    }

    /// The `wgpu` backends this choice allows. `Auto` allows all of them, which is
    /// what `wgpu` does when left alone.
    pub fn bits(self) -> eframe::wgpu::Backends {
        use eframe::wgpu::Backends;
        match self {
            Backend::Auto => Backends::all(),
            Backend::Vulkan => Backends::VULKAN,
            Backend::Dx12 => Backends::DX12,
            Backend::Metal => Backends::METAL,
            Backend::OpenGl => Backends::GL,
        }
    }

    /// Publish the choice to `wgpu` through the environment, for the parts
    /// of the program that can only be reached that way - `burn`'s compute
    /// backend, several layers down inside `cubecl`.
    ///
    /// A value the user set themselves is never overwritten: it is the
    /// documented escape hatch, and someone who typed it is debugging.
    ///
    /// # Safety of the environment write
    ///
    /// Setting an environment variable is only sound while the process is
    /// single-threaded, which is why this must be called from the top of
    /// `main` before anything has spawned a thread - including the `rayon`
    /// pool, which is built lazily on first use.
    pub fn export(self) {
        if std::env::var_os(ENV_VAR).is_some() {
            return;
        }
        if self == Backend::Auto {
            return;
        }
        std::env::set_var(ENV_VAR, self.key());
    }
}

/// The variable `wgpu` reads, and the one the workaround used.
pub const ENV_VAR: &str = "WGPU_BACKEND";

/// The backend the environment asks for, if it asks for one.
///
/// This is checked before the settings file so that a user who has been told
/// to set the variable - or who has it in a launch script - keeps getting
/// what they asked for after an upgrade.
pub fn from_env() -> Option<Backend> {
    let raw = std::env::var(ENV_VAR).ok()?;
    Backend::from_key(&raw)
}

/// The order to try, starting with `first`.
///
/// The point is that a machine which cannot do the first choice still starts.
/// Windows always ends at DirectX 12, because a Windows machine that has
/// neither a working Vulkan nor a working D3D12 is not going to run anything;
/// everywhere else OpenGL is the floor.
pub fn candidates(first: Backend) -> Vec<Backend> {
    let mut out = vec![first];
    let rest: [Backend; 4] = if cfg!(target_os = "windows") {
        // The reported failure is Vulkan-on-Windows, so DirectX 12 is the
        // first thing to reach for after it.
        [
            Backend::Dx12,
            Backend::Vulkan,
            Backend::OpenGl,
            Backend::Auto,
        ]
    } else if cfg!(target_os = "macos") {
        [
            Backend::Metal,
            Backend::Auto,
            Backend::OpenGl,
            Backend::Auto,
        ]
    } else {
        [
            Backend::Vulkan,
            Backend::OpenGl,
            Backend::Auto,
            Backend::Auto,
        ]
    };
    for b in rest {
        if b.available_here() && !out.contains(&b) {
            out.push(b);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_spelling_round_trips() {
        for b in Backend::ALL {
            assert_eq!(Backend::from_key(b.key()), Some(b), "{}", b.key());
        }
        // …and the ones people type by hand.
        assert_eq!(Backend::from_key("DX12"), Some(Backend::Dx12));
        assert_eq!(Backend::from_key(" Vulkan "), Some(Backend::Vulkan));
        assert_eq!(Backend::from_key("d3d12"), Some(Backend::Dx12));
        assert_eq!(Backend::from_key("gl"), Some(Backend::OpenGl));
        assert_eq!(Backend::from_key(""), Some(Backend::Auto));
        assert_eq!(Backend::from_key("nonsense"), None);
    }

    #[test]
    fn the_fallback_list_starts_where_asked_and_repeats_nothing() {
        for first in Backend::offered() {
            let list = candidates(first);
            assert_eq!(list[0], first, "starts with the choice");
            let mut seen = list.clone();
            seen.sort_by_key(|b| b.key());
            seen.dedup();
            assert_eq!(seen.len(), list.len(), "no repeats in {list:?}");
            assert!(list.len() > 1, "there is always somewhere to fall back to");
            assert!(
                list.iter().all(|b| b.available_here()),
                "nothing impossible on this platform: {list:?}"
            );
        }
    }

    #[test]
    fn a_windows_vulkan_failure_reaches_directx_first() {
        // The whole point of the exercise: the machine that cannot do Vulkan
        // must try Windows' own backend next, not OpenGL.
        if cfg!(target_os = "windows") {
            assert_eq!(candidates(Backend::Vulkan)[1], Backend::Dx12);
        }
        // And everywhere, the automatic choice is in the list somewhere, so
        // a machine that defeats every explicit backend still gets whatever
        // wgpu would have picked on its own.
        assert!(candidates(Backend::Vulkan).contains(&Backend::Auto));
    }

    #[test]
    fn only_backends_this_platform_could_have_are_offered() {
        let offered = Backend::offered();
        assert!(offered.contains(&Backend::Auto));
        assert_eq!(
            cfg!(target_os = "windows"),
            offered.contains(&Backend::Dx12)
        );
        assert_eq!(cfg!(target_os = "macos"), offered.contains(&Backend::Metal));
    }
}
