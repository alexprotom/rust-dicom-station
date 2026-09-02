//! Which characters the interface is allowed to draw - and the font stack
//! that makes them all draw.
//!
//! egui carries four fonts of its own: Ubuntu-Light for text, Hack for
//! monospaced text, Noto Emoji and a small icon font for pictures. What it
//! does *not* carry is a way to notice that a character is in none of them:
//! the missing glyph comes out as an empty box, invisible to the compiler,
//! to every test and to whoever wrote the line, and visible only on the
//! user's screen. That is how ✎, 🤖, 🧠, 🖌, 🧪, ✕, ❐, 🧹 and a handful of
//! others once ended up in the interface as squares.
//!
//! Two things keep that from happening again.
//!
//! **[`install`] widens the font stack.** By default only Ubuntu-Light and
//! the two emoji fonts are used for ordinary (proportional) text, so
//! everything that lives *only* in Hack - the arrows ↑ ↓ ⇄ ⇤ ⇥, the set
//! algebra ∩ ∪ ⊕ ⊖, ● ◐ ▸ ▼ ⬚ - was a box in a button label while rendering
//! perfectly in the status bar's monospaced text. Hack is appended as the
//! last proportional fallback: no new font files, no new dependency, and the
//! mathematical symbols the structure algebra needs are simply there.
//!
//! **The test below is the guard.** Every non-ASCII character in the sources
//! has to appear in [`ALLOWED`], which lists what those four fonts really
//! draw - "really" meaning a `cmap` entry that is not `.notdef`. A new glyph
//! fails the build until somebody has checked it against
//! `epaint_default_fonts-*/fonts/*.ttf`.

/// Give ordinary text the monospaced font as its last fallback, so a symbol
/// that only Hack has still draws in a menu or on a button.
pub(super) fn install(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    if let Some(proportional) = fonts.families.get_mut(&egui::FontFamily::Proportional) {
        // The name is epaint's own key for its bundled Hack; if a future egui
        // renames it the push is harmless and the guard test catches the
        // glyphs that stop rendering.
        if !proportional.iter().any(|f| f == "Hack") {
            proportional.push("Hack".to_owned());
        }
    }
    ctx.set_fonts(fonts);
}

/// Every non-ASCII character the interface may use: verified present in
/// egui's bundled fonts. Add to this list only after checking the fonts.
#[cfg(test)]
pub(super) const ALLOWED: &str = "«°±²³·»Ö×ĊĠΔβμσφ–—“”…⁴↑→↓↺⇄⇤⇥⇧⇩⇮⇯⇱⇲−∩∪≈⊎⊕⊖⊗⊘⊞⋂⋃⌖⌚⌨⏩⏭⏮⏱⏳⏸⏹⏺■▣▲▶▸▼◀◉◌◍◎●◐◑◼☑☢♻⚒⚖⚙⚛⚠⚡⛶✂✏✒✔✖✚✨✱➕➖⟲⟳⤴⤵⬆⬇⬚🅰🎞🎨🎯🎲🏥🏷👁👤💡💬💾📁📂📄📈📊📋📌📎📐📝📤📥📦📩🔀🔁🔍🔏🔒🔓🔔🔗🔤🔧🔩🔬🔮🕐🕹🖊🖥🗑";

#[cfg(test)]
mod tests {
    use super::ALLOWED;

    /// Walk the crate's own sources and refuse any glyph not on the list.
    #[test]
    fn the_interface_draws_no_glyph_the_bundled_fonts_lack() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut offenders: Vec<String> = Vec::new();
        let mut stack = vec![root];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).expect("read src") {
                let path = entry.expect("dir entry").path();
                if path.is_dir() {
                    stack.push(path);
                    continue;
                }
                if path.extension().is_none_or(|e| e != "rs") {
                    continue;
                }
                let text = std::fs::read_to_string(&path).expect("read source");
                for (n, line) in text.lines().enumerate() {
                    // Comments are for the reader of the code, not for the
                    // screen; this file's own doc comment names the culprits.
                    if line.trim_start().starts_with("//") {
                        continue;
                    }
                    for ch in line.chars() {
                        if ch.is_ascii() || ALLOWED.contains(ch) {
                            continue;
                        }
                        offenders.push(format!(
                            "{}:{}: U+{:04X} {ch}",
                            path.file_name().unwrap_or_default().to_string_lossy(),
                            n + 1,
                            ch as u32
                        ));
                    }
                }
            }
        }
        assert!(
            offenders.is_empty(),
            "these characters are not known to render in egui's bundled fonts \
             and would come out as empty boxes:\n  {}\n\
             Pick one from `app::glyphs::ALLOWED`, or add it there once you have \
             found it in the `cmap` of one of \
             epaint_default_fonts-*/fonts/*.ttf",
            offenders.join("\n  ")
        );
    }
}
