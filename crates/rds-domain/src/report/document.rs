//! Encapsulated documents: a PDF filed with the study as a DICOM object.
//!
//! A radiologist's findings (*Befund*), a referral, a consent form come to
//! a PACS as *Encapsulated PDF Storage* (`1.2.840.10008.5.1.4.1.1.104.1`):
//! the file's whole content is the PDF, in `EncapsulatedDocument`
//! (0042,0011), with its title in `DocumentTitle` (0042,0010). The
//! Modality is whatever the sender chose - `DOC`, or `OT` as often - and
//! the object has no Pixel Data and no position, so neither a volume nor a
//! planar image can be made of it.
//!
//! So a document is a report ([`super::ReportKind::Document`]) that keeps
//! the PDF's bytes ([`Document`]) and draws its pages on demand. The pages
//! are rasterized on the CPU by `hayro` (pure Rust, so every build renders
//! the same pages), on a thread of their own the first time something asks
//! for them ([`Document::pages`]), and kept from then on: a loader that
//! never shows the document (the MCP server, a workflow) never pays for
//! the rendering.
//!
//! The document's title stays in [`Document::title`] and out of the
//! report's facts: a findings title often carries the patient's ID and an
//! order number, and the facts are what the MCP server returns.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use anyhow::{bail, Result};
use dicom_dictionary_std::tags;
use dicom_object::InMemDicomObject;

use crate::loader::str_of;

/// The pixels per PDF point the pages are drawn at: 2 is 144 dots per inch,
/// an A4 page of 1190 × 1684, sharp at the pane's usual sizes.
pub const RENDER_SCALE: f32 = 2.0;

/// More pages than a findings report has: a document past this is drawn
/// up to here, and says so.
pub const MAX_PAGES: usize = 40;

/// One rendered page: RGBA, row by row, on a white ground.
#[derive(Clone)]
pub struct Page {
    pub width: usize,
    pub height: usize,
    pub rgba: Vec<u8>,
}

/// The pages as drawn, or why they could not be.
#[derive(Clone)]
pub struct Rendered {
    pub pages: Vec<Page>,
    /// Pages in the document, which is more than `pages` past [`MAX_PAGES`].
    pub total: usize,
}

/// The document an encapsulated object carries.
#[derive(Clone, Default)]
pub struct Document {
    /// `application/pdf`, as the file states it.
    pub mime: String,
    /// `DocumentTitle`; empty when not stated.
    pub title: String,
    pub bytes: Arc<Vec<u8>>,
    pages: Arc<OnceLock<Result<Rendered, String>>>,
    started: Arc<AtomicBool>,
}

impl Document {
    /// Is this a PDF, which is what can be drawn?
    pub fn is_pdf(&self) -> bool {
        self.mime.eq_ignore_ascii_case("application/pdf") || self.bytes.starts_with(b"%PDF-")
    }

    /// The pages, once they are drawn: `None` while they are being drawn
    /// (the first call starts that, on a thread of its own), then the
    /// pages or the reason there are none. Clones share the drawing.
    pub fn pages(&self) -> Option<&Result<Rendered, String>> {
        if let Some(r) = self.pages.get() {
            return Some(r);
        }
        if !self.started.swap(true, Ordering::AcqRel) {
            let (bytes, cell) = (self.bytes.clone(), self.pages.clone());
            let pdf = self.is_pdf();
            std::thread::spawn(move || {
                let r = if pdf {
                    render(&bytes, RENDER_SCALE).map_err(|e| e.to_string())
                } else {
                    Err("not a PDF: only PDF documents are drawn".to_string())
                };
                let _ = cell.set(r);
            });
        }
        None
    }

    /// The pages, drawn here and now when nobody has yet: for a caller
    /// with no frame to come back in (the tests, a command line).
    pub fn pages_blocking(&self) -> &Result<Rendered, String> {
        self.pages.get_or_init(|| {
            if self.is_pdf() {
                render(&self.bytes, RENDER_SCALE).map_err(|e| e.to_string())
            } else {
                Err("not a PDF: only PDF documents are drawn".to_string())
            }
        })
    }
}

/// The document of an encapsulated object; `None` when it carries none.
pub fn read(obj: &InMemDicomObject) -> Option<Document> {
    let el = obj.element(tags::ENCAPSULATED_DOCUMENT).ok()?;
    let mut bytes = el.to_bytes().ok()?.into_owned();
    // The value is padded to an even length; the stated length, when there
    // is one, is the document's own.
    if let Some(n) = str_of(obj, tags::ENCAPSULATED_DOCUMENT_LENGTH)
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&n| n > 0 && n <= bytes.len())
    {
        bytes.truncate(n);
    }
    if bytes.is_empty() {
        return None;
    }
    Some(Document {
        mime: str_of(obj, tags::MIME_TYPE_OF_ENCAPSULATED_DOCUMENT).unwrap_or_default(),
        title: str_of(obj, tags::DOCUMENT_TITLE).unwrap_or_default(),
        bytes: Arc::new(bytes),
        ..Default::default()
    })
}

/// Draw the pages of a PDF at `scale` pixels per point, white behind.
pub fn render(bytes: &[u8], scale: f32) -> Result<Rendered> {
    use hayro::hayro_interpret::InterpreterSettings;
    use hayro::hayro_syntax::Pdf;
    use hayro::vello_cpu::color::palette::css::WHITE;
    use hayro::{PixmapSettings, RenderCache, RenderSettings};

    let pdf = match Pdf::new(bytes.to_vec()) {
        Ok(p) => p,
        Err(e) => bail!("the PDF cannot be read ({e:?})"),
    };
    let cache = RenderCache::new();
    let settings = InterpreterSettings::default();
    let all = pdf.pages();
    let total = all.len();
    let mut pages = Vec::new();
    for page in all.iter().take(MAX_PAGES) {
        let px = hayro::render(
            page,
            &cache,
            &settings,
            &RenderSettings::default(),
            &PixmapSettings {
                x_scale: scale,
                y_scale: scale,
                bg_color: WHITE,
            },
        );
        let (width, height) = (px.width() as usize, px.height() as usize);
        if width == 0 || height == 0 {
            continue;
        }
        // Premultiplied, but over an opaque white ground every alpha is
        // 255, so these are the colours as they are.
        pages.push(Page {
            width,
            height,
            rgba: px.data_as_u8_slice().to_vec(),
        });
    }
    if pages.is_empty() {
        bail!("the PDF has no page that can be drawn");
    }
    Ok(Rendered { pages, total })
}

/// The smallest PDF that draws something: one 200 × 100 pt page with a
/// black rectangle in its lower left quarter, for the tests and the
/// synthetic data. Offsets are written out so the cross-reference table is
/// exact.
pub fn sample_pdf() -> Vec<u8> {
    let content = "0 0 0 rg 0 0 100 50 re f\n";
    let objs = [
        "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_string(),
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] /Contents 4 0 R >>".to_string(),
        format!(
            "<< /Length {} >>\nstream\n{content}endstream",
            content.len()
        ),
    ];
    let mut out = b"%PDF-1.4\n".to_vec();
    let mut offsets = Vec::new();
    for (i, o) in objs.iter().enumerate() {
        offsets.push(out.len());
        out.extend_from_slice(format!("{} 0 obj\n{o}\nendobj\n", i + 1).as_bytes());
    }
    let xref = out.len();
    let n = objs.len() + 1;
    out.extend_from_slice(format!("xref\n0 {n}\n0000000000 65535 f \n").as_bytes());
    for off in offsets {
        out.extend_from_slice(format!("{off:010} 00000 n \n").as_bytes());
    }
    out.extend_from_slice(
        format!("trailer\n<< /Size {n} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n").as_bytes(),
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pdf_page_is_drawn_at_scale_on_white() {
        let r = render(&sample_pdf(), 2.0).unwrap();
        assert_eq!(r.total, 1);
        let p = &r.pages[0];
        assert_eq!((p.width, p.height), (400, 200));
        let at = |x: usize, y: usize| &p.rgba[(y * p.width + x) * 4..][..4];
        // PDF space runs upwards: the rectangle is at the bottom left.
        assert_eq!(at(50, 150), &[0, 0, 0, 255]);
        assert_eq!(at(350, 50), &[255, 255, 255, 255]);
    }

    #[test]
    fn what_is_not_a_pdf_says_so_rather_than_failing_the_load() {
        assert!(render(b"not a pdf", 1.0).is_err());
        let d = Document {
            mime: "text/xml".into(),
            bytes: Arc::new(b"<ClinicalDocument/>".to_vec()),
            ..Default::default()
        };
        assert!(!d.is_pdf());
        assert!(d.pages_blocking().is_err());
    }
}
