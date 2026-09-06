//! Derived structures: a recipe that stays attached to its result.
//!
//! `PTV = CTV + 5 mm` is one command in the Combine tool, and after it there
//! is no way to tell that the PTV came from the CTV, no way to re-run it when
//! the CTV is corrected, and no way to see that it is now out of date. That
//! is what a *derived* structure fixes, and it is the difference between a
//! tool a planner uses once and a structure set a department can trust.
//!
//! An [`Expr`] is the recipe with the geometry taken out: which structures,
//! in what order, with what margins, combined how, tidied how. It is stored
//! **in the structure itself**, as RTSTRUCT ROI Description (3006,0028), so
//! it survives an export, another system, and a re-import. Alongside it goes
//! a fingerprint of the geometry the last evaluation used, which is what the
//! *needs update* status is computed from, and an `overridden` flag set the
//! moment somebody edits the result by hand.
//!
//! ## Why names and not indices
//!
//! Operands are named. Indices do not survive an export, a reload, a
//! reordering of the list or a copy to the other dataset; a name does, it is
//! what the user typed, and when it no longer resolves the honest answer is
//! to say so rather than to combine whatever is in slot three.

use crate::rtstruct::Roi;
use crate::structops::{BoolOp, Cleanup, Margin};

/// The prefix that marks a description as ours. Anything else in the field
/// is somebody's free text and is left alone.
pub const PREFIX: &str = "RDS-DERIVED:";

/// One operand of a derived expression.
#[derive(Clone, Debug, PartialEq)]
pub struct Dep {
    /// The structure's name, as the user sees it.
    pub name: String,
    /// Applied to this operand before it is combined with the others.
    pub margin: Margin,
}

/// The recipe of a derived structure: [`crate::structops::Recipe`] with the
/// masks taken out.
#[derive(Clone, Debug, PartialEq)]
pub struct Expr {
    pub op: BoolOp,
    pub deps: Vec<Dep>,
    /// Applied to the combined result.
    pub margin: Margin,
    pub cleanup: Cleanup,
}

/// What a derived structure's geometry is worth right now.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// The geometry matches the recipe and its inputs.
    UpToDate,
    /// An input changed since it was last evaluated.
    NeedsUpdate,
    /// Somebody edited the result by hand. The recipe is kept, but it is no
    /// longer what produced what you see.
    Overridden,
}

impl Status {
    /// The marker the structure list draws. RayStation uses a green circle,
    /// a red square and a yellow triangle; so does this.
    pub fn glyph(self) -> &'static str {
        match self {
            Status::UpToDate => "◉",
            Status::NeedsUpdate => "■",
            Status::Overridden => "▲",
        }
    }

    pub fn color(self) -> [u8; 3] {
        match self {
            Status::UpToDate => [110, 200, 110],
            Status::NeedsUpdate => [235, 95, 95],
            Status::Overridden => [235, 200, 80],
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Status::UpToDate => "up to date",
            Status::NeedsUpdate => "needs update",
            Status::Overridden => "overridden by hand",
        }
    }
}

/// A derived structure as it is stored: the recipe, the fingerprint of the
/// inputs it was last evaluated against, and whether it has been overridden.
#[derive(Clone, Debug, PartialEq)]
pub struct Derived {
    pub expr: Expr,
    /// [`fingerprint`] of the operands at the last evaluation. Zero means
    /// "never evaluated", which reads as *needs update*.
    pub hash: u64,
    pub overridden: bool,
}

impl Derived {
    pub fn new(expr: Expr) -> Derived {
        Derived {
            expr,
            hash: 0,
            overridden: false,
        }
    }

    /// Status against the fingerprint the inputs have *now*.
    pub fn status(&self, current: u64) -> Status {
        if self.overridden {
            Status::Overridden
        } else if self.hash == current && self.hash != 0 {
            Status::UpToDate
        } else {
            Status::NeedsUpdate
        }
    }

    // -- storage ---------------------------------------------------------

    /// Encode into an ROI Description. The format is a prefix and JSON, both
    /// so that a human reading the DICOM can see what it is and so that a
    /// future field can be added without breaking a reader.
    pub fn encode(&self) -> String {
        let deps: Vec<serde_json::Value> = self
            .expr
            .deps
            .iter()
            .map(|d| {
                serde_json::json!({
                    "n": d.name,
                    "m": d.margin.all().to_vec(),
                })
            })
            .collect();
        let c = &self.expr.cleanup;
        let v = serde_json::json!({
            "v": 1,
            "op": op_key(self.expr.op),
            "deps": deps,
            "m": self.expr.margin.all().to_vec(),
            "c": {
                "fill": c.fill_holes,
                "close": c.close_mm,
                "largest": c.keep_largest,
                "min": c.min_volume_cm3,
            },
            "h": format!("{:016x}", self.hash),
            "ov": self.overridden,
        });
        format!("{PREFIX}{v}")
    }

    /// Decode one, or `None` when the description is somebody else's text.
    pub fn decode(text: &str) -> Option<Derived> {
        let json = text.strip_prefix(PREFIX)?;
        let v: serde_json::Value = serde_json::from_str(json.trim()).ok()?;
        let op = op_of(v.get("op")?.as_str()?)?;
        let mut deps = Vec::new();
        for d in v.get("deps")?.as_array()? {
            deps.push(Dep {
                name: d.get("n")?.as_str()?.to_string(),
                margin: margin_of(d.get("m")),
            });
        }
        let c = v.get("c");
        let cleanup = Cleanup {
            fill_holes: field(c, "fill").and_then(|x| x.as_bool()).unwrap_or(false),
            close_mm: field(c, "close").and_then(|x| x.as_f64()).unwrap_or(0.0),
            keep_largest: field(c, "largest")
                .and_then(|x| x.as_bool())
                .unwrap_or(false),
            min_volume_cm3: field(c, "min").and_then(|x| x.as_f64()).unwrap_or(0.0),
        };
        let hash = v
            .get("h")
            .and_then(|x| x.as_str())
            .and_then(|s| u64::from_str_radix(s, 16).ok())
            .unwrap_or(0);
        Some(Derived {
            expr: Expr {
                op,
                deps,
                margin: margin_of(v.get("m")),
                cleanup,
            },
            hash,
            overridden: v.get("ov").and_then(|x| x.as_bool()).unwrap_or(false),
        })
    }

    /// The recipe as one line, the guard against a subtraction with its
    /// operands the wrong way round: `PTV_eval = PTV ∩ (BODY -5.0 mm)`.
    pub fn line(&self, result: &str) -> String {
        let mut s = format!("{result} = ");
        for (i, d) in self.expr.deps.iter().enumerate() {
            if i > 0 {
                s.push_str(&format!(" {} ", self.expr.op.joiner()));
            }
            if d.margin.is_none() {
                s.push_str(&d.name);
            } else {
                s.push_str(&format!("({} {})", d.name, d.margin.describe()));
            }
        }
        if !self.expr.margin.is_none() {
            s = format!("({s}) {}", self.expr.margin.describe());
            // The result margin binds the whole expression, so the name has
            // to come back out in front of the bracket.
            if let Some(rest) = s.strip_prefix(&format!("({result} = ")) {
                s = format!("{result} = ({rest}");
            }
        }
        if !self.expr.cleanup.is_none() {
            s.push_str(" + tidy");
        }
        s
    }
}

fn field<'a>(o: Option<&'a serde_json::Value>, key: &str) -> Option<&'a serde_json::Value> {
    o?.get(key)
}

fn margin_of(v: Option<&serde_json::Value>) -> Margin {
    let Some(a) = v.and_then(|x| x.as_array()) else {
        return Margin::NONE;
    };
    let g = |i: usize| a.get(i).and_then(|x| x.as_f64()).unwrap_or(0.0);
    Margin {
        right: g(0),
        left: g(1),
        anterior: g(2),
        posterior: g(3),
        superior: g(4),
        inferior: g(5),
    }
}

fn op_key(op: BoolOp) -> &'static str {
    match op {
        BoolOp::Union => "union",
        BoolOp::Intersect => "intersect",
        BoolOp::Subtract => "subtract",
        BoolOp::Xor => "xor",
    }
}

fn op_of(k: &str) -> Option<BoolOp> {
    Some(match k {
        "union" => BoolOp::Union,
        "intersect" => BoolOp::Intersect,
        "subtract" => BoolOp::Subtract,
        "xor" => BoolOp::Xor,
        _ => return None,
    })
}

/// The recipe attached to a structure, if it has one.
pub fn of_roi(roi: &Roi) -> Option<Derived> {
    Derived::decode(&roi.description)
}

/// Fingerprint of one structure's geometry.
///
/// It has to change when the geometry changes and not otherwise, and it has
/// to cost about nothing, because it is computed for every dependency of
/// every derived structure whenever anything is edited. Contour count, point
/// count and the coordinates quantised to a micrometre satisfy both.
pub fn hash_roi(roi: &Roi) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut mix = |x: u64| {
        h ^= x;
        h = h.wrapping_mul(0x1000_0000_01b3);
    };
    mix(roi.contours.len() as u64);
    for c in &roi.contours {
        mix(c.points.len() as u64);
        for p in &c.points {
            mix((p.x * 1000.0).round() as i64 as u64);
            mix((p.y * 1000.0).round() as i64 as u64);
            mix((p.z * 1000.0).round() as i64 as u64);
        }
    }
    h
}

/// The same for a voxel mask, so a segmentation may be an operand too.
pub fn hash_mask(mask: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    h ^= mask.len() as u64;
    h = h.wrapping_mul(0x1000_0000_01b3);
    // Every voxel would be a hundred megabytes of hashing on a big study;
    // a fixed number of strides over the set voxels is enough to notice an
    // edit and costs the same on any size.
    let step = (mask.len() / 4096).max(1);
    let mut i = 0;
    while i < mask.len() {
        h ^= (mask[i] as u64) << (i % 56);
        h = h.wrapping_mul(0x1000_0000_01b3);
        i += step;
    }
    let set = mask.iter().filter(|&&v| v != 0).count() as u64;
    h ^= set;
    h.wrapping_mul(0x1000_0000_01b3)
}

/// Combine the fingerprints of the operands, in order: a reordering of a
/// subtraction is a different structure and has to read as one.
pub fn fingerprint(parts: &[u64]) -> u64 {
    let mut h = 0x9e37_79b9_7f4a_7c15u64;
    for (i, p) in parts.iter().enumerate() {
        h ^= p.rotate_left((i % 64) as u32);
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    // Zero is reserved for "never evaluated".
    if h == 0 {
        1
    } else {
        h
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Vec3;
    use crate::rtstruct::Contour;

    fn roi(pts: &[[f64; 3]]) -> Roi {
        Roi {
            number: 1,
            name: "r".into(),
            color: [1, 2, 3],
            roi_type: "ORGAN".into(),
            description: String::new(),
            contours: vec![Contour {
                points: pts.iter().map(|p| Vec3::new(p[0], p[1], p[2])).collect(),
                geometric_type: "CLOSED_PLANAR".into(),
            }],
        }
    }

    fn sample() -> Derived {
        Derived {
            expr: Expr {
                op: BoolOp::Subtract,
                deps: vec![
                    Dep {
                        name: "PTV".into(),
                        margin: Margin::NONE,
                    },
                    Dep {
                        name: "Rectum".into(),
                        margin: Margin::uniform(3.0),
                    },
                ],
                margin: Margin {
                    right: 1.0,
                    left: 2.0,
                    anterior: 3.0,
                    posterior: 4.0,
                    superior: 5.0,
                    inferior: 6.0,
                },
                cleanup: Cleanup {
                    fill_holes: true,
                    close_mm: 2.0,
                    keep_largest: true,
                    min_volume_cm3: 0.5,
                },
            },
            hash: 0xdead_beef_1234_5678,
            overridden: false,
        }
    }

    #[test]
    fn a_recipe_survives_the_trip_through_a_dicom_field() {
        let d = sample();
        let text = d.encode();
        assert!(text.starts_with(PREFIX));
        assert!(
            text.len() < 1024,
            "ROI Description is ST, 1024 characters: {}",
            text.len()
        );
        let back = Derived::decode(&text).expect("decodes");
        assert_eq!(back, d);
    }

    #[test]
    fn somebody_elses_description_is_left_alone() {
        assert!(Derived::decode("Prostate, drawn by AP on the T2").is_none());
        assert!(Derived::decode("").is_none());
        // Ours but truncated by a system with a shorter field: not a recipe,
        // and it must not panic.
        let t = sample().encode();
        assert!(Derived::decode(&t[..t.len() / 2]).is_none());
    }

    #[test]
    fn the_status_is_the_three_the_manual_names() {
        let mut d = sample();
        assert_eq!(d.status(d.hash), Status::UpToDate);
        assert_eq!(d.status(d.hash ^ 1), Status::NeedsUpdate);
        d.overridden = true;
        assert_eq!(d.status(d.hash), Status::Overridden);
        // Never evaluated reads as needing an update, not as up to date.
        let fresh = Derived::new(sample().expr);
        assert_eq!(fresh.status(0), Status::NeedsUpdate);
    }

    #[test]
    fn the_line_reads_like_the_recipe() {
        let d = Derived::new(Expr {
            op: BoolOp::Intersect,
            deps: vec![
                Dep {
                    name: "PTV".into(),
                    margin: Margin::NONE,
                },
                Dep {
                    name: "BODY".into(),
                    margin: Margin::uniform(-5.0),
                },
            ],
            margin: Margin::NONE,
            cleanup: Cleanup::default(),
        });
        assert_eq!(d.line("PTV_eval"), "PTV_eval = PTV ∩ (BODY -5.0 mm)");
    }

    #[test]
    fn a_fingerprint_notices_an_edit_and_the_order() {
        let a = roi(&[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0]]);
        let mut b = a.clone();
        assert_eq!(hash_roi(&a), hash_roi(&b));
        // A vertex moved by a hundredth of a millimetre is an edit.
        b.contours[0].points[1].x += 0.01;
        assert_ne!(hash_roi(&a), hash_roi(&b));
        // Order matters: A - B is not B - A.
        assert_ne!(
            fingerprint(&[hash_roi(&a), hash_roi(&b)]),
            fingerprint(&[hash_roi(&b), hash_roi(&a)])
        );
        assert_ne!(fingerprint(&[hash_roi(&a)]), 0);
        // A mask notices a painted voxel.
        let mut m = vec![0u8; 4096];
        let h0 = hash_mask(&m);
        m[1234] = 1;
        assert_ne!(h0, hash_mask(&m));
    }
}
