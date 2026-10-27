//! The X-Ray Radiation Dose SR of a CT scanner (TID 10011, *CT Radiation
//! Dose*), read into the table a person wants from it: one row per
//! acquisition with its protocol, technique and dose indices, and the
//! study's totals above.
//!
//! The template is walked by concept code (DCM codes of PS3.16), never by
//! the meaning strings, which scanners translate. A code the report does
//! not carry leaves its cell empty; a report with no acquisition container
//! at all still yields its totals. Everything the template holds beyond
//! this table (dose check alerts, device identification, the irradiation
//! event UIDs) stays in the SR tree the generic view lays out.

use super::sr::Node;
use super::Table;

/// The root concept of the dose report.
pub const CODE_DOSE_REPORT: &str = "113701";
const CODE_PROCEDURE: &str = "121058";
const CODE_START: &str = "113809";
const CODE_END: &str = "113810";
const CODE_DEVICE_NAME: &str = "121013";
const CODE_DEVICE_MODEL: &str = "121015";
const CODE_DEVICE_MAKER: &str = "121014";
const CODE_ACCUMULATED: &str = "113811";
const CODE_EVENTS: &str = "113812";
const CODE_DLP_TOTAL: &str = "113813";
const CODE_ACQUISITION: &str = "113819";
const CODE_PROTOCOL: &str = "125203";
const CODE_REGION: &str = "123014";
const CODE_ACQ_TYPE: &str = "113820";
const CODE_EXPOSURE_TIME: &str = "113824";
const CODE_SCAN_LENGTH: &str = "113825";
const CODE_COLLIMATION_TOTAL: &str = "113827";
const CODE_PITCH: &str = "113828";
const CODE_KVP: &str = "113733";
const CODE_MA_MAX: &str = "113833";
const CODE_MA_MEAN: &str = "113734";
const CODE_ROTATION_TIME: &str = "113834";
const CODE_CTDIVOL: &str = "113830";
const CODE_PHANTOM: &str = "113835";
const CODE_DLP: &str = "113838";
const CODE_MODULATION: &str = "113842";

/// Is this SR a CT dose report?
pub fn is_dose_report(root: &Node) -> bool {
    root.code == CODE_DOSE_REPORT
}

/// The facts above the table: what was done, when, on which scanner, and
/// the totals.
pub fn facts(root: &Node) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut add = |k: &str, v: String| {
        if !v.trim().is_empty() {
            out.push((k.to_string(), v.trim().to_string()));
        }
    };
    add("Procedure", root.text_of(CODE_PROCEDURE, true));
    let device = [CODE_DEVICE_MAKER, CODE_DEVICE_MODEL]
        .iter()
        .map(|c| root.text_of(c, true))
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    add("Scanner", device);
    add("Station", root.text_of(CODE_DEVICE_NAME, true));
    add("Irradiation from", root.text_of(CODE_START, true));
    add("Irradiation to", root.text_of(CODE_END, true));
    if let Some(acc) = root.find(CODE_ACCUMULATED) {
        add("Irradiation events", acc.num_text(CODE_EVENTS));
        let dlp = acc.num_text(CODE_DLP_TOTAL);
        if !dlp.is_empty() {
            add(
                "Total DLP",
                format!("{dlp} {}", acc.unit_of(CODE_DLP_TOTAL)),
            );
        }
        // Whatever else the scanner accumulated (an effective dose, say).
        for n in &acc.children {
            if n.value_type == "NUM" && n.code != CODE_EVENTS && n.code != CODE_DLP_TOTAL {
                add(&n.name, n.value.text(true));
            }
        }
    }
    out
}

/// One row per CT acquisition, in the order of the report.
pub fn acquisitions(root: &Node) -> Table {
    let acqs: Vec<&Node> = root.children_coded(CODE_ACQUISITION).collect();
    // The unit of a column is the first acquisition's that states it.
    let unit = |code: &str| {
        acqs.iter()
            .map(|a| a.unit_of(code))
            .find(|u| !u.is_empty())
            .unwrap_or_default()
    };
    let with_unit = |name: &str, code: &str| {
        let u = unit(code);
        if u.is_empty() {
            name.to_string()
        } else {
            format!("{name} {u}")
        }
    };
    let columns = vec![
        "#".to_string(),
        "Protocol".to_string(),
        "Type".to_string(),
        "Region".to_string(),
        "kV".to_string(),
        "mA mean / max".to_string(),
        with_unit("Rotation", CODE_ROTATION_TIME),
        "Pitch".to_string(),
        with_unit("Collimation", CODE_COLLIMATION_TOTAL),
        with_unit("Length", CODE_SCAN_LENGTH),
        with_unit("Time", CODE_EXPOSURE_TIME),
        with_unit("CTDIvol", CODE_CTDIVOL),
        with_unit("DLP", CODE_DLP),
        "Phantom".to_string(),
        "Modulation".to_string(),
    ];
    let rows = acqs
        .iter()
        .enumerate()
        .map(|(i, a)| {
            let ma = match (a.num_text(CODE_MA_MEAN), a.num_text(CODE_MA_MAX)) {
                (m, x) if m.is_empty() => x,
                (m, x) if x.is_empty() || x == m => m,
                (m, x) => format!("{m} / {x}"),
            };
            vec![
                (i + 1).to_string(),
                a.text_of(CODE_PROTOCOL, true),
                short_type(&a.text_of(CODE_ACQ_TYPE, true)),
                a.text_of(CODE_REGION, true),
                a.num_text(CODE_KVP),
                ma,
                a.num_text(CODE_ROTATION_TIME),
                a.num_text(CODE_PITCH),
                a.num_text(CODE_COLLIMATION_TOTAL),
                a.num_text(CODE_SCAN_LENGTH),
                a.num_text(CODE_EXPOSURE_TIME),
                a.num_text(CODE_CTDIVOL),
                a.num_text(CODE_DLP),
                short_phantom(&a.text_of(CODE_PHANTOM, true)),
                a.text_of(CODE_MODULATION, true),
            ]
        })
        .collect();
    Table {
        title: "CT acquisitions".to_string(),
        columns,
        rows,
    }
}

/// `Spiral Acquisition` as `Spiral`, `Constant Angle Acquisition` as
/// `Constant angle` (a topogram).
fn short_type(t: &str) -> String {
    let t = t.trim();
    let t = t.strip_suffix(" Acquisition").unwrap_or(t);
    let mut s = t.to_string();
    if let Some(rest) = s.get(1..) {
        s = format!("{}{}", &s[..1], rest.to_lowercase());
    }
    s
}

/// `IEC Body Dosimetry Phantom` as `Body (32 cm)`, `IEC Head Dosimetry
/// Phantom` as `Head (16 cm)`.
fn short_phantom(t: &str) -> String {
    let l = t.to_ascii_lowercase();
    if l.contains("body") {
        "Body (32 cm)".to_string()
    } else if l.contains("head") {
        "Head (16 cm)".to_string()
    } else {
        t.trim().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::super::sr::Value;
    use super::*;

    fn num(code: &str, name: &str, text: &str, unit: &str) -> Node {
        Node {
            relationship: "CONTAINS".into(),
            value_type: "NUM".into(),
            code: code.into(),
            scheme: "DCM".into(),
            name: name.into(),
            value: Value::Num {
                text: text.into(),
                value: text.parse().ok(),
                unit: unit.into(),
            },
            children: Vec::new(),
        }
    }

    fn text(code: &str, name: &str, t: &str) -> Node {
        Node {
            value_type: "TEXT".into(),
            code: code.into(),
            name: name.into(),
            value: Value::Text(t.into()),
            ..Default::default()
        }
    }

    fn coded(code: &str, name: &str, meaning: &str) -> Node {
        Node {
            value_type: "CODE".into(),
            code: code.into(),
            name: name.into(),
            value: Value::Code {
                meaning: meaning.into(),
                code: String::new(),
            },
            ..Default::default()
        }
    }

    fn container(code: &str, name: &str, children: Vec<Node>) -> Node {
        Node {
            value_type: "CONTAINER".into(),
            code: code.into(),
            name: name.into(),
            children,
            ..Default::default()
        }
    }

    /// The shape of a Siemens report: totals, then one container per
    /// acquisition with the parameters, the source and the dose nested.
    fn report() -> Node {
        let acq = |protocol: &str, kind: &str, kv: &str, ma: &str, ctdi: &str, dlp: &str| {
            container(
                CODE_ACQUISITION,
                "CT Acquisition",
                vec![
                    text(CODE_PROTOCOL, "Acquisition Protocol", protocol),
                    coded(CODE_REGION, "Target Region", "Abdomen"),
                    coded(CODE_ACQ_TYPE, "CT Acquisition Type", kind),
                    container(
                        "113822",
                        "CT Acquisition Parameters",
                        vec![
                            num(CODE_EXPOSURE_TIME, "Exposure Time", "10.64", "s"),
                            num(CODE_SCAN_LENGTH, "Scanning Length", "425.5", "mm"),
                            num(
                                CODE_COLLIMATION_TOTAL,
                                "Nominal Total Collimation Width",
                                "19.2",
                                "mm",
                            ),
                            num(CODE_PITCH, "Pitch Factor", "1.25", "ratio"),
                            container(
                                "113831",
                                "CT X-Ray Source Parameters",
                                vec![
                                    num(CODE_KVP, "KVP", kv, "kV"),
                                    num(CODE_MA_MAX, "Maximum X-Ray Tube Current", "181", "mA"),
                                    num(CODE_MA_MEAN, "Mean X-Ray Tube Current", ma, "mA"),
                                    num(
                                        CODE_ROTATION_TIME,
                                        "Exposure Time per Rotation",
                                        "0.6",
                                        "s",
                                    ),
                                ],
                            ),
                        ],
                    ),
                    container(
                        "113829",
                        "CT Dose",
                        vec![
                            num(CODE_CTDIVOL, "Mean CTDIvol", ctdi, "mGy"),
                            coded(
                                CODE_PHANTOM,
                                "CTDIw Phantom Type",
                                "IEC Body Dosimetry Phantom",
                            ),
                            num(CODE_DLP, "DLP", dlp, "mGy.cm"),
                        ],
                    ),
                    text(CODE_MODULATION, "X-Ray Modulation Type", "XYZ_EC"),
                ],
            )
        };
        Node {
            value_type: "CONTAINER".into(),
            code: CODE_DOSE_REPORT.into(),
            name: "X-Ray Radiation Dose Report".into(),
            children: vec![
                coded(
                    CODE_PROCEDURE,
                    "Procedure reported",
                    "Computed Tomography X-ray",
                ),
                text(CODE_DEVICE_MAKER, "Device Observer Manufacturer", "SIEMENS"),
                text(
                    CODE_DEVICE_MODEL,
                    "Device Observer Model Name",
                    "SOMATOM Emotion 16",
                ),
                container(
                    CODE_ACCUMULATED,
                    "CT Accumulated Dose Data",
                    vec![
                        num(
                            CODE_EVENTS,
                            "Total Number of Irradiation Events",
                            "2",
                            "events",
                        ),
                        num(
                            CODE_DLP_TOTAL,
                            "CT Dose Length Product Total",
                            "508.44",
                            "mGy.cm",
                        ),
                    ],
                ),
                acq(
                    "Steinsuche",
                    "Spiral Acquisition",
                    "110",
                    "81",
                    "2.8",
                    "119.11",
                ),
                acq(
                    "AbdRoutine",
                    "Spiral Acquisition",
                    "130",
                    "128",
                    "8.18",
                    "389.33",
                ),
            ],
            ..Default::default()
        }
    }

    #[test]
    fn the_acquisitions_become_one_row_each_with_the_units_in_the_header() {
        let r = report();
        assert!(is_dose_report(&r));
        let t = acquisitions(&r);
        assert_eq!(t.rows.len(), 2);
        assert_eq!(t.columns[4], "kV");
        assert_eq!(t.columns[6], "Rotation s");
        assert_eq!(t.columns[11], "CTDIvol mGy");
        assert_eq!(t.columns[12], "DLP mGy.cm");
        let row = &t.rows[1];
        assert_eq!(row[1], "AbdRoutine");
        assert_eq!(row[2], "Spiral");
        assert_eq!(row[3], "Abdomen");
        assert_eq!(row[4], "130");
        assert_eq!(row[5], "128 / 181");
        assert_eq!(row[7], "1.25");
        assert_eq!(row[11], "8.18");
        assert_eq!(row[12], "389.33");
        assert_eq!(row[13], "Body (32 cm)");
        assert_eq!(row[14], "XYZ_EC");
    }

    #[test]
    fn the_facts_carry_the_scanner_and_the_totals() {
        let f = facts(&report());
        let get = |k: &str| {
            f.iter()
                .find(|(key, _)| key == k)
                .map(|(_, v)| v.clone())
                .unwrap_or_default()
        };
        assert_eq!(get("Procedure"), "Computed Tomography X-ray");
        assert_eq!(get("Scanner"), "SIEMENS SOMATOM Emotion 16");
        assert_eq!(get("Irradiation events"), "2");
        assert_eq!(get("Total DLP"), "508.44 mGy.cm");
        assert!(get("Irradiation from").is_empty(), "not stated, not listed");
    }
}
