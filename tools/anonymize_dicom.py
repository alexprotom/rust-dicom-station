#!/usr/bin/env python3
"""
Anonymize the 4DCT example dataset (2 CT series + 1 RTSTRUCT each) in place.

Pure standard library: parses/writes Implicit VR Little Endian DICOM directly,
so it runs anywhere python3 is available (no pydicom needed).

Result:
    PatientName / PatientID       -> lung_p1
    StudyInstanceUID              -> 1.2.3.4.5.1        (shared by both series)
    FrameOfReferenceUID           -> 1.2.3.4.5.2        (shared by both series)
    phase_000 CT series           -> 1.2.3.4.5.10       instances .10.<InstanceNumber>
    phase_000 RTSTRUCT series     -> 1.2.3.4.5.11       instance  .11.1
    phase_050 CT series           -> 1.2.3.4.5.20       instances .20.<InstanceNumber>
    phase_050 RTSTRUCT series     -> 1.2.3.4.5.21       instance  .21.1
Everything not needed to render the images / contours is removed.
"""

import os
import struct
import sys

ROOT = "1.2.3.4.5"
STUDY_UID = ROOT + ".1"
FOR_UID = ROOT + ".2"
IMPL_CLASS_UID = ROOT + ".0"
DATE = "20000101"
TIME = "000000"

SERIES = [
    # folder,                  ct_series_uid, rt_series_uid, description,      ct_series_no, rt_series_no
    ("lung_p1_4DCT_phase_000", ROOT + ".10", ROOT + ".11", "4DCT_phase_000", "1", "11"),
    ("lung_p1_4DCT_phase_050", ROOT + ".20", ROOT + ".21", "4DCT_phase_050", "2", "12"),
]

CT_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.2"
RT_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.481.3"
STUDY_COMPONENT_SOP_CLASS = "1.2.840.10008.3.1.2.3.1"

ITEM = (0xFFFE, 0xE000)
ITEM_DELIM = (0xFFFE, 0xE00D)
SEQ_DELIM = (0xFFFE, 0xE0DD)
UNDEFINED = 0xFFFFFFFF

KNOWN_SQ = {
    (0x0012, 0x0064), (0x3006, 0x0010), (0x3006, 0x0012), (0x3006, 0x0014),
    (0x3006, 0x0016), (0x3006, 0x0020), (0x3006, 0x0039), (0x3006, 0x0040),
    (0x3006, 0x0080), (0x3006, 0x0086), (0x0008, 0x1140), (0x0008, 0x1110),
    (0x0008, 0x1115), (0x0008, 0x1120), (0x0040, 0xA730),
}
PIXEL_DATA = (0x7FE0, 0x0010)

# ---------------------------------------------------------------- parsing ---


class Elem(object):
    __slots__ = ("tag", "value", "items")

    def __init__(self, tag, value=None, items=None):
        self.tag = tag          # (group, element)
        self.value = value      # bytes, for non-sequence elements
        self.items = items      # list[list[Elem]], for sequences

    @property
    def is_sq(self):
        return self.items is not None


def _looks_like_sq(tag, data):
    if tag == PIXEL_DATA:
        return False
    if tag in KNOWN_SQ:
        return True
    if len(data) >= 8:
        g, e = struct.unpack("<HH", data[:4])
        if (g, e) == ITEM:
            return True
    return False


def parse_dataset(buf, start, end):
    """Parse Implicit VR Little Endian elements in buf[start:end]."""
    elems = []
    pos = start
    while pos + 8 <= end:
        g, e, length = struct.unpack("<HHI", buf[pos:pos + 8])
        pos += 8
        tag = (g, e)
        if tag in (ITEM_DELIM, SEQ_DELIM):
            break
        if length == UNDEFINED:
            items, pos = parse_items(buf, pos, end)
            elems.append(Elem(tag, items=items))
            continue
        data = buf[pos:pos + length]
        pos += length
        if _looks_like_sq(tag, data):
            items, _ = parse_items(data, 0, len(data))
            elems.append(Elem(tag, items=items))
        else:
            elems.append(Elem(tag, value=data))
    return elems


def parse_items(buf, pos, end):
    """Parse a run of sequence items starting at pos; returns (items, new_pos)."""
    items = []
    while pos + 8 <= end:
        g, e, length = struct.unpack("<HHI", buf[pos:pos + 8])
        pos += 8
        tag = (g, e)
        if tag == SEQ_DELIM:
            break
        if tag != ITEM:
            raise ValueError("expected item tag, got (%04X,%04X)" % tag)
        if length == UNDEFINED:
            item_start = pos
            depth_pos = pos
            # find matching item delimiter at this level
            sub = parse_dataset(buf, depth_pos, end)
            # re-scan to find where the item ended
            pos = _skip_dataset(buf, item_start, end)
            items.append(sub)
        else:
            items.append(parse_dataset(buf, pos, pos + length))
            pos += length
    return items, pos


def _skip_dataset(buf, pos, end):
    """Advance past an undefined-length item's content, returning pos after the delimiter."""
    while pos + 8 <= end:
        g, e, length = struct.unpack("<HHI", buf[pos:pos + 8])
        pos += 8
        if (g, e) == ITEM_DELIM:
            return pos
        if length == UNDEFINED:
            _, pos = parse_items(buf, pos, end)
        else:
            pos += length
    return pos


# ---------------------------------------------------------------- writing ---


def pad(value, tag):
    """Encode a text value, padded to even length."""
    b = value.encode("ascii")
    if len(b) % 2:
        b += b"\x00" if tag == PIXEL_DATA else b" "
    return b


def pad_uid(value):
    b = value.encode("ascii")
    if len(b) % 2:
        b += b"\x00"
    return b


def encode_dataset(elems):
    out = []
    for el in sorted(elems, key=lambda x: x.tag):
        if el.is_sq:
            body = b"".join(
                struct.pack("<HHI", ITEM[0], ITEM[1], len(d)) + d
                for d in (encode_dataset(it) for it in el.items)
            )
        else:
            body = el.value
            if len(body) % 2:
                body += b"\x00"
        out.append(struct.pack("<HHI", el.tag[0], el.tag[1], len(body)) + body)
    return b"".join(out)


def encode_file_meta(sop_class, sop_instance):
    """Explicit VR Little Endian file meta group."""
    def ui(tag, value):
        b = pad_uid(value)
        return struct.pack("<HH2sH", tag[0], tag[1], b"UI", len(b)) + b

    # (0002,0001) OB carries 2 reserved bytes and a 4-byte length
    body = struct.pack("<HH2sHI", 0x0002, 0x0001, b"OB", 0, 2) + b"\x00\x01"
    body += ui((0x0002, 0x0002), sop_class)
    body += ui((0x0002, 0x0003), sop_instance)
    body += ui((0x0002, 0x0010), "1.2.840.10008.1.2")  # Implicit VR Little Endian
    body += ui((0x0002, 0x0012), IMPL_CLASS_UID)
    glen = struct.pack("<HH2sH", 0x0002, 0x0000, b"UL", 4) + struct.pack("<I", len(body))
    return b"\x00" * 128 + b"DICM" + glen + body


def read_dicom(path):
    with open(path, "rb") as fh:
        buf = fh.read()
    if buf[128:132] != b"DICM":
        raise ValueError("not a Part-10 DICOM file: " + path)
    pos = 132
    meta = {}
    while pos + 8 <= len(buf):
        g, e = struct.unpack("<HH", buf[pos:pos + 4])
        if g != 0x0002:
            break
        vr = buf[pos + 4:pos + 6]
        if vr in (b"OB", b"OW", b"OF", b"SQ", b"UT", b"UN"):
            length = struct.unpack("<I", buf[pos + 8:pos + 12])[0]
            vpos = pos + 12
        else:
            length = struct.unpack("<H", buf[pos + 6:pos + 8])[0]
            vpos = pos + 8
        meta[(g, e)] = buf[vpos:vpos + length]
        pos = vpos + length
    ts = meta.get((0x0002, 0x0010), b"").rstrip(b"\x00 ").decode("ascii")
    if ts != "1.2.840.10008.1.2":
        raise ValueError("expected Implicit VR Little Endian, got %r in %s" % (ts, path))
    return meta, parse_dataset(buf, pos, len(buf))


def get(elems, tag, default=""):
    for el in elems:
        if el.tag == tag and not el.is_sq:
            return el.value.rstrip(b"\x00 ").decode("ascii", "replace")
    return default


# ------------------------------------------------------------ anonymizing ---

CT_KEEP = {
    (0x0008, 0x0005), (0x0008, 0x0008), (0x0008, 0x0016), (0x0008, 0x0060),
    (0x0018, 0x0015), (0x0018, 0x0050), (0x0018, 0x5100),
    (0x0020, 0x0013), (0x0020, 0x0032), (0x0020, 0x0037), (0x0020, 0x1041),
    (0x0028, 0x0002), (0x0028, 0x0004), (0x0028, 0x0010), (0x0028, 0x0011),
    (0x0028, 0x0030), (0x0028, 0x0100), (0x0028, 0x0101), (0x0028, 0x0102),
    (0x0028, 0x0103), (0x0028, 0x1052), (0x0028, 0x1053),
    PIXEL_DATA,
}

RT_KEEP = {
    (0x0008, 0x0005), (0x0008, 0x0016), (0x0008, 0x0060), (0x0018, 0x0015),
    (0x3006, 0x0010), (0x3006, 0x0020), (0x3006, 0x0039), (0x3006, 0x0080),
    (0x300E, 0x0002),
}


def remap_uids(elems, uid_map):
    """Recursively replace any UID value found in uid_map (used inside sequences)."""
    for el in elems:
        if el.is_sq:
            for item in el.items:
                remap_uids(item, uid_map)
        else:
            key = el.value.rstrip(b"\x00 ").decode("ascii", "replace")
            if key in uid_map:
                el.value = pad_uid(uid_map[key])
            elif el.tag == (0x0008, 0x1150) and key not in (CT_SOP_CLASS, RT_SOP_CLASS):
                # RT Referenced Study points at a vendor-specific SOP class -> standard one
                el.value = pad_uid(STUDY_COMPONENT_SOP_CLASS)


def build(tag, value):
    return Elem(tag, pad(value, tag))


def anonymize_ct(elems, uid_map, series_uid, series_no, description, new_sop):
    kept = [el for el in elems if el.tag in CT_KEEP]
    remap_uids(kept, uid_map)
    kept += [
        build((0x0008, 0x0018), new_sop),
        build((0x0008, 0x0020), DATE),
        build((0x0008, 0x0030), TIME),
        build((0x0008, 0x0023), DATE),
        build((0x0008, 0x0033), TIME),
        build((0x0008, 0x1030), "4DCT"),
        build((0x0008, 0x103E), description),
        build((0x0010, 0x0010), "lung_p1"),
        build((0x0010, 0x0020), "lung_p1"),
        build((0x0010, 0x0030), ""),
        build((0x0010, 0x0040), ""),
        build((0x0012, 0x0062), "YES"),
        build((0x0012, 0x0063), "BASIC"),
        build((0x0020, 0x000D), STUDY_UID),
        build((0x0020, 0x000E), series_uid),
        build((0x0020, 0x0010), "1"),
        build((0x0020, 0x0011), series_no),
        build((0x0020, 0x0052), FOR_UID),
    ]
    return kept


def anonymize_rt(elems, uid_map, series_uid, series_no, description, new_sop, label):
    kept = [el for el in elems if el.tag in RT_KEEP]
    remap_uids(kept, uid_map)
    kept += [
        build((0x0008, 0x0018), new_sop),
        build((0x0008, 0x0020), DATE),
        build((0x0008, 0x0030), TIME),
        build((0x0008, 0x0023), DATE),
        build((0x0008, 0x1030), "4DCT"),
        build((0x0008, 0x103E), description),
        build((0x0010, 0x0010), "lung_p1"),
        build((0x0010, 0x0020), "lung_p1"),
        build((0x0010, 0x0030), ""),
        build((0x0010, 0x0040), ""),
        build((0x0012, 0x0062), "YES"),
        build((0x0012, 0x0063), "BASIC"),
        build((0x0020, 0x000D), STUDY_UID),
        build((0x0020, 0x000E), series_uid),
        build((0x0020, 0x0010), "1"),
        build((0x0020, 0x0011), series_no),
        build((0x3006, 0x0002), label),
        build((0x3006, 0x0004), "ROIs"),
        build((0x3006, 0x0008), DATE),
        build((0x3006, 0x0009), TIME),
    ]
    return kept


# ------------------------------------------------------------------- main ---


def main(base):
    plan = []       # (path, kind, folder_index)
    uid_map = {}
    parsed = {}

    for idx, (folder, ct_uid, rt_uid, desc, ct_no, rt_no) in enumerate(SERIES):
        d = os.path.join(base, folder)
        if not os.path.isdir(d):
            sys.exit("missing folder: " + d)
        for name in sorted(os.listdir(d)):
            if not name.lower().endswith(".dcm"):
                continue
            path = os.path.join(d, name)
            meta, elems = read_dicom(path)
            sop_class = get(elems, (0x0008, 0x0016))
            sop_inst = get(elems, (0x0008, 0x0018))
            parsed[path] = (meta, elems)
            if sop_class == RT_SOP_CLASS:
                new_sop = rt_uid + ".1"
                plan.append((path, "RT", idx))
            elif sop_class == CT_SOP_CLASS:
                inst_no = get(elems, (0x0020, 0x0013)).strip() or "0"
                new_sop = "%s.%d" % (ct_uid, int(inst_no))
                plan.append((path, "CT", idx))
            else:
                sys.exit("unexpected SOP class %s in %s" % (sop_class, path))
            uid_map[sop_inst] = new_sop
            uid_map[get(elems, (0x0020, 0x000D))] = STUDY_UID
            uid_map[get(elems, (0x0020, 0x000E))] = ct_uid if sop_class == CT_SOP_CLASS else rt_uid
            for_uid = get(elems, (0x0020, 0x0052))
            if for_uid:
                uid_map[for_uid] = FOR_UID

    # the RTSTRUCT carries the frame of reference only inside sequences
    uid_map.pop("", None)

    counts = {}
    for path, kind, idx in plan:
        folder, ct_uid, rt_uid, desc, ct_no, rt_no = SERIES[idx]
        meta, elems = parsed[path]
        sop_inst = get(elems, (0x0008, 0x0018))
        new_sop = uid_map[sop_inst]
        if kind == "CT":
            out = anonymize_ct(elems, uid_map, ct_uid, ct_no, desc, new_sop)
            sop_class = CT_SOP_CLASS
        else:
            out = anonymize_rt(elems, uid_map, rt_uid, rt_no, desc, new_sop,
                               desc.replace("4DCT_", ""))
            sop_class = RT_SOP_CLASS
        blob = encode_file_meta(sop_class, new_sop) + encode_dataset(out)
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            fh.write(blob)
        os.replace(tmp, path)
        counts[kind] = counts.get(kind, 0) + 1

    print("rewrote %d CT slices and %d RTSTRUCT files" % (counts.get("CT", 0), counts.get("RT", 0)))
    print("unique original UIDs remapped: %d" % len(uid_map))
    verify(base, plan)


STALE = (b"1.3.6.1.4.1.14519", b"2.25.1", b"1.2.528.", b"P102", b"HM10395",
         b"ADAC", b"Pinnacle", b"dcm4che", b"4D-Lung", b"2819497684894126",
         b"19980325", b"Warped", b"POIandROIandBOLUS", b"Plan_0")


def collect_uids(elems, out):
    for el in elems:
        if el.is_sq:
            for item in el.items:
                collect_uids(item, out)
        elif el.tag in ((0x0008, 0x1155), (0x0008, 0x0018)):
            out.add(el.value.rstrip(b"\x00 ").decode("ascii", "replace"))


def verify(base, plan):
    problems = []
    ct_sops = {}
    rt_files = []
    for path, kind, idx in plan:
        with open(path, "rb") as fh:
            blob = fh.read()
        cut = blob.find(b"\xe0\x7f\x10\x00")   # everything before PixelData
        head = blob[:cut] if cut > 0 else blob
        for needle in STALE:
            if needle in head:
                problems.append("%s still contains %r" % (path, needle))
        meta, elems = read_dicom(path)
        if kind == "CT":
            ct_sops.setdefault(idx, set()).add(get(elems, (0x0008, 0x0018)))
        else:
            rt_files.append((path, idx, elems))

    for path, idx, elems in rt_files:
        refs = set()
        collect_uids(elems, refs)
        refs.discard(get(elems, (0x0008, 0x0018)))
        refs.discard(STUDY_UID)
        refs.discard("")
        missing = refs - ct_sops.get(idx, set())
        if missing:
            problems.append("%s references %d SOP UIDs with no matching CT slice: %s"
                            % (path, len(missing), sorted(missing)[:3]))
        else:
            print("%s: all %d image references resolve to CT slices in its own series"
                  % (os.path.basename(os.path.dirname(path)), len(refs)))

    for idx, sops in sorted(ct_sops.items()):
        print("%s: %d CT slices, %d unique SOP UIDs" % (SERIES[idx][0], len(sops), len(sops)))

    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  " + p)
        sys.exit(1)
    print("verification OK")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
