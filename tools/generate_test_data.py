#!/usr/bin/env python3
"""Generate a synthetic DICOM RT study for testing rust-dicom-viewer.

Creates in test_data/:
  * CT series: 40 slices, 96x96 px, 2 mm isotropic. Water cylinder phantom
    (r = 70 mm), spherical target (r = 25 mm, HU 100) at the origin and a
    small "cord" cylinder (r = 8 mm, HU 40) at (0, 60).
  * RTSTRUCT: BODY (EXTERNAL), TARGET (PTV), CORD (ORGAN).
  * RTDOSE: 3D Gaussian, 60 Gy at the isocenter, sigma 20 mm, 32-bit, 4 mm grid.
  * RTPLAN: ion (proton) plan, 2 beams, 60 Gy / 30 fx prescription.

Requires: pydicom >= 2.4
"""

import argparse
import os
import datetime

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--out", default="test_data", help="output directory (relative to repo root)")
ap.add_argument("--target-shift-y", type=float, default=0.0, help="target/dose Y shift in mm")
ap.add_argument("--peak", type=float, default=60.0, help="dose peak in Gy")
ap.add_argument("--plan-label", default="SynthProton")
args = ap.parse_args()

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.out)
os.makedirs(OUT, exist_ok=True)

SHIFT_Y = args.target_shift_y

# --- shared identifiers -----------------------------------------------------
study_uid = generate_uid()
for_uid = generate_uid()
ct_series_uid = generate_uid()
ct_sop_uids = []

NX = NY = 96
NZ = 40
SPACING = 2.0
X0 = -(NX - 1) / 2.0 * SPACING  # -95
Y0 = -(NY - 1) / 2.0 * SPACING
Z0 = -(NZ - 1) / 2.0 * SPACING  # -39

now = datetime.datetime.now()
DATE = now.strftime("%Y%m%d")
TIME = now.strftime("%H%M%S")


def base_dataset(sop_class, sop_uid, modality):
    fm = FileMetaDataset()
    fm.MediaStorageSOPClassUID = sop_class
    fm.MediaStorageSOPInstanceUID = sop_uid
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset("", {}, file_meta=fm, preamble=b"\0" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.SOPClassUID = sop_class
    ds.SOPInstanceUID = sop_uid
    ds.PatientName = "PHANTOM^RT"
    ds.PatientID = "RTTEST001"
    ds.PatientBirthDate = "19700101"
    ds.PatientSex = "O"
    ds.StudyInstanceUID = study_uid
    ds.StudyDate = DATE
    ds.StudyTime = TIME
    ds.StudyDescription = "Synthetic RT study"
    ds.AccessionNumber = "1"
    ds.ReferringPhysicianName = ""
    ds.Modality = modality
    ds.Manufacturer = "rust-dicom-viewer synthetic"
    return ds


# --- CT series --------------------------------------------------------------
xs = X0 + np.arange(NX) * SPACING
ys = Y0 + np.arange(NY) * SPACING
XX, YY = np.meshgrid(xs, ys)  # [row=y, col=x]

for k in range(NZ):
    z = Z0 + k * SPACING
    hu = np.full((NY, NX), -1000.0, dtype=np.float64)
    body = XX**2 + YY**2 <= 70.0**2
    hu[body] = 0.0
    target = XX**2 + (YY - SHIFT_Y) ** 2 + z**2 <= 25.0**2
    hu[target] = 100.0
    cord = XX**2 + (YY - 60.0) ** 2 <= 8.0**2
    hu[cord] = 40.0

    sop_uid = generate_uid()
    ct_sop_uids.append(sop_uid)
    ds = base_dataset("1.2.840.10008.5.1.4.1.1.2", sop_uid, "CT")
    ds.SeriesInstanceUID = ct_series_uid
    ds.SeriesNumber = 1
    ds.SeriesDescription = "Synthetic CT"
    ds.InstanceNumber = k + 1
    ds.FrameOfReferenceUID = for_uid
    ds.PositionReferenceIndicator = ""
    ds.ImagePositionPatient = [f"{X0:.3f}", f"{Y0:.3f}", f"{z:.3f}"]
    ds.ImageOrientationPatient = ["1", "0", "0", "0", "1", "0"]
    ds.PixelSpacing = [f"{SPACING:.3f}", f"{SPACING:.3f}"]
    ds.SliceThickness = f"{SPACING:.3f}"
    ds.KVP = "120"
    ds.Rows = NY
    ds.Columns = NX
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1  # signed
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.RescaleIntercept = "-1024"
    ds.RescaleSlope = "1"
    ds.WindowCenter = "40"
    ds.WindowWidth = "400"
    stored = np.round(hu + 1024.0).astype(np.int16)  # raw = HU - intercept
    ds.PixelData = stored.tobytes()
    ds.save_as(os.path.join(OUT, f"CT_{k:03d}.dcm"), enforce_file_format=True)

print(f"CT: {NZ} slices written")

# --- RTSTRUCT ---------------------------------------------------------------
ss_uid = generate_uid()
ds = base_dataset("1.2.840.10008.5.1.4.1.1.481.3", ss_uid, "RTSTRUCT")
ds.SeriesInstanceUID = generate_uid()
ds.SeriesNumber = 2
ds.StructureSetLabel = "SynthStructs"
ds.StructureSetDate = DATE
ds.StructureSetTime = TIME

ref_frame = Dataset()
ref_frame.FrameOfReferenceUID = for_uid
rt_ref_study = Dataset()
rt_ref_study.ReferencedSOPClassUID = "1.2.840.10008.3.1.2.3.1"
rt_ref_study.ReferencedSOPInstanceUID = study_uid
rt_ref_series = Dataset()
rt_ref_series.SeriesInstanceUID = ct_series_uid
cis = []
for uid in ct_sop_uids:
    ci = Dataset()
    ci.ReferencedSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    ci.ReferencedSOPInstanceUID = uid
    cis.append(ci)
rt_ref_series.ContourImageSequence = cis
rt_ref_study.RTReferencedSeriesSequence = [rt_ref_series]
ref_frame.RTReferencedStudySequence = [rt_ref_study]
ds.ReferencedFrameOfReferenceSequence = [ref_frame]


def circle_points(cx, cy, r, z, n=64):
    ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    pts = []
    for a in ang:
        pts += [cx + r * np.cos(a), cy + r * np.sin(a), z]
    return [f"{v:.3f}" for v in pts]


rois = [
    (1, "BODY", "EXTERNAL", [0, 255, 0]),
    (2, "TARGET", "PTV", [255, 0, 0]),
    (3, "CORD", "ORGAN", [255, 255, 0]),
]

ssr, rcs, obs = [], [], []
for num, name, typ, color in rois:
    s = Dataset()
    s.ROINumber = num
    s.ReferencedFrameOfReferenceUID = for_uid
    s.ROIName = name
    s.ROIGenerationAlgorithm = "AUTOMATIC"
    ssr.append(s)

    rc = Dataset()
    rc.ReferencedROINumber = num
    rc.ROIDisplayColor = color
    contours = []
    for k in range(NZ):
        z = Z0 + k * SPACING
        if name == "BODY":
            r = 70.0
        elif name == "CORD":
            r = 8.0
        else:
            r2 = 25.0**2 - z**2
            if r2 <= 4.0:
                continue
            r = float(np.sqrt(r2))
        c = Dataset()
        c.ContourGeometricType = "CLOSED_PLANAR"
        cy = 60.0 if name == "CORD" else (SHIFT_Y if name == "TARGET" else 0.0)
        pts = circle_points(0.0, cy, r, z)
        c.NumberOfContourPoints = len(pts) // 3
        c.ContourData = pts
        ci = Dataset()
        ci.ReferencedSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        ci.ReferencedSOPInstanceUID = ct_sop_uids[k]
        c.ContourImageSequence = [ci]
        contours.append(c)
    rc.ContourSequence = contours
    rcs.append(rc)

    o = Dataset()
    o.ObservationNumber = num
    o.ReferencedROINumber = num
    o.RTROIInterpretedType = typ
    o.ROIInterpreter = ""
    obs.append(o)

ds.StructureSetROISequence = ssr
ds.ROIContourSequence = rcs
ds.RTROIObservationsSequence = obs
ds.save_as(os.path.join(OUT, "RS_synth.dcm"), enforce_file_format=True)
print("RTSTRUCT written")

# --- RTPLAN (ion) -----------------------------------------------------------
plan_uid = generate_uid()
ds = base_dataset("1.2.840.10008.5.1.4.1.1.481.8", plan_uid, "RTPLAN")
ds.SeriesInstanceUID = generate_uid()
ds.SeriesNumber = 3
ds.FrameOfReferenceUID = for_uid
ds.RTPlanLabel = args.plan_label
ds.RTPlanName = "Synthetic proton plan"
ds.RTPlanDate = DATE
ds.RTPlanTime = TIME
ds.RTPlanGeometry = "PATIENT"

dr = Dataset()
dr.DoseReferenceNumber = 1
dr.DoseReferenceStructureType = "SITE"
dr.DoseReferenceDescription = "TARGET"
dr.DoseReferenceType = "TARGET"
dr.TargetPrescriptionDose = f"{args.peak:.1f}"
ds.DoseReferenceSequence = [dr]

fg = Dataset()
fg.FractionGroupNumber = 1
fg.NumberOfFractionsPlanned = 30
fg.NumberOfBeams = 2
fg.NumberOfBrachyApplicationSetups = 0
rbs = []
for num, mset, bdose in [(1, 120.5, "1.05"), (2, 98.3, "0.95")]:
    rb = Dataset()
    rb.ReferencedBeamNumber = num
    rb.BeamMeterset = f"{mset}"
    rb.BeamDose = bdose
    rbs.append(rb)
fg.ReferencedBeamSequence = rbs
ds.FractionGroupSequence = [fg]

beams = []
for num, name, gantry in [(1, "G000", 0.0), (2, "G090", 90.0)]:
    b = Dataset()
    b.BeamNumber = num
    b.BeamName = name
    b.BeamType = "STATIC"
    b.RadiationType = "PROTON"
    b.ScanMode = "MODULATED"
    b.TreatmentMachineName = "SYNTH-PBS"
    b.TreatmentDeliveryType = "TREATMENT"
    b.NumberOfWedges = 0
    b.NumberOfCompensators = 0
    b.NumberOfBoli = 0
    b.NumberOfBlocks = 0
    b.FinalCumulativeMetersetWeight = "1.0"
    b.NumberOfControlPoints = 4
    b.NumberOfRangeShifters = 0
    b.NumberOfLateralSpreadingDevices = 0
    b.NumberOfRangeModulators = 0
    b.PatientSupportType = "TABLE"
    cps = []
    for ci, energy in enumerate([180.0, 160.0, 140.0, 120.0]):
        cp = Dataset()
        cp.ControlPointIndex = ci
        cp.NominalBeamEnergy = f"{energy}"
        cp.CumulativeMetersetWeight = f"{ci / 3.0:.4f}"
        if ci == 0:
            cp.GantryAngle = f"{gantry}"
            cp.GantryRotationDirection = "NONE"
            cp.PatientSupportAngle = "0"
            cp.PatientSupportRotationDirection = "NONE"
            cp.IsocenterPosition = ["0.0", f"{SHIFT_Y:.1f}", "0.0"]
        cps.append(cp)
    b.IonControlPointSequence = cps
    beams.append(b)
ds.IonBeamSequence = beams
ds.save_as(os.path.join(OUT, "RP_synth.dcm"), enforce_file_format=True)
print("RTPLAN written")

# --- RTDOSE -----------------------------------------------------------------
# Odd grid counts so that (0,0,0) — the dose peak — is exactly on the grid.
DNX = DNY = 47
DNZ = 41
DSP = 4.0
DX0 = -(DNX - 1) / 2.0 * DSP  # -92
DY0 = -(DNY - 1) / 2.0 * DSP
DZ0 = -(DNZ - 1) / 2.0 * SPACING  # -40, 2 mm frame steps

dxs = DX0 + np.arange(DNX) * DSP
dys = DY0 + np.arange(DNY) * DSP
dzs = DZ0 + np.arange(DNZ) * SPACING
DXX, DYY = np.meshgrid(dxs, dys)

SIGMA = 20.0
PEAK = args.peak
frames = []
for z in dzs:
    r2 = DXX**2 + (DYY - SHIFT_Y) ** 2 + z**2
    dose = PEAK * np.exp(-r2 / (2.0 * SIGMA**2))
    frames.append(dose)
dose3d = np.stack(frames, axis=0)  # [frame, row, col]

SCALING = 1.0e-3
stored = np.round(dose3d / SCALING).astype(np.uint32)

dose_uid = generate_uid()
ds = base_dataset("1.2.840.10008.5.1.4.1.1.481.2", dose_uid, "RTDOSE")
ds.SeriesInstanceUID = generate_uid()
ds.SeriesNumber = 4
ds.FrameOfReferenceUID = for_uid
ds.ImagePositionPatient = [f"{DX0:.3f}", f"{DY0:.3f}", f"{DZ0:.3f}"]
ds.ImageOrientationPatient = ["1", "0", "0", "0", "1", "0"]
ds.PixelSpacing = [f"{DSP:.3f}", f"{DSP:.3f}"]
ds.SliceThickness = ""
ds.Rows = DNY
ds.Columns = DNX
ds.NumberOfFrames = str(DNZ)
ds.FrameIncrementPointer = pydicom.tag.Tag(0x3004, 0x000C)
ds.GridFrameOffsetVector = [f"{(z - DZ0):.3f}" for z in dzs]
ds.BitsAllocated = 32
ds.BitsStored = 32
ds.HighBit = 31
ds.PixelRepresentation = 0
ds.SamplesPerPixel = 1
ds.PhotometricInterpretation = "MONOCHROME2"
ds.DoseUnits = "GY"
ds.DoseType = "PHYSICAL"
ds.DoseSummationType = "PLAN"
ds.DoseGridScaling = f"{SCALING:.9f}"
rp = Dataset()
rp.ReferencedSOPClassUID = "1.2.840.10008.5.1.4.1.1.481.8"
rp.ReferencedSOPInstanceUID = plan_uid
ds.ReferencedRTPlanSequence = [rp]
ds.PixelData = stored.tobytes()
ds.save_as(os.path.join(OUT, "RD_synth.dcm"), enforce_file_format=True)
print(f"RTDOSE written (max {dose3d.max():.2f} Gy)")

print(f"\nDone. Study in {OUT}")
