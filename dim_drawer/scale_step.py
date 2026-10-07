"""Scale STEP parts to the millimetre geometry of third-angle DXF drawings."""

import argparse
import math
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory

import ezdxf
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.gp import gp_Pnt, gp_Trsf
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.Interface import Interface_Static
from OCC.Core.STEPControl import (
    STEPControl_AsIs,
    STEPControl_Reader,
    STEPControl_Writer,
)

from dim_drawer.extract import extract_doc
from dim_drawer.views import split_views


def drawing_extents(dxf_path):
    """Six model lengths: front XY, top XZ, right ZY, excluding center marks."""
    doc = ezdxf.readfile(dxf_path)
    if doc.units != ezdxf.units.MM:
        raise ValueError(f"DXF must use millimetres: {dxf_path}")
    views = split_views(extract_doc(doc))
    if len(views) != 3:
        raise ValueError(f"expected three views, found {len(views)}: {dxf_path}")
    top = max(views, key=lambda v: v["bbox"][1] + v["bbox"][3])
    right = max(views, key=lambda v: v["bbox"][0] + v["bbox"][2])
    if top is right:
        raise ValueError(f"expected a third-angle L arrangement: {dxf_path}")
    front = next(v for v in views if v is not top and v is not right)
    lengths = tuple(
        length
        for v in (front, top, right)
        for length in (
            v["bbox"][2] - v["bbox"][0],
            v["bbox"][3] - v["bbox"][1],
        )
    )
    if not all(math.isfinite(v) and v > 0 for v in lengths):
        raise ValueError(f"invalid DXF view extents: {lengths}")
    return lengths


def _read_step(path):
    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone or not reader.TransferRoots():
        raise ValueError(f"failed to read STEP: {path}")
    shape = reader.OneShape()
    if shape.IsNull():
        raise ValueError(f"empty STEP: {path}")
    return shape


def _extents(shape):
    box = Bnd_Box()
    brepbndlib.AddOptimal(shape, box, False, False)
    bounds = box.Get()
    xyz = tuple(bounds[i + 3] - bounds[i] for i in range(3))
    if not all(math.isfinite(v) and v > 0 for v in xyz):
        raise ValueError(f"invalid STEP extents: {xyz}")
    x, y, z = xyz
    return x, y, x, z, z, y


def _check_extents(actual, expected, tolerance_mm):
    error = max(abs(a - b) for a, b in zip(actual, expected, strict=True))
    if error > tolerance_mm:
        raise ValueError(
            f"view extents disagree by {error:.6f} mm "
            f"(tolerance {tolerance_mm:g} mm): {actual} vs {expected}"
        )
    return error


def scale_step(step_path, dxf_path, output_path, tolerance_mm=0.01):
    """Apply one uniform scale, then verify the exported STEP before replacing it."""
    if not math.isfinite(tolerance_mm) or tolerance_mm <= 0:
        raise ValueError("tolerance_mm must be finite and positive")
    output_path = Path(output_path)
    if Path(step_path).resolve() == output_path.resolve():
        raise ValueError("output must differ from the source STEP")
    expected = drawing_extents(dxf_path)
    shape = _read_step(step_path)
    source = _extents(shape)
    # ponytail: fit extents of paired, axis-aligned views; match edges for other poses.
    factor = median(b / a for a, b in zip(source, expected, strict=True))
    _check_extents(tuple(v * factor for v in source), expected, tolerance_mm)

    transform = gp_Trsf()
    transform.SetScale(gp_Pnt(0, 0, 0), factor)
    scaled = BRepBuilderAPI_Transform(shape, transform, True).Shape()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_path.parent) as scratch:
        temporary = Path(scratch) / output_path.name
        Interface_Static.SetCVal("write.step.schema", "AP203")
        Interface_Static.SetCVal("write.step.unit", "MM")
        writer = STEPControl_Writer()
        if writer.Transfer(scaled, STEPControl_AsIs) != IFSelect_RetDone:
            raise RuntimeError(f"failed to transfer STEP: {step_path}")
        if writer.Write(str(temporary)) != IFSelect_RetDone:
            raise RuntimeError(f"failed to write STEP: {output_path}")
        error = _check_extents(_extents(_read_step(temporary)), expected, tolerance_mm)
        temporary.replace(output_path)
    return factor, error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step-dir", type=Path, required=True)
    parser.add_argument("--dxf-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tolerance-mm", type=float, default=0.01)
    args = parser.parse_args()
    steps = sorted(args.step_dir.glob("*.step"))
    if not steps:
        parser.error(f"no STEP files in {args.step_dir}")
    for step in steps:
        factor, error = scale_step(
            step,
            args.dxf_dir / f"{step.stem}.dxf",
            args.out / step.name,
            args.tolerance_mm,
        )
        print(f"{step.stem}: scale={factor:.9f} max_extent_error={error:.6f} mm")


if __name__ == "__main__":
    main()
