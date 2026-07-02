#!/usr/bin/env python
"""Rebuild the canonical AMVDG example solid: spec/flange.step.

The worked example (example_flange_v0.3.json) is the renderer's output on THIS
solid, so the example is reproducible rather than hand-authored. Run in the
`drawing2cad` conda env (needs cadquery):

    python spec/build_flange.py            # -> spec/flange.step

A round bolt flange chosen to exercise every AMVDG geometry case in one part:
  * concentric circles in the top view (plate OD, raised hub, central bore),
  * a polar bolt-circle (6 through-holes) -> repeated circles + a bolt_circle feature,
  * a stepped front-view profile (plate + hub) -> visible + hidden edges,
  * axis-aligned cylinders throughout -> the dimensioner/oracle handle them exactly.
"""
import os
import cadquery as cq

# --- parameters (these ARE the intended dimensions of the part) --------------
PLATE_OD = 120.0   # base plate outer diameter
PLATE_T  = 16.0    # base plate thickness
HUB_OD   = 76.0    # raised central hub outer diameter
HUB_H    = 22.0    # hub height above the plate
BORE_D   = 40.0    # central through bore diameter
BOLT_PCD = 95.0    # bolt-hole pitch-circle diameter
BOLT_D   = 11.0    # bolt-hole diameter
BOLT_N   = 6       # number of bolt holes


def build():
    r = (
        cq.Workplane("XY")
        .circle(PLATE_OD / 2.0).extrude(PLATE_T)                 # base plate
        .faces(">Z").workplane()
        .circle(HUB_OD / 2.0).extrude(HUB_H)                     # raised hub
        .faces(">Z").workplane()
        .circle(BORE_D / 2.0).cutThruAll()                       # central bore
        .faces("<Z").workplane()
        .polarArray(BOLT_PCD / 2.0, 0, 360, BOLT_N)
        .circle(BOLT_D / 2.0).cutThruAll()                       # bolt holes
    )
    return r


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "flange.step")
    r = build()
    solid = r.val()
    assert solid.isValid(), "flange solid failed OCC validity check"
    cq.exporters.export(r, out)
    print("wrote %s  (volume=%.1f mm^3, faces=%d)"
          % (out, solid.Volume(), len(solid.Faces())))
