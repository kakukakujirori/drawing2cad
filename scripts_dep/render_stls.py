#!/usr/bin/env python
"""Offscreen-render STL files to PNG (pyvista/VTK) for eyeballing B2 outputs.
Run in py312 (has pyvista): python scripts/render_stls.py <stl_dir> <out_dir> [ids...]"""
import os, sys, glob
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
import pyvista as pv
pv.OFF_SCREEN = True

stl_dir, out_dir = sys.argv[1], sys.argv[2]
ids = sys.argv[3:]
os.makedirs(out_dir, exist_ok=True)
files = sorted(glob.glob(os.path.join(stl_dir, "*.stl")))
if ids:
    files = [f for f in files if os.path.splitext(os.path.basename(f))[0] in ids]
for f in files:
    fid = os.path.splitext(os.path.basename(f))[0]
    try:
        mesh = pv.read(f)
        p = pv.Plotter(off_screen=True, window_size=(480, 480))
        p.add_mesh(mesh, color="#ffd966", show_edges=True, edge_color="#333333")
        p.view_isometric()
        p.add_text(f"{fid}  bbox={[round(x,1) for x in mesh.bounds]}", font_size=8)
        out = os.path.join(out_dir, f"{fid}.png")
        p.screenshot(out)
        p.close()
        print(f"rendered {fid} -> {out}  bounds={[round(x,1) for x in mesh.bounds]}")
    except Exception as e:
        print(f"FAIL {fid}: {type(e).__name__}: {e}")
