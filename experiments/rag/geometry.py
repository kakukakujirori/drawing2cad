"""How a build differs from its part: the scorer's IoU and pose, and the
connected pieces of material the build lacks or adds, found on a voxel grid
and drawn for the judge.

    python -m experiments.rag.geometry PAIRS.jsonl OUT_DIR [WORKERS]

Writes OUT_DIR/comparisons.jsonl and, per pair, gt.png, build.png and diff.png
in OUT_DIR/<key>/. Everything is drawn in the build's frame, where the axes of
its reading hold.
"""

import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.rag.schema import Comparison, Component, Pair, append_row, read_rows

# Voxel size, as a share of the part's longest side; finer for thin-walled parts.
PITCH = 1 / 256
MAX_VOXELS = 4e7
# Material closer than this to the other solid, as a share of the part's
# longest side, only shows where the surfaces nearly coincide. On 150 final
# builds, the pieces left appear on 15 of 75 with IoU >= 0.99, and are missing
# from 1 of 34 with IoU < 0.9, whose wall is only too thick.
NEAR = 0.005
# Pieces below this share of the part's volume are not reported.
MIN_SHARE = 1e-4
# Eye directions of the four panels: two from above, two from below.
EYES = [(1, -1, 1), (-1, 1, 1), (1, 1, -1), (-1, -1, -1)]
PANEL = 700


def _occupancy(solid, pitch: float) -> tuple[np.ndarray, np.ndarray]:
    """A solid's voxels, centred on multiples of pitch so that all grids share one
    lattice, and the grid index of the first one. Each layer is its cross
    section, filled even-odd so that holes stay empty."""
    from PIL import Image, ImageChops, ImageDraw

    bounds = np.array(solid.bounding_box()).reshape(2, 3)
    first = np.floor(bounds[0] / pitch).astype(int)
    shape = np.ceil(bounds[1] / pitch).astype(int) - first + 1
    grid = np.zeros(shape, bool)
    for k in range(shape[2]):
        layer = Image.new("1", (int(shape[0]), int(shape[1])))
        for polygon in solid.slice((first[2] + k) * pitch).to_polygons():
            ring = Image.new("1", layer.size)
            # Pixel i covers [i, i + 1), so a voxel centre lies half a pixel in.
            ImageDraw.Draw(ring).polygon([tuple(xy) for xy in polygon / pitch - first[:2] + 0.5], fill=1)
            layer = ImageChops.logical_xor(layer, ring)
        grid[:, :, k] = np.array(layer).T
    return grid, first


def _one_grid(a: np.ndarray, at: np.ndarray, b: np.ndarray, bt: np.ndarray):
    start = np.minimum(at, bt)
    shape = np.maximum(at + a.shape, bt + b.shape) - start

    def place(grid: np.ndarray, first: np.ndarray) -> np.ndarray:
        out = np.zeros(shape, bool)
        out[tuple(slice(o, o + n) for o, n in zip(first - start, grid.shape, strict=True))] = grid
        return out

    return place(a, at), place(b, bt), start


def _pieces(difference: np.ndarray, other: np.ndarray, total: int, near: int,
            prefix: str) -> list[tuple[str, float, np.ndarray]]:
    """Connected pieces of a difference, each grown back from its material more
    than `near` voxels from the other solid. Surfaces that nearly coincide leave
    skins all over the part, and no piece of them is that far."""
    from scipy import ndimage

    core = difference & (ndimage.distance_transform_edt(~other) > near)
    cube = np.ones((3, 3, 3), bool)
    labels, _ = ndimage.label(ndimage.binary_dilation(core, cube, iterations=near, mask=difference))
    sizes = np.bincount(labels.ravel())[1:]
    kept = [n for n in np.argsort(-sizes) if sizes[n] >= MIN_SHARE * total]
    return [(f"{prefix}{i}", sizes[n] / total, labels == n + 1) for i, n in enumerate(kept, 1)]


def _pitch(gt, extent: float) -> float:
    """A step that puts three voxels across a plate-like part's wall, within MAX_VOXELS."""
    bounds = np.array(gt.bounding_box()).reshape(2, 3)
    wall = 2 * gt.volume() / gt.surface_area()  # a plate's thickness
    pitch = min(extent * PITCH, wall / 3)
    return max(pitch, float(np.prod(bounds[1] - bounds[0]) / MAX_VOXELS) ** (1 / 3))


def _mesh(solid):
    import pyvista as pv

    mesh = solid.to_mesh()
    faces = np.asarray(mesh.tri_verts)
    return pv.PolyData(np.asarray(mesh.vert_properties)[:, :3], np.c_[np.full(len(faces), 3), faces].ravel())


def _voxel_mesh(mask: np.ndarray, start: np.ndarray, pitch: float):
    import pyvista as pv

    # Voxel indices are centres; the image's cells start half a voxel before them.
    grid = pv.ImageData(dimensions=np.array(mask.shape) + 1, spacing=(pitch,) * 3, origin=(start - 0.5) * pitch)
    grid.cell_data["inside"] = mask.ravel(order="F").astype(np.uint8)
    return grid.threshold(0.5).extract_surface()


def _draw(path: Path, meshes: list[tuple[object, dict]], labels: list[tuple[str, np.ndarray]]) -> None:
    import pyvista as pv

    plotter = pv.Plotter(off_screen=True, shape=(2, 2), window_size=(2 * PANEL, 2 * PANEL))
    plotter.set_background("white")
    for n, eye in enumerate(EYES):
        plotter.subplot(n // 2, n % 2)
        for mesh, style in meshes:
            plotter.add_mesh(mesh, smooth_shading=False, **style)
            if style.get("opacity", 1) == 1:
                plotter.add_mesh(mesh.extract_feature_edges(30), color="black", line_width=1)
        if labels:
            plotter.add_point_labels(np.array([at for _, at in labels]), [name for name, _ in labels],
                                     font_size=14, always_visible=True, shape_opacity=0.7)
        plotter.add_text(f"from {'+-'[eye[0] < 0]}X {'+-'[eye[1] < 0]}Y {'+-'[eye[2] < 0]}Z", font_size=9)
        plotter.add_axes()
        plotter.view_vector(eye, viewup=(0, 0, 1))
        plotter.reset_camera()
    plotter.screenshot(str(path))
    plotter.close()


def compare(pair: Pair, out: Path) -> Comparison:
    from zeroshot.evaluation.preprocess import ORIENTATIONS, prepare_pair
    from zeroshot.pipeline.verification.solid_mesh import PreprocessConfig, solid_mesh

    prepared = prepare_pair(pair.build, pair.gt)
    # The scorer turns the build onto the part; turn both back to the build's frame.
    back = np.column_stack((ORIENTATIONS[prepared.rotation_index].T, np.zeros(3)))
    gt = solid_mesh(prepared.gt_shape, PreprocessConfig()).transform(back)
    build = solid_mesh(prepared.pred_shape, PreprocessConfig()).transform(back)
    extent = PreprocessConfig().reference_extent
    pitch = _pitch(gt, extent)
    part, made, start = _one_grid(*_occupancy(gt, pitch), *_occupancy(build, pitch))
    total, near = int(part.sum()), max(1, round(NEAR * extent / pitch))
    pieces = _pieces(part & ~made, made, total, near, "M") + _pieces(made & ~part, part, total, near, "E")
    folder = out / pair.key
    folder.mkdir(parents=True, exist_ok=True)
    grey = {"color": "#c8c8c8"}
    _draw(folder / "gt.png", [(_mesh(gt), grey)], [])
    _draw(folder / "build.png", [(_mesh(build), grey)], [])
    labels = [(name, (np.argwhere(mask).mean(axis=0) + start) * pitch) for name, _, mask in pieces]
    colours = [(_voxel_mesh(mask, start, pitch), {"color": "red" if name[0] == "M" else "royalblue"})
               for name, _, mask in pieces]
    _draw(folder / "diff.png", [(_mesh(gt), {"color": "#c8c8c8", "opacity": 0.25}), *colours], labels)
    return Comparison(
        key=pair.key,
        iou=prepared.iou,
        rotation_index=prepared.rotation_index,
        components=[Component(id=name, share=share) for name, share, _ in pieces],
        images={name: str(folder / f"{name}.png") for name in ("gt", "build", "diff")},
    )


def _work(job: tuple[Pair, Path]) -> Comparison | str:
    # Software rendering, as run_parallel.sh sets it: no GPU memory needed.
    os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/50_mesa.json")
    os.environ.setdefault("VTK_DEFAULT_OPENGL_WINDOW", "vtkEGLRenderWindow")
    try:
        return compare(*job)
    except Exception as error:  # one bad build must not stop the rest
        return f"{job[0].key}: {type(error).__name__}: {error}"


if __name__ == "__main__":
    pairs, out = list(read_rows(sys.argv[1], Pair)), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    done = {row.key for row in read_rows(out / "comparisons.jsonl", Comparison)}
    jobs = [(pair, out) for pair in pairs if pair.key not in done]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for result in pool.map(_work, jobs):
            if isinstance(result, str):
                print("FAILED", result, flush=True)
            else:
                append_row(out / "comparisons.jsonl", result)
                print(result.key, f"IoU {result.iou:.3f}", [(c.id, round(c.share, 3)) for c in result.components], flush=True)
