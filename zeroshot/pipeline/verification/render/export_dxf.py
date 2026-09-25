"""Export one projected view as a DXF of its own, and a picture of it.

The view is written at 1:1 in model millimetres, in the U and V the view was
drawn in, so a coordinate read off it is a measurement of the solid. The
picture is the same linework rasterised, so a reader can look at a view
without rasterising it first.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import cadquery as cq
import ezdxf
from cadquery.occ_impl.exporters.dxf import DxfDocument
from ezdxf import bbox
from ezdxf.addons.drawing import Frontend, RenderContext
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
from ezdxf.addons.drawing.properties import LayoutProperties
from ezdxf.layouts import Modelspace
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

_TEMPLATE = Path(__file__).with_name("techdraw_template.dxf")

# Enough to read a hole from, and small enough that several views in a message
# stay affordable.
_PNG_DPI = 100
_PNG_SHORT_SIDE = 480
_PNG_LONG_SIDE = 1600
DEFAULT_PNG_MARGIN_RATIO = 0.04

# A view narrower than this on either axis carries no recoverable shape. It
# only fires on pathological inputs: an empty projection, or a knife-edge view
# of a near-zero-thickness plate that collapses to a single line.
MIN_VIEW_EXTENT_MM = 0.05


class DegenerateDrawingError(RuntimeError):
    """A view holds nothing a drawing could be made of."""


def write_view_dxf(
    dxf_path: Path, visible: Sequence[cq.Edge], hidden: Sequence[cq.Edge], layer: str
) -> None:
    """Write one view's edges, hidden ones dashed, to an existing output directory.

    `layer` names the view, and says which one was refused when it has no area.
    """
    doc = ezdxf.readfile(_TEMPLATE)
    modelspace = doc.modelspace()
    if layer not in doc.layers:
        doc.layers.add(layer)
    for edges, linetype in ((visible, "Continuous"), (hidden, "HIDDEN")):
        if not edges:
            continue
        # Refit splines: ezdxf refuses the high-degree ones HLR can return.
        converted = DxfDocument(approx="spline").add_shape(cq.Workplane().add(edges))
        for entity in converted.msp:
            entity.dxf.layer, entity.dxf.linetype = layer, linetype
            modelspace.add_foreign_entity(entity)

    extents = bbox.extents(modelspace)
    if not extents.has_data:
        raise DegenerateDrawingError(f"view {layer!r} has no edges")
    width, height = extents.size.x, extents.size.y
    if width < MIN_VIEW_EXTENT_MM or height < MIN_VIEW_EXTENT_MM:
        raise DegenerateDrawingError(
            f"view {layer!r} has near-zero extent (w={width:.4f}, h={height:.4f} mm)"
        )
    doc.header["$EXTMIN"] = (extents.extmin.x, extents.extmin.y, 0.0)
    doc.header["$EXTMAX"] = (extents.extmax.x, extents.extmax.y, 0.0)
    doc.saveas(Path(dxf_path))


def png_bounds(
    modelspace: Modelspace,
    *,
    margin_ratio: float,
) -> tuple[float, float, float, float]:
    """UV bounds, with each margin a fraction of the drawing's short side."""
    if not math.isfinite(margin_ratio) or margin_ratio < 0:
        raise ValueError("margin_ratio must be finite and non-negative")
    extents = bbox.extents(modelspace)
    margin = min(extents.size.x, extents.size.y) * margin_ratio
    return (
        extents.extmin.x - margin,
        extents.extmin.y - margin,
        extents.extmax.x + margin,
        extents.extmax.y + margin,
    )


def export_to_png(
    dxf_path: Path,
    image_path: Path | None = None,
    *,
    margin_ratio: float = DEFAULT_PNG_MARGIN_RATIO,
) -> Path:
    """Rasterise black on white, with space around all four silhouette edges."""
    image_path = image_path or Path(dxf_path).with_suffix(".png")
    modelspace = ezdxf.readfile(dxf_path).modelspace()
    x_min, y_min, x_max, y_max = png_bounds(modelspace, margin_ratio=margin_ratio)
    width, height = x_max - x_min, y_max - y_min
    pixels = min(
        _PNG_SHORT_SIDE / min(width, height), _PNG_LONG_SIDE / max(width, height)
    )
    figure = Figure(
        figsize=(width * pixels / _PNG_DPI, height * pixels / _PNG_DPI),
        dpi=_PNG_DPI,
    )
    FigureCanvasAgg(figure)
    axes = figure.add_axes((0, 0, 1, 1))
    axes.set_axis_off()
    properties = LayoutProperties.from_layout(modelspace)
    properties.set_colors("#FFFFFF", "#000000")
    Frontend(
        RenderContext(modelspace.doc), MatplotlibBackend(axes, adjust_figure=False)
    ).draw_layout(modelspace, layout_properties=properties)
    axes.set_xlim(x_min, x_max)
    axes.set_ylim(y_min, y_max)
    figure.savefig(image_path, dpi=_PNG_DPI, facecolor=axes.get_facecolor())
    return image_path
