"""Choose dimensions for one projection and draw them.

Dimension style randomisation (arrow blocks, text styles, unit suffix, decimal
separator, background fill) follows ParaCAD's `draw_random_dimension`; the
choice of *what* to dimension does not. ParaCAD picks random pairs from all
endpoints, which suits a single sketch but on a multi-view drawing produces
dimensions spanning two projections.
"""

import math

# Font table taken from ParaCAD (dxfWriter_cyw_white.py, Apache-2.0). Faces the
# host lacks fall back in the renderer, which is itself a source of variety.
TEXT_STYLES = {
    "Style1": {"font": "Arial.ttf", "width": 1.0, "height": 0.0},
    "Style2": {"font": "Times New Roman.ttf", "width": 0.8, "height": 0.0},
    "Style3": {"font": "Courier New.ttf", "width": 1.2, "height": 0.0},
    "Style4": {"font": "Verdana.ttf", "width": 1.0, "height": 0.0},
    "Style5": {"font": "Calibri.ttf", "width": 1.0, "height": 0.0},
    "Style6": {"font": "SimSun.ttf", "width": 1.0, "height": 0.0, "oblique": 15},
    "Style7": {"font": "SimHei.ttf", "width": 1.0, "height": 0.0, "oblique": -15},
    "Style8": {"font": "Microsoft YaHei.ttf", "width": 0.9, "height": 0.0},
    "Style9": {"font": "Comic Sans MS.ttf", "width": 1.0, "height": 0.0},
    "Style10": {"font": "Georgia.ttf", "width": 1.0, "height": 0.0},
}

ARROW_STYLES = (
    "",
    "CLOSED",
    "CLOSEDBLANK",
    "OPEN",
    "OPEN30",
    "ARCHTICK",
    "EZ_ARROW",
    "EZ_ARROW_FILLED",
    "DOTSMALL",
)

BACKGROUND_COLORS = (9, 41, 42, 43, 44, 45, 46, 47, 50)


def sample_style(rng):
    """Draw one style for a whole sheet, so notation stays consistent on it."""
    with_unit = rng.random() < 0.25
    with_fill = rng.random() < 0.30
    return {
        "arrow": rng.choice(ARROW_STYLES),
        "text_style": rng.choice(list(TEXT_STYLES)),
        "dimdsep": ord(".") if rng.random() < 0.9 else ord(","),
        "dimtad": rng.choice([1, 1, 1, 0]),
        "dimdec": rng.choice([0, 1, 1, 2]),
        "unit": "mm" if with_unit else "",
        "fill_color": rng.choice(BACKGROUND_COLORS) if with_fill else None,
    }


def create_text_styles(doc):
    for name, attribs in TEXT_STYLES.items():
        if name not in doc.styles:
            doc.styles.new(name, dxfattribs=attribs)


class DimensionWriter:
    """Draws DIMENSION entities sized relative to the drawing, not in absolute
    units: a fixed dimtxt renders text as large as the part on an A4 sheet."""

    def __init__(self, doc, msp, style, txt_height=3.5, lineweight=25):
        self.doc = doc
        self.msp = msp
        self.style = style
        self.txt = txt_height
        self.lineweight = lineweight

    def _override(self):
        s = self.style
        override = {
            "dimtxsty": s["text_style"],
            "dimblk": s["arrow"],
            "dimtxt": self.txt,
            "dimasz": self.txt * 0.85,
            "dimgap": self.txt * 0.30,
            "dimexe": self.txt * 0.40,
            "dimexo": self.txt * 0.50,
            "dimtad": s["dimtad"],
            "dimdec": s["dimdec"],
            "dimdsep": s["dimdsep"],
            "dimlfac": 1,
            "dimlwd": self.lineweight,
            "dimlwe": self.lineweight,
            "dimclrd": 7,
            "dimclre": 7,
            "dimclrt": 7,
        }
        if s["unit"]:
            override["dimpost"] = f"<>{s['unit']}"
        if s["fill_color"] is not None:
            override["dimtfill"] = 1
            override["dimtfillclr"] = s["fill_color"]
        return override

    def linear(self, base, p1, p2, angle):
        return self.msp.add_linear_dim(
            base=base, p1=p1, p2=p2, angle=angle, override=self._override()
        )

    def _leader_location(self, center, radius, angle, reach):
        distance = radius + reach * self.txt
        return (
            center[0] + distance * math.cos(math.radians(angle)),
            center[1] + distance * math.sin(math.radians(angle)),
        )

    def diameter(self, center, radius, angle, reach):
        override = self._override()
        override["dimtoh"] = 1  # keep leader text horizontal and readable
        override["dimtih"] = 1
        return self.msp.add_diameter_dim(
            center=center,
            radius=radius,
            location=self._leader_location(center, radius, angle, reach),
            angle=angle,
            override=override,
        )

    def radius(self, center, radius, angle, reach):
        override = self._override()
        override["dimtoh"] = 1
        override["dimtih"] = 1
        return self.msp.add_radius_dim(
            center=center,
            radius=radius,
            location=self._leader_location(center, radius, angle, reach),
            override=override,
        )

    def commit(self, dim, occupancy):
        """Render, read back the real text box, and undo the dimension if it
        collides. Predicting the text position fails for leader dimensions,
        where landing length and dimtoh handling depend on the dimstyle."""
        dim.render()
        entity = dim.dimension
        block_name = entity.dxf.geometry

        rect = None
        for sub in self.doc.blocks.get(block_name):
            if sub.dxftype() == "MTEXT":
                rect = _mtext_rect(sub)
                break

        if rect is None or occupancy.fits(rect):
            if rect is not None:
                occupancy.add(rect)
            return True

        self.msp.delete_entity(entity)
        self.doc.blocks.delete_block(block_name, safe=False)
        return False


def _mtext_rect(mtext):
    height = mtext.dxf.char_height
    text = mtext.text.replace("%%c", "D").replace("\\P", " ")
    width = max(len(text), 1) * height * 0.62
    rotation = mtext.get_dxf_attrib("rotation", 0.0) % 180.0
    if 45.0 < rotation < 135.0:
        width, height = height * 1.2, width
    else:
        height = height * 1.2
    cx, cy = mtext.dxf.insert.x, mtext.dxf.insert.y
    return (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2)


class Occupancy:
    """Ledger of placed text boxes, consulted before committing a dimension."""

    def __init__(self, pad=0.0):
        self.rects = []
        self.pad = pad

    def fits(self, r):
        p = self.pad
        return all(
            r[2] + p <= o[0] or r[0] - p >= o[2] or r[3] + p <= o[1] or r[1] - p >= o[3]
            for o in self.rects
        )

    def add(self, rect):
        self.rects.append(rect)
