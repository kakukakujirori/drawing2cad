import math
from pathlib import Path
import tempfile
import unittest

import ezdxf
import torch

from src.data import (
    DXFParseError,
    DXFPrimitiveConfig,
    DXFPrimitiveParser,
    DXF_PRIMITIVE_TYPE_TO_ID,
    VIEW_DIRECTION_TO_ID,
)


class DXFPrimitiveParserTest(unittest.TestCase):
    """The parser reads each primitive's view from its DXF layer.

    Fixtures place geometry directly on the ``front`` / ``top`` / ``right``
    layers the renderer and stamping tool write; there is no sidecar metadata.
    """

    def _write_fixture(self, directory: str, *, unsupported: bool = False) -> Path:
        path = Path(directory) / "drawing.dxf"
        document = ezdxf.new("R2000")
        if "HIDDEN" not in document.linetypes:
            document.linetypes.add(
                "HIDDEN",
                pattern=[0.2, 0.1, -0.1],
                description="hidden test line",
            )
        for layer in ("front", "top", "right"):
            if layer not in document.layers:
                document.layers.add(layer)
        modelspace = document.modelspace()
        # front view: LINE, ARC, CIRCLE (hidden)
        modelspace.add_line(
            (148.5, 105.0), (158.5, 105.0), dxfattribs={"layer": "front"}
        )
        modelspace.add_arc(
            (160.0, 105.0), 5.0, 0.0, 180.0, dxfattribs={"layer": "front"}
        )
        modelspace.add_circle(
            (170.0, 105.0),
            4.0,
            dxfattribs={"layer": "front", "linetype": "HIDDEN"},
        )
        # top view: ELLIPSE, SPLINE
        modelspace.add_ellipse(
            (180.0, 105.0),
            major_axis=(5.0, 0.0),
            ratio=0.5,
            dxfattribs={"layer": "top"},
        )
        modelspace.add_spline(
            fit_points=[(190.0, 100.0), (194.0, 108.0), (200.0, 105.0)],
            dxfattribs={"layer": "top"},
        )
        # right view: LWPOLYLINE
        modelspace.add_lwpolyline(
            [(205.0, 100.0, 0.5), (210.0, 105.0, 0.0), (215.0, 100.0, 0.0)],
            format="xyb",
            dxfattribs={"layer": "right"},
        )
        # Projection artifact outside the declared A4 sheet, on a view layer:
        # kept by the layer filter but dropped by the inside-sheet check.
        modelspace.add_line(
            (1000.0, 50.0), (1001.0, 50.0), dxfattribs={"layer": "front"}
        )
        # Content on a non-view layer is excluded by the layer filter.
        modelspace.add_line((0.0, 0.0), (297.0, 210.0), dxfattribs={"layer": "10"})
        if unsupported:
            modelspace.add_text("dimension", dxfattribs={"layer": "front"})
        document.saveas(path)
        return path

    def test_supported_entities_are_uniformly_sampled_and_featured(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            path = self._write_fixture(directory)
            config = DXFPrimitiveConfig(samples_per_primitive=17)
            parsed = DXFPrimitiveParser(config).parse(path, sample_id="fixture")

        self.assertEqual(parsed.sample_features.shape, (6, 17, 7))
        self.assertEqual(parsed.primitive_type_ids.shape, (6,))
        self.assertEqual(parsed.view_direction_ids.shape, (6,))
        self.assertTrue(torch.isfinite(parsed.sample_features).all())
        self.assertNotIn("INSERT", parsed.entity_type_names)

        line_index = parsed.entity_type_names.index("LINE")
        line = parsed.sample_features[line_index]
        scale = 1.8 / 297.0
        torch.testing.assert_close(
            line[0], torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, -0.5])
        )
        torch.testing.assert_close(
            line[-1],
            torch.tensor([10.0 * scale, 0.0, 1.0, 1.0, 0.0, 0.0, 0.5]),
        )
        line_steps = torch.linalg.vector_norm(line[1:, :2] - line[:-1, :2], dim=-1)
        torch.testing.assert_close(
            line_steps, torch.full_like(line_steps, line_steps.mean())
        )

        circle_index = parsed.entity_type_names.index("CIRCLE")
        circle = parsed.sample_features[circle_index]
        torch.testing.assert_close(circle[0, :2], circle[-1, :2])
        self.assertTrue(torch.all(circle[:, 2] == -1.0))
        circle_steps = torch.linalg.vector_norm(
            circle[1:, :2] - circle[:-1, :2], dim=-1
        )
        self.assertLess(
            (circle_steps.max() - circle_steps.min()).item(),
            1e-5,
        )

        expected_ids = [
            DXF_PRIMITIVE_TYPE_TO_ID[name] for name in parsed.entity_type_names
        ]
        self.assertEqual(parsed.primitive_type_ids.tolist(), expected_ids)
        # View direction is the entity's layer, in modelspace iteration order.
        self.assertEqual(
            parsed.view_direction_ids.tolist(),
            [
                VIEW_DIRECTION_TO_ID["front"],
                VIEW_DIRECTION_TO_ID["front"],
                VIEW_DIRECTION_TO_ID["front"],
                VIEW_DIRECTION_TO_ID["top"],
                VIEW_DIRECTION_TO_ID["top"],
                VIEW_DIRECTION_TO_ID["right"],
            ],
        )

    def test_geometry_channels_match_analytic_values(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            path = self._write_fixture(directory)
            config = DXFPrimitiveConfig(samples_per_primitive=17)
            parsed = DXFPrimitiveParser(config).parse(path, sample_id="fixture")
        scale = 1.8 / 297.0
        features = parsed.sample_features

        tangent_norms = torch.linalg.vector_norm(features[..., 3:5], dim=-1)
        torch.testing.assert_close(tangent_norms, torch.ones_like(tangent_norms))

        arc_positions = features[..., 6]
        torch.testing.assert_close(
            arc_positions[:, 0], torch.full_like(arc_positions[:, 0], -0.5)
        )
        torch.testing.assert_close(
            arc_positions[:, -1], torch.full_like(arc_positions[:, -1], 0.5)
        )
        self.assertTrue(torch.all(arc_positions[:, 1:] > arc_positions[:, :-1]))

        line = features[parsed.entity_type_names.index("LINE")]
        torch.testing.assert_close(line[:, 3], torch.ones(17))
        torch.testing.assert_close(line[:, 4], torch.zeros(17))
        torch.testing.assert_close(line[:, 5], torch.zeros(17))

        # ezdxf circles/arcs are traversed counterclockwise, so the signed
        # curvature is positive and Menger curvature on exact circle points is
        # exact: log1p(1 / normalized_radius) at every sample.
        circle = features[parsed.entity_type_names.index("CIRCLE")]
        expected_circle = math.log1p(1.0 / (4.0 * scale))
        torch.testing.assert_close(circle[:, 5], torch.full((17,), expected_circle))
        torch.testing.assert_close(circle[0, 3:5], circle[-1, 3:5])

        arc = features[parsed.entity_type_names.index("ARC")]
        expected_arc = math.log1p(1.0 / (5.0 * scale))
        torch.testing.assert_close(arc[:, 5], torch.full((17,), expected_arc))

    def test_visibility_filters_and_strict_unsupported_entities(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            path = self._write_fixture(directory, unsupported=True)
            visible_only = DXFPrimitiveParser(
                DXFPrimitiveConfig(include_hidden=False)
            ).parse(path)
            self.assertTrue(torch.all(visible_only.sample_features[..., 2] == 1.0))

            with self.assertRaisesRegex(DXFParseError, "unsupported entity TEXT"):
                DXFPrimitiveParser(DXFPrimitiveConfig(strict_entity_types=True)).parse(
                    path
                )

    def test_view_direction_is_read_from_the_layer(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            path = Path(directory) / "views.dxf"
            document = ezdxf.new("R2000")
            for layer in ("front", "top", "right"):
                document.layers.add(layer)
            modelspace = document.modelspace()
            modelspace.add_line((10.0, 10.0), (20.0, 10.0), dxfattribs={"layer": "top"})
            modelspace.add_line(
                (30.0, 10.0), (40.0, 10.0), dxfattribs={"layer": "front"}
            )
            modelspace.add_line(
                (50.0, 10.0), (60.0, 10.0), dxfattribs={"layer": "right"}
            )
            # A primitive on a non-view layer is not model input.
            modelspace.add_line((70.0, 10.0), (80.0, 10.0), dxfattribs={"layer": "0"})
            document.saveas(path)

            parsed = DXFPrimitiveParser().parse(path)
            self.assertEqual(parsed.num_primitives, 3)
            self.assertEqual(
                parsed.view_direction_ids.tolist(),
                [
                    VIEW_DIRECTION_TO_ID["top"],
                    VIEW_DIRECTION_TO_ID["front"],
                    VIEW_DIRECTION_TO_ID["right"],
                ],
            )

            # An included non-view layer is silently dropped, or raises in strict
            # mode so an unexpected layer surfaces during audits.
            config = DXFPrimitiveConfig(included_layers=("front", "top", "right", "0"))
            self.assertEqual(DXFPrimitiveParser(config).parse(path).num_primitives, 3)
            strict = DXFPrimitiveConfig(
                included_layers=("front", "top", "right", "0"),
                strict_entity_types=True,
            )
            with self.assertRaisesRegex(DXFParseError, "is not a view direction"):
                DXFPrimitiveParser(strict).parse(path)

    def test_degenerate_entities_are_skipped_or_reported(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            path = Path(directory) / "degenerate.dxf"
            document = ezdxf.new("R2000")
            document.layers.add("front")
            modelspace = document.modelspace()
            modelspace.add_line((1.0, 1.0), (1.0, 1.0), dxfattribs={"layer": "front"})
            modelspace.add_line((2.0, 2.0), (3.0, 2.0), dxfattribs={"layer": "front"})
            document.saveas(path)
            parsed = DXFPrimitiveParser().parse(path)
            self.assertEqual(parsed.num_primitives, 1)
            self.assertEqual(parsed.num_skipped_degenerate_entities, 1)
            with self.assertRaisesRegex(DXFParseError, "zero geometric length"):
                DXFPrimitiveParser(DXFPrimitiveConfig(strict_entity_types=True)).parse(
                    path
                )

    def test_drawing_with_only_out_of_sheet_geometry_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            path = Path(directory) / "invalid_projection.dxf"
            document = ezdxf.new("R2000")
            document.layers.add("front")
            document.modelspace().add_line(
                (1000.0, 50.0), (1001.0, 50.0), dxfattribs={"layer": "front"}
            )
            document.saveas(path)

            with self.assertRaisesRegex(DXFParseError, "no usable primitives"):
                DXFPrimitiveParser().parse(path)


if __name__ == "__main__":
    unittest.main()
