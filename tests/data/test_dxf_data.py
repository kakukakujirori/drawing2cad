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
)


class DXFPrimitiveParserTest(unittest.TestCase):
    def _write_fixture(self, directory: str, *, unsupported: bool = False) -> Path:
        path = Path(directory) / "drawing.dxf"
        document = ezdxf.new("R2000")
        if "HIDDEN" not in document.linetypes:
            document.linetypes.add(
                "HIDDEN",
                pattern=[0.2, 0.1, -0.1],
                description="hidden test line",
            )
        modelspace = document.modelspace()
        modelspace.add_line((148.5, 105.0), (158.5, 105.0))
        modelspace.add_arc((160.0, 105.0), 5.0, 0.0, 180.0)
        modelspace.add_circle(
            (170.0, 105.0),
            4.0,
            dxfattribs={"linetype": "HIDDEN"},
        )
        modelspace.add_ellipse(
            (180.0, 105.0),
            major_axis=(5.0, 0.0),
            ratio=0.5,
        )
        modelspace.add_spline(
            fit_points=[(190.0, 100.0), (194.0, 108.0), (200.0, 105.0)]
        )
        modelspace.add_lwpolyline(
            [(205.0, 100.0, 0.5), (210.0, 105.0, 0.0), (215.0, 100.0, 0.0)],
            format="xyb",
        )
        # Projection artifacts outside the declared A4 sheet are discarded.
        modelspace.add_line((1000.0, 50.0), (1001.0, 50.0))
        # Center/annotation layer content is intentionally excluded by default.
        modelspace.add_line((0.0, 0.0), (297.0, 210.0), dxfattribs={"layer": "10"})
        if unsupported:
            modelspace.add_text("dimension", dxfattribs={"layer": "0"})
        document.saveas(path)
        return path

    def test_supported_entities_are_uniformly_sampled_and_featured(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            path = self._write_fixture(directory)
            config = DXFPrimitiveConfig(samples_per_primitive=17)
            parsed = DXFPrimitiveParser(config).parse(path)

        self.assertEqual(parsed.sample_features.shape, (6, 17, 3))
        self.assertEqual(parsed.primitive_type_ids.shape, (6,))
        self.assertTrue(torch.isfinite(parsed.sample_features).all())
        self.assertNotIn("INSERT", parsed.entity_type_names)

        line_index = parsed.entity_type_names.index("LINE")
        line = parsed.sample_features[line_index]
        scale = 1.8 / 297.0
        torch.testing.assert_close(line[0], torch.tensor([0.0, 0.0, 1.0]))
        torch.testing.assert_close(
            line[-1], torch.tensor([10.0 * scale, 0.0, 1.0])
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

    def test_visibility_filters_and_strict_unsupported_entities(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            path = self._write_fixture(directory, unsupported=True)
            visible_only = DXFPrimitiveParser(
                DXFPrimitiveConfig(include_hidden=False)
            ).parse(path)
            self.assertTrue(torch.all(visible_only.sample_features[..., 2] == 1.0))

            with self.assertRaisesRegex(DXFParseError, "unsupported entity TEXT"):
                DXFPrimitiveParser(
                    DXFPrimitiveConfig(strict_entity_types=True)
                ).parse(path)

    def test_degenerate_entities_are_skipped_or_reported(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            path = Path(directory) / "degenerate.dxf"
            document = ezdxf.new("R2000")
            modelspace = document.modelspace()
            modelspace.add_line((1.0, 1.0), (1.0, 1.0))
            modelspace.add_line((2.0, 2.0), (3.0, 2.0))
            document.saveas(path)
            parsed = DXFPrimitiveParser().parse(path)
            self.assertEqual(parsed.num_primitives, 1)
            with self.assertRaisesRegex(DXFParseError, "zero geometric length"):
                DXFPrimitiveParser(
                    DXFPrimitiveConfig(strict_entity_types=True)
                ).parse(path)

    def test_drawing_with_only_out_of_sheet_geometry_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            path = Path(directory) / "invalid_projection.dxf"
            document = ezdxf.new("R2000")
            document.modelspace().add_line((1000.0, 50.0), (1001.0, 50.0))
            document.saveas(path)

            with self.assertRaisesRegex(DXFParseError, "no usable primitives"):
                DXFPrimitiveParser().parse(path)


if __name__ == "__main__":
    unittest.main()
