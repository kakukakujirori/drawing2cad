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
    ViewBBox,
)


class DXFPrimitiveParserTest(unittest.TestCase):
    @staticmethod
    def _view_bboxes() -> tuple[ViewBBox, ...]:
        return (
            ViewBBox("front", (140.0, 90.0, 175.0, 120.0)),
            ViewBBox("top", (175.01, 90.0, 202.0, 120.0)),
            ViewBBox("right", (202.01, 90.0, 220.0, 120.0)),
        )

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
            parsed = DXFPrimitiveParser(config).parse(
                path, view_bboxes=self._view_bboxes(), sample_id="fixture"
            )

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
            parsed = DXFPrimitiveParser(config).parse(
                path, view_bboxes=self._view_bboxes(), sample_id="fixture"
            )
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
            ).parse(path, view_bboxes=self._view_bboxes())
            self.assertTrue(torch.all(visible_only.sample_features[..., 2] == 1.0))

            with self.assertRaisesRegex(DXFParseError, "unsupported entity TEXT"):
                DXFPrimitiveParser(DXFPrimitiveConfig(strict_entity_types=True)).parse(
                    path, view_bboxes=self._view_bboxes()
                )

    def test_degenerate_entities_are_skipped_or_reported(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            path = Path(directory) / "degenerate.dxf"
            document = ezdxf.new("R2000")
            modelspace = document.modelspace()
            modelspace.add_line((1.0, 1.0), (1.0, 1.0))
            modelspace.add_line((2.0, 2.0), (3.0, 2.0))
            document.saveas(path)
            parsed = DXFPrimitiveParser().parse(
                path,
                view_bboxes=(
                    ViewBBox("front", (0.0, 0.0, 4.0, 4.0)),
                    ViewBBox("top", (10.0, 10.0, 11.0, 11.0)),
                    ViewBBox("right", (12.0, 12.0, 13.0, 13.0)),
                ),
            )
            self.assertEqual(parsed.num_primitives, 1)
            self.assertEqual(parsed.num_skipped_degenerate_entities, 1)
            with self.assertRaisesRegex(DXFParseError, "zero geometric length"):
                DXFPrimitiveParser(DXFPrimitiveConfig(strict_entity_types=True)).parse(
                    path,
                    view_bboxes=(
                        ViewBBox("front", (0.0, 0.0, 4.0, 4.0)),
                        ViewBBox("top", (10.0, 10.0, 11.0, 11.0)),
                        ViewBBox("right", (12.0, 12.0, 13.0, 13.0)),
                    ),
                )

    def test_entities_in_overlapping_view_bboxes_are_skipped_or_reported(
        self,
    ) -> None:
        # A few real drawings have view bounding boxes that clip at a shared
        # corner; an entity whose midpoint lands in the overlap belongs to no
        # single view. Non-strict parsing must skip and count it (so one bad
        # entity never kills a training run), while strict parsing still raises.
        overlapping_bboxes = (
            ViewBBox("front", (0.0, 0.0, 10.0, 10.0)),
            ViewBBox("top", (5.0, 5.0, 15.0, 15.0)),  # clips front's corner
            ViewBBox("right", (20.0, 20.0, 30.0, 30.0)),
        )
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            path = Path(directory) / "overlap.dxf"
            document = ezdxf.new("R2000")
            modelspace = document.modelspace()
            modelspace.add_line((1.0, 1.0), (3.0, 1.0))  # midpoint (2, 1): front only
            modelspace.add_line((7.0, 7.0), (8.0, 8.0))  # midpoint (7.5, 7.5): overlap
            document.saveas(path)

            parsed = DXFPrimitiveParser().parse(
                path, view_bboxes=overlapping_bboxes, sample_id="overlap"
            )
            self.assertEqual(parsed.num_primitives, 1)
            self.assertEqual(parsed.num_skipped_unassigned_view_entities, 1)
            self.assertEqual(parsed.num_skipped_degenerate_entities, 0)
            self.assertEqual(
                parsed.view_direction_ids.tolist(), [VIEW_DIRECTION_TO_ID["front"]]
            )

            with self.assertRaisesRegex(
                DXFParseError, "requires exactly one bbox match"
            ):
                DXFPrimitiveParser(DXFPrimitiveConfig(strict_entity_types=True)).parse(
                    path, view_bboxes=overlapping_bboxes, sample_id="overlap"
                )

    def test_drawing_with_only_out_of_sheet_geometry_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            path = Path(directory) / "invalid_projection.dxf"
            document = ezdxf.new("R2000")
            document.modelspace().add_line((1000.0, 50.0), (1001.0, 50.0))
            document.saveas(path)

            with self.assertRaisesRegex(DXFParseError, "no usable primitives"):
                DXFPrimitiveParser().parse(
                    path,
                    view_bboxes=(
                        ViewBBox("front", (0.0, 0.0, 10.0, 10.0)),
                        ViewBBox("top", (20.0, 20.0, 30.0, 30.0)),
                        ViewBBox("right", (40.0, 40.0, 50.0, 50.0)),
                    ),
                )

    def test_assignment_uses_raw_mean_and_reports_zero_or_multiple_matches(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            path = Path(directory) / "assignment.dxf"
            document = ezdxf.new("R2000")
            document.modelspace().add_line((100.0, 50.0), (110.0, 50.0))
            document.saveas(path)
            disjoint = (
                ViewBBox("front", (99.0, 49.0, 111.0, 51.0)),
                ViewBBox("top", (120.0, 49.0, 130.0, 51.0)),
                ViewBBox("right", (140.0, 49.0, 150.0, 51.0)),
            )
            parsed = DXFPrimitiveParser().parse(
                path, view_bboxes=disjoint, sample_id="raw-coordinate-test"
            )
            self.assertEqual(
                parsed.view_direction_ids.tolist(),
                [VIEW_DIRECTION_TO_ID["front"]],
            )

            # Strict mode raises a detailed diagnostic for zero and multiple
            # matches (audit path). Non-strict mode instead skips and counts the
            # entity -- covered by
            # test_entities_in_overlapping_view_bboxes_are_skipped_or_reported.
            strict = DXFPrimitiveConfig(strict_entity_types=True)
            no_match = (
                ViewBBox("front", (0.0, 0.0, 1.0, 1.0)),
                ViewBBox("top", (2.0, 2.0, 3.0, 3.0)),
                ViewBBox("right", (4.0, 4.0, 5.0, 5.0)),
            )
            with self.assertRaisesRegex(
                DXFParseError,
                "sample_id='zero-match'.*entity_handle=.*representative_point",
            ):
                DXFPrimitiveParser(strict).parse(
                    path, view_bboxes=no_match, sample_id="zero-match"
                )

            ambiguous = (
                ViewBBox("front", (99.0, 49.0, 111.0, 51.0)),
                ViewBBox("top", (100.0, 49.0, 110.0, 51.0)),
                ViewBBox("right", (140.0, 49.0, 150.0, 51.0)),
            )
            with self.assertRaisesRegex(
                DXFParseError, "matched_directions=.*front.*top"
            ):
                DXFPrimitiveParser(strict).parse(
                    path, view_bboxes=ambiguous, sample_id="multiple-match"
                )

    def test_small_bbox_rounding_tolerance_accepts_boundary_mean(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            path = Path(directory) / "rounded.dxf"
            document = ezdxf.new("R2000")
            document.modelspace().add_line((10.0, 20.07), (12.0, 20.07))
            document.saveas(path)
            parsed = DXFPrimitiveParser(
                DXFPrimitiveConfig(view_bbox_tolerance_mm=0.1)
            ).parse(
                path,
                view_bboxes=(
                    ViewBBox("front", (9.0, 19.0, 13.0, 20.0)),
                    ViewBBox("top", (30.0, 30.0, 31.0, 31.0)),
                    ViewBBox("right", (40.0, 40.0, 41.0, 41.0)),
                ),
            )
            self.assertEqual(parsed.view_direction_ids.tolist(), [0])


if __name__ == "__main__":
    unittest.main()
