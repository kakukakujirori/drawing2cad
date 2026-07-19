import importlib.util
import unittest

import numpy as np

from src.metrics.geometry import (
    aggregate_geometry_metrics,
    bbox_dimension_error_mm,
    surface_chamfer_distance,
    symmetric_chamfer_distance,
    translation_aligned_voxel_iou,
)


HAS_TRIMESH = importlib.util.find_spec("trimesh") is not None


class GeometryMetricsTest(unittest.TestCase):
    def test_bbox_dimension_error_is_orientation_independent_by_default(self) -> None:
        error = bbox_dimension_error_mm([2.0, 3.0, 1.0], [1.0, 2.0, 4.5])
        np.testing.assert_allclose(error, [0.0, 0.0, 1.5])

    def test_symmetric_chamfer_for_point_sets(self) -> None:
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        self.assertEqual(symmetric_chamfer_distance(points, points), 0.0)
        shifted = points + np.array([0.0, 1.0, 0.0])
        self.assertAlmostEqual(symmetric_chamfer_distance(points, shifted), 2.0)

    @unittest.skipUnless(HAS_TRIMESH, "trimesh is not installed")
    def test_voxel_iou_is_translation_aligned_but_scale_preserving(self) -> None:
        import trimesh

        target = trimesh.creation.box(extents=[2.0, 3.0, 4.0])
        translated = target.copy()
        translated.apply_translation([30.0, -12.0, 8.0])
        self.assertGreater(
            translation_aligned_voxel_iou(translated, target, resolution=24),
            0.99,
        )

        wrong_scale = trimesh.creation.box(extents=[1.0, 1.5, 2.0])
        self.assertLess(
            translation_aligned_voxel_iou(wrong_scale, target, resolution=24),
            0.3,
        )

    @unittest.skipUnless(HAS_TRIMESH, "trimesh is not installed")
    def test_surface_chamfer_is_translation_aligned(self) -> None:
        import trimesh

        target = trimesh.creation.box(extents=[2.0, 3.0, 4.0])
        translated = target.copy()
        translated.apply_translation([10.0, 20.0, 30.0])
        self.assertLess(
            surface_chamfer_distance(translated, target, num_points=256, seed=7),
            1e-20,
        )

    def test_aggregation_scores_failures_as_zero_iou(self) -> None:
        metrics = aggregate_geometry_metrics(
            [
                {"valid": True, "iou": 1.0, "max_bbox_error_mm": 0.0},
                {"valid": True, "iou": 0.5, "max_bbox_error_mm": 2.0},
                {"valid": False, "iou": None, "max_bbox_error_mm": None},
            ]
        )
        self.assertEqual(metrics["val/iou_scored_n"], 2)
        self.assertAlmostEqual(metrics["val/mean_iou_including_failures"], 0.5)
        self.assertAlmostEqual(metrics["val/mean_iou_valid_only"], 0.75)
        self.assertAlmostEqual(metrics["val/median_iou"], 0.5)
        self.assertAlmostEqual(metrics["val/mean_max_bbox_error_mm"], 1.0)


if __name__ == "__main__":
    unittest.main()
