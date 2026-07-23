import importlib.util
import unittest

import numpy as np

from src.metrics.geometry import (
    aggregate_geometry_metrics,
    align_meshes,
    bbox_dimension_error_mm,
    max_bbox_error_relative,
    normalized_voxel_iou,
    surface_chamfer_distance,
    symmetric_chamfer_distance,
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
    def test_align_meshes_centers_and_optionally_normalizes(self) -> None:
        import trimesh

        target = trimesh.creation.box(extents=[2.0, 3.0, 4.0])
        predicted = target.copy()
        predicted.apply_translation([30.0, -12.0, 8.0])

        for normalize in (False, True):
            pred, gt = align_meshes(predicted, target, normalize_scale=normalize)
            for mesh in (pred, gt):
                center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
                np.testing.assert_allclose(center, [0.0, 0.0, 0.0], atol=1e-9)
                if normalize:
                    self.assertAlmostEqual(float(np.max(mesh.extents)), 1.0)

    @unittest.skipUnless(HAS_TRIMESH, "trimesh is not installed")
    def test_voxel_iou_ignores_translation_and_scale_but_not_shape(self) -> None:
        import trimesh

        target = trimesh.creation.box(extents=[2.0, 3.0, 4.0])

        translated = target.copy()
        translated.apply_translation([30.0, -12.0, 8.0])
        self.assertGreater(normalized_voxel_iou(translated, target, resolution=24), 0.99)

        # Uniformly rescaled: the same shape, so the score must not drop. This
        # is the assertion that reverses under the old scale-preserving metric.
        half_size = trimesh.creation.box(extents=[1.0, 1.5, 2.0])
        self.assertGreater(normalized_voxel_iou(half_size, target, resolution=24), 0.99)

        wrong_shape = trimesh.creation.box(extents=[4.0, 4.0, 4.0])
        self.assertLess(normalized_voxel_iou(wrong_shape, target, resolution=24), 0.6)

    @unittest.skipUnless(HAS_TRIMESH, "trimesh is not installed")
    def test_voxel_grid_is_bounded_regardless_of_prediction_size(self) -> None:
        import trimesh

        from src.metrics.geometry import _voxel_indices

        # A normalized target against a millimetre-scale prediction: the exact
        # pairing that exhausted host memory when the pitch came from the target
        # and was then applied to the prediction. Normalizing first bounds both
        # grids by (resolution + 1) ** 3, so assert the property, not a threshold.
        target = trimesh.creation.box(extents=[1.8, 1.6, 1.6])
        millimetre_scale = trimesh.creation.box(extents=[80.0, 80.0, 80.0])

        resolution = 64
        score = normalized_voxel_iou(millimetre_scale, target, resolution=resolution)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

        pred, gt = align_meshes(millimetre_scale, target)
        for mesh in (pred, gt):
            cells = _voxel_indices(mesh, 1.0 / resolution)
            self.assertLessEqual(len(cells), (resolution + 1) ** 3)

    @unittest.skipUnless(HAS_TRIMESH, "trimesh is not installed")
    def test_relative_bbox_error_is_scaled_by_the_target(self) -> None:
        import trimesh

        target = trimesh.creation.box(extents=[2.0, 3.0, 4.0])
        predicted = trimesh.creation.box(extents=[2.0, 3.0, 5.0])
        # One millimetre of error against a 4 mm longest edge.
        self.assertAlmostEqual(max_bbox_error_relative(predicted, target), 0.25)

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
                {
                    "valid": True,
                    "iou": 1.0,
                    "max_bbox_error_mm": 0.0,
                    "max_bbox_error_relative": 0.0,
                },
                {
                    "valid": True,
                    "iou": 0.5,
                    "max_bbox_error_mm": 2.0,
                    "max_bbox_error_relative": 0.5,
                },
                {
                    "valid": False,
                    "iou": None,
                    "max_bbox_error_mm": None,
                    "max_bbox_error_relative": None,
                },
            ]
        )
        self.assertEqual(metrics["val/iou_scored_n"], 2)
        self.assertAlmostEqual(metrics["val/mean_iou_including_failures"], 0.5)
        self.assertAlmostEqual(metrics["val/mean_iou_valid_only"], 0.75)
        self.assertAlmostEqual(metrics["val/median_iou"], 0.5)
        self.assertAlmostEqual(metrics["val/mean_max_bbox_error_mm"], 1.0)
        self.assertAlmostEqual(metrics["val/mean_max_bbox_error_relative"], 0.25)
        self.assertNotIn("val/mean_iou_normalized", metrics)


if __name__ == "__main__":
    unittest.main()
