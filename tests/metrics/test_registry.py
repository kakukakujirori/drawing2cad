import subprocess
import sys
import unittest

from src.metrics import (
    BoundingBoxMetric,
    METRIC_REGISTRY,
    VoxelIoUMetric,
)
from src.metrics.base import CADMetric
from src.metrics.registry import (
    build_metrics,
    empty_metric_row,
    register_metric,
    required_artifacts,
)


class MetricRegistryTest(unittest.TestCase):
    def test_every_family_is_registered_under_its_class_name(self) -> None:
        for name in (
            "CadExecutionMetric",
            "VoxelIoUMetric",
            "BoundingBoxMetric",
            "ChamferMetric",
            "CADGenBenchScoreMetric",
            "ECCVChallengeMetric",
        ):
            self.assertIn(name, METRIC_REGISTRY)
            self.assertEqual(METRIC_REGISTRY[name].name, name)

    def test_builds_from_class_names_and_mappings(self) -> None:
        metrics = build_metrics(
            ["CadExecutionMetric", {"name": "VoxelIoUMetric", "resolution": 32}]
        )
        self.assertEqual(len(metrics), 2)
        voxel = metrics[1]
        assert isinstance(voxel, VoxelIoUMetric)
        self.assertEqual(voxel.resolution, 32)

    def test_none_selects_nothing(self) -> None:
        self.assertEqual(build_metrics(None), ())

    def test_unknown_name_lists_the_known_ones(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown metric 'VoxelIoU'"):
            build_metrics(["VoxelIoU"])

    def test_mapping_without_a_name_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "'name' key"):
            build_metrics([{"resolution": 32}])

    def test_bad_parameter_names_the_metric(self) -> None:
        with self.assertRaisesRegex(TypeError, "VoxelIoUMetric"):
            build_metrics([{"name": "VoxelIoUMetric", "resolutionn": 32}])

    def test_a_list_is_required(self) -> None:
        with self.assertRaisesRegex(TypeError, "list of metric entries"):
            build_metrics("VoxelIoUMetric")

    def test_duplicate_family_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "listed more than once"):
            build_metrics(["VoxelIoUMetric", "VoxelIoUMetric"])

    def test_colliding_row_keys_are_rejected(self) -> None:
        class _ShadowIoUMetric(CADMetric):
            requires = frozenset()
            row_keys = ("iou",)

            def score(self, sample):
                return {}

            def reduce(self, rows, *, prefix):
                return {}

        try:
            register_metric(_ShadowIoUMetric)
            with self.assertRaisesRegex(ValueError, "both write row key 'iou'"):
                build_metrics(["VoxelIoUMetric", "_ShadowIoUMetric"])
        finally:
            METRIC_REGISTRY.pop("_ShadowIoUMetric", None)

    def test_unknown_artifact_is_rejected_at_registration(self) -> None:
        class _BadArtifactMetric(CADMetric):
            requires = frozenset({"pred_pointcloud"})
            row_keys = ()

            def score(self, sample):
                return {}

            def reduce(self, rows, *, prefix):
                return {}

        try:
            with self.assertRaisesRegex(ValueError, "unknown artifact"):
                register_metric(_BadArtifactMetric)
        finally:
            METRIC_REGISTRY.pop("_BadArtifactMetric", None)

    def test_required_artifacts_is_the_union(self) -> None:
        metrics = build_metrics(["VoxelIoUMetric", "ECCVChallengeMetric"])
        self.assertEqual(
            required_artifacts(metrics),
            frozenset({"pred_mesh", "gt_mesh", "pred_step", "gt_step"}),
        )
        self.assertEqual(
            required_artifacts(build_metrics(["CadExecutionMetric"])), frozenset()
        )

    def test_registering_every_family_loads_no_cad_kernel(self) -> None:
        """The training process must stay free of CAD kernels.

        Registration happens by importing every family, so a stray module-level
        ``import cadquery`` there would pull OCC into the trainer -- where a
        native fault has no containment. Checked in a fresh interpreter because
        this suite's other tests deliberately load those kernels.
        """

        script = (
            "import sys; import src.metrics; import src.evaluation; "
            "loaded = sorted(m for m in sys.modules "
            "if m.split('.')[0] in {'OCC', 'OCP', 'cadquery', 'cadgenbench'}); "
            "print(loaded)"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(completed.stdout.strip(), "[]")

    def test_empty_row_covers_every_selected_column(self) -> None:
        row = empty_metric_row([VoxelIoUMetric(), BoundingBoxMetric()])
        self.assertEqual(
            sorted(row),
            [
                "bbox_error_mm",
                "bbox_pred_mm",
                "bbox_target_mm",
                "iou",
                "max_bbox_error_mm",
                "max_bbox_error_relative",
            ],
        )
        self.assertTrue(all(value is None for value in row.values()))


if __name__ == "__main__":
    unittest.main()
