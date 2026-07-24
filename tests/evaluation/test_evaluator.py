from pathlib import Path
import tempfile
import unittest

from src.evaluation.evaluator import (
    EvaluationConfig,
    EvaluationItem,
    aggregate_evaluation_metrics,
    evaluate_prediction,
    evaluation_error_histogram,
)
from src.metrics import (
    BoundingBoxMetric,
    CadExecutionMetric,
    VoxelIoUMetric,
)


try:
    import cadquery as cq

    HAS_CADQUERY = True
except Exception:
    cq = None
    HAS_CADQUERY = False


def _mesh_metrics(resolution: int) -> tuple:
    return (
        CadExecutionMetric(),
        VoxelIoUMetric(resolution=resolution),
        BoundingBoxMetric(),
    )


class EvaluatorTest(unittest.TestCase):
    def test_aggregate_metrics_exposes_checkpoint_monitor(self) -> None:
        rows = [
            {
                "exec_ok": True,
                "has_result": True,
                "valid": True,
                "iou": 0.8,
                "max_bbox_error_mm": 1.0,
                "max_bbox_error_relative": 0.25,
                "error": None,
            },
            {
                "exec_ok": False,
                "has_result": False,
                "valid": False,
                "iou": None,
                "max_bbox_error_mm": None,
                "max_bbox_error_relative": None,
                "error": "timeout",
            },
        ]
        metrics = aggregate_evaluation_metrics(rows, metrics=_mesh_metrics(64))
        self.assertEqual(metrics["val/mean_iou_including_failures"], 0.4)
        self.assertEqual(metrics["val/mean_iou_valid_only"], 0.8)
        self.assertEqual(metrics["val/mean_max_bbox_error_relative"], 0.25)
        self.assertEqual(metrics["val/valid_rate"], 0.5)
        self.assertEqual(evaluation_error_histogram(rows), {"ok": 1, "timeout": 1})

    def test_unselected_family_contributes_no_keys(self) -> None:
        rows = [{"exec_ok": True, "has_result": True, "valid": True, "iou": 0.5}]
        metrics = aggregate_evaluation_metrics(rows, metrics=(VoxelIoUMetric(),))
        self.assertIn("val/mean_iou_including_failures", metrics)
        self.assertNotIn("val/valid_rate", metrics)
        self.assertNotIn("val/mean_max_bbox_error_mm", metrics)

    def test_missing_target_is_a_failed_row(self) -> None:
        row = evaluate_prediction(
            EvaluationItem("missing", "result = None", "/not/a/real/target.step"),
            config=EvaluationConfig(metrics=_mesh_metrics(16)),
        )
        self.assertFalse(row["exec_ok"])
        self.assertEqual(row["error"], "missing_target_step")
        # The row still carries every selected family's columns, so a failure
        # aggregates identically to a scored sample.
        self.assertIn("iou", row)
        self.assertIsNone(row["iou"])

    def test_item_requires_exactly_one_target_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one"):
            EvaluationItem("bad", "result = None")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            EvaluationItem(
                "bad",
                "result = None",
                target_step_path="target.step",
                target_code="result = None",
            )

    @unittest.skipUnless(HAS_CADQUERY, "CadQuery cannot be imported")
    def test_identical_box_integration(self) -> None:
        assert cq is not None
        with tempfile.TemporaryDirectory() as temporary:
            target_path = Path(temporary) / "target.step"
            target = cq.Workplane("XY").box(2, 3, 4)
            cq.exporters.export(target, str(target_path))
            row = evaluate_prediction(
                EvaluationItem(
                    "box",
                    "result = cq.Workplane('XY').box(2, 3, 4).translate((10, 20, 30))",
                    target_path,
                ),
                config=EvaluationConfig(timeout_s=30.0, metrics=_mesh_metrics(20)),
            )
            self.assertTrue(row["valid"], row["error"])
            self.assertGreater(float(row["iou"]), 0.99)
            self.assertLess(float(row["max_bbox_error_mm"]), 1e-6)

    @unittest.skipUnless(HAS_CADQUERY, "CadQuery cannot be imported")
    def test_target_cadquery_code_integration(self) -> None:
        row = evaluate_prediction(
            EvaluationItem(
                "code-target",
                "result = cq.Workplane('XY').box(2, 3, 4)",
                target_code="result = cq.Workplane('XY').box(2, 3, 4)",
            ),
            config=EvaluationConfig(timeout_s=30.0, metrics=_mesh_metrics(16)),
        )
        self.assertTrue(row["valid"], row["error"])
        self.assertGreater(float(row["iou"]), 0.99)


if __name__ == "__main__":
    unittest.main()
