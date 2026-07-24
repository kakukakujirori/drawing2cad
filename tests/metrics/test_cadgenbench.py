"""Tests for the CADGenBench CAD Score family.

The score definitions come from the installed benchmark, so these tests check
the wiring: that a candidate identical to the target scores ~1, that the
normalization option is what makes a scale-shifted candidate scoreable, and
that the aggregation counts unscored samples as zeros.
"""

import importlib.util
from pathlib import Path
import tempfile
import unittest

from src.evaluation.scoring import score_sample
from src.metrics.cadgenbench import CADGenBenchScoreMetric


try:
    import cadquery as cq

    HAS_CADQUERY = True
except Exception:
    cq = None
    HAS_CADQUERY = False

HAS_CADGENBENCH = importlib.util.find_spec("cadgenbench") is not None


@unittest.skipUnless(
    HAS_CADQUERY and HAS_CADGENBENCH, "CadQuery/cadgenbench unavailable"
)
class CADGenBenchScoreMetricTest(unittest.TestCase):
    def _score(self, predicted: Path, target: Path, **kwargs) -> dict:
        # Through the isolated worker: cadgenbench meshes with OCP, which this
        # suite must not load into the test process.
        with tempfile.TemporaryDirectory() as workdir:
            result = score_sample(
                "box",
                workdir=workdir,
                metrics=(CADGenBenchScoreMetric(**kwargs),),
                pred_step_path=predicted,
                gt_step_path=target,
                timeout_s=600.0,
            )
        self.assertIsNone(result.error)
        self.assertEqual(result.metric_errors, {})
        return result.columns

    # The benchmark's rigid alignment samples points without a fixed seed, and on
    # a symmetric solid it can settle into a visibly worse pose from one run to
    # the next. These tests therefore feed already-aligned inputs (``align`` off)
    # so a score can be asserted exactly; ``test_alignment_path_runs`` covers the
    # default path without pinning its value.

    def test_identical_solid_scores_one(self) -> None:
        assert cq is not None
        with tempfile.TemporaryDirectory() as temporary:
            step = Path(temporary) / "box.step"
            cq.exporters.export(cq.Workplane("XY").box(20, 30, 40), str(step))
            row = self._score(step, step, normalize_to_gt_bbox=False, align=False)
        self.assertTrue(row["cadgenbench_is_valid"])
        self.assertGreater(row["cad_score"], 0.99)
        self.assertGreater(row["shape_volume_iou"], 0.99)
        self.assertAlmostEqual(row["topology_match_score"], 1.0, places=6)

    def test_alignment_path_runs(self) -> None:
        assert cq is not None
        with tempfile.TemporaryDirectory() as temporary:
            step = Path(temporary) / "box.step"
            cq.exporters.export(cq.Workplane("XY").box(20, 30, 40), str(step))
            row = self._score(step, step, normalize_to_gt_bbox=False, align=True)
        self.assertIsNotNone(row["cadgenbench_alignment_rmse"])
        self.assertGreaterEqual(row["cad_score"], 0.0)
        self.assertLessEqual(row["cad_score"], 1.0)

    def test_normalization_recovers_a_rescaled_prediction(self) -> None:
        assert cq is not None
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            target = temporary / "target.step"
            predicted = temporary / "pred.step"
            solid = cq.Workplane("XY").box(20, 30, 40)
            cq.exporters.export(solid, str(target))
            cq.exporters.export(solid.val().scale(7.0), str(predicted))
            normalized = self._score(
                predicted, target, normalize_to_gt_bbox=True, align=False
            )
            raw = self._score(
                predicted, target, normalize_to_gt_bbox=False, align=False
            )
        self.assertAlmostEqual(normalized["cadgenbench_normalization_scale"], 1 / 7.0)
        self.assertGreater(normalized["cad_score"], 0.99)
        # Shape axes collapse at the wrong scale -- the smaller box merely sits
        # inside the larger one -- and only topology survives, which is exactly
        # the 1/3 weight the generation axes give it once interface is absent.
        self.assertLess(raw["shape_volume_iou"], 0.01)
        self.assertAlmostEqual(raw["cad_score"], 1 / 3, places=2)

    def test_reduce_scores_unscored_samples_as_zero(self) -> None:
        rows = [
            {
                "cad_score": 0.8,
                "shape_surface_distance_f1": 0.9,
                "shape_volume_iou": 0.7,
                "shape_similarity_score": 0.8,
                "topology_match_score": 1.0,
            },
            {key: None for key in CADGenBenchScoreMetric.row_keys},
        ]
        metrics = CADGenBenchScoreMetric().reduce(rows, prefix="val")
        self.assertEqual(metrics["val/cad_score_scored_n"], 1)
        self.assertAlmostEqual(metrics["val/mean_cad_score"], 0.4)
        self.assertAlmostEqual(metrics["val/mean_cad_score_scored_only"], 0.8)
        self.assertAlmostEqual(metrics["val/mean_shape_volume_iou"], 0.35)


if __name__ == "__main__":
    unittest.main()
