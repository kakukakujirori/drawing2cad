import tempfile
import unittest

from src.evaluation.scoring import score_sample
from src.metrics.base import CADMetric
from src.metrics.registry import METRIC_REGISTRY, register_metric


@register_metric
class _ConstantMetric(CADMetric):
    """Writes one column without touching any artifact."""

    requires = frozenset()
    row_keys = ("constant",)

    def score(self, sample):
        del sample
        return {"constant": 1.0}

    def reduce(self, rows, *, prefix):
        del rows, prefix
        return {}


@register_metric
class _RaisingMetric(CADMetric):
    requires = frozenset()
    row_keys = ("never_written",)

    def score(self, sample):
        raise RuntimeError("deliberate metric failure")

    def reduce(self, rows, *, prefix):
        del rows, prefix
        return {}


@register_metric
class _CrashingMetric(CADMetric):
    """Kills its own process, the way a native CAD kernel fault would."""

    requires = frozenset()
    row_keys = ("never_written_either",)

    def score(self, sample):
        import os

        os._exit(9)

    def reduce(self, rows, *, prefix):
        del rows, prefix
        return {}


@register_metric
class _HangingMetric(CADMetric):
    requires = frozenset()
    row_keys = ("never_written_at_all",)

    def score(self, sample):
        import time

        time.sleep(30)
        return {}

    def reduce(self, rows, *, prefix):
        del rows, prefix
        return {}


class ScoringTest(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        for name in (
            "_ConstantMetric",
            "_RaisingMetric",
            "_CrashingMetric",
            "_HangingMetric",
        ):
            METRIC_REGISTRY.pop(name, None)

    def test_no_metrics_costs_no_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            result = score_sample("s", workdir=workdir, metrics=())
        self.assertEqual(result.columns, {})
        self.assertIsNone(result.error)

    def test_columns_come_back_from_the_child(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            result = score_sample(
                "s", workdir=workdir, metrics=(_ConstantMetric(),), timeout_s=120.0
            )
        self.assertEqual(result.columns, {"constant": 1.0})
        self.assertEqual(result.metric_errors, {})
        self.assertIsNone(result.error)

    def test_one_failing_metric_does_not_cost_the_others(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            result = score_sample(
                "s",
                workdir=workdir,
                metrics=(_RaisingMetric(), _ConstantMetric()),
                timeout_s=120.0,
            )
        self.assertEqual(result.columns, {"constant": 1.0})
        self.assertIn("_RaisingMetric", result.metric_errors)
        self.assertIn(
            "deliberate metric failure", result.metric_errors["_RaisingMetric"]
        )
        self.assertIsNone(result.error)

    def test_a_crashed_child_is_reported_not_raised(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            result = score_sample(
                "s", workdir=workdir, metrics=(_CrashingMetric(),), timeout_s=120.0
            )
        self.assertEqual(result.columns, {})
        self.assertIsNotNone(result.error)
        assert result.error is not None
        self.assertTrue(result.error.startswith("process_exit:"), result.error)

    def test_a_hung_child_times_out(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            result = score_sample(
                "s", workdir=workdir, metrics=(_HangingMetric(),), timeout_s=3.0
            )
        self.assertEqual(result.error, "timeout")


if __name__ == "__main__":
    unittest.main()
