import unittest

from src.metrics.cad import cad_error_histogram, cad_execution_metrics


class CadMetricsTest(unittest.TestCase):
    def test_rates_include_failed_rows(self) -> None:
        rows = [
            {
                "exec_ok": True,
                "has_result": True,
                "valid": True,
                "error": None,
            },
            {
                "exec_ok": True,
                "has_result": False,
                "valid": False,
                "error": "no_result",
            },
            {
                "exec_ok": False,
                "has_result": False,
                "valid": False,
                "error": "exec:SyntaxError: bad",
            },
        ]
        metrics = cad_execution_metrics(rows)
        self.assertEqual(metrics["val/num_samples"], 3)
        self.assertAlmostEqual(metrics["val/exec_ok_rate"], 2 / 3)
        self.assertAlmostEqual(metrics["val/has_result_rate"], 1 / 3)
        self.assertAlmostEqual(metrics["val/valid_rate"], 1 / 3)
        self.assertEqual(
            cad_error_histogram(rows),
            {"exec": 1, "no_result": 1, "ok": 1},
        )

    def test_empty_rates_are_stable(self) -> None:
        self.assertEqual(
            cad_execution_metrics([], prefix=""),
            {
                "num_samples": 0,
                "exec_ok_rate": 0.0,
                "has_result_rate": 0.0,
                "valid_rate": 0.0,
            },
        )


if __name__ == "__main__":
    unittest.main()
