import json
from pathlib import Path
import tempfile
import unittest

import torch

from src.utils.logging import ExperimentLogger, JSONLMetricLogger, normalize_metrics


class MetricLoggingTest(unittest.TestCase):
    def test_jsonl_logger_flushes_scalar_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metrics.jsonl"
            logger = JSONLMetricLogger(path)
            logger.log(
                {"train/loss": torch.tensor(1.25), "train/step_kind": "optimizer"},
                step=3,
            )
            record = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(record["step"], 3)
            self.assertEqual(record["train/loss"], 1.25)
            self.assertIn("timestamp", record)
            logger.finish()

    def test_invalid_metrics_are_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, "must be scalar"):
            normalize_metrics({"train/vector": torch.ones(2)})
        with self.assertRaisesRegex(ValueError, "reserved"):
            normalize_metrics({"step": 1})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metrics.jsonl"
            logger = JSONLMetricLogger(path)
            logger.log({"val/valid_only": float("nan")}, step=0)
            logger.finish()
            self.assertIsNone(json.loads(path.read_text())["val/valid_only"])

    def test_experiment_logger_is_local_first_and_rank_aware(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {
                "jsonl": {"enabled": True, "filename": "custom.jsonl"},
                "wandb": {"enabled": False},
            }
            logger = ExperimentLogger.from_config(root, config)
            logger.log({"val/loss": 2.0}, step=4)
            logger.finish()
            self.assertTrue((root / "custom.jsonl").is_file())

            worker = ExperimentLogger.from_config(
                root / "worker", config, is_main_process=False
            )
            worker.log({"ignored": 1}, step=0)
            worker.finish()
            self.assertFalse((root / "worker").exists())


if __name__ == "__main__":
    unittest.main()
