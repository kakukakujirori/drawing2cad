from __future__ import annotations

from pathlib import Path
import unittest

import torch

from src.utils.metric_router import LoggedMetric, MetricRouter


class _Logger:
    def __init__(self) -> None:
        self.records: list[tuple[dict[str, object], int]] = []
        self.finished = False

    def log(self, metrics, *, step: int) -> None:
        self.records.append((dict(metrics), step))

    def log_artifact(
        self, path: str | Path, *, name: str | None = None, kind: str = "artifact"
    ) -> None:
        del path, name, kind

    def finish(self) -> None:
        self.finished = True


class _Progress:
    def __init__(self) -> None:
        self.metrics: dict[str, str] = {}

    def update_metrics(self, metrics) -> None:
        self.metrics.update(metrics)


class MetricRouterTest(unittest.TestCase):
    def test_logs_every_metric_but_displays_only_prog_bar_metrics(self) -> None:
        logger = _Logger()
        progress = _Progress()
        router = MetricRouter(logger, progress)

        router.log(
            {
                "train/loss": LoggedMetric(
                    torch.tensor(1.23456),
                    prog_bar=True,
                    display_name="loss",
                    format_spec=".3f",
                ),
                "train/grad_norm": 2.0,
            },
            step=7,
        )

        self.assertEqual(logger.records[0][1], 7)
        self.assertEqual(set(logger.records[0][0]), {"train/loss", "train/grad_norm"})
        self.assertEqual(progress.metrics, {"loss": "1.235"})

    def test_rejects_non_scalar_progress_values_and_forwards_finish(self) -> None:
        logger = _Logger()
        router = MetricRouter(logger, _Progress())
        with self.assertRaisesRegex(TypeError, "must be scalar"):
            router.log(
                {"train/vector": LoggedMetric(torch.ones(2), prog_bar=True)},
                step=1,
            )
        router.finish()
        self.assertTrue(logger.finished)


if __name__ == "__main__":
    unittest.main()
