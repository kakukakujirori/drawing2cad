import json
from pathlib import Path
import tempfile
import unittest

from src.utils.checkpoint import CheckpointManager


class CheckpointManagerTest(unittest.TestCase):
    @staticmethod
    def _save_value(value: str):
        def save(path: Path) -> None:
            (path / "state.txt").write_text(value, encoding="utf-8")

        return save

    def test_latest_and_max_topk_retention(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "checkpoints"
            manager = CheckpointManager(root, monitor="val/iou", mode="max", top_k=2)
            for step, value in [(1, 0.2), (2, 0.8), (3, 0.5), (4, 0.1)]:
                manager.save(
                    step=step,
                    metrics={"val/iou": value},
                    save_state=self._save_value(f"state-{step}"),
                )
            self.assertEqual(
                [(entry.step, entry.value) for entry in manager.entries],
                [(2, 0.8), (3, 0.5)],
            )
            self.assertEqual((root / "latest" / "state.txt").read_text(), "state-4")
            self.assertEqual(len(list((root / "topk").iterdir())), 2)
            index = json.loads((root / "topk.json").read_text(encoding="utf-8"))
            self.assertEqual([entry["step"] for entry in index["checkpoints"]], [2, 3])
            self.assertFalse(any(root.glob(".topk.json.tmp-*")))

            reloaded = CheckpointManager(root, monitor="val/iou", mode="max", top_k=2)
            self.assertEqual(reloaded.entries, manager.entries)

    def test_min_mode_and_latest_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "checkpoints"
            manager = CheckpointManager(root, monitor="val/loss", mode="min", top_k=1)
            manager.save(
                step=5,
                metrics={"val/loss": 3.0},
                save_state=self._save_value("optimizer-and-rng-state"),
                extra_metadata={"epoch": 2},
            )
            loaded: list[str] = []

            def load(path: Path) -> None:
                loaded.append((path / "state.txt").read_text(encoding="utf-8"))

            metadata = manager.load_latest(load)
            self.assertEqual(loaded, ["optimizer-and-rng-state"])
            self.assertEqual(metadata["step"], 5)
            self.assertEqual(metadata["epoch"], 2)

    def test_monitor_must_be_present_finite_and_scalar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = CheckpointManager(
                Path(temporary), monitor="val/metric", mode="max", top_k=0
            )
            with self.assertRaisesRegex(KeyError, "absent"):
                manager.save(step=0, metrics={}, save_state=self._save_value("x"))
            with self.assertRaisesRegex(ValueError, "finite"):
                manager.save(
                    step=0,
                    metrics={"val/metric": float("nan")},
                    save_state=self._save_value("x"),
                )
            with self.assertRaisesRegex(TypeError, "numeric scalar"):
                manager.save(
                    step=0,
                    metrics={"val/metric": "bad"},
                    save_state=self._save_value("x"),
                )
            with self.assertRaisesRegex(ValueError, "reserved keys"):
                manager.save(
                    step=0,
                    metrics={"val/metric": 0.5},
                    save_state=self._save_value("x"),
                    extra_metadata={"step": 99},
                )

    def test_failed_save_does_not_replace_previous_latest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manager = CheckpointManager(root, monitor="val/loss", mode="min", top_k=0)
            manager.save(
                step=1,
                metrics={"val/loss": 1.0},
                save_state=self._save_value("good"),
            )

            def fail(path: Path) -> None:
                (path / "partial.txt").write_text("partial")
                raise RuntimeError("save failed")

            with self.assertRaisesRegex(RuntimeError, "save failed"):
                manager.save(step=2, metrics={"val/loss": 0.5}, save_state=fail)
            self.assertEqual((root / "latest" / "state.txt").read_text(), "good")

    def test_non_main_manager_is_a_noop(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "worker"
            manager = CheckpointManager(
                root,
                monitor="val/loss",
                mode="min",
                top_k=1,
                is_main_process=False,
            )
            kept = manager.save(
                step=0,
                metrics={},
                save_state=lambda _: self.fail("save callback must not run"),
            )
            self.assertFalse(kept)
            self.assertFalse(root.exists())

    def test_accelerate_style_barrier_save_runs_callback_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            barriers: list[None] = []
            manager = CheckpointManager.from_config(
                Path(temporary),
                {
                    "monitor": "val/loss",
                    "mode": "min",
                    "top_k": 1,
                    "root_dirname": "states",
                },
                barrier=lambda: barriers.append(None),
            )
            calls: list[Path] = []

            def save(path: Path) -> None:
                calls.append(path)
                (path / "accelerate-state.txt").write_text("complete")

            manager.save(step=7, metrics={"val/loss": 0.25}, save_state=save)
            self.assertEqual(len(calls), 1)
            self.assertEqual(len(barriers), 3)
            self.assertEqual(
                (Path(temporary) / "states/latest/accelerate-state.txt").read_text(),
                "complete",
            )


if __name__ == "__main__":
    unittest.main()
