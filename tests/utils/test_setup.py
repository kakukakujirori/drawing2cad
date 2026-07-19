from datetime import datetime, timezone
import json
from pathlib import Path
import random
import tempfile
import unittest

import torch
import yaml

from src.utils.setup import seed_everything, setup_run


class SetupRunTest(unittest.TestCase):
    def _config(self, root: Path) -> dict:
        return {
            "paths": {"project_root": str(root), "log_root": str(root / "logs")},
            "model": {"model_name_or_path": "local/model"},
            "data": {"train_root": "train-data", "val_root": "val-data"},
            "logger": {"jsonl": {"filename": "scalars.jsonl"}},
            "checkpoint": {"root_dirname": "states"},
            "evaluation": {
                "generation_enabled": True,
                "predictions_dirname": "predictions",
            },
        }

    def test_writes_resolved_config_metadata_and_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "logs" / "train_sft" / "run"
            context = setup_run(
                self._config(root),
                output_dir=output,
                timestamp=datetime(2026, 7, 19, tzinfo=timezone.utc),
            )
            self.assertEqual(context.run_dir, output.resolve())
            self.assertEqual(context.metrics_path.name, "scalars.jsonl")
            self.assertTrue(context.checkpoints_dir.is_dir())
            self.assertTrue(context.predictions_dir.is_dir())
            saved_config = yaml.safe_load(
                context.config_path.read_text(encoding="utf-8")
            )
            self.assertEqual(saved_config["model"]["model_name_or_path"], "local/model")
            metadata = json.loads(context.metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["base_checkpoint"], "local/model")
            self.assertEqual(metadata["dataset_roots"]["validation"], "val-data")
            self.assertEqual(metadata["world_size"], 1)
            self.assertNotIn("environment", metadata)

    def test_hydra_precreated_empty_directory_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "already-created"
            output.mkdir()
            context = setup_run(self._config(root), output_dir=output)
            self.assertTrue(context.config_path.is_file())
            with self.assertRaises(FileExistsError):
                setup_run(self._config(root), output_dir=output)

    def test_resume_keeps_directory_and_updates_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "run"
            setup_run(self._config(root), output_dir=output)
            setup_run(self._config(root), output_dir=output, resume=True)
            metadata = json.loads((output / "run_metadata.json").read_text())
            self.assertEqual(metadata["resume_count"], 1)
            self.assertIn("last_resumed_at", metadata)

    def test_non_main_process_performs_no_filesystem_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "worker-run"
            context = setup_run(
                self._config(root),
                output_dir=output,
                is_main_process=False,
                rank=1,
                world_size=2,
            )
            self.assertFalse(output.exists())
            self.assertEqual(context.rank, 1)

    def test_seed_everything_repeats_python_and_torch_streams(self) -> None:
        seed_everything(123)
        first = (random.random(), torch.rand(3))
        seed_everything(123)
        second = (random.random(), torch.rand(3))
        self.assertEqual(first[0], second[0])
        torch.testing.assert_close(first[1], second[1])
        with self.assertRaises(ValueError):
            seed_everything(-1)


if __name__ == "__main__":
    unittest.main()
