from pathlib import Path
import unittest

import yaml
from hydra import compose, initialize_config_dir

from src.data import DXF_ORIENTED_SAMPLE_FEATURE_INDICES, DXFPrimitiveConfig
from src.models import PrimitiveEncoderConfig


CONFIG_ROOT = Path(__file__).parents[2] / "configs"


class TrainingConfigTest(unittest.TestCase):
    def _load(self, relative_path: str) -> dict:
        path = CONFIG_ROOT / relative_path
        self.assertTrue(path.is_file(), path)
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertIsInstance(value, dict)
        return value

    def test_root_composes_all_required_groups(self) -> None:
        root = self._load("train_sft.yaml")
        defaults = root["defaults"]
        groups = {next(iter(item)) for item in defaults if isinstance(item, dict)}
        self.assertTrue(
            {
                "data",
                "model",
                "optimizer",
                "scheduler",
                "logger",
                "checkpoint",
                "hydra",
                "debug",
            }.issubset(groups)
        )
        self.assertEqual(root["task_name"], "train_sft")

    def test_data_and_primitive_configs_match_constructor_contracts(self) -> None:
        data = self._load("data/z2c_smoke.yaml")
        dxf = DXFPrimitiveConfig(**data["dxf"])
        self.assertEqual(dxf.sample_feature_dim, 7)
        self.assertFalse(data["scale_augmentation"])
        self.assertEqual(
            data["image_sources"],
            [
                {
                    "style": "isometric",
                    "directory": "render_3d/hlg_perspective",
                }
            ],
        )
        model = self._load("model/qwen3vl_primitive_lora.yaml")
        primitive = PrimitiveEncoderConfig(**model["primitive"])
        self.assertEqual(primitive.view_directions, ("front", "top", "right"))
        self.assertEqual(primitive.sample_feature_dim, dxf.sample_feature_dim)
        self.assertEqual(primitive.num_primitive_types, dxf.num_primitive_types)
        self.assertEqual(
            primitive.oriented_feature_indices,
            DXF_ORIENTED_SAMPLE_FEATURE_INDICES,
        )

    def test_checkpoint_monitor_is_a_direct_log_key(self) -> None:
        checkpoint = self._load("checkpoint/topk.yaml")
        self.assertIn("/", checkpoint["monitor"])
        self.assertIn(checkpoint["mode"], {"min", "max"})
        self.assertGreater(checkpoint["top_k"], 0)
        self.assertEqual(checkpoint["latest_dirname"], "latest")

    def test_configured_metrics_build_and_cover_the_checkpoint_monitor(self) -> None:
        from src.metrics.registry import build_metrics

        evaluation = self._load("train_sft.yaml")["evaluation"]
        metrics = build_metrics(evaluation["metrics"])
        self.assertTrue(metrics)
        monitor = self._load("checkpoint/topk.yaml")["monitor"]
        # `val/<root>/<key>`: the root is a dataset name known only at runtime,
        # so only the metric key itself can be checked here.
        produced = set()
        for metric in metrics:
            produced.update(
                key.split("/")[-1] for key in metric.reduce([], prefix="val")
            )
        # The teacher-forced loss pass in src/training/sft.py emits its keys
        # alongside the generation metrics and is monitorable on the same terms.
        produced.update(("loss", "loss_shuffled_primitives", "primitive_gain"))
        self.assertIn(monitor.split("/")[-1], produced)

    def test_hydra_composes_production_defaults(self) -> None:
        with initialize_config_dir(
            config_dir=str(CONFIG_ROOT.resolve()), version_base="1.3"
        ):
            config = compose(config_name="train_sft", return_hydra_config=True)
        self.assertIsNone(config.hydra.runtime.choices["debug"])
        self.assertEqual(config.hydra.runtime.choices["data"], "z2c")
        # The production budget is expressed in epochs, so max_steps is resolved
        # from the sharded loader length at runtime rather than pinned here.
        self.assertIsNone(config.training.max_steps)
        self.assertEqual(config.training.num_train_epochs, 3)
        self.assertEqual(config.training.gradient_accumulation_steps, 8)
        self.assertIsNone(config.data.train_max_samples)
        self.assertTrue(config.utils.progress_bar.enabled)

    def test_hydra_composes_explicit_smoke_overrides(self) -> None:
        with initialize_config_dir(
            config_dir=str(CONFIG_ROOT.resolve()), version_base="1.3"
        ):
            config = compose(
                config_name="train_sft",
                overrides=["data=z2c_smoke", "debug=smoke"],
                return_hydra_config=True,
            )
        self.assertEqual(config.hydra.runtime.choices["debug"], "smoke")
        self.assertEqual(config.hydra.runtime.choices["data"], "z2c_smoke")
        self.assertEqual(config.hydra.runtime.choices["utils/progress_bar"], "rich")
        self.assertEqual(config.training.max_steps, 2)
        self.assertEqual(config.data.train_max_samples, 4)
        self.assertEqual(
            list(config.model.primitive.view_directions), ["front", "top", "right"]
        )


if __name__ == "__main__":
    unittest.main()
