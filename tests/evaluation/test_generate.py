from pathlib import Path
import tempfile
import unittest

import torch

from src.data import Drawing2CADBatch, SampleMetadata, ViewBBox
from src.data.layout import REQUIRED_SUBDIRS, resolve_dataset_roots
from src.evaluation.generate import CADGenerationEvaluator
from src.models import PrimitiveBatch, PrimitiveEncoderConfig


class _Accelerator:
    is_main_process = True
    num_processes = 1
    process_index = 0
    device = torch.device("cpu")

    @staticmethod
    def wait_for_everyone():
        return None

    @staticmethod
    def unwrap_model(model):
        return model


class _Tokenizer:
    pad_token_id = 0


class _Processor:
    tokenizer = _Tokenizer()

    @staticmethod
    def batch_decode(tokens, **kwargs):
        del kwargs
        return [f"generated-{int(row[0])}" for row in tokens]


class _Model(torch.nn.Module):
    def generate(self, input_ids, **kwargs):
        del kwargs
        suffix = torch.full(
            (input_ids.shape[0], 1), 42, dtype=input_ids.dtype, device=input_ids.device
        )
        return torch.cat((input_ids, suffix), dim=1)


def _metadata() -> SampleMetadata:
    return SampleMetadata(
        sample_id="sample",
        view_bboxes=(
            ViewBBox("front", (0, 0, 1, 1)),
            ViewBBox("top", (2, 2, 3, 3)),
            ViewBBox("right", (4, 4, 5, 5)),
        ),
        drawing_scale=1.0,
    )


def _batch() -> Drawing2CADBatch:
    primitive_batch = PrimitiveBatch(
        sample_features=torch.zeros(1, 1, 2, 3),
        sample_mask=torch.ones(1, 1, 2, dtype=torch.bool),
        primitive_mask=torch.ones(1, 1, dtype=torch.bool),
        primitive_type_ids=torch.zeros(1, 1, dtype=torch.long),
        view_direction_ids=torch.zeros(1, 1, dtype=torch.long),
    )
    return Drawing2CADBatch(
        model_inputs={
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
            "mm_token_type_ids": torch.zeros(1, 2, dtype=torch.long),
            "primitive_token_mask": torch.tensor([[False, True]]),
            "primitive_batch": primitive_batch,
            "logits_to_keep": 1,
        },
        sample_ids=("sample",),
        sample_metadata=(_metadata(),),
    )


class CADGenerationEvaluatorTest(unittest.TestCase):
    def test_generation_writes_artifacts_and_stable_checkpoint_metric(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            # The root's basename names the dataset in every metric key, so it
            # must be stable rather than the random temporary directory name.
            root = Path(temporary) / "bench_val"
            root.mkdir()
            manifest = {
                "name": "sample",
                "ok": True,
                "extra": {
                    "techdraw": {
                        "scale": 1.0,
                        "bbox_format": "xyxy",
                        "bbox_coordinate_system": {
                            "unit": "mm",
                            "origin": "sheet_bottom_left",
                            "x_axis": "right",
                            "y_axis": "up",
                        },
                        "views": {
                            "front": {"bbox": [0, 0, 1, 1]},
                            "top": {"bbox": [2, 2, 3, 3]},
                            "right": {"bbox": [4, 4, 5, 5]},
                        },
                    }
                },
            }
            import json

            (root / "manifest.jsonl").write_text(
                json.dumps(manifest) + "\n", encoding="utf-8"
            )
            for name in REQUIRED_SUBDIRS:
                (root / name).mkdir()
            (dataset_root,) = resolve_dataset_roots(root, split="val")
            evaluator = CADGenerationEvaluator(
                accelerator=_Accelerator(),
                processor=_Processor(),
                data_config={},
                dataset_root=dataset_root,
                evaluation_config={
                    "generation_subset_size": 1,
                    "generation_seed": 7,
                    "max_new_tokens": 1,
                    "metrics": ["CadExecutionMetric", "VoxelIoUMetric"],
                },
                primitive_config=PrimitiveEncoderConfig(
                    sample_feature_dim=3,
                    num_primitive_types=7,
                    primitive_dim=8,
                    num_primitive_latents=1,
                    primitive_encoder_layers=1,
                    resampler_layers=1,
                    resampler_heads=2,
                ),
                predictions_dir=root / "predictions",
            )
            evaluator._loader = lambda sample_ids: [_batch()]
            model = _Model().train()
            metrics = evaluator(model, step=3)

            self.assertEqual(metrics["val/bench_val/mean_iou_including_failures"], 0.0)
            self.assertTrue(model.training)
            step_dir = root / "predictions" / "step_00000003"
            self.assertEqual(
                (step_dir / "sample.cadquery.py").read_text(encoding="utf-8"),
                "generated-42",
            )
            self.assertTrue((step_dir / "metrics.json").is_file())
            self.assertTrue((root / "predictions" / "subset.json").is_file())


if __name__ == "__main__":
    unittest.main()
