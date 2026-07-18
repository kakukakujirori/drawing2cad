from pathlib import Path
import unittest

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Qwen3VLConfig

from src.data import (
    DXFPrimitiveConfig,
    DXFPrimitiveParser,
    Drawing2CADCollator,
    Drawing2CADDataset,
)
from src.models import (
    Drawing2CADQwen3VLForConditionalGeneration,
    PrimitiveEncoderConfig,
)


DATASET_ROOT = Path("experiments/dataset_z2c_eccv_val")
CHECKPOINT = "ADSKAILab/Zero-To-CAD-Qwen3-VL-2B"


class DatasetModelIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not DATASET_ROOT.is_dir():
            raise unittest.SkipTest(f"sample dataset unavailable: {DATASET_ROOT}")
        try:
            cls.processor = AutoProcessor.from_pretrained(
                CHECKPOINT,
                trust_remote_code=True,
                local_files_only=True,
            )
        except OSError as error:
            raise unittest.SkipTest(f"local Qwen processor unavailable: {error}") from error
        cls.dxf_config = DXFPrimitiveConfig(samples_per_primitive=8)
        cls.primitive_config = PrimitiveEncoderConfig(
            sample_feature_dim=cls.dxf_config.sample_feature_dim,
            num_primitive_types=cls.dxf_config.num_primitive_types,
            primitive_dim=16,
            num_primitive_latents=2,
            primitive_encoder_layers=2,
            resampler_layers=1,
            resampler_heads=4,
            dropout=0.0,
        )

    def _dataset(self, *, include_target: bool, max_samples: int):
        return Drawing2CADDataset(
            DATASET_ROOT,
            dxf_parser=DXFPrimitiveParser(self.dxf_config),
            include_target=include_target,
            max_samples=max_samples,
            image_max_edge=64,
        )

    def test_real_dataset_item_and_batched_padding(self) -> None:
        dataset = self._dataset(include_target=True, max_samples=2)
        first = dataset[0]
        self.assertGreater(first.primitives.num_primitives, 0)
        self.assertEqual(first.primitives.sample_features.shape[1:], (8, 3))
        self.assertEqual(len(first.images), 1)
        self.assertLessEqual(max(first.images[0].size), 64)
        self.assertIn("result", first.target_code)

        collator = Drawing2CADCollator(
            self.processor,
            self.primitive_config,
            include_labels=True,
        )
        batch = next(
            iter(DataLoader(dataset, batch_size=2, collate_fn=collator, num_workers=0))
        )
        inputs = batch.model_inputs
        primitive_batch = inputs["primitive_batch"]
        self.assertEqual(batch.sample_ids, tuple(r.sample_id for r in dataset.records))
        self.assertEqual(primitive_batch.sample_features.shape[0:2], (2, 1))
        self.assertEqual(primitive_batch.sample_features.shape[-2:], (8, 3))
        self.assertEqual(inputs["primitive_token_mask"].sum(dim=-1).tolist(), [2, 2])
        self.assertTrue(
            torch.all(inputs["attention_mask"][inputs["primitive_token_mask"]] == 1)
        )
        self.assertTrue(
            torch.all(inputs["mm_token_type_ids"][inputs["primitive_token_mask"]] == 0)
        )
        self.assertEqual(inputs["labels"].shape[1], inputs["logits_to_keep"])
        self.assertTrue(torch.all(inputs["labels"][:, 0] == -100))
        self.assertTrue(torch.any(inputs["labels"] != -100))

    def _tiny_model(self):
        tokenizer = self.processor.tokenizer
        text_config = {
            "vocab_size": len(tokenizer),
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 4096,
            "rope_parameters": {
                "rope_type": "default",
                "mrope_section": [2, 1, 1],
                "mrope_interleaved": True,
            },
            "pad_token_id": tokenizer.pad_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        vision_config = {
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 4,
            "in_channels": 3,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 32,
            "num_position_embeddings": 2304,
            "deepstack_visual_indexes": [0],
        }
        config = Qwen3VLConfig(
            text_config=text_config,
            vision_config=vision_config,
            image_token_id=tokenizer.convert_tokens_to_ids("<|image_pad|>"),
            video_token_id=tokenizer.convert_tokens_to_ids("<|video_pad|>"),
            vision_start_token_id=tokenizer.convert_tokens_to_ids("<|vision_start|>"),
            vision_end_token_id=tokenizer.convert_tokens_to_ids("<|vision_end|>"),
        )
        config.primitive_config = self.primitive_config.to_dict()
        return Drawing2CADQwen3VLForConditionalGeneration(config).eval()

    def test_dataloader_to_native_image_and_primitive_model_forward(self) -> None:
        dataset = self._dataset(include_target=False, max_samples=1)
        collator = Drawing2CADCollator(
            self.processor,
            self.primitive_config,
            include_labels=False,
        )
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=collator,
        )
        batch = next(iter(loader))
        model = self._tiny_model()
        with torch.inference_mode():
            output = model(**batch.model_inputs, use_cache=False)
        self.assertEqual(output.logits.shape[0:2], (1, 1))
        self.assertTrue(torch.isfinite(output.logits).all())


if __name__ == "__main__":
    unittest.main()
