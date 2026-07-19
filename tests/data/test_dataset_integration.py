from pathlib import Path
from dataclasses import replace
import unittest

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Qwen3VLConfig

from src.data import (
    DXFPrimitiveConfig,
    DXFPrimitiveParser,
    Drawing2CADCollator,
    Drawing2CADDataset,
    Drawing2CADPreprocessor,
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

    def _dataset(
        self,
        *,
        include_target: bool,
        max_samples: int,
        transform=None,
    ):
        return Drawing2CADDataset(
            DATASET_ROOT,
            dxf_parser=DXFPrimitiveParser(self.dxf_config),
            include_target=include_target,
            max_samples=max_samples,
            image_max_edge=64,
            transform=transform,
        )

    def test_real_dataset_item_and_batched_padding(self) -> None:
        dataset = self._dataset(include_target=True, max_samples=2)
        first = dataset[0]
        self.assertGreater(first.primitives.num_primitives, 0)
        self.assertEqual(first.primitives.sample_features.shape[1:], (8, 3))
        self.assertEqual(
            first.primitives.view_direction_ids.shape,
            (first.primitives.num_primitives,),
        )
        self.assertTrue(
            torch.all(
                (first.primitives.view_direction_ids >= 0)
                & (first.primitives.view_direction_ids < 3)
            )
        )
        self.assertEqual(len(first.images), 1)
        self.assertEqual(first.image_styles, ("isometric",))
        self.assertLessEqual(max(first.images[0].size), 64)
        self.assertIn("result", first.target_code)

        preprocessor = Drawing2CADPreprocessor(
            self.processor,
            self.primitive_config.num_primitive_latents,
            include_labels=True,
        )
        prepared_dataset = self._dataset(
            include_target=True,
            max_samples=2,
            transform=preprocessor,
        )
        collator = Drawing2CADCollator(
            self.processor.tokenizer.pad_token_id,
            padding_side=self.processor.tokenizer.padding_side,
        )
        batch = next(
            iter(
                DataLoader(
                    prepared_dataset,
                    batch_size=2,
                    collate_fn=collator,
                    num_workers=0,
                )
            )
        )
        inputs = batch.model_inputs
        primitive_batch = inputs["primitive_batch"]
        self.assertEqual(batch.sample_ids, tuple(r.sample_id for r in dataset.records))
        self.assertEqual(primitive_batch.sample_features.shape[0], 2)
        self.assertEqual(primitive_batch.sample_features.shape[-2:], (8, 3))
        self.assertEqual(inputs["primitive_token_mask"].sum(dim=-1).tolist(), [2, 2])
        self.assertTrue(
            torch.all(inputs["attention_mask"][inputs["primitive_token_mask"]] == 1)
        )
        self.assertTrue(
            torch.all(inputs["mm_token_type_ids"][inputs["primitive_token_mask"]] == 0)
        )
        self.assertEqual(inputs["labels"].shape[1], inputs["logits_to_keep"])
        self.assertLess(inputs["labels"].shape[1], inputs["input_ids"].shape[1])
        self.assertTrue(torch.all(inputs["labels"][:, 0] == -100))
        self.assertTrue(torch.any(inputs["labels"] != -100))
        for batch_index in range(2):
            prepared = prepared_dataset[batch_index]
            primitive_count = prepared.primitives.num_primitives
            torch.testing.assert_close(
                primitive_batch.view_direction_ids[batch_index, :primitive_count],
                prepared.primitives.view_direction_ids,
            )
            self.assertTrue(
                torch.all(
                    primitive_batch.view_direction_ids[
                        batch_index, primitive_count:
                    ]
                    == -1
                )
            )

    def test_single_sample_preprocessing_and_two_image_concat_order(self) -> None:
        sample = self._dataset(include_target=True, max_samples=1)[0]
        preprocessor = Drawing2CADPreprocessor(
            self.processor,
            self.primitive_config.num_primitive_latents,
            include_labels=True,
        )
        one_image = preprocessor(sample)
        two_image = preprocessor(
            replace(
                sample,
                images=(sample.images[0], sample.images[0].copy()),
                image_styles=("isometric", "drawing"),
            )
        )
        self.assertEqual(one_image.model_inputs["image_grid_thw"].shape[0], 1)
        self.assertEqual(two_image.model_inputs["image_grid_thw"].shape[0], 2)
        collator = Drawing2CADCollator(self.processor.tokenizer.pad_token_id)
        batch = collator([one_image, two_image])
        torch.testing.assert_close(
            batch.model_inputs["pixel_values"],
            torch.cat(
                [
                    one_image.model_inputs["pixel_values"],
                    two_image.model_inputs["pixel_values"],
                ],
                dim=0,
            ),
        )
        torch.testing.assert_close(
            batch.model_inputs["image_grid_thw"],
            torch.cat(
                [
                    one_image.model_inputs["image_grid_thw"],
                    two_image.model_inputs["image_grid_thw"],
                ],
                dim=0,
            ),
        )

    def test_dataset_accepts_substituted_typed_metadata_provider(self) -> None:
        baseline = self._dataset(include_target=False, max_samples=1)
        sample_id = baseline.records[0].sample_id
        metadata = baseline.metadata_provider.get(sample_id)

        class StaticProvider:
            def get(self, requested_id):
                if requested_id != sample_id:
                    raise KeyError(requested_id)
                return metadata

        dataset = Drawing2CADDataset(
            DATASET_ROOT,
            metadata_provider=StaticProvider(),
            dxf_parser=DXFPrimitiveParser(self.dxf_config),
            include_target=False,
            sample_ids=(sample_id,),
            image_max_edge=64,
        )
        self.assertEqual(dataset[0].metadata, metadata)

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
        preprocessor = Drawing2CADPreprocessor(
            self.processor,
            self.primitive_config.num_primitive_latents,
            include_labels=False,
        )
        dataset = self._dataset(
            include_target=False, max_samples=1, transform=preprocessor
        )
        collator = Drawing2CADCollator(self.processor.tokenizer.pad_token_id)
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
