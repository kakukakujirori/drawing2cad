"""Modality dropout that forces answers through the drawing primitives."""

from __future__ import annotations

import random
import unittest

from PIL import Image
import torch

from src.data.dxf import DXFPrimitiveData
from src.data.preprocessing import Drawing2CADPreprocessor
from src.data.dataset import Drawing2CADSample


class StubTokenizer:
    pad_token = "<|pad|>"
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [1, 2, 3]

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [1] * len(text.split())}


class StubProcessor:
    """Minimal processor: only the image content of the conversation matters."""

    def __init__(self) -> None:
        self.tokenizer = StubTokenizer()
        self.seen_images: list[tuple[Image.Image, ...]] = []

    def apply_chat_template(self, conversation, **kwargs):
        images = tuple(
            part["image"]
            for message in conversation
            for part in message["content"]
            if isinstance(part, dict) and part.get("type") == "image"
        )
        self.seen_images.append(images)
        input_ids = torch.tensor([[0, 0, 5, 6]])
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "mm_token_type_ids": torch.zeros_like(input_ids, dtype=torch.int32),
        }


def sample(width: int = 8, height: int = 4) -> Drawing2CADSample:
    return Drawing2CADSample(
        sample_id="s0",
        primitives=DXFPrimitiveData(
            sample_features=torch.zeros(1, 2, 7),
            primitive_type_ids=torch.zeros(1, dtype=torch.long),
            view_direction_ids=torch.zeros(1, dtype=torch.long),
            entity_handles=("0",),
            entity_type_names=("LINE",),
        ),
        images=(
            Image.new("RGB", (width, height), (200, 30, 30)),
            Image.new("RGB", (width, height), (30, 200, 30)),
        ),
        image_styles=("isometric", "drawing"),
        target_code="result = 1",
    )


def preprocessor(**kwargs) -> Drawing2CADPreprocessor:
    return Drawing2CADPreprocessor(
        StubProcessor(), num_primitive_latents=2, include_labels=False, **kwargs
    )


def blanked(result: Drawing2CADSample) -> bool:
    return all(
        image.getextrema() == ((0, 0), (0, 0), (0, 0)) for image in result.images
    )


class ImageDropoutTest(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(7)

    def test_disabled_by_default(self) -> None:
        self.assertEqual(preprocessor().image_dropout, 0.0)

    def test_zero_probability_returns_the_sample_untouched(self) -> None:
        prep = preprocessor()
        original = sample()
        self.assertIs(prep.apply_image_dropout(original), original)

    def test_probability_one_blanks_every_view(self) -> None:
        prep = preprocessor(image_dropout=1.0)
        result = prep.apply_image_dropout(sample())
        self.assertTrue(blanked(result))

    def test_all_views_are_blanked_together(self) -> None:
        # Blanking views independently would leave one to answer from, which
        # defeats the point of the augmentation.
        prep = preprocessor(image_dropout=0.5)
        for _ in range(50):
            result = prep.apply_image_dropout(sample())
            per_view = [
                image.getextrema() == ((0, 0), (0, 0), (0, 0))
                for image in result.images
            ]
            self.assertIn(len(set(per_view)), (1,), per_view)

    def test_blanking_preserves_image_size_so_token_length_is_unchanged(self) -> None:
        prep = preprocessor(image_dropout=1.0)
        original = sample(width=13, height=7)
        result = prep.apply_image_dropout(original)
        self.assertEqual(
            [image.size for image in result.images],
            [image.size for image in original.images],
        )

    def test_fill_colour_is_configurable(self) -> None:
        prep = preprocessor(image_dropout=1.0, image_dropout_fill=255)
        result = prep.apply_image_dropout(sample())
        for image in result.images:
            self.assertEqual(image.getextrema(), ((255, 255), (255, 255), (255, 255)))

    def test_rate_is_honoured_over_many_draws(self) -> None:
        prep = preprocessor(image_dropout=0.25)
        random.seed(0)
        dropped = sum(blanked(prep.apply_image_dropout(sample())) for _ in range(2000))
        self.assertAlmostEqual(dropped / 2000, 0.25, delta=0.04)

    def test_the_original_sample_is_never_mutated(self) -> None:
        prep = preprocessor(image_dropout=1.0)
        original = sample()
        prep.apply_image_dropout(original)
        self.assertNotEqual(original.images[0].getextrema(), ((0, 0), (0, 0), (0, 0)))

    def test_call_routes_blanked_images_into_the_conversation(self) -> None:
        prep = preprocessor(image_dropout=1.0)
        prep(sample())
        (images,) = prep.processor.seen_images
        self.assertEqual(len(images), 2)
        for image in images:
            self.assertEqual(image.getextrema(), ((0, 0), (0, 0), (0, 0)))

    def test_sequence_length_never_sees_dropout(self) -> None:
        # Length filtering must be deterministic and measured on real views.
        prep = preprocessor(image_dropout=1.0)
        prep.sequence_length(sample())
        (images,) = prep.processor.seen_images
        self.assertNotEqual(images[0].getextrema(), ((0, 0), (0, 0), (0, 0)))

    def test_out_of_range_settings_are_rejected(self) -> None:
        for kwargs in (
            {"image_dropout": -0.1},
            {"image_dropout": 1.5},
            {"image_dropout_fill": -1},
            {"image_dropout_fill": 256},
        ):
            with self.assertRaises(ValueError):
                preprocessor(**kwargs)


if __name__ == "__main__":
    unittest.main()
