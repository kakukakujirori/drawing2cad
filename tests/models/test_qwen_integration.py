import copy
import tempfile
import unittest
from unittest import mock

import torch
from transformers import Qwen3VLForConditionalGeneration

from src.models import Drawing2CADQwen3VLForConditionalGeneration
from tests.model_helpers import (
    make_primitive_batch,
    primitive_config,
    tiny_drawing_model,
    tiny_qwen_config,
)


class QwenIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)
        self.model = tiny_drawing_model().eval()
        self.primitive_batch = make_primitive_batch((1,))
        self.input_ids = torch.tensor([[5, 6, 7, 8, 9]])
        self.attention_mask = torch.ones_like(self.input_ids)
        self.mm_token_type_ids = torch.zeros_like(
            self.input_ids, dtype=torch.int32
        )
        self.primitive_token_mask = torch.tensor(
            [[False, True, True, False, False]]
        )

    def _primitive_forward(self, **overrides):
        inputs = {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "mm_token_type_ids": self.mm_token_type_ids,
            "primitive_batch": self.primitive_batch,
            "primitive_token_mask": self.primitive_token_mask,
        }
        inputs.update(overrides)
        return self.model(**inputs)

    def test_no_primitive_path_matches_native_qwen(self) -> None:
        base_config = copy.deepcopy(self.model.config)
        native = Qwen3VLForConditionalGeneration(base_config).eval()
        qwen_state = {
            key: value
            for key, value in self.model.state_dict().items()
            if not key.startswith("primitive_encoder.")
        }
        native.load_state_dict(qwen_state, strict=True)

        with torch.no_grad():
            wrapped_output = self.model(
                input_ids=self.input_ids,
                attention_mask=self.attention_mask,
                use_cache=False,
            )
            native_output = native(
                input_ids=self.input_ids,
                attention_mask=self.attention_mask,
                use_cache=False,
            )
        torch.testing.assert_close(wrapped_output.logits, native_output.logits)

    def test_only_requested_embeddings_are_replaced(self) -> None:
        captured = {}

        def capture_inputs(_module, _args, kwargs):
            captured["inputs_embeds"] = kwargs["inputs_embeds"].detach().clone()

        handle = self.model.model.register_forward_pre_hook(
            capture_inputs, with_kwargs=True
        )
        try:
            with torch.no_grad():
                expected_latents, _ = self.model.encode_primitives(
                    self.primitive_batch
                )
                original = self.model.get_input_embeddings()(self.input_ids)
                self._primitive_forward(use_cache=False)
        finally:
            handle.remove()

        injected = captured["inputs_embeds"]
        torch.testing.assert_close(
            injected[self.primitive_token_mask],
            expected_latents.to(injected),
        )
        torch.testing.assert_close(
            injected[~self.primitive_token_mask],
            original[~self.primitive_token_mask],
        )

    def test_placeholder_contract_errors_before_scatter(self) -> None:
        mismatch = self.primitive_token_mask.clone()
        mismatch[0, 2] = False
        with self.assertRaisesRegex(ValueError, "placeholder count mismatch"):
            self._primitive_forward(primitive_token_mask=mismatch)

        labels = torch.full_like(self.input_ids, -100)
        labels[self.primitive_token_mask] = 1
        with self.assertRaisesRegex(ValueError, "labels=-100"):
            self._primitive_forward(labels=labels)

        invalid_types = self.mm_token_type_ids.clone()
        invalid_types[self.primitive_token_mask] = 1
        with self.assertRaisesRegex(ValueError, "mm_token_type_ids=0"):
            self._primitive_forward(mm_token_type_ids=invalid_types)

    def test_native_image_path_and_multimodal_rope_remain_active(self) -> None:
        # Grid [1, 4, 4] produces four merged Qwen image tokens.
        input_ids = torch.tensor([[5, 120, 120, 120, 120, 6, 7, 8]])
        mm_token_type_ids = torch.tensor(
            [[0, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.int32
        )
        primitive_mask = torch.tensor(
            [[False, False, False, False, False, True, True, False]]
        )
        pixel_values = torch.randn(16, 24)
        image_grid_thw = torch.tensor([[1, 4, 4]])

        with mock.patch.object(
            self.model.model,
            "get_image_features",
            wraps=self.model.model.get_image_features,
        ) as image_features:
            with torch.no_grad():
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    mm_token_type_ids=mm_token_type_ids,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    primitive_batch=self.primitive_batch,
                    primitive_token_mask=primitive_mask,
                    use_cache=False,
                )
        image_features.assert_called_once()
        self.assertIs(image_features.call_args.args[0], pixel_values)
        self.assertIs(image_features.call_args.args[1], image_grid_thw)
        self.assertEqual(output.logits.shape, (1, 8, 128))
        self.assertIsNotNone(self.model.model.rope_deltas)

    def test_mixed_primitive_counts_use_one_placeholder_block_each(self) -> None:
        primitive_batch = make_primitive_batch((1, 7))
        input_ids = torch.tensor(
            [
                [5, 6, 7, 8, 9, 10, 11, 12],
                [13, 14, 15, 16, 17, 18, 19, 20],
            ]
        )
        primitive_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        primitive_mask[:, 1:3] = True
        with torch.no_grad():
            output = self.model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                mm_token_type_ids=torch.zeros_like(input_ids, dtype=torch.int32),
                primitive_batch=primitive_batch,
                primitive_token_mask=primitive_mask,
            )
        self.assertEqual(output.logits.shape, (2, 8, 128))

    def test_labels_logits_to_keep_and_standard_output(self) -> None:
        labels = torch.tensor([[8, 9]])
        output = self._primitive_forward(labels=labels, logits_to_keep=2)
        self.assertEqual(output.logits.shape, (1, 2, 128))
        self.assertIsNotNone(output.loss)
        self.assertTrue(torch.isfinite(output.loss))
        self.assertIsNotNone(output.past_key_values)

    def test_greedy_generation_encodes_primitives_once(self) -> None:
        with mock.patch.object(
            self.model.primitive_encoder,
            "forward",
            wraps=self.model.primitive_encoder.forward,
        ) as encode:
            with torch.no_grad():
                generated = self.model.generate(
                    input_ids=self.input_ids,
                    attention_mask=self.attention_mask,
                    mm_token_type_ids=self.mm_token_type_ids,
                    primitive_batch=self.primitive_batch,
                    primitive_token_mask=self.primitive_token_mask,
                    max_new_tokens=3,
                    do_sample=False,
                )
        self.assertEqual(encode.call_count, 1)
        self.assertEqual(generated.shape, (1, self.input_ids.shape[1] + 3))

    def test_saved_checkpoint_round_trip_restores_config_and_weights(self) -> None:
        original_weight = self.model.primitive_encoder.resampler.queries.detach().clone()
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            self.model.save_pretrained(directory)
            reloaded = (
                Drawing2CADQwen3VLForConditionalGeneration.from_pretrained(directory)
            )
        self.assertEqual(
            reloaded.primitive_config.to_dict(),
            self.model.primitive_config.to_dict(),
        )
        torch.testing.assert_close(
            reloaded.primitive_encoder.resampler.queries,
            original_weight,
        )

    def test_from_qwen_pretrained_allows_only_new_primitive_weights(self) -> None:
        config = tiny_qwen_config()
        native = Qwen3VLForConditionalGeneration(config)
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            native.save_pretrained(directory)
            loaded = Drawing2CADQwen3VLForConditionalGeneration.from_qwen_pretrained(
                directory,
                primitive_config=primitive_config(),
                attn_implementation="sdpa",
            )
        self.assertIsInstance(
            loaded, Drawing2CADQwen3VLForConditionalGeneration
        )
        self.assertEqual(loaded.config.text_config.hidden_size, 32)
        encoded, counts = loaded.encode_primitives(make_primitive_batch((1,)))
        self.assertEqual(counts.tolist(), [4])
        self.assertTrue(torch.isfinite(encoded).all())


if __name__ == "__main__":
    unittest.main()
