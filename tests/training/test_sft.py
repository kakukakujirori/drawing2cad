from pathlib import Path
import tempfile
import unittest

from accelerate import Accelerator
import torch

from src.models.factory import apply_language_lora, freeze_vision_encoder
from src.training.checkpoint import AdapterCheckpointIO
from src.training.state import TrainingProgress
from tests.model_helpers import make_primitive_batch, tiny_drawing_model


LORA_CONFIG = {
    "enabled": True,
    "rank": 2,
    "alpha": 4,
    "dropout": 0.0,
    "bias": "none",
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
}


def tiny_inputs() -> dict:
    input_ids = torch.tensor([[5, 6, 7, 8, 9, 10]])
    labels = torch.tensor([[-100, -100, -100, -100, 9, 10]])
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "mm_token_type_ids": torch.zeros_like(input_ids, dtype=torch.int32),
        "labels": labels,
        "primitive_batch": make_primitive_batch((2,)),
        "primitive_token_mask": torch.tensor(
            [[False, True, True, False, False, False]]
        ),
        "use_cache": False,
    }


def tiny_peft_model():
    model = tiny_drawing_model()
    visual = freeze_vision_encoder(model)
    model = apply_language_lora(model, LORA_CONFIG)
    freeze_vision_encoder(model)
    return model, visual


class SFTTrainingTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(91)

    def test_one_step_trains_lora_and_primitive_but_not_vision(self) -> None:
        model, visual = tiny_peft_model()
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=1e-3,
        )
        before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

        output = model(**tiny_inputs())
        self.assertIsNotNone(output.loss)
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()

        lora_gradients = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if "lora_" in name and parameter.requires_grad
        ]
        primitive_gradients = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if "primitive_encoder.modules_to_save" in name
        ]
        self.assertTrue(lora_gradients)
        self.assertTrue(any(gradient is not None for gradient in lora_gradients))
        self.assertTrue(primitive_gradients)
        self.assertTrue(any(gradient is not None for gradient in primitive_gradients))
        self.assertTrue(
            all(not parameter.requires_grad for parameter in visual.parameters())
        )
        self.assertTrue(
            all(parameter.grad is None for parameter in visual.parameters())
        )

        optimizer.step()
        self.assertTrue(
            any(
                not torch.equal(before[name], parameter.detach())
                for name, parameter in model.named_parameters()
                if name in before
            )
        )

    def test_adapter_checkpoint_restores_model_optimizer_scheduler_rng_and_step(
        self,
    ) -> None:
        model, _ = tiny_peft_model()
        accelerator = Accelerator(cpu=True)
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=1e-3,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: 1.0 / (step + 1)
        )
        output = model(**tiny_inputs())
        output.loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        generator = torch.Generator().manual_seed(1234)
        checkpoint = AdapterCheckpointIO(
            accelerator=accelerator,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloader_generator=generator,
        )
        progress = TrainingProgress(global_step=3, epoch=1, batches_seen_in_epoch=2)
        checkpoint.set_progress(progress)
        saved_parameters = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        expected_lr = optimizer.param_groups[0]["lr"]
        expected_generator_state = generator.get_state().clone()

        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            directory = Path(temporary)
            checkpoint.save(directory)
            self.assertTrue((directory / "adapter/adapter_model.safetensors").is_file())
            self.assertFalse((directory / "model.safetensors").exists())
            self.assertFalse((directory / "adapter/model.safetensors").exists())

            with torch.no_grad():
                for parameter in model.parameters():
                    if parameter.requires_grad:
                        parameter.add_(10.0)
            optimizer.param_groups[0]["lr"] = 0.25
            scheduler.last_epoch = 99
            generator.manual_seed(9999)

            restored = checkpoint.load(directory)

        self.assertEqual(restored, progress)
        self.assertEqual(optimizer.param_groups[0]["lr"], expected_lr)
        torch.testing.assert_close(generator.get_state(), expected_generator_state)
        for name, parameter in model.named_parameters():
            if name in saved_parameters:
                torch.testing.assert_close(parameter, saved_parameters[name])


if __name__ == "__main__":
    unittest.main()
