"""Adapter checkpoints must not resume against a different model definition."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from peft.utils.save_and_load import load_peft_weights
import torch

from src.models.factory import apply_language_lora, freeze_vision_encoder
from src.training.checkpoint import AdapterCheckpointIO
from src.training.state import TrainingProgress
from tests.model_helpers import tiny_drawing_model


LORA_CONFIG = {
    "enabled": True,
    "rank": 2,
    "alpha": 4,
    "dropout": 0.0,
    "bias": "none",
    "target_modules": ["q_proj", "v_proj"],
}


class StubAccelerator:
    """Everything AdapterCheckpointIO uses.

    A real Accelerator is deliberately avoided: it initializes a process-wide
    AcceleratorState singleton, so constructing one per test method conflicts
    with whichever test built the first one.
    """

    device = torch.device("cpu")

    @staticmethod
    def unwrap_model(model):
        return model


class AdapterCheckpointKeyTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(3)
        model = tiny_drawing_model()
        freeze_vision_encoder(model)
        self.model = apply_language_lora(model, LORA_CONFIG)
        freeze_vision_encoder(self.model)
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=1e-3
        )
        self.io = AdapterCheckpointIO(
            accelerator=StubAccelerator(),
            model=self.model,
            optimizer=optimizer,
            scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0),
        )
        self.io.set_progress(TrainingProgress(global_step=1))

    def _saved(self, directory: Path) -> Path:
        self.io.save(directory)
        return directory / "adapter"

    def _rewrite_without(self, adapter_dir: Path, predicate) -> None:
        from safetensors.torch import save_file

        state = load_peft_weights(str(adapter_dir), device="cpu")
        kept = {key: value for key, value in state.items() if not predicate(key)}
        self.assertLess(len(kept), len(state))
        (adapter_dir / "adapter_model.safetensors").unlink()
        save_file(kept, str(adapter_dir / "adapter_model.safetensors"))

    def test_a_complete_checkpoint_loads(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            directory = Path(temporary)
            self._saved(directory)
            self.assertEqual(self.io.load(directory).global_step, 1)

    def test_a_missing_lora_tensor_is_rejected(self) -> None:
        # PEFT reports these in missing_keys rather than raising, and
        # missing_keys also lists the whole frozen backbone; without filtering
        # to trainable names this would resume on a fresh adapter in silence.
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            directory = Path(temporary)
            adapter = self._saved(directory)
            self._rewrite_without(adapter, lambda key: "lora_A" in key)
            with self.assertRaises(RuntimeError) as caught:
                self.io.load(directory)
            self.assertIn("missing trainable keys", str(caught.exception))
            self.assertIn("lora_A", str(caught.exception))

    def test_a_missing_primitive_tensor_reports_the_checkpoint(self) -> None:
        # PEFT raises a bare KeyError for modules_to_save entries; the wrapper
        # has to say which checkpoint and why. Adding a module to the primitive
        # encoder (as normalize_output did) lands exactly here.
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            directory = Path(temporary)
            adapter = self._saved(directory)
            self._rewrite_without(adapter, lambda key: "local_feature_norm" in key)
            with self.assertRaises(RuntimeError) as caught:
                self.io.load(directory)
            message = str(caught.exception)
            self.assertIn("different model definition", message)
            self.assertIn("adapter", message)
            self.assertNotIsInstance(caught.exception, KeyError)


if __name__ == "__main__":
    unittest.main()
