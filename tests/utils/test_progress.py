from io import StringIO
import unittest

from rich.console import Console

from src.utils import RichEpochProgressBar


class RichEpochProgressBarTest(unittest.TestCase):
    def test_renders_requested_epoch_and_optimizer_step_format(self) -> None:
        stream = StringIO()
        console = Console(file=stream, force_terminal=False, width=160)
        bar = RichEpochProgressBar(
            console=console,
            leave=True,
            auto_refresh=False,
        )

        bar.start_epoch(3, total_steps=5)
        bar.advance(2)
        bar.update_metrics({"loss": "1.234", "lr": "2.00e-04"})
        bar.refresh()
        bar.stop()

        rendered = stream.getvalue()
        self.assertIn("[epoch 3] |", rendered)
        self.assertIn("|[2/5]", rendered)
        self.assertIn("loss: 1.234", rendered)
        self.assertIn("lr: 2.00e-04", rendered)
        self.assertIn(" • ", rendered)
        self.assertIn("it/s", rendered)

    def test_non_main_process_and_zero_refresh_rate_are_silent(self) -> None:
        for kwargs in (
            {"is_main_process": False},
            {"refresh_rate": 0},
            {"enabled": False},
        ):
            with self.subTest(kwargs=kwargs):
                stream = StringIO()
                bar = RichEpochProgressBar(
                    console=Console(file=stream, force_terminal=False),
                    leave=True,
                    auto_refresh=False,
                    **kwargs,
                )
                bar.start_epoch(1, total_steps=2)
                bar.advance()
                bar.stop()
                self.assertEqual(stream.getvalue(), "")

    def test_resume_offset_and_configuration_validation(self) -> None:
        stream = StringIO()
        bar = RichEpochProgressBar.from_config(
            {
                "enabled": True,
                "refresh_rate": 5,
                "leave": True,
                "styles": {"complete": "magenta"},
            },
            is_main_process=True,
            console=Console(file=stream, force_terminal=False, width=60),
        )
        bar.start_epoch(2, total_steps=8, completed_steps=6)
        bar.advance()
        bar.stop()
        self.assertIn("[epoch 2] |", stream.getvalue())
        self.assertIn("|[7/8]", stream.getvalue())

        with self.assertRaisesRegex(ValueError, "refresh_rate"):
            RichEpochProgressBar(refresh_rate=-1)
        with self.assertRaisesRegex(ValueError, "unknown progress styles"):
            RichEpochProgressBar(styles={"unknown": "red"})

    def test_multiple_epochs_replace_the_previous_task_when_not_left(self) -> None:
        bar = RichEpochProgressBar(
            console=Console(file=StringIO(), force_terminal=False),
            leave=False,
            auto_refresh=False,
        )
        bar.start_epoch(1, total_steps=2)
        bar.advance(2)
        bar.finish_epoch()
        bar.start_epoch(2, total_steps=3)
        bar.advance(3)
        bar.finish_epoch()
        bar.stop()


if __name__ == "__main__":
    unittest.main()
